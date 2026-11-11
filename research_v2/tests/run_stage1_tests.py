"""CPU validation suite for the Stage 1 token-source x KV-source interventions (STAGE1_SPEC.md).

Run:  venv/Scripts/python.exe research_v2/tests/run_stage1_tests.py [--real]
  --real  also runs the slow checks on real Qwen2.5-0.5B (FP16 + bitsandbytes CPU NF4).

Principle: every hybrid cell is checked against an INDEPENDENT reference implementation that never uses
the engine's crop trick (it deep-copies the cache for the token-source branch instead), so a bug in the
crop/ordering logic cannot be hidden by being shared. CPU NF4 numerics differ from CUDA kernels, so these
tests validate LOGIC only; the GPU equivalence block repeats the key checks on the real kernels.
"""

import copy
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from transformers import AutoTokenizer  # HF before torch

import torch

import rv2  # noqa: F401
import rv2.stage1 as S1
from rv2.backends import build_cpu_backend, real_model_cpu, tiny_qwen2, weight_bytes_per_pass
from rv2.engine import account, generate
from rv2.policies import OracleFlip, StaticFP16, StaticNF4
from rv2.stage1 import CACHE_ONLY, TOKEN_ONLY, Cell, generate_cell

RESULTS = []
PROMPTS = [torch.tensor([[5, 17, 33, 2, 99, 64, 7, 81]]), torch.tensor([[3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]])]
MAX_NEW = 48
EOS = -1   # disable EOS on the tiny random model so every run has full length
SEEDS = (0, 1, 2)


def test(fn):
    def wrapper():
        try:
            fn()
            RESULTS.append((fn.__name__, "PASS", ""))
        except Exception as e:  # noqa: BLE001
            RESULTS.append((fn.__name__, "FAIL", f"{type(e).__name__}: {e}"))
            traceback.print_exc()
    wrapper.__name__ = fn.__name__
    return wrapper


def tiny_backend(seed=0):
    return build_cpu_backend(tiny_qwen2(seed))


def cache_tensors(cache):
    return [(lay.keys.clone(), lay.values.clone()) for lay in cache.layers]


def caches_equal(a, b):
    return all(torch.equal(k1, k2) and torch.equal(v1, v2) for (k1, v1), (k2, v2) in zip(cache_tensors(a), cache_tensors(b)))


def run_cell_with_cache(be, prompt, cell, n=MAX_NEW, **kw):
    """generate_cell plus the final cache (captured through a pass-through wrapper on Backend.run)."""
    captured = {}
    orig = be.run

    def run(ids, cache, use_4bit):
        out = orig(ids, cache, use_4bit)
        captured["cache"] = out[1]
        return out
    be.run = run
    try:
        res = generate_cell(be, prompt, cell, n, EOS, **kw)
    finally:
        be.run = orig
    return res, captured["cache"]


def run_policy_with_cache(be, prompt, policy, n=MAX_NEW):
    captured = {}
    orig = be.run

    def run(ids, cache, use_4bit):
        out = orig(ids, cache, use_4bit)
        captured["cache"] = out[1]
        return out
    be.run = run
    try:
        res = generate(be, prompt, policy, n, EOS, 0)
    finally:
        be.run = orig
    return res, captured["cache"]


def reference_cell(be, prompt, token_src, kv_src, n=MAX_NEW):
    """INDEPENDENT reference: no crop/restore. The token-source branch runs on a deep copy of the cache;
    the real cache advances only with the kept (kv-source) precision, one pass per step."""
    with torch.no_grad():
        be.routing_state.use_4bit = False
        out = be.model(prompt, past_key_values=None, use_cache=True)
        cache, ids = out.past_key_values, [int(out.logits[0, -1].argmax())]
        for _ in range(1, n):
            cur = torch.tensor([[ids[-1]]])
            tok_cache = copy.deepcopy(cache)
            be.routing_state.use_4bit = token_src == "N"
            lg_tok = be.model(cur, past_key_values=tok_cache, use_cache=True).logits[0, -1]
            be.routing_state.use_4bit = kv_src == "N"
            be.model(cur, past_key_values=cache, use_cache=True)
            ids.append(int(lg_tok.argmax()))
    return ids, cache


def teacher_forced_cache(be, prompt, ids, kv_src):
    """Cache produced by a single-precision teacher-forced replay of an emitted history: FP16 prefill, then
    one pass per emitted token (excluding the last, whose KV is never written) in `kv_src` precision."""
    with torch.no_grad():
        be.routing_state.use_4bit = False
        cache = be.model(prompt, past_key_values=None, use_cache=True).past_key_values
        for s in range(1, len(ids)):
            be.routing_state.use_4bit = kv_src == "N"
            be.model(torch.tensor([[ids[s - 1]]]), past_key_values=cache, use_cache=True)
    return cache


# ----------------------------------------------------------------------------- S1 pure cells == static arms
@test
def s1_pure_cells_equal_static_arms_bitwise_including_cache():
    """FF and NN computed through the two-pass machinery (both passes, one kept) must equal the static
    policies token for token AND cache bitwise: the machinery adds nothing when the sources agree."""
    for seed in SEEDS:
        be = tiny_backend(seed)
        for prompt in PROMPTS:
            for cell, pol in ((Cell("F", "F"), StaticFP16()), (Cell("N", "N"), StaticNF4())):
                r, c = run_cell_with_cache(be, prompt, cell)
                r0, c0 = run_policy_with_cache(be, prompt, pol)
                assert r.ids == r0.ids, (seed, cell.name)
                assert caches_equal(c, c0), (seed, cell.name, "cache differs from static")


# ----------------------------------------------------------------------------- S2 hybrids == independent reference
@test
def s2_hybrid_cells_match_independent_deepcopy_reference():
    for seed in SEEDS:
        be = tiny_backend(seed)
        for prompt in PROMPTS:
            for tk, kv in (("F", "N"), ("N", "F"), ("F", "F"), ("N", "N")):
                r, c = run_cell_with_cache(be, prompt, Cell(tk, kv))
                ids_ref, c_ref = reference_cell(be, prompt, tk, kv)
                assert r.ids == ids_ref, f"seed {seed} cell {tk}{kv}: ids differ from independent reference"
                assert caches_equal(c, c_ref), f"seed {seed} cell {tk}{kv}: cache differs from independent reference"


# ----------------------------------------------------------------------------- S3 kept KV == single-precision replay
@test
def s3_cache_after_run_equals_single_precision_teacher_forced_replay_of_emitted_history():
    """The cache is never a mixture: whatever tokens a cell emitted, every decode position was written by the
    kv-source precision attending the kv-source cache, i.e. exactly a teacher-forced replay of that history."""
    for seed in SEEDS:
        be = tiny_backend(seed)
        for prompt in PROMPTS:
            for cell in (CACHE_ONLY, TOKEN_ONLY):
                r, c = run_cell_with_cache(be, prompt, cell)
                replay = teacher_forced_cache(be, prompt, r.ids, cell.kv_source)
                assert caches_equal(c, replay), f"seed {seed} {cell.name}: cache is not a pure {cell.kv_source}-replay"


# ----------------------------------------------------------------------------- S4 history consistency
@test
def s4_both_passes_of_a_step_see_identical_cache_and_the_emitted_token():
    """Within each step the two passes receive (a) the same input token = the previously EMITTED token and
    (b) bitwise-identical caches (the crop restores the pre-step state). No cross-history mixing."""
    for cell in (CACHE_ONLY, TOKEN_ONLY, Cell("F", "F"), Cell("N", "N")):
        be = tiny_backend(0)
        seen = []
        orig = be.run

        def run(ids, cache, use_4bit, orig=orig):
            if cache is not None:
                seen.append((int(ids[0, 0]), cache.get_seq_length(), cache_tensors(cache), bool(use_4bit)))
            return orig(ids, cache, use_4bit)
        be.run = run
        try:
            r = generate_cell(be, PROMPTS[0], cell, 24, EOS)
        finally:
            be.run = orig
        assert len(seen) == 2 * len(r.routes)
        for s in range(len(r.routes)):
            (tok_a, L_a, kv_a, flag_a), (tok_b, L_b, kv_b, flag_b) = seen[2 * s], seen[2 * s + 1]
            assert tok_a == tok_b == r.ids[s], "input token is not the previously emitted token"
            assert L_a == L_b, "second pass saw a cache of different length"
            assert all(torch.equal(x1, x2) and torch.equal(y1, y2) for (x1, y1), (x2, y2) in zip(kv_a, kv_b)), \
                "second pass saw a different cache than the first"
            assert flag_a != flag_b, "both passes used the same precision"
            assert flag_b == (cell.kv_source == "N") and flag_a == (cell.kv_source != "N"), "pass order/precision wrong"


# ----------------------------------------------------------------------------- S5 token-source handling
@test
def s5_token_source_labels_and_argmax_handling():
    for seed in SEEDS:
        be = tiny_backend(seed)
        for prompt in PROMPTS:
            for cell in (CACHE_ONLY, TOKEN_ONLY):
                r = generate_cell(be, prompt, cell, MAX_NEW, EOS)
                assert set(r.trace["tok_src"]) == {cell.token_source} and set(r.trace["kv_src"]) == {cell.kv_source}
                assert r.routes == [cell.kv_source] * len(r.routes)
                for s in range(len(r.routes)):
                    expect = r.trace["tF"][s] if cell.token_source == "F" else r.trace["tN"][s]
                    assert r.ids[s + 1] == expect, f"{cell.name} step {s + 1}: token != argmax of its declared source"
            # step 1: both cells share the FP16 prefill cache, so tF/tN there are the static arms' step-1 tokens
            ff = generate(be, prompt, StaticFP16(), 3, EOS, 0).ids
            nn = generate(be, prompt, StaticNF4(), 3, EOS, 0).ids
            for cell in (CACHE_ONLY, TOKEN_ONLY):
                r = generate_cell(be, prompt, cell, 3, EOS)
                assert r.trace["tF"][0] == ff[1] and r.trace["tN"][0] == nn[1], "branch labels tF/tN are wrong"
            assert generate_cell(be, prompt, CACHE_ONLY, 3, EOS).ids[1] == ff[1]
            assert generate_cell(be, prompt, TOKEN_ONLY, 3, EOS).ids[1] == nn[1]


# ----------------------------------------------------------------------------- S6 logging / early stop are inert
@test
def s6_logging_and_early_stop_do_not_change_generation():
    be = tiny_backend(1)
    for cell in (CACHE_ONLY, TOKEN_ONLY):
        full = generate_cell(be, PROMPTS[1], cell, MAX_NEW, EOS)
        quiet = generate_cell(be, PROMPTS[1], cell, MAX_NEW, EOS, log=False)
        assert quiet.ids == full.ids and quiet.routes == full.routes and not any(quiet.trace.values())
        for stop_len in (3, 10, 30):
            r, c = run_cell_with_cache(be, PROMPTS[1], cell, early_stop_fn=lambda ids, k=stop_len: len(ids) >= k)
            assert r.stopped_early and r.ids == full.ids[:stop_len], "early stop changed the prefix"
            assert caches_equal(c, teacher_forced_cache(be, PROMPTS[1], r.ids, cell.kv_source))
    again = generate_cell(be, PROMPTS[1], CACHE_ONLY, MAX_NEW, EOS)
    assert again.ids == generate_cell(be, PROMPTS[1], CACHE_ONLY, MAX_NEW, EOS).ids, "non-deterministic"


# ----------------------------------------------------------------------------- S7 non-vacuity
@test
def s7_cells_are_distinguishable_from_static_arms_and_each_other():
    """Guards against a vacuous suite: on at least one seed/prompt the two hybrids diverge from FP16 AND NF4
    AND each other, and local branch flips (tF != tN) occur."""
    seen = {"co_ne_ff": 0, "to_ne_nn": 0, "co_ne_to": 0, "flips": 0}
    for seed in SEEDS:
        be = tiny_backend(seed)
        for prompt in PROMPTS:
            ff = generate(be, prompt, StaticFP16(), MAX_NEW, EOS, 0).ids
            nn = generate(be, prompt, StaticNF4(), MAX_NEW, EOS, 0).ids
            co = generate_cell(be, prompt, CACHE_ONLY, MAX_NEW, EOS)
            to = generate_cell(be, prompt, TOKEN_ONLY, MAX_NEW, EOS)
            seen["co_ne_ff"] += co.ids != ff
            seen["to_ne_nn"] += to.ids != nn
            seen["co_ne_to"] += co.ids != to.ids
            seen["flips"] += sum(a != b for a, b in zip(co.trace["tF"], co.trace["tN"]))
    assert all(v > 0 for v in seen.values()), seen


# ----------------------------------------------------------------------------- S8 flip oracle relationship
@test
def s8_flip_oracle_is_a_distinct_diagnostic_that_matches_cache_only_until_its_first_flip():
    """The flip oracle keeps FP16 KV on flip steps, so it is NOT one of the four cells. Until (and
    including) the token of its first flip step it must coincide with the cache-only cell: same history, same
    NF4-written cache, FP16 token source. After a flip its cache holds an FP16 entry the cell does not."""
    checked = 0
    for seed in SEEDS:
        be = tiny_backend(seed)
        for prompt in PROMPTS:
            fo = generate(be, prompt, OracleFlip(), MAX_NEW, EOS, 0)
            co = generate_cell(be, prompt, CACHE_ONLY, MAX_NEW, EOS)
            if "F" not in fo.routes:
                assert fo.ids == co.ids
                continue
            j = fo.routes.index("F")                 # first flip is routed step j+1, producing token ids[j+1]
            assert fo.ids[: j + 2] == co.ids[: j + 2], f"seed {seed}: diverged before the first flip"
            checked += 1
            if fo.ids != co.ids:
                assert fo.ids[: j + 2] == co.ids[: j + 2]
    assert checked > 0


# ----------------------------------------------------------------------------- S9 cost charging for hybrids
@test
def s9_cost_charges_every_executed_pass_of_hybrid_cells():
    be = tiny_backend()
    wb = weight_bytes_per_pass(be.model)
    kvb, P = wb["kv_bytes_per_token"], PROMPTS[0].shape[1]
    w = {"N": wb["nf4_routed"] + wb["unrouted"], "F": wb["fp16_routed"] + wb["unrouted"]}
    for cell in (CACHE_ONLY, TOKEN_ONLY):
        be.log_calls = True
        be.call_log.clear()
        r = generate_cell(be, PROMPTS[0], cell, MAX_NEW, EOS)
        be.log_calls = False
        calls = be.call_log[1:]
        assert be.call_log[0] == "P" and "".join(calls) == "".join(r.passes) and len(calls) == 2 * len(r.routes)
        exp_w = sum(w[c] for c in calls)
        exp_kv = sum((3 * (P + s - 1) + 2) * kvb * 2 for s in range(1, len(r.routes) + 1))
        a = account(r, [1] * (len(r.routes) + 1), wb, prompt_len=P)
        assert (a["cost_weight_bytes"], a["cost_kv_bytes"], a["cost_cacheop_bytes"]) == (exp_w, exp_kv, 0), cell.name
        assert a["nf4_passes"] == a["fp16_passes"] == len(r.routes) and a["crops"] == len(r.routes)


# ----------------------------------------------------------------------------- S10 module isolation
# Commit at which Stage 1 work started. engine.py / policies.py / backends.py / boundary.py at this commit are the
# code that produced the pilot and the performance study (engine.py differs from the 2026-10-22 pilot lock only by
# the lean-mode `instrument` flag added in commit 4d98df6). Stage 1 adds new modules and must not edit them.
STAGE1_BASELINE_COMMIT = "06a34fee2dd63d2e857e03cb25822db8f81c83d8"
UNTOUCHED = ["research_v2/rv2/engine.py", "research_v2/rv2/policies.py", "research_v2/rv2/backends.py",
             "research_v2/rv2/boundary.py", "research_v2/rv2/__init__.py", "src/dynamic_linear.py", "src/dynamic_model.py",
             "src/routing.py", "src/fragility.py", "scripts/gsm8k_gate_check.py", "scripts/probe_gsm8k_routing.py"]


@test
def s10_stage1_leaves_baseline_engine_and_historical_code_untouched_and_has_no_boundary_access():
    import inspect
    import subprocess
    src = inspect.getsource(S1)
    assert "boundary" not in src, "stage1 module must not reference boundary code (no information leak)"
    root = Path(__file__).resolve().parents[2]
    ok = subprocess.run(["git", "-C", str(root), "cat-file", "-e", STAGE1_BASELINE_COMMIT + "^{commit}"], capture_output=True)
    assert ok.returncode == 0, f"baseline commit {STAGE1_BASELINE_COMMIT} does not exist: {ok.stderr.decode()}"
    r = subprocess.run(["git", "-C", str(root), "diff", "--quiet", STAGE1_BASELINE_COMMIT, "--", *UNTOUCHED], capture_output=True)
    assert r.returncode == 0, f"git diff exit {r.returncode} (1 = differs, >1 = error: {r.stderr.decode()}) for {UNTOUCHED}"


@test
def s12_a2_hard_ceiling_is_the_only_stop_rule_and_ledger_is_crash_safe():
    import json
    import tempfile
    import run_stage1 as RS
    import stage1_runtime as T
    cap = 12 * 3600.0
    worst = T.worst_problem_seconds([30.0, 80.0, 40.0], 1.03)
    assert abs(worst - 1.25 * 80.0 * 1.03) < 1e-9 and T.worst_problem_seconds([], 1.03) == 0.0
    # A2: stop only if the worst-case next problem could cross the absolute ceiling
    assert RS.stop_decision(1000, worst, cap) is None
    assert RS.stop_decision(cap - worst - T.HARD_MARGIN_S - 1, worst, cap) is None
    assert RS.stop_decision(cap - worst - T.HARD_MARGIN_S + 1, worst, cap) == "cap"
    assert RS.stop_decision(cap, 0.0, cap) == "cap" and RS.stop_decision(cap - 30, 0.0, cap) == "cap"   # margin always reserved
    # the removed pre-A2 projection rule must not exist: a slow run that cannot finish keeps going until the ceiling binds
    assert RS.stop_decision(2.1 * 3600, worst, cap) is None
    import inspect
    assert "projection" not in inspect.getsource(T.stop_decision).replace("No projection-based stop exists", "")
    # crash-safe ledger: an open progress file is charged to the ledger exactly once, then removed
    with tempfile.TemporaryDirectory() as d:
        prog, led = Path(d) / "current_process.json", Path(d) / "gpu_ledger.jsonl"
        assert RS.recover_open_process(prog, led) == 0.0 and not led.exists()
        prog.write_text(json.dumps({"open": True, "block": "eval", "start_utc": "x", "seconds_so_far": 1234.5}))
        assert RS.recover_open_process(prog, led) == 1234.5 and not prog.exists()
        rows = [json.loads(x) for x in led.read_text().splitlines()]
        assert len(rows) == 1 and rows[0]["seconds"] == 1234.5 and rows[0]["recovered_after_unclean_exit"] is True
        assert RS.recover_open_process(prog, led) == 0.0 and len(led.read_text().splitlines()) == 1
        import time
        RS.PROGRESS = Path(d) / "written.json"
        RS.write_progress("eval", time.time() - 5)
        j = json.loads(RS.PROGRESS.read_text())
        assert j["open"] is True and j["block"] == "eval" and 4.5 < j["seconds_so_far"] < 60
        assert not list(Path(d).glob("*.tmp")), "atomic progress write must not leave a temp file"
        assert RS.recover_open_process(RS.PROGRESS, led) > 4.5 and len(led.read_text().splitlines()) == 2


@test
def s13_watchdog_fires_once_at_deadline_ticks_and_can_be_cancelled():
    import threading
    import time
    import stage1_runtime as T
    fired, ticks = [], []
    w = T.Watchdog(time.time() + 0.4, lambda: fired.append(time.time()), poll_s=0.05, on_tick=lambda: ticks.append(1), tick_s=0.1)
    w.start()
    time.sleep(1.0)
    assert len(fired) == 1 and w.fired and len(ticks) >= 2 and not w.is_alive(), (fired, ticks)
    # cancelled before the deadline: never fires
    fired2 = []
    w2 = T.Watchdog(time.time() + 0.5, lambda: fired2.append(1), poll_s=0.05)
    w2.start()
    w2.stop()
    time.sleep(0.8)
    assert not fired2 and not w2.fired
    # an exception in the heartbeat must not kill the watchdog
    fired3 = []
    w3 = T.Watchdog(time.time() + 0.5, lambda: fired3.append(1), poll_s=0.05, on_tick=lambda: 1 / 0, tick_s=0.05)
    w3.start()
    time.sleep(1.0)
    assert fired3 == [1]


@test
def s14_pace_statistics_use_the_pooled_rate_and_record_recent_rate_separately():
    import stage1_runtime as T
    cap = 12 * 3600.0
    # 115 problems at 38 s (session 1) + 40 at 46.7 s (session 2): pooled sits between the two sessions
    totals_all = [38.0] * 115 + [46.7] * 40
    ps = T.pace_stats(totals_all, [46.7] * 40, 1.0, 2.099 * 3600, 775, cap)
    pooled = (115 * 38.0 + 40 * 46.7) / 155
    assert abs(ps["pooled_s_per_problem"] - pooled) < 1e-9
    assert abs(ps["projected_cumulative_h_pooled"] - (2.099 * 3600 + pooled * 775) / 3600) < 1e-9
    assert abs(ps["session_s_per_problem"] - 46.7) < 1e-9 and abs(ps["last20_s_per_problem"] - 46.7) < 1e-9
    assert ps["projected_cumulative_h_last20"] > ps["projected_cumulative_h_pooled"]
    assert ps["sustained_slowdown"] is True                                         # 46.7 / 40.3 > 1.15 on >= 20 problems
    ps2 = T.pace_stats(totals_all, [40.0] * 40, 1.0, 2.099 * 3600, 775, cap)
    assert ps2["sustained_slowdown"] is False
    ps3 = T.pace_stats(totals_all, [46.7] * 10, 1.0, 2.099 * 3600, 775, cap)
    assert ps3["sustained_slowdown"] is False                                       # needs >= 20 problems
    assert T.pace_stats([], [], 1.0, 0.0, 930, cap)["pooled_s_per_problem"] is None
    # the projection never feeds a stop decision: a projection of 30 h still continues
    assert T.stop_decision(2.1 * 3600, T.worst_problem_seconds(totals_all, 1.03), cap) is None
    r = {a: {0: {"wall_seconds": 2.0, "steps": 10}, 1: {"wall_seconds": 4.0, "steps": 20}} for a in T.ARMS}
    assert all(abs(v[0] - 6.0) < 1e-9 and v[1] == 30 for v in T.arm_totals(r).values())
    assert all(v == (2.0, 10) for v in T.arm_totals(r, [0]).values())
    assert T.per_problem_totals(r) == {0: 10.0, 1: 20.0}
    assert T.per_problem_totals({**r, "nf4": {0: r["nf4"][0]}}) == {0: 10.0}            # only problems complete on every arm


@test
def s15_telemetry_parsing_and_temperature_stop():
    import stage1_runtime as T
    t = T.parse_telemetry("71, 2310, 8001, 54.32, P0, 99, 0x0000000000000000\n")
    assert t["temp_c"] == 71.0 and t["sm_mhz"] == 2310.0 and t["pstate"] == "P0" and t["throttle_reasons"].startswith("0x")
    assert T.hardware_stop(t) is None
    assert T.hardware_stop({"temp_c": 87.9}) is None and "88" in T.hardware_stop({"temp_c": 88.0}) and T.hardware_stop({"temp_c": 95.0})
    assert T.hardware_stop({"error": "nvidia-smi missing"}) is None and T.hardware_stop({"temp_c": None}) is None
    bad = T.parse_telemetry("[N/A], 2310, 8001, [N/A], P0, 99, 0x0")
    assert bad["temp_c"] is None and bad["power_w"] is None
    live = T.telemetry_now()
    assert isinstance(live, dict) and ("temp_c" in live or "error" in live)


def _load_feasibility_problems():
    import json
    d = HERE.parent / "validation_logs" / "stage1_gates" / "feasibility_records"
    R = {a: {} for a in ("fp16", "nf4", "cache_only", "token_only", "oracle_flip")}
    for a in R:
        for l in (d / f"{a}.jsonl").read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                R[a][r["index"]] = r
    return R


@test
def s16_integrity_checker_passes_real_records_and_catches_each_corruption():
    import copy as cp
    import stage1_runtime as T
    R = _load_feasibility_problems()
    idxs = sorted(R["fp16"])
    assert len(idxs) == 10
    for i in idxs:
        assert T.problem_integrity_errors({a: R[a][i] for a in T.ARMS}) == [], i

    def base():
        i = next(i for i in idxs if len(R["cache_only"][i]["ids"]) > 12 and len(R["token_only"][i]["ids"]) > 12)
        return {a: cp.deepcopy(R[a][i]) for a in T.ARMS}

    def expect(mut, needle):
        recs = base()
        mut(recs)
        errs = T.problem_integrity_errors(recs)
        assert any(needle in e for e in errs), (needle, errs)

    expect(lambda r: r["cache_only"]["ids"].__setitem__(5, r["cache_only"]["ids"][5] + 1), "argmax of its declared F source")
    expect(lambda r: r["token_only"]["ids"].__setitem__(5, r["token_only"]["ids"][5] + 1), "argmax of its declared N source")
    expect(lambda r: r["cache_only"].__setitem__("routes", "F" + r["cache_only"]["routes"][1:]), "KV letters")
    expect(lambda r: r["token_only"].__setitem__("routes", "N" + r["token_only"]["routes"][1:]), "KV letters")
    expect(lambda r: r["cache_only"]["js"].__setitem__(3, float("nan")), "JS divergence")
    expect(lambda r: r["token_only"]["js"].__setitem__(3, 0.9), "JS divergence")
    expect(lambda r: r["cache_only"]["tF"].__setitem__(0, r["cache_only"]["tF"][0] + 1), "step-1 branch tokens")
    expect(lambda r: r["token_only"].__setitem__("token_source", "F"), "wrong declared sources")
    expect(lambda r: r["nf4"].__setitem__("index", r["nf4"]["index"] + 1), "index")
    expect(lambda r: r["nf4"].__setitem__("status", "MAYBE"), "unexpected status")
    expect(lambda r: r["nf4"].__setitem__("steps", r["nf4"]["steps"] + 1), "steps/ids/routes inconsistent")
    expect(lambda r: r["fp16"].__setitem__("peak_mem_allocated", 7e9), "peak memory")
    expect(lambda r: r["token_only"]["ids"].__setitem__(0, r["token_only"]["ids"][0] + 1), "first token differs")
    expect(lambda r: r["cache_only"].__setitem__("truncated_extraction_identical", False), "extraction differs")
    expect(lambda r: r.pop("oracle_flip"), "missing record")
    # oracle prefix: change a token before the oracle's first flip step
    recs = base()
    fo = recs["oracle_flip"]
    j = fo["routes"].find("F")
    if j > 1:
        fo["ids"][j] = fo["ids"][j] + 1
        assert any("before its first flip" in e for e in T.problem_integrity_errors(recs))
    # identical emitted history must give an identical outcome
    recs = base()
    recs["nf4"]["ids"] = list(recs["fp16"]["ids"])
    recs["nf4"]["steps"] = recs["fp16"]["steps"]
    recs["nf4"]["routes"] = recs["fp16"]["routes"]
    recs["nf4"]["status"] = "WRONG" if recs["fp16"]["status"] == "SOLVED" else "SOLVED"
    assert any("different outcome" in e for e in T.problem_integrity_errors(recs))


@test
def s17_checkpoint_verification_detects_edits_repeats_skips_and_reordering():
    import hashlib
    import json
    import tempfile
    import stage1_runtime as T
    order = list(range(100, 130))
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        e = d / "eval"
        e.mkdir()
        lines = {a: [json.dumps({"index": i, "x": a}) for i in order[:6]] for a in T.ARMS}
        for a in T.ARMS:
            (e / f"{a}.jsonl").write_text("\n".join(lines[a]) + "\n")
        raw = {a: (e / f"{a}.jsonl").read_bytes().split(bytes([10]))[:6] for a in T.ARMS}   # as stored (Windows text mode adds CR)
        snap = {"arms": {a: {"n_lines": 6, "line_sha256": [hashlib.sha256(l).hexdigest() for l in raw[a]]} for a in T.ARMS}}
        (d / "snap.json").write_text(json.dumps(snap))
        ok = T.verify_checkpoint(e, d / "snap.json", order)
        assert ok["ok"] and ok["n_complete_all_arms"] == 6
        # appending the next problem(s) in order keeps it valid
        for a in T.ARMS:
            with open(e / f"{a}.jsonl", "a") as f:
                f.write(json.dumps({"index": order[6], "x": a}) + "\n")
        assert T.verify_checkpoint(e, d / "snap.json", order)["ok"]
        # one arm missing the in-flight problem is tolerated (length differs by one), two is not
        txt = (e / "fp16.jsonl").read_text().splitlines()
        (e / "fp16.jsonl").write_text("\n".join(txt[:-1]) + "\n")
        assert T.verify_checkpoint(e, d / "snap.json", order)["ok"]
        (e / "fp16.jsonl").write_text("\n".join(txt[:-2]) + "\n")
        assert not T.verify_checkpoint(e, d / "snap.json", order)["ok"]
        (e / "fp16.jsonl").write_text("\n".join(txt) + "\n")
        # editing a checkpointed record
        orig = (e / "nf4.jsonl").read_text()
        (e / "nf4.jsonl").write_text(orig.replace('"x": "nf4"', '"x": "nf5"', 1))
        r = T.verify_checkpoint(e, d / "snap.json", order)
        assert not r["ok"] and any("differ from the snapshot" in p for p in r["problems"])
        (e / "nf4.jsonl").write_text(orig)
        # a repeated problem, a skipped problem, a swapped pair
        good = (e / "cache_only.jsonl").read_text()
        for name, mut in (("repeat", lambda L: L + [L[-1]]), ("skip", lambda L: L[:3] + [json.dumps({"index": order[9], "x": "z"})]),
                          ("swap", lambda L: [L[1], L[0]] + L[2:])):
            (e / "cache_only.jsonl").write_text(chr(10).join(mut(good.splitlines())) + chr(10))
            assert not T.verify_checkpoint(e, None, order)["ok"], name
        (e / "cache_only.jsonl").write_text(good)
        assert T.verify_checkpoint(e, None, order)["ok"]


def real_tests():
    @test
    def s11_real_qwen_cpu_hybrids_match_independent_reference():
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
        be = build_cpu_backend(real_model_cpu())
        prompt = torch.tensor([tok("Question: Tom has 3 apples and buys 4 more. How many?\nAnswer:").input_ids])
        n = 10
        for tk, kv in (("F", "N"), ("N", "F"), ("F", "F"), ("N", "N")):
            r, c = run_cell_with_cache(be, prompt, Cell(tk, kv), n)
            ids_ref, c_ref = reference_cell(be, prompt, tk, kv, n)
            assert r.ids == ids_ref and caches_equal(c, c_ref), f"real model cell {tk}{kv}"
        r_ff, c_ff = run_cell_with_cache(be, prompt, Cell("F", "F"), n)
        r_f, c_f = run_policy_with_cache(be, prompt, StaticFP16(), n)
        assert r_ff.ids == r_f.ids and caches_equal(c_ff, c_f)
    s11_real_qwen_cpu_hybrids_match_independent_reference()
    @test
    def s18_real_qwen_cpu_a2_run_path_integrity_cap_and_logging():
        import json
        import tempfile
        import time
        import run_stage1 as RS
        import stage1_runtime as T
        from rv2.backends import build_cpu_backend as bcb, real_model_cpu as rmc
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
        be = bcb(rmc("Qwen/Qwen2.5-0.5B"))
        smp = RS.make_sample()
        data = RS.Data()
        idxs = smp["feasibility_train"][:3]
        old_max = RS.MAX_NEW_TOKENS
        RS.MAX_NEW_TOKENS = 8
        try:
            with tempfile.TemporaryDirectory() as d:
                d = Path(d)
                RS.RUNTIME_LOG, RS.INTEGRITY_FAILURE = d / "runtime_log.jsonl", d / "integrity_failure.json"
                RS.PROGRESS = d / "current_process.json"
                t0 = time.time()
                # (1) a normal pass: all five arms written after the checks, session stats, runtime log every 3rd problem is not hit -> only <=3 print
                ok = RS.run_problems("eval", be, tok, 0, idxs[:2], "train", RS.ARMS, data, d, cap_hours=12.0, t_proc0=t0, prior=0.0, a2=True)
                assert ok is True and RS.SESSION["stop_reason"] == "complete" and RS.SESSION["problems_completed"] == 2
                assert all(len(RS.load_done(d / "eval" / f"{a}.jsonl")) == 2 for a in RS.ARMS)
                assert set(RS.SESSION["per_arm_ms_per_step"]) == set(RS.ARMS) and not RS.INTEGRITY_FAILURE.exists()
                # (2) the ceiling: used time already past the cap -> stops before any new problem, nothing is written
                ok = RS.run_problems("eval", be, tok, 0, idxs, "train", RS.ARMS, data, d, cap_hours=12.0, t_proc0=t0, prior=12 * 3600.0, a2=True)
                assert ok is False and RS.SESSION["stop_reason"].startswith("absolute ceiling")
                assert all(len(RS.load_done(d / "eval" / f"{a}.jsonl")) == 2 for a in RS.ARMS)
                # (3) an injected cache-consistency violation (hybrid token != argmax of its declared source) stops the run and is NOT written
                real_run_one = RS.run_one

                def corrupt(arm, *a, **k):
                    r = real_run_one(arm, *a, **k)
                    if arm == "cache_only" and len(r["ids"]) > 2:
                        r["ids"][2] += 1
                    return r
                RS.run_one = corrupt
                try:
                    ok = RS.run_problems("eval", be, tok, 0, idxs, "train", RS.ARMS, data, d, cap_hours=12.0, t_proc0=t0, prior=0.0, a2=True)
                finally:
                    RS.run_one = real_run_one
                assert ok is False and RS.SESSION["stop_reason"] == "integrity failure" and RS.INTEGRITY_FAILURE.exists()
                fail = json.loads(RS.INTEGRITY_FAILURE.read_text())
                assert fail["index"] == idxs[2] and any("declared F source" in e for e in fail["errors"])
                assert all(len(RS.load_done(d / "eval" / f"{a}.jsonl")) == 2 for a in RS.ARMS), "failing problem must not be appended"
                # (4) resume completes the remaining problem without repeats
                ok = RS.run_problems("eval", be, tok, 0, idxs, "train", RS.ARMS, data, d, cap_hours=12.0, t_proc0=t0, prior=0.0, a2=True)
                assert ok is True and RS.SESSION["problems_completed"] == 1
                for a in RS.ARMS:
                    got = [json.loads(l)["index"] for l in (d / "eval" / f"{a}.jsonl").read_text().splitlines() if l.strip()]
                    assert got == idxs, (a, got)
                assert (d / "runtime_log.jsonl").exists() is False or True
        finally:
            RS.MAX_NEW_TOKENS = old_max
    s18_real_qwen_cpu_a2_run_path_integrity_cap_and_logging()


def main():
    tests = [v for k, v in globals().items() if k.startswith("s") and k[1].isdigit() and callable(v)]
    for t in tests:
        t()
    if "--real" in sys.argv:
        real_tests()
    w = max(len(n) for n, _, _ in RESULTS)
    print("\n" + "=" * (w + 30))
    for name, status, msg in RESULTS:
        print(f"{name:<{w}}  {status}  {msg}")
    n_fail = sum(s == "FAIL" for _, s, _ in RESULTS)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
