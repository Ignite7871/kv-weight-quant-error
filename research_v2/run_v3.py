"""v3 driver (PROTOCOL_V3.md): origin and transfer of weight-induced cache error, four models, Tier A only.

The step, arms, routing, predictors and calibration accumulation are in rv3/core.py; the ledger, watchdog and hardware stops are the Stage 1 machinery
(run_stage1.gpu_process) and the problem loop is the E1 block loop (run_controls.block_loop), both imported unchanged.

Subcommands (CPU unless marked)
  plan                     write v3_sample.json (ids copied from the locked E1 sample; nothing new is drawn)
  ledger                   cumulative v3 GPU time (one ledger for all four models; absolute 25.0 h ceiling)
  status                   per-model progress and prerequisites
  gates     --model M      [GPU] G1 alpha=0 bitwise (all arms), G2/G3 integrity (all arms), G2b subspace geometry, G4 determinism,
                                 G5 NAT vs engine cache-only, G7 cache geometry + model-file hash, G8 LOC/INH routing
  calibrate --model M      [GPU] 100 train problems: FP16 reference (query second moments), NAT (Delta statistics), C1 at alpha 0.5/1/2
                           then: analyze_v3.py calibrate --model M ; tools/lock_v3.py --append-calibration M
  g6        --model M      [GPU] runtime gate on 3 calibration problems with all test arms. First model: projects all four models and fixes the
                           design (reductions CV -> C2 -> n160, uniform) -> v3_runs/design.json; then tools/lock_v3.py --append-design.
                           Later models: stop if the measured time exceeds the projection by > 15 %.
  tierA     --model M      [GPU] the fixed-order test problems, the design's arms, teacher-forced, predictors on

Before the first model: tools/v3_gpu_mutation_check.py --authorized-gpu-run (once; all mutants on CUDA).
GPU subcommands are REFUSED unless ALL of: --authorized-gpu-run; the GPU mutation check passed; research_v2/v3_runs/AUTHORIZED_BY_AUTHORS_V3.txt exists and contains the text
"PROTOCOL_V3" and the lock id printed by tools/lock_v3.py (created only by the authors); the v3 lock, the E1 lock and the E1L lock verify; the
model is the next one in the fixed order (or already started); gates passed (calibrate, g6, tierA); calibration frozen and its hash in the lock
(g6, tierA); design fixed and in the lock and this model's G6 passed (tierA). The 25.0 h ceiling is absolute; a run cut by it is TRUNCATED.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_controls as E                        # noqa: E402  (locked E1 module, unchanged; imports run_stage1 -> HF before torch)
import e1l_support as LS                        # noqa: E402

import torch                                    # noqa: E402

import rv3.core as V                            # noqa: E402
from rv2.boundary import would_stop_at_boundary  # noqa: E402
from rv2.engine import generate                 # noqa: E402
from rv2.policies import StaticFP16             # noqa: E402
from rv2.stage1 import CACHE_ONLY, generate_cell  # noqa: E402

RS1 = E.RS1
OUT = HERE / "v3_runs"
SAMPLE_FILE = HERE / "v3_sample.json"
AUTH = OUT / "AUTHORIZED_BY_AUTHORS_V3.txt"
DESIGN = OUT / "design.json"
CAP_HOURS = 25.0
S_STEPS = 64
N_GATE_ALPHA0, N_GATE_NUM, N_GATE_G5, N_G6 = 10, 6, 8, 3
MODELS = {"qwen-base": "Qwen/Qwen2.5-0.5B", "llama-instruct": "meta-llama/Llama-3.2-1B-Instruct",
          "qwen-instruct": "Qwen/Qwen2.5-0.5B-Instruct", "llama-base": "meta-llama/Llama-3.2-1B"}
ORDER = list(MODELS)                            # fixed run order (section 9)
GEOMETRY = {"qwen": (24, 2, 64), "llama": (16, 8, 64)}
N_DYNAMIC = {"qwen": 168, "llama": 112}
LLAMA_TIME_RATIO = 0.616                        # E1L / E1 measured time per arm-problem (budget_projection.json)
REDUCTIONS = [("drop", "CV"), ("drop", "C2"), ("n", 160)]
DRIFT = 0.15


def family(m):
    return m.split("-")[0]


def mdir(m):
    return OUT / m


# ----------------------------------------------------------------------------- sample
def make_sample() -> dict:
    if SAMPLE_FILE.exists():
        return json.loads(SAMPLE_FILE.read_text())
    q = json.loads((HERE / "e1_sample.json").read_text())
    s = {"source": "ids copied from the locked E1 sample (e1_sample.json); no new draw, no outcome data",
         "e1_sample_sha256": hashlib.sha256((HERE / "e1_sample.json").read_bytes().replace(bytes([13, 10]), bytes([10]))).hexdigest(),
         "models": MODELS, "run_order": ORDER, "tierA_ids": q["tierA_ids"], "calibration_train_ids": q["calibration_train_ids"],
         "gate_alpha0_train_ids": q["calibration_train_ids"][:N_GATE_ALPHA0], "gate_numerics_train_ids": q["calibration_train_ids"][:N_GATE_NUM],
         "gate_test_ids_for_G5": q["tierA_ids"][:N_GATE_G5], "g6_train_ids": q["calibration_train_ids"][-N_G6:],
         "exposed_test_ids_in_sample": sorted(set(q["tierA_ids"]) & set(q["exposed_test_ids_in_sample"])),
         "seed_tier_labels": {"calibration": "V3calib", "tierA": "V3A", "gates": "V3gate", "g6": "V3g6"},
         "note": "Each model's shared teacher-forced history is its in-session FP16 greedy generation (<= 65 tokens, drift-marker stop)."}
    SAMPLE_FILE.write_text(json.dumps(s, indent=1))
    return s


# ----------------------------------------------------------------------------- guards
def lock_id():
    r = subprocess.run([sys.executable, str(HERE / "tools" / "lock_v3.py"), "--lock-id"], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def locks_ok(model=None) -> bool:
    args = ["--verify"] + (["--verify-model", model] if model else [])
    ok = subprocess.run([sys.executable, str(HERE / "tools" / "lock_v3.py"), *args], capture_output=True).returncode == 0
    for t in ("lock_e1.py", "lock_e1l.py"):
        ok = ok and subprocess.run([sys.executable, str(HERE / "tools" / t), "--verify"], capture_output=True).returncode == 0
    return ok


def lock_text():
    p = HERE / "V3_PROTOCOL_LOCK.txt"
    return p.read_text(encoding="utf-8") if p.exists() else ""


def gates_passed(m):
    g = mdir(m) / "gates" / "gates.json"
    return g.exists() and json.loads(g.read_text()).get("all_passed")


def calibration_locked(m):
    p = mdir(m) / "calibration.json"
    return p.exists() and f"CALIBRATION {m}:" in lock_text() and hashlib.sha256(p.read_bytes()).hexdigest() in lock_text()


def design_locked():
    return DESIGN.exists() and "DESIGN:" in lock_text() and hashlib.sha256(DESIGN.read_bytes()).hexdigest() in lock_text()


def g6_passed(m):
    p = mdir(m) / "g6.json"
    return p.exists() and json.loads(p.read_text()).get("pass")


def require_gpu_permission(a, m, cmd):
    if not a.authorized_gpu_run:
        sys.exit("Refusing to run: GPU runs need explicit author authorization (--authorized-gpu-run).")
    lid = lock_id()
    if lid is None:
        sys.exit("Refusing to run: no v3 lock (tools/lock_v3.py).")
    txt = AUTH.read_text(encoding="utf-8", errors="replace") if AUTH.exists() else ""
    if "PROTOCOL_V3" not in txt or lid not in txt:
        sys.exit(f"Refusing to run: {AUTH.relative_to(HERE.parent)} must exist and contain 'PROTOCOL_V3' and the lock id {lid}. "
                 "Only the authors create it, after reviewing the v3 lock.")
    if not locks_ok(m):
        sys.exit("Refusing to run: the v3 lock (or the E1 / E1L lock, or this model's weight-file hash) does not verify.")
    k = ORDER.index(m)
    for prev in ORDER[:k]:
        if not (mdir(prev) / "tierA_complete.json").exists():
            sys.exit(f"Refusing to run: fixed run order; {prev} has not completed Tier A.")
    gm = OUT / "gpu_mutation_check.json"
    if not (gm.exists() and json.loads(gm.read_text()).get("all_passed")):
        sys.exit("Refusing to run: the GPU mutation check (tools/v3_gpu_mutation_check.py) has not passed.")
    if cmd in ("calibrate", "g6", "tierA") and not gates_passed(m):
        sys.exit(f"Refusing to run: gates for {m} have not passed.")
    if cmd in ("g6", "tierA") and not calibration_locked(m):
        sys.exit(f"Refusing to run: calibration for {m} is not frozen and appended to the lock.")
    if cmd == "g6" and k > 0 and not design_locked():
        sys.exit("Refusing to run: the design (reductions) has not been fixed and locked by the first model's G6.")
    if cmd == "tierA" and not (design_locked() and g6_passed(m)):
        sys.exit(f"Refusing to run: design not locked or G6 not passed for {m}.")


def configure(m):
    OUT.mkdir(parents=True, exist_ok=True)
    RS1.OUT, RS1.LEDGER, RS1.PROGRESS = OUT, OUT / "gpu_ledger.jsonl", OUT / "current_process.json"
    RS1.MODEL = MODELS[m] if m else RS1.MODEL


@contextmanager
def ledger_heartbeat(block: str, cap_hours: float = CAP_HOURS):
    """Amendment 1: ledger accounting for GPU work that does not run inside run_stage1.gpu_process (the GPU mutation check). Charges an earlier
    unclosed process first, refuses at the ceiling, writes the Stage 1 progress file every 15 s (so a hard kill is charged by the next GPU process),
    and on exit writes one ledger line and removes the progress file. Yields a dict; set ["completed"] = True on success."""
    RS1.OUT, RS1.LEDGER, RS1.PROGRESS = OUT, OUT / "gpu_ledger.jsonl", OUT / "current_process.json"
    OUT.mkdir(parents=True, exist_ok=True)
    RS1.recover_open_process(RS1.PROGRESS, RS1.LEDGER)
    if RS1.ledger_seconds() >= cap_hours * 3600:
        sys.exit(f"Refusing: the {cap_hours} h ceiling is reached.")
    t0, stop, state = time.time(), threading.Event(), {"completed": False}

    def beat():
        while not stop.is_set():
            RS1.write_progress(block, t0)
            stop.wait(15)
    hb = threading.Thread(target=beat, daemon=True)
    hb.start()
    try:
        yield state
    finally:
        stop.set()
        hb.join()
        with open(RS1.LEDGER, "a") as f:
            f.write(json.dumps({"start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)), "seconds": time.time() - t0, "block": block,
                                "completed": bool(state["completed"])}) + chr(10))
        RS1.PROGRESS.unlink(missing_ok=True)


# ----------------------------------------------------------------------------- per-problem work
def prompt_ids(tok, data, split, idx, device):
    _, prompt = data.problem(split, idx)
    return tok(prompt, return_tensors="pt").input_ids.to(device)


def fp16_history(be, tok, ids_t):
    res = generate(be, ids_t, StaticFP16(), S_STEPS + 1, tok.eos_token_id, 0, early_stop_fn=lambda ids: would_stop_at_boundary(tok, ids), instrument=False)
    return res.ids


def arm_list(design):
    arms = [a for a in V.TEST_ARMS if a not in design.get("dropped_arms", [])]
    return [(a, 1.0, V.file_tag(a)) for a in arms]


def run_problem(be, tok, data, m, idx, split, arms, tier, ctx, cap, acc=None, predictors=True, hist=None):
    ids_t = prompt_ids(tok, data, split, idx, be.device)
    ref = hist if hist is not None else fp16_history(be, tok, ids_t)
    S_i = min(S_STEPS, len(ref) - 1)
    rl = V.reference_logits_v3(be, ids_t, ref, S_i, cap if acc is not None else None, acc, be.model.config.num_key_value_heads)
    recs = {}
    for arm, alpha, tag in arms:
        t0 = time.perf_counter()
        out = V.teacher_forced_arm_v3(be, ids_t, ref, S_i, arm, alpha, V.arm_seeds(m, arm, alpha, idx, tier), rl, ctx, cap, predictors,
                                      acc if (acc is not None and arm == "NAT") else None)
        recs[tag] = {"index": idx, "split": split, "model": m, "arm": tag, "alpha": alpha, "S": S_i, "ref_len": len(ref), **out,
                     "wall_seconds": time.perf_counter() - t0}
    return recs, ref


def integrity_errors(recs, predictors=True):
    errs = []
    for tag, r in recs.items():
        js = r["js"]
        if not all(x == x and abs(x) < 1e3 for x in js):
            errs.append(f"{tag}: non-finite JS")
        if js and js[0] != 0.0:
            errs.append(f"{tag}: step-1 JS is {js[0]!r}, must be exactly 0")
        if js and min(js) < 0:
            errs.append(f"{tag}: negative JS")
        # amendment 2: split check (construction pre-rounding in float64; correctly rounded FP16 storage); the post-rounding norm is reported only
        if not r.get("construction_ok", False):
            errs.append(f"{tag}: construction (float64, pre-rounding) deviates from the target norm by more than {V.CONSTR_TOL:g} "
                        f"(max {max(r.get('constr_dev_max') or [float('nan')]):.2e})")
        if not r.get("rounding_ok", False):
            errs.append(f"{tag}: stored entry is not the correctly rounded FP16 of the float64 pre-rounding entry")
        if not r["exact_ok"]:
            errs.append(f"{tag}: a path that must equal kappa^F bitwise was changed")
        if not r["natural_ok"]:
            errs.append(f"{tag}: natural entry not stored bitwise")
        if not r["routing_reset"]:
            errs.append(f"{tag}: routing overrides not reset")
        if max(r["leak"]) > 1e-5:
            errs.append(f"{tag}: subspace leak {max(r['leak']):.2e} > 1e-5")
        if predictors and "P_logit" in r and not all(x == x for x in r["P_logit"] + r["P_norm"] + r["P_val"]):
            errs.append(f"{tag}: non-finite predictor")
    return errs


def append_records(d, recs):
    E.append_records(d, recs)


def done_ids(d, tag):
    return RS1.load_done(d / f"{tag}.jsonl")


def file_sha256(p: Path):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


# ----------------------------------------------------------------------------- gates
def gpu_gates(be, tok, m, prior, t0, data, s):
    fam = family(m)
    cfg = be.model.config
    rep = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "model": m, "hf_id": MODELS[m]}
    ids0 = prompt_ids(tok, data, "train", s["gate_alpha0_train_ids"][0], be.device)
    geo = LS.cache_geometry(be, ids0)
    n_dyn = len(V.dynamic_modules(be.model))
    rep["G7_cache_geometry"] = {**geo, "n_dynamic_modules": n_dyn,
                                "pass": bool((geo["n_layers"], geo["n_kv_heads"], geo["head_dim"]) == GEOMETRY[fam] and geo["all_fp16"] and n_dyn == N_DYNAMIC[fam])}
    cap = V.QueryCapture(be.model)
    ctx = V.synthetic_ctx(cfg, be.device)
    U = ctx["U"]
    eye = torch.eye(U.shape[-1], device=U.device)
    rep["G2b_U_orthonormal_err"] = float((U.transpose(-1, -2) @ U - eye).abs().max())
    # G1: every arm at scale 0 equals pure FP16 bitwise (JS exactly 0 at every step; final cache identical)
    g1 = []
    for idx in s["gate_alpha0_train_ids"]:
        ids_t = prompt_ids(tok, data, "train", idx, be.device)
        ref = fp16_history(be, tok, ids_t)
        S_i = min(S_STEPS, len(ref) - 1)
        rl = V.reference_logits_v3(be, ids_t, ref, S_i)
        cF = V.fp16_teacher_forced_cache(be, ids_t, ref, S_i)
        for arm in V.TEST_ARMS:
            r, c0 = V.teacher_forced_arm_v3(be, ids_t, ref, S_i, arm, 0.0, V.arm_seeds(m, arm, 0.0, idx, "V3gate"), rl, ctx, cap, False, return_cache=True)
            same = all(torch.equal(a.keys, b.keys) and torch.equal(a.values, b.values) for a, b in zip(c0.layers, cF.layers))
            g1.append({"index": idx, "arm": arm, "ok": bool(same and all(x == 0.0 for x in r["js"]) and r["routing_reset"])})
    rep["G1_alpha0_bitwise"] = {"n": len(g1), "all_identical": all(x["ok"] for x in g1), "failures": [x for x in g1 if not x["ok"]]}
    # G2/G3 (all arms, alpha 1) + G2b leak + G4 determinism
    g2, leak = [], 0.0
    g4 = []
    for idx in s["gate_numerics_train_ids"]:
        recs, ref = run_problem(be, tok, data, m, idx, "train", [(a, 1.0, a) for a in V.TEST_ARMS], "V3gate", ctx, cap)
        g2 += integrity_errors(recs)
        leak = max(leak, max(max(r["leak"]) for r in recs.values()))
        ids_t = prompt_ids(tok, data, "train", idx, be.device)
        S_i = min(S_STEPS, len(ref) - 1)
        rl = V.reference_logits_v3(be, ids_t, ref, S_i)
        a1 = V.teacher_forced_arm_v3(be, ids_t, ref, S_i, "C1", 2.0, V.arm_seeds(m, "C1", 2.0, idx, "V3gate"), rl, ctx, cap, False)["js"]
        a2 = V.teacher_forced_arm_v3(be, ids_t, ref, S_i, "C1", 2.0, V.arm_seeds(m, "C1", 2.0, idx, "V3gate"), rl, ctx, cap, False)["js"]
        a3 = V.teacher_forced_arm_v3(be, ids_t, ref, S_i, "C1", 2.0, V.arm_seeds(m, "C1", 2.0, idx, "V3gate-other"), rl, ctx, cap, False)["js"]
        g4.append((a1 == a2, a1 != a3))
    rep["G2_G3_integrity_errors"] = g2
    rep["G2b_max_leak"] = leak
    rep["G4_determinism"] = {"same_seed_identical": all(x for x, _ in g4), "different_seed_differs_in_n_of": [sum(y for _, y in g4), len(g4)]}
    # G5: the v3 NAT path reproduces the engine's cache-only generation (teacher-forced on the engine's tokens: every FP16-pass argmax equals the next engine token)
    g5 = []
    for idx in s["gate_test_ids_for_G5"]:
        ids_t = prompt_ids(tok, data, "test", idx, be.device)
        e = generate_cell(be, ids_t, CACHE_ONLY, S_STEPS + 1, tok.eos_token_id, early_stop_fn=lambda ids: would_stop_at_boundary(tok, ids)).ids
        S_i = len(e) - 1
        if S_i < 1:
            continue
        rl = V.reference_logits_v3(be, ids_t, e, S_i)
        r = V.teacher_forced_arm_v3(be, ids_t, e, S_i, "NAT", 1.0, V.arm_seeds(m, "NAT", 1.0, idx, "V3gate"), rl, ctx, cap, False)
        g5.append(bool(r["top"] == e[1:S_i + 1]))
    rep["G5_nat_equals_engine_cache_only"] = {"n": len(g5), "identical": int(sum(g5)), "pass": bool(g5) and all(g5)}
    # G8: LOC/INH routing module by module, overrides reset
    ids_t = ids0
    exp = {mode: V.routing_map(be.model, mode) for mode in ("LOC", "INH")}
    g8 = {}
    for mode in ("LOC", "INH"):
        _, cache = be.run(ids_t, None, use_4bit=False)
        with V.precision_probe(be.model) as log:
            V.measure_pass(be, torch.tensor([[int(ids_t[0, -1])]], device=be.device), cache, mode)
        got = dict(log)
        g8[mode] = {"n_modules_called": len(got), "n_nf4": int(sum(got.values())), "matches_map": got == exp[mode], "reset": V.overrides_clear(be.model)}
    rep["G8_routing"] = {**g8, "pass": all(v["matches_map"] and v["reset"] for v in g8.values())}
    cap.remove()
    rep["all_passed"] = bool(rep["G7_cache_geometry"]["pass"] and rep["G1_alpha0_bitwise"]["all_identical"] and not g2 and leak <= 1e-5
                             and rep["G2b_U_orthonormal_err"] <= 1e-6 and rep["G4_determinism"]["same_seed_identical"]
                             and rep["G4_determinism"]["different_seed_differs_in_n_of"][0] == len(g4)
                             and rep["G5_nat_equals_engine_cache_only"]["pass"] and rep["G8_routing"]["pass"])
    rep["used_h_after_gates"] = (prior + time.time() - t0) / 3600
    (mdir(m) / "gates").mkdir(parents=True, exist_ok=True)
    (mdir(m) / "gates" / "gates.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({k: v for k, v in rep.items() if k != "G1_alpha0_bitwise"}, indent=1, default=float))
    print("G1:", rep["G1_alpha0_bitwise"]["all_identical"], "ALL PASSED:", rep["all_passed"])
    return rep["all_passed"]


# ----------------------------------------------------------------------------- calibration (accumulator persisted with each problem)
def aux_path(m):
    return mdir(m) / "calib" / "aux_state.pt"


def merge_aux(m, idx, st):
    p = aux_path(m)
    p.parent.mkdir(parents=True, exist_ok=True)
    cur = torch.load(p) if p.exists() else {"included": []}
    if idx in cur["included"]:
        return
    if not cur["included"]:
        cur.update({k: v for k, v in st.items()})
    else:
        for k in ("M", "d2", "absF"):
            cur[k] = cur[k] + st[k]
        for k in ("n_q", "n_meas"):
            cur[k] = cur[k] + st[k]
        for k in ("rhoK", "rhoV"):
            cur[k] = torch.cat([cur[k], st[k]])
    cur["included"] = cur["included"] + [idx]
    tmp = p.with_suffix(".tmp")
    torch.save(cur, tmp)
    tmp.replace(p)


def aux_included(m):
    p = aux_path(m)
    return set(torch.load(p)["included"]) if p.exists() else set()


def calib_step(be, tok, data, m, idx, cap):
    d = mdir(m) / "calib"
    acc = V.CalibAccumulator()
    recs, ref = run_problem(be, tok, data, m, idx, "train", V.CAL_ARMS, "V3calib", None, cap, acc=acc, predictors=False)
    errs = integrity_errors(recs, predictors=False)
    if errs:
        (mdir(m) / "integrity_failure.json").write_text(json.dumps({"index": idx, "errors": errs}))
        print("[INTEGRITY FAILURE]", errs[:3], flush=True)
        return False
    merge_aux(m, idx, acc.state())
    append_records(d, {t: r for t, r in recs.items() if idx not in done_ids(d, t)})     # a crash between merge and append leaves no duplicates
    return True


def calib_done(m, idx):
    d = mdir(m) / "calib"
    return idx in aux_included(m) and all(idx in done_ids(d, tag) for _, _, tag in V.CAL_ARMS)


def tier_step(be, tok, data, m, idx, arms, ctx, cap, stored_e1=None):
    d = mdir(m) / "tierA"
    recs, ref = run_problem(be, tok, data, m, idx, "test", arms, "V3A", ctx, cap)
    errs = integrity_errors(recs)
    if errs:
        (mdir(m) / "integrity_failure.json").write_text(json.dumps({"index": idx, "errors": errs}))
        print("[INTEGRITY FAILURE]", errs[:3], flush=True)
        return False
    append_records(d, {t: r for t, r in recs.items() if idx not in done_ids(d, t)})     # amendment 1: a crash mid-append leaves no duplicates
    h = {"index": idx, "ids": ref}
    if stored_e1:
        h["equals_stored_e1_history"] = bool(stored_e1[idx][:len(ref)] == ref)
    if idx not in done_ids(d, "FP16_history"):
        append_records(d, {"FP16_history": h})
    return True


def torn_record_files(m):
    """Amendment 1: record files whose last line is incomplete (no trailing newline) or that contain a line that is not valid JSON."""
    bad = []
    for p in sorted(mdir(m).glob("*/*.jsonl")):
        b = p.read_bytes()
        if not b:
            continue
        ok = b.endswith(bytes([10]))
        if ok:
            try:
                for line in b.decode("utf-8").splitlines():
                    if line.strip():
                        json.loads(line)
            except (ValueError, UnicodeDecodeError):
                ok = False
        if not ok:
            bad.append(p)
    return bad


def require_intact_records(m):
    bad = torn_record_files(m)
    if bad:
        sys.exit("Refusing to run: incomplete or invalid record file(s): " + ", ".join(str(p) for p in bad)
                 + ". Run tools/v3_repair_torn_line.py on each (logged repair; the problem is then rerun) - never edit records by hand.")


def mark_complete(m, ids, arms, ok):
    d = mdir(m) / "tierA"
    if ok and all(all(i in done_ids(d, tag) for _, _, tag in arms) for i in ids):
        (mdir(m) / "tierA_complete.json").write_text(json.dumps({"model": m, "n": len(ids), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))


# ----------------------------------------------------------------------------- G6 runtime gate and design
def g6(be, tok, m, prior, t0, data, s):
    cal = json.loads((mdir(m) / "calibration.json").read_text())
    ctx = V.ctx_from_calibration(cal, be.device)
    cap = V.QueryCapture(be.model)
    design = json.loads(DESIGN.read_text()) if DESIGN.exists() else {"dropped_arms": [], "n_test": len(s["tierA_ids"])}
    arms = [(a, 1.0, a) for a in V.TEST_ARMS]
    per_arm, per_hist = [], []
    for idx in s["g6_train_ids"]:
        t = time.perf_counter()
        ids_t = prompt_ids(tok, data, "train", idx, be.device)
        ref = fp16_history(be, tok, ids_t)
        per_hist.append(time.perf_counter() - t)
        recs, _ = run_problem(be, tok, data, m, idx, "train", arms, "V3g6", ctx, cap, hist=ref)
        errs = integrity_errors(recs)
        if errs:
            raise RuntimeError(f"G6 integrity failure: {errs[:3]}")
        per_arm.append(sum(r["wall_seconds"] for r in recs.values()) / len(recs))
    cap.remove()
    t_arm = sum(per_arm) / len(per_arm)
    t_over = sum(per_hist) / len(per_hist) + 1.0                         # history + reference logits per problem (reference ~ one arm's FP16 pass)
    used_h = (prior + time.time() - t0) / 3600
    rep = {"model": m, "t_arm_problem_s": t_arm, "t_overhead_problem_s": t_over, "used_h": used_h}
    k = ORDER.index(m)
    if k == 0:
        # project everything still to run, for each candidate design, with the Qwen measurement and the measured Llama/Qwen ratio
        def projection(dropped, n):
            arms_n = len([a for a in V.TEST_ARMS if a not in dropped])
            tot = used_h
            for mm in ORDER:
                ratio = 1.0 if family(mm) == "qwen" else LLAMA_TIME_RATIO
                test = n * (arms_n * t_arm + t_over) * ratio / 3600
                calib = 0.0 if mm == m else 100 * (len(V.CAL_ARMS) * t_arm + t_over) * ratio / 3600
                gates = 0.0 if mm == m else 0.35
                tot += (test + calib + gates) * 1.05
            return tot
        dropped, n = [], len(s["tierA_ids"])
        steps = [{"design": "full", "projected_total_h": projection(dropped, n)}]
        for kind, val in REDUCTIONS:
            if steps[-1]["projected_total_h"] <= CAP_HOURS:
                break
            if kind == "drop":
                dropped = dropped + [val]
            else:
                n = val
            steps.append({"design": f"{kind} {val}", "projected_total_h": projection(dropped, n)})
        ok = steps[-1]["projected_total_h"] <= CAP_HOURS
        rep.update({"projection_steps": steps, "pass": ok})
        if ok:
            design = {"dropped_arms": dropped, "n_test": n, "fixed_by": f"G6 of {m}", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      "projection_steps": steps, "t_arm_problem_s_qwen": t_arm}
            DESIGN.write_text(json.dumps(design, indent=1))
            rep["design_written"] = True
        else:
            rep["stop"] = "projection exceeds the ceiling after all pre-declared reductions: stop and ask the authors"
    else:
        exp = json.loads(DESIGN.read_text())["t_arm_problem_s_qwen"] * (1.0 if family(m) == "qwen" else LLAMA_TIME_RATIO)
        rep.update({"expected_t_arm_problem_s": exp, "drift": t_arm / exp - 1.0, "pass": bool(t_arm <= exp * (1 + DRIFT))})
        if not rep["pass"]:
            rep["stop"] = "runtime more than 15 % above projection: stop and report (no new reduction)"
    (mdir(m) / "g6.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep, indent=1, default=float))
    return rep["pass"]


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "ledger", "status", "gates", "calibrate", "g6", "tierA"])
    ap.add_argument("--model", choices=ORDER)
    ap.add_argument("--authorized-gpu-run", action="store_true")
    ap.add_argument("--cap-hours", type=float, default=CAP_HOURS)
    a = ap.parse_args()
    if a.cap_hours > CAP_HOURS:
        sys.exit(f"cap above the approved {CAP_HOURS} h is not permitted")
    configure(a.model)
    if a.cmd == "plan":
        s = make_sample()
        print({k: (len(v) if isinstance(v, list) else v) for k, v in s.items() if k not in ("note", "source")})
        return
    if a.cmd == "ledger":
        print(f"v3 cumulative GPU wall time: {RS1.ledger_seconds() / 3600:.3f} h of {a.cap_hours} h")
        return
    if a.cmd == "status":
        for m in ORDER:
            print(m, {"gates": bool(gates_passed(m)), "calibration_locked": bool(calibration_locked(m)), "g6": bool(g6_passed(m)),
                      "tierA_complete": (mdir(m) / "tierA_complete.json").exists()})
        print("design locked:", design_locked())
        return
    if not a.model:
        sys.exit("--model is required")
    m = a.model
    require_gpu_permission(a, m, a.cmd)
    require_intact_records(m)
    s = make_sample()
    data = RS1.Data()
    mdir(m).mkdir(parents=True, exist_ok=True)
    if a.cmd == "gates":
        RS1.gpu_process(f"gates:{m}", lambda be, dyn, tok, wb, prior, t0: gpu_gates(be, tok, m, prior, t0, data, s), a.cap_hours)
        return
    if a.cmd == "calibrate":
        d = mdir(m) / "calib"

        def run(be, dyn, tok, wb, prior, t0):
            cap = V.QueryCapture(be.model)
            return E.block_loop(f"calib:{m}", be, tok, wb, prior, t0, a.cap_hours, s["calibration_train_ids"], lambda i: calib_step(be, tok, data, m, i, cap),
                                lambda i: calib_done(m, i))
        RS1.gpu_process(f"calibrate:{m}", run, a.cap_hours)
        return
    if a.cmd == "g6":
        RS1.gpu_process(f"g6:{m}", lambda be, dyn, tok, wb, prior, t0: g6(be, tok, m, prior, t0, data, s), a.cap_hours)
        return
    design = json.loads(DESIGN.read_text())
    cal = json.loads((mdir(m) / "calibration.json").read_text())
    arms = arm_list(design)
    ids = s["tierA_ids"][:design["n_test"]]
    d = mdir(m) / "tierA"
    stored_e1 = E.load_fp16_ids() if m == "qwen-base" else {}

    def run(be, dyn, tok, wb, prior, t0):
        ctx = V.ctx_from_calibration(cal, be.device)
        cap = V.QueryCapture(be.model)
        ok = E.block_loop(f"tierA:{m}", be, tok, wb, prior, t0, a.cap_hours, ids, lambda i: tier_step(be, tok, data, m, i, arms, ctx, cap, stored_e1),
                          lambda i: all(i in done_ids(d, tag) for _, _, tag in arms))
        mark_complete(m, ids, arms, ok)
        return ok
    RS1.gpu_process(f"tierA:{m}", run, a.cap_hours)


if __name__ == "__main__":
    main()
