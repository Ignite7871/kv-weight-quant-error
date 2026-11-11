"""CPU validation suite for research_v2 instrumentation (no GPU, no network beyond the HF cache).

Run:  venv/Scripts/python.exe research_v2/tests/run_cpu_tests.py [--real]
  --real  additionally runs the slow end-to-end checks on real Qwen2.5-0.5B (FP16 + bitsandbytes CPU NF4).

Each test targets a failure mode the historical smoke tests could not detect (PILOT_SPEC.md section 7):
off-by-one routing, wrong-branch KV entries after crop/restore, information leakage into deployable
policies, boundary/scoring errors, RNG order dependence, and accounting inconsistencies.
"""

import math
import random
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoTokenizer  # HF before torch

import torch

import rv2  # noqa: F401  (sets sys.path for historical modules)
from rv2.backends import build_cpu_backend, real_model_cpu, tiny_qwen2, weight_bytes_per_pass
from rv2.boundary import find_boundary, score, would_stop_at_boundary
from rv2.engine import account, generate
from rv2.policies import (CascadeNF4Entropy, CascadeNF4Margin, InformationAccessError, LagEntropy, LagMargin,
                          OracleFlip, OracleFP16Entropy, OracleJS, Policy, RandomBernoulli, StaticFP16,
                          StaticNF4, StepView)
from dynamic_model import DynamicQwen                          # historical
from gsm8k_random_matched_budget_n930 import generate_with_random_routing  # historical

RESULTS = []


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


# ----------------------------------------------------------------------------- fixtures
class StubTok:
    """Minimal tokenizer so the HISTORICAL generate functions can drive a tiny model."""
    eos_token_id = 0

    def __init__(self, prompt_ids):
        self.prompt_ids = prompt_ids

    def __call__(self, prompt, return_tensors="pt"):
        ids = self.prompt_ids

        class Enc(dict):
            def to(self, device):
                return self
        return Enc(input_ids=ids)

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i) for i in ids if not (skip_special_tokens and i == self.eos_token_id))


def historical_dyn(backend, tau):
    """A DynamicQwen object wired to an existing CPU model, bypassing the CUDA-only __init__."""
    dyn = DynamicQwen.__new__(DynamicQwen)
    dyn.model, dyn.routing_state, dyn.entropy_threshold = backend.model, backend.routing_state, tau
    dyn.stats = {"fp16_tokens": 0, "4bit_tokens": 0}
    return dyn


def tiny_backend(seed=0):
    return build_cpu_backend(tiny_qwen2(seed))


PROMPTS = [torch.tensor([[5, 17, 33, 2, 99, 64, 7, 81]]), torch.tensor([[3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]])]
MAX_NEW = 48
EOS = -1  # tiny model: disable EOS so every run has full length (EOS handling tested separately)


def cache_tensors(cache):
    return [(lay.keys.clone(), lay.values.clone()) for lay in cache.layers]


def run_with_cache(backend, prompt, policy, idx=0, **kw):
    """generate() plus the final cache, via a probing pass-through wrapper."""
    captured = {}
    orig = backend.run

    def run(ids, cache, use_4bit):
        logits, c = orig(ids, cache, use_4bit)
        captured["cache"] = c
        return logits, c
    backend.run = run
    try:
        res = generate(backend, prompt, policy, MAX_NEW, EOS, idx, **kw)
    finally:
        backend.run = orig
    return res, captured["cache"]


class Scripted(Policy):
    """Test-only decide-before-step policy replaying a fixed route sequence."""
    name = "scripted"
    access = frozenset({"step"})

    def __init__(self, routes):
        self.routes = routes
        super().__init__()

    def decide(self, view):
        return self.routes[view.step - 1] == "F"


def entropy_quantiles(backend, prompt):
    res = generate(backend, prompt, StaticFP16(), MAX_NEW, EOS, 0)
    hs = sorted(res.trace["H_lag1"])
    return [hs[int(q * (len(hs) - 1))] for q in (0.2, 0.5, 0.8)]


# ----------------------------------------------------------------------------- T1 historical identity
@test
def t1_lag_entropy_matches_historical_DynamicQwen_generate():
    be = tiny_backend()
    for prompt in PROMPTS:
        taus = [-1.0, 100.0] + entropy_quantiles(be, prompt)
        for tau in taus:
            dyn = historical_dyn(be, tau)
            _, stats, hist_routes = dyn.generate("p", StubTok(prompt), MAX_NEW)
            hist_ids = dyn.last_generated_ids
            res = generate(be, prompt, LagEntropy(tau), MAX_NEW, StubTok.eos_token_id, 0)
            assert res.ids == hist_ids, f"ids differ at tau={tau}"
            assert [("F" if r == "fp16" else "N") for r in hist_routes] == res.routes, f"routes differ at tau={tau}"


@test
def t1b_random_matches_historical_generate_with_random_routing():
    be = tiny_backend(1)
    for prompt in PROMPTS:
        for p in (0.0, 0.1, 0.43, 1.0):
            _, _, hist_routes, hist_ids, _, _ = generate_with_random_routing(
                historical_dyn(be, 0.0), "p", StubTok(prompt), MAX_NEW, p, random.Random(789))
            res = generate(be, prompt, RandomBernoulli(p, "t"), MAX_NEW, StubTok.eos_token_id, 0,
                           rng_override=random.Random(789))
            assert res.ids == hist_ids and [("F" if r == "fp16" else "N") for r in hist_routes] == res.routes


@test
def t1c_static_policies_equal_extreme_thresholds():
    be = tiny_backend()
    for prompt in PROMPTS:
        assert generate(be, prompt, StaticFP16(), MAX_NEW, EOS, 0).ids == generate(be, prompt, LagEntropy(-1.0), MAX_NEW, EOS, 0).ids
        assert generate(be, prompt, StaticNF4(), MAX_NEW, EOS, 0).ids == generate(be, prompt, LagEntropy(1e9), MAX_NEW, EOS, 0).ids


@test
def t1e_static_fp16_equals_unwrapped_model_absolute_reference():
    """External reference independent of DynamicLinear/RoutingState: greedy decoding with an untouched
    copy of the model (plain nn.Linear, never wrapped) must match the engine's StaticFP16 exactly.
    Catches a globally inverted or ignored precision flag, which internal comparisons cannot."""
    import copy
    base = tiny_qwen2(0).to(torch.float16)
    plain = copy.deepcopy(base)
    be = build_cpu_backend(base)
    for prompt in PROMPTS:
        with torch.no_grad():
            out = plain(prompt, use_cache=True)
            cache, ids = out.past_key_values, [int(out.logits[0, -1].argmax())]
            for _ in range(1, MAX_NEW):
                out = plain(torch.tensor([[ids[-1]]]), past_key_values=cache, use_cache=True)
                ids.append(int(out.logits[0, -1].argmax()))
        assert generate(be, prompt, StaticFP16(), MAX_NEW, EOS, 0).ids == ids
        assert generate(be, prompt, StaticNF4(), MAX_NEW, EOS, 0).ids != ids


@test
def t1d_precisions_actually_differ():
    """Guards against a vacuous suite: FP16 and NF4 must produce different outputs on the tiny model."""
    be = tiny_backend()
    a = generate(be, PROMPTS[0], StaticFP16(), MAX_NEW, EOS, 0)
    b = generate(be, PROMPTS[0], StaticNF4(), MAX_NEW, EOS, 0)
    assert a.ids != b.ids, "tiny model too insensitive to NF4 -- tests would be vacuous"
    assert sum(x is not None and x > 0 for x in b.trace["H_lag1"]) > 0


# ----------------------------------------------------------------------------- T2 cache correctness
@test
def t2_cascade_extremes_equal_static_bitwise_including_cache():
    be = tiny_backend()
    for prompt in PROMPTS:
        r_f, c_f = run_with_cache(be, prompt, StaticFP16())
        r_n, c_n = run_with_cache(be, prompt, StaticNF4())
        r_ce, c_ce = run_with_cache(be, prompt, CascadeNF4Margin(2.0))    # margin < 2 always -> escalate
        r_cn, c_cn = run_with_cache(be, prompt, CascadeNF4Margin(-1.0))   # never escalate
        assert r_ce.ids == r_f.ids and all(p == "NF" for p in r_ce.passes)
        assert r_cn.ids == r_n.ids and all(p == "N" for p in r_cn.passes)
        for (k1, v1), (k2, v2) in zip(cache_tensors(c_ce), cache_tensors(c_f)):
            assert torch.equal(k1, k2) and torch.equal(v1, v2), "escalated cascade cache != FP16 cache"
        for (k1, v1), (k2, v2) in zip(cache_tensors(c_cn), cache_tensors(c_n)):
            assert torch.equal(k1, k2) and torch.equal(v1, v2)


@test
def t2b_oracle_extremes_equal_static_bitwise_including_cache():
    be = tiny_backend()
    for prompt in PROMPTS:
        r_f, c_f = run_with_cache(be, prompt, StaticFP16())
        r_n, c_n = run_with_cache(be, prompt, StaticNF4())
        r_of, c_of = run_with_cache(be, prompt, OracleJS(-1.0))   # always FP16
        r_on, c_on = run_with_cache(be, prompt, OracleJS(10.0))   # never FP16 -> every step restores NF4 KV
        assert r_of.ids == r_f.ids and r_on.ids == r_n.ids
        assert r_on.counts["restores"] == len(r_on.routes)
        for (a, b) in ((c_of, c_f), (c_on, c_n)):
            for (k1, v1), (k2, v2) in zip(cache_tensors(a), cache_tensors(b)):
                assert torch.equal(k1, k2) and torch.equal(v1, v2), "oracle cache != static cache"


@test
def t2c_mixed_routes_equal_single_pass_replay_bitwise():
    """The strongest cache test: for intermediate thresholds, replaying the oracle's / cascade's route
    sequence with a single-pass router must give identical tokens AND identical final cache."""
    be = tiny_backend()
    for prompt in PROMPTS:
        r0 = generate(be, prompt, OracleFlip(), MAX_NEW, EOS, 0)
        js_sorted = sorted(x for x in r0.trace["js"] if x is not None)
        m_sorted = sorted(x for x in r0.trace["M_nf4"] if x is not None)
        pols = [OracleJS(js_sorted[len(js_sorted) // 2]), OracleFlip(),
                OracleFP16Entropy(sorted(r0.trace["H_fp16"])[len(js_sorted) // 2]),
                CascadeNF4Margin(m_sorted[len(m_sorted) // 3]), CascadeNF4Entropy(1.0)]
        for pol in pols:
            r, c = run_with_cache(be, prompt, pol)
            mixed = 0 < r.routes.count("F") < len(r.routes)
            r2, c2 = run_with_cache(be, prompt, Scripted(r.routes))
            assert r2.ids == r.ids, f"{pol.name}: replay ids differ"
            for (k1, v1), (k2, v2) in zip(cache_tensors(c), cache_tensors(c2)):
                assert torch.equal(k1, k2) and torch.equal(v1, v2), f"{pol.name}: replay cache differs"
            if pol.name.startswith(("oracle_js", "cascade_nf4_margin")):
                assert mixed, f"{pol.name}: threshold did not produce mixed routes; test would be weak"


@test
def t2d_oracle_flip_semantics():
    be = tiny_backend()
    for prompt in PROMPTS:
        r = generate(be, prompt, OracleFlip(), MAX_NEW, EOS, 0)
        for s, route in enumerate(r.routes):
            differ = r.trace["top_fp16"][s] != r.trace["top_nf4"][s]
            assert (route == "F") == differ
            assert r.ids[s + 1] == r.trace["top_fp16"][s], "flip oracle token must equal FP16 argmax"


# ----------------------------------------------------------------------------- T3 leakage
@test
def t3_stepview_blocks_undeclared_fields():
    v = StepView(frozenset({"prev_logits"}), prev_logits=1, fp16_logits=2, nf4_logits=3)
    assert v.prev_logits == 1
    for f in ("fp16_logits", "nf4_logits", "rng", "step"):
        try:
            getattr(v, f)
            raise AssertionError(f"{f} leaked")
        except InformationAccessError:
            pass


@test
def t3b_deployable_policies_cannot_declare_fp16_current():
    class Bad(Policy):
        name, kind, access = "bad", "cascade", frozenset({"fp16_logits"})
    class Bad2(Policy):
        name, kind, access = "bad2", "pre", frozenset({"nf4_logits"})
    for cls in (Bad, Bad2):
        try:
            cls()
            raise AssertionError(f"{cls.__name__} accepted")
        except AssertionError as e:
            if "accepted" in str(e):
                raise
    for pol in (LagEntropy(1), LagMargin(.5), RandomBernoulli(.1, "x"), CascadeNF4Margin(.3), StaticFP16(), StaticNF4()):
        assert pol.deployable and "fp16_logits" not in pol.access
    for pol in (OracleJS(.1), OracleFlip(), OracleFP16Entropy(1.0)):
        assert not pol.deployable


@test
def t3c_pass_order_never_computes_fp16_before_deployable_decision():
    be = tiny_backend()
    be.log_calls = True
    for pol in (LagEntropy(1.0), LagMargin(0.3), RandomBernoulli(0.3, "s"), CascadeNF4Margin(0.3)):
        be.call_log.clear()
        r = generate(be, PROMPTS[0], pol, MAX_NEW, EOS, 0)
        calls = be.call_log[1:]  # drop prefill
        assert be.call_log[0] == "P"
        flat = "".join(r.passes)
        assert "".join(calls) == flat, "backend calls do not match recorded passes"
        if pol.kind == "pre":
            assert all(p in ("N", "F") for p in r.passes)
        else:
            assert all(p in ("N", "NF") for p in r.passes)  # FP16 only ever AFTER the NF4 pass + decision
    be.log_calls = False


@test
def t3d_cascade_decisions_invariant_to_fp16_weights_until_first_escalation():
    """Counterfactual leak test: corrupting the FP16 weights AFTER the (always-FP16, shared) prefill
    cannot change any cascade decision made up to and including the first escalation, nor any token
    before it (those may depend only on NF4 outputs and history)."""
    be = tiny_backend()
    # Until its first escalation the cascade follows the static-NF4 trajectory, so choose tau between
    # the running minimum of NF4 margins and the first later new minimum (escalation at step j >= 3).
    m = generate(be, PROMPTS[1], StaticNF4(), MAX_NEW, EOS, 0).trace["M_nf4"]
    j = next(k for k in range(3, len(m)) if m[k] < min(m[:k]))
    pol = CascadeNF4Margin((m[j] + min(m[:j])) / 2)
    r_clean = generate(be, PROMPTS[1], pol, MAX_NEW, EOS, 0)
    assert "F" in r_clean.routes and r_clean.routes.index("F") > 0, "threshold gives no informative escalation"
    first = r_clean.routes.index("F")
    from dynamic_linear import DynamicLinear
    saved = [(m, m.fp16_weight.clone()) for m in be.model.modules() if isinstance(m, DynamicLinear)]
    orig_run = be.run

    def run(ids, cache, use_4bit):          # corrupt FP16 weights right after the prefill
        out = orig_run(ids, cache, use_4bit)
        if cache is None:
            for m, _ in saved:
                m.fp16_weight.mul_(-3.0)
        return out
    be.run = run
    try:
        r_bad = generate(be, PROMPTS[1], pol, MAX_NEW, EOS, 0)
    finally:
        be.run = orig_run
        for m, w in saved:
            m.fp16_weight.copy_(w)
    assert r_bad.ids != r_clean.ids, "corruption had no effect; test would be vacuous"
    assert r_bad.routes[: first + 1] == r_clean.routes[: first + 1]
    assert r_bad.ids[: first + 1] == r_clean.ids[: first + 1]


# ----------------------------------------------------------------------------- T4 RNG
@test
def t4_random_streams_order_independent_and_distinct():
    pol = RandomBernoulli(0.3, "seedA")
    a = [pol.make_rng(17).random() for _ in range(5)]
    _ = [pol.make_rng(i).random() for i in range(10)]
    b = [pol.make_rng(17).random() for _ in range(5)]
    assert a == b
    assert [pol.make_rng(18).random() for _ in range(5)] != a
    assert [RandomBernoulli(0.3, "seedB").make_rng(17).random() for _ in range(5)] != a
    rng = pol.make_rng(1)
    frac = sum(rng.random() < 0.3 for _ in range(20000)) / 20000
    assert abs(frac - 0.3) < 0.015


# ----------------------------------------------------------------------------- T5 accounting
@test
def t5_accounting_consistency():
    be = tiny_backend()
    for pol in (StaticFP16(), StaticNF4(), LagEntropy(1.0), CascadeNF4Margin(0.3), OracleFlip()):
        r = generate(be, PROMPTS[0], pol, MAX_NEW, EOS, 0)
        n = len(r.routes)
        mask = [1] * (n // 2 + 1) + [0] * (n - n // 2)       # synthetic boundary
        a = account(r, mask, weight_bytes_per_pass(be.model), prompt_len=PROMPTS[0].shape[1])
        assert a["fp16_steps_full"] + a["nf4_steps_full"] == n == MAX_NEW - 1
        assert a["routed_steps_answer"] == n // 2
        assert a["fp16_steps_answer"] <= a["fp16_steps_full"]
        if pol.kind == "oracle":
            assert a["nf4_passes"] == a["fp16_passes"] == n and a["recomputes"] == n
        if pol.kind == "cascade":
            assert a["nf4_passes"] == n and a["fp16_passes"] == a["fp16_steps_full"] == a["recomputes"]
        if pol.kind == "pre":
            assert a["nf4_passes"] + a["fp16_passes"] == n and a["recomputes"] == 0
    wb = weight_bytes_per_pass(be.model)
    assert wb["nf4_routed"] < wb["fp16_routed"] and wb["unrouted"] > 0


@test
def t5c_cost_charges_every_executed_pass_and_cache_op():
    """Independent recomputation of total cost from the backend's actual call log and cache lengths:
    every NF4 pass, every FP16 recomputation and every oracle snapshot/restore must be charged."""
    be = tiny_backend()
    wb = weight_bytes_per_pass(be.model)
    kvb = wb["kv_bytes_per_token"]
    P = PROMPTS[0].shape[1]
    w = {"N": wb["nf4_routed"] + wb["unrouted"], "F": wb["fp16_routed"] + wb["unrouted"]}
    m = generate(be, PROMPTS[0], StaticNF4(), MAX_NEW, EOS, 0).trace["M_nf4"]
    for pol in (StaticFP16(), StaticNF4(), LagEntropy(1.0), CascadeNF4Margin(sorted(m)[len(m) // 3]),
                OracleFlip(), OracleJS(0.0)):
        be.log_calls = True
        be.call_log.clear()
        r = generate(be, PROMPTS[0], pol, MAX_NEW, EOS, 0)
        be.log_calls = False
        calls = be.call_log[1:]
        # rebuild per-step call groups from the raw log (one group per routed step)
        groups, i = [], 0
        for p in r.passes:
            groups.append("".join(calls[i:i + len(p)])); i += len(p)
        assert i == len(calls) and groups == r.passes
        exp_w = sum(w[c] for c in calls)
        exp_kv = sum((3 * (P + s - 1) + 2) * kvb * len(g) for s, g in enumerate(groups, start=1))
        exp_ops = 0
        if pol.kind == "oracle":
            exp_ops = sum(2 * kvb + ((2 * (P + s - 1) + 1) * kvb if rt == "N" else 0)
                          for s, rt in enumerate(r.routes, start=1))
            assert r.counts["restores"] == r.routes.count("N")
        a = account(r, [1] * (len(r.routes) + 1), wb, prompt_len=P)
        assert (a["cost_weight_bytes"], a["cost_kv_bytes"], a["cost_cacheop_bytes"]) == (exp_w, exp_kv, exp_ops), pol.name
        if pol.kind == "cascade":
            esc = r.routes.count("F")
            assert 0 < esc < len(r.routes) and a["recomputes"] == esc and a["fp16_passes"] == esc
            assert a["nf4_passes"] == len(r.routes)          # the discarded NF4 pass is still charged
        if pol.kind == "oracle":
            assert a["nf4_passes"] == a["fp16_passes"] == len(r.routes)


@test
def t3e_answer_boundary_cannot_influence_generation():
    """Boundaries are retrospective: the engine and policies never import the boundary module, the
    pilot driver never passes an early-stop function, and computing a boundary changes nothing."""
    import inspect
    import rv2.engine as eng
    import rv2.policies as pol_mod
    for mod in (eng, pol_mod):
        src = inspect.getsource(mod)
        assert "boundary" not in src.replace("decision boundary", ""), f"{mod.__name__} references boundary code"
    import run_pilot
    assert "early_stop_fn" not in inspect.getsource(run_pilot)
    be = tiny_backend()
    a = generate(be, PROMPTS[0], LagEntropy(1.0), MAX_NEW, EOS, 0)
    find_boundary(qtok(), a.ids, -1, MAX_NEW)   # tiny-model ids decoded with the real tokenizer
    b = generate(be, PROMPTS[0], LagEntropy(1.0), MAX_NEW, EOS, 0)
    assert a.ids == b.ids and a.routes == b.routes


@test
def t8_tie_rule_and_cost_matching():
    from run_pilot import _cost_matched_p, tie_rule_threshold
    # '>' : FP16 iff v > tau. Values with a tie block: share 0.2 at tau=0, 0.0 at tau=1 -> both 0.1 away;
    # rule prefers the LOWER share -> tau = 1 (share 0).
    assert tie_rule_threshold([0.0] * 8 + [1.0, 1.0], 0.1, ">") == (1.0, 0.0)
    assert tie_rule_threshold([0.0] * 8 + [1.0, 2.0], 0.1, ">") == (1.0, 0.1)
    # '<' : FP16 iff v < tau; exact hit at tau = 0.5 (one value below)
    assert tie_rule_threshold([0.1, 0.5, 0.7, 0.9] + [1.0] * 6, 0.1, "<") == (0.5, 0.1)
    assert tie_rule_threshold([1.0] * 10, 0.3, ">") == (1.0, 0.0)   # all tied: lower share wins
    # cost matching: for a cascade with escalation fraction e and negligible KV terms,
    # p = e * W_F / (W_F - W_N); check the implementation against a hand computation on a real run.
    be = tiny_backend()
    wb = weight_bytes_per_pass(be.model)
    m = generate(be, PROMPTS[0], StaticNF4(), MAX_NEW, EOS, 0).trace["M_nf4"]
    r = generate(be, PROMPTS[0], CascadeNF4Margin(sorted(m)[len(m) // 4]), MAX_NEW, EOS, 0)
    P = PROMPTS[0].shape[1]
    rec = {"accounting": account(r, [1] * (len(r.routes) + 1), wb, prompt_len=P), "prompt_tokens": P}
    n, esc = len(r.routes), r.routes.count("F")
    wN, wF, kvb = wb["nf4_routed"] + wb["unrouted"], wb["fp16_routed"] + wb["unrouted"], wb["kv_bytes_per_token"]
    extra_kv = sum((3 * (P + s - 1) + 2) * kvb for s, rt in enumerate(r.routes, start=1) if rt == "F")
    expected = (esc * wF + extra_kv) / (n * (wF - wN))
    assert abs(_cost_matched_p([rec], wb) - expected) < 1e-12 and expected > esc / n


@test
def t9_lean_mode_is_token_and_route_identical():
    """instrument=False must change nothing but timing/logging, for every policy kind."""
    be = tiny_backend()
    m = generate(be, PROMPTS[0], StaticNF4(), MAX_NEW, EOS, 0).trace["M_nf4"]
    pols = [StaticFP16(), StaticNF4(), LagEntropy(1.0), LagMargin(0.3), RandomBernoulli(0.3, "lean"),
            CascadeNF4Margin(sorted(m)[len(m) // 3]), OracleFlip(), OracleJS(0.0)]
    for prompt in PROMPTS:
        for pol in pols:
            a = generate(be, prompt, pol, MAX_NEW, EOS, 0)
            b = generate(be, prompt, pol, MAX_NEW, EOS, 0, instrument=False)
            assert (a.ids, a.routes, a.passes, a.counts) == (b.ids, b.routes, b.passes, b.counts), pol.name
            assert all(len(v) == 0 for v in b.trace.values()) and b.timing["total"] > 0


@test
def t5b_eos_handling_matches_historical():
    """EOS: generation stops right after the EOS token, and an EOS produced by the prefill skips
    the loop -- same as DynamicQwen.generate. Uses the historical function as the reference."""
    be = tiny_backend()
    ref = generate(be, PROMPTS[0], StaticNF4(), MAX_NEW, EOS, 0)
    for eos in (ref.ids[0], ref.ids[5], ref.ids[20]):
        tok = StubTok(PROMPTS[0]); tok.eos_token_id = eos
        dyn = historical_dyn(be, 100.0)
        dyn.generate("p", tok, MAX_NEW)
        r = generate(be, PROMPTS[0], LagEntropy(100.0), MAX_NEW, eos, 0)
        assert r.ids == dyn.last_generated_ids and r.ids[-1] == eos


# ----------------------------------------------------------------------------- T6 boundary (real tokenizer)
_TOK = None


def qtok():
    global _TOK
    if _TOK is None:
        _TOK = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    return _TOK


def ids_of(text):
    return qtok()(text, add_special_tokens=False).input_ids


@test
def t6_boundary_cases():
    tok = qtok()
    eos = tok.eos_token_id
    prob = {"answer": "blah #### 18"}
    cases = [
        (" She has 9+9=<<9+9=18>>18 eggs.\n#### 18\n\nQuestion: A man buys 3 apples", "drift_marker", True),
        (" 18 eggs. #### 18\n question : what", "drift_marker", True),   # case-insensitive, spaces
        (" The answer is 18.", "eos", True),
        ("\nQuestion: immediately drifting", "drift_marker", False),
        (" Cost is 18 €. 😀 Done.\n#### 18\nQuestion: x", "drift_marker", True),
    ]
    for text, expected_type, solvable in cases:
        ids = ids_of(text) + ([eos] if expected_type == "eos" else [])
        b = find_boundary(tok, ids, eos, max_new_tokens=256)
        assert b.boundary_type == expected_type, (text, b.boundary_type)
        full = tok.decode(ids, skip_special_tokens=True)
        if expected_type == "drift_marker":
            assert full[b.boundary_char:].lstrip(" ").startswith("\n"), (text, full[b.boundary_char:][:20])
        sc = score(prob, full, b)
        assert sc["truncated_extraction_identical"]
        assert (sc["status"] == "SOLVED") == solvable, (text, sc)
        # every token wholly after the boundary is excluded; every token starting before is included
        assert sum(b.answer_token_mask) == b.n_answer_tokens
        assert b.decision_char <= b.boundary_char and b.n_decision_tokens <= b.n_answer_tokens
        assert all(a >= d for a, d in zip(b.answer_token_mask, b.decision_token_mask))
        if "####" in full[:b.boundary_char]:
            assert b.decision_type == "hash_number" and sc["decision_extraction_identical"]
            assert full[:b.decision_char].rstrip().endswith("18"), full[:b.decision_char][-15:]
        else:
            assert b.decision_type == "no_hash" and b.decision_char == b.boundary_char
        if expected_type == "eos":
            assert b.answer_token_mask[-1] == 0  # EOS itself never counts
    # cap: no marker, no EOS, length == max_new_tokens
    ids = ids_of(" " + "and " * 300)[:256]
    b = find_boundary(tok, ids, eos, max_new_tokens=256)
    assert b.boundary_type == "cap" and b.n_answer_tokens == 256


@test
def t6b_early_stop_is_prefix_stable():
    """If generation were stopped as soon as the marker appears, the boundary (and thus the scored
    text) must equal the one computed on the full generation. Checked over all token prefixes."""
    tok = qtok()
    text = " Step 1: 3*4=12.\n Step 2: 12+6=18.\n#### 18\n\n  Question: Tom has 5 cats and 2 dogs. How many?\nAnswer: 7"
    ids = ids_of(text)
    full_b = find_boundary(tok, ids, tok.eos_token_id, 256)
    stop_k = next(k for k in range(1, len(ids) + 1) if would_stop_at_boundary(tok, ids[:k]))
    pre_b = find_boundary(tok, ids[:stop_k], tok.eos_token_id, 256)
    assert pre_b.boundary_type == "drift_marker" and pre_b.boundary_char == full_b.boundary_char
    assert pre_b.answer_token_mask == full_b.answer_token_mask[:stop_k]


@test
def t6c_offset_inconsistency_is_flagged_not_hidden():
    tok = qtok()
    ids = ids_of(" 😀😀 emoji #### 3")
    b = find_boundary(tok, ids + [tok.eos_token_id], tok.eos_token_id, 256)
    # whether or not the emoji splits across tokens, the flag must be a bool and masks well-formed
    assert isinstance(b.offsets_consistent, bool) and len(b.answer_token_mask) == len(ids) + 1


# ----------------------------------------------------------------------------- T7 real model (slow)
def real_tests():
    @test
    def t7_real_qwen_cpu_lag_entropy_matches_historical():
        tok = qtok()
        be = build_cpu_backend(real_model_cpu())
        prompt = torch.tensor([tok("Question: Tom has 3 apples and buys 4 more. How many?\nAnswer:").input_ids])
        n = 12
        for tau in (0.5, 1.943):
            dyn = historical_dyn(be, tau)
            class T(StubTok):
                eos_token_id = tok.eos_token_id
            dyn.generate("p", T(prompt), n)
            r = generate(be, prompt, LagEntropy(tau), n, tok.eos_token_id, 0)
            assert r.ids == dyn.last_generated_ids
        r_f, c_f = None, None
        global MAX_NEW
        old, MAX_NEW = MAX_NEW, n
        try:
            r_f, c_f = run_with_cache(be, prompt, StaticFP16())
            r_c, c_c = run_with_cache(be, prompt, CascadeNF4Margin(2.0))
            r_o, c_o = run_with_cache(be, prompt, OracleJS(10.0))
            r_n, c_n = run_with_cache(be, prompt, StaticNF4())
            assert r_c.ids == r_f.ids and r_o.ids == r_n.ids
            for (a, b) in ((c_c, c_f), (c_o, c_n)):
                for (k1, v1), (k2, v2) in zip(cache_tensors(a), cache_tensors(b)):
                    assert torch.equal(k1, k2) and torch.equal(v1, v2)
            assert all(isinstance(m, float) for m in r_o.trace["js"])
        finally:
            MAX_NEW = old
        wb = weight_bytes_per_pass(be.model)
        print(f"    real-model weight bytes per pass: {wb}")
    t7_real_qwen_cpu_lag_entropy_matches_historical()


def main():
    tests = [v for k, v in globals().items() if k.startswith("t") and k[1].isdigit() and callable(v)]
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
