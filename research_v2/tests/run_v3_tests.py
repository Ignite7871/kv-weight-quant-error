"""CPU validation for v3 (PROTOCOL_V3.md section 10). Runs every test on a tiny random Qwen2 (4 query / 2 KV heads, head_dim 16) and a tiny random GQA
Llama (8 query / 2 KV heads, head_dim 64), both with the real DynamicLinear / bitsandbytes NF4 path. LOGIC only (CPU NF4 numerics differ from CUDA).
Checks are computed independently of the code under test wherever possible (own hooks, own loops, own module maps).

  venv/Scripts/python.exe research_v2/tests/run_v3_tests.py            tiny models (both families)
  venv/Scripts/python.exe research_v2/tests/run_v3_tests.py --real     + real Qwen2.5-0.5B and Llama-3.2-1B on CPU, a few steps
  venv/Scripts/python.exe research_v2/tests/run_v3_tests.py --static   + locked-file checks (E1 / E1L locks verify)
"""

from __future__ import annotations

import inspect
import json
import math
import struct
import subprocess
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
RV = HERE.parent
sys.path.insert(0, str(RV))

import run_v3 as R                    # noqa: E402  (HF before torch)
import e1l_support as LS              # noqa: E402

import torch                          # noqa: E402

import rv2.controls as C              # noqa: E402
import rv3.core as V                  # noqa: E402
from rv2.backends import build_cpu_backend, tiny_qwen2  # noqa: E402
from rv2.stage1 import CACHE_ONLY, generate_cell        # noqa: E402

DEV = "cpu"
FAMILIES = {"qwen": lambda: build_cpu_backend(tiny_qwen2(0)), "llama": lambda: LS.tiny_backend_cpu(0)}
PROMPT = torch.tensor([[5, 17, 33, 2, 99, 64, 7, 81, 3, 1, 4]])


def install(device):
    """Run the same suite on another device (the GPU mutation check uses 'cuda')."""
    global DEV, PROMPT
    DEV = device
    PROMPT = PROMPT.to(device)
    FAMILIES["qwen"] = lambda: LS.build_backend(tiny_qwen2(0), device)
    FAMILIES["llama"] = lambda: LS.build_backend(LS.tiny_llama(0), device)
    _BE.clear()


def tok(x):
    return torch.tensor([[x]], device=DEV)
REF = [3, 9, 27, 81, 1, 5, 7, 11, 13, 17, 19]
S = 8
RESULTS = []
_BE = {}


def be(fam):
    if fam not in _BE:
        _BE[fam] = FAMILIES[fam]()
    return _BE[fam]


def t(fn):
    TESTS.append(fn)
    return fn


TESTS = []


def run_arm(b, arm, alpha=1.0, ctx=None, cap=None, preds=True, seeds=None, prompt=None, ref=REF, return_cache=False):
    prompt = PROMPT if prompt is None else prompt
    ctx = ctx if ctx is not None else V.synthetic_ctx(b.model.config, DEV)
    rl = V.reference_logits_v3(b, prompt, ref, S)
    seeds = seeds or V.arm_seeds("test", arm, alpha, 0, "T")
    return V.teacher_forced_arm_v3(b, prompt, ref, S, arm, alpha, seeds, rl, ctx, cap, preds, return_cache=return_cache)


def capture_entries(arms_out):
    """Wrap V.stored_entry_v3 (as called by the step) to record (arm, alpha, kF, kM, stored, float64 pre-rounding entry)."""
    orig = V.stored_entry_v3

    def w(arm, alpha, kF, kM, gen, vgen, ctx):
        out, info = orig(arm, alpha, kF, kM, gen, vgen, ctx)
        arms_out.append((arm, alpha, kF.clone(), kM.clone(), out.clone(), info["pre"].clone()))
        return out, info
    return orig, w


def struct_fp16_bits(u):
    """Test oracle (independent of both v3 implementations): Python's float -> binary16 packing, round to nearest, ties to even."""
    return torch.tensor([struct.unpack("<h", struct.pack("<e", float(x)))[0] for x in u.double().flatten().cpu()],
                        dtype=torch.int16).view(u.shape)


def rounding_cases():
    """float64 values covering every regime: random magnitudes, signed zeros, subnormals, exact FP16 ties and near-ties (double-rounding traps)."""
    g = torch.Generator().manual_seed(7)
    x = (torch.randn(4000, generator=g, dtype=torch.float64) * torch.exp(torch.randn(4000, generator=g, dtype=torch.float64) * 3)).clamp(-6e4, 6e4)
    h = torch.arange(0, 0x7BFF, 37, dtype=torch.int32).to(torch.int16).view(torch.float16).double()
    mid = (h[:-1] + h[1:]) / 2                                   # exact ties (representable in float64)
    eps = mid.abs() * 2 ** -40 + 1e-300
    traps = torch.tensor([1 + 2 ** -11 + 2 ** -30, 1 + 2 ** -11 - 2 ** -30, 2 ** -24 * 1.5, 2 ** -25, 0.0, -0.0], dtype=torch.float64)
    return torch.cat([x, mid, -mid, mid + eps, mid - eps, -(mid + eps), traps])


# ----------------------------------------------------------------------------- tests (each takes the family name)
@t
def v01_alpha0_bitwise_every_arm(fam):
    b = be(fam)
    cF = V.fp16_teacher_forced_cache(b, PROMPT, REF, S)
    for arm in V.TEST_ARMS:
        r, c0 = run_arm(b, arm, 0.0, preds=False, return_cache=True)
        assert all(x == 0.0 for x in r["js"]), f"{arm}: JS not exactly 0 at alpha 0"
        assert all(torch.equal(a.keys, c.keys) and torch.equal(a.values, c.values) for a, c in zip(c0.layers, cF.layers)), f"{arm}: cache differs from FP16 at alpha 0"
        assert V.overrides_clear(b.model), f"{arm}: overrides left set"


@t
def v02_stored_entries_norms_and_paths_independent(fam):
    b = be(fam)
    ctx = V.synthetic_ctx(b.model.config, DEV, rho_star=0.07)
    rec = []
    orig, w = capture_entries(rec)
    V.stored_entry_v3 = w
    try:
        for arm in V.TEST_ARMS:
            run_arm(b, arm, 1.0, ctx, preds=False)
    finally:
        V.stored_entry_v3 = orig
    assert len(rec) == len(V.TEST_ARMS) * S
    for arm, alpha, kF, kM, out, pre in rec:
        assert out.dtype == torch.float16 and torch.equal(out.view(torch.int16).cpu(), struct_fp16_bits(pre)), \
            f"{arm}: stored entry is not the correctly rounded FP16 of the pre-rounding entry"
        F, M, O = kF.double(), kM.double(), pre.double()
        d, D = O - F, M - F
        nF = F.norm(dim=-1)
        K, Vv = slice(0, None, 2), slice(1, None, 2)
        if arm in ("NAT", "LOC", "INH"):
            assert torch.equal(out, kM), f"{arm}: not the measured entry bitwise"
        elif arm == "NAT-K":
            assert torch.equal(out[K], kM[K]) and torch.equal(out[Vv], kF[Vv]), "NAT-K paths wrong"
        elif arm == "NAT-V":
            assert torch.equal(out[Vv], kM[Vv]) and torch.equal(out[K], kF[K]), "NAT-V paths wrong"
        else:
            tgt = torch.zeros_like(nF)
            if arm in ("C1", "C2", "CV"):
                tgt = D.norm(dim=-1)
            elif arm in ("QA", "QO"):
                tgt = D.norm(dim=-1)
            elif arm == "NAT-K*":
                tgt[K] = 0.07 * nF[K] * (D[K].norm(dim=-1) > 0)
            elif arm == "NAT-V*":
                tgt[Vv] = 0.07 * nF[Vv] * (D[Vv].norm(dim=-1) > 0)
            elif arm == "C1-K":
                tgt[K] = D[K].norm(dim=-1)
            elif arm == "C1-V":
                tgt[Vv] = D[Vv].norm(dim=-1)
            zero_rows = {"NAT-K*": Vv, "C1-K": Vv, "NAT-V*": K, "C1-V": K}.get(arm)
            if zero_rows is not None:
                assert torch.equal(out[zero_rows], kF[zero_rows]), f"{arm}: other path changed"
            dev = (d.norm(dim=-1) - tgt).abs()
            pos = tgt > 0
            assert bool((dev[pos] <= 1e-6 * tgt[pos]).all()) and bool((d.norm(dim=-1)[~pos] == 0).all()), \
                f"{arm}: pre-rounding per-slot norm off (max rel dev {float((dev[pos] / tgt[pos]).max()):.3e})"
            if arm in ("NAT-K*", "NAT-V*"):
                rows = K if arm == "NAT-K*" else Vv
                cos = torch.nn.functional.cosine_similarity(d[rows], D[rows], dim=-1)
                m = D[rows].norm(dim=-1) > 0
                assert bool((cos[m] > 0.99).all()), f"{arm}: direction is not the natural Delta direction"
            if arm == "C2":
                assert bool(((d.abs() - D.abs()).abs() <= 1e-12 * D.abs().clamp_min(1e-30)).all()), "C2: per-coordinate magnitudes not kept"


@t
def v03_qa_qo_geometry_prerounding(fam):
    b = be(fam)
    cfg = b.model.config
    ctx = V.synthetic_ctx(cfg, DEV, seed=3)
    U = ctx["U"]
    L, h, hd = U.shape[0], U.shape[1], U.shape[2]
    assert U.shape[-1] == 8, "r must be 8"
    assert float((U.transpose(-1, -2) @ U - torch.eye(8, device=DEV)).abs().max()) < 1e-6, "U not orthonormal"
    g = torch.Generator().manual_seed(1)
    kF = torch.randn(2 * L, h, hd, generator=g).to(DEV)
    kM = kF + 0.05 * torch.randn(2 * L, h, hd, generator=g).to(DEV)
    outs = {}
    for arm in ("QA", "QO"):
        gen, vgen = C.make_generator(DEV, 11), C.make_generator(DEV, 12)        # same streams for both arms in this test
        out, info = V.stored_entry_v3(arm, 1.0, kF, kM, gen, vgen, ctx)
        outs[arm] = info["pre"]                                                 # amendment 2: geometry judged on the float64 pre-rounding entry
        dK = (info["pre"] - kF.double())[0::2]
        Ud = U.double()
        proj = torch.einsum("lhdr,lhr->lhd", Ud, torch.einsum("lhdr,lhd->lhr", Ud, dK))
        off = (dK - proj) if arm == "QA" else proj
        ratio = off.norm(dim=-1) / dK.norm(dim=-1)
        assert float(ratio.max()) < 1e-5, f"{arm}: key perturbation not in the declared subspace ({float(ratio.max()):.2e})"
        assert torch.allclose(dK.norm(dim=-1), (kM - kF)[0::2].double().norm(dim=-1), rtol=1e-5), f"{arm}: key norm not natural"
    assert torch.equal(outs["QA"][1::2], outs["QO"][1::2]), "QA and QO value draws differ with the same value stream"
    sa, sv = V.arm_seeds("m", "QA", 1.0, 3, "T"), V.arm_seeds("m", "QO", 1.0, 3, "T")
    assert sa[1] == sv[1] and sa[0] != sv[0], "QA/QO must share the value stream only"


@t
def v04_only_newest_entry_changes(fam):
    b = be(fam)
    orig = V.controlled_step_v3

    def w(backend, cur, cache, *a, **k):
        L = cache.get_seq_length()
        snap = [(x.keys[..., :L, :].clone(), x.values[..., :L, :].clone()) for x in cache.layers]
        out = orig(backend, cur, cache, *a, **k)
        assert cache.get_seq_length() == L + 1, "cache length must grow by one"
        for (k0, v0), lay in zip(snap, cache.layers):
            assert torch.equal(lay.keys[..., :L, :], k0) and torch.equal(lay.values[..., :L, :], v0), "an earlier entry changed"
        return out
    V.controlled_step_v3 = w
    try:
        for arm in V.TEST_ARMS:
            run_arm(b, arm, 1.0, preds=False)
    finally:
        V.controlled_step_v3 = orig


@t
def v05_loc_inh_routing_module_by_module(fam):
    b = be(fam)
    names = [n for n, m in b.model.named_modules() if type(m).__name__ == "DynamicLinear"]
    kv = {n for n in names if n.endswith(".k_proj") or n.endswith(".v_proj")}
    n_layers = b.model.config.num_hidden_layers
    assert len(kv) == 2 * n_layers and len(names) == 7 * n_layers, "unexpected module set"
    _, cache = b.run(PROMPT, None, use_4bit=False)
    for mode, want in (("LOC", {n: n in kv for n in names}), ("INH", {n: n not in kv for n in names})):
        L = cache.get_seq_length()
        with V.precision_probe(b.model) as log:
            V.measure_pass(b, tok(7), cache, mode)
        cache.crop(L)
        got = dict(log)
        assert got == want, f"{mode}: precision map differs from the declared one"
        assert all(m.layer_force_4bit is None for n, m in b.model.named_modules() if type(m).__name__ == "DynamicLinear"), f"{mode}: override left set"
    with V.precision_probe(b.model) as log:
        L = cache.get_seq_length()
        V.measure_pass(b, tok(7), cache, "NF4")
        cache.crop(L)
    assert all(v for _, v in log), "NF4 measurement must use NF4 everywhere"
    # independent: the LOC entry equals a pass with overrides set by this test
    for mode, want in (("LOC", {n: n in kv for n in names}), ("INH", {n: n not in kv for n in names})):
        L = cache.get_seq_length()
        V.measure_pass(b, tok(9), cache, mode)
        e1 = [(x.keys[..., L:, :].clone(), x.values[..., L:, :].clone()) for x in cache.layers]
        cache.crop(L)
        mods = dict(b.model.named_modules())
        for n in names:
            mods[n].layer_force_4bit = want[n]
        b.run(tok(9), cache, False)
        for n in names:
            mods[n].layer_force_4bit = None
        e2 = [(x.keys[..., L:, :].clone(), x.values[..., L:, :].clone()) for x in cache.layers]
        cache.crop(L)
        assert all(torch.equal(a, c) and torch.equal(x, y) for (a, x), (c, y) in zip(e1, e2)), f"{mode}: entry differs from the independent routing"


@t
def v06_nat_equals_engine_cache_only(fam):
    b = be(fam)
    e = generate_cell(b, PROMPT, CACHE_ONLY, S + 1, None).ids
    rl = V.reference_logits_v3(b, PROMPT, e, S)
    r = V.teacher_forced_arm_v3(b, PROMPT, e, S, "NAT", 1.0, V.arm_seeds("t", "NAT", 1.0, 0, "T"), rl, None, None, False)
    assert r["top"] == e[1:S + 1], "NAT teacher-forced on the engine's cache-only tokens must reproduce them"


@t
def v07_seeds_label_only(fam):
    b = be(fam)
    sig = list(inspect.signature(V.seed_for).parameters)
    assert sig == ["model_label", "arm", "alpha", "index", "tier", "repeat", "master"], sig
    a = V.seed_for("m", "C1", 1.0, 5, "T")
    assert a == V.seed_for("m", "C1", 1.0, 5, "T") and len({a, V.seed_for("m2", "C1", 1.0, 5, "T"), V.seed_for("m", "C2", 1.0, 5, "T"),
                                                           V.seed_for("m", "C1", 2.0, 5, "T"), V.seed_for("m", "C1", 1.0, 6, "T"), V.seed_for("m", "C1", 1.0, 5, "U")}) == 6
    draws = {}
    orig = V._gauss                                             # amendment 2: C1-K draws through V._gauss (the same float32 draw as before)

    def w(shape, gen, device):
        u = orig(shape, gen, device)
        draws.setdefault(tag[0], []).append(u.clone())
        return u
    V._gauss = w
    try:
        for name, prompt in (("A", PROMPT), ("B", torch.tensor([[11, 12, 13, 14, 15, 16, 17]], device=DEV))):
            tag = [name]
            run_arm(b, "C1-K", 1.0, preds=False, seeds=V.arm_seeds("m", "C1-K", 1.0, 5, "T"), prompt=prompt)
    finally:
        V._gauss = orig
    assert len(draws["A"]) == len(draws["B"]) and all(torch.equal(x, y) for x, y in zip(draws["A"], draws["B"])), "random draws depend on the data"
    r1 = run_arm(b, "C1", 1.0, preds=False)
    r2 = run_arm(b, "C1", 1.0, preds=False)
    r3 = run_arm(b, "C1", 1.0, preds=False, seeds=V.arm_seeds("test", "C1", 1.0, 0, "other"))
    assert r1["js"] == r2["js"] and r1["js"] != r3["js"], "determinism / seed sensitivity"


def slow_predictors(b, arm, ctx):
    """Independent float64 loop implementation: own hooks capture, at every FP16 pass of the arm run, the attention input, rotary tables and the keys
    and values as read; deltas come from the final cache minus the FP16 entries captured at write time."""
    layers = [lay.self_attn for lay in b.model.model.layers]
    cfg = b.model.config
    nq, nkv = cfg.num_attention_heads, cfg.num_key_value_heads
    hd = layers[0].head_dim
    rep = nq // nkv
    steps = []
    cur = {}

    def is_fp16_pass():
        mods = V.dynamic_modules(b.model)
        return (not b.routing_state.use_4bit) and all(m.layer_force_4bit is None for _, m in mods)

    hs = []
    for i, attn in enumerate(layers):
        def pre(mod, args, kwargs, i=i):
            if is_fp16_pass() and kwargs["hidden_states"].shape[1] == 1 and kwargs.get("past_key_values") is not None and kwargs["past_key_values"].get_seq_length() > PROMPT.shape[1] - 1:
                cur[i] = {"h": kwargs["hidden_states"].detach().clone(), "pe": tuple(x.detach().clone() for x in kwargs["position_embeddings"])}

        def post(mod, args, kwargs, out, i=i):
            if i in cur and "K" not in cur[i]:
                c = kwargs["past_key_values"]
                cur[i]["K"] = c.layers[i].keys[0].detach().double().clone()
                cur[i]["Vv"] = c.layers[i].values[0].detach().double().clone()
                if i == len(layers) - 1:
                    steps.append(dict(cur))
                    cur.clear()
        hs.append(attn.register_forward_pre_hook(pre, with_kwargs=True))
        hs.append(attn.register_forward_hook(post, with_kwargs=True))
    try:
        r, cache = run_arm(b, arm, 1.0, ctx, V.QueryCapture(b.model), True, return_cache=True)
    finally:
        for h in hs:
            h.remove()
    # the reference pass also runs FP16 single-token steps first: keep only the last S captured steps (the arm run)
    steps = steps[-S:]
    P0 = PROMPT.shape[1]

    def rot_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)
    out = {k: [] for k in ("P_norm", "P_logit", "P_val", "P_anormK", "P_out")}
    final = [(lay.keys[0].double(), lay.values[0].double()) for lay in cache.layers]
    for t_, st in enumerate(steps, start=1):
        n = t_ - 1
        if n == 0:
            for k in out:
                out[k].append(0.0 if k != "P_out" else float("nan"))
            continue
        pl = pv = pa = 0.0
        sq_slot = []
        shares = []
        for l, attn in enumerate(layers):
            q = attn.q_proj(st[l]["h"]).view(1, 1, nq, hd).transpose(1, 2)[0, :, 0, :].double()
            cos, sin = (x.double()[0, -1] for x in st[l]["pe"])
            q = q * cos + rot_half(q) * sin
            K, Vv = st[l]["K"], st[l]["Vv"]
            for g in range(nq):
                hh = g // rep
                lg = torch.tensor([float(q[g] @ K[hh, j]) for j in range(K.shape[1])], dtype=torch.float64) * float(attn.scaling)
                p = torch.softmax(lg, dim=0)
                acc_v = torch.zeros(hd, dtype=torch.float64, device=DEV)
                for j in range(P0, P0 + n):
                    wstep = j - P0 + 1                                  # the step that wrote position j
                    dk = final[l][0][hh, j] - steps[wstep - 1][l]["K"][hh, j]
                    dv = final[l][1][hh, j] - steps[wstep - 1][l]["Vv"][hh, j]
                    pl += float(p[j]) * abs(float(q[g] @ dk)) / math.sqrt(hd)
                    pa += float(p[j]) * float(dk.norm())
                    acc_v += float(p[j]) * dv
                pv += float(acc_v.norm())
        for l in range(len(layers)):
            for hh in range(nkv):
                for X in (0, 1):
                    tot = 0.0
                    for j in range(P0, P0 + n):
                        wstep = j - P0 + 1
                        dd = final[l][X][hh, j] - steps[wstep - 1][l]["K" if X == 0 else "Vv"][hh, j]
                        tot += float(dd @ dd)
                    sq_slot.append(math.sqrt(tot / n))
        for j in range(P0, P0 + n):
            wstep = j - P0 + 1
            sh = []
            for l in range(len(layers)):
                for hh in range(nkv):
                    for X in (0, 1):
                        dd = final[l][X][hh, j] - steps[wstep - 1][l]["K" if X == 0 else "Vv"][hh, j]
                        tot = float(dd @ dd)
                        if tot > 0:
                            o = ctx["outliers"][2 * l + X, hh]
                            sh.append(float((dd[o] ** 2).sum()) / tot)
            if sh:
                shares.append(sum(sh) / len(sh))
        den = len(layers) * nq
        out["P_logit"].append(pl / den)
        out["P_val"].append(pv / den)
        out["P_anormK"].append(pa / den)
        out["P_norm"].append(sum(sq_slot) / len(sq_slot))
        out["P_out"].append(sum(shares) / len(shares) if shares else float("nan"))
    return r, out


@t
def v08_predictors_match_slow_reference(fam):
    b = be(fam)
    ctx = V.synthetic_ctx(b.model.config, DEV, seed=5)
    for arm in ("NAT", "C1", "NAT-K*", "C1-V", "LOC"):
        r, slow = slow_predictors(b, arm, ctx)
        for k in slow:
            for s_, (a, c) in enumerate(zip(r[k], slow[k]), start=1):
                if c != c:
                    assert a != a, f"{arm} {k} step {s_}: expected nan"
                    continue
                assert abs(a - c) <= 1e-6 * max(abs(c), 1e-12) + 1e-14, f"{arm} {k} step {s_}: fast {a!r} vs slow {c!r}"


@t
def v09_query_capture_equals_hand_rotation(fam):
    b = be(fam)
    cap = V.QueryCapture(b.model)
    seen = {}
    layers = [lay.self_attn for lay in b.model.model.layers]

    def pre(mod, args, kwargs):
        seen["h"] = kwargs["hidden_states"].detach().clone()
        seen["pe"] = kwargs["position_embeddings"]
    h = layers[-1].register_forward_pre_hook(pre, with_kwargs=True)
    try:
        _, cache = b.run(PROMPT, None, use_4bit=False)
        L = cache.get_seq_length()
        cap.active = True
        b.run(tok(13), cache, False)
        cap.active = False
    finally:
        h.remove()
    q = cap.queries()[-1].double()
    attn = layers[-1]
    nq, hd = b.model.config.num_attention_heads, attn.head_dim
    raw = attn.q_proj(seen["h"]).view(1, 1, nq, hd).transpose(1, 2)[0, :, 0, :].double()
    cos, sin = (x.double()[0, -1] for x in seen["pe"])
    half = hd // 2
    rh = torch.cat((-raw[..., half:], raw[..., :half]), dim=-1)
    want = raw * cos + rh * sin
    assert torch.allclose(q, want, rtol=1e-9, atol=1e-12), "captured post-RoPE query differs from the hand rotation"
    nkv = b.model.config.num_key_value_heads
    kraw = attn.k_proj(seen["h"]).view(1, 1, nkv, hd).transpose(1, 2)[0, :, 0, :].double()
    krot = kraw * cos + torch.cat((-kraw[..., half:], kraw[..., :half]), dim=-1) * sin
    assert torch.allclose(krot, cache.layers[-1].keys[0, :, L, :].double(), rtol=1e-2, atol=1e-2), "hand-rotated key differs from the cached key"
    cap.remove()


@t
def v10_reference_logits_equal_e1(fam):
    b = be(fam)
    assert torch.equal(V.reference_logits_v3(b, PROMPT, REF, S), C.reference_logits(b, PROMPT, REF, S))
    cap = V.QueryCapture(b.model)
    acc = V.CalibAccumulator()
    assert torch.equal(V.reference_logits_v3(b, PROMPT, REF, S, cap, acc, b.model.config.num_key_value_heads), C.reference_logits(b, PROMPT, REF, S)), "query capture changed the computation"
    cap.remove()


@t
def v11_calibration_accumulator_and_subspace(fam):
    b = be(fam)
    cfg = b.model.config
    nq, nkv = cfg.num_attention_heads, cfg.num_key_value_heads
    g = torch.Generator().manual_seed(2)
    hd = 16 if fam == "qwen" else 64
    Ls = cfg.num_hidden_layers
    qs = [torch.randn(nq, hd, generator=g) for _ in range(Ls)]
    acc = V.CalibAccumulator()
    acc.on_queries(qs, nkv)
    rep = nq // nkv
    for l in range(Ls):
        for hh in range(nkv):
            M = sum(torch.outer(qs[l][gg].double(), qs[l][gg].double()) for gg in range(nq) if gg // rep == hh)
            assert torch.allclose(acc.M[l, hh], M), "query second moment not pooled within the KV group"
    U, E8, r80 = V.subspace_from_M(acc.M)
    assert U.shape[-1] == 8
    for l in range(Ls):
        for hh in range(nkv):
            ev = torch.linalg.eigvalsh(acc.M[l, hh]).flip(0)
            c = ev.cumsum(0) / ev.sum()
            assert abs(float(c[7]) - float(E8[l, hh])) < 1e-9, "E8 wrong"
            assert int(r80[l, hh]) == int((c < 0.8).sum()) + 1, "r80 wrong"
            Mu = acc.M[l, hh].double() @ U[l, hh].double()
            ray = (U[l, hh].double() * Mu).sum(0)
            assert torch.allclose(ray.sort(descending=True).values, ev[:8], rtol=1e-4), "U8 is not the top-8 eigenspace"
    kF = torch.randn(2 * Ls, nkv, hd, generator=g).half()
    kM = (kF.float() + 0.1 * torch.randn(2 * Ls, nkv, hd, generator=g)).half()
    acc.on_measure(kF, kM)
    D = kM.double() - kF.double()
    assert torch.allclose(acc.d2, D * D), "Delta^2 accumulation wrong"
    assert torch.allclose(acc.rhoK[0].double(), (D[0::2].norm(dim=-1) / kF[0::2].double().norm(dim=-1)).flatten()), "rho_K wrong"
    assert torch.allclose(acc.rhoV[0].double(), (D[1::2].norm(dim=-1) / kF[1::2].double().norm(dim=-1)).flatten()), "rho_V wrong"


@t
def v12_integrity_check_detects(fam):
    base = {"js": [0.0, 0.1, 0.2], "norm_dev_mean": [0.0, 0.001, 0.0], "norm_dev_max": [0.0, 0.01, 0.0], "exact_ok": True, "natural_ok": True,
            "routing_reset": True, "leak": [0.0, 0.0, 0.0], "P_logit": [0.0, 1.0, 2.0], "P_norm": [0.0, 1.0, 1.0], "P_val": [0.0, 1.0, 1.0],
            "construction_ok": True, "rounding_ok": True, "constr_dev_max": [0.0, 1e-7, 0.0]}
    assert R.integrity_errors({"a": dict(base)}) == []
    for k, v in (("js", [0.1, 0.1, 0.2]), ("js", [0.0, float("nan"), 0.1]), ("construction_ok", False), ("rounding_ok", False),
                 ("exact_ok", False), ("natural_ok", False), ("routing_reset", False), ("leak", [0, 1e-4, 0]), ("P_logit", [0, float("nan"), 1])):
        bad = dict(base)
        bad[k] = v
        assert R.integrity_errors({"a": bad}), f"integrity check misses {k}={v}"
    for k in ("construction_ok", "rounding_ok"):                 # a record without the amendment-2 flags must fail, not pass silently
        bad = dict(base)
        bad.pop(k)
        assert R.integrity_errors({"a": bad}), f"integrity check accepts a record without {k}"
    ok = dict(base, norm_dev_mean=[0, 0.02, 0], norm_dev_max=[0, 0.2, 0])   # amendment 2: post-rounding norm is reported, not gated
    assert R.integrity_errors({"a": ok}) == [], "post-rounding norm deviation must not be an integrity criterion (amendment 2)"


@t
def v13_cv_follows_channel_profile(fam):
    b = be(fam)
    ctx = V.synthetic_ctx(b.model.config, DEV)
    s2 = torch.full_like(ctx["sigma2"], 1e-6)
    s2[..., 0] = 1.0
    ctx["sigma2"] = s2
    g = torch.Generator().manual_seed(4)
    sh = ctx["sigma2"].shape
    kF = torch.randn(*sh, generator=g).to(DEV)
    kM = kF + 0.1 * torch.randn(*sh, generator=g).to(DEV)
    out, _ = V.stored_entry_v3("CV", 1.0, kF, kM, C.make_generator(DEV, 1), C.make_generator(DEV, 2), ctx)
    d = out - kF
    share = (d[..., 0] ** 2) / (d ** 2).sum(-1)
    assert float(share.mean()) > 0.9, "CV does not follow the calibration channel-variance profile"


@t
def v14_scale_factors_and_hd_rule(fam):
    import analyze_v3 as A
    rng = torch.Generator().manual_seed(0)
    L, h, hd = 2, 2, 16
    aux = {"M": (lambda X: X @ X.transpose(-1, -2))(torch.randn(L, h, hd, hd, generator=rng, dtype=torch.float64)).numpy(),
           "d2": torch.rand(2 * L, h, hd, generator=rng).numpy(), "absF": torch.rand(2 * L, h, hd, generator=rng).numpy(), "n_meas": 4, "n_q": 4,
           "rhoK": torch.full((40,), 0.02, dtype=torch.float64).numpy(), "rhoV": torch.full((40,), 0.08, dtype=torch.float64).numpy()}
    recs = {t_: {i: {"js": [0.0] + [0.001 * (a if t_ != "NAT" else 1) * (1 + i)] * 5, "target_norm_mean": [1, 1.1, 0.9, 1.0, 1.2, 1.0]}
                 for i in range(5)} for t_, a in (("NAT", 1), ("C1_a0.5", 0.5), ("C1_a1", 1), ("C1_a2", 2))}
    out = A.calibrate_from(recs, aux)
    assert abs(out["rho_star"] - 0.04) < 1e-12 and abs(out["scale_factor_K"] - 2.0) < 1e-12 and abs(out["scale_factor_V"] - 0.5) < 1e-12
    assert out["HD_descriptive"] is False, "s in [0.5, 2] (closed) must keep H-D a test"
    aux["rhoV"] = torch.full((40,), 0.09, dtype=torch.float64).numpy()
    out = A.calibrate_from(recs, aux)
    assert out["HD_descriptive"] is True, "s outside [0.5, 2] must make H-D descriptive"
    assert abs(out["dose_slope_b_hat"] - 1.0) < 1e-9, out["dose_slope_b_hat"]
    assert len(out["U8"][0][0][0]) == 8


@t
def v15_fp16_rounding_is_correct(fam):
    """Amendment 2: the production rounding (V.to_fp16) and the runtime-check rounding (V.fp16_rne_reference) both equal an independent oracle,
    bitwise, on random values, signed zeros, subnormals, exact ties and double-rounding traps; the output is FP16."""
    x = rounding_cases().to(DEV)
    ref = struct_fp16_bits(x)
    for name, f in (("to_fp16", V.to_fp16), ("fp16_rne_reference", V.fp16_rne_reference)):
        y = f(x)
        assert y.dtype == torch.float16, f"{name}: output is {y.dtype}, not float16"
        bad = int((y.view(torch.int16).cpu() != ref).sum())
        assert bad == 0, f"{name}: {bad} of {x.numel()} values not correctly rounded"


@t
def v16_runtime_split_check(fam):
    """Amendment 2: on real steps every arm passes the construction and rounding checks; the checks fail when the construction is off by 1.001 in
    norm, when the stored entry is one FP16 step away from the correct rounding, and when the stored entry is not FP16."""
    b = be(fam)
    for arm in V.TEST_ARMS:
        r = run_arm(b, arm, 1.0, preds=False)
        assert r["construction_ok"] and r["rounding_ok"], f"{arm}: split check fails on a correct run (constr max {max(r['constr_dev_max']):.2e})"
        assert R.integrity_errors({arm: r}, predictors=False) == [], f"{arm}: {R.integrity_errors({arm: r}, predictors=False)}"
    orig = V.stored_entry_v3

    def scaled(arm, alpha, kF, kM, gen, vgen, ctx):
        out, info = orig(arm, alpha, kF, kM, gen, vgen, ctx)
        if info["target"] is not None and alpha != 0:
            F = kF.double()
            info["pre"] = F + 1.001 * (info["pre"] - F)
            out = V.to_fp16(info["pre"])
        return out, info

    def off_by_one(arm, alpha, kF, kM, gen, vgen, ctx):
        out, info = orig(arm, alpha, kF, kM, gen, vgen, ctx)
        bits = out.view(torch.int16).clone()
        bits.view(-1)[5] += 1
        return bits.view(torch.float16), info

    def not_fp16(arm, alpha, kF, kM, gen, vgen, ctx):
        out, info = orig(arm, alpha, kF, kM, gen, vgen, ctx)
        return info["pre"].clone(), info
    for name, fn, flag in (("construction x1.001", scaled, "construction_ok"), ("one FP16 step off", off_by_one, "rounding_ok"),
                           ("not FP16", not_fp16, "rounding_ok")):
        V.stored_entry_v3 = fn
        try:
            r = run_arm(b, "C1", 1.0, preds=False)
        finally:
            V.stored_entry_v3 = orig
        assert not r[flag] and R.integrity_errors({"C1": r}, predictors=False), f"runtime check misses: {name}"


# ----------------------------------------------------------------------------- real models (optional)
def real_checks():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rep = {}
    for label, mid in R.MODELS.items():                     # amendment 1: all four models
        fam = R.family(label)
        b = LS.build_backend(AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16), "cpu")
        tok = AutoTokenizer.from_pretrained(mid)
        p = tok("Question: Tom has 3 apples and buys 4 more. How many apples does he have?\nAnswer:", return_tensors="pt").input_ids
        ref = R.fp16_history(b, tok, p)[:5]
        geo = LS.cache_geometry(b, p)
        ctx = V.synthetic_ctx(b.model.config, "cpu")
        cap = V.QueryCapture(b.model)
        Sr = len(ref) - 1
        rl = V.reference_logits_v3(b, p, ref, Sr)
        cF = V.fp16_teacher_forced_cache(b, p, ref, Sr)
        a0 = []
        errs = []
        for arm in V.TEST_ARMS:
            r0, c0 = V.teacher_forced_arm_v3(b, p, ref, Sr, arm, 0.0, V.arm_seeds("real", arm, 0.0, 0, "T"), rl, ctx, cap, False, return_cache=True)
            a0.append(all(x == 0 for x in r0["js"]) and all(torch.equal(x.keys, y.keys) and torch.equal(x.values, y.values) for x, y in zip(c0.layers, cF.layers)))
            r1 = V.teacher_forced_arm_v3(b, p, ref, Sr, arm, 1.0, V.arm_seeds("real", arm, 1.0, 0, "T"), rl, ctx, cap, True)
            errs += R.integrity_errors({arm: r1})
        n_dyn = len(V.dynamic_modules(b.model))
        rep[label] = {"hf_id": mid, "geometry": geo, "n_dynamic_modules": n_dyn, "alpha0_bitwise_all_arms": all(a0), "integrity_errors": errs,
                    "pass": bool(all(a0) and not errs and n_dyn == R.N_DYNAMIC[fam] and (geo["n_layers"], geo["n_kv_heads"], geo["head_dim"]) == R.GEOMETRY[fam])}
        cap.remove()
        del b
    return rep


def static_checks():
    out = {}
    for tname in ("lock_e1.py", "lock_e1l.py"):
        out[tname] = subprocess.run([sys.executable, str(RV / "tools" / tname), "--verify"], capture_output=True).returncode == 0
    return out


def run_suite(families=("qwen", "llama"), verbose=False):
    fails = []
    for fn in TESTS:
        for fam in families:
            try:
                fn(fam)
            except Exception as e:  # noqa: BLE001
                fails.append((f"{fn.__name__}[{fam}]", f"{type(e).__name__}: {e}"))
                if verbose:
                    traceback.print_exc()
    return fails


def main():
    fails = run_suite(verbose=True)
    n = len(TESTS) * len(FAMILIES)
    print(f"{n - len(fails)}/{n} passed")
    for f in fails:
        print("FAIL", *f)
    rep = {"tiny_tests_total": n, "tiny_tests_failed": [list(f) for f in fails]}
    if "--static" in sys.argv:
        rep["static"] = static_checks()
        print("static:", rep["static"])
    if "--real" in sys.argv:
        rep["real"] = real_checks()
        print(json.dumps(rep["real"], indent=1, default=str))
    if "--write" in sys.argv:
        (RV / "validation_logs").mkdir(exist_ok=True)
        (RV / "validation_logs" / "v3_cpu_validation.json").write_text(json.dumps(rep, indent=1, default=str))
    ok = not fails and all(rep.get("static", {}).values()) and all(v["pass"] for v in rep.get("real", {}).values())
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
