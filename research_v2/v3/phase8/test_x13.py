"""CPU tests for X13 (v3/phase8/FOLLOWUP_X13.md section 5). Tiny Qwen2 and tiny GQA Llama of tests/run_v3_tests.py (real DynamicLinear / NF4
path); --real adds a few teacher-forced steps on the four real models on CPU with their frozen calibration contexts. LOGIC only.

  venv/Scripts/python.exe research_v2/v3/phase8/test_x13.py [--real] [--write]
"""

from __future__ import annotations

import json
import math
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

P8 = Path(__file__).resolve().parent
RV = P8.parents[1]
sys.path.insert(0, str(RV / "tests"))
sys.path.insert(0, str(RV))
sys.path.insert(0, str(P8))

import run_v3_tests as T                         # noqa: E402  (tiny backends and run_arm; HF before torch)

import numpy as np                               # noqa: E402
import torch                                     # noqa: E402

import rv3.core as V                             # noqa: E402
import rv3x13 as X                               # noqa: E402
import run_x13 as RX                             # noqa: E402
import analyze_x13 as AX                         # noqa: E402

R = T.R
FAMILIES = ("qwen", "llama")
TESTS = []
LOCKED_ARMS = ["NAT", "NAT-K", "C1", "QA", "LOC"]


def t(fn):
    TESTS.append(fn)
    return fn


def ctx_for(b, seed=5, rho=0.07):
    return V.synthetic_ctx(b.model.config, T.DEV, seed=seed, rho_star=rho)


def synth_M(b, seed=3):
    cfg = b.model.config
    L_, h = cfg.num_hidden_layers, cfg.num_key_value_heads
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(L_, h, hd, hd, generator=g, dtype=torch.float64)
    return A @ A.transpose(-1, -2) + 0.1 * torch.eye(hd, dtype=torch.float64)


def seeds(arm, label="x13t"):
    return V.arm_seeds(label, X.SEED_LABEL.get(arm, arm), 1.0, 0, "T")


def run_arm(b, arm, ctx, x13=False, M=None, alpha=1.0, capture=False, arm_seeds=None):
    """One teacher-forced arm with predictors. x13: X13 overrides installed. capture: also return the stored entries per step
    (alpha, kF, kM, out, info). Returns (record, predictor state or None, step log, captured)."""
    cap = V.QueryCapture(b.model)
    seen = []
    try:
        if x13:
            X.install(M)
            X.clear()
        inner = V.stored_entry_v3
        if capture:
            def w(a_, al, kF, kM, gen, vgen, c):
                out, info = inner(a_, al, kF, kM, gen, vgen, c)
                if a_ == arm:
                    seen.append((al, kF.clone(), kM.clone(), out.clone(), {k: (v.clone() if torch.is_tensor(v) else v) for k, v in info.items()}))
                return out, info
            V.stored_entry_v3 = w
        r = T.run_arm(b, arm, alpha, ctx, cap, True, seeds=arm_seeds or seeds(arm))
        inst = X.INSTANCES[-1] if x13 else None
        log = list(X.STEP_LOG)
    finally:
        cap.remove()
        X.uninstall()
    return r, inst, log, seen


def roundtrip(r):
    return json.loads(json.dumps(r))


def nat_mu(b, ctx, M=None):
    _, inst, _, _ = run_arm(b, "NAT", ctx, x13=True, M=M)
    return X.mu_from(inst), inst


# ----------------------------------------------------------------------------- tests
@t
def x01_locked_records_bitwise_with_x13_state(fam):
    b = T.be(fam)
    ctx = ctx_for(b)
    M = synth_M(b)
    for arm in LOCKED_ARMS:
        r0, _, _, _ = run_arm(b, arm, ctx)
        r1, inst, _, _ = run_arm(b, arm, ctx, x13=True, M=M)
        d = RX.compare_records(r1, roundtrip(r0))
        assert not d, f"{arm}: X13 record differs from the locked one: {list(d)[:4]}"
        assert len(inst.plc) == len(r1["js"]) and all(math.isfinite(x) for x in inst.plc), f"{arm}: P_logit^c missing or non-finite"
        assert inst.plc[0] == 0.0 and any(x > 0 for x in inst.plc[1:]), f"{arm}: P_logit^c is not 0 at step 1 or never positive"


@t
def x02_centred_arm_construction(fam):
    b = T.be(fam)
    ctx = ctx_for(b)
    mu, _ = nat_mu(b, ctx)
    assert float(mu.abs().max()) > 0, "vacuous: mu is zero"
    for arm, path in X.CEN_ARMS.items():
        r, _, log, seen = run_arm(b, arm, dict(ctx, mu=mu), x13=True, capture=True)
        assert not R.integrity_errors({arm: r}), R.integrity_errors({arm: r})
        assert len(seen) == T.S and len(log) == T.S and all("degenerate" in s for s in log)
        rows, other = (slice(0, None, 2), slice(1, None, 2)) if path == "K" else (slice(1, None, 2), slice(0, None, 2))
        for alpha, kF, kM, out, info in seen:
            F, D = kF.double(), kM.double() - kF.double()
            d = info["pre"][rows] - F[rows]
            nD = D[rows].norm(dim=-1)
            ok = nD > 0
            assert torch.allclose(d.norm(dim=-1)[ok], nD[ok], rtol=1e-10, atol=0), "pre-rounding norm is not ||Delta||"
            c = D[rows] - mu.double()[rows]
            cos = (d * c).sum(-1)[ok] / (d.norm(dim=-1)[ok] * c.norm(dim=-1)[ok])
            assert bool((cos > 1 - 1e-10).all()), f"direction is not that of Delta - mu (min cos {float(cos.min())})"
            assert torch.equal(out[other], kF[other]), "the other path is not kappa^F bitwise"
    for arm, path in X.CEN_ARMS.items():                                           # mu = 0: the natural path entry, bitwise
        rows = slice(0, None, 2) if path == "K" else slice(1, None, 2)
        _, _, _, seen = run_arm(b, arm, dict(ctx, mu=torch.zeros_like(mu)), x13=True, capture=True)
        for _, kF, kM, out, _ in seen:
            assert torch.equal(out[rows], kM[rows]), f"{arm} with mu = 0 differs from the natural entry"
    cF = V.fp16_teacher_forced_cache(b, T.PROMPT, T.REF, T.S)                    # alpha = 0: the FP16 cache
    X.install()
    try:
        r0, c0 = T.run_arm(b, "NAT-K-cen", 0.0, dict(ctx, mu=mu), None, False, return_cache=True)
    finally:
        X.uninstall()
    assert all(x == 0.0 for x in r0["js"]), "alpha 0: JS not exactly 0"
    assert all(torch.equal(x.keys, y.keys) and torch.equal(x.values, y.values) for x, y in zip(c0.layers, cF.layers)), "alpha 0: cache differs"
    g = torch.Generator().manual_seed(4)                                           # degenerate fallback: mu = Delta stores the natural entry
    kF = (torch.randn(4, 2, 16, generator=g) * 3).half()
    kM = (kF.double() + torch.randn(4, 2, 16, generator=g, dtype=torch.float64) * 0.05).half()
    X.STEP_LOG.clear()
    out, _ = X._centred("NAT-K-cen", 1.0, kF, kM, {"mu": kM.double() - kF.double()})
    nd = int(((kM.double() - kF.double())[0::2].norm(dim=-1) > 0).sum())
    assert X.STEP_LOG[-1]["degenerate"] == nd and torch.equal(out[0::2], kM[0::2]), "degenerate slots do not keep the natural entry"
    X.STEP_LOG.clear()


@t
def x03_mu_is_mean_of_nat_delta(fam):
    b = T.be(fam)
    ctx = ctx_for(b)
    r, inst, _, seen = run_arm(b, "NAT", ctx, x13=True, capture=True)
    D = torch.stack([kM.double() - kF.double() for _, kF, kM, _, _ in seen])       # [S, 2L, n_kv, hd]
    ref = D.mean(0).numpy()
    mu = X.mu_from(inst).numpy()
    assert mu.shape == ref.shape and np.allclose(mu, ref, rtol=1e-12, atol=1e-15), "mu is not the per-slot mean of NAT's Delta"
    sl = X.slot_arrays(inst, nat=True)
    assert np.allclose(sl["S1K"], D[:, 0::2].sum(0).numpy(), rtol=1e-12, atol=1e-15)
    assert np.allclose(sl["normsV"], D[:, 1::2].norm(dim=-1).numpy(), rtol=1e-12, atol=0)


def _nc_case(seed, scale_F, scale_d, L2=4, h=2, hd=16):
    g = torch.Generator().manual_seed(seed)
    F = (torch.randn(L2, h, hd, generator=g, dtype=torch.float64) * scale_F).half().double()
    d = torch.randn(L2, h, hd, generator=g, dtype=torch.float64) * scale_d
    return F, d, torch.zeros(L2, dtype=torch.bool), d.norm(dim=-1, keepdim=True)


def _resid(r16, F, t):
    return ((r16.double() - F).norm(dim=-1) - t).abs() / t


@t
def x04_norm_corrected_rounding(fam):
    for seed, sF, sd in ((1, 60.0, 0.02), (2, 300.0, 0.05), (3, 8.0, 0.002), (4, 1.0, 0.3)):
        F, d, ex, tg = _nc_case(seed, sF, sd)
        t_ = tg.squeeze(-1)
        r, st = X.nc_round(F, d.clone(), ex, tg)
        r1 = V.fp16_rne_reference(F + d)
        e, e1 = _resid(r, F, t_), _resid(r1, F, t_)
        assert r.dtype == torch.float16 and bool((e <= e1).all()) and st["nc_ok"], f"case {seed}: corrected residual above nearest"
        small = e1 <= X.NC_SEARCH_MIN
        assert torch.equal(r[small], r1[small]), f"case {seed}: a slot within tolerance was changed"
        ref = AX_signed(r, F, t_)
        assert all(math.isclose(st[k], v, rel_tol=1e-12, abs_tol=1e-15) for k, v in ref.items()), f"case {seed}: signed stats {st} vs {ref}"
        assert all(math.isclose(st["nn_" + k], v, rel_tol=1e-12, abs_tol=1e-15) for k, v in AX_signed(r1, F, t_).items())
        sel = ~small
        if bool(sel.any()) and seed in (1, 2):
            assert float(e[sel].mean()) < 0.5 * float(e1[sel].mean()), f"case {seed}: the search did not reduce the residual"
    F, d, ex, tg = _nc_case(1, 60.0, 0.02)                                        # lambda* equals an independent loop over the grid
    t_ = tg.squeeze(-1)
    grid = X.lambda_grid("cpu")
    r1 = V.fp16_rne_reference(F + d)
    for (i, j) in [(i, j) for i in range(F.shape[0]) for j in range(F.shape[1])][:6]:
        if float(_resid(r1, F, t_)[i, j]) <= X.NC_SEARCH_MIN:
            continue
        best, best_l = float(_resid(r1, F, t_)[i, j]), 1.0
        for lam in grid.tolist():
            rr = V.fp16_rne_reference(F[i, j] + lam * d[i, j])
            ee = abs(float((rr.double() - F[i, j]).norm()) - float(t_[i, j])) / float(t_[i, j])
            if ee < best:
                best, best_l = ee, lam
        sub_r, _ = X.nc_round(F[i:i + 1, j:j + 1].clone(), d[i:i + 1, j:j + 1].clone(), torch.zeros(1, dtype=torch.bool), tg[i:i + 1, j:j + 1])
        got = float(_resid(sub_r, F[i:i + 1, j:j + 1], t_[i:i + 1, j:j + 1]))
        if best > X.NC_REFINE_MIN:
            assert got <= best + 1e-15
        else:
            assert math.isclose(got, best, rel_tol=1e-12, abs_tol=1e-15), f"slot {(i, j)}: {got} vs independent {best} (lambda {best_l})"
    F, d, ex, tg = _nc_case(7, 8.0, 0.0015)                                        # refinement: local optimum, terminated before 64 moves
    t_ = tg.squeeze(-1)
    r, st = X.nc_round(F, d.clone(), ex, tg)
    e = _resid(r, F, t_)
    assert st["moves"] > 0, "vacuous: refinement never moved"
    up, dn = X.fp16_neighbors(r)
    for c in range(r.shape[-1]):
        for cand in (up, dn):
            trial = r.clone()
            trial[..., c] = cand[..., c]
            refined = e > X.NC_REFINE_MIN
            assert not bool((_resid(trial, F, t_)[refined] < e[refined]).any()), "refined slot is not a local optimum"
    assert st["moves"] < 64 * int((_resid(V.fp16_rne_reference(F + d), F, t_) > X.NC_SEARCH_MIN).sum()), "refinement hit the move cap"
    allh = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).view(torch.float16)   # fp16_neighbors is exact
    fin = allh[torch.isfinite(allh)].double().unique()
    g = torch.Generator().manual_seed(9)
    v = torch.cat([torch.tensor([0.0, -0.0, 6e-8, -6e-8, 1.0, -1.0, 65504.0 / 2]).half(),
                   (torch.randn(300, generator=g) * torch.tensor(10.0) ** torch.randint(-6, 4, (300,), generator=g)).half()])
    up, dn = X.fp16_neighbors(v)
    k = torch.searchsorted(fin, v.double(), right=True)
    k0 = torch.searchsorted(fin, v.double(), right=False) - 1
    assert torch.equal(up.double(), fin[k]) and torch.equal(dn.double(), fin[k0]), "fp16_neighbors is not the adjacent FP16 value"
    b = T.be(fam)                                                                   # C1-nc: the locked C1 stream, step-1 pre-rounding entry
    ctx = ctx_for(b)
    _, _, _, lock = run_arm(b, "C1", ctx, capture=True, arm_seeds=V.arm_seeds("x13t", "C1", 1.0, 0, "T"))
    pre_nc = []
    orig = X._LOCKED["stored_entry_v3"]

    def w(a_, al, kF, kM, gen, vgen, c):
        out, info = orig(a_, al, kF, kM, gen, vgen, c)
        pre_nc.append(info["pre"].clone())
        return out, info
    X._LOCKED["stored_entry_v3"] = w
    try:
        r, _, log, _ = run_arm(b, "C1-nc", ctx, x13=True, arm_seeds=seeds("C1-nc"))
    finally:
        X._LOCKED["stored_entry_v3"] = orig
    assert torch.equal(pre_nc[0], lock[0][4]["pre"]), "C1-nc does not reproduce the locked C1 pre-rounding entry at step 1"
    errs = RX.integrity_x13({"C1-nc": {**r, **X.step_fields(log)}})               # the driver merges the step log into the record
    assert not errs and len(log) == T.S and all(s["nc_ok"] for s in log), errs


def AX_signed(r, F, t_):
    s = ((r.double() - F).norm(dim=-1) - t_) / t_
    return {"sdev_sum": float(s.sum()), "sdev_n": int(s.numel()), "sdev_min": float(s.min()), "sdev_max": float(s.max()),
            "sdev_pos5": int((s > 0.05).sum()), "sdev_neg5": int((s < -0.05).sum())}


@t
def x05_centred_logit_term_finite_difference(fam):
    g = torch.Generator().manual_seed(21)
    n_q, n_kv, hd, T_, lo, n = 4, 2, 16, 12, 7, 4
    q = torch.randn(n_q, hd, generator=g, dtype=torch.float64)
    K = torch.randn(n_kv, T_, hd, generator=g, dtype=torch.float64)
    hmap = V.group_map(n_q, n_kv, "cpu")
    dk = torch.randn(n_q, n, hd, generator=g, dtype=torch.float64)
    sc = hd ** -0.5
    term = X.centred_logit_term(q, K, hmap, sc, lo, n, dk, hd)
    lg = torch.einsum("gd,gjd->gj", q, K[hmap]) * sc
    dl = torch.zeros_like(lg)
    dl[:, lo:lo + n] = torch.einsum("gd,gjd->gj", q, dk)
    eps = 1e-7
    fd = float((torch.softmax(lg + eps * sc * dl, -1) - torch.softmax(lg, -1)).abs().sum()) / eps
    assert math.isclose(term, fd, rel_tol=1e-5), f"P_logit^c term {term} vs finite difference {fd}"


@t
def x06_rayleigh(fam):
    g = torch.Generator().manual_seed(31)
    L_, h, hd = 3, 2, 8
    A = torch.randn(L_, h, hd, hd, generator=g, dtype=torch.float64)
    M = A @ A.transpose(-1, -2)
    dk = torch.randn(L_, h, hd, generator=g, dtype=torch.float64)
    num = X.rayleigh_num(dk, M).numpy()
    for l in range(L_):
        for k in range(h):
            x = dk[l, k].numpy()
            assert math.isclose(num[l, k], float(x @ M[l, k].numpy() @ x), rel_tol=1e-12)
    u = torch.randn(20000, L_, h, hd, generator=g, dtype=torch.float64)
    u = u / u.norm(dim=-1, keepdim=True)
    tr = torch.diagonal(M, dim1=-2, dim2=-1).sum(-1)
    ratio = float(torch.einsum("nlhd,lhde,nlhe->", u, M, u) / (tr / hd).sum() / u.shape[0])
    assert abs(ratio - 1) < 0.02, f"isotropic Rayleigh ratio {ratio}"
    b = T.be(fam)
    Mb = synth_M(b)
    _, inst, _, _ = run_arm(b, "NAT", ctx_for(b), x13=True, M=Mb)
    dK = inst.dK[:, :, :inst.n]                                                    # [L, n_kv, n, hd]
    ref = torch.einsum("lhnd,lhde,lhne->lh", dK, Mb, dK)
    assert torch.allclose(inst.RN, ref, rtol=1e-10, atol=0), "state Rayleigh numerator differs from the explicit sum"
    trb = torch.diagonal(Mb, dim1=-2, dim2=-1).sum(-1)
    assert torch.allclose(inst.RD, trb / dK.shape[-1] * (dK * dK).sum((-1, -2)), rtol=1e-10, atol=0)


@t
def x07_common_mode_share(fam):
    g = np.random.default_rng(41)
    n, L_, h, hd = 30, 3, 2, 16
    D = g.standard_normal((n, L_, h, hd)) + 0.5
    norms, S1 = np.linalg.norm(D, axis=-1), D.sum(0)
    direct = sum(np.sum((S1[l, k] / n) ** 2) for l in range(L_) for k in range(h)) / sum(np.sum(D[:, l, k] ** 2) / n for l in range(L_) for k in range(h))
    assert math.isclose(AX.cm_share(norms, S1), direct, rel_tol=1e-12)
    fixed = np.abs(g.standard_normal((n, L_, h))) + 0.1
    vals = []
    for _ in range(3000):
        u = g.standard_normal((n, L_, h, hd))
        u /= np.linalg.norm(u, axis=-1, keepdims=True)
        vals.append(n * AX.cm_share(fixed, (u * fixed[..., None]).sum(0)))
    assert abs(np.mean(vals) - 1) < 0.02, f"isotropic mean n*C {np.mean(vals)}"
    nu = AX.null_lognc(fixed, np.random.default_rng(5), draws=400, hd=hd)
    ref = float(np.mean(np.log(vals)))
    assert abs(nu - ref) < 0.05, f"null mean ln(nC) {nu} vs independent {ref}"


@t
def x08_comparator_auth_manifest_subset_design(fam):
    base = {"index": 1, "js": [0.0, 0.125, 0.25], "P_out": [float("nan"), 0.5, 0.5], "top": [3, 4, 5], "exact_ok": True, "wall_seconds": 2.0}
    lk = roundtrip(base)
    assert RX.compare_records(dict(base), lk) == {}
    assert RX.compare_records({**base, "wall_seconds": 9.0, "P_logit_c": [0.1]}, lk) == {}, "wall_seconds / X13 fields must be ignored"
    assert "js" in RX.compare_records({**base, "js": [0.0, math.nextafter(0.125, 1.0), 0.25]}, lk), "1-ulp change not detected"
    assert "P_out" in RX.compare_records({**base, "P_out": [0.5, float("nan"), 0.5]}, lk), "NaN position change not detected"
    assert "js" in RX.compare_records({**base, "js": [-0.0, 0.125, 0.25]}, lk), "sign of zero not detected"
    assert "top" in RX.compare_records({k: v for k, v in base.items() if k != "top"}, lk), "missing key not detected"
    assert "sdev_n" in RX.compare_records({**base, "sdev_n": [1]}, lk), "an unexpected key must be reported"
    assert RX.history_matches([1, 2, 3], [1, 2, 3]) and not RX.history_matches([1, 2, 3], [1, 2, 4]) and not RX.history_matches([1, 2], [1, 2, 3])
    tmp = Path(tempfile.mkdtemp())
    try:
        for n in RX.MANIFEST_FILES:
            if (P8 / n).exists():
                shutil.copy(P8 / n, tmp / n)
            else:
                (tmp / n).write_text("placeholder\n")
        runs = tmp / "runs"
        for m in RX.ORDER:
            (runs / m / "calib").mkdir(parents=True)
            shutil.copy(RV / "v3_runs" / m / "calib" / "aux_state.pt", runs / m / "calib" / "aux_state.pt")
        mid = RX.manifest_id(tmp, runs=runs)
        p = tmp / "rv3x13.py"
        p.write_bytes(p.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
        assert RX.manifest_id(tmp, runs=runs) == mid, "CRLF vs LF changes the manifest id"
        p.write_bytes(p.read_bytes() + b"#")
        assert RX.manifest_id(tmp, runs=runs) != mid, "a changed byte does not change the manifest id"
        p.write_bytes(p.read_bytes()[:-1])
        a = runs / "llama-base" / "calib" / "aux_state.pt"
        a.write_bytes(a.read_bytes() + b"\0")
        assert RX.manifest_id(tmp, runs=runs) != mid, "a changed aux_state.pt does not change the manifest id"
    finally:
        shutil.rmtree(tmp)
    assert not RX.auth_ok("", mid) and not RX.auth_ok(mid, mid) and not RX.auth_ok("FOLLOWUP_X13", mid)
    assert not RX.auth_ok("FOLLOWUP_X13 0123456789abcdef", mid) and RX.auth_ok(f"FOLLOWUP_X13\n{mid}\n", mid)
    s = json.loads((RV / "v3_sample.json").read_text())
    ids = RX.subset_ids(s["tierA_ids"])
    assert ids == json.loads(RX.X9_SUBSET.read_text())["ids"], "the X13 draw does not reproduce the X9 subset"
    if RX.SUBSET.exists():
        assert json.loads(RX.SUBSET.read_text())["ids"] == ids and json.loads(RX.SUBSET.read_text())["seed"] == 20261209
    d = RX.full_design()
    exp = 100.0
    for k, m in enumerate(RX.ORDER):
        ratio = 1.0 if m.startswith("qwen") else R.LLAMA_TIME_RATIO
        exp += (50 - (3 if k == 0 else 0)) * (len(RX.arms_for(m)) * 7.0 + 2.7) * ratio + (60.0 if k else 0.0)
    assert math.isclose(RX.projection_h(d, 100.0, 7.0, 2.7, 60.0, 3), exp * 1.05 / 3600, rel_tol=1e-12)
    assert [len(RX.arms_for(m)) for m in RX.ORDER] == [5, 5, 3, 3]
    fit = RX.choose_design(100.0, 7.0, 2.7, 60.0, 3)
    assert fit["pass_"] and fit["applied"] == [], fit["projection_steps"]
    slow = RX.choose_design(100.0, 9.5, 3.0, 60.0, 3)
    assert slow["applied"][:1] == ["n = 40 in Llama"] and slow["n"]["llama-base"] == 40 and slow["n"]["qwen-base"] in (40, 50)
    worst = RX.choose_design(100.0, 30.0, 3.0, 60.0, 3)
    assert worst["applied"] == RX.REDUCTIONS and not worst["pass_"] and worst["n"] == {m: 40 for m in RX.ORDER}


def _rec(**kw):
    r = {"js": [0.0, 0.1], "construction_ok": True, "constr_dev_max": [0.0, 0.0], "rounding_ok": True, "exact_ok": True, "natural_ok": True,
         "routing_reset": True, "leak": [0.0, 0.0], "P_logit": [0.0, 0.1], "P_norm": [0.0, 0.1], "P_val": [0.0, 0.1]}
    r.update(kw)
    return r


@t
def x09_integrity_rule(fam):
    assert RX.integrity_x13({"C1-nc": _rec(construction_ok=False, constr_dev_max=[0.0, 0.3], nc_ok=[True, True])}) == []
    assert RX.integrity_x13({"C1-nc": _rec(construction_ok=False, nc_ok=[True, False])}), "nc_ok false must be an error"
    assert RX.integrity_x13({"C1-K-nc": _rec(nc_ok=[])}), "missing nc_ok must be an error"
    assert RX.integrity_x13({"NAT-K-cen": _rec(construction_ok=False, constr_dev_max=[0.0, 0.3])}), "centred construction error must be kept"
    assert RX.integrity_x13({"NAT": _rec(rounding_ok=False)}), "locked errors must be kept"


@t
def x10_analysis_rules(fam):
    rng = np.random.default_rng(0)
    G, ci = AX.gap_closed([2.0, 2.0, 2.0], [1.0, 1.0, 1.0], [3.0, 3.0, 3.0], rng)
    assert math.isclose(G, 0.5) and np.allclose(ci, [0.5, 0.5])
    Gv, _ = AX.gap_closed([0.5, 0.5], [1.0, 1.0], [0.0, 0.0], rng)                  # value path: denominator negative, movement toward C1 -> G > 0
    assert math.isclose(Gv, 0.5)
    assert AX.verdict_a({"ci95": [1.5, 2.0]}, 0.01) == "supported" and AX.verdict_a({"ci95": [1.0, 2.0]}, 0.01) == "inconclusive"
    assert AX.verdict_a({"ci95": [1.5, 2.0]}, 0.2) == "inconclusive" and AX.verdict_a({"ci95": [-1.0, 0.6]}, 0.01) == "contradicted"
    s = {"estimate": 0.3, "ci95": [0.1, 0.5]}
    assert AX.verdict_direction(s, 0.01, 1.0) == "supported" and AX.verdict_direction(s, 0.2, 1.0) == "inconclusive"
    assert AX.verdict_direction(s, 0.01, -1.0) == "contradicted"
    assert AX.verdict_direction({"estimate": -0.3, "ci95": [-0.5, -0.1]}, 0.01, -1.0) == "supported"
    assert AX.size_band(0.7, [0.3, 0.9]) == "most of the gap" and AX.size_band(0.1, [0.0, 0.2]) == "small part"
    assert AX.size_band(1.5, [1.2, 1.8]) == "most of the gap; overshoot" and AX.size_band(0.4, [0.1, 0.7]) == "partial"
    row = lambda d, ok=True, band="most of the gap": {"direction": d, "manipulation_ok": ok, "band": band}
    pm = lambda rows: {f"m{k}": {"b": {"K": r}} for k, r in enumerate(rows)}
    assert AX.reading(pm([row("supported")] * 3 + [row("inconclusive")]), "K") == "supported; accounts for most of the provenance gap"
    assert AX.reading(pm([row("supported", band="partial")] * 3 + [row("inconclusive")]), "K") == "supported"
    assert AX.reading(pm([row("supported", ok=False)] * 3 + [row("inconclusive")]), "K") == "mixed"
    assert AX.reading(pm([row("contradicted")] * 2 + [row("supported")] * 2), "K") == "contradicted"
    assert math.isclose(AX.half_width([-0.307, -0.176]), 0.0655)
    assert AX.verdict_d([-0.03, 0.04], 0.0655) == "supported" and AX.verdict_d([0.07, 0.2], 0.0655) == "contradicted"
    assert AX.verdict_d([-0.02, 0.1], 0.0655).startswith("inconclusive") and "rounding confound" in AX.verdict_d([-0.02, 0.1], 0.0655)


@t
def x11_run_arms_end_to_end(fam):
    b = T.be(fam)
    ctx = ctx_for(b)
    M = synth_M(b)
    cap = V.QueryCapture(b.model)
    X.install(M)
    try:
        recs, slots = RX.run_arms(b, T.PROMPT, T.REF, T.S, RX.ARMS_QWEN, ctx, cap, "x13t", 0, raw=True)
        insts = list(X.INSTANCES)
    finally:
        cap.remove()
        X.uninstall()
    assert "mu" not in ctx, "mu left in the context"
    assert list(recs) == [t_ for _, _, t_ in RX.ARMS_QWEN] and not RX.integrity_x13(recs), RX.integrity_x13(recs)
    r0, _, _, _ = run_arm(b, "NAT", ctx)
    meta = ("index", "split", "model", "arm", "alpha", "S", "ref_len")
    assert not RX.compare_records({k: v for k, v in recs["NAT"].items() if k not in meta}, roundtrip(r0)), "run_arms NAT differs from the locked NAT"
    assert [recs["NAT"][k] for k in meta] == [0, "test", "x13t", "NAT", 1.0, T.S, len(T.REF)], "record metadata differs from run_problem's"
    assert all(k in recs[t_] for t_ in ("NAT-K-cen", "NAT-V-cen") for k in ("sdev_n", "degenerate", "P_logit_c"))
    assert all(k in recs[t_] for t_ in ("C1-nc", "C1-K-nc") for k in ("nn_sdev_n", "lam_min", "nc_ok", "cos_min"))
    assert "sdev_n" not in recs["NAT"] and {"RN", "RD", "rawK"} <= set(slots["NAT"]) and "RN" not in slots["NAT-K-cen"]
    mu = X.mu_from(insts[0])
    d1, D1 = insts[1].dK[:, :, 0], insts[0].dK[:, :, 0]                           # step 1: K-cen's Delta equals NAT's (same cache before step 1)
    c = D1 - mu[0::2]
    ok = (D1.norm(dim=-1) > 0) & (d1.norm(dim=-1) > 0)
    cos = (d1 * c).sum(-1)[ok] / (d1.norm(dim=-1)[ok] * c.norm(dim=-1)[ok])
    assert bool((cos > 0.99).all()), f"centred arm at step 1 does not follow Delta - mu of NAT (min cos {float(cos.min())})"


# ----------------------------------------------------------------------------- real models (optional)
def real_checks(steps=4):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rep = {}
    for label, mid in R.MODELS.items():
        b = T.LS.build_backend(AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16), "cpu")
        tok = AutoTokenizer.from_pretrained(mid)
        p = tok("Question: Tom has 3 apples and buys 4 more. How many apples does he have?\nAnswer:", return_tensors="pt").input_ids
        ref = R.fp16_history(b, tok, p)[:steps + 1]
        Sr = len(ref) - 1
        cal = json.loads((RV / "v3_runs" / label / "calibration.json").read_text())
        ctx = V.ctx_from_calibration(cal, "cpu")
        M = torch.load(RV / "v3_runs" / label / "calib" / "aux_state.pt", map_location="cpu", weights_only=False)["M"]
        rl = V.reference_logits_v3(b, p, ref, Sr)
        cap = V.QueryCapture(b.model)
        try:
            r0 = V.teacher_forced_arm_v3(b, p, ref, Sr, "NAT", 1.0, V.arm_seeds(label, "NAT", 1.0, 0, "T"), rl, ctx, cap, True)
            X.install(M)
            recs, _ = RX.run_arms(b, p, ref, Sr, RX.arms_for(label), ctx, cap, label, 0)
        finally:
            cap.remove()
            X.uninstall()
        d = RX.compare_records({k: v for k, v in recs["NAT"].items() if k not in ("index", "split", "model", "arm", "alpha", "S", "ref_len")},
                               roundtrip(r0))
        errs = RX.integrity_x13(recs)
        rep[label] = {"nat_bitwise_equal": not d, "diff_fields": list(d), "integrity_errors": errs, "arms": list(recs), "steps": Sr,
                      "pass": bool(not d and not errs)}
        del b
    return rep


def run_suite(families=FAMILIES, verbose=False):
    fails = []
    for fn in TESTS:
        for fam in families:
            try:
                fn(fam)
            except Exception as e:  # noqa: BLE001
                fails.append((f"{fn.__name__}[{fam}]", f"{type(e).__name__}: {e}"))
                if verbose:
                    traceback.print_exc()
            finally:
                X.uninstall()
                X.clear()
    return fails


def main():
    fails = run_suite(verbose=True)
    n = len(TESTS) * len(FAMILIES)
    print(f"{n - len(fails)}/{n} passed")
    for f in fails:
        print("FAIL", *f)
    rep = {"tiny_tests_total": n, "tiny_tests_failed": [list(f) for f in fails]}
    if "--real" in sys.argv:
        rep["real"] = real_checks()
        print(json.dumps(rep["real"], indent=1, default=str))
    if "--write" in sys.argv:
        (P8 / "x13_cpu_validation.json").write_text(json.dumps(rep, indent=1, default=str))
    ok = not fails and all(v["pass"] for v in rep.get("real", {}).values())
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
