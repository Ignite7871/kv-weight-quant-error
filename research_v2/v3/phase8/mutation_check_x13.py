"""Mutation check for X13 (v3/phase8/FOLLOWUP_X13.md section 5): every injected fault must make at least one test of test_x13.py fail.

  venv/Scripts/python.exe research_v2/v3/phase8/mutation_check_x13.py [--write]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

P8 = Path(__file__).resolve().parent
sys.path.insert(0, str(P8))

import test_x13 as TX                            # noqa: E402  (imports run_v3_tests, rv3x13, run_x13, analyze_x13)

import torch                                     # noqa: E402

X, RX, AX, V = TX.X, TX.RX, TX.AX, TX.V
SNAP = {"X": dict(vars(X)), "RX": dict(vars(RX)), "AX": dict(vars(AX)), "PS": dict(vars(X.CentredPredictorState)),
        "LOCKED": dict(X._LOCKED), "CEN": dict(X.CEN_ARMS), "SEED": dict(X.SEED_LABEL)}
O = SNAP["X"]


def restore():
    X.uninstall()
    X.clear()
    for name, mod in (("X", X), ("RX", RX), ("AX", AX)):
        for k, v in SNAP[name].items():
            if not k.startswith("__"):
                setattr(mod, k, v)
    for k in ("evaluate", "record_write"):
        setattr(X.CentredPredictorState, k, SNAP["PS"][k])
    X._LOCKED.clear()
    X._LOCKED.update(SNAP["LOCKED"])
    X.CEN_ARMS.clear()
    X.CEN_ARMS.update(SNAP["CEN"])
    X.SEED_LABEL.clear()
    X.SEED_LABEL.update(SNAP["SEED"])


def m_mu_not_subtracted():
    X._centred = lambda arm, alpha, kF, kM, ctx: O["_centred"](arm, alpha, kF, kM, {**ctx, "mu": torch.zeros_like(ctx["mu"])})


def m_mu_wrong_axis():
    def f(inst):
        mu = O["mu_from"](inst)
        return mu.mean(1, keepdim=True).expand_as(mu).clone()                       # averaged over KV heads as well
    X.mu_from = f


def m_renorm_to_centred_norm():
    def f(arm, alpha, kF, kM, ctx):
        sl = slice(0, None, 2) if X.CEN_ARMS[arm] == "K" else slice(1, None, 2)
        D = kM.double() - kF.double()
        c = D[sl] - ctx["mu"].double()[sl]
        scale = (c.norm(dim=-1, keepdim=True) / D[sl].norm(dim=-1, keepdim=True).clamp_min(1e-300))
        fake = kF.double().clone()
        fake[sl] = kF.double()[sl] + D[sl] * scale                                  # Delta rescaled so that ||Delta|| becomes ||Delta - mu||
        return O["_centred"](arm, alpha, kF, fake.half(), ctx)
    X._centred = f


def m_centred_wrong_path():
    X.CEN_ARMS.update({"NAT-K-cen": "V", "NAT-V-cen": "K"})


def m_lambda_search_maximizes():
    def f(e, dim):
        mx = e.max(dim).values
        return mx, (e == mx.unsqueeze(dim)).int().argmax(dim)
    X._first_min = f


def m_greedy_accepts_worse():
    def f(r, F, t, e):
        moves = torch.zeros(r.shape[0], dtype=torch.long, device=r.device)
        hd = r.shape[1]
        for _ in range(X.NC_MAX_MOVES):
            up, dn = X.fp16_neighbors(r)
            cands = torch.stack([up, dn])
            diff = r.double() - F
            n2 = (diff * diff).sum(-1)
            dc = cands.double() - F.unsqueeze(0)
            new = (n2[None, :, None] - diff.unsqueeze(0) ** 2 + dc * dc).clamp_min(0).sqrt()
            ce = ((new - t[None, :, None]).abs() / t[None, :, None]).permute(1, 0, 2).reshape(r.shape[0], 2 * hd)
            arg = ce.argmin(-1)
            rows = torch.arange(r.shape[0])
            r = r.clone()
            r[rows, arg % hd] = cands[arg // hd, rows, arg % hd]                   # applied whether or not it reduces the residual
            moves += 1
        return r, X._resid(r, F, t), moves
    X._refine = f


def m_c1nc_new_seed_stream():
    X.SEED_LABEL.clear()


def m_signed_logged_unsigned():
    def f(rn, t, pos, prefix=""):
        return O["signed_stats"](t + (rn - t).abs(), t, pos, prefix)
    X.signed_stats = f


def m_plc_decode_only():
    def f(q, K, hmap, scaling, lo, n, dk, hd):
        p = V._attn_probs(q, K, hmap, scaling)[:, lo:lo + n]
        dl = torch.einsum("gd,gjd->gj", q, dk)
        m = (p * dl).sum(-1, keepdim=True) / p.sum(-1, keepdim=True)
        return float((p * (dl - m).abs()).sum()) / hd ** 0.5
    X.centred_logit_term = f


def m_rayleigh_wrong_head():
    X.rayleigh_num = lambda dk, M: O["rayleigh_num"](dk, torch.roll(M, 1, dims=1))


def m_cm_share_unscaled():
    def f(norms, S1):
        n = norms.shape[0]
        return float((S1 ** 2).sum() / ((norms ** 2).sum(0) / n).sum())
    AX.cm_share = f


def m_comparator_tolerance():
    same = SNAP["RX"]["same"]

    def f(a, b):
        if isinstance(a, float) and isinstance(b, float) and a == a and b == b:
            return abs(a - b) <= 1e-12 * max(abs(b), 1e-300)
        return same(a, b)
    RX.same = f


def m_auth_ignores_manifest_id():
    RX.auth_ok = lambda text, mid: RX.AUTH_TEXT in text


def m_history_always_matches():
    RX.history_matches = lambda regen, locked: True


def m_subset_seed_changed():
    RX.SUBSET_SEED = 20261208


def m_integrity_drops_all_construction():
    def f(recs):
        errs = [e for e in RX.R.integrity_errors(recs) if ": construction" not in e]
        for tag, r in recs.items():
            if tag in X.NC_ARMS and not all(r.get("nc_ok") or [False]):
                errs.append(f"{tag}: nc_ok false")
        return errs
    RX.integrity_x13 = f


def m_gap_inverted():
    g = SNAP["AX"]["gap_closed"]

    def f(a, b, c, rng):
        G, ci = g(a, b, c, rng)
        return (1.0 / G if G else float("inf")), [1.0 / ci[1] if ci[1] else float("inf"), 1.0 / ci[0] if ci[0] else float("inf")]
    AX.gap_closed = f


def m_d_full_width():
    AX.half_width = lambda ci: ci[1] - ci[0]


MUTANTS = [m_mu_not_subtracted, m_mu_wrong_axis, m_renorm_to_centred_norm, m_centred_wrong_path, m_lambda_search_maximizes,
           m_greedy_accepts_worse, m_c1nc_new_seed_stream, m_signed_logged_unsigned, m_plc_decode_only, m_rayleigh_wrong_head,
           m_cm_share_unscaled, m_comparator_tolerance, m_auth_ignores_manifest_id, m_history_always_matches, m_subset_seed_changed,
           m_integrity_drops_all_construction, m_gap_inverted, m_d_full_width]


def main():
    restore()
    base = TX.run_suite()
    if base:
        print("baseline fails; fix the suite first:", base)
        sys.exit(1)
    rep, survived = {}, []
    for m in MUTANTS:
        t0 = time.time()
        try:
            m()
            fails = TX.run_suite()
        finally:
            restore()
        rep[m.__name__] = {"caught": bool(fails), "failing_tests": sorted({f[0] for f in fails})[:6], "seconds": round(time.time() - t0, 1)}
        print(m.__name__, "CAUGHT" if fails else "SURVIVED", rep[m.__name__]["failing_tests"][:3], flush=True)
        if not fails:
            survived.append(m.__name__)
    print(f"{len(MUTANTS) - len(survived)}/{len(MUTANTS)} mutants caught")
    if "--write" in sys.argv:
        (P8 / "x13_mutation_check.json").write_text(json.dumps({"mutants": rep, "caught": len(MUTANTS) - len(survived), "total": len(MUTANTS)}, indent=1))
    sys.exit(0 if not survived else 1)


if __name__ == "__main__":
    main()
