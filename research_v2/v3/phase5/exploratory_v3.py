"""EXPLORATORY analyses X1 and X2 (formerly E1, E2) of v3/phase5/EXPLORATORY_PLAN_V3.md (2026-10-31). Not locked. X3 is not computable from the records.

Uses the locked analysis primitives (analyze_v3.py, imported unchanged): bootstrap, sign-flip test, verdicts, Holm, ranking, Spearman, H-A D.
  venv/Scripts/python.exe research_v2/v3/phase5/exploratory_v3.py   -> v3/phase5/exploratory_v3.json
"""
import json
import sys
from pathlib import Path

import numpy as np

RV = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RV))
import analyze_v3 as L                           # noqa: E402

MODELS = L.ORDER
RUNS = RV / "v3_runs"
SEED_X1, SEED_X2 = 20261201, 20261202
LABEL = "EXPLORATORY (EXPLORATORY_PLAN_V3.md, 2026-10-31)"


def load(m, arms):
    return {a: L.load_jsonl(RUNS / m / "tierA" / f"{a.replace('*', 'star')}.jsonl") for a in arms}


def m_(r):
    return float(np.exp(L.logm(r)))                                   # per-problem mean JS, floored at 1e-9


def with_holm(res):
    for (m, v), h in zip(res.items(), L.holm([v["p"] for v in res.values()])):
        v["holm_p"] = float(h)


def pooled(per_problem, rng):
    sh = sorted(set.intersection(*[set(d) for d in per_problem.values()]))
    return L.summarize([np.mean([per_problem[m][i] for m in MODELS]) for i in sh], rng) if sh else None


def spearman_ci(x, y, rng):
    x, y = np.asarray(x), np.asarray(y)
    bi = L.boot_idx(len(x), rng)
    b = [L.spearman(x[k], y[k]) for k in bi]
    return {"rho": L.spearman(x, y), "ci95": [float(v) for v in np.nanpercentile(b, [2.5, 97.5])]}


def x1():
    rng = np.random.default_rng(SEED_X1)
    out, diffs = {}, {}
    for m in MODELS:
        A = load(m, ["NAT", "C1", "NAT-K", "NAT-V", "C1-K", "C1-V"])
        cal = json.loads((RUNS / m / "calibration.json").read_text())
        idx = sorted(set.intersection(*[set(v) for v in A.values()]))
        wK = np.array([m_(A["NAT-K"][i]) / (m_(A["NAT-K"][i]) + m_(A["NAT-V"][i])) for i in idx])
        tK = L.theta(A, "NAT-K", "C1-K", idx)
        tV = L.theta(A, "NAT-V", "C1-V", idx)
        pred = wK * tK + (1 - wK) * tV
        obs = L.theta(A, "NAT", "C1", idx)
        sp, so = L.summarize(pred, rng), L.summarize(obs, rng)
        d = L.summarize(obs - pred, rng, cal["margin_log_js"], cal["dose_response_informative"])
        excl = lambda s: s["ci95"][0] > 0 or s["ci95"][1] < 0
        out[m] = {"n": len(idx), "mean_wK": float(wK.mean()), "wK_median": float(np.median(wK)),
                  "mean_theta_K": float(tK.mean()), "mean_theta_V": float(tV.mean()),
                  "pred": {k: sp[k] for k in ("estimate", "ci95", "p")}, "obs": {k: so[k] for k in ("estimate", "ci95", "p")},
                  "obs_minus_pred": d, "margin": cal["margin_log_js"],
                  "sign_predicted": bool(np.sign(sp["estimate"]) == np.sign(so["estimate"]) and excl(sp) and excl(so)),
                  "spearman_pred_obs": spearman_ci(pred, obs, rng)}
        diffs[m] = d
    with_holm(diffs)
    po = [out[m]["pred"]["estimate"] for m in MODELS]
    oo = [out[m]["obs"]["estimate"] for m in MODELS]
    order = {"models": MODELS, "pred_means": po, "obs_means": oo,
             "pred_order": [MODELS[k] for k in np.argsort(po)], "obs_order": [MODELS[k] for k in np.argsort(oo)]}
    order["order_matches"] = order["pred_order"] == order["obs_order"]
    return {"label": LABEL, "per_model": out, "model_order": order}


def comb_records(nat):
    out = {}
    for i, r in nat.items():
        S = r["S"]
        rr = dict(r)
        rr["P_comb"] = [0.0] + list(L.rankdata(r["P_logit"][1:S]) + L.rankdata(r["P_val"][1:S]))
        out[i] = rr
    return out


def x2():
    rng = np.random.default_rng(SEED_X2)
    parts = {"X2a P_val vs P_norm": {}, "X2b P_comb vs P_norm": {}, "X2c D_comb - D_logit": {}}
    pp = {k: {} for k in parts}
    for m in MODELS:
        nat = comb_records(load(m, ["NAT"])["NAT"])
        idx = sorted(nat)
        Dv, ev = L.ha_D(nat, idx, pred="P_val")
        Dc, ec = L.ha_D(nat, idx, pred="P_comb")
        Dl, el = L.ha_D(nat, idx, pred="P_logit")
        Dv, Dc, Dl = dict(Dv), dict(Dc), dict(Dl)
        both = sorted(set(Dc) & set(Dl))
        for name, D, ex in (("X2a P_val vs P_norm", Dv, ev), ("X2b P_comb vs P_norm", Dc, ec)):
            s = L.summarize(list(D.values()), rng)
            s["verdict"] = "difference" if s["verdict"] == "difference" else "inconclusive"
            parts[name][m] = {**s, "excluded_problems": ex}
            pp[name][m] = D
        dd = {i: Dc[i] - Dl[i] for i in both}
        s = L.summarize(list(dd.values()), rng)
        s["verdict"] = "difference" if s["verdict"] == "difference" else "inconclusive"
        parts["X2c D_comb - D_logit"][m] = {**s, "excluded_problems": len(idx) - len(both)}
        pp["X2c D_comb - D_logit"][m] = dd
    res = {"label": LABEL}
    for name in parts:
        with_holm(parts[name])
        res[name] = {"per_model": parts[name], "pooled_mean_over_models": pooled(pp[name], rng)}
    return res


def main():
    out = {"X1": x1(), "X2": x2(), "X3": {"status": "not computable from the locked records (predictors are summed over layers before they are "
                                                    "written); not run; needs an author decision (EXPLORATORY_PLAN_V3.md)"}}
    p = Path(__file__).with_name("exploratory_v3.json")
    p.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
