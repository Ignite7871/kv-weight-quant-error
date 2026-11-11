"""EXPLORATORY analysis X10 of v3/phase5/EXPLORATORY_PLAN_V3.md (dated entry 2026-10-31, committed 7697aaf before computing). Not locked.

Path scaling of NAT-K* / NAT-V* against the calibrated dose slope; descriptive. Locked records and frozen calibration files only.
  venv/Scripts/python.exe research_v2/v3/phase5/exploratory_v3_x10.py   -> v3/phase5/exploratory_v3_x10.json, EXPLORATORY_RESULTS_X10.md
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

RV = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RV))
import analyze_v3 as L                           # noqa: E402

MODELS = L.ORDER
RUNS = RV / "v3_runs"
HERE = Path(__file__).resolve().parent
SEED_X10 = 20261210
LABEL = "EXPLORATORY, descriptive (EXPLORATORY_PLAN_V3.md, dated entry X10, 2026-10-31; not blind)"
PATHS = {"K": ("NAT-K*", "NAT-K", "scale_factor_K"), "V": ("NAT-V*", "NAT-V", "scale_factor_V")}


def load(m, arms):
    return {a: L.load_jsonl(RUNS / m / "tierA" / f"{a.replace('*', 'star')}.jsonl") for a in arms}


def x10():
    rng = np.random.default_rng(SEED_X10)
    out = {}
    for m in MODELS:
        cal = json.loads((RUNS / m / "calibration.json").read_text())
        b, se = cal["dose_slope_b_hat"], cal["dose_slope_problem_sd"] / math.sqrt(cal["n_calibration_problems"])
        out[m] = {"b_hat": b, "se_b": se}
        for p, (star, nat, sk) in PATHS.items():
            A = load(m, [star, nat])
            idx = sorted(set(A[star]) & set(A[nat]))
            s = L.summarize(L.theta(A, star, nat, idx), rng)
            sp = cal[sk]
            lp = b * math.log(sp)
            out[m][p] = {"n": len(idx), "s": sp, "theta": s["estimate"], "theta_ci95": s["ci95"],
                         "R_obs": math.exp(s["estimate"]), "R_obs_ci95": [math.exp(x) for x in s["ci95"]],
                         "R_pred": math.exp(lp), "R_pred_range": sorted([sp ** (b - 1.96 * se), sp ** (b + 1.96 * se)]),
                         "excess_E": s["estimate"] - lp, "excess_E_ci95": [x - lp for x in s["ci95"]],
                         "R_obs_over_R_pred": math.exp(s["estimate"] - lp)}
    return out


def to_md(o):
    Lm = ["# EXPLORATORY results X10 — 2026-10-31", "",
          "**Exploratory, descriptive and not blind** (EXPLORATORY_PLAN_V3.md, dated entry committed 7697aaf before this script ran; "
          "the point values had already been computed by hand). Source: `exploratory_v3_x10.json`. n = 200 per model.", "",
          "R_pred = s^b̂ is a uniform-scaling benchmark: b̂ is C1's dose slope (both paths, isotropic), and NAT-P* sets each slot's error "
          "to ρ*·‖κ^F_P‖, so its per-slot effective scale is ρ*/ρ_slot, not s.", "",
          "| Model | Path | s | b̂ (SE) | R_obs [95 % CI] | R_pred = s^b̂ [b̂ ± 1.96 SE] | E = ln R_obs − b̂ ln s [95 % CI] | R_obs / R_pred |",
          "|---|---|---|---|---|---|---|---|"]
    for m in MODELS:
        v = o[m]
        for p in PATHS:
            r = v[p]
            Lm.append(f"| {m} | {p} | {r['s']:.3f} | {v['b_hat']:.3f} ({v['se_b']:.3f}) | {r['R_obs']:.3f} [{r['R_obs_ci95'][0]:.3f}, {r['R_obs_ci95'][1]:.3f}] | "
                      f"{r['R_pred']:.3f} [{r['R_pred_range'][0]:.3f}, {r['R_pred_range'][1]:.3f}] | "
                      f"{r['excess_E']:+.3f} [{r['excess_E_ci95'][0]:+.3f}, {r['excess_E_ci95'][1]:+.3f}] | {r['R_obs_over_R_pred']:.2f} |")
    return "\n".join(Lm) + "\n"


def main():
    res = {"label": LABEL, "per_model": x10()}
    (HERE / "exploratory_v3_x10.json").write_text(json.dumps(res, indent=1, default=float))
    md = to_md(res["per_model"])
    (HERE / "EXPLORATORY_RESULTS_X10.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
