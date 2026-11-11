"""EXPLORATORY analysis X12 of v3/phase5/EXPLORATORY_PLAN_V3.md (dated entry 2026-11-01, committed fcdc74d before computing). Not locked.

Per-family tuning simple effects of the locked H-F decomposition; descriptive. Reads only the locked per-problem theta(NAT, C1).
  venv/Scripts/python.exe research_v2/v3/phase5/exploratory_v3_x12.py   -> v3/phase5/exploratory_v3_x12.json, EXPLORATORY_RESULTS_X12.md
"""
import json
import sys
from pathlib import Path

import numpy as np

RV = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RV))
import analyze_v3 as L                           # noqa: E402

HERE = Path(__file__).resolve().parent
SEED_X12 = 20261213
LABEL = "EXPLORATORY, descriptive (EXPLORATORY_PLAN_V3.md, dated entry X12, 2026-11-01; not blind)"


def main():
    rep = json.loads((HERE / "v3_report_locked.json").read_text())
    T = {m: rep["per_model"][m]["theta_NAT_C1_per_problem"] for m in L.ORDER}
    ids = sorted(set.intersection(*[set(v) for v in T.values()]))
    th = {m: np.array([T[m][i] for i in ids]) for m in L.ORDER}
    rng = np.random.default_rng(SEED_X12)
    eff = {"Qwen tuning effect (instruct - base)": th["qwen-instruct"] - th["qwen-base"],
           "Llama tuning effect (instruct - base)": th["llama-instruct"] - th["llama-base"]}
    out = {k: L.summarize(v, rng) for k, v in eff.items()}
    for (k, v), h in zip(out.items(), L.holm([v["p"] for v in out.values()])):
        v["holm_p"] = float(h)
        v.pop("verdict", None)
    tq, tl = (float(v.mean()) for v in eff.values())
    check = -0.5 * (tq - tl) - rep["HF"]["contrasts"]["interaction"]["estimate"]
    res = {"label": LABEL, "n_shared": len(ids), "effects": out, "check: -1/2 (T_Q - T_L) minus locked interaction": check}
    (HERE / "exploratory_v3_x12.json").write_text(json.dumps(res, indent=1, default=float))
    md = ["# EXPLORATORY results X12 — 2026-11-01", "", "**Descriptive and not blind** (EXPLORATORY_PLAN_V3.md, dated entry committed fcdc74d "
          "before this script ran). Source: `exploratory_v3_x12.json`.", "",
          "| Effect | mean [95 % CI] | 90 % CI | p | Holm p (2) |", "|---|---|---|---|---|"]
    for k, v in out.items():
        md.append(f"| {k} | {v['estimate']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] | [{v['ci90'][0]:+.3f}, {v['ci90'][1]:+.3f}] | "
                  f"{v['p']:.4f} | {v['holm_p']:.4f} |")
    md += ["", f"n = {len(ids)} shared problems. Check against the locked interaction: {check:+.2e}."]
    (HERE / "EXPLORATORY_RESULTS_X12.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
