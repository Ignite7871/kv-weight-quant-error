"""Follow-up (not part of the locked analysis): prevalence of the Tier B per-slot perturbation-norm deviation above 10% and sensitivity of the Tier B
conclusions to the affected observations. Rule fixed BEFORE looking at any contrast: a problem is 'affected' for a contrast if any step of any arm used in
that contrast has norm_dev_max > 0.10. No other exclusion rule is tried. Reads the raw records only.
  venv/Scripts/python.exe research_v2/followup/e1_tierB_norm_tolerance.py   -> e1_runs/tierB_norm_tolerance_assessment.json"""
import json
import sys
from pathlib import Path

import numpy as np

RV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RV))
import analyze_controls as A  # noqa: E402

THR = 0.10
load = lambda p: {json.loads(l)["index"]: json.loads(l) for l in open(p, encoding="utf-8")}   # noqa: E731
B = {a: load(RV / "e1_runs" / "tierB" / f"{a}.jsonl") for a in ("FP16", "NAT", "C1", "C2")}
ib = sorted(B["NAT"])
out = {"threshold": THR, "n_problems": len(ib), "prevalence": {}}
aff = {}
for a in ("C1", "C2"):
    steps = ex = 0
    ids = set()
    mx = 0.0
    for i in ib:
        m = np.asarray(B[a][i]["diag"]["norm_dev_max"])
        steps += len(m); ex += int((m > THR).sum()); mx = max(mx, float(m.max()))
        if (m > THR).any():
            ids.add(i)
    aff[a] = ids
    out["prevalence"][a] = {"steps": steps, "steps_above": ex, "share_of_steps_pct": 100 * ex / steps, "problems_above": sorted(ids), "max_deviation": mx,
                            "sweep_steps_above": {str(t): int(sum((np.asarray(B[a][i]["diag"]["norm_dev_max"]) > t).sum() for i in ib)) for t in (0.05, 0.075, 0.10, 0.125)}}
ref = {i: B["FP16"][i]["ids"] for i in ib}
Y = {a: {i: A.restricted_first_div(B[a][i]["ids"], ref[i]) for i in ib} for a in ("NAT", "C1", "C2")}
sens = {}
for name, (x, y), drop in (("NAT - C1", ("NAT", "C1"), aff["C1"]), ("NAT - C2", ("NAT", "C2"), aff["C2"])):
    keep = [i for i in ib if i not in drop]
    full = np.array([Y[x][i] - Y[y][i] for i in ib])
    sub = np.array([Y[x][i] - Y[y][i] for i in keep])
    rng = np.random.default_rng(A.SEED)
    sens[name] = {"locked_full": A.summarize(full, rng, None), "excluding_affected": A.summarize(sub, np.random.default_rng(A.SEED), None),
                  "n_excluded": len(drop), "affected_problem_values_NAT_minus_arm": {str(i): float(Y[x][i] - Y[y][i]) for i in sorted(drop)}}
out["sensitivity_tierB_restricted_first_divergence"] = sens
acc = {}
for a in ("FP16", "NAT", "C1", "C2"):
    acc[a] = float(np.mean([B[a][i]["status"] == "SOLVED" for i in ib]))
out["accuracy"] = acc
(RV / "e1_runs" / "tierB_norm_tolerance_assessment.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
