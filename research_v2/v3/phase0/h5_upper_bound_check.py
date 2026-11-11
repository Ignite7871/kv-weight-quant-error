"""v3 Phase 0 follow-up (author decision 1): distribution of the H5 90% upper bootstrap limit over 1,000 independent
10,000-resample streams for Stage 1 (n=930), held-out (n=389) and held-out excluding 47 exposed (n=342), and the fraction
of streams whose upper limit is >= 3.0 points. Same stream construction as recompute_phase0.py (multinomial over the
{-1,0,+1} value classes, numpy default percentile). Imports nothing from the existing analysis code. CPU only.

  venv/Scripts/python.exe research_v2/v3/phase0/h5_upper_bound_check.py
"""

import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RV = HERE.parents[1]


def status(d, arm):
    rs = [json.loads(x) for x in (d / f"{arm}.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    return {r["index"]: r["status"] == "SOLVED" for r in rs}


def norm_sf(z):
    return 0.5 * math.erfc(z / math.sqrt(2))


def main():
    exposed = set(json.loads((RV / "stage2_sample.json").read_text())["exposed_gold_trajectory_test_0_199"])
    o1, c1 = status(RV / "stage1_runs" / "eval", "oracle_flip"), status(RV / "stage1_runs" / "eval", "cache_only")
    o2, c2 = status(RV / "stage2_runs" / "eval", "oracle_flip"), status(RV / "stage2_runs" / "eval", "cache_only")
    samples = {"stage1": np.array([o1[i] - c1[i] for i in sorted(o1)], dtype=int),
               "heldout389": np.array([o2[i] - c2[i] for i in sorted(o2)], dtype=int),
               "heldout342": np.array([o2[i] - c2[i] for i in sorted(o2) if i not in exposed], dtype=int)}
    assert [len(v) for v in samples.values()] == [930, 389, 342]
    out = {"streams": 1000, "B_per_stream": 10_000, "threshold_pp": 3.0, "samples": {}}
    for k, d in samples.items():
        n = len(d)
        probs = np.array([(d == 1).sum(), (d == 0).sum(), (d == -1).sum()]) / n
        rng = np.random.default_rng(int.from_bytes(f"v3-h5-upper-{k}".encode(), "big") % 2**63)
        his = np.empty(1000)
        los = np.empty(1000)
        for s in range(1000):
            draw = rng.multinomial(n, probs, size=10_000)
            m = (draw[:, 0] - draw[:, 2]) / n * 100
            los[s], his[s] = np.percentile(m, [5, 95])
        est, se = d.mean(), d.std(ddof=1) / math.sqrt(n)
        tost = max(norm_sf((est + 0.03) / se), norm_sf((0.03 - est) / se))
        u, c = np.unique(np.round(his, 2), return_counts=True)
        ul, cl = np.unique(np.round(los, 2), return_counts=True)
        out["samples"][k] = {"n": n, "estimate_pp": 100 * est, "lattice_step_pp": 100 / n,
                             "upper_limit_distribution": {f"{a:+.2f}": int(b) for a, b in zip(u, c)},
                             "lower_limit_distribution": {f"{a:+.2f}": int(b) for a, b in zip(ul, cl)},
                             "upper_min": float(his.min()), "upper_max": float(his.max()),
                             "fraction_upper_ge_3": float((his >= 3.0).mean()),
                             "fraction_lower_le_minus3": float((los <= -3.0).mean()),
                             "tost_p_pm3_normal": tost}
    (HERE / "h5_upper_bound_check.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
