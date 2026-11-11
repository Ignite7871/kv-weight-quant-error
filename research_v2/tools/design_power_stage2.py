"""Design power of the Stage 2 held-out validation (n = 389), computed from Stage 1 data only (no held-out data exist).

For each primary-family contrast: the per-problem contrast values d_i observed in Stage 1 (n = 930) define a sampling distribution; a
held-out sample of 389 is simulated by resampling Stage 1 problems with replacement, as if the Stage 1 distribution were the truth.
The held-out test is approximated by the normal test on the mean (the locked analysis uses a sign-flip permutation test and a
percentile bootstrap, which agree closely at this n). Reported: standard error and the 80%-power minimum detectable effect at
alpha = 0.05 (raw), the probability that the raw two-sided p < 0.05, that the Holm-adjusted p (over the five contrasts) < 0.05, that
the point estimate has the Stage 1 sign, and (oracle_flip - cache_only) that the 90% CI lies within +-3 pp.

Caveat stated in the spec: Stage 1 estimates selected for attention are probably somewhat too large (winner's curse), so the real
power is probably lower than these figures.

  python research_v2/tools/design_power_stage2.py [--out research_v2/validation_logs/stage2_design_power.json]
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

RV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RV))
import analyze_stage1 as A                   # noqa: E402


def norm_sf(z):
    return 0.5 * math.erfc(z / math.sqrt(2))


def holm_rows(P):
    """Row-wise Holm adjustment for an (S, m) matrix of raw p-values."""
    S, m = P.shape
    order = np.argsort(P, axis=1)
    sorted_p = np.take_along_axis(P, order, axis=1)
    adj_sorted = np.minimum(1.0, np.maximum.accumulate(sorted_p * (m - np.arange(m)), axis=1))
    adj = np.empty_like(adj_sorted)
    np.put_along_axis(adj, order, adj_sorted, axis=1)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RV / "validation_logs" / "stage2_design_power.json"))
    ap.add_argument("--n", type=int, default=389)
    ap.add_argument("--sims", type=int, default=20000)
    a = ap.parse_args()
    R, common = A.load_all(RV / "stage1_runs" / "eval")
    Y = {x: np.array([1 if R[x][i]["status"] == "SOLVED" else 0 for i in common]) for x in A.ARMS}
    names = [n for n, _ in A.FAMILY]
    D = np.stack([sum(c * Y[x] for x, c in coef.items()).astype(float) for _, coef in A.FAMILY], axis=1)   # (930, 5)
    rng = np.random.default_rng(20261025)
    idx = rng.integers(0, len(D), size=(a.sims, a.n))
    samp = D[idx]                                                    # (sims, n, 5)
    mean = samp.mean(axis=1)
    se = samp.std(axis=1, ddof=1) / math.sqrt(a.n)
    z = mean / np.maximum(se, 1e-12)
    P = np.vectorize(lambda x: 2 * norm_sf(abs(x)))(z)
    H = holm_rows(P)
    est = D.mean(axis=0)
    out = {"n_heldout": a.n, "sims": a.sims, "stage1_n": len(common), "contrasts": {}}
    for j, name in enumerate(names):
        se_true = D[:, j].std(ddof=1) / math.sqrt(a.n)
        row = {"stage1_estimate_pp": 100 * est[j], "se_pp_at_n": 100 * se_true, "mde80_pp": 100 * 2.8016 * se_true,
               "discordant_share_stage1": float((D[:, j] != 0).mean()),
               "p_raw_lt_0.05": float((P[:, j] < 0.05).mean()), "p_holm_lt_0.05": float((H[:, j] < 0.05).mean()),
               "p_same_sign_as_stage1": float((np.sign(mean[:, j]) == np.sign(est[j])).mean())}
        if name == "oracle_flip - cache_only":
            row["p_ci90_within_pm3pp"] = float(((np.abs(mean[:, j]) + 1.645 * se[:, j]) < 0.03).mean())
        out["contrasts"][name] = row
    out["note"] = ("Assumes the Stage 1 distribution is the truth; Stage 1 estimates are likely optimistic (winner's curse), so true "
                   "power is probably lower. The five Holm-adjusted figures use the same family as the locked analysis.")
    Path(a.out).write_text(json.dumps(out, indent=1))
    for k, v in out["contrasts"].items():
        print(f"{k:55s} est {v['stage1_estimate_pp']:+.2f} pp  SE {v['se_pp_at_n']:.2f}  MDE80 {v['mde80_pp']:.2f}  "
              f"P(raw p<.05) {v['p_raw_lt_0.05']:.2f}  P(Holm<.05) {v['p_holm_lt_0.05']:.2f}  P(same sign) {v['p_same_sign_as_stage1']:.2f}"
              + (f"  P(TOST) {v['p_ci90_within_pm3pp']:.2f}" if 'p_ci90_within_pm3pp' in v else ""))


if __name__ == "__main__":
    main()
