# EXPLORATORY feasibility check (post hoc, not a confirmatory result).
# Compares router's lag-1 signal H(p_{t-1}) with same-step FP16 confidence H(p_t)
# as predictors of EQF_t, on stored teacher-forced data. Problem-level bootstrap.
import sys, numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
rng = np.random.default_rng(0)
for path in sys.argv[1:]:
    d = pd.read_csv(path)
    d = d.sort_values(["problem_id", "token_idx"])
    g = d.groupby("problem_id")
    d["H_same"] = g["entropy"].shift(-1)      # FP16 entropy of the step whose JS is in this row
    d["M_same"] = g["margin"].shift(-1)
    d = d.dropna(subset=["H_same"])
    tail = (d.js_divergence > 0.01).astype(int)
    print(path.split("_")[-1], "n=", len(d), "tail rate=%.3f" % tail.mean(), "flip rate=%.3f" % d.argmax_flip.mean())
    for name, x, sign in [("H lag1 (router)", d.entropy, 1), ("H same-step", d.H_same, 1),
                          ("margin lag1", d.margin, -1), ("margin same-step", d.M_same, -1)]:
        rho = spearmanr(x, d.js_divergence).correlation
        a_tail = roc_auc_score(tail, sign * x); a_flip = roc_auc_score(d.argmax_flip, sign * x)
        # problem bootstrap for rho
        pids = d.problem_id.unique(); idx = {p: np.where(d.problem_id.values == p)[0] for p in pids}
        bs = []
        for _ in range(300):
            s = np.concatenate([idx[p] for p in rng.choice(pids, len(pids))])
            bs.append(spearmanr(x.values[s], d.js_divergence.values[s]).correlation)
        print(f"  {name:18s} rho={sign*rho:+.3f} [{np.percentile(np.array(bs)*sign,2.5):+.3f},{np.percentile(np.array(bs)*sign,97.5):+.3f}]  AUC(EQF>0.01)={a_tail:.3f}  AUC(flip)={a_flip:.3f}")
