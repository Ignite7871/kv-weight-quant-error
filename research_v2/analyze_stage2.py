"""Stage 2 analysis (CPU only): protocol-locked held-out validation. Implements STAGE2_SPEC.md section 6; written and tested before any
held-out data exist. It reuses the Stage 1 analysis (`analyze_stage1.analyze`) UNCHANGED for every within-sample statistic, so the
held-out and Stage 1 numbers are computed by identical code, and adds only (a) the Stage 1 vs held-out replication table,
(b) outcome-pattern counts, (c) a combined 1,319-problem analysis labelled secondary, (d) the pre-declared sensitivity analysis.

Usage: venv/Scripts/python.exe research_v2/analyze_stage2.py [--stage1 stage1_runs/eval] [--stage2 stage2_runs/eval] [--out stage2_analysis]

Only problems completed for ALL five arms are analysed. Accuracy counts only status == SOLVED.
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_stage1 as A                      # noqa: E402

ARMS = A.ARMS
B_HET = 10_000
SEED2 = 20261025
ALPHA = 0.05
PRIMARY = A.FAMILY                                # the same five contrasts, same Holm family as Stage 1
# Stage 1's primary-family point estimates (pp), copied from stage1_analysis/stage1_report.json for the record; the analysis
# recomputes them from the Stage 1 records and asserts they match.
STAGE1_ESTIMATES_PP = {"cache_only - fp16": -2.3656, "token_only - fp16": -3.5484,
                       "interaction (nf4 - cache_only - token_only + fp16)": -4.0860, "cache_only - nf4": 7.6344,
                       "oracle_flip - cache_only": -0.7527}


# ----------------------------------------------------------------------------- pure helpers (unit-tested)
def contrast_vector(R: dict, common: list, coef: dict) -> np.ndarray:
    Y = {a: np.array([1 if R[a][i]["status"] == "SOLVED" else 0 for i in common]) for a in ARMS}
    return sum(c * Y[a] for a, c in coef.items()).astype(float)


def heterogeneity(d1: np.ndarray, d2: np.ndarray, rng: np.random.Generator, B: int = B_HET) -> dict:
    """Held-out minus Stage 1 mean contrast. CI: independent bootstrap of the two samples. p: permutation of the stage label over the
    pooled problems (exchangeability under 'same contrast in both samples'). Two-sided."""
    n1, n2 = len(d1), len(d2)
    obs = d2.mean() - d1.mean()
    b1 = d1[rng.integers(0, n1, size=(B, n1))].mean(axis=1)
    b2 = d2[rng.integers(0, n2, size=(B, n2))].mean(axis=1)
    lo, hi = np.percentile(b2 - b1, [2.5, 97.5])
    pooled = np.concatenate([d1, d2])
    ge, done, chunk = 0, 0, 1000
    while done < B:
        m = min(chunk, B - done)
        perm = np.argsort(rng.random((m, n1 + n2)), axis=1)
        sel2 = pooled[perm[:, :n2]].mean(axis=1)
        sel1 = pooled[perm[:, n2:]].mean(axis=1)
        ge += int((np.abs(sel2 - sel1) >= abs(obs) - 1e-15).sum())
        done += m
    return {"diff_pp": 100 * obs, "ci95_pp": [100 * lo, 100 * hi], "p_permutation": (1 + ge) / (B + 1)}


def replication_category(e1_pp, e2_pp, p2_raw, p2_holm, het_ci_pp):
    """Pre-declared classification (STAGE2_SPEC.md section 6.3) for a directional contrast. Statistical uncertainty is kept apart from
    disagreement in direction: a sign flip counts as a directional DISagreement only if the held-out-minus-Stage-1 CI excludes 0."""
    same_sign = e2_pp != 0 and (e2_pp > 0) == (e1_pp > 0)
    compatible = het_ci_pp[0] <= 0 <= het_ci_pp[1]
    if same_sign and p2_holm < ALPHA:
        return "replicated: same direction and Holm-adjusted p < 0.05"
    if same_sign and p2_raw < ALPHA:
        return "same direction; raw p < 0.05 but not Holm-significant"
    if same_sign:
        return "same direction; not significant (inconclusive)"
    if compatible:
        return "opposite or zero point estimate, but compatible with Stage 1 (difference CI includes 0): statistical uncertainty"
    return "opposite direction and inconsistent with Stage 1 (difference CI excludes 0): directional disagreement"


def equivalence_category(ci90_pp, bound_pp=3.0):
    lo, hi = ci90_pp
    if lo > -bound_pp and hi < bound_pp:
        return "equivalence replicated (90% CI within +-3 pp)"
    if lo > 0 or hi < 0:
        return "difference from 0 (contradicts equivalence)"
    return "equivalence not demonstrated (inconclusive)"


def pattern_counts(R: dict, common: list) -> dict:
    """Counts of the joint outcome pattern over the five arms, written S (solved) / W (not) in ARMS order, plus pairwise disagreement counts."""
    pats = {}
    for i in common:
        p = "".join("S" if R[a][i]["status"] == "SOLVED" else "W" for a in ARMS)
        pats[p] = pats.get(p, 0) + 1
    dis = {f"{a} vs {b}": sum((R[a][i]["status"] == "SOLVED") != (R[b][i]["status"] == "SOLVED") for i in common)
           for a, b in itertools.combinations(ARMS, 2)}
    return {"arm_order": ARMS, "patterns": dict(sorted(pats.items(), key=lambda kv: -kv[1])), "n_distinct_patterns": len(pats),
            "all_five_agree": pats.get("SSSSS", 0) + pats.get("WWWWW", 0), "pairwise_outcome_disagreements": dis}


def replication_table(R1, c1, R2, c2, rep2_family, rng, B=B_HET) -> dict:
    table = {}
    for name, coef in PRIMARY:
        d1, d2 = contrast_vector(R1, c1, coef), contrast_vector(R2, c2, coef)
        het = heterogeneity(d1, d2, rng, B)
        f2 = rep2_family[name]
        row = {"stage1_estimate_pp": 100 * d1.mean(), "heldout_estimate_pp": f2["estimate_pp"], "heldout_ci95_pp": f2["ci95_pp"],
               "heldout_ci90_pp": f2["ci90_pp"], "heldout_p_raw": f2["p_signflip"], "heldout_p_holm": f2["p_holm"],
               "heldout_vs_stage1": het,
               "stage1_estimate_inside_heldout_ci95": f2["ci95_pp"][0] <= 100 * d1.mean() <= f2["ci95_pp"][1],
               "same_sign": bool(f2["estimate_pp"] != 0 and (f2["estimate_pp"] > 0) == (d1.mean() > 0))}
        if name == "oracle_flip - cache_only":
            row["category"] = equivalence_category(f2["ci90_pp"])
        else:
            row["category"] = replication_category(100 * d1.mean(), f2["estimate_pp"], f2["p_signflip"], f2["p_holm"], het["ci95_pp"])
        table[name] = row
    return table


def merge_samples(R1: dict, R2: dict) -> tuple[dict, list]:
    assert not (set.intersection(*[set(R1[a]) for a in ARMS]) & set.intersection(*[set(R2[a]) for a in ARMS])), "stage samples overlap"
    R = {a: {**R1[a], **R2[a]} for a in ARMS}
    return R, sorted(set.intersection(*[set(R[a]) for a in ARMS]))


def divergence_side_by_side(rep1: dict, rep2: dict) -> dict:
    out = {}
    for a in rep2["divergence_from_fp16"]:
        d1, d2 = rep1["divergence_from_fp16"][a], rep2["divergence_from_fp16"][a]
        out[a] = {"stage1": {"share_diverged": d1["share_diverged"], **d1["first_divergence_token"]},
                  "heldout": {"share_diverged": d2["share_diverged"], **d2["first_divergence_token"]}}
    return out


# ----------------------------------------------------------------------------- driver
def run(R1, c1, R2, c2, out: Path, exposed: list, B=B_HET) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED2)
    rep1 = A.analyze(R1, c1, None, None)
    for k, v in STAGE1_ESTIMATES_PP.items():
        assert abs(rep1["primary_family"][k]["estimate_pp"] - v) < 1e-3, f"Stage 1 reference estimate changed for {k}"
    rep2 = A.analyze(R2, c2, out / "heldout", None)
    (out / "heldout" / "stage1_report.json").replace(out / "heldout" / "heldout_report.json")
    final = {"label": "Stage 2 protocol-locked held-out validation (primary). Combined and sensitivity analyses below are SECONDARY.",
             "heldout": rep2,
             "accuracy_side_by_side": {a: {"stage1_pct": rep1["accuracy"][a]["pct"], "heldout_pct": rep2["accuracy"][a]["pct"]} for a in ARMS},
             "replication_vs_stage1": replication_table(R1, c1, R2, c2, rep2["primary_family"], rng, B),
             "outcome_patterns_heldout": pattern_counts(R2, c2), "outcome_patterns_stage1": pattern_counts(R1, c1),
             "first_divergence_side_by_side": divergence_side_by_side(rep1, rep2)}
    # ---- secondary: combined 1,319 problems (NOT independent confirmation: includes the exploratory Stage 1 data)
    Rc, cc = merge_samples(R1, R2)
    repc = A.analyze(Rc, cc, out / "combined_secondary", None)
    (out / "combined_secondary" / "stage1_report.json").replace(out / "combined_secondary" / "combined_report.json")
    final["combined_secondary"] = {"label": "SECONDARY. Pools the exploratory Stage 1 sample with the held-out sample. Not independent "
                                            "confirmation; the Stage 1 data motivated the contrasts.",
                                   "n": len(cc), "accuracy": repc["accuracy"], "primary_family": repc["primary_family"],
                                   "decomposition_pp_description_only": repc["decomposition_pp_description_only"],
                                   "divergence_from_fp16": repc["divergence_from_fp16"]}
    # ---- pre-declared sensitivity: exclude held-out problems whose gold trajectory was labelled in earlier exploration
    keep = [i for i in c2 if i not in set(exposed)]
    reps = A.analyze(R2, keep, None, None)
    final["sensitivity_excluding_prior_exposure"] = {"label": "SECONDARY sensitivity analysis (pre-declared): held-out problems whose gold solutions "
                                                              "were token-labelled in earlier exploration (test indices 0-199) are excluded.",
                                                     "n_excluded": len(c2) - len(keep), "n": len(keep), "accuracy": reps["accuracy"],
                                                     "primary_family": reps["primary_family"]}
    (out / "stage2_report.json").write_text(json.dumps(final, indent=1, default=float))
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", default=str(HERE / "stage1_runs" / "eval"))
    ap.add_argument("--stage2", default=str(HERE / "stage2_runs" / "eval"))
    ap.add_argument("--out", default=str(HERE / "stage2_analysis"))
    ap.add_argument("--sample", default=str(HERE / "stage2_sample.json"))
    a = ap.parse_args()
    R1, c1 = A.load_all(Path(a.stage1))
    R2, c2 = A.load_all(Path(a.stage2))
    s = json.loads(Path(a.sample).read_text())
    if not c1 or not c2:
        sys.exit("need completed problems for all five arms in both stages")
    assert set(c2) <= set(s["heldout389"]) and not set(c2) & set(s["pool930"]), "held-out records outside the held-out set"
    final = run(R1, c1, R2, c2, Path(a.out), s["exposed_gold_trajectory_test_0_199"])
    print(json.dumps({"n_heldout": len(c2), "accuracy": final["heldout"]["accuracy"],
                      "replication": {k: v["category"] for k, v in final["replication_vs_stage1"].items()}}, indent=1, default=float))


if __name__ == "__main__":
    main()
