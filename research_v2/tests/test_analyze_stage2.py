"""Tests for analyze_stage2.py on synthetic records with known answers (CPU, seconds). No held-out data are used.

Run: venv/Scripts/python.exe research_v2/tests/test_analyze_stage2.py
"""

import copy
import json
import sys
import tempfile
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np

import analyze_stage1 as A
import analyze_stage2 as B
import test_analyze_stage1 as T1

RESULTS = []


def test(fn):
    def wrapper():
        try:
            fn()
            RESULTS.append((fn.__name__, "PASS", ""))
        except Exception as e:  # noqa: BLE001
            RESULTS.append((fn.__name__, "FAIL", f"{type(e).__name__}: {e}"))
            traceback.print_exc()
    wrapper.__name__ = fn.__name__
    return wrapper


def reindexed(R, common, offset):
    R2 = {a: {} for a in A.ARMS}
    for a in A.ARMS:
        for i in common:
            r = copy.deepcopy(R[a][i])
            r["index"] = i + offset
            R2[a][i + offset] = r
    return R2, [i + offset for i in common]


@test
def b1_replication_categories_separate_uncertainty_from_directional_disagreement():
    rc = B.replication_category
    assert rc(-3.5, -4.0, 0.01, 0.03, [-3, 2]).startswith("replicated")
    assert rc(-3.5, -4.0, 0.01, 0.07, [-3, 2]).startswith("same direction; raw p")
    assert rc(-3.5, -1.0, 0.40, 0.80, [-3, 2]).startswith("same direction; not significant")
    assert "statistical uncertainty" in rc(-3.5, +1.0, 0.60, 0.90, [-1, 7])          # sign flip, difference CI includes 0
    assert "directional disagreement" in rc(-3.5, +6.0, 0.01, 0.04, [4, 14])         # sign flip, difference CI excludes 0
    assert "statistical uncertainty" in rc(-3.5, 0.0, 1.0, 1.0, [-1, 7])             # exactly zero is not "same direction"
    assert rc(+7.6, +9.0, 1e-4, 4e-4, [-3, 6]).startswith("replicated")
    assert B.equivalence_category([-2.0, 2.0]).startswith("equivalence replicated")
    assert B.equivalence_category([-4.0, 1.0]).startswith("equivalence not demonstrated")
    assert B.equivalence_category([0.5, 5.0]).startswith("difference from 0")
    assert B.equivalence_category([-3.0, 2.0]).startswith("equivalence not demonstrated")      # boundary is strict


@test
def b2_heterogeneity_has_a_calibrated_null_and_detects_a_shift():
    rng = np.random.default_rng(0)
    base = (rng.random(900) < 0.30).astype(float) - (rng.random(900) < 0.35).astype(float)       # contrast values in {-1,0,1}
    d1, d2 = base[:600], base[600:900]
    same = B.heterogeneity(d1, d2, np.random.default_rng(1), B=2000)
    assert same["p_permutation"] > 0.05 and same["ci95_pp"][0] < same["diff_pp"] < same["ci95_pp"][1]
    assert abs(same["diff_pp"] - 100 * (d2.mean() - d1.mean())) < 1e-9
    shifted = d2 + 0.15 * (rng.random(300) < 1.0)                                                # +15 pp shift
    het = B.heterogeneity(d1, shifted, np.random.default_rng(2), B=2000)
    assert het["p_permutation"] < 0.01 and het["ci95_pp"][0] > 0 and abs(het["diff_pp"] - 100 * (shifted.mean() - d1.mean())) < 1e-9
    # null calibration: under exchangeability the rejection rate at 0.05 is near 0.05
    rej = 0
    r2 = np.random.default_rng(3)
    for _ in range(100):
        x = (r2.random(500) < 0.4).astype(float)
        rej += B.heterogeneity(x[:300], x[300:], r2, B=200)["p_permutation"] < 0.05
    assert rej <= 14, rej


@test
def b3_pattern_counts_hand_checked():
    def mk(statuses):
        return {a: {i: {"status": s} for i, s in enumerate(col)} for a, col in zip(A.ARMS, statuses)}
    # 4 problems; arm order fp16, nf4, cache_only, token_only, oracle_flip
    R = mk([["SOLVED", "SOLVED", "WRONG", "WRONG"], ["WRONG", "SOLVED", "WRONG", "WRONG"], ["SOLVED", "SOLVED", "WRONG", "SOLVED"],
            ["SOLVED", "SOLVED", "WRONG", "WRONG"], ["SOLVED", "SOLVED", "WRONG", "SOLVED"]])
    pc = B.pattern_counts(R, [0, 1, 2, 3])
    assert pc["patterns"] == {"SWSSS": 1, "SSSSS": 1, "WWWWW": 1, "WWSWS": 1}, pc["patterns"]
    assert pc["all_five_agree"] == 2 and pc["n_distinct_patterns"] == 4
    assert pc["pairwise_outcome_disagreements"]["fp16 vs nf4"] == 1 and pc["pairwise_outcome_disagreements"]["cache_only vs token_only"] == 1
    assert pc["pairwise_outcome_disagreements"]["cache_only vs oracle_flip"] == 0


@test
def b4_merge_requires_disjoint_samples_and_unions_them():
    R1, c1 = T1.build(n=20, seed=1)
    R2, c2 = reindexed(*T1.build(n=15, seed=2), 1000)
    Rc, cc = B.merge_samples(R1, R2)
    assert len(cc) == 35 and cc == sorted(c1 + c2)
    try:
        B.merge_samples(R1, R1)
    except AssertionError:
        pass
    else:
        raise AssertionError("overlapping stages were merged")


@test
def b5_full_run_on_synthetic_stages_is_consistent_with_the_stage1_code_and_writes_all_outputs():
    R1, c1 = T1.build(n=60, seed=1)
    R2, c2 = reindexed(*T1.build(n=45, seed=2), 1000)
    exposed = c2[:7]
    old = B.STAGE1_ESTIMATES_PP
    B.STAGE1_ESTIMATES_PP = {}
    try:
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            f = B.run(R1, c1, R2, c2, out, exposed, B=300)
            for p in ("stage2_report.json", "heldout/heldout_report.json", "heldout/paired_outcomes.csv", "heldout/divergence_survival.csv",
                      "combined_secondary/combined_report.json", "combined_secondary/paired_outcomes.csv"):
                assert (out / p).exists(), p
            assert not (out / "heldout" / "stage1_report.json").exists()
            direct = A.analyze(R2, c2, None, None)
            assert f["heldout"]["accuracy"] == direct["accuracy"]
            assert f["heldout"]["primary_family"]["token_only - fp16"]["estimate_pp"] == direct["primary_family"]["token_only - fp16"]["estimate_pp"]
            assert f["combined_secondary"]["n"] == 105 and "SECONDARY" in f["combined_secondary"]["label"]
            assert f["sensitivity_excluding_prior_exposure"]["n"] == 38 and f["sensitivity_excluding_prior_exposure"]["n_excluded"] == 7
            assert set(f["replication_vs_stage1"]) == {n for n, _ in A.FAMILY}
            for name, row in f["replication_vs_stage1"].items():
                d1 = B.contrast_vector(R1, c1, dict(A.FAMILY)[name])
                assert abs(row["stage1_estimate_pp"] - 100 * d1.mean()) < 1e-9 and abs(row["heldout_estimate_pp"] - direct["primary_family"][name]["estimate_pp"]) < 1e-9
                assert row["category"]
            assert f["outcome_patterns_heldout"]["pairwise_outcome_disagreements"]["fp16 vs nf4"] == \
                direct["paired_tables"]["nf4_vs_fp16"]["nf4_only"] + direct["paired_tables"]["nf4_vs_fp16"]["fp16_only"]
            # the combined family equals a direct analysis of the pooled records (same code, same data)
            Rc, cc = B.merge_samples(R1, R2)
            pooled = A.analyze(Rc, cc, None, None)
            assert f["combined_secondary"]["primary_family"]["cache_only - nf4"]["estimate_pp"] == pooled["primary_family"]["cache_only - nf4"]["estimate_pp"]
            json.loads((out / "stage2_report.json").read_text())
    finally:
        B.STAGE1_ESTIMATES_PP = old


@test
def b6_stage1_reference_estimates_are_asserted_against_the_stage1_records():
    R1, c1 = T1.build(n=40, seed=1)
    R2, c2 = reindexed(*T1.build(n=30, seed=2), 1000)
    with tempfile.TemporaryDirectory() as d:
        try:
            B.run(R1, c1, R2, c2, Path(d), [], B=100)
        except AssertionError as e:
            assert "Stage 1 reference estimate" in str(e)
        else:
            raise AssertionError("a mismatching Stage 1 reference was accepted")


@test
def b7_real_stage1_records_reproduce_the_recorded_reference_estimates():
    root = HERE.parent / "stage1_runs" / "eval"
    if not (root / "fp16.jsonl").exists():
        return                                                                     # raw records are not in git
    R1, c1 = A.load_all(root)
    rep = A.analyze(R1, c1, None, None)
    for k, v in B.STAGE1_ESTIMATES_PP.items():
        assert abs(rep["primary_family"][k]["estimate_pp"] - v) < 1e-3, k


def main():
    tests = [v for k, v in globals().items() if k.startswith("b") and k[1:2].isdigit() and callable(v)]
    for t in tests:
        t()
    w = max(len(n) for n, _, _ in RESULTS)
    print("\n" + "=" * (w + 30))
    for name, status, msg in RESULTS:
        print(f"{name:<{w}}  {status}  {msg}")
    n_fail = sum(s == "FAIL" for _, s, _ in RESULTS)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
