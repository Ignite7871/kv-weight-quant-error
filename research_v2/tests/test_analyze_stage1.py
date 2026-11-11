"""Tests for analyze_stage1.py on synthetic records with known answers (CPU, seconds).

Run: venv/Scripts/python.exe research_v2/tests/test_analyze_stage1.py
"""

import copy
import sys
import tempfile
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np

import analyze_stage1 as A

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


REF = [9, 3, 4, 5, 6, 7, 8, 9, 1, 2]          # synthetic FP16 trajectory (token ids), 10 tokens, 9 routed steps


def rec(i, arm, status, ids, hybrid=False):
    n = len(ids) - 1
    r = {"index": i, "arm": arm, "status": status, "ids": ids, "answer_token_mask": [1] * 8 + [0, 0],
         "steps": n, "wall_seconds": 1.0, "gen_seconds": 0.9, "peak_mem_allocated": 1e9, "stopped_early": True,
         "accounting": {"cost_fpe_per_step": 0.5}, "boundary": {"boundary_type": "drift_marker"}}
    if hybrid:
        r.update({"tF": [1] * n, "tN": [1] * (n - 2) + [2, 2], "js": [0.01] * n})
    return r


def diverge_at(d):
    if d is None:
        return list(REF)
    ids = list(REF)
    ids[d] = 100 + d
    return ids


def build(n=60, seed=0):
    """Known construction: FP16 solves problems i%3 != 0 ... outcomes defined by simple rules so contrasts are exact."""
    rng = np.random.default_rng(seed)
    R = {a: {} for a in A.ARMS}
    for i in range(n):
        fp = "SOLVED" if i % 3 != 0 else "WRONG"
        D = {"nf4": int(rng.integers(1, 8)) if i % 2 == 0 else None,
             "cache_only": int(rng.integers(1, 8)) if i % 5 == 0 else None,
             "token_only": int(rng.integers(1, 8)) if i % 4 == 0 else None,
             "oracle_flip": int(rng.integers(1, 8)) if i % 6 == 0 else None}
        R["fp16"][i] = rec(i, "fp16", fp, diverge_at(None))
        for a, d in D.items():
            # identical history => identical outcome (must hold); diverged problems may flip outcome
            st = fp if d is None else ("WRONG" if (i + d) % 2 == 0 else "SOLVED")
            R[a][i] = rec(i, a, st, diverge_at(d), hybrid=a in ("cache_only", "token_only"))
    return R, list(range(n))


@test
def a1_first_divergence_and_hazard_arrays_hand_checked():
    assert A.first_divergence(REF, REF) is None
    assert A.first_divergence(REF, diverge_at(4)) == 4
    assert A.first_divergence(REF[:5], REF) == 5                         # strict prefix -> divergence at shorter length
    risk, event, D = A.hazard_arrays(diverge_at(4), REF, [1] * 8 + [0, 0])
    assert D == 4 and sum(risk) == 4 and sum(event) == 1                  # steps 1..4 at risk, event at step 4
    assert risk[0] == 4 and event[0] == 1                                 # all in bin 1-20
    risk, event, D = A.hazard_arrays(REF, REF, [1] * 8 + [0, 0])
    assert D is None and sum(event) == 0 and sum(risk) == 7              # steps 1..7 (mask excludes tokens 8,9); token 0 never at risk
    risk, event, D = A.hazard_arrays(diverge_at(9), REF, [1] * 8 + [0, 0])
    assert D == 9 and sum(event) == 0                                     # divergence outside the answer-region mask is not an event


@test
def a2_holm_matches_textbook_example():
    got = A.holm([0.01, 0.04, 0.03, 0.005])
    assert np.allclose(got, [0.03, 0.06, 0.06, 0.02]), got
    assert all(g >= p for g, p in zip(A.holm([0.2, 0.01, 0.5]), [0.2, 0.01, 0.5]))


@test
def a3_contrasts_identity_and_exact_values():
    R, common = build()
    rep = A.analyze(R, common, None, None)
    Y = {a: np.array([R[a][i]["status"] == "SOLVED" for i in common]).astype(int) for a in A.ARMS}
    fam = rep["primary_family"]
    names = list(fam)
    assert abs(fam[names[0]]["estimate_pp"] - 100 * (Y["cache_only"] - Y["fp16"]).mean()) < 1e-9
    assert abs(fam[names[1]]["estimate_pp"] - 100 * (Y["token_only"] - Y["fp16"]).mean()) < 1e-9
    assert abs(fam[names[2]]["estimate_pp"] - 100 * (Y["nf4"] - Y["cache_only"] - Y["token_only"] + Y["fp16"]).mean()) < 1e-9
    d = rep["decomposition_pp_description_only"]
    assert abs(d["total_gap"] - (d["cache_path"] + d["token_path"] + d["interaction"])) < 1e-9
    for v in fam.values():
        assert v["ci95_pp"][0] <= v["estimate_pp"] <= v["ci95_pp"][1] and 0 < v["p_signflip"] <= 1 and v["p_holm"] >= v["p_signflip"] - 1e-12
    assert rep["identity_check_total_equals_cache_plus_token_plus_interaction"] is True
    assert rep["accuracy"]["fp16"]["solved"] == int(Y["fp16"].sum())


@test
def a4_analysis_is_deterministic():
    R, common = build()
    a = A.analyze(R, common, None, None)
    b = A.analyze(copy.deepcopy(R), common, None, None)
    assert a["primary_family"] == b["primary_family"] and a["hazard_under_identical_history_vs_fp16"] == b["hazard_under_identical_history_vs_fp16"]


@test
def a5_divergence_counts_and_non_diverged_outcome_identity():
    R, common = build()
    rep = A.analyze(R, common, None, None)
    for a, step in (("nf4", 2), ("cache_only", 5), ("token_only", 4), ("oracle_flip", 6)):
        expect = sum(1 for i in common if i % step == 0)
        assert rep["divergence_from_fp16"][a]["n_diverged"] == expect, a
        assert rep["outcome_transitions_vs_fp16"][a]["non_diverged_with_different_outcome(must be 0)"] == 0
        assert rep["outcome_transitions_vs_fp16"][a]["non_diverged"] == len(common) - expect
    # corrupt one non-diverged problem: the anomaly must be reported, not hidden
    R2 = copy.deepcopy(R)
    i = next(i for i in common if i % 5 != 0)
    R2["cache_only"][i]["status"] = "WRONG" if R2["fp16"][i]["status"] == "SOLVED" else "SOLVED"
    assert A.analyze(R2, common, None, None)["outcome_transitions_vs_fp16"]["cache_only"]["non_diverged_with_different_outcome(must be 0)"] == 1


@test
def a6_hazard_events_equal_number_of_diverged_problems_in_the_answer_region():
    R, common = build()
    rep = A.analyze(R, common, None, None)
    for a in ("nf4", "cache_only", "token_only", "oracle_flip"):
        ev = rep["hazard_under_identical_history_vs_fp16"][a]["pooled"]["events"]
        n_div = rep["divergence_from_fp16"][a]["n_diverged"]
        assert ev == n_div, (a, ev, n_div)           # divergence is always at steps 1..7 here, all inside the answer mask (tokens 0..7)


@test
def a7_tost_equivalence_label():
    rng = np.random.default_rng(0)
    idx = rng.integers(0, 200, size=(A.B_BOOT, 200))
    Y = {"a": np.zeros(200, dtype=int), "b": np.zeros(200, dtype=int)}
    assert A.contrast(Y, {"a": 1, "b": -1}, idx, rng)["tost_pm3pp"].startswith("equivalent")
    Y = {"a": np.ones(200, dtype=int), "b": np.zeros(200, dtype=int)}
    assert A.contrast(Y, {"a": 1, "b": -1}, idx, rng)["tost_pm3pp"].startswith("differs")


@test
def a8_outputs_written_and_complete_cases_only():
    R, common = build(30)
    del R["token_only"][7]                                    # problem 7 incomplete for one arm -> excluded
    with tempfile.TemporaryDirectory() as d:
        runs = Path(d) / "eval"
        runs.mkdir()
        import json
        for a in A.ARMS:
            (runs / f"{a}.jsonl").write_text("\n".join(json.dumps(r) for r in R[a].values()) + "\n")
        R2, common2 = A.load_all(runs)
        assert 7 not in common2 and len(common2) == 29
        out = Path(d) / "out"
        A.analyze(R2, common2, out, None)
        for f in ("stage1_report.json", "paired_outcomes.csv", "divergence_survival.csv"):
            assert (out / f).exists(), f


def main():
    tests = [v for k, v in globals().items() if k.startswith("a") and k[1].isdigit() and callable(v)]
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
