"""Tests of analyze_controls.py on simulated data with known truth (CPU). Run: venv/Scripts/python.exe research_v2/tests/test_analyze_controls.py"""

import math
import sys
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import analyze_controls as A

RES = []


def test(fn):
    try:
        fn()
        RES.append((fn.__name__, "PASS", ""))
    except Exception as e:  # noqa: BLE001
        RES.append((fn.__name__, "FAIL", f"{type(e).__name__}: {e}"))
        traceback.print_exc()
    return fn


def sim_records(n, S, base, shifts, seed, noise=0.4, alpha_exp=None):
    """records for arms in `shifts` (name -> log-effect shift or (alpha, ...)); per-problem log level ~ N(log base, noise) shared across arms (paired)."""
    rng = np.random.default_rng(seed)
    lvl = rng.normal(math.log(base), noise, n)
    out = {}
    for arm, sh in shifts.items():
        recs = {}
        for i in range(n):
            js = np.exp(lvl[i] + sh + rng.normal(0, 0.15, S))
            recs[i] = {"index": i, "js": js.tolist(), "flip": (rng.random(S) < 0.05).astype(float).tolist(),
                       "norm_dev_mean": [0.001] * S, "norm_dev_max": [0.01] * S, "rel_scale_mean": [0.15] * S,
                       "target_norm_mean": (1.0 * np.exp(rng.normal(0, 0.25, S))).tolist()}
        out[arm] = recs
    return out


@test
def t01_holm_matches_manual():
    p = [0.0274, 0.0353, 0.8391, 0.1292, 0.6846]
    assert np.allclose(A.holm(p), [0.137, 0.1412, 1.0, 0.3876, 1.0], atol=1e-4)


@test
def t02_estimate_recovers_known_effect_and_ci_covers():
    tierA = sim_records(200, 40, 1e-3, {"NAT": 0.30, "C1_a1": 0.0, "C2_a1": 0.10}, 1)
    calib = {"margin_tierA_log_js": 0.2, "dose_response_informative": True, "tierB_relative_margin_c": 0.2}
    rep = A.analyze(tierA, {}, calib, None, B=2000)
    p = rep["tierA"]["primary_log_JS_NAT_minus_C1"]
    assert abs(p["estimate"] - 0.30) < 0.03, p["estimate"]
    assert p["ci95"][0] < 0.30 < p["ci95"][1] and p["verdict"] == "difference"
    s = rep["tierA"]["secondary"]
    assert abs(s["A: NAT - C2 (log JS)"]["estimate"] - 0.20) < 0.03 and abs(s["A: C2 - C1 (log JS)"]["estimate"] - 0.10) < 0.03


@test
def t03_null_has_nominal_type_I_error_and_equivalence_verdict():
    calib = {"margin_tierA_log_js": 0.15, "dose_response_informative": True, "tierB_relative_margin_c": 0.2}
    rej = 0
    eq = 0
    N = 150
    for s in range(N):
        tierA = sim_records(60, 30, 1e-3, {"NAT": 0.0, "C1_a1": 0.0, "C2_a1": 0.0}, 100 + s)
        r = A.analyze(tierA, {}, calib, None, B=400)["tierA"]["primary_log_JS_NAT_minus_C1"]
        rej += r["verdict"] == "difference"
        eq += r["verdict"] == "equivalent"
    assert rej / N < 0.12, f"type I error {rej / N:.3f}"
    assert eq / N > 0.5, f"equivalence under a true null declared only {eq / N:.2f} of the time"


@test
def t04_equivalence_not_claimable_when_dose_response_flat():
    calib = {"margin_tierA_log_js": None, "dose_response_informative": False, "tierB_relative_margin_c": 0.2}
    tierA = sim_records(120, 30, 1e-3, {"NAT": 0.0, "C1_a1": 0.0, "C2_a1": 0.0}, 5)
    v = A.analyze(tierA, {}, calib, None, B=500)["tierA"]["primary_log_JS_NAT_minus_C1"]["verdict"]
    assert v == "inconclusive", v


@test
def t05_dose_slope_and_calibration_recover_truth():
    b, k, n, S = 0.8, 2e-3, 100, 30
    shifts = {"NAT": math.log(k * 1.6 ** b / k)}                       # NAT effect equals C1 at alpha = 1.6
    calib = sim_records(n, S, k, {"NAT": shifts["NAT"]}, 7)
    for a in A.ALPHA_GRID:
        calib[f"C1_a{a:g}"] = sim_records(n, S, k, {"x": b * math.log(a)}, 7)["x"]
    for a in (0.5, 1.0, 2.0):
        pass
    res = A.calibrate(calib)
    assert abs(res["alpha_star"] - 1.6) < 0.12, res["alpha_star"]
    assert abs(res["dose_slope_b_hat"] - b) < 0.05, res["dose_slope_b_hat"]
    assert res["dose_response_informative"] and abs(res["margin_tierA_log_js"] - b * math.log(1 + res["c_used"])) < 1e-9
    assert A.C_LO <= res["c_used"] <= A.C_HI


@test
def t06_calibration_clamps_and_flags():
    n, S = 40, 20
    calib = sim_records(n, S, 1e-3, {"NAT": 3.0}, 8)                    # NAT effect far above every C1 dose
    for a in A.ALPHA_GRID:
        calib[f"C1_a{a:g}"] = sim_records(n, S, 1e-3, {"x": 0.8 * math.log(a)}, 8)["x"]
    res = A.calibrate(calib)
    assert res["alpha_star"] == A.ALPHA_GRID[-1] and res["alpha_star_flag"]


@test
def t07_first_divergence_and_restriction():
    assert A.first_div([1, 2, 3], [1, 2, 4]) == 2 and A.first_div([1, 2], [1, 2]) is None and A.first_div([1, 2, 3], [1, 2]) == 2
    assert A.restricted_first_div([1, 2], [1, 2], 64) == 64.0 and A.restricted_first_div([5, 2, 3], [1, 2, 3], 64) == 0.0
    assert A.restricted_first_div(list(range(100)), list(range(70)) + [999] * 30, 64) == 64.0


@test
def t08_tierB_contrast_matches_manual_computation():
    rng = np.random.default_rng(3)
    n = 120
    ref = {i: list(rng.integers(0, 50, 80)) for i in range(n)}
    def make(shift):
        out = {}
        for i in range(n):
            ids = list(ref[i])
            k = int(min(79, max(1, rng.integers(5, 40) + shift)))
            ids[k] = ids[k] + 1
            out[i] = {"index": i, "ids": ids, "status": "SOLVED" if rng.random() < 0.3 else "FAILED"}
        return out
    tb = {"FP16": {i: {"index": i, "ids": ref[i], "status": "SOLVED"} for i in range(n)}, "NAT": make(0), "C1": make(8)}
    rep = A.analyze({}, tb, {"tierB_relative_margin_c": 0.2}, None, B=500)
    est = rep["tierB"]["primary_restricted_first_divergence_NAT_minus_C1"]["estimate"]
    man = np.mean([A.restricted_first_div(tb["NAT"][i]["ids"], ref[i]) - A.restricted_first_div(tb["C1"][i]["ids"], ref[i]) for i in range(n)])
    assert abs(est - man) < 1e-9 and est < -4


@test
def t09_bootstrap_ci_agrees_with_independent_normal_approximation():
    rng = np.random.default_rng(11)
    d = rng.normal(0.1, 0.5, 300)
    s = A.summarize(d, np.random.default_rng(1), None, 5000)
    half = 1.96 * d.std(ddof=1) / math.sqrt(len(d))
    assert abs(s["ci95"][0] - (d.mean() - half)) < 0.01 and abs(s["ci95"][1] - (d.mean() + half)) < 0.01
    assert abs(s["p_signflip"] - math.erfc(abs(d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))) / math.sqrt(2))) < 0.02


@test
def t10_floor_prevents_log_of_zero():
    rec = {"js": [0.0] * 10}
    assert A.per_problem_mean_js(rec) == A.FLOOR


if __name__ == "__main__":
    w = max(len(n) for n, _, _ in RES)
    for n, s, m in RES:
        print(f"{n:<{w}}  {s}  {m}")
    bad = [r for r in RES if r[1] != "PASS"]
    print(f"\n{len(RES) - len(bad)}/{len(RES)} passed")
    sys.exit(1 if bad else 0)
