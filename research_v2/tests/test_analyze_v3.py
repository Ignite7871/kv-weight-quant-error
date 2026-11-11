"""Analysis tests on simulated truth for analyze_v3.py (PROTOCOL_V3.md section 10). CPU only, no model.
  venv/Scripts/python.exe research_v2/tests/test_analyze_v3.py [--quick]
--quick uses fewer simulations (for the mutation check)."""

from __future__ import annotations

import math
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze_v3 as A  # noqa: E402

QUICK = "--quick" in sys.argv
TESTS = []


def t(fn):
    TESTS.append(fn)
    return fn


def small_B():
    A.B_BOOT, A.B_PERM = 1000, 1000


@t
def a01_type1_error_of_paired_contrast():
    small_B()
    rng = np.random.default_rng(1)
    sims = 150 if QUICK else 400
    rej_ci = rej_p = 0
    for _ in range(sims):
        d = rng.normal(0, 1, 200)
        s = A.summarize(d, rng)
        rej_ci += s["verdict"] == "difference"
        rej_p += s["p"] < 0.05
    assert 0.015 <= rej_ci / sims <= 0.10 and 0.015 <= rej_p / sims <= 0.10, (rej_ci / sims, rej_p / sims)


@t
def a02_power_and_direction():
    small_B()
    rng = np.random.default_rng(2)
    s = A.summarize(rng.normal(0.3, 1, 200), rng)
    assert s["verdict"] == "difference" and s["estimate"] > 0 and s["ci95"][0] > 0


@t
def a03_equivalence_logic():
    small_B()
    rng = np.random.default_rng(3)
    d = rng.normal(0, 0.05, 200)
    assert A.summarize(d, rng, margin=0.2, informative=True)["verdict"] == "equivalent"
    assert A.summarize(d, rng, margin=0.2, informative=False)["verdict"] == "inconclusive", "no equivalence when the dose response is flat"
    assert A.summarize(d, rng, margin=0.001, informative=True)["verdict"] == "inconclusive"
    assert A.summarize(rng.normal(0, 3, 30), rng, margin=0.2)["verdict"] == "inconclusive", "never equivalence from non-significance"


@t
def a04_holm():
    p = [0.01, 0.04, 0.03, 0.005]
    want = [0.03, 0.06, 0.06, 0.02]
    assert np.allclose(A.holm(p), want), A.holm(p)


@t
def a05_spearman_with_ties():
    assert abs(A.spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1.0) < 1e-12
    assert abs(A.spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1.0) < 1e-12
    # hand-computed: x = [1, 2, 2, 3] ranks [1, 2.5, 2.5, 4]; y = [1, 3, 2, 4] ranks [1, 3, 2, 4]
    rx, ry = np.array([1, 2.5, 2.5, 4]), np.array([1, 3, 2, 4.0])
    want = np.corrcoef(rx, ry)[0, 1]
    assert abs(A.spearman([1, 2, 2, 3], [1, 3, 2, 4]) - want) < 1e-12
    assert math.isnan(A.spearman([1, 1, 1], [1, 2, 3]))


def hc_data(rng, n, beta, gamma, mx, steps=40, noise=0.3):
    nat, c1 = {}, {}
    for i in range(n):
        x = rng.normal(mx, 1.0, steps)
        y = beta + gamma * x + rng.normal(0, noise, steps)
        jsC = np.exp(rng.normal(-6, 0.5, steps))
        pC = np.exp(rng.normal(0, 0.5, steps))
        nat[i] = {"js": [0.0] + list(jsC * np.exp(y)), "P_logit": [0.0] + list(pC * np.exp(x))}
        c1[i] = {"js": [0.0] + list(jsC), "P_logit": [0.0] + list(pC)}
    return nat, c1


@t
def a06_hc_recovers_planted_confounder_and_absence():
    small_B()
    rng = np.random.default_rng(4)
    nat, c1 = hc_data(rng, 120, beta=0.0, gamma=0.5, mx=1.0)      # contrast fully explained by P_logit
    r, _ = A.hc_model(A.hc_stats(nat, c1, range(120)), rng)
    assert r["applicable"] and r["band"] == "substantial" and r["fraction"] > 0.8, r
    nat, c1 = hc_data(rng, 120, beta=0.5, gamma=0.5, mx=0.0)      # P_logit unrelated to the contrast's mean
    r, _ = A.hc_model(A.hc_stats(nat, c1, range(120)), rng)
    assert r["applicable"] and r["band"] == "no", r
    nat, c1 = hc_data(rng, 120, beta=0.0, gamma=0.5, mx=0.0)      # no contrast at all
    r, _ = A.hc_model(A.hc_stats(nat, c1, range(120)), rng)
    assert not r["applicable"] and r["band"] == "not applicable", r
    nat, c1 = hc_data(rng, 120, beta=0.6, gamma=0.5, mx=0.5, noise=0.2)   # partially explained: beta0 = 0.85, beta1 = 0.6, fraction ~ 0.29
    r, _ = A.hc_model(A.hc_stats(nat, c1, range(120)), rng)
    assert r["band"] == "partial" and 0.2 < r["fraction"] < 0.4, r


@t
def a07_hc_betas_exact():
    rng = np.random.default_rng(5)
    x = rng.normal(1, 1, 500)
    y = 0.3 + 0.7 * x
    S = np.array([[len(x), y.sum(), x.sum(), (x * y).sum(), (x * x).sum()]])
    b0, b1 = A.hc_betas(S.sum(0))
    assert abs(b0 - y.mean()) < 1e-12 and abs(b1 - 0.3) < 1e-10


@t
def a08_hc_cross_decision_and_nonreplication_finding():
    small_B()
    rng = np.random.default_rng(6)
    na, ca = hc_data(rng, 150, 0.0, 0.5, 1.0)
    nb, cb = hc_data(rng, 150, 0.0, 0.5, 0.0)
    r = A.hc_cross(A.hc_stats(na, ca, range(150)), A.hc_stats(nb, cb, range(150)), rng)
    assert r["precondition_met"] and r["band"] == "substantial", r
    nb2, cb2 = hc_data(rng, 150, 0.0, 0.5, 1.0)
    r = A.hc_cross(A.hc_stats(na, ca, range(150)), A.hc_stats(nb2, cb2, range(150)), rng)
    assert not r["precondition_met"] and "did not replicate" in r.get("finding", ""), r


def fake_records(rng, n, theta_nat_c1=0.0, ha_signal=True, he=0.3, hd=0.0, hb=0.2, sd=0.4, steps=30):
    arms = ["NAT", "C1", "C2", "QA", "QO", "CV", "NAT-K", "NAT-V", "NAT-K*", "NAT-V*", "C1-K", "C1-V", "LOC", "INH"]
    base = {i: rng.normal(-6, 0.5) for i in range(n)}
    shift = {"NAT": theta_nat_c1, "QA": hb / 2, "QO": -hb / 2, "NAT-K*": hd / 2, "NAT-V*": -hd / 2, "INH": he / 2, "LOC": -he / 2}
    data = {}
    for a in arms:
        data[a] = {}
        for i in range(n):
            lvl = base[i] + shift.get(a, 0.0) + rng.normal(0, sd / math.sqrt(2))
            pl = np.exp(rng.normal(0, 1, steps))
            pn = np.exp(rng.normal(0, 1, steps))
            js = np.exp(lvl + (0.8 * np.log(pl) if ha_signal else 0.8 * np.log(pn)) + rng.normal(0, 0.3, steps))
            js[0] = 0.0
            data[a][i] = {"index": i, "S": steps, "js": list(js), "P_logit": [0.0] + list(pl[1:]), "P_norm": [0.0] + list(pn[1:]),
                          "P_val": [0.0] + list(np.exp(rng.normal(0, 1, steps - 1))), "P_anormK": [0.0] + list(np.exp(rng.normal(0, 1, steps - 1))),
                          "P_out": [float("nan")] + list(rng.random(steps - 1)), "dn2_K": [1.0] * steps, "dn2_V": [1.0] * steps,
                          "rel_K": [0.01] * steps, "rel_V": [0.02] * steps, **fake_rdev(rng, a, steps)}
    return data


NATURAL_ARMS = ("NAT", "NAT-K", "NAT-V", "LOC", "INH")


def fake_rdev(rng, arm, steps, n_slots=8, p_flag=0.0, vals=None):
    """Amendment-2 record fields as written by rv3.core: per step the largest realized deviation, its slot and the count above 5 %; per record
    the histogram (bins of A.RDEV_EDGES), the sum and the count. Natural arms have no constructed slots."""
    nb = len(A.RDEV_EDGES) + 1
    if arm in NATURAL_ARMS:
        return {"rdev_max": [0.0] * steps, "rdev_n_gt5": [0] * steps, "rdev_arg": [None] * steps, "rdev_hist": [0] * nb, "rdev_sum": 0.0, "rdev_count": 0}
    v = np.exp(rng.normal(-7, 1.5, (steps, n_slots))) if vals is None else vals
    flag = rng.random(steps) < p_flag
    v[flag, 0] = 0.06 + rng.random(int(flag.sum()))
    hist = np.bincount(np.searchsorted(A.RDEV_EDGES, v.ravel(), side="left"), minlength=nb)
    return {"rdev_max": v.max(1).tolist(), "rdev_n_gt5": (v > A.RDEV_FLAG).sum(1).tolist(),
            "rdev_arg": [[0, "K", int(j % 2)] for j in v.argmax(1)], "rdev_hist": hist.tolist(), "rdev_sum": float(v.sum()), "rdev_count": int(v.size)}


def fake_cal(desc=False):
    return {"margin_log_js": 0.2, "dose_response_informative": True, "HD_descriptive": desc, "E8": {}, "r80": {}, "rho_K": 0.01, "rho_V": 0.02,
            "rho_star": 0.014, "scale_factor_K": 1.4, "scale_factor_V": 0.7}


@t
def a09_full_pipeline_planted_truth():
    small_B()
    rng = np.random.default_rng(7)
    n = 60 if QUICK else 100
    data = {"qwen-base": fake_records(rng, n, theta_nat_c1=-0.4), "llama-instruct": fake_records(rng, n, theta_nat_c1=0.4, ha_signal=False),
            "qwen-instruct": fake_records(rng, n, theta_nat_c1=-0.4), "llama-base": fake_records(rng, n, theta_nat_c1=0.4)}
    cals = {m: fake_cal(desc=(m == "llama-base")) for m in data}
    rep = A.analyze_all(data, cals)
    pm = rep["per_model"]
    assert pm["qwen-base"]["HA"]["supported"] and not pm["llama-instruct"]["HA"]["supported"], "H-A per model"
    assert pm["llama-instruct"]["HA"]["falsified"], "planted reversed predictor must falsify H-A"
    assert pm["qwen-base"]["HE_reading"].startswith("INH > LOC"), pm["qwen-base"]["HE_reading"]
    assert "HD K*-V*" not in pm["llama-base"]["family"] and "HD_descriptive" in pm["llama-base"], "H-D rule"
    assert len(pm["qwen-base"]["family"]) in (6, 7)
    X = rep["HF"]["contrasts"]
    assert X["family (Llama - Qwen)"]["verdict"] == "difference" and X["family (Llama - Qwen)"]["estimate"] > 0.5, X["family (Llama - Qwen)"]
    assert X["tuning (instruct - base)"]["verdict"] != "difference" or abs(X["tuning (instruct - base)"]["estimate"]) < 0.2
    assert rep["HF"]["readings"]["family account"] == "supported"
    assert rep["HC_cross"]["qwen-base vs llama-instruct"]["role"] == "decision rule"


@t
def a10_sd_guard():
    small_B()
    rng = np.random.default_rng(8)
    n = 50
    data = {m: fake_records(rng, n, sd=0.4 if m != "qwen-instruct" else 1.5) for m in A.ORDER}
    rep = A.analyze_all(data, {m: fake_cal() for m in data})
    assert rep["HF"]["label"].startswith("descriptive"), rep["HF"]["label"]


@t
def a11_hb_falsification_flag():
    small_B()
    rng = np.random.default_rng(9)
    data = {"qwen-base": fake_records(rng, 80, hb=-0.6)}
    rep = A.analyze_all(data, {"qwen-base": fake_cal()})
    assert rep["per_model"]["qwen-base"]["HB_falsified"], "QA < QO must falsify H-B"
    assert rep["HF"]["label"].startswith("not estimable")


@t
def a12_duplicate_record_fails_loudly():
    import json
    import tempfile
    p = Path(tempfile.mkdtemp()) / "NAT.jsonl"
    p.write_text(json.dumps({"index": 3, "js": [0.0]}) + chr(10) + json.dumps({"index": 3, "js": [0.0]}) + chr(10), encoding="utf-8")
    try:
        A.load_jsonl(p)
    except ValueError:
        return
    raise AssertionError("a duplicate (model, arm, problem) record must raise")


@t
def a13_rounding_report_and_sensitivity():
    """Amendment 2: the realized-deviation report equals a direct computation; the sensitivity keeps exactly the steps no flagged write reaches
    (1..f for the first flagged write f; min over the arms of a contrast), drops problems with < 8 retained steps, and leaves NAT untouched."""
    small_B()
    rng = np.random.default_rng(11)
    n, steps = 40, 30
    data = fake_records(rng, n, steps=steps)
    first = {}
    for i in range(n):                                            # plant: C1 flags at step f_i (or never), QA never
        f = int(rng.integers(1, steps + 1)) if i % 3 else None
        r = data["C1"][i]
        r["rdev_n_gt5"] = [0] * steps
        if f is not None:
            r["rdev_n_gt5"][f - 1] = 1
        first[i] = f if f is not None else steps
    V_ = {i: np.exp(rng.normal(-7, 1.5, (steps, 8))) for i in range(n)}
    for i in range(n):
        keep = data["C1"][i]["rdev_n_gt5"]
        data["C1"][i].update(fake_rdev(rng, "C1", steps, vals=V_[i]))
        data["C1"][i]["rdev_n_gt5"] = keep
    vals = np.concatenate([V_[i].ravel() for i in range(n)])
    rep = A.rdev_report(data, list(range(n)))
    c = rep["C1"]
    assert c["n_slots"] == vals.size and abs(c["mean"] - vals.mean()) < 1e-12 and abs(c["max"] - vals.max()) < 1e-15, c
    p99 = np.quantile(vals, 0.99, method="inverted_cdf")
    k = int(np.searchsorted(A.RDEV_EDGES, p99, side="left"))
    assert c["p99_at_most"] == A.RDEV_EDGES[k] and A.RDEV_EDGES[k - 1] < p99 <= A.RDEV_EDGES[k], (c["p99_at_most"], p99)
    assert rep["NAT"]["n_slots"] == 0 and rep["LOC"]["n_slots"] == 0, "natural arms have no constructed slots"
    Ad, kept, dropped, _ = A.sens_data(data, ("NAT", "C1"), list(range(n)))
    exp_kept = [i for i in range(n) if first[i] >= A.SENS_MIN_STEPS]
    assert kept == exp_kept and dropped == n - len(exp_kept), (kept, exp_kept)
    for i in kept:
        assert Ad["C1"][i]["js"] == data["C1"][i]["js"][:first[i]] and Ad["NAT"][i]["js"] == data["NAT"][i]["js"][:first[i]], i
        assert Ad["C1"][i]["S"] == first[i] and len(Ad["NAT"][i]["P_logit"]) == first[i]
    An, kn, dn, _ = A.sens_data(data, ("NAT",), list(range(n)))
    assert kn == list(range(n)) and dn == 0 and all(An["NAT"][i]["js"] == data["NAT"][i]["js"] for i in kn), "NAT alone must be untouched"
    out = A.rounding_sensitivity(data, fake_cal(), list(range(n)))
    fq = {i: min(next((s + 1 for s, x in enumerate(data[a][i]["rdev_n_gt5"]) if x > 0), steps) for a in ("QA", "QO")) for i in range(n)}
    assert out["label"].startswith("EXPLORATORY") and out["contrasts"]["HB QA-QO"]["n_dropped"] == sum(fq[i] < A.SENS_MIN_STEPS for i in range(n))
    assert out["contrasts"]["NAT-C1"]["n_kept"] == len(exp_kept)


def run_all():
    fails = []
    for fn in TESTS:
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            fails.append((fn.__name__, f"{type(e).__name__}: {e}"))
    return fails


def main():
    fails = run_all()
    for fn in TESTS:
        print(("FAIL " if fn.__name__ in dict(fails) else "PASS ") + fn.__name__)
    for f in fails:
        print(*f)
    print(f"{len(TESTS) - len(fails)}/{len(TESTS)} passed")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
