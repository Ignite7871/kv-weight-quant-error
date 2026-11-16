"""CPU tests of v6b/analyze_v6b.py and v6b/recompute_v6b_independent.py on synthetic records with known shares.
python research_v2/v6b/tests/test_analyze_v6b.py"""
from __future__ import annotations

import contextlib
import io
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import analyze_v6b as A  # noqa: E402
import recompute_v6b_independent as R  # noqa: E402

IDS = json.loads((A.RV / "v6" / "v6_sample.json").read_text(encoding="utf-8"))["test_ids"][:70]
FL = {k: True for k in A.FLAGS}


def lnm_table(p, n=70, seed=0):
    """per-arm ln m_i from target contrasts: p = dict(tV, dirR1, dirR2, tN, dirU1); small per-problem noise."""
    g = np.random.default_rng(seed)
    e = lambda: g.normal(0, 0.05, n)
    c1 = -8 + g.normal(0, 0.5, n)
    nat = c1 + p["tV"] + e()
    cen = c1 + p["dirR1"] + e()
    b = nat - p["dirR2"] + e()
    cc = c1 + p["tN"] + e()
    cen0 = cc + p["dirU1"] + e()
    return {"NAT-V": nat, "C1-V": c1, "NAT-V-cen": cen, "C1-V+mu": b, "C1-V@cen": cc, "NAT-V-cen0": cen0}


def write_runs(root, table, cut1=False, dup=False, badflag=False, models=A.MODELS):
    out, v6 = root / "v6b", root / "v6"
    for m in models:
        d, d6 = out / "test" / m, v6 / "test" / m
        d.mkdir(parents=True, exist_ok=True)
        d6.mkdir(parents=True, exist_ok=True)
        for arm, x in table.items():
            rows, rows6 = [], []
            for k, i in enumerate(IDS):
                r = {"index": i, "arm": arm, "js": [0.0] + [math.exp(x[k]) * 4 / 3] * 3, "degenerate": [0, 0, 0, 0],
                     "rel_V": [0.1] * 4, "stored_rel_V": [0.1] * 4, **FL}
                if arm == "C1-V+mu":
                    r["b_norm_ratio"] = [1.0, 1.01, 0.99, 1.0]
                if badflag and arm == "NAT-V-cen0" and k == 3:
                    r["rounding_ok"] = False
                if arm in A.FROM_V6:
                    rows6.append(r)
                if cut1 and arm in A.FROM_V6 and k >= 10:
                    continue
                rows.append(r)
            if dup and arm == "C1-V@cen":
                rows.append(rows[0])
            (d / f"{arm}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            if rows6:
                (d6 / f"{arm}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows6), encoding="utf-8")
        (out / f"status_{m}.json").write_text(json.dumps({"model": m, "exit_code": 0}), encoding="utf-8")
    return out, v6


def run(table, **kw):
    with tempfile.TemporaryDirectory() as t:
        out, v6 = write_runs(Path(t), table, **kw)
        A.OUT, A.V6, R.OUT, R.V6 = out, v6, out, v6
        with contextlib.redirect_stdout(io.StringIO()):
            A.main()
            R.main()
        rep = json.loads((out / "v6b_report.json").read_text(encoding="utf-8"))
        rec = json.loads((out / "v6b_recomputation.json").read_text(encoding="utf-8"))
    return rep, rec


KEEP = {"tV": 1.0, "dirR1": 0.8, "dirR2": 0.8, "tN": -0.3, "dirU1": 0.8}


def test_keep_unrestricted_and_identities():
    rep, rec = run(lnm_table(KEEP))
    assert rep["verdict"]["H_star"] == "keep, unrestricted", rep["verdict"]
    m = rep["models"][A.MODELS[0]]
    s = {p: m["paths"][p]["share_dir"]["share"] for p in A.PATHS}
    assert abs(s["R1"] - 0.8) < 0.03 and abs(s["R2"] - 0.8) < 0.03, s
    assert abs(s["U1"] - 0.8 / 1.3) < 0.03 and abs(s["U2"] - 0.8 / 1.3) < 0.03, s
    assert abs(m["theta_N"]["estimate"] + 0.3) < 0.03 and abs(m["theta_U"]["estimate"] - 1.3) < 0.03
    assert abs(m["theta_V"]["estimate"] - m["theta_U"]["estimate"] - m["theta_N"]["estimate"]) < 1e-12
    assert abs(m["renorm_effect"]["estimate"] - 0.3) < 0.03, "renorm effect = theta(cen, cen0)"
    assert rec["outside_tolerance"] == 0 and rec["verdict"] == "keep, unrestricted", rec["disagreements"][:3]
    assert rep["verdict"]["order_dependent"] == [], "no interaction constructed"


def test_interaction_flagged():
    rep, _ = run(lnm_table({**KEEP, "dirU1": 1.0}))       # U1 direction 1.0 vs U2 direction 0.8
    assert rep["verdict"]["order_dependent"] and all(x.endswith("interaction_U") for x in rep["verdict"]["order_dependent"])


def test_keep_at_fixed_norm():
    p = {**KEEP, "tN": -0.6}                     # U1, U2 shares = 0.8 / 1.6 = 0.5: CIs straddle 0.5
    rep, rec = run(lnm_table(p))
    assert rep["verdict"]["H_star"] == "keep at fixed norm", rep["verdict"]
    assert rec["verdict"] == "keep at fixed norm"


def test_reject_single_path():
    p = {**KEEP, "dirR2": 0.3}                   # R2 share 0.3 -> upper bound < 0.5
    rep, rec = run(lnm_table(p))
    assert rep["verdict"]["H_star"] == "reject", rep["verdict"]
    assert any(x.endswith(":R2") for x in rep["verdict"]["rejecting_paths"])
    assert rec["verdict"] == "reject"
    assert rep["verdict"]["order_dependent"], "R1 0.8 vs R2 0.3 is an order interaction"


def test_undecided():
    p = {**KEEP, "dirR1": 0.5}                   # R1 share straddles 0.5, nothing below
    rep, rec = run(lnm_table(p))
    assert rep["verdict"]["H_star"] == "undecided", rep["verdict"]
    assert rec["verdict"] == "undecided"


def test_cut1_fill_from_v6():
    full, _ = run(lnm_table(KEEP))
    cut, rec = run(lnm_table(KEEP), cut1=True)
    m = A.MODELS[1]
    assert cut["models"][m]["records_from_v6"] == 2 * 60
    assert cut["models"][m]["paths"]["R1"]["share_dir"] == full["models"][m]["paths"]["R1"]["share_dir"]
    assert rec["outside_tolerance"] == 0 and rec["models"][m]["records_from_v6"] == 120


def test_duplicates_refused():
    """The locked analysis and the recomputation must each refuse duplicate records on their own."""
    with tempfile.TemporaryDirectory() as t:
        out, v6 = write_runs(Path(t), lnm_table(KEEP), dup=True)
        A.OUT, A.V6, R.OUT, R.V6 = out, v6, out, v6
        A.load_sample()
        try:
            A.per_model(A.MODELS[0])
            raise AssertionError("analyze_v6b accepted duplicate records")
        except SystemExit as e:
            assert "duplicate" in str(e), e
        try:
            R.model(A.MODELS[0], 0)
            raise AssertionError("recompute accepted duplicate records")
        except AssertionError as e:
            assert "duplicate" in str(e), e


def test_recompute_flags_ci_disagreement():
    """compare() must flag a CI endpoint off by more than 5% of the locked width, and a different verdict."""
    rep, _ = run(lnm_table(KEEP))
    with tempfile.TemporaryDirectory() as t:
        out, v6 = write_runs(Path(t), lnm_table(KEEP))
        R.OUT, R.V6 = out, v6
        Rm = {m: R.model(m, k) for k, m in enumerate(R.MODELS)}
    assert all(c["ok"] for c in R.compare(rep, Rm, rep["verdict"]["H_star"])), "clean comparison must agree"
    m, p = R.MODELS[0], "R1"
    lo, hi = Rm[m]["paths"][p]["share_dir"]["ci95"]
    Rm[m]["paths"][p]["share_dir"]["ci95"] = [lo - 0.2 * (hi - lo), hi]
    bad = [c["q"] for c in R.compare(rep, Rm, rep["verdict"]["H_star"]) if not c["ok"]]
    assert bad == [f"{m}:{p}:share:ci95"], bad
    bad = [c["q"] for c in R.compare(rep, {**Rm, m: Rm[m]}, "reject") if not c["ok"]]
    assert "verdict H*" in bad


def test_flags_counted():
    rep, rec = run(lnm_table(KEEP), badflag=True)
    assert rep["models"][A.MODELS[0]]["records_with_a_failed_flag"] == 1
    assert rec["models"][A.MODELS[0]]["records_with_a_failed_flag"] == 1


def test_status_required():
    with tempfile.TemporaryDirectory() as t:
        out, v6 = write_runs(Path(t), lnm_table(KEEP))
        (out / f"status_{A.MODELS[2]}.json").unlink()
        A.OUT, A.V6 = out, v6
        try:
            A.main()
            raise AssertionError("must refuse")
        except SystemExit as e:
            assert "refused" in str(e)


def test_threshold_and_paths():
    assert A.THRESH == 0.5
    assert A.PATHS["R1"] == (("NAT-V-cen", "C1-V"), ("NAT-V", "NAT-V-cen"), ("NAT-V", "C1-V"))
    assert A.PATHS["R2"] == (("NAT-V", "C1-V+mu"), ("C1-V+mu", "C1-V"), ("NAT-V", "C1-V"))
    assert A.PATHS["U1"] == (("NAT-V-cen0", "C1-V@cen"), ("NAT-V", "NAT-V-cen0"), ("NAT-V", "C1-V@cen"))
    assert A.PATHS["U2"] == (("NAT-V", "C1-V+mu"), ("C1-V+mu", "C1-V@cen"), ("NAT-V", "C1-V@cen"))


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


def main():
    bad = []
    for t in TESTS:
        try:
            t()
        except Exception as e:  # noqa: BLE001
            bad.append((t.__name__, repr(e)[:300]))
    print(f"test_analyze_v6b: {len(TESTS) - len(bad)}/{len(TESTS)} pass")
    for n, e in bad:
        print("  FAIL", n, e)
    return not bad


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
