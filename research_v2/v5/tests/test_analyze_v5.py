"""Analysis tests on simulated truth (PROTOCOL_V5.md sections 4, 7, 8, 11). A simulator writes v5-format records (six arms, X13a slot sums,
calibration files) with KNOWN effects for the seven units; each scenario checks the readings the protocol prescribes: the A, B and D families with
Holm within each, the A-X classification (head-dimension pattern / size pattern / unresolved, on both sides of each boundary) and the joint reading,
a cut unit leaving a one-model family, unresolved units, the refusal before every unit ends, the anchor criteria, and agreement with the independent
recomputation. The v4 reference report is synthetic here (the locked values of the five source models); the real report is checked by
real_model_cpu_v5.py on the laptop.

  venv/Scripts/python.exe research_v2/v5/tests/test_analyze_v5.py [--write <json>] [--only <name>]
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
V5 = HERE.parent
RV = V5.parent
sys.path.insert(0, str(RV))
sys.path.insert(0, str(V5))

import numpy as np  # noqa: E402

import analyze_v5 as A  # noqa: E402
import recompute_v5_independent as RC  # noqa: E402

TESTS, RESULTS = [], []
S = 10
FLAGS = ("exact_ok", "natural_ok", "routing_reset", "construction_ok", "rounding_ok", "finite_ok")
V4_VALUES = {"qwen2.5-0.5b": (-0.2233, -1.2901), "qwen2.5-1.5b": (0.2036, -0.9840), "qwen2.5-3b": (0.3005, -0.8513), "llama3.2-1b": (0.0230, -0.9461),
             "llama3.2-3b": (0.4228, -0.4350)}
TRUTH = {"H1": -1.0, "H2": 0.8, "H3": -0.9, "NAT-C1": 0.0}
HD_PRED = {"U1": {"NAT-C1": 0.2036, "H1": -0.9840}, "U2": {"NAT-C1": (-0.2233 + 0.0230) / 2, "H1": (-1.2901 - 0.9461) / 2}}
SIZE_PRED = {"U1": {"NAT-C1": -0.2233, "H1": -1.2901}, "U2": {"NAT-C1": (0.2036 + 0.3005 + 0.4228) / 3, "H1": (-0.9840 - 0.8513 - 0.4350) / 3}}


def t(fn):
    TESTS.append(fn)
    return fn


def synthetic_v4_report():
    return {"per_model": {m: {"contrasts": {"NAT-C1": {"estimate": a}, "H1": {"estimate": b}}} for m, (a, b) in V4_VALUES.items()}}


def write_unit(out: Path, u: str, eff=None, sd=0.4, n=100, cm=True, hd=64, L_=2, nkv=2, seed=0, margin=0.21, informative=True, complete=True):
    eff = {**TRUTH, **(eff or {})}
    rng = np.random.default_rng(seed)
    d = out / u / "test"
    (d / "slots").mkdir(parents=True, exist_ok=True)
    recs = {a: [] for a in A.ARMS}
    noise = rng.normal(0, sd, (6, n))
    noise[1:] -= noise[1:].mean(axis=1, keepdims=True)
    for i in range(n):
        base = rng.normal(-6, 0.3)
        lNV = base + noise[0, i]
        lNK = lNV + eff["H3"] + noise[1, i]
        lCK = lNK - eff["H1"] + noise[2, i]
        lCV = lNV - eff["H2"] + noise[3, i]
        lN = base + 0.5 + noise[4, i]
        lC = lN - eff["NAT-C1"] + noise[5, i]
        for a, l in (("NAT", lN), ("C1", lC), ("NAT-K", lNK), ("C1-K", lCK), ("NAT-V", lNV), ("C1-V", lCV)):
            recs[a].append({"index": i, "arm": a, "S": S, "js": [0.0] + [math.exp(l)] * (S - 1), "rdev_n_gt5": [0] * S, "rel_K": [0.05] * S,
                            "rel_V": [0.2] * S, "unit": u, **{k: True for k in FLAGS}})
        mu = rng.normal(0, 1, (L_, nkv, hd)) if cm else np.zeros((L_, nkv, hd))
        dK = mu[None] + rng.normal(0, 0.3 if cm else 1.0, (S, L_, nkv, hd))
        dV = mu[None] + rng.normal(0, 0.3 if cm else 1.0, (S, L_, nkv, hd))
        np.savez_compressed(d / "slots" / f"NAT_{i}.npz", normsK=np.linalg.norm(dK, axis=-1), normsV=np.linalg.norm(dV, axis=-1), S1K=dK.sum(0), S1V=dV.sum(0))
    for a, rs in recs.items():
        (d / f"{a}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rs))
    (out / u / "calibration.json").write_text(json.dumps({"margin_log_js": margin, "dose_response_informative": informative, "rho_K": 0.06, "rho_V": 0.2,
                                                          "rho_V_over_rho_K": 0.2 / 0.06, "dose_slope_b_hat": 2.2}))
    if complete:
        (out / u / "test_complete.json").write_text("{}")


@contextlib.contextmanager
def study(overrides=None, not_run=(), design=None, hd=None):
    """overrides: unit -> kwargs for write_unit. The A units default to the head-dimension prediction on both A-X contrasts."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        out = Path(td) / "v5_runs"
        out.mkdir()
        saved = (A.OUT, A.v4_report, A.manifest)
        A.OUT = out
        A.v4_report = synthetic_v4_report
        man = A.manifest()
        A.manifest = lambda: {m: {**e, "geometry": {**e["geometry"], "head_dim": 16}} for m, e in man.items()}
        try:
            for k, u in enumerate(A.UNITS):
                if u in not_run:
                    (out / u).mkdir()
                    (out / u / "not_run.json").write_text("{}")
                    continue
                kw = {"seed": k, "hd": 16}
                if u in ("U1", "U2"):
                    kw["eff"] = {"NAT-C1": HD_PRED[u]["NAT-C1"], "H1": HD_PRED[u]["H1"]}
                kw.update((overrides or {}).get(u, {}))
                if "eff" in (overrides or {}).get(u, {}) and u in ("U1", "U2"):
                    kw["eff"] = {"NAT-C1": HD_PRED[u]["NAT-C1"], "H1": HD_PRED[u]["H1"], **overrides[u]["eff"]}
                write_unit(out, u, **kw)
            if design:
                (out / "design.json").write_text(json.dumps(design))
            yield out
        finally:
            A.OUT, A.v4_report, A.manifest = saved


def run_analyze():
    with contextlib.redirect_stdout(io.StringIO()):
        return A.analyze()


def refused(fn):
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            fn()
    except SystemExit as e:
        return str(e)
    return None


# ----------------------------------------------------------------------------- A-X units
@t
def a01_ax_references_and_sources_are_the_protocol_table():
    r = A.ax_references(synthetic_v4_report())
    assert abs(r["U1"]["NAT-C1"]["boundary"] - (-0.2233 + 0.2036) / 2) < 1e-12 and r["U1"]["NAT-C1"]["hd_side"] == 1
    assert abs(r["U2"]["NAT-C1"]["size_pred"] - (0.2036 + 0.3005 + 0.4228) / 3) < 1e-12 and r["U2"]["NAT-C1"]["hd_side"] == -1
    assert abs(r["U2"]["H1"]["boundary"] - (-0.9374)) < 1e-3 and abs(r["U1"]["H1"]["boundary"] - (-1.13705)) < 1e-9
    assert A.AX_SOURCES["U2"]["NAT-C1"] == {"size": ["qwen2.5-1.5b", "qwen2.5-3b", "llama3.2-3b"], "hd": ["qwen2.5-0.5b", "llama3.2-1b"]}
    assert A.AX_SOURCES["U1"]["H1"] == {"size": ["qwen2.5-0.5b"], "hd": ["qwen2.5-1.5b"]}
    with tempfile.TemporaryDirectory() as td:
        saved = A.V4_RUNS
        A.V4_RUNS = Path(td)
        (Path(td) / "v4_report.json").write_text(json.dumps(synthetic_v4_report()))
        try:
            A.v4_report()
            raise AssertionError("a report that is not the locked v4 report must be refused")
        except ValueError:
            pass
        finally:
            A.V4_RUNS = saved


@t
def a02_ax_classify_and_one_sided_p():
    ref = {"boundary": 0.0, "hd_side": 1}
    assert A.ax_classify({"ci95": [0.1, 0.3]}, 0.01, ref) == "head-dimension pattern"
    assert A.ax_classify({"ci95": [-0.3, -0.1]}, 0.01, ref) == "size pattern"
    assert A.ax_classify({"ci95": [-0.1, 0.3]}, 0.01, ref) == "unresolved"
    assert A.ax_classify({"ci95": [0.1, 0.3]}, 0.06, ref) == "unresolved", "the CI alone is not enough: Holm p < 0.05 too"
    assert A.ax_classify({"ci95": [0.1, 0.3]}, 0.01, {"boundary": 0.0, "hd_side": -1}) == "size pattern"
    rng = np.random.default_rng(1)
    d = rng.normal(0.3, 0.5, 100)
    p, s = A.one_sided_flip_p(d, ("x",))
    assert s == 1 and p < 0.001
    p2, s2 = A.one_sided_flip_p(-d, ("x",))
    assert s2 == -1 and p2 < 0.001
    p3, _ = A.one_sided_flip_p(rng.normal(0, 0.5, 100), ("y",))
    assert p3 > 0.01


@t
def a03_truth_at_head_dim_everywhere():
    with study():
        rep = run_analyze()
        fam = rep["all_units"]
        for k in A.FAMILIES:
            assert fam[k]["result"] == "supported in every model", (k, fam[k])
        ax = fam["A-X"]
        for c in ("NAT-C1", "H1"):
            for u in ("U1", "U2"):
                assert ax[c]["units"][u]["classification"] == "head-dimension pattern", (c, u, ax[c]["units"][u])
        assert ax["joint_reading_NAT-C1"]["reading"] == "in these checkpoints the v4 size trend tracks head dimension"
        assert "family offset" in ax["NAT-C1"]["units"]["U2"]["note"]
        assert set(rep["per_unit"]) == set(A.UNITS) and "history source" in rep["notes"]["D"]


@t
def a04_truth_at_size_and_mixed_and_between():
    sz = {u: {"eff": {"NAT-C1": SIZE_PRED[u]["NAT-C1"], "H1": SIZE_PRED[u]["H1"]}} for u in ("U1", "U2")}
    with study(sz):
        ax = run_analyze()["all_units"]["A-X"]
        assert all(ax["NAT-C1"]["units"][u]["classification"] == "size pattern" for u in ("U1", "U2"))
        assert ax["joint_reading_NAT-C1"]["reading"] == "in these checkpoints the v4 size trend tracks size"
    with study({"U2": sz["U2"]}):
        ax = run_analyze()["all_units"]["A-X"]
        assert ax["NAT-C1"]["units"]["U1"]["classification"] == "head-dimension pattern"
        assert ax["NAT-C1"]["units"]["U2"]["classification"] == "size pattern"
        assert ax["joint_reading_NAT-C1"]["reading"] == "neither factor alone accounts for it in these checkpoints"
    b = A.ax_references(synthetic_v4_report())["U1"]["NAT-C1"]["boundary"]
    with study({"U1": {"eff": {"NAT-C1": b + 0.01}}}):
        ax = run_analyze()["all_units"]["A-X"]
        assert ax["NAT-C1"]["units"]["U1"]["classification"] == "unresolved"
        assert ax["joint_reading_NAT-C1"]["reading"] == "neither factor alone accounts for it in these checkpoints"


@t
def a05_families_falsified_and_inconclusive():
    with study({"U5": {"eff": {"H2": -0.6}}}):
        fam = run_analyze()["all_units"]
        assert fam["D-H2"]["readings"]["U5"] == "falsified" and fam["D-H2"]["result"] == "falsified"
        assert fam["B-H2"]["result"] == "supported in every model", "other families unaffected"
    with study({"U4": {"eff": {"H2": 0.0}, "sd": 0.2, "margin": 0.3}}):
        fam = run_analyze()["all_units"]
        assert fam["B-H2"]["readings"]["U4"] == "falsified", "equivalent to 0 falsifies"
    with study({"U4": {"eff": {"H2": 0.0}, "sd": 0.2, "margin": 0.3, "informative": False}}):
        assert run_analyze()["all_units"]["B-H2"]["readings"]["U4"] == "inconclusive", "equivalence needs an informative dose response"
    with study({"U3": {"cm": False}}):
        fam = run_analyze()["all_units"]
        assert fam["D-H5"]["readings"]["U3/K"] != "supported" and fam["D-H5"]["result"] != "supported in every model"


@t
def a06_holm_within_each_family():
    with study():
        rep = run_analyze()
        assert set(rep["all_units"]["A-H1"]["holm_p"]) == {"U1", "U2"} and set(rep["all_units"]["B-H3"]["holm_p"]) == {"U4", "U6"}
        assert set(rep["all_units"]["D-H5"]["holm_p"]) == {"U3/K", "U3/V", "U5/K", "U5/V"}
        assert "U3" not in rep["all_units"]["B-H2"]["units"] and "U0" not in json.dumps({k: v["units"] for k, v in rep["all_units"].items() if k != "A-X"})


@t
def a07_cut_unit_leaves_one_model_family():
    with study(not_run=("U6",), design={"n_test": {"U3": 100, "U4": 100, "U5": 100}, "cut_units": ["U6"]}):
        rep = run_analyze()
        assert rep["status"]["U6"] == "not run" and rep["all_units"]["B-H2"]["units"] == ["U4"] and rep["all_units"]["B-H2"]["n_models"] == 1
        assert rep["all_units"]["A-H1"]["n_models"] == 2


@t
def a08_r1_n60_and_unresolved_units():
    with study({u: {"n": 60} for u in ("U3", "U4", "U5", "U6")}, design={"n_test": {u: 60 for u in ("U3", "U4", "U5", "U6")}}):
        rep = run_analyze()
        assert all(rep["status"][u] == "complete" for u in ("U3", "U4", "U5", "U6")) and rep["per_unit"]["U3"]["n_problems"] == 60
    with study({"U2": {"n": 40, "complete": False}}) as out:
        assert "Refusing" in refused(A.analyze)
        (out / "U2" / "drift_stop.json").write_text("{}")
        rep = run_analyze()
        assert rep["status"]["U2"] == "unresolved" and "U2" not in rep["per_unit"] and rep["all_units"]["A-H1"]["units"] == ["U1"]
        assert rep["all_units"]["A-X"]["joint_reading_NAT-C1"]["reading"] == "neither factor alone accounts for it in these checkpoints"
    with study({"U1": {"n": 90}}):
        rep = run_analyze()
        assert rep["status"]["U1"] == "unresolved", "complete but below the design n"


@t
def a09_flagged_with_and_without():
    with study() as out:
        (out / "U5" / "precision.json").write_text(json.dumps({"precision": "bf16"}))
        rep = run_analyze()
        assert rep["flagged_bf16"] == ["U5"] and rep["without_flagged"]["D-H2"]["units"] == ["U3"] and rep["abstract_level"]["source"] == "without_flagged"


@t
def a10_anchor_criteria():
    with study() as out:
        v4 = out.parent / "v4_runs"
        shutil.copytree(out / "U0", v4 / "qwen2.5-0.5b")
        for p in (v4 / "qwen2.5-0.5b" / "test").glob("*.jsonl"):
            p.write_text("".join(json.dumps({k: v for k, v in json.loads(l).items() if k != "unit"}) + "\n" for l in p.read_text().splitlines()))
        hist = "".join(json.dumps({"index": i, "ids": [1, 2, i]}) + "\n" for i in range(100))
        for d in (out / "U0" / "test", v4 / "qwen2.5-0.5b" / "test"):
            (d / "FP16_history.jsonl").write_text(hist)
        saved = A.V4_RUNS
        A.V4_RUNS = v4
        try:
            Au = A.load_test("U0")
            idx = list(range(100))
            cal = json.loads((out / "U0" / "calibration.json").read_text())
            rep4 = {"contrasts": {h: A.summarize(A.L.theta(Au, a, b, idx), ("v4", h), cal["margin_log_js"]) for h, (a, b) in A.CONTRASTS.items()},
                    "H5": {"K": {"estimate": 2.5}, "V": {"estimate": 2.4}}}
            x = A.x13a("U0", idx, 16)
            r = A.anchor_core(Au, cal, cal, rep4, idx, x)
            assert r["pass"] and all(abs(r["contrasts"][h]["delta"]["estimate"]) < 1e-12 for h in A.CONTRASTS)
            bw = A.bitwise_share(idx)
            assert bw["share"] == 1.0 and bw["nat_slots_equal"] == 100 and bw["histories_equal"] == 100
            r2 = A.anchor_core(Au, dict(cal, rho_V=cal["rho_V"] * 1.08), cal, rep4, idx, x)
            assert not r2["pass"] and r2["criterion_3"]["rho_V"] is False
            p = v4 / "qwen2.5-0.5b" / "test" / "NAT-V.jsonl"
            rows = [json.loads(l) for l in p.read_text().splitlines()]
            for rr in rows:
                rr["js"] = [x_ * math.exp(0.3) for x_ in rr["js"]]
            p.write_text("".join(json.dumps(rr) + "\n" for rr in rows))
            r3 = A.anchor_core(Au, cal, cal, rep4, idx, x)
            assert not r3["pass"] and r3["contrasts"]["H2"]["criterion_1"] is False and r3["contrasts"]["H3"]["criterion_1"] is False
            assert A.bitwise_share(idx)["share"] < 1.0
        finally:
            A.V4_RUNS = saved


@t
def a11_independent_recomputation_agrees():
    with study() as out:
        rep = run_analyze()
        v4 = out.parent / "v4r"
        v4.mkdir()
        (v4 / "v4_report.json").write_text(json.dumps(synthetic_v4_report()))
        saved = (RC.V4_REPORT_SHA256, RC.HERE)
        import hashlib
        RC.V4_REPORT_SHA256 = hashlib.sha256((v4 / "v4_report.json").read_bytes()).hexdigest()
        man_dir = out.parent / "man"
        man_dir.mkdir()
        man = A.manifest()
        (man_dir / "V5_MODEL_MANIFEST.json").write_text(json.dumps({"models": man}))
        RC.HERE = man_dir
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                code = RC.main(["--runs", str(out), "--v4-runs", str(v4)])
            res = json.loads((out / "v5_recomputation.json").read_text())
            assert code == 0 and res["n_disagree"] == 0 and res["n_compared"] > 150, (res["n_compared"], res["disagreements"][:3])
            assert any("A-X" in r["quantity"] and "classification" in r["quantity"] for r in res["rows"])
        finally:
            RC.V4_REPORT_SHA256, RC.HERE = saved
        assert rep["all_units"]["A-X"]["joint_reading_NAT-C1"]["reading"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--write")
    ap.add_argument("--only")
    a = ap.parse_args(argv)
    RESULTS.clear()
    for fn in TESTS:
        if a.only and a.only not in fn.__name__:
            continue
        t0 = time.time()
        try:
            fn()
            RESULTS.append({"test": fn.__name__, "ok": True})
            print(f"PASS {fn.__name__} ({time.time() - t0:.1f}s)", flush=True)
        except Exception as e:  # noqa: BLE001
            RESULTS.append({"test": fn.__name__, "ok": False, "error": f"{type(e).__name__}: {e}"[:500]})
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
    n_ok = sum(r["ok"] for r in RESULTS)
    print(f"{n_ok}/{len(RESULTS)} passed")
    if a.write:
        Path(a.write).write_text(json.dumps({"passed": n_ok, "n": len(RESULTS), "results": RESULTS}, indent=1))
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
