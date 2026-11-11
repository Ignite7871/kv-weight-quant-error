"""Analysis tests on simulated truth (PROTOCOL_V4.md sections 4, 7, 9, 12). A simulator writes v4-format records (six arms, X13a slot sums,
calibration files) with KNOWN effects for the 8 ladder models; each scenario checks the readings the protocol prescribes:
"supported in every model" / "falsified" / "not established", Holm within each family, H1-H3 equivalence falsification, the H4 exact model-level
permutation (35 labellings, complete separation, sign readings, not estimable), H5 thresholds with the model's own head_dim, the BF16-flagged
with/without reporting, unresolved models, FQ and the cross-quantizer label, and the Step 0 criteria on the locked v3 records themselves.

  venv/Scripts/python.exe research_v2/v4/tests/test_analyze_v4.py [--quick] [--write <json>]
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
V4 = HERE.parent
RV = V4.parent
sys.path.insert(0, str(RV))
sys.path.insert(0, str(V4))

import numpy as np  # noqa: E402

import analyze_v3 as L  # noqa: E402
import analyze_v4 as A  # noqa: E402

TESTS, RESULTS = [], []
QUICK = False
S = 10
HD = {m: e["geometry"]["head_dim"] for m, e in A.manifest().items()}
FLAGS = ("exact_ok", "natural_ok", "routing_reset", "construction_ok", "rounding_ok", "finite_ok")
TRUTH = {"H1": -1.0, "H2": 0.8, "H3": -0.9}
NATC1 = {"qwen2.5-0.5b": -0.25, "qwen2.5-1.5b": -0.3, "qwen2.5-3b": -0.28, "qwen2.5-7b": -0.22,
         "llama3.2-1b": 0.05, "llama3.2-3b": 0.08, "llama3.1-8b": 0.03, "mistral-7b-v0.3": -0.1}


def t(fn):
    TESTS.append(fn)
    return fn


def n_problems():
    return 100                                                       # r5: the design's n (a model below it is unresolved)


def write_model(out: Path, tag: str, n=None, eff=None, sd=0.5, cm=True, hd=64, L_=2, nkv=2, seed=0, margin=0.21, informative=True):
    """Arms are built from per-problem log mean-JS values: l(NAT-K) - l(C1-K) = H1, l(NAT-V) - l(C1-V) = H2, l(NAT-K) - l(NAT-V) = H3,
    l(NAT) - l(C1) = NAT-C1, each plus N(0, sd^2) problem noise. js = [0, e^l, ..., e^l] (the common factor (S-1)/S cancels in theta)."""
    eff = {**TRUTH, "NAT-C1": NATC1.get(tag, 0.0), **(eff or {})}
    n = n or n_problems()
    rng = np.random.default_rng(seed)
    d = out / tag / "test"
    (d / "slots").mkdir(parents=True, exist_ok=True)
    recs = {a: [] for a in ("NAT", "C1", "NAT-K", "C1-K", "NAT-V", "C1-V")}
    noise = rng.normal(0, sd, (6, n))
    noise[1:] -= noise[1:].mean(axis=1, keepdims=True)              # demeaned: each model's estimates equal its set truth exactly
    for i in range(n):
        base = rng.normal(-6, 0.3)
        lNV = base + noise[0, i]
        lNK = lNV + eff["H3"] + noise[1, i]
        lCK = lNK - eff["H1"] + noise[2, i]
        lCV = lNV - eff["H2"] + noise[3, i]
        lN = base + 0.5 + noise[4, i]
        lC = lN - eff["NAT-C1"] + noise[5, i]
        for a, l in (("NAT", lN), ("C1", lC), ("NAT-K", lNK), ("C1-K", lCK), ("NAT-V", lNV), ("C1-V", lCV)):
            recs[a].append({"index": i, "arm": a, "S": S, "js": [0.0] + [math.exp(l)] * (S - 1), "rdev_n_gt5": [0] * S,
                            "rel_K": [0.05] * S, "rel_V": [0.2] * S, **{k: True for k in FLAGS}})
        mu = rng.normal(0, 1, (L_, nkv, hd)) if cm else np.zeros((L_, nkv, hd))
        dK = mu[None] + rng.normal(0, 0.3 if cm else 1.0, (S, L_, nkv, hd))
        dV = mu[None] + rng.normal(0, 0.3 if cm else 1.0, (S, L_, nkv, hd))
        np.savez_compressed(d / "slots" / f"NAT_{i}.npz", normsK=np.linalg.norm(dK, axis=-1), normsV=np.linalg.norm(dV, axis=-1),
                            S1K=dK.sum(0), S1V=dV.sum(0))
    for a, rs in recs.items():
        (d / f"{a}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rs))
    (out / tag / "calibration.json").write_text(json.dumps({"margin_log_js": margin, "dose_response_informative": informative, "rho_K": 0.06,
                                                            "rho_V": 0.2, "rho_V_over_rho_K": 0.2 / 0.06, "dose_slope_b_hat": 2.2}))
    (out / tag / "test_complete.json").write_text("{}")


@contextlib.contextmanager
def study(overrides=None, flagged=(), stopped=None, not_run=()):
    """overrides: tag -> kwargs for write_model."""
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        saved = A.OUT
        A.OUT = out
        try:
            for k, m in enumerate(A.LADDER):
                kw = dict(hd=HD[m], seed=k)
                kw.update((overrides or {}).get(m, {}))
                write_model(out, m, **kw)
            for m in flagged:
                (out / m / "precision.json").write_text(json.dumps({"precision": "bf16"}))
            for m, n in (stopped or {}).items():
                shutil.rmtree(out / m)
                write_model(out, m, n=n, hd=HD[m], seed=99)
                (out / m / "test_complete.json").unlink()
                (out / m / "ceiling_stop.json").write_text("{}")
            for m in not_run:
                shutil.rmtree(out / m)
                (out / m).mkdir()
                (out / m / "not_run.json").write_text(json.dumps({"reason": "dropped"}))
            with contextlib.redirect_stdout(io.StringIO()):
                rep = A.analyze()
            yield rep
        finally:
            A.OUT = saved


def res(rep, key, part="all_models"):
    return rep[part][key]["result"]


# ----------------------------------------------------------------------------- unit tests
@t
def a01_h4_permutation_exact():
    qwen, llama = A.QWEN, A.LLAMA
    th = {**{m: -0.3 + 0.01 * k for k, m in enumerate(qwen)}, **{m: 0.1 + 0.01 * k for k, m in enumerate(llama)}}
    p = A.h4_permutation(th, qwen, llama)
    assert p["n_labellings"] == 35 and abs(p["p_one_sided"] - 1 / 35) < 1e-15 and p["complete_separation"]
    th2 = {**{m: 0.3 for m in qwen}, **{m: -0.3 for m in llama}}
    assert A.h4_permutation(th2, qwen, llama)["p_one_sided"] == 1.0
    th3 = dict(th, **{"llama3.2-1b": -0.275})                                        # one inversion: one Llama below one Qwen
    p3 = A.h4_permutation(th3, qwen, llama)
    assert not p3["complete_separation"] and abs(p3["p_one_sided"] - 2 / 35) < 1e-15
    assert A.h4_permutation(th, qwen[:3], llama)["n_labellings"] == 20


@t
def a02_readings_rules():
    s = {"ci95": [-0.5, -0.1], "verdict": "difference"}
    assert A.reading_directional(s, 0.01, -1) == "supported" and A.reading_directional(s, 0.2, -1) == "inconclusive"
    assert A.reading_directional({"ci95": [0.1, 0.4], "verdict": "difference"}, 0.01, -1) == "falsified"
    assert A.reading_directional({"ci95": [-0.1, 0.1], "verdict": "equivalent"}, 0.9, -1) == "falsified"
    assert A.reading_directional({"ci95": [0.1, 0.4], "verdict": "difference"}, 0.01, +1) == "supported"
    assert A.reading_h5({"ci95": [1.5, 2.0]}, 0.001) == "supported" and A.reading_h5({"ci95": [1.2, 2.0]}, 0.001) == "inconclusive"
    assert A.reading_h5({"ci95": [0.1, 0.6]}, 0.001) == "falsified"
    assert A.universal({"a": "supported", "b": "supported"}) == "supported in every model"
    assert A.universal({"a": "supported", "b": "falsified"}) == "falsified"
    assert A.universal({"a": "supported", "b": "inconclusive"}) == "not established in all models"


@t
def a03_x13a_null_uses_model_head_dim():
    """An isotropic delta in R^128 gives L close to 0 with the head_dim-128 null and a biased value with a 64-dim null."""
    rng = np.random.default_rng(3)
    norms = np.abs(rng.normal(1, 0.2, (30, 2, 2)))
    g = rng.normal(0, 1, (30, 2, 2, 128))
    S1 = (g / np.linalg.norm(g, axis=-1, keepdims=True) * norms[..., None]).sum(0)
    import analyze_x13 as X13
    x128 = math.log(30 * X13.cm_share(norms, S1)) - X13.null_lognc(norms, np.random.default_rng(1), 200, 128)
    x64 = math.log(30 * X13.cm_share(norms, S1)) - X13.null_lognc(norms, np.random.default_rng(1), 200, 64)
    assert abs(x128) < 0.6 and x128 != x64


@t
def a04_calibrate_core_reproduces_v3():
    import torch
    for m in ("qwen-base", "llama-base"):
        d = RV / "v3_runs" / m / "calib"
        assert d.is_dir(), f"r8: the locked v3 records are required, {d} is missing (tools/v4_v3_records.py)"
        recs = {t_: L.load_jsonl(d / f"{t_}.jsonl") for t_ in ("NAT", "C1_a0.5", "C1_a1", "C1_a2")}
        aux = torch.load(d / "aux_state.pt")
        c = A.calibrate_core(recs, aux["rhoK"].numpy(), aux["rhoV"].numpy())
        v3 = json.loads((RV / "v3_runs" / m / "calibration.json").read_text())
        for k in ("dose_slope_b_hat", "c_used", "margin_log_js", "rho_K", "rho_V"):
            assert abs(c[k] - v3[k]) <= 1e-12 * max(1, abs(v3[k])), (m, k)


# ----------------------------------------------------------------------------- scenarios
@t
def a05_truth_supported_everywhere():
    with study() as rep:
        for k in ("F1 H1", "F2 H2", "F3 H3"):
            assert res(rep, k) == "supported in every model", (k, rep["all_models"][k]["readings"])
            fam = rep["all_models"][k]
            assert len(fam["holm_p"]) == 8
            want = L.holm([rep["per_model"][m]["contrasts"][k.split()[1]]["p"] for m in fam["models"]])
            assert np.allclose([fam["holm_p"][m] for m in fam["models"]], want, rtol=0, atol=0), k
        assert res(rep, "F5 H5") == "supported in every model" and len(rep["all_models"]["F5 H5"]["holm_p"]) == 16
        h4 = rep["all_models"]["H4"]
        assert h4["result"] == "supported" and h4["permutation"]["complete_separation"] and abs(h4["permutation"]["p_one_sided"] - 1 / 35) < 1e-15
        assert "mistral_out_of_sample" in h4 and "mistral-7b-v0.3" not in h4["qwen"] + h4["llama"]
        assert rep["abstract_level"]["source"] == "all_models"
        for m in A.LADDER:
            assert rep["per_model"][m]["head_dim"] == HD[m]


@t
def a06_one_model_falsifies_h1():
    with study({"mistral-7b-v0.3": {"eff": {"H1": 0.5}}}) as rep:
        assert res(rep, "F1 H1") == "falsified" and rep["all_models"]["F1 H1"]["readings"]["mistral-7b-v0.3"] == "falsified"
        assert res(rep, "F2 H2") == "supported in every model"


@t
def a07_equivalence_falsifies_h2():
    with study({"qwen2.5-3b": {"eff": {"H2": 0.0}, "sd": 0.15}}) as rep:
        r = rep["per_model"]["qwen2.5-3b"]["contrasts"]["H2"]
        assert r["verdict"] == "equivalent" and rep["all_models"]["F2 H2"]["readings"]["qwen2.5-3b"] == "falsified"


@t
def a08_equivalence_needs_informative_dose():
    with study({"qwen2.5-3b": {"eff": {"H2": 0.0}, "sd": 0.15, "informative": False}}) as rep:
        assert rep["per_model"]["qwen2.5-3b"]["contrasts"]["H2"]["verdict"] != "equivalent"
        assert rep["all_models"]["F2 H2"]["readings"]["qwen2.5-3b"] != "falsified"


@t
def a09_h4_qwen_equivalent_falsifies():
    with study({"qwen2.5-7b": {"eff": {"NAT-C1": -0.02}, "sd": 0.15}}) as rep:
        assert rep["all_models"]["H4"]["result"] == "falsified"


@t
def a10_h4_incomplete_separation_inconclusive():
    with study({"qwen2.5-7b": {"eff": {"NAT-C1": -0.15}, "sd": 0.6}, "llama3.1-8b": {"eff": {"NAT-C1": -0.17}, "sd": 1.5}}) as rep:
        h4 = rep["all_models"]["H4"]
        assert not h4["permutation"]["complete_separation"] and h4["permutation"]["p_one_sided"] > 0.05
        assert h4["result"] == "inconclusive", h4


@t
def a11_flagged_with_and_without():
    with study({"llama3.2-3b": {"eff": {"H1": 0.4}}}, flagged=("llama3.2-3b",)) as rep:
        assert rep["flagged_bf16"] == ["llama3.2-3b"]
        assert res(rep, "F1 H1") == "falsified" and res(rep, "F1 H1", "without_flagged") == "supported in every model"
        assert "llama3.2-3b" not in rep["without_flagged"]["F1 H1"]["holm_p"] and len(rep["without_flagged"]["F1 H1"]["holm_p"]) == 7
        assert rep["abstract_level"]["source"] == "without_flagged" and "F1 H1" in rep["abstract_level"]["disagreements"]
        wf = rep["without_flagged"]["H4"]["permutation"]                 # 4 Qwen + 2 Llama: C(6, 2) = 15 labellings, min p 1/15 > 0.05
        assert wf["n_labellings"] == 15 and abs(wf["min_attainable_p"] - 1 / 15) < 1e-15
        assert rep["without_flagged"]["H4"]["result"] != "supported" and "H4" in rep["abstract_level"]["disagreements"]


@t
def a12_unresolved_model_and_h4_not_estimable():
    with study(stopped={"qwen2.5-3b": 90, "qwen2.5-7b": 80}) as rep:
        assert rep["status"]["qwen2.5-3b"] == "unresolved" and "qwen2.5-3b" not in rep["per_model"]
        assert rep["all_models"]["H4"]["result"] == "not estimable"
        assert len(rep["all_models"]["F1 H1"]["holm_p"]) == 6


@t
def a13_h5_isotropic_falsified():
    with study({"llama3.1-8b": {"cm": False}}) as rep:
        rd = rep["all_models"]["F5 H5"]["readings"]
        assert rd["llama3.1-8b/K"] == "falsified" and rd["llama3.1-8b/V"] == "falsified" and res(rep, "F5 H5") == "falsified"


@t
def a19_mistral_dropped_leaves_seven_and_h4_intact():
    with study(not_run=("mistral-7b-v0.3",)) as rep:
        assert rep["status"]["mistral-7b-v0.3"] == "not run" and "mistral-7b-v0.3" not in rep["per_model"]
        assert len(rep["all_models"]["F1 H1"]["holm_p"]) == 7 and len(rep["all_models"]["F5 H5"]["holm_p"]) == 14
        h4 = rep["all_models"]["H4"]
        assert h4["result"] == "supported" and abs(h4["permutation"]["p_one_sided"] - 1 / 35) < 1e-15 and "mistral_out_of_sample" not in h4


@t
def a20_h4_needs_all_seven():
    with study(stopped={"qwen2.5-7b": 80}) as rep:                       # 3 Qwen + 3 Llama would allow p = 1/20; r5: not estimable
        assert rep["status"]["qwen2.5-7b"] == "unresolved" and rep["all_models"]["H4"]["result"] == "not estimable" and "seven" in rep["all_models"]["H4"]["reason"]
        assert len(rep["all_models"]["F1 H1"]["holm_p"]) == 7


@t
def a15_refuses_before_all_models_finish():
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        saved = A.OUT
        A.OUT = out
        try:
            for k, m in enumerate(A.LADDER):
                write_model(out, m, n=20, hd=HD[m], seed=k)
            (out / "llama3.1-8b" / "test_complete.json").unlink()
            try:
                A.analyze()
                raise AssertionError("analyze must refuse while a model is running")
            except SystemExit:
                pass
        finally:
            A.OUT = saved


@t
def a16_step0_on_v3_records_themselves():
    """Step 0 with the locked v3 records of the first 100 ids and v3's calibration on the first 50 copied in as the 'cloud' records: every paired difference is
    exactly 0, criterion 1 holds, the verdict categories equal v3's, criterion 3 holds; shifting one arm by +0.3 log units fails criterion 1."""
    for lab in A.STEP0.values():                                      # r8: fail loudly, never pass without the records
        for f in ("tierA/NAT.jsonl", "calib", "calibration.json"):
            assert (RV / "v3_runs" / lab / f).exists(), f"r8: the locked v3 records are required, v3_runs/{lab}/{f} is missing (tools/v4_v3_records.py)"
    FIRST_100 = set(json.loads(A.SAMPLE.read_text(encoding="utf-8"))["tierA_ids"])
    assert len(FIRST_100) == 100
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        saved = A.OUT
        A.OUT = out
        try:
            for m, lab in A.STEP0.items():
                src = RV / "v3_runs" / lab
                d = out / m / "test"
                (d / "slots").mkdir(parents=True)
                for a in ("NAT", "C1", "NAT-K", "C1-K", "NAT-V", "C1-V", "FP16_history"):
                    rows = [l for l in (src / "tierA" / f"{a}.jsonl").read_text().splitlines() if l.strip() and json.loads(l)["index"] in FIRST_100]
                    assert len(rows) == 100
                    (d / f"{a}.jsonl").write_text("\n".join(rows) + "\n")
                cal = A.v3_calibration_on(lab)
                assert cal["n_calibration_problems"] == 50
                (out / m / "calibration.json").write_text(json.dumps(cal))
                rng = np.random.default_rng(0)
                for i in L.load_jsonl(d / "NAT.jsonl"):
                    v = rng.normal(0, 1, (4, 2, 2, 64))
                    np.savez(d / "slots" / f"NAT_{i}.npz", normsK=np.linalg.norm(v, axis=-1), normsV=np.linalg.norm(v, axis=-1), S1K=v.sum(0), S1V=v.sum(0))
            with contextlib.redirect_stdout(io.StringIO()):
                d0 = A.step0()
            assert d0["decision"] == "pass", json.dumps(d0, default=str)[:2000]
            for m, r in d0["models"].items():
                for h, row in r["contrasts"].items():
                    assert row["delta"]["estimate"] == 0.0 and row["criterion_1"], (m, h)
                assert r["reported"]["history_token_identical_share"] == 1.0 and r["n_paired"] == 100
                c3 = r["criterion_3"]
                assert c3["v3_n_problems"] == 50 and c3["rho_K"] and c3["rho_V"] and c3["b_hat"] and c3["v4"] == c3["v3"], "criterion 3 compares like with like"
            assert d0["models"]["llama3.2-1b"]["contrasts"]["NAT-C1"]["criterion_2"].startswith("not applicable")    # D9: |theta_v3| = 0.013
            assert d0["models"]["qwen2.5-0.5b"]["contrasts"]["NAT-C1"]["criterion_2"] is True                         # |theta_v3| = 0.241
            (out / "step0.json").unlink()
            p = out / "llama3.2-1b" / "test" / "C1-K.jsonl"
            rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
            for r in rows:
                r["js"] = [x * math.exp(0.3) for x in r["js"]]
            p.write_text("".join(json.dumps(r) + "\n" for r in rows))
            with contextlib.redirect_stdout(io.StringIO()):
                d1 = A.step0()
            assert d1["decision"].startswith("fail") and not d1["models"]["llama3.2-1b"]["contrasts"]["H1"]["criterion_1"]
            (out / "step0.json").unlink()                                    # net C1-K shift +0.15: delta = -0.15, 90 % CI inside the margin, |delta| > 0.10
            for r in rows:
                r["js"] = [x * math.exp(-0.15) for x in r["js"]]
            p.write_text("".join(json.dumps(r) + "\n" for r in rows))
            with contextlib.redirect_stdout(io.StringIO()):
                d2 = A.step0()
            row = d2["models"]["llama3.2-1b"]["contrasts"]["H1"]
            assert abs(row["delta"]["estimate"] + 0.15) < 1e-9 and -0.2095 < row["delta"]["ci90"][0] and row["delta"]["ci90"][1] < 0.2095
            assert not row["criterion_1"] and d2["decision"].startswith("fail")
            (out / "step0.json").unlink()                                    # criterion 3: rho_K 10 % above v3 fails, b_hat 0.2 above passes
            cp = out / "qwen2.5-0.5b" / "calibration.json"
            cal = json.loads(cp.read_text())
            cal["rho_K"] *= 1.10
            cal["dose_slope_b_hat"] += 0.2
            cp.write_text(json.dumps(cal))
            with contextlib.redirect_stdout(io.StringIO()):
                d3 = A.step0()
            c3 = d3["models"]["qwen2.5-0.5b"]["criterion_3"]
            assert c3["rho_K"] is False and c3["rho_V"] is True and c3["b_hat"] is True and not d3["models"]["qwen2.5-0.5b"]["pass"]
        finally:
            A.OUT = saved


@t
def a18_h4_sign_rules_unit():
    def est(qv, lv, qci, lci, qverd="difference"):
        e = {}
        for m, v, ci in zip(A.QWEN, qv, qci):
            e[m] = {"contrasts": {"NAT-C1": {"estimate": v, "ci95": ci, "verdict": qverd}}}
        for m, v, ci in zip(A.LLAMA, lv, lci):
            e[m] = {"contrasts": {"NAT-C1": {"estimate": v, "ci95": ci, "verdict": "inconclusive"}}}
        return e
    ms = A.QWEN + A.LLAMA
    q, qc = [-0.3, -0.29, -0.28, -0.27], [[-0.4, -0.2]] * 4
    ok = est(q, [0.1, 0.11, 0.12], qc, [[-0.05, 0.2]] * 3)
    assert A.h4(ms, ok)["result"] == "supported"
    llama_neg = est(q, [0.1, 0.11, 0.12], qc, [[-0.05, 0.2], [-0.05, 0.2], [-0.3, -0.01]])
    assert A.h4(ms, llama_neg)["result"] == "falsified"
    qwen_pos = est(q, [0.1, 0.11, 0.12], [[-0.4, -0.2]] * 3 + [[0.01, 0.2]], [[-0.05, 0.2]] * 3)
    assert A.h4(ms, qwen_pos)["result"] == "falsified"
    qwen_unsigned = est(q, [0.1, 0.11, 0.12], [[-0.4, -0.2]] * 3 + [[-0.5, 0.05]], [[-0.05, 0.2]] * 3)
    assert A.h4(ms, qwen_unsigned)["result"] == "inconclusive"


@t
def a17_independent_recomputation_agrees():
    """recompute_v4_independent.py (no project imports) agrees with analyze_v4 within the declared tolerances, with and without a flagged model;
    a tampered report is caught."""
    import recompute_v4_independent as RC
    with study({"llama3.2-3b": {"eff": {"H1": 0.4}}}, flagged=("llama3.2-3b",)) as rep:
        with contextlib.redirect_stdout(io.StringIO()):
            assert RC.main(["--runs", str(A.OUT)]) == 0, json.loads((A.OUT / "v4_recomputation.json").read_text())["disagreements"][:3]
        r = json.loads((A.OUT / "v4_report.json").read_text())
        r["per_model"]["qwen2.5-1.5b"]["contrasts"]["H2"]["estimate"] += 1e-6
        r["all_models"]["F3 H3"]["result"] = "falsified"
        (A.OUT / "v4_report.json").write_text(json.dumps(r))
        with contextlib.redirect_stdout(io.StringIO()):
            assert RC.main(["--runs", str(A.OUT)]) == 1
        bad = json.loads((A.OUT / "v4_recomputation.json").read_text())
        assert any("F3 H3 result" in q for q in bad["unresolved_verdicts"]) and any("H2 estimate" in d["quantity"] for d in bad["disagreements"])


def main(argv=None):
    global QUICK
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--write")
    ap.add_argument("--only")
    a = ap.parse_args(argv)
    QUICK = a.quick
    RESULTS.clear()
    for fn in TESTS:
        if a.only and a.only not in fn.__name__:
            continue
        t0 = time.time()
        try:
            fn()
            RESULTS.append({"test": fn.__name__, "ok": True, "seconds": round(time.time() - t0, 1)})
            print(f"PASS {fn.__name__} ({time.time() - t0:.1f}s)", flush=True)
        except Exception as e:  # noqa: BLE001
            RESULTS.append({"test": fn.__name__, "ok": False, "error": f"{type(e).__name__}: {e}"[:800]})
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}"[:1500], flush=True)
            traceback.print_exc()
    n_ok = sum(r["ok"] for r in RESULTS)
    print(f"{n_ok}/{len(RESULTS)} passed")
    if a.write:
        Path(a.write).write_text(json.dumps({"quick": QUICK, "passed": n_ok, "n": len(RESULTS), "results": RESULTS}, indent=1))
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
