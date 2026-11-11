"""CPU tests for X9 (v3/phase7/FOLLOWUP_X9.md section 5). Tiny Qwen2 and tiny GQA Llama of tests/run_v3_tests.py (real DynamicLinear / NF4 path);
--real adds a few teacher-forced steps on the four real models on CPU with their frozen calibration contexts. LOGIC only.

  venv/Scripts/python.exe research_v2/v3/phase7/test_x9.py [--real] [--write]
"""

from __future__ import annotations

import json
import math
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

P7 = Path(__file__).resolve().parent
RV = P7.parents[1]
sys.path.insert(0, str(RV / "tests"))
sys.path.insert(0, str(RV))
sys.path.insert(0, str(P7))

import run_v3_tests as T                         # noqa: E402  (tiny backends and run_arm; HF before torch)

import numpy as np                               # noqa: E402
import torch                                     # noqa: E402

import rv3.core as V                             # noqa: E402
import rv3x9 as X                                # noqa: E402
import run_x9 as RX                              # noqa: E402
import analyze_x9 as AX                          # noqa: E402

R = T.R
FAMILIES = ("qwen", "llama")
TESTS = []
GATE_ARMS = ["NAT", "NAT-K*", "NAT-V*", "QA", "C1-K", "LOC"]


def t(fn):
    TESTS.append(fn)
    return fn


def ctx_for(b, seed=5, rho=0.07):
    return V.synthetic_ctx(b.model.config, T.DEV, seed=seed, rho_star=rho)


def arm_rec(b, arm, ctx, alpha=1.0, x9=False, prompt=None, ref=None, S=None):
    """One teacher-forced arm with predictors; x9=True runs it with the X9 overrides installed. Returns (record, X9 predictor state or None)."""
    cap = V.QueryCapture(b.model)
    try:
        if x9:
            X.install(ctx["U"])
            X.clear()
        if prompt is None:
            r = T.run_arm(b, arm, alpha, ctx, cap, True)
        else:
            rl = V.reference_logits_v3(b, prompt, ref, S)
            r = V.teacher_forced_arm_v3(b, prompt, ref, S, arm, alpha, V.arm_seeds("x9t", arm, alpha, 0, "T"), rl, ctx, cap, True)
        inst = X.INSTANCES[-1] if x9 else None
    finally:
        cap.remove()
        if x9:
            X.uninstall()
    return r, inst


def roundtrip(r):
    return json.loads(json.dumps(r))


def per_layer_checks(r, inst):
    lf = X.layer_fields(inst)
    for s in range(len(r["js"])):
        for k, pk in (("PL_logit", "P_logit"), ("PL_val", "P_val"), ("PL_anormK", "P_anormK")):
            a, b = sum(lf[k][s]), r[pk][s]
            assert abs(a - b) <= 1e-12 * max(abs(b), 1e-300) or a == b, f"step {s + 1}: sum {k} {a!r} vs {pk} {b!r}"
        a, b = float(np.mean(lf["PL_norm"][s])), r["P_norm"][s]
        assert abs(a - b) <= 1e-12 * max(abs(b), 1e-300) or a == b, f"step {s + 1}: mean PL_norm {a!r} vs P_norm {b!r}"


# ----------------------------------------------------------------------------- tests
@t
def x01_records_bitwise_identical_with_x9_state(fam):
    b = T.be(fam)
    ctx = ctx_for(b)
    for arm in GATE_ARMS:
        r0, _ = arm_rec(b, arm, ctx)
        r1, inst = arm_rec(b, arm, ctx, x9=True)
        d = RX.compare_records(r1, roundtrip(r0))
        assert not d, f"{arm}: X9 record differs from the locked one: {list(d)[:4]}"
        per_layer_checks(r1, inst)
        assert len(inst.layer_terms) == len(r1["js"]) == len(inst.slots["eK"]), f"{arm}: one log entry per step"


@t
def x02_per_layer_p_norm_independent(fam):
    b = T.be(fam)
    ctx = ctx_for(b)
    r, inst = arm_rec(b, "NAT", ctx, x9=True)
    eK, eV = np.stack(inst.slots["eK"]), np.stack(inst.slots["eV"])                 # [S, L, n_kv]
    lf = X.layer_fields(inst)
    for s in range(1, len(r["js"])):                                                 # step s+1 reads the s writes before it
        ck, cv = eK[:s].sum(0), eV[:s].sum(0)
        ref = np.concatenate([np.sqrt(ck / s), np.sqrt(cv / s)], axis=1).mean(1)     # per layer: mean over K heads and V heads
        assert np.allclose(lf["PL_norm"][s], ref, rtol=1e-12, atol=0), f"step {s + 1}: PL_norm {lf['PL_norm'][s]} vs {ref.tolist()}"


@t
def x03_l0nat_construction(fam):
    b = T.be(fam)
    ctx = ctx_for(b)
    seen = []
    X.install(ctx["U"])
    inner = V.stored_entry_v3

    def w(arm, alpha, kF, kM, gen, vgen, c):
        out, info = inner(arm, alpha, kF, kM, gen, vgen, c)
        if arm == X.X9_ARM:
            seen.append((alpha, kF.clone(), kM.clone(), out.clone(), info["pre"].clone()))
        return out, info
    V.stored_entry_v3 = w
    cap = V.QueryCapture(b.model)
    try:
        r = T.run_arm(b, X.X9_ARM, 1.0, ctx, cap, True)
    finally:
        cap.remove()
        X.uninstall()
    assert len(seen) == T.S, "one stored entry per step"
    differs = False
    for alpha, kF, kM, out, pre in seen:
        lk, _ = X._LOCKED["stored_entry_v3"]("NAT-K*", alpha, kF, kM, None, None, ctx)
        assert torch.equal(out[0], kM[0]), "layer-0 K row is not kappa^M bitwise"
        assert torch.equal(out[2::2], lk[2::2]), "K rows of layers >= 1 differ from NAT-K*"
        assert torch.equal(out[1::2], kF[1::2]), "a V row differs from kappa^F"
        assert torch.equal(pre[0], kM[0].double()), "pre-rounding layer-0 K row is not kappa^M"
        differs = differs or not torch.equal(lk[0], kM[0])
    assert differs, "test is vacuous: NAT-K* never changed layer 0"
    errs = R.integrity_errors({X.X9_ARM: r})
    assert not errs, errs
    cF = V.fp16_teacher_forced_cache(b, T.PROMPT, T.REF, T.S)
    X.install(ctx["U"])
    try:
        r0, c0 = T.run_arm(b, X.X9_ARM, 0.0, ctx, None, False, return_cache=True)
    finally:
        X.uninstall()
    assert all(x == 0.0 for x in r0["js"]), "alpha 0: JS not exactly 0"
    assert all(torch.equal(x.keys, y.keys) and torch.equal(x.values, y.values) for x, y in zip(c0.layers, cF.layers)), "alpha 0: cache differs from FP16"


@t
def x04_slot_energies_independent(fam):
    g = torch.Generator().manual_seed(11)
    L_, h, hd = 3, 2, 16
    U = torch.linalg.qr(torch.randn(L_, h, hd, hd, generator=g, dtype=torch.float64))[0][..., :8]
    d = torch.randn(2 * L_, h, hd, generator=g, dtype=torch.float64)
    F = torch.randn(2 * L_, h, hd, generator=g, dtype=torch.float64) * 5
    e = X.slot_energies(d, F, U)
    Un, dn, Fn = U.numpy(), d.numpy(), F.numpy()
    for l in range(L_):
        for k in range(h):
            P = Un[l, k] @ Un[l, k].T
            for row, ek, pk in ((2 * l, "eK", "pK"), (2 * l + 1, "eV", "pV")):
                x = dn[row, k]
                assert math.isclose(e[ek][l, k].item(), float(x @ x), rel_tol=1e-12)
                assert math.isclose(e[pk][l, k].item(), float(np.sum((P @ x) ** 2)), rel_tol=1e-12)
            assert math.isclose(e["fK"][l, k].item(), float(np.linalg.norm(Fn[2 * l, k])), rel_tol=1e-12)
            assert math.isclose(e["fV"][l, k].item(), float(np.linalg.norm(Fn[2 * l + 1, k])), rel_tol=1e-12)
    b = T.be(fam)
    ctx = ctx_for(b)
    r, inst = arm_rec(b, "NAT", ctx, x9=True)
    eK = np.stack(inst.slots["eK"])
    for s in range(len(r["js"])):
        a, ref = float(eK[s].mean()), r["dn2_K"][s]
        assert math.isclose(a, ref, rel_tol=1e-6, abs_tol=1e-30), f"step {s + 1}: mean eK {a!r} vs locked dn2_K {ref!r}"


@t
def x05_reproduction_comparator(fam):
    base = {"index": 1, "js": [0.0, 0.125, 0.25], "P_out": [float("nan"), 0.5, 0.5], "top": [3, 4, 5], "exact_ok": True,
            "rdev_arg": [None, [0, "K", 1], None], "wall_seconds": 2.0}
    lk = roundtrip(base)
    assert RX.compare_records(dict(base), lk) == {}
    assert RX.compare_records({**base, "wall_seconds": 9.0}, lk) == {}, "wall_seconds must be ignored"
    assert RX.compare_records({**base, "PL_logit": [[0.0]]}, lk) == {}, "X9-only fields must be ignored"
    ulp = {**base, "js": [0.0, math.nextafter(0.125, 1.0), 0.25]}
    assert "js" in RX.compare_records(ulp, lk), "1-ulp change not detected"
    assert RX.compare_records(ulp, lk)["js"]["first_differing_step"] == 2
    assert "top" in RX.compare_records({**base, "top": [3, 4, 6]}, lk), "token change not detected"
    assert "P_out" in RX.compare_records({**base, "P_out": [0.5, float("nan"), 0.5]}, lk), "NaN position change not detected"
    assert "js" in RX.compare_records({**base, "js": [-0.0, 0.125, 0.25]}, lk), "sign of zero not detected"
    miss = dict(base)
    del miss["top"]
    assert "top" in RX.compare_records(miss, lk), "missing key not detected"
    assert "extra" in RX.compare_records({**base, "extra": 1}, lk), "unexpected key not detected"
    assert RX.history_matches([1, 2, 3], [1, 2, 3]) and not RX.history_matches([1, 2, 3], [1, 2, 4]) and not RX.history_matches([1, 2], [1, 2, 3])


@t
def x06_manifest_auth_subset_design(fam):
    tmp = Path(tempfile.mkdtemp())
    try:
        for n in RX.MANIFEST_FILES:
            shutil.copy(P7 / n, tmp / n)
        mid = RX.manifest_id(tmp)
        assert mid == RX.manifest_id(P7), "manifest depends on the directory"
        p = tmp / "rv3x9.py"
        p.write_bytes(p.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
        assert RX.manifest_id(tmp) == mid, "CRLF vs LF changes the manifest id"
        p.write_bytes(p.read_bytes() + b"#")
        assert RX.manifest_id(tmp) != mid, "a changed byte does not change the manifest id"
    finally:
        shutil.rmtree(tmp)
    assert not RX.auth_ok("", mid) and not RX.auth_ok(mid, mid) and not RX.auth_ok("FOLLOWUP_X9", mid)
    assert not RX.auth_ok("FOLLOWUP_X9 0123456789abcdef", mid) and not RX.auth_ok("FOLLOWUP_X9 " + mid, "")
    assert RX.auth_ok(f"FOLLOWUP_X9\n{mid}\n", mid)
    s = json.loads((RV / "v3_sample.json").read_text())
    ids = RX.subset_ids(s["tierA_ids"])
    assert ids == RX.subset_ids(s["tierA_ids"]) and len(set(ids)) == 50 and ids == sorted(ids) and set(ids) <= set(s["tierA_ids"])
    fixed = json.loads(RX.SUBSET.read_text())
    assert fixed["ids"] == ids and fixed["seed"] == 20261209, "x9_subset.json does not match the seeded draw"
    d = RX.full_design()
    exp = 100.0
    for k, m in enumerate(RX.ORDER):
        ratio = 1.0 if m.startswith("qwen") else R.LLAMA_TIME_RATIO
        exp += (50 - (3 if k == 0 else 0)) * (3 * 7.0 + 2.7) * ratio + (60.0 if k else 0.0)
    assert math.isclose(RX.projection_h(d, 100.0, 7.0, 2.7, 60.0, 3), exp * 1.05 / 3600, rel_tol=1e-12)
    fit = RX.choose_design(100.0, 7.0, 2.7, 60.0, 3)
    assert fit["pass_"] and fit["applied"] == [] and fit["llama_l0nat"]
    slow = RX.choose_design(100.0, 9.0, 3.0, 60.0, 3)
    assert slow["applied"][:1] == ["drop L0nat in Llama"] and not slow["llama_l0nat"]
    assert [x["design"] for x in slow["projection_steps"]][1:] == slow["applied"]
    worst = RX.choose_design(100.0, 30.0, 3.0, 60.0, 3)
    assert worst["applied"] == RX.REDUCTIONS and not worst["pass_"] and worst["n"] == {m: 40 for m in RX.ORDER}
    assert [t[0] for t in RX.arms_for("llama-base", slow)] == ["NAT", "NAT-K*"] and len(RX.arms_for("qwen-base", slow)) == 3


@t
def x07_analysis_on_synthetic_records(fam):
    S, L_, h = 4, 3, 2
    eK = np.ones((S, L_, h))
    pK = np.full((S, L_, h), 0.1)
    pK[:, 0] = 0.5
    slots = {7: {"eK": eK, "pK": pK, "eV": eK, "pV": pK}, 9: {"eK": 2 * eK, "pK": 2 * pK, "eV": eK, "pV": pK}}
    ids, F, F0, num, den = AX.a_fractions(slots)
    assert ids == [7, 9] and np.allclose(F, (0.5 + 0.1 + 0.1) / 3) and np.allclose(F0, 0.1)
    assert np.allclose(num.sum(0)[:, 0] / den.sum(0)[:, 0], [0.5, 0.1, 0.1])
    rng = np.random.default_rng(0)
    G, ci = AX.share_G([4.0, 4.0, 4.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], rng)
    assert math.isclose(G, 0.75) and np.allclose(ci, [0.75, 0.75])
    G2, _ = AX.share_G([4.0, 5.0, 3.0], [3.0, 2.0, 3.0], [1.0, 1.0, 1.0], rng)
    assert math.isclose(G2, (1 + 3 + 0) / 3 / ((3 + 4 + 2) / 3))
    assert AX.verdict_b(0.75, [0.6, 0.9]) == "supported" and AX.verdict_b(0.75, [0.1, 0.9]) == "inconclusive"
    assert AX.verdict_b(0.45, [0.3, 0.6]) == "inconclusive" and AX.verdict_b(0.05, [-0.1, 0.19]) == "contradicted"
    assert AX.verdict_a({"ci95": [-0.5, -0.1]}, 0.01) == "supported" and AX.verdict_a({"ci95": [-0.5, -0.1]}, 0.2) == "inconclusive"
    assert AX.verdict_a({"ci95": [0.1, 0.5]}, 0.01) == "contradicted" and AX.verdict_a({"ci95": [-0.1, 0.5]}, 0.01) == "inconclusive"
    rec = {"PL_logit": [[1.0, 2.0, 3.0]], "PL_norm": [[1.0, 2.0, 4.0]]}
    assert AX.no_l0(rec, "PL_logit") == [5.0] and AX.no_l0(rec, "PL_norm") == [3.0]
    es = AX.eff_scale({1: {"eK": np.full((2, 1, 1), 4.0), "fK": np.full((2, 1, 1), 10.0)}}, 0.1)
    assert math.isclose(es[0]["median"], 0.5) and es[0]["n_zero_error_slots"] == 0


_BE3 = {}


def be3(fam):
    """3-layer tiny variants: the predictor denominator L*H_q (12, 24) is not a power of two and the layer sum has more than two terms, so
    a different accumulation order of the summed predictor changes its bits (with 2 layers and L*H_q = 8 or 16 it cannot)."""
    if fam not in _BE3:
        from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM
        torch.manual_seed(3)
        if fam == "qwen":
            m = Qwen2ForCausalLM(Qwen2Config(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=4,
                                             num_key_value_heads=2, max_position_embeddings=2048, tie_word_embeddings=True,
                                             initializer_range=0.25, use_sliding_window=False)).eval()
        else:
            m = LlamaForCausalLM(LlamaConfig(vocab_size=128, hidden_size=512, intermediate_size=256, num_hidden_layers=3, num_attention_heads=8,
                                             num_key_value_heads=2, max_position_embeddings=2048, tie_word_embeddings=True,
                                             initializer_range=0.1, rope_theta=500000.0)).eval()
        _BE3[fam] = T.LS.build_backend(m, T.DEV)
    return _BE3[fam]


@t
def x08_records_bitwise_three_layers(fam):
    b = be3(fam)
    ctx = ctx_for(b)
    for arm in ("NAT", "NAT-K*", "QA"):
        r0, _ = arm_rec(b, arm, ctx)
        r1, inst = arm_rec(b, arm, ctx, x9=True)
        d = RX.compare_records(r1, roundtrip(r0))
        assert not d, f"{arm} (3 layers): X9 record differs from the locked one: {list(d)[:4]}"
        per_layer_checks(r1, inst)


# ----------------------------------------------------------------------------- real models (optional)
def real_checks(steps=4):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rep = {}
    for label, mid in R.MODELS.items():
        b = T.LS.build_backend(AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16), "cpu")
        tok = AutoTokenizer.from_pretrained(mid)
        p = tok("Question: Tom has 3 apples and buys 4 more. How many apples does he have?\nAnswer:", return_tensors="pt").input_ids
        ref = R.fp16_history(b, tok, p)[:steps + 1]
        Sr = len(ref) - 1
        cal = json.loads((RV / "v3_runs" / label / "calibration.json").read_text())
        ctx = V.ctx_from_calibration(cal, "cpu")
        out = {}
        for arm in ("NAT", "NAT-K*"):
            r0, _ = arm_rec(b, arm, ctx, prompt=p, ref=ref, S=Sr)
            r1, inst = arm_rec(b, arm, ctx, x9=True, prompt=p, ref=ref, S=Sr)
            d = RX.compare_records(r1, roundtrip(r0))
            try:
                per_layer_checks(r1, inst)
                pl = True
            except AssertionError as e:
                pl = str(e)
            out[arm] = {"bitwise_equal": not d, "diff_fields": list(d), "per_layer_sums": pl}
        r2, inst2 = arm_rec(b, X.X9_ARM, ctx, x9=True, prompt=p, ref=ref, S=Sr)
        out[X.X9_ARM] = {"integrity_errors": R.integrity_errors({X.X9_ARM: r2}), "steps": Sr}
        out["pass"] = bool(all(out[a]["bitwise_equal"] and out[a]["per_layer_sums"] is True for a in ("NAT", "NAT-K*"))
                           and not out[X.X9_ARM]["integrity_errors"])
        rep[label] = out
        del b
    return rep


def run_suite(families=FAMILIES, verbose=False):
    fails = []
    for fn in TESTS:
        for fam in families:
            try:
                fn(fam)
            except Exception as e:  # noqa: BLE001
                fails.append((f"{fn.__name__}[{fam}]", f"{type(e).__name__}: {e}"))
                if verbose:
                    traceback.print_exc()
            finally:
                X.uninstall()
    return fails


def main():
    fails = run_suite(verbose=True)
    n = len(TESTS) * len(FAMILIES)
    print(f"{n - len(fails)}/{n} passed")
    for f in fails:
        print("FAIL", *f)
    rep = {"tiny_tests_total": n, "tiny_tests_failed": [list(f) for f in fails]}
    if "--real" in sys.argv:
        rep["real"] = real_checks()
        print(json.dumps(rep["real"], indent=1, default=str))
    if "--write" in sys.argv:
        (P7 / "x9_cpu_validation.json").write_text(json.dumps(rep, indent=1, default=str))
    ok = not fails and all(v["pass"] for v in rep.get("real", {}).values())
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
