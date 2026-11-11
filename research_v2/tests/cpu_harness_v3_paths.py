"""CPU harness for the GPU-only code paths of run_v3.py and analyze_v3.py: the driver's own gate, calibration, G6, Tier A and analysis functions are
executed end to end on CPU with a tiny random Qwen2 sized to the real tokenizer, real GSM8K prompts, 12 steps and a temporary output folder.
Everything the GPU run would call is called (except model loading and the ledger/watchdog wrapper). G7 is expected to FAIL here (tiny geometry).
  venv/Scripts/python.exe research_v2/tests/cpu_harness_v3_paths.py [--write]
"""
import json
import sys
import tempfile
import time
from pathlib import Path

RV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RV))

import run_v3 as R                        # noqa: E402
from transformers import AutoTokenizer    # noqa: E402

import analyze_v3 as A                    # noqa: E402
import torch                              # noqa: E402
import rv3.core as V                      # noqa: E402
from rv2.backends import build_cpu_backend, tiny_qwen2  # noqa: E402


def main():
    t_start = time.time()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    be = build_cpu_backend(tiny_qwen2(0, vocab=151936))
    data = R.RS1.Data()
    s0 = R.make_sample()
    s = {**s0, "gate_alpha0_train_ids": s0["calibration_train_ids"][:2], "gate_numerics_train_ids": s0["calibration_train_ids"][:2],
         "gate_test_ids_for_G5": s0["tierA_ids"][:2], "g6_train_ids": s0["calibration_train_ids"][-2:]}
    cal_ids, test_ids = s0["calibration_train_ids"][:3], s0["tierA_ids"][:3]
    R.S_STEPS = 12
    A.B_BOOT, A.B_PERM = 500, 500
    rep = {}
    with tempfile.TemporaryDirectory() as tmp:
        R.OUT = Path(tmp)
        R.DESIGN = R.OUT / "design.json"
        m0 = "qwen-base"
        R.mdir(m0).mkdir(parents=True)
        R.gpu_gates(be, tok, m0, 0.0, time.time(), data, s)
        g = json.loads((R.mdir(m0) / "gates" / "gates.json").read_text())
        rep["gates"] = {"G7_expected_fail_tiny": not g["G7_cache_geometry"]["pass"], "G1": g["G1_alpha0_bitwise"]["all_identical"],
                        "G2_G3_errors": g["G2_G3_integrity_errors"], "G2b_leak": g["G2b_max_leak"], "G2b_U": g["G2b_U_orthonormal_err"],
                        "G4": g["G4_determinism"], "G5": g["G5_nat_equals_engine_cache_only"], "G8": g["G8_routing"]}
        cals = {}
        for m in R.ORDER:
            R.mdir(m).mkdir(parents=True, exist_ok=True)
            cap = V.QueryCapture(be.model)
            for i in cal_ids:
                assert R.calib_step(be, tok, data, m, i, cap), "calibration step failed"
                assert R.calib_done(m, i)
            assert R.calib_step(be, tok, data, m, cal_ids[0], cap) and len(A.load_jsonl(R.mdir(m) / "calib" / "NAT.jsonl")) == 3, "resume must not duplicate"
            cap.remove()
            d = R.mdir(m) / "calib"
            recs = {t: A.load_jsonl(d / f"{t}.jsonl") for t in ("NAT", "C1_a0.5", "C1_a1", "C1_a2")}
            aux = {k: (v.numpy() if hasattr(v, "numpy") else v) for k, v in torch.load(d / "aux_state.pt").items()}
            assert aux["included"] == cal_ids, aux["included"]
            cals[m] = A.calibrate_from(recs, aux)
            (R.mdir(m) / "calibration.json").write_text(json.dumps(cals[m]))
        rep["calibration_keys_ok"] = all(k in cals[m0] for k in ("U8", "sigma2", "outliers", "rho_star", "scale_factor_K", "HD_descriptive", "margin_log_js", "E8", "r80"))
        assert R.g6(be, tok, m0, 0.0, time.time(), data, s) is not None
        rep["g6_first_model"] = json.loads((R.mdir(m0) / "g6.json").read_text())
        design = json.loads(R.DESIGN.read_text())
        rep["design"] = {k: design[k] for k in ("dropped_arms", "n_test")}
        arms = R.arm_list(design)
        for m in R.ORDER:
            ctx = V.ctx_from_calibration(cals[m], "cpu")
            cap = V.QueryCapture(be.model)
            for i in test_ids:
                assert R.tier_step(be, tok, data, m, i, arms, ctx, cap, {}), f"tier step failed {m} {i}"
            cap.remove()
            R.mark_complete(m, test_ids, arms, True)
            assert (R.mdir(m) / "tierA_complete.json").exists()
        rep["g6_later_model"] = R.g6(be, tok, "llama-instruct", 0.0, time.time(), data, s)
        dat = {m: {a: A.load_jsonl(R.mdir(m) / "tierA" / f"{tag}.jsonl") for a, _, tag in arms} for m in R.ORDER}
        rep["record_files"] = sorted(p.name for p in (R.mdir(m0) / "tierA").glob("*.jsonl"))
        out = A.analyze_all(dat, cals, set())
        md = A.to_markdown(out)
        rep["analysis_sections"] = {"HA": all("HA" in out["per_model"][m] for m in R.ORDER), "HF": "contrasts" in out["HF"],
                                    "HC_cross": "qwen-base vs llama-instruct" in out["HC_cross"], "markdown_lines": len(md.splitlines())}
    gates_ok = (rep["gates"]["G7_expected_fail_tiny"] and rep["gates"]["G1"] and not rep["gates"]["G2_G3_errors"] and rep["gates"]["G2b_leak"] <= 1e-5
                and rep["gates"]["G2b_U"] <= 1e-6 and rep["gates"]["G4"]["same_seed_identical"] and rep["gates"]["G5"]["pass"] and rep["gates"]["G8"]["pass"])
    rep["pass"] = bool(gates_ok and rep["calibration_keys_ok"] and rep["design"]["n_test"] == 200 and all(rep["analysis_sections"][k] for k in ("HA", "HF", "HC_cross")))
    rep["seconds"] = round(time.time() - t_start, 1)
    print(json.dumps(rep, indent=1, default=str))
    if "--write" in sys.argv:
        (RV / "validation_logs" / "v3_cpu_harness.json").write_text(json.dumps(rep, indent=1, default=str))
    sys.exit(0 if rep["pass"] else 1)


if __name__ == "__main__":
    main()
