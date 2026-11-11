"""Exercise the GPU-only code paths of run_e1l.py (gpu_gates_l, calibration step, Tier A step incl. the in-session FP16 reference) on CPU with tiny settings and the REAL
Llama-3.2-1B-Instruct checkpoint, to catch plumbing bugs before any GPU time is spent. Not a validation of numerics.
  venv/Scripts/python.exe research_v2/tests/cpu_harness_e1l_paths.py"""
import json
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import run_e1l as R  # noqa: E402
import e1l_support as LS  # noqa: E402
import run_stage1 as RS1  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(LS.MODEL)
be = LS.real_backend_cpu()
data = RS1.Data()
RS1.MAX_NEW_TOKENS = 6
R.S_STEPS = 4
s = R.make_sample()
s = {**s, "gate_train_ids": s["gate_train_ids"][:1], "gate_numerics_train_ids": s["gate_numerics_train_ids"][:1], "gate_test_ids_for_G5": s["gate_test_ids_for_G5"][:1],
     "tierA_ids": s["tierA_ids"][:1], "calibration_train_ids": s["calibration_train_ids"][:2]}
tmp = Path(tempfile.mkdtemp())
R.OUT = tmp
ok = R.gpu_gates_l(be, tok, None, 0.0, time.time(), data, s)
rep = json.loads((tmp / "gates" / "gates.json").read_text())
print({k: (v.get("pass", v.get("all_identical")) if isinstance(v, dict) else v) for k, v in rep.items() if k.startswith("G")})
print("geometry:", rep["G7_cache_geometry"])
recs = R.E.run_tier_a_problem(be, tok, data, s["calibration_train_ids"][0], "train", R.CAL_ARMS, None, 4, "Lcalib")
print("calib arms", list(recs), R.E.integrity_errors_a(recs))
ref = R.fp16_reference(be, tok, data, s["tierA_ids"][0])
recs = R.E.run_tier_a_problem(be, tok, data, s["tierA_ids"][0], "test", R.TIER_A_ARMS + [("C1", 0.9, "C1star")], {s["tierA_ids"][0]: ref}, 4, "LA")
print("tierA arms", list(recs), R.E.integrity_errors_a(recs), "ref len", len(ref))
R.E.append_records(tmp / "tierA", recs)
R.E.append_records(tmp / "tierA", {"FP16_ref": {"index": s["tierA_ids"][0], "ids": ref}})
print("harness complete; outputs in", tmp)
