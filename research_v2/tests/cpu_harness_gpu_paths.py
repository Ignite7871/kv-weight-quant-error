"""Exercise the GPU-only code paths (run_pilot2.pilot, run_controls.gpu_gates / calibrate / tierA / tierB steps) on CPU with tiny settings, to catch
plumbing bugs before any GPU time is spent. Real Qwen2.5-0.5B on CPU; not a validation of numerics.  Run: venv/Scripts/python.exe research_v2/tests/cpu_harness_gpu_paths.py"""
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import run_pilot2 as P
import run_controls as E
import run_stage1 as RS1
from rv2.backends import build_cpu_backend, real_model_cpu
from transformers import AutoTokenizer
import torch

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
be = build_cpu_backend(real_model_cpu())
data = RS1.Data()
RS1.MAX_NEW_TOKENS = 6
from rv2.backends import weight_bytes_per_pass
wb = weight_bytes_per_pass(be.model)
tmp = Path(tempfile.mkdtemp())

# ---- pilot() with tiny settings
P.EQUIV_TOKENS = 4
s = {"equiv_ids": P.make_sample()["train_ids"][:1], "train_ids": P.make_sample()["train_ids"][:2]}
import time
ok = P.pilot(be, tok, wb, 0.0, time.time(), "llama32_1b_instruct", data, s, tmp / "pilot")
print("pilot() returned", ok, "(gates may fail on a 2-problem CPU run; the point is that it runs and writes a report)")
rep = json.loads((tmp / "pilot" / "pilot_report.json").read_text())
print({k: v.get("pass") for k, v in rep["gates"].items()})

# ---- gpu_gates() with tiny settings
E.S_STEPS, E.N_GATE = 4, 1
es = E.make_sample()
es = {**es, "gate_train_ids": es["gate_train_ids"][:2], "gate_test_ids_for_G5": es["gate_test_ids_for_G5"][:1], "tierA_ids": es["tierA_ids"][:1], "tierB_ids": es["tierB_ids"][:2]}
E.OUT = tmp / "e1"
fp16 = E.load_fp16_ids()
r = E.gpu_gates(be, tok, wb, 0.0, time.time(), data, es, fp16)
print("gpu_gates() returned", r)

# ---- calibration / tier A / tier B step functions
recs = E.run_tier_a_problem(be, tok, data, es["gate_train_ids"][0], "train", E.CAL_ARMS, None, 4, "calib")
print("calib arms:", list(recs), "integrity:", E.integrity_errors_a(recs))
recs = E.run_tier_a_problem(be, tok, data, es["tierA_ids"][0], "test", E.TIER_A_ARMS + [("C1", 1.7, "C1star")], fp16, 4, "A")
print("tierA arms:", list(recs), "integrity:", E.integrity_errors_a(recs))
E.append_records(tmp / "e1" / "tierA", recs)
for arm in E.TIER_B_ARMS:
    E.append_records(tmp / "e1" / "tierB", {arm: E.run_tier_b_one(be, tok, data, es["tierB_ids"][0], arm)})
print("harness complete; outputs in", tmp)
