"""CPU interruption/resume test of the LOCKED v3 driver code (imported unchanged). Tiny Qwen2 sized to the real tokenizer, real prompts, 8 steps."""
import json, sys, tempfile, time, hashlib
from pathlib import Path
RV = Path(r"D:\Projects\Dynamic Precision Scaling\entropy-dps-project\research_v2"); sys.path.insert(0, str(RV))
import run_v3 as R
from transformers import AutoTokenizer
import torch, rv3.core as V, analyze_v3 as A
from rv2.backends import build_cpu_backend, tiny_qwen2
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B"); be = build_cpu_backend(tiny_qwen2(0, vocab=151936)); data = R.RS1.Data()
s = R.make_sample(); R.S_STEPS = 8
ids = s["tierA_ids"][:3]; cal_ids = s["calibration_train_ids"][:3]
arms = R.arm_list({"dropped_arms": []})
ctx = V.synthetic_ctx(be.model.config, "cpu"); cap = V.QueryCapture(be.model)
def lines(p): return p.read_bytes().splitlines(keepends=True) if p.exists() else []
def done(m, i): return all(i in R.done_ids(R.mdir(m) / "tierA", tag) for _, _, tag in arms)
def loop(m, step):
    for i in ids:
        if not done(m, i): step(i)
out = {}
def strip(b):
    r = json.loads(b); r.pop("wall_seconds", None); return json.dumps(r, sort_keys=True)
def fresh():
    R.OUT = Path(tempfile.mkdtemp()); R.DESIGN = R.OUT / "design.json"
m = "qwen-base"
fresh(); loop(m, lambda i: R.tier_step(be, tok, data, m, i, arms, ctx, cap))
ref = {p.name: [strip(l) for l in lines(p)] for p in (R.mdir(m) / "tierA").glob("*.jsonl")}
# A: interruption during computation of problem 2
fresh(); orig_rp = R.run_problem
def boom(*a, **k):
    if a[4] == ids[1]: raise KeyboardInterrupt
    return orig_rp(*a, **k)
R.run_problem = boom
try: loop(m, lambda i: R.tier_step(be, tok, data, m, i, arms, ctx, cap))
except KeyboardInterrupt: pass
R.run_problem = orig_rp
before = {p.name: lines(p) for p in (R.mdir(m) / "tierA").glob("*.jsonl")}
loop(m, lambda i: R.tier_step(be, tok, data, m, i, arms, ctx, cap))
after = {p.name: lines(p) for p in (R.mdir(m) / "tierA").glob("*.jsonl")}
out["A_records_before_interruption_byte_identical_after_resume"] = all(after[k][:len(v)] == v for k, v in before.items())
out["A_exactly_one_record_per_problem_per_arm"] = all(len(v) == 3 for v in after.values())
out["A_rerun_equals_uninterrupted_run_except_wall_seconds"] = all([strip(l) for l in after[k]] == ref[k] for k in ref)
# B: interruption while appending problem 2 (5 of 15 arm files written)
fresh(); orig_ap = R.append_records
def boom2(d, recs):
    if any(r.get("index") == ids[1] for r in recs.values()) and "FP16_history" not in recs:
        for k, (tag, r) in enumerate(recs.items()):
            if k == 5: raise KeyboardInterrupt
            orig_ap(d, {tag: r})
    return orig_ap(d, recs)
R.append_records = boom2
try: loop(m, lambda i: R.tier_step(be, tok, data, m, i, arms, ctx, cap))
except KeyboardInterrupt: pass
R.append_records = orig_ap
loop(m, lambda i: R.tier_step(be, tok, data, m, i, arms, ctx, cap))
got = {p.name: [strip(l) for l in lines(p)] for p in (R.mdir(m) / "tierA").glob("*.jsonl")}
out["B_files_with_a_duplicate_record"] = sorted(k for k, v in got.items() if len(v) != 3)
out["B_duplicates_identical_to_each_other_and_to_reference"] = all(set(v) == set(ref[k]) for k, v in got.items())
# C: torn last line
fresh(); loop(m, lambda i: R.tier_step(be, tok, data, m, i, arms, ctx, cap))
p = R.mdir(m) / "tierA" / "NAT.jsonl"; p.write_bytes(p.read_bytes()[:-200])
try:
    loop(m, lambda i: R.tier_step(be, tok, data, m, i, arms, ctx, cap)); out["C_torn_line"] = "resumed silently"
except Exception as e:
    out["C_torn_line"] = f"driver stops: {type(e).__name__}"
# D: calibration interrupted after the accumulator merge, during the record append
fresh(); R.calib_step(be, tok, data, m, cal_ids[0], cap); orig_ap = R.append_records
def boom3(d, recs):
    for k, (tag, r) in enumerate(recs.items()):
        if k == 2: raise KeyboardInterrupt
        orig_ap(d, {tag: r})
R.append_records = boom3
try: R.calib_step(be, tok, data, m, cal_ids[1], cap)
except KeyboardInterrupt: pass
R.append_records = orig_ap
for i in cal_ids:
    if not R.calib_done(m, i): R.calib_step(be, tok, data, m, i, cap)
aux = torch.load(R.aux_path(m))
out["D_calib_lines_per_arm"] = sorted({len(lines(R.mdir(m) / "calib" / f"{t}.jsonl")) for _, _, t in V.CAL_ARMS})
out["D_accumulator_counts_each_problem_once"] = aux["included"] == cal_ids and aux["n_meas"] == sum(len(A.load_jsonl(R.mdir(m) / "calib" / "NAT.jsonl")[i]["js"]) for i in cal_ids)
# E: ledger across sessions
fresh(); RS1 = R.RS1; RS1.OUT = R.OUT; RS1.LEDGER = R.OUT / "gpu_ledger.jsonl"; RS1.PROGRESS = R.OUT / "current_process.json"
with open(RS1.LEDGER, "a") as f: f.write(json.dumps({"seconds": 3600.0, "block": "s1", "completed": True}) + chr(10))
with open(RS1.LEDGER, "a") as f: f.write(json.dumps({"seconds": 120.0, "block": "gpu_mutation_check", "completed": True}) + chr(10))
RS1.write_progress("tierA:qwen-base", time.time() - 1800)
out["E_recovered_seconds_charged"] = round(RS1.recover_open_process(RS1.PROGRESS, RS1.LEDGER))
out["E_cumulative_hours_(1.0+0.0333+0.5)"] = round(RS1.ledger_seconds() / 3600, 4)
print(json.dumps(out, indent=1))
