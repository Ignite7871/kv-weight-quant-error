"""Llama-3.2-1B-Instruct E1 Tier A replication driver (E1L_SPEC.md). Generation, analysis and integrity code are the locked Qwen E1 modules, imported unchanged;
this file only sets the model, samples, folders, ledger (absolute 2.5 h cap) and the Llama gates. No Tier B exists in this driver.

Subcommands (CPU unless marked)
  plan         write e1l_sample.json (ids copied from the locked Qwen E1 sample; nothing new is drawn)
  ledger       cumulative Llama-E1 GPU time
  gates        [GPU] G1 zero perturbation, G2/G3 numerics + token-history consistency, G4 determinism, G5 NAT vs in-session engine cache-only, G6 runtime projection, G7 cache geometry
  calibrate    [GPU] 100 train problems: NAT + C1 at alpha 0.5/1/2/4  (then: analyze_e1l.py calibrate; lock_e1l.py --append-calibration)
  tierA        [GPU] 200 test problems, teacher-forced, 6 arms
GPU subcommands are REFUSED unless: --authorized-gpu-run; research_v2/e1l_runs/AUTHORIZED_BY_AUTHORS_LLAMA.txt exists, is non-empty and contains the word "Llama"
(created only by the corresponding author after reviewing the Llama lock); the Llama lock AND the Qwen E1 lock verify; gates passed (calibrate/tierA);
calibration frozen (tierA). The 2.5 h cap is absolute; a run cut by it is TRUNCATED, never extended; no flag raises it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_controls as E                        # noqa: E402  (locked Qwen E1 module, unchanged)
import e1l_support as LS                        # noqa: E402

import torch                                    # noqa: E402

import rv2.controls as C                        # noqa: E402
from rv2.boundary import would_stop_at_boundary  # noqa: E402
from rv2.engine import generate                 # noqa: E402
from rv2.policies import StaticFP16             # noqa: E402
from rv2.stage1 import CACHE_ONLY, generate_cell  # noqa: E402

RS1 = E.RS1
OUT = HERE / "e1l_runs"
SAMPLE_FILE = HERE / "e1l_sample.json"
AUTH = OUT / "AUTHORIZED_BY_AUTHORS_LLAMA.txt"
CAP_HOURS = 2.5
S_STEPS = 64
N_GATE_TRAIN, N_GATE_NUM, N_GATE_G5 = 10, 6, 8
TIER_A_ARMS, CAL_ARMS = E.TIER_A_ARMS, E.CAL_ARMS


def make_sample() -> dict:
    if SAMPLE_FILE.exists():
        return json.loads(SAMPLE_FILE.read_text())
    q = json.loads((HERE / "e1_sample.json").read_text())
    s = {"model": LS.MODEL, "source": "ids copied from the locked Qwen E1 sample (e1_sample.json); no new draw, no outcome data",
         "qwen_e1_sample_sha256": hashlib.sha256((HERE / "e1_sample.json").read_bytes().replace(bytes([13, 10]), bytes([10]))).hexdigest(),
         "tierA_ids": q["tierA_ids"], "calibration_train_ids": q["calibration_train_ids"], "gate_train_ids": q["calibration_train_ids"][:N_GATE_TRAIN],
         "gate_numerics_train_ids": q["calibration_train_ids"][:N_GATE_NUM], "gate_test_ids_for_G5": q["tierA_ids"][:N_GATE_G5],
         "exposed_test_ids_in_sample": sorted(set(q["tierA_ids"]) & set(q["exposed_test_ids_in_sample"])),
         "seed_tier_labels": {"calibration": "Lcalib", "tierA": "LA", "gates": "Lgate"},
         "note": "Llama has no stored FP16 trajectories: the shared teacher-forced token history of each test problem is its in-session FP16 greedy generation (<= 65 tokens, drift-marker stop)."}
    SAMPLE_FILE.write_text(json.dumps(s, indent=1))
    return s


def lock_ok() -> bool:
    ok1 = subprocess.run([sys.executable, str(HERE / "tools" / "lock_e1l.py"), "--verify"], capture_output=True).returncode == 0
    ok2 = subprocess.run([sys.executable, str(HERE / "tools" / "lock_e1.py"), "--verify"], capture_output=True).returncode == 0
    return ok1 and ok2


def require_gpu_permission(a, need_gates=False, need_calibration=False):
    if not a.authorized_gpu_run:
        sys.exit("Refusing to run: GPU runs need explicit author authorization (--authorized-gpu-run).")
    if not AUTH.exists() or "Llama" not in AUTH.read_text(encoding="utf-8", errors="replace"):
        sys.exit("Refusing to run: e1l_runs/AUTHORIZED_BY_AUTHORS_LLAMA.txt must exist and contain the word 'Llama'. Only the corresponding author creates it, after reviewing the Llama lock.")
    if not lock_ok():
        sys.exit("Refusing to run: the Llama lock and/or the Qwen E1 lock do not verify.")
    if need_gates:
        g = OUT / "gates" / "gates.json"
        if not g.exists() or not json.loads(g.read_text()).get("all_passed"):
            sys.exit("Refusing to run: the Llama gates have not passed.")
    if need_calibration and not (OUT / "e1l_calibration.json").exists():
        sys.exit("Refusing to run: calibration constants are not frozen (e1l_runs/e1l_calibration.json).")


def configure():
    OUT.mkdir(parents=True, exist_ok=True)
    RS1.OUT, RS1.LEDGER, RS1.PROGRESS = OUT, OUT / "gpu_ledger.jsonl", OUT / "current_process.json"
    RS1.MODEL = LS.MODEL


def fp16_reference(be, tok, data, idx):
    _, prompt = data.problem("test", idx)
    ids_t = tok(prompt, return_tensors="pt").input_ids.to(be.device)
    res = generate(be, ids_t, StaticFP16(), S_STEPS + 1, tok.eos_token_id, 0, early_stop_fn=lambda ids: would_stop_at_boundary(tok, ids), instrument=False)
    return res.ids


def gpu_gates_l(be, tok, wb, prior, t0, data, s):
    rep = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "model": LS.MODEL}
    # G7 cache geometry as it appears in the real cache
    _, prompt = data.problem("train", s["gate_train_ids"][0])
    ids_t = tok(prompt, return_tensors="pt").input_ids.to(be.device)
    geo = LS.cache_geometry(be, ids_t)
    rep["G7_cache_geometry"] = {**geo, "pass": bool(geo["n_layers"] == 16 and geo["n_kv_heads"] == 8 and geo["head_dim"] == 64 and geo["all_fp16"])}
    # G1 alpha = 0 bitwise (tokens + cache)
    g1 = []
    for idx in s["gate_train_ids"]:
        _, prompt = data.problem("train", idx)
        ids_t = tok(prompt, return_tensors="pt").input_ids.to(be.device)
        cap = {}
        orig = be.run

        def run(i, c, use_4bit, orig=orig, cap=cap):
            o = orig(i, c, use_4bit)
            cap["c"] = o[1]
            return o
        be.run = run
        try:
            ref = generate(be, ids_t, StaticFP16(), S_STEPS + 1, tok.eos_token_id, 0, instrument=False)
            c_ref = [(l.keys.clone(), l.values.clone()) for l in cap["c"].layers]
            for arm in ("C1", "C2"):
                z = C.generate_control(be, ids_t, arm, 0.0, 1, S_STEPS + 1, tok.eos_token_id)
                c_z = [(l.keys.clone(), l.values.clone()) for l in cap["c"].layers]
                g1.append(bool(z.ids == ref.ids and all(torch.equal(a, b) and torch.equal(x, y) for (a, x), (b, y) in zip(c_ref, c_z))))
        finally:
            be.run = orig
    rep["G1_alpha0_bitwise"] = {"n": len(g1), "all_identical": all(g1)}
    # G2/G3/G4 numerics, integrity (incl. step-1 JS exactly 0 = token-history/cache-state consistency), determinism
    g2, g4 = [], []
    for idx in s["gate_numerics_train_ids"]:
        recs = E.run_tier_a_problem(be, tok, data, idx, "train", TIER_A_ARMS + [("C1", 0.91, "C1star")], None, S_STEPS, "Lgate")
        g2.append(E.integrity_errors_a(recs))
        _, prompt = data.problem("train", idx)
        ids_t = tok(prompt, return_tensors="pt").input_ids.to(be.device)
        a1 = C.generate_control(be, ids_t, "C1", 2.0, C.seed_for("C1", 2.0, idx, "Lgate"), 40, tok.eos_token_id).ids
        a2 = C.generate_control(be, ids_t, "C1", 2.0, C.seed_for("C1", 2.0, idx, "Lgate"), 40, tok.eos_token_id).ids
        a3 = C.generate_control(be, ids_t, "C1", 2.0, C.seed_for("C1", 2.0, idx, "Lgate-other"), 40, tok.eos_token_id).ids
        g4.append((a1 == a2, a1 != a3))
    rep["G2_G3_integrity_errors"] = [e for e in g2 if e]
    rep["G4_determinism"] = {"same_seed_identical": all(x for x, _ in g4), "different_seed_differs_in_n_of": [sum(y for _, y in g4), len(g4)]}
    # G5 NAT through the new path equals the engine's cache-only generation (in-session; no stored Llama records exist)
    g5 = []
    for idx in s["gate_test_ids_for_G5"]:
        _, prompt = data.problem("test", idx)
        ids_t = tok(prompt, return_tensors="pt").input_ids.to(be.device)
        stop = lambda ids: would_stop_at_boundary(tok, ids)       # noqa: E731
        e = generate_cell(be, ids_t, CACHE_ONLY, RS1.MAX_NEW_TOKENS, tok.eos_token_id, early_stop_fn=stop)
        n = C.generate_control(be, ids_t, "NAT", 1.0, 1, RS1.MAX_NEW_TOKENS, tok.eos_token_id, early_stop_fn=stop)
        g5.append(bool(e.ids == n.ids))
    rep["G5_nat_equals_engine_cache_only"] = {"n": len(g5), "identical": int(sum(g5)), "pass": all(g5)}
    # G6 runtime projection from measured per-problem times (3 test-type + 2 calibration-type problems), against the remaining ceiling
    tA = []
    for idx in s["tierA_ids"][:3]:
        t = time.perf_counter()
        ref = fp16_reference(be, tok, data, idx)
        E.run_tier_a_problem(be, tok, data, idx, "test", TIER_A_ARMS + [("C1", 0.91, "C1star")], {idx: ref}, S_STEPS, "Lgate")
        tA.append(time.perf_counter() - t)
    tC = []
    for idx in s["calibration_train_ids"][-2:]:
        t = time.perf_counter()
        E.run_tier_a_problem(be, tok, data, idx, "train", CAL_ARMS, None, S_STEPS, "Lgate")
        tC.append(time.perf_counter() - t)
    used_h = (prior + time.time() - t0) / 3600
    proj = used_h + (100 * (sum(tC) / len(tC)) + 200 * (sum(tA) / len(tA))) / 3600 * 1.05 + 0.03     # 5 % margin, process-start overhead
    rep["G6_runtime"] = {"tierA_problem_s_mean": sum(tA) / len(tA), "calibration_problem_s_mean": sum(tC) / len(tC), "used_h_after_gates": used_h,
                         "projected_total_h": proj, "ceiling_h": CAP_HOURS, "pass": bool(proj <= CAP_HOURS)}
    rep["all_passed"] = bool(rep["G7_cache_geometry"]["pass"] and rep["G1_alpha0_bitwise"]["all_identical"] and not rep["G2_G3_integrity_errors"]
                             and rep["G4_determinism"]["same_seed_identical"] and rep["G5_nat_equals_engine_cache_only"]["pass"] and rep["G6_runtime"]["pass"])
    (OUT / "gates").mkdir(parents=True, exist_ok=True)
    (OUT / "gates" / "gates.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep, indent=1, default=float))
    return rep["all_passed"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "ledger", "gates", "calibrate", "tierA"])
    ap.add_argument("--authorized-gpu-run", action="store_true")
    ap.add_argument("--cap-hours", type=float, default=CAP_HOURS)
    a = ap.parse_args()
    if a.cap_hours > CAP_HOURS:
        sys.exit(f"cap above the approved {CAP_HOURS} h is not permitted")
    configure()
    if a.cmd == "plan":
        s = make_sample()
        print({k: (len(v) if isinstance(v, list) else v) for k, v in s.items() if k not in ("note", "source")})
        return
    if a.cmd == "ledger":
        print(f"Llama E1 cumulative GPU wall time: {RS1.ledger_seconds() / 3600:.3f} h of {a.cap_hours} h")
        return
    require_gpu_permission(a, need_gates=a.cmd in ("calibrate", "tierA"), need_calibration=a.cmd == "tierA")
    s = make_sample()
    data = RS1.Data()
    if a.cmd == "gates":
        RS1.gpu_process("gates", lambda be, dyn, tok, wb, prior, t0: gpu_gates_l(be, tok, wb, prior, t0, data, s), a.cap_hours)
        return
    if a.cmd == "calibrate":
        d = OUT / "calib"

        def step(be, tok, idx):
            recs = E.run_tier_a_problem(be, tok, data, idx, "train", CAL_ARMS, None, S_STEPS, "Lcalib")
            errs = E.integrity_errors_a(recs)
            if errs:
                (OUT / "integrity_failure.json").write_text(json.dumps({"index": idx, "errors": errs}))
                print("[INTEGRITY FAILURE]", errs[:3], flush=True)
                return False
            E.append_records(d, recs)
            return True
        RS1.gpu_process("calibrate", lambda be, dyn, tok, wb, prior, t0: E.block_loop(
            "calib", be, tok, wb, prior, t0, a.cap_hours, s["calibration_train_ids"], lambda i: step(be, tok, i),
            lambda i: all(i in E.done_ids(d, tag) for _, _, tag in CAL_ARMS)), a.cap_hours)
        return
    calib = json.loads((OUT / "e1l_calibration.json").read_text())
    arms = TIER_A_ARMS + [("C1", float(calib["alpha_star"]), "C1star")]
    d = OUT / "tierA"

    def stepA(be, tok, idx):
        ref = fp16_reference(be, tok, data, idx)
        recs = E.run_tier_a_problem(be, tok, data, idx, "test", arms, {idx: ref}, S_STEPS, "LA")
        errs = E.integrity_errors_a(recs)
        if errs:
            (OUT / "integrity_failure.json").write_text(json.dumps({"index": idx, "errors": errs}))
            print("[INTEGRITY FAILURE]", errs[:3], flush=True)
            return False
        E.append_records(d, recs)
        E.append_records(d, {"FP16_ref": {"index": idx, "ids": ref}})
        return True
    RS1.gpu_process("tierA", lambda be, dyn, tok, wb, prior, t0: E.block_loop(
        "tierA", be, tok, wb, prior, t0, a.cap_hours, s["tierA_ids"], lambda i: stepA(be, tok, i),
        lambda i: all(i in E.done_ids(d, tag) for _, _, tag in arms)), a.cap_hours)


if __name__ == "__main__":
    main()
