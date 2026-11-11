"""E1 driver (E1_SPEC.md): perturbation-matched cache controls. The generation code is rv2/controls.py on the VERIFIED engine; GPU ledger, watchdog,
telemetry and hardware stops are the Stage 1 machinery (run_stage1.gpu_process, stage1_runtime) imported unchanged.

Subcommands (CPU unless marked)
  plan         write e1_sample.json (seeded orders; no outcome data used) and print a summary
  cpu-smoke    end-to-end I/O check on real Qwen2.5-0.5B on CPU (a few tokens): records, schema, analysis pipeline
  ledger       print cumulative E1 GPU time
  gates        [GPU] G1 alpha=0 bitwise, G2/G3 numerics, G4 determinism, G5 NAT vs stored cache-only, G6 runtime projection
  calibrate    [GPU] calibration problems (train split): NAT + C1 at alpha 0.5/1/2/4; then `analyze_controls.py calibrate`
  tierA        [GPU] teacher-forced Tier A on the test sample
  tierB        [GPU] free-running Tier B (FP16, NAT, C1, C2)

GPU subcommands are REFUSED unless ALL of: --authorized-gpu-run; e1_runs/AUTHORIZED_BY_AUTHORS.txt exists (created only by the authors); the
protocol lock verifies (tools/lock_e1.py --verify); gates passed (for calibrate/tierA/tierB); calibration frozen (for tierA/tierB). The cap is
ABSOLUTE (8.0 h cumulative for everything E1 does on the GPU) and there is no flag to raise it; a run cut short by it is TRUNCATED, never extended.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_stage1 as RS1                       # noqa: E402  (HF datasets / transformers before torch)

import torch                                   # noqa: E402

import rv2.controls as C                       # noqa: E402
from rv2.boundary import find_boundary, score, would_stop_at_boundary   # noqa: E402
from rv2.engine import generate                # noqa: E402
from rv2.policies import StaticFP16            # noqa: E402

OUT = HERE / "e1_runs"
SAMPLE_FILE = HERE / "e1_sample.json"
CAP_HOURS = 8.0
S_STEPS = 64
N_A, N_B, N_CAL, N_GATE = 200, 400, 100, 20
TEST_ORDER_SEED = "rv2-E1-test-order-2026-10-27"
CAL_SEED = "rv2-E1-calibration-train-2026-10-27"
FEWSHOT = {0, 1, 2}                            # train problems used inside every prompt
TIER_A_ARMS = [("NAT", 1.0, "NAT"), ("C1", 1.0, "C1_a1.0"), ("C2", 1.0, "C2_a1.0"), ("C1", 0.5, "C1_a0.5"), ("C1", 2.0, "C1_a2.0")]
CAL_ARMS = [("NAT", 1.0, "NAT"), ("C1", 0.5, "C1_a0.5"), ("C1", 1.0, "C1_a1"), ("C1", 2.0, "C1_a2"), ("C1", 4.0, "C1_a4")]
TIER_B_ARMS = ["FP16", "NAT", "C1", "C2"]
STAGE_FP16 = [HERE / "stage1_runs" / "eval" / "fp16.jsonl", HERE / "stage2_runs" / "eval" / "fp16.jsonl"]


# ----------------------------------------------------------------------------- sample (no outcome data enters)
def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def make_sample() -> dict:
    if SAMPLE_FILE.exists():
        return json.loads(SAMPLE_FILE.read_text())
    s1 = json.loads((HERE / "stage1_sample.json").read_text())
    s2 = json.loads((HERE / "stage2_sample.json").read_text())
    pilot = json.loads((HERE / "pilot_sample.json").read_text())
    n_test = s1["n_test"]
    order = random.Random(TEST_ORDER_SEED).sample(range(n_test), n_test)           # seeded permutation: prefixes are random subsets
    tier_b, tier_a = order[:N_B], order[:N_A]
    used_train = set(FEWSHOT)
    for k in ("equiv_train", "feasibility_train", "speed_train"):
        used_train |= set(s1.get(k, []))
    for k in ("train_equiv", "train_calib"):
        used_train |= set(pilot.get(k, []))
    pool = [i for i in range(7473) if i not in used_train]                          # GSM8K train split has 7,473 problems
    cal = random.Random(CAL_SEED).sample(pool, N_CAL)
    exposed = set(s2["exposed_gold_trajectory_test_0_199"])
    s = {"test_order_seed": TEST_ORDER_SEED, "calibration_seed": CAL_SEED, "n_test": n_test,
         "tierB_ids": tier_b, "tierA_ids": tier_a, "calibration_train_ids": cal, "gate_train_ids": cal[:N_GATE],
         "gate_test_ids_for_G5": tier_a[:24], "exposed_test_ids_in_sample": sorted(set(tier_b) & exposed),
         "excluded_train_ids": sorted(used_train),
         "fp16_reference_files_sha256": {p.relative_to(HERE).as_posix(): sha256_file(p) for p in STAGE_FP16 if p.exists()},
         "note": "Tier A reuses the stored FP16 token sequences of Stage 1/2 only as the shared teacher-forced history; the FP16 reference distributions are recomputed in-session."}
    SAMPLE_FILE.write_text(json.dumps(s, indent=1))
    return s


def load_fp16_ids() -> dict:
    out = {}
    for p in STAGE_FP16:
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["index"]] = r["ids"]
    return out


# ----------------------------------------------------------------------------- guards
def lock_ok() -> bool:
    return subprocess.run([sys.executable, str(HERE / "tools" / "lock_e1.py"), "--verify"], capture_output=True).returncode == 0


def require_gpu_permission(a, need_gates=False, need_calibration=False):
    if not a.authorized_gpu_run:
        sys.exit("Refusing to run: GPU runs need explicit author authorization (--authorized-gpu-run).")
    if not (OUT / "AUTHORIZED_BY_AUTHORS.txt").exists():
        sys.exit("Refusing to run: e1_runs/AUTHORIZED_BY_AUTHORS.txt does not exist. Only the authors create it, after approving the E1 GPU evaluation.")
    if not lock_ok():
        sys.exit("Refusing to run: the E1 protocol lock does not verify (tools/lock_e1.py --verify).")
    if need_gates:
        g = OUT / "gates" / "gates.json"
        if not g.exists() or not json.loads(g.read_text()).get("all_passed"):
            sys.exit("Refusing to run: the E1 gates have not passed.")
    if need_calibration and not (OUT / "e1_calibration.json").exists():
        sys.exit("Refusing to run: calibration constants are not frozen (e1_runs/e1_calibration.json).")


def configure():
    OUT.mkdir(parents=True, exist_ok=True)
    RS1.OUT = OUT
    RS1.LEDGER = OUT / "gpu_ledger.jsonl"
    RS1.PROGRESS = OUT / "current_process.json"


# ----------------------------------------------------------------------------- record builders
def _rec_base(arm_tag, alpha, idx, split):
    return {"index": idx, "split": split, "arm": arm_tag, "alpha": alpha}


def run_tier_a_problem(be, tok, data, idx, split, arms, fp16_ids, S, tier_label, wb=None):
    prob, prompt = data.problem(split, idx)
    ids_t = tok(prompt, return_tensors="pt").input_ids.to(be.device)
    ref = fp16_ids[idx] if split == "test" else None
    if ref is None:                                                              # calibration (train) problems: generate the FP16 reference in-session
        res = generate(be, ids_t, StaticFP16(), S + 1, tok.eos_token_id, 0, early_stop_fn=lambda ids: would_stop_at_boundary(tok, ids), instrument=False)
        ref = res.ids
    S_i = min(S, len(ref) - 1)
    rl = C.reference_logits(be, ids_t, ref, S_i)
    recs = {}
    for arm, alpha, tag in arms:
        t0 = time.perf_counter()
        out = C.teacher_forced_arm(be, ids_t, ref, S_i, arm, alpha, C.seed_for(arm, alpha, idx, tier_label), rl)
        recs[tag] = {**_rec_base(tag, alpha, idx, split), "S": S_i, "ref_len": len(ref), **out, "wall_seconds": time.perf_counter() - t0}
    return recs


def run_tier_b_one(be, tok, data, idx, arm, ref_ids=None):
    prob, prompt = data.problem("test", idx)
    ids_t = tok(prompt, return_tensors="pt").input_ids.to(be.device)
    stop = lambda ids: would_stop_at_boundary(tok, ids)       # noqa: E731
    t0 = time.perf_counter()
    if arm == "FP16":
        res = generate(be, ids_t, StaticFP16(), RS1.MAX_NEW_TOKENS, tok.eos_token_id, 0, early_stop_fn=stop, instrument=False)
        diag = None
    else:
        res = C.generate_control(be, ids_t, arm, 1.0, C.seed_for(arm, 1.0, idx, "B"), RS1.MAX_NEW_TOKENS, tok.eos_token_id, early_stop_fn=stop)
        diag = res.trace["diag"]
    wall = time.perf_counter() - t0
    b = find_boundary(tok, res.ids, tok.eos_token_id, RS1.MAX_NEW_TOKENS)
    text = tok.decode(res.ids, skip_special_tokens=True)
    return {"index": idx, "split": "test", "arm": arm, **score(prob, text, b), "ids": res.ids, "text": text, "steps": len(res.ids) - 1,
            "wall_seconds": wall, "diag": diag}


def integrity_errors_a(recs):
    errs = []
    for tag, r in recs.items():
        js = r["js"]
        if not all(x == x and abs(x) < 1e3 for x in js):
            errs.append(f"{tag}: non-finite JS")
        if js and js[0] != 0.0:
            errs.append(f"{tag}: step-1 JS is {js[0]!r}, must be exactly 0 (no perturbed entry exists yet)")
        if min(js) < 0:
            errs.append(f"{tag}: negative JS")
        if tag != "NAT" and (max(r["norm_dev_mean"]) > 0.01 or max(r["norm_dev_max"]) > 0.10):
            errs.append(f"{tag}: realized perturbation norm outside tolerance (per-step mean up to {max(r['norm_dev_mean']):.4f}, per-slot max up to {max(r['norm_dev_max']):.3f}; limits 0.01 / 0.10)")
        if tag == "NAT" and max(r["norm_dev_max"]) != 0.0:
            errs.append("NAT: norm deviation must be exactly 0")
    return errs


def append_records(dirpath: Path, recs):
    dirpath.mkdir(parents=True, exist_ok=True)
    for tag, r in recs.items():
        with open(dirpath / f"{tag}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(r) + "\n")


def done_ids(dirpath: Path, tag):
    return RS1.load_done(dirpath / f"{tag}.jsonl")


# ----------------------------------------------------------------------------- GPU blocks
def block_loop(name, be, tok, wb, prior, t0, cap_hours, indices, step_fn, done_fn):
    """Problem loop with the absolute ceiling: stop BEFORE a problem whose worst-case duration could cross the cap; hardware stops as in Stage 1."""
    import stage1_runtime as R
    worst, n = 0.0, 0
    for k, idx in enumerate(indices):
        if done_fn(idx):
            continue
        used = prior + (time.time() - t0)
        if used + 1.5 * worst + 90 > cap_hours * 3600:
            print(f"[CAP] stopping before problem #{k}: used {used / 3600:.3f} h; the experiment is TRUNCATED, the budget is not extended.", flush=True)
            return False
        tele = R.telemetry_now()
        why = R.hardware_stop(tele)
        if why:
            print(f"[HARDWARE STOP] {why}", flush=True)
            return False
        tp = time.perf_counter()
        ok = step_fn(idx)
        if not ok:
            return False
        worst = max(worst, time.perf_counter() - tp)
        n += 1
        if n <= 3 or n % 10 == 0:
            print(f"[{name}] #{k} idx={idx} {time.perf_counter() - tp:.1f}s | cum GPU {(prior + time.time() - t0) / 3600:.3f} h | {tele}", flush=True)
    return True


def gpu_gates(be, tok, wb, prior, t0, data, s, fp16_ids):
    import torch as T
    rep = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    stop = lambda ids: would_stop_at_boundary(tok, ids)       # noqa: E731
    # G1: alpha = 0 reproduces FP16 bitwise (tokens and cache) on gate train problems
    g1 = []
    for idx in s["gate_train_ids"][:N_GATE]:
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
                same = z.ids == ref.ids and all(T.equal(a, b) and T.equal(x, y) for (a, x), (b, y) in zip(c_ref, c_z))
                g1.append(bool(same))
        finally:
            be.run = orig
    rep["G1_alpha0_bitwise"] = {"n": len(g1), "all_identical": all(g1)}
    # G2/G3/G4: norms, finiteness, determinism on a few gate problems (teacher forced, 64 steps)
    g2, g4 = [], []
    for idx in s["gate_train_ids"][:8]:
        recs = run_tier_a_problem(be, tok, data, idx, "train", TIER_A_ARMS, None, S_STEPS, "gate")
        g2.append(integrity_errors_a(recs))
        _, prompt = data.problem("train", idx)
        ids_t = tok(prompt, return_tensors="pt").input_ids.to(be.device)
        a1 = C.generate_control(be, ids_t, "C1", 2.0, C.seed_for("C1", 2.0, idx, "gate"), 40, tok.eos_token_id).ids
        a2 = C.generate_control(be, ids_t, "C1", 2.0, C.seed_for("C1", 2.0, idx, "gate"), 40, tok.eos_token_id).ids
        a3 = C.generate_control(be, ids_t, "C1", 2.0, C.seed_for("C1", 2.0, idx, "gate-other"), 40, tok.eos_token_id).ids
        g4.append((a1 == a2, a1 != a3))
    rep["G2_G3_integrity_errors"] = [e for e in g2 if e]
    rep["G4_determinism"] = {"same_seed_identical": all(x for x, _ in g4), "different_seed_differs_in_n_of": [sum(y for _, y in g4), len(g4)]}
    # G5: NAT through the new path equals the STORED engine cache-only generation (reported; not a stop condition)
    stored = {}
    for p in (HERE / "stage1_runs" / "eval" / "cache_only.jsonl", HERE / "stage2_runs" / "eval" / "cache_only.jsonl"):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                stored[r["index"]] = r["ids"]
    g5 = []
    for idx in s["gate_test_ids_for_G5"]:
        r = run_tier_b_one(be, tok, data, idx, "NAT")
        g5.append(r["ids"] == stored[idx])
    rep["G5_nat_equals_stored_cache_only"] = {"n": len(g5), "identical": int(sum(g5))}
    # G6: runtime measurement -> projection
    t = time.perf_counter()
    idx = s["tierA_ids"][0]
    recs = run_tier_a_problem(be, tok, data, idx, "test", TIER_A_ARMS, fp16_ids, S_STEPS, "A")
    tA = time.perf_counter() - t
    tb = {}
    for arm in TIER_B_ARMS:
        t = time.perf_counter()
        run_tier_b_one(be, tok, data, s["tierB_ids"][1], arm)
        tb[arm] = time.perf_counter() - t
    per_a = tA * (len(TIER_A_ARMS) + 1) / len(TIER_A_ARMS)         # five arms measured; the sixth (C1*) costs about the same as one arm
    proj_h = (N_A * per_a + N_B * sum(tb.values()) + N_CAL * (5 * tA / len(TIER_A_ARMS) + 5)) / 3600 + 0.3
    rep["G6_runtime"] = {"tierA_problem_s_for_5_arms": tA, "tierB_problem_s_by_arm": tb, "projected_total_h_incl_calibration_and_checks": proj_h,
                         "ceiling_h": CAP_HOURS, "fits": proj_h <= CAP_HOURS}
    ok = (rep["G1_alpha0_bitwise"]["all_identical"] and not rep["G2_G3_integrity_errors"] and rep["G4_determinism"]["same_seed_identical"]
          and rep["G6_runtime"]["fits"])
    rep["all_passed"] = bool(ok)
    (OUT / "gates").mkdir(parents=True, exist_ok=True)
    (OUT / "gates" / "gates.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep, indent=1, default=float))
    return ok


# ----------------------------------------------------------------------------- CPU smoke
def cpu_smoke():
    from rv2.backends import build_cpu_backend, real_model_cpu
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(RS1.MODEL)
    be = build_cpu_backend(real_model_cpu(RS1.MODEL))
    data = RS1.Data()
    s = make_sample()
    out = HERE / "validation_logs" / "e1_cpu_smoke"
    RS1.MAX_NEW_TOKENS = 12
    idx = s["tierA_ids"][0]
    fp16 = load_fp16_ids()
    recs = run_tier_a_problem(be, tok, data, idx, "test", TIER_A_ARMS, fp16, 6, "A")
    print("Tier A integrity errors:", integrity_errors_a(recs))
    append_records(out / "tierA", recs)
    for arm in TIER_B_ARMS:
        r = run_tier_b_one(be, tok, data, s["tierB_ids"][0], arm)
        append_records(out / "tierB", {arm: r})
        print("Tier B", arm, r["status"], r["steps"])
    print("records written to", out)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "cpu-smoke", "ledger", "gates", "calibrate", "tierA", "tierB"])
    ap.add_argument("--authorized-gpu-run", action="store_true")
    ap.add_argument("--cap-hours", type=float, default=CAP_HOURS)
    a = ap.parse_args()
    if a.cap_hours > CAP_HOURS:
        sys.exit(f"cap above the approved {CAP_HOURS} h is not permitted")
    configure()
    if a.cmd == "plan":
        s = make_sample()
        print({k: (len(v) if isinstance(v, list) else v) for k, v in s.items() if k != "excluded_train_ids"})
        assert len(set(s["tierB_ids"])) == N_B and set(s["tierA_ids"]) <= set(s["tierB_ids"]) and not set(s["calibration_train_ids"]) & set(s["excluded_train_ids"])
        print("tier A subset of tier B:", set(s["tierA_ids"]) <= set(s["tierB_ids"]), "| exposed test problems in sample:", len(s["exposed_test_ids_in_sample"]))
        return
    if a.cmd == "ledger":
        print(f"E1 cumulative GPU wall time: {RS1.ledger_seconds() / 3600:.3f} h of {a.cap_hours} h")
        return
    if a.cmd == "cpu-smoke":
        cpu_smoke()
        return
    require_gpu_permission(a, need_gates=a.cmd in ("calibrate", "tierA", "tierB"), need_calibration=a.cmd in ("tierA", "tierB"))
    s = make_sample()
    data = RS1.Data()
    fp16 = load_fp16_ids()

    if a.cmd == "gates":
        RS1.gpu_process("gates", lambda be, dyn, tok, wb, prior, t0: gpu_gates(be, tok, wb, prior, t0, data, s, fp16), a.cap_hours)
        return

    if a.cmd == "calibrate":
        d = OUT / "calib"

        def step(be, tok, idx):
            recs = run_tier_a_problem(be, tok, data, idx, "train", CAL_ARMS, None, S_STEPS, "calib")
            errs = integrity_errors_a(recs)
            if errs:
                (OUT / "integrity_failure.json").write_text(json.dumps({"index": idx, "errors": errs}))
                print("[INTEGRITY FAILURE]", errs[:3], flush=True)
                return False
            append_records(d, recs)
            return True
        RS1.gpu_process("calibrate", lambda be, dyn, tok, wb, prior, t0: block_loop(
            "calib", be, tok, wb, prior, t0, a.cap_hours, s["calibration_train_ids"], lambda i: step(be, tok, i),
            lambda i: all(i in done_ids(d, tag) for _, _, tag in CAL_ARMS)), a.cap_hours)
        return

    calib = json.loads((OUT / "e1_calibration.json").read_text())
    if a.cmd == "tierA":
        arms = TIER_A_ARMS + [("C1", float(calib["alpha_star"]), "C1star")]
        d = OUT / "tierA"

        def step(be, tok, idx):
            recs = run_tier_a_problem(be, tok, data, idx, "test", arms, fp16, S_STEPS, "A")
            errs = integrity_errors_a(recs)
            if errs:
                (OUT / "integrity_failure.json").write_text(json.dumps({"index": idx, "errors": errs}))
                print("[INTEGRITY FAILURE]", errs[:3], flush=True)
                return False
            append_records(d, recs)
            return True
        RS1.gpu_process("tierA", lambda be, dyn, tok, wb, prior, t0: block_loop(
            "tierA", be, tok, wb, prior, t0, a.cap_hours, s["tierA_ids"], lambda i: step(be, tok, i),
            lambda i: all(i in done_ids(d, tag) for _, _, tag in arms)), a.cap_hours)
        return

    if a.cmd == "tierB":
        d = OUT / "tierB"

        def step(be, tok, idx):
            recs = {}
            for k, arm in enumerate(TIER_B_ARMS[idx % 4:] + TIER_B_ARMS[:idx % 4]):        # rotated order balances clock drift
                recs[arm] = run_tier_b_one(be, tok, data, idx, arm)
            if recs["NAT"]["ids"][0] != recs["FP16"]["ids"][0] or recs["C1"]["ids"][0] != recs["FP16"]["ids"][0]:
                print("[INTEGRITY FAILURE] first token differs from the shared FP16 prefill token", flush=True)
                return False
            append_records(d, recs)
            return True
        RS1.gpu_process("tierB", lambda be, dyn, tok, wb, prior, t0: block_loop(
            "tierB", be, tok, wb, prior, t0, a.cap_hours, s["tierB_ids"], lambda i: step(be, tok, i),
            lambda i: all(i in done_ids(d, arm) for arm in TIER_B_ARMS)), a.cap_hours)


if __name__ == "__main__":
    main()
