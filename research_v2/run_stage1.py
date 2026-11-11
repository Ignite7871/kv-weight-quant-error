"""Stage 1 driver (STAGE1_SPEC.md): cache-mechanism experiment. GPU subcommands need --authorized-gpu-run.

Subcommands
  plan          write stage1_sample.json (pool, run order, reserved 389, gate sets) and print a summary (CPU)
  gate-equiv    GPU equivalence block vs the untouched FP16/NF4 implementations and the independent reference
  gate-feasible GPU feasibility subset (train problems): both hybrid cells behave as specified; timing for projection
  speed         brief fixed-workload speed re-check (GPU)
  project       runtime projection from measured rates -> stage1_runs/projection.json (CPU); required before `run`
  run           main evaluation, problem-major over the seeded order, resumable (GPU)
  ledger        print cumulative GPU time
  cpu-smoke     end-to-end I/O check with the real model on CPU (few tokens)

GPU budget: every GPU process appends its full wall time (model load included) to stage1_runs/gpu_ledger.jsonl.
The cap (12.0 h cumulative, ALL stage-1 GPU use, absolute) is enforced by the driver (amendment A2): the run stops before
any problem whose worst-case duration could cross it, and a watchdog closes the ledger and exits shortly before it. The
completion projection is informational and never stops the run; a run cut short by the ceiling is TRUNCATED, not extended.
There is no flag to override the cap.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from datasets import load_dataset          # HF before torch
from transformers import AutoTokenizer

import torch

import rv2  # noqa: F401
from rv2 import PROJECT_ROOT
from rv2.boundary import find_boundary, score, would_stop_at_boundary
from rv2.engine import account, generate
from rv2.policies import OracleFlip, StaticFP16, StaticNF4
from rv2.stage1 import CACHE_ONLY, TOKEN_ONLY, Cell, generate_cell
from probe_gsm8k_routing import build_fewshot_prefix   # historical prompt builder

MODEL = "Qwen/Qwen2.5-0.5B"
MAX_NEW_TOKENS = 256
N_FEWSHOT = 3
OUT = HERE / "stage1_runs"
LEDGER = OUT / "gpu_ledger.jsonl"
SAMPLE_FILE = HERE / "stage1_sample.json"
PROJECTION_FILE = OUT / "projection.json"
ORDER_SEED = "rv2-stage1-order-2026-10-23"
ARMS = ["fp16", "nf4", "cache_only", "token_only", "oracle_flip"]
CAP_HOURS = 12.0
PILOT_SAMPLE = HERE / "pilot_sample.json"


# ----------------------------------------------------------------------------- sample
def make_sample():
    if SAMPLE_FILE.exists():
        return json.loads(SAMPLE_FILE.read_text())
    pool = sorted(r["index"] for r in json.load(open(PROJECT_ROOT / "results" / "gsm8k_multibudget_p10_entropy_only.json"))["results"])
    assert len(pool) == 930 and len(set(pool)) == 930
    n_test = len(load_dataset("openai/gsm8k", "main", split="test"))
    reserved = sorted(set(range(n_test)) - set(pool))
    assert n_test == 1319 and len(reserved) == 389
    order = random.Random(ORDER_SEED).sample(pool, len(pool))          # seeded permutation: a prefix is a random subset
    pilot = json.loads(PILOT_SAMPLE.read_text())
    s = {"pool930": pool, "run_order": order, "reserved389": reserved, "order_seed": ORDER_SEED,
         "equiv_train": pilot["train_equiv"],                         # the pilot's 5 equivalence problems (train split)
         "feasibility_train": pilot["train_calib"][:10],              # first 10 pilot calibration problems (train split)
         "speed_train": pilot["train_equiv"][:2], "n_test": n_test}
    SAMPLE_FILE.write_text(json.dumps(s, indent=1))
    return s


def assert_test_indices_allowed(indices, s):
    bad = set(indices) & set(s["reserved389"])
    assert not bad, f"refusing to evaluate reserved validation problems: {sorted(bad)[:5]}"
    assert set(indices) <= set(s["pool930"]), "indices outside the 930-problem exploratory pool"


# ----------------------------------------------------------------------------- provenance / ledger
def sha256(p: Path):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def provenance():
    code = sorted((HERE / "rv2").glob("*.py")) + [Path(__file__), HERE / "STAGE1_SPEC.md"]
    hist = [PROJECT_ROOT / "src" / f for f in ("dynamic_linear.py", "dynamic_model.py", "routing.py", "fragility.py")]
    hist += [PROJECT_ROOT / "scripts" / f for f in ("gsm8k_gate_check.py", "probe_gsm8k_routing.py")]
    git = subprocess.run(["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(PROJECT_ROOT), "status", "--porcelain", "research_v2/rv2", "research_v2/run_stage1.py",
                            "research_v2/STAGE1_SPEC.md", "src", "scripts"], capture_output=True, text=True).stdout.strip()
    import bitsandbytes, transformers
    return {"git_head": git, "git_dirty_paths": dirty.splitlines(),
            "code_sha256": {str(p.relative_to(PROJECT_ROOT)): sha256(p) for p in code + hist if p.exists()},
            "versions": {"python": platform.python_version(), "torch": torch.__version__,
                         "transformers": transformers.__version__, "bitsandbytes": bitsandbytes.__version__,
                         "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
            "model": MODEL, "max_new_tokens": MAX_NEW_TOKENS}


def ledger_seconds():
    if not LEDGER.exists():
        return 0.0
    return sum(json.loads(l)["seconds"] for l in LEDGER.read_text().splitlines() if l.strip())


PROGRESS = OUT / "current_process.json"
RUNTIME_LOG = OUT / "runtime_log.jsonl"
SESSIONS = OUT / "sessions.jsonl"
INTEGRITY_FAILURE = OUT / "integrity_failure.json"
CHECKPOINT_SNAPSHOT = OUT / "checkpoint_155_integrity.json"
SESSION: dict = {}          # per-process statistics, merged into the process's ledger line and sessions.jsonl (A2)

# Amendment A2 (STAGE1_SPEC.md section 10): the ONLY stopping rule is the absolute cumulative ceiling. The earlier
# projection-based stop was removed; the projection is informational. See stage1_runtime.py.
from stage1_runtime import (HARD_CAP_HOURS, TEMP_STOP_C, WATCHDOG_MARGIN_S, Watchdog, append_jsonl, arm_totals,  # noqa: E402
                            hardware_stop, pace_stats, per_problem_totals, problem_integrity_errors, stop_decision,
                            telemetry_now, verify_checkpoint, worst_problem_seconds, DEFAULT_OVERHEAD)


def recover_open_process(progress_path, ledger_path):
    """If a previous GPU process ended without closing its ledger entry (crash, kill, power loss), charge its last recorded
    elapsed time to the ledger so the cap cannot be silently exceeded. Returns the recovered seconds (0.0 if none)."""
    if not progress_path.exists():
        return 0.0
    try:
        d = json.loads(progress_path.read_text())
    except Exception:  # noqa: BLE001
        return 0.0
    if not d.get("open"):
        return 0.0
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger_path, "a") as f:
        f.write(json.dumps({"start_utc": d.get("start_utc"), "seconds": float(d.get("seconds_so_far", 0.0)), "block": d.get("block"),
                            "completed": False, "recovered_after_unclean_exit": True}) + "\n")
    progress_path.unlink()
    return float(d.get("seconds_so_far", 0.0))


def write_progress(block, t_proc0):
    tmp = PROGRESS.with_suffix(".tmp")
    tmp.write_text(json.dumps({"open": True, "block": block, "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_proc0)),
                               "seconds_so_far": time.time() - t_proc0, "last_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
    os.replace(tmp, PROGRESS)        # atomic: a crash never leaves a torn progress file


class KeepAwake:
    """Prevents Windows idle sleep while a long GPU process runs (process-scoped; released on exit).
    It changes no power plan or system setting."""

    def __enter__(self):
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        except Exception:  # noqa: BLE001
            pass
        return self

    def __exit__(self, *a):
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        except Exception:  # noqa: BLE001
            pass


def gpu_process(name, fn, cap_hours):
    """Run fn(backend, dyn, tok, wb, prior_seconds, t_proc0) as one ledgered GPU process."""
    if not torch.cuda.is_available():
        sys.exit("CUDA not available.")
    rec = recover_open_process(PROGRESS, LEDGER)
    if rec:
        print(f"[ledger] charged {rec / 3600:.3f} h from a previous process that did not exit cleanly", flush=True)
    prior = ledger_seconds()
    if prior >= cap_hours * 3600:
        sys.exit(f"GPU cap reached ({prior / 3600:.3f} h of {cap_hours} h).")
    t0 = time.time()
    cap_s = cap_hours * 3600
    from rv2.backends import load_gpu_backend, weight_bytes_per_pass
    ok = False
    state = {"done": False}
    lock = threading.Lock()
    SESSION.clear()

    def finalize(ok_, extra=None):
        """Write this process's ledger line exactly once (main thread or watchdog, whichever comes first)."""
        with lock:
            if state["done"]:
                return
            state["done"] = True
            OUT.mkdir(parents=True, exist_ok=True)
            line = {"start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)), "seconds": time.time() - t0,
                    "block": name, "completed": bool(ok_), **SESSION, **(extra or {})}
            with open(LEDGER, "a") as f:
                f.write(json.dumps(line, default=float) + "\n")
            if name == "eval":
                append_jsonl(SESSIONS, line)
            if PROGRESS.exists():
                PROGRESS.unlink()
            print(f"[ledger] cumulative GPU wall time: {ledger_seconds() / 3600:.3f} h of {cap_hours} h", flush=True)

    def heartbeat():
        with lock:
            if not state["done"]:
                write_progress(name, t0)         # a crash or power loss is charged at most ~15 s short

    def on_expire():
        SESSION["stop_reason"] = "watchdog: absolute ceiling reached"
        finalize(False, {"watchdog_fired": True})
        print(f"[WATCHDOG] absolute {cap_hours} h ceiling reached; ledger closed, exiting. Experiment is TRUNCATED.", flush=True)
        os._exit(3)

    wd = Watchdog(t0 + (cap_s - prior) - WATCHDOG_MARGIN_S, on_expire, on_tick=heartbeat)
    wd.start()
    try:
        with KeepAwake():
            backend, dyn = load_gpu_backend(MODEL)
            tok = AutoTokenizer.from_pretrained(MODEL)
            ok = fn(backend, dyn, tok, weight_bytes_per_pass(backend.model), prior, t0)
    finally:
        wd.stop()
        finalize(ok)


# ----------------------------------------------------------------------------- generation of one arm on one problem
def make_policy_runner(arm):
    if arm == "fp16":
        return lambda be, ids_t, tok, stop: generate(be, ids_t, StaticFP16(), MAX_NEW_TOKENS, tok.eos_token_id, 0,
                                                     early_stop_fn=stop, instrument=False)
    if arm == "nf4":
        return lambda be, ids_t, tok, stop: generate(be, ids_t, StaticNF4(), MAX_NEW_TOKENS, tok.eos_token_id, 0,
                                                     early_stop_fn=stop, instrument=False)
    if arm == "oracle_flip":
        return lambda be, ids_t, tok, stop: generate(be, ids_t, OracleFlip(), MAX_NEW_TOKENS, tok.eos_token_id, 0,
                                                     early_stop_fn=stop, instrument=False)
    cell = {"cache_only": CACHE_ONLY, "token_only": TOKEN_ONLY}[arm]
    return lambda be, ids_t, tok, stop: generate_cell(be, ids_t, cell, MAX_NEW_TOKENS, tok.eos_token_id, early_stop_fn=stop)


def run_one(arm, backend, tok, prompt_ids, prob, idx, split, wb, early_stop=True):
    cuda = backend.device.startswith("cuda")
    stop = (lambda ids: would_stop_at_boundary(tok, ids)) if early_stop else None
    if cuda:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    res = make_policy_runner(arm)(backend, prompt_ids, tok, stop)
    if cuda:
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    b = find_boundary(tok, res.ids, tok.eos_token_id, MAX_NEW_TOKENS)
    text = tok.decode(res.ids, skip_special_tokens=True)
    rec = {"index": idx, "split": split, "arm": arm, "kind": res.policy.get("kind"), "prompt_tokens": int(prompt_ids.shape[1]),
           **score(prob, text, b),
           "boundary": {k: v for k, v in b.to_dict().items() if not k.endswith("_mask")},
           "answer_token_mask": b.answer_token_mask, "ids": res.ids, "text": text,
           "routes": "".join(res.routes), "passes": "".join(res.passes) if len(res.passes) < 3 else None,
           "steps": len(res.routes), "stopped_early": res.stopped_early, "early_stop_rule": "drift marker" if early_stop else None,
           "wall_seconds": wall, "gen_seconds": res.timing["total"],
           "peak_mem_allocated": torch.cuda.max_memory_allocated() if cuda else None,
           "accounting": account(res, b.answer_token_mask, wb, None, prompt_len=int(prompt_ids.shape[1])),
           "routes_meaning": "KV-source letter per step (F/N); for static arms the precision; for oracle_flip the kept KV"}
    if arm in ("cache_only", "token_only"):
        rec.update({"tF": res.trace["tF"], "tN": res.trace["tN"], "js": res.trace["js"],
                    "token_source": res.policy["token_source"], "kv_source": res.policy["kv_source"]})
    return rec


class Data:
    def __init__(self):
        self.train = load_dataset("openai/gsm8k", "main", split="train")
        self.test = load_dataset("openai/gsm8k", "main", split="test")
        self.prefix = build_fewshot_prefix(self.train.select(range(N_FEWSHOT)))

    def problem(self, split, idx):
        prob = (self.train if split == "train" else self.test)[idx]
        return prob, self.prefix + f"Question: {prob['question']}\nAnswer:"          # historical template


def load_done(path):
    if not path.exists():
        return set()
    return {json.loads(l)["index"] for l in path.read_text().splitlines() if l.strip()}


def load_slim(path):
    """index -> {wall_seconds, steps} for every record in an arm file (pace statistics only)."""
    out = {}
    if path.exists():
        for l in path.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                out[r["index"]] = {"wall_seconds": r["wall_seconds"], "steps": r["steps"]}
    return out


def load_record(path, idx):
    for l in path.read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            if r["index"] == idx:
                return r
    raise KeyError(idx)


def run_problems(block, backend, tok, wb, indices, split, arms, data, out_dir, cap_hours=None, t_proc0=None, prior=0.0,
                 projection=None, early_stop=True, tag="", a2=False):
    """Problem-major: for each problem run every missing arm (rotated order, so clock drift is balanced across
    arms). Resumable per (arm, problem).

    Stopping (amendment A2): the ONLY rule is the absolute cumulative ceiling (`stop_decision`: stop before a problem whose
    worst-case duration could cross it). With a2=True (main run) the problem's records are integrity-checked before they are
    written, GPU telemetry is logged every 10 problems, over-temperature stops the run, and pooled/session pace statistics
    are logged (informational: they never stop the run). Returns True iff every index in `indices` is complete."""
    (out_dir / block).mkdir(parents=True, exist_ok=True)
    (out_dir / block / f"manifest{tag}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json").write_text(json.dumps(
        {"block": block, "split": split, "n_indices": len(indices), "arms": arms, **provenance(),
         "weight_bytes_per_pass": wb, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1))
    paths = {a: out_dir / block / f"{a}.jsonl" for a in arms}
    done = {a: load_done(p) for a, p in paths.items()}
    hist = {a: load_slim(p) for a, p in paths.items()} if a2 else None
    totals_all = list(per_problem_totals(hist, arms).values()) if a2 else []
    max_hist_total = max(totals_all) if totals_all else 0.0
    hist_idx = set.intersection(*[set(v) for v in hist.values()]) if a2 else set()
    pooled_arm = arm_totals(hist, None, arms) if a2 else None
    sess_arm = {a: [0.0, 0] for a in arms}
    totals_session, secs, n_done_problems = [], [], 0
    t_loop0 = time.time()
    SESSION.update({"problems_completed": 0, "stop_reason": None, "first_order_position": None,
                    "problems_complete_at_start": len(hist_idx), "temps_c": [], "sm_mhz": []})
    try:
        for k, idx in enumerate(indices):
            todo = [a for a in arms if idx not in done[a]]
            if not todo:
                continue
            if cap_hours is not None:
                used = prior + (time.time() - t_proc0)
                overhead = DEFAULT_OVERHEAD
                if totals_session:
                    overhead = max(DEFAULT_OVERHEAD, (time.time() - t_loop0) / sum(totals_session))
                worst = worst_problem_seconds(totals_all + totals_session, overhead)
                if stop_decision(used, worst, cap_hours * 3600) == "cap":
                    SESSION["stop_reason"] = "absolute ceiling (truncated)"
                    print(f"[CAP] stopping before problem #{k} (idx {idx}): used {used / 3600:.3f} h + worst-case problem "
                          f"{worst:.0f} s + margin > absolute cap {cap_hours:.2f} h. {n_done_problems} problems completed in this "
                          f"process. The experiment is TRUNCATED; the budget is not extended.", flush=True)
                    return False
            if a2:
                tele = telemetry_now()
                if isinstance(tele, dict) and tele.get("temp_c") is not None:
                    SESSION["temps_c"].append(tele["temp_c"])
                    SESSION["sm_mhz"].append(tele.get("sm_mhz"))
                why = hardware_stop(tele)
                if why:
                    SESSION["stop_reason"] = f"hardware safety: {why}"
                    print(f"[HARDWARE STOP] before problem #{k} (idx {idx}): {why}. Completed records are kept.", flush=True)
                    return False
            prob, prompt = data.problem(split, idx)
            ids_t = tok(prompt, return_tensors="pt").input_ids.to(backend.device)         # historical tokenization
            rot = k % len(arms)
            order = [a for a in (arms[rot:] + arms[:rot]) if a in todo]
            t_problem0 = time.perf_counter()
            line, recs = [], {}
            for arm in order:
                rec = run_one(arm, backend, tok, ids_t, prob, idx, split, wb, early_stop)
                recs[arm] = rec
                line.append(f"{arm}:{rec['status'][:1]}/{rec['steps']}st/{rec['wall_seconds']:.1f}s")
                if not a2:
                    with open(paths[arm], "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec) + "\n")
            secs.append(time.perf_counter() - t_problem0)
            if a2:
                full = {a: (recs[a] if a in recs else load_record(paths[a], idx)) for a in arms}
                errs = problem_integrity_errors(full)
                if errs:
                    SESSION["stop_reason"] = "integrity failure"
                    INTEGRITY_FAILURE.write_text(json.dumps({"index": idx, "order_position": k, "errors": errs, "records": full},
                                                            default=float))
                    print(f"[INTEGRITY FAILURE] problem idx {idx} (order position {k}): {errs[:5]}. Stopped immediately; the "
                          f"failing records are saved in {INTEGRITY_FAILURE.name} and were NOT appended to the arm files.", flush=True)
                    return False
                for arm in order:                                      # all five arms are appended only after the checks pass
                    with open(paths[arm], "a", encoding="utf-8") as f:
                        f.write(json.dumps(recs[arm]) + "\n")
                tot = sum(full[a]["wall_seconds"] for a in arms)
                totals_all.append(tot)
                totals_session.append(tot)
                for a in arms:
                    sess_arm[a][0] += full[a]["wall_seconds"]
                    sess_arm[a][1] += full[a]["steps"]
                if SESSION["first_order_position"] is None:
                    SESSION["first_order_position"] = k
            n_done_problems += 1
            SESSION["problems_completed"] = n_done_problems
            SESSION["last_order_position"] = k
            if t_proc0 is not None:
                write_progress(block, t_proc0)
            if n_done_problems % 10 == 0 or n_done_problems <= 3:
                used = prior + (time.time() - t_proc0) if t_proc0 else 0.0
                print(f"[{block}] #{k} idx={idx} " + " ".join(line) + f" | cum GPU {used / 3600:.3f} h", flush=True)
                if a2:
                    overhead = max(DEFAULT_OVERHEAD, (time.time() - t_loop0) / sum(totals_session))
                    remaining = len(indices) - (len(hist_idx) + n_done_problems)
                    ps = pace_stats(totals_all, totals_session, overhead, used, remaining, cap_hours * 3600)
                    ms_sess = {a: 1000 * sess_arm[a][0] / sess_arm[a][1] for a in arms if sess_arm[a][1]}
                    ms_pool = {a: 1000 * pooled_arm[a][0] / pooled_arm[a][1] for a in arms if pooled_arm[a][1]}
                    entry = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "order_position": k, "idx": idx,
                             "complete_all_arms": len(hist_idx) + n_done_problems, "overhead": overhead, **ps,
                             "ms_per_step_session": ms_sess, "ms_per_step_pooled_at_session_start": ms_pool, "telemetry": tele}
                    append_jsonl(RUNTIME_LOG, entry)
                    print(f"    pace: pooled {ps['pooled_s_per_problem']:.1f} s/problem | session {ps['session_s_per_problem']:.1f} | "
                          f"last20 {ps['last20_s_per_problem']:.1f} | projected cumulative {ps['projected_cumulative_h_pooled']:.2f} h "
                          f"(pooled) / {ps['projected_cumulative_h_last20']:.2f} h (last20) of {cap_hours:.1f} h"
                          f"{' | SUSTAINED SLOWDOWN FLAG' if ps['sustained_slowdown'] else ''} | GPU {tele.get('temp_c')} C "
                          f"{tele.get('sm_mhz')} MHz", flush=True)
        SESSION["stop_reason"] = SESSION["stop_reason"] or "complete"
        return True
    finally:
        if a2 and totals_session:
            SESSION["mean_s_per_problem_wall_sum"] = sum(totals_session) / len(totals_session)
            SESSION["per_arm_mean_wall_s"] = {a: sess_arm[a][0] / len(totals_session) for a in arms}
            SESSION["per_arm_ms_per_step"] = {a: 1000 * sess_arm[a][0] / sess_arm[a][1] for a in arms if sess_arm[a][1]}


# ----------------------------------------------------------------------------- CPU smoke
def cpu_smoke():
    from rv2.backends import build_cpu_backend, real_model_cpu, weight_bytes_per_pass
    tok = AutoTokenizer.from_pretrained(MODEL)
    be = build_cpu_backend(real_model_cpu(MODEL))
    s = make_sample()
    data = Data()
    global MAX_NEW_TOKENS
    MAX_NEW_TOKENS = 10
    out = HERE / "validation_logs" / "stage1_cpu_smoke"
    run_problems("smoke", be, tok, weight_bytes_per_pass(be.model), s["feasibility_train"][:2], "train", ARMS, data, out)
    for a in ARMS:
        rec = json.loads((out / "smoke" / f"{a}.jsonl").read_text().splitlines()[0])
        print(a, rec["status"], rec["steps"], rec["routes"], list(rec)[:6])


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "gate-equiv", "gate-feasible", "speed", "project", "run", "ledger", "cpu-smoke"])
    ap.add_argument("--cap-hours", type=float, default=CAP_HOURS)
    ap.add_argument("--limit", type=int, default=None, help="run only the first N problems of the order (diagnostic)")
    ap.add_argument("--authorized-gpu-run", action="store_true",
                    help="required for any GPU run; only pass after explicit author authorization")
    a = ap.parse_args()
    if a.cap_hours > CAP_HOURS:
        sys.exit(f"cap above the authorized {CAP_HOURS} h is not permitted")

    if a.cmd == "plan":
        s = make_sample()
        print({k: (len(v) if isinstance(v, list) else v) for k, v in s.items()})
        assert_test_indices_allowed(s["run_order"], s)
        print("run order is a permutation of the pool:", sorted(s["run_order"]) == s["pool930"],
              "| reserved disjoint:", not set(s["reserved389"]) & set(s["pool930"]))
        return
    if a.cmd == "ledger":
        print(f"cumulative GPU wall time: {ledger_seconds() / 3600:.3f} h of {a.cap_hours} h")
        return
    if a.cmd == "cpu-smoke":
        return cpu_smoke()
    if a.cmd == "project":
        import stage1_gates
        return stage1_gates.project(OUT, make_sample(), ARMS, a.cap_hours, ledger_seconds())
    if not a.authorized_gpu_run:
        sys.exit("Refusing to run: GPU runs need explicit author authorization (--authorized-gpu-run).")

    import stage1_gates
    s = make_sample()
    data = Data()
    if a.cmd == "gate-equiv":
        return gpu_process("gate-equiv", lambda be, dyn, tok, wb, prior, t0: stage1_gates.gate_equiv(be, dyn, tok, wb, s, data, OUT), a.cap_hours)
    if a.cmd == "gate-feasible":
        def f(be, dyn, tok, wb, prior, t0):
            ok = run_problems("feasibility", be, tok, wb, s["feasibility_train"], "train", ARMS, data, OUT,
                              cap_hours=a.cap_hours, t_proc0=t0, prior=prior)
            stage1_gates.check_feasibility(OUT / "feasibility", s["feasibility_train"])
            return ok
        return gpu_process("gate-feasible", f, a.cap_hours)
    if a.cmd == "speed":
        return gpu_process("speed", lambda be, dyn, tok, wb, prior, t0: stage1_gates.speed_check(be, tok, data, s, OUT), a.cap_hours)
    if a.cmd == "run":
        order = s["run_order"][: a.limit] if a.limit else s["run_order"]
        assert_test_indices_allowed(order, s)
        rep = verify_checkpoint(OUT / "eval", CHECKPOINT_SNAPSHOT if CHECKPOINT_SNAPSHOT.exists() else None, s["run_order"])
        print(f"[checkpoint] {rep['n_complete_all_arms']} problems complete on all arms; verification "
              f"{'OK' if rep['ok'] else 'FAILED'}", flush=True)
        if not rep["ok"]:
            sys.exit("Refusing to run: checkpoint verification failed: " + "; ".join(rep["problems"]))
        done_all = set.intersection(*[load_done(OUT / "eval" / f"{x}.jsonl") for x in ARMS])
        remaining = [i for i in order if i not in done_all]
        left = a.cap_hours * 3600 - ledger_seconds()
        print(f"[A2] remaining problems {len(remaining)}; next = run_order[{len(done_all)}] = {remaining[0] if remaining else None}; "
              f"{left / 3600:.3f} h left under the absolute {a.cap_hours} h ceiling", flush=True)
        if PROJECTION_FILE.exists():
            need = json.loads(PROJECTION_FILE.read_text())["sec_per_problem_central"] * len(remaining)
            print(f"[A2] informational only: gate-time projection {need / 3600:.2f} h vs {left / 3600:.2f} h left. The run proceeds "
                  f"and, if the ceiling is reached first, ends TRUNCATED (no extension).", flush=True)
        gpu_process("eval", lambda be, dyn, tok, wb, prior, t0: run_problems(
            "eval", be, tok, wb, order, "test", ARMS, data, OUT, cap_hours=a.cap_hours, t_proc0=t0, prior=prior, a2=True), a.cap_hours)
        reason = SESSION.get("stop_reason")
        print(f"[A2] process ended: {reason}; problems completed this process: {SESSION.get('problems_completed')}", flush=True)
        if reason in ("integrity failure",) or (reason or "").startswith("hardware"):
            sys.exit(2)
        if reason and reason.startswith("absolute ceiling"):
            sys.exit(4)

if __name__ == "__main__":
    main()
