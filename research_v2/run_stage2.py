"""Stage 2 driver: protocol-locked held-out validation of the Stage 1 cache/token experiment (STAGE2_SPEC.md).

The generation code, the five arms, the per-problem integrity checks, the telemetry and the absolute-ceiling logic are the VERIFIED
Stage 1 / amendment-A2 code in run_stage1.py, rv2/ and stage1_runtime.py, imported unchanged. This file only
  * defines the held-out sample (the 389 test problems outside the 930-problem pool) and its seeded run order,
  * points the Stage 1 driver at Stage 2's own folders and its own GPU ledger (cap 5.5 h, additional to Stage 1),
  * adds Stage 2 subcommands (plan, check-clean, gate-equiv, speed, project, run, ledger).

Subcommands
  plan          write stage2_sample.json and print a summary (CPU)
  check-clean   verify the held-out problems are disjoint from the pool and absent from every stored outcome file (CPU)
  gate-equiv    short GPU correctness check: the Stage 1 equivalence block on the first 3 pilot train problems
  speed         brief fixed-workload speed re-check (GPU)
  project       runtime projection from the Stage 1 measurements -> stage2_runs/projection.json (CPU)
  run           held-out evaluation, problem-major over the seeded order, resumable (GPU)
  ledger        print cumulative Stage 2 GPU time

GPU budget: ADDITIONAL 5.5 h cumulative for everything in Stage 2 (checks included), counted in stage2_runs/gpu_ledger.jsonl.
The ceiling is absolute (amendment A2 semantics): no flag raises it, and a run cut short by it is TRUNCATED, never extended.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_stage1 as RS1                       # noqa: E402  (imports HF datasets, torch, rv2)

OUT = HERE / "stage2_runs"
SAMPLE_FILE = HERE / "stage2_sample.json"
STAGE1_SAMPLE = HERE / "stage1_sample.json"
ORDER_SEED = "rv2-stage2-order-2026-10-25"
CAP_HOURS = 5.5
ARMS = RS1.ARMS
N_EQUIV = 3                                    # "short" GPU correctness check: first 3 of the pilot's 5 equivalence problems


# ----------------------------------------------------------------------------- sample
def make_sample2() -> dict:
    if SAMPLE_FILE.exists():
        return json.loads(SAMPLE_FILE.read_text())
    s1 = json.loads(STAGE1_SAMPLE.read_text())
    held = sorted(s1["reserved389"])
    pool = sorted(s1["pool930"])
    assert len(held) == 389 and len(pool) == 930 and not set(held) & set(pool) and set(held) | set(pool) == set(range(1319))
    order = random.Random(ORDER_SEED).sample(held, len(held))              # seeded permutation: any prefix is a random subset
    s = {"heldout389": held, "run_order": order, "order_seed": ORDER_SEED, "pool930": pool,
         "equiv_train": s1["equiv_train"][:N_EQUIV], "speed_train": s1["speed_train"],
         "exposed_gold_trajectory_test_0_199": [i for i in held if i < 200],
         "exposed_probe_test_0_7": [i for i in held if i < 8], "n_test": s1["n_test"]}
    SAMPLE_FILE.write_text(json.dumps(s, indent=1))
    return s


def assert_heldout_indices(indices, s):
    """Only held-out problems may be evaluated, and none of the 930 pool (Stage 1) problems."""
    bad = set(indices) & set(s["pool930"])
    assert not bad, f"refusing to evaluate Stage 1 pool problems in Stage 2: {sorted(bad)[:5]}"
    assert set(indices) <= set(s["heldout389"]), "indices outside the 389 held-out problems"


def configure(out: Path = OUT):
    """Point the verified Stage 1 driver at Stage 2's folders and ledger. Nothing in rv2/ or the generation path changes."""
    out.mkdir(parents=True, exist_ok=True)
    RS1.OUT = out
    RS1.LEDGER = out / "gpu_ledger.jsonl"
    RS1.PROGRESS = out / "current_process.json"
    RS1.RUNTIME_LOG = out / "runtime_log.jsonl"
    RS1.SESSIONS = out / "sessions.jsonl"
    RS1.INTEGRITY_FAILURE = out / "integrity_failure.json"
    RS1.CHECKPOINT_SNAPSHOT = out / "checkpoint_snapshot.json"
    RS1.PROJECTION_FILE = out / "projection.json"
    RS1.SAMPLE_FILE = SAMPLE_FILE
    return out


# ----------------------------------------------------------------------------- projection (CPU)
def project(out: Path, cap_hours: float, used_s: float, n_problems: int = 389) -> dict:
    """Projected cumulative Stage 2 GPU time from the Stage 1 per-session measurements (arms' wall seconds per problem; the
    held-out problems are a random remainder of the same pool). Checks and the model load are added explicitly."""
    sess = [json.loads(l) for l in (HERE / "stage1_runs" / "sessions.jsonl").read_text().splitlines() if l.strip()]
    per_session = [s["mean_s_per_problem_wall_sum"] for s in sess]
    n_each = [s["problems_completed"] for s in sess]
    pooled = sum(m * n for m, n in zip(per_session, n_each)) / sum(n_each)
    overhead = 1.03
    check_s = {"gate-equiv (3 problems, incl. load)": 0.14 * 3600, "speed (incl. load)": 0.03 * 3600}
    load_s = 90.0
    rates = {"fastest_session": min(per_session), "pooled": pooled, "slowest_session": max(per_session)}
    proj = {k: {"s_per_problem": v * overhead, "main_run_h": v * overhead * n_problems / 3600,
                "cumulative_h": (used_s + sum(check_s.values()) + load_s + v * overhead * n_problems) / 3600} for k, v in rates.items()}
    res = {"n_problems": n_problems, "cap_hours": cap_hours, "used_h_so_far": used_s / 3600, "reserved_for_checks_h": sum(check_s.values()) / 3600,
           "checks_assumed_s": check_s, "load_s": load_s, "stage1_session_rates_s_per_problem": per_session, "projection": proj,
           "fits_central": proj["pooled"]["cumulative_h"] <= cap_hours,
           "fits_even_at_slowest_session_rate": proj["slowest_session"]["cumulative_h"] <= cap_hours,
           "max_problems_at_slowest_rate": int(max(0.0, cap_hours * 3600 - 60 - used_s - sum(check_s.values()) - load_s)
                                               // (rates["slowest_session"] * overhead))}
    (out / "projection.json").write_text(json.dumps(res, indent=1))
    return res


def ledger_hours():
    return RS1.ledger_seconds() / 3600


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "check-clean", "gate-equiv", "speed", "project", "run", "ledger"])
    ap.add_argument("--cap-hours", type=float, default=CAP_HOURS)
    ap.add_argument("--authorized-gpu-run", action="store_true", help="required for any GPU run; only pass after explicit author authorization")
    a = ap.parse_args()
    if a.cap_hours > CAP_HOURS:
        sys.exit(f"cap above the authorized {CAP_HOURS} h is not permitted")
    configure(OUT)
    s = make_sample2()
    if a.cmd == "plan":
        print({k: (len(v) if isinstance(v, list) else v) for k, v in s.items()})
        assert_heldout_indices(s["run_order"], s)
        print("run order is a permutation of the held-out set:", sorted(s["run_order"]) == s["heldout389"])
        return
    if a.cmd == "ledger":
        print(f"Stage 2 cumulative GPU wall time: {ledger_hours():.3f} h of {a.cap_hours} h")
        return
    if a.cmd == "check-clean":
        import subprocess
        sys.exit(subprocess.run([sys.executable, str(HERE / "tools" / "check_heldout_clean.py")]).returncode)
    if a.cmd == "project":
        r = project(OUT, a.cap_hours, RS1.ledger_seconds())
        print(json.dumps(r, indent=1))
        return
    if not a.authorized_gpu_run:
        sys.exit("Refusing to run: GPU runs need explicit author authorization (--authorized-gpu-run).")

    import stage1_gates
    data = RS1.Data()
    if a.cmd == "gate-equiv":
        def f(be, dyn, tok, wb, prior, t0):
            ok = stage1_gates.gate_equiv(be, dyn, tok, wb, s, data, OUT)
            if not ok:
                print("[STOP] correctness check failed; Stage 2 does not proceed.", flush=True)
            return ok
        RS1.gpu_process("gate-equiv", f, a.cap_hours)
        rep = json.loads((OUT / "gates" / "gate_equiv.json").read_text())
        sys.exit(0 if rep.get("all_passed") else 3)
    if a.cmd == "speed":
        RS1.gpu_process("speed", lambda be, dyn, tok, wb, prior, t0: stage1_gates.speed_check(be, tok, data, s, OUT), a.cap_hours)
        return
    if a.cmd == "run":
        gate = OUT / "gates" / "gate_equiv.json"
        if not gate.exists() or not json.loads(gate.read_text()).get("all_passed"):
            sys.exit("Refusing to run: the short GPU correctness check has not passed.")
        order = s["run_order"]
        assert_heldout_indices(order, s)
        rep = RS1.verify_checkpoint(OUT / "eval", None, order, ARMS)
        print(f"[checkpoint] {rep['n_complete_all_arms']} problems complete on all arms; order/repeat/skip verification "
              f"{'OK' if rep['ok'] else 'FAILED'}", flush=True)
        if not rep["ok"]:
            sys.exit("Refusing to run: checkpoint verification failed: " + "; ".join(rep["problems"]))
        done_all = set.intersection(*[RS1.load_done(OUT / "eval" / f"{x}.jsonl") for x in ARMS])
        remaining = [i for i in order if i not in done_all]
        left = a.cap_hours * 3600 - RS1.ledger_seconds()
        print(f"[stage2] remaining problems {len(remaining)}; next = run_order[{len(done_all)}]; {left / 3600:.3f} h left under the "
              f"absolute {a.cap_hours} h ceiling (no extension)", flush=True)
        RS1.gpu_process("eval", lambda be, dyn, tok, wb, prior, t0: RS1.run_problems(
            "eval", be, tok, wb, order, "test", ARMS, data, OUT, cap_hours=a.cap_hours, t_proc0=t0, prior=prior, a2=True), a.cap_hours)
        reason = RS1.SESSION.get("stop_reason")
        print(f"[stage2] process ended: {reason}; problems completed this process: {RS1.SESSION.get('problems_completed')}", flush=True)
        if reason == "integrity failure" or (reason or "").startswith("hardware"):
            sys.exit(2)
        if reason and reason.startswith("absolute ceiling"):
            sys.exit(4)


if __name__ == "__main__":
    main()
