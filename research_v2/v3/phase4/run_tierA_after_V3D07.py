"""Launcher for llama-base Tier A after the G6 failure (DEVIATION_LOG_V3 V3-D07, author decision 2026-10-31). Not locked.

The locked driver (run_v3.py) is imported and run unchanged, except for ONE guard: run_v3.g6_passed returns True for llama-base, and only
while v3_runs/llama-base/g6.json is byte-identical to the failed G6 record that the authors' decision refers to (sha256 below). Every other
guard is the locked code's own:
  - authorization file and lock id;
  - lock, E1/E1L lock and model-hash verification;
  - fixed run order; gates passed; calibration frozen in the lock; design locked;
  - the 25.0 h ceiling: the locked CAP_HOURS, and block_loop stops before a problem that could cross it (TRUNCATED).
No arm, sample, seed, hypothesis or rule changes.

  venv/Scripts/python.exe research_v2/v3/phase4/run_tierA_after_V3D07.py            # runs: run_v3.py tierA --model llama-base --authorized-gpu-run
  venv/Scripts/python.exe research_v2/v3/phase4/run_tierA_after_V3D07.py --check    # CPU only: all permission checks, then exit
"""
import hashlib
import sys
from pathlib import Path

RV = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RV))

import run_v3 as R                               # noqa: E402

MODEL = "llama-base"
G6_SHA256 = "210d58ca917684a3f57d4604468ba096672e66317c9491fce1cb3db20e794348"   # the failed G6 record (drift +18.06 %, pass false)


def waived(m):
    if m != MODEL:
        return False
    p = R.mdir(m) / "g6.json"
    return p.exists() and hashlib.sha256(p.read_bytes()).hexdigest() == G6_SHA256


def main():
    check = "--check" in sys.argv[1:]
    if [x for x in sys.argv[1:] if x != "--check"]:
        sys.exit("no arguments other than --check: this launcher runs only `tierA --model llama-base --authorized-gpu-run`")
    if not waived(MODEL):
        sys.exit(f"refusing: {MODEL} g6.json is missing or differs from the failed G6 record named in V3-D07")
    orig = R.g6_passed
    R.g6_passed = lambda m: True if (m == MODEL and waived(m)) else orig(m)
    print(f"[V3-D07] G6 guard waived for {MODEL} only (failed record sha256 {G6_SHA256[:16]}...); all other guards and the "
          f"{R.CAP_HOURS} h ceiling unchanged", flush=True)
    if check:
        a = type("A", (), {"authorized_gpu_run": True})()
        R.configure(MODEL)
        R.require_gpu_permission(a, MODEL, "tierA")
        R.require_intact_records(MODEL)
        print("[V3-D07] --check: every other permission check passes; no GPU used", flush=True)
        return
    sys.argv = [str(RV / "run_v3.py"), "tierA", "--model", MODEL, "--authorized-gpu-run"]
    R.main()


if __name__ == "__main__":
    main()
