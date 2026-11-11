"""v3 CPU validation before the lock (PROTOCOL_V3.md section 10). Runs, in order, and records the outcome of each:
  1. tests/run_v3_tests.py --static --real --write   (tiny Qwen2 + tiny GQA Llama; real Qwen2.5-0.5B and Llama-3.2-1B on CPU; E1/E1L locks verify)
  2. tests/test_analyze_v3.py                        (analysis on simulated truth)
  2b. tests/cpu_harness_v3_paths.py --write          (the driver's gate / calibration / G6 / Tier A / analysis code end to end on CPU)
  2c. tests/test_v3_repair.py, tests/test_v3_resume.py --write   (amendment 1: repair script; crash injection A-G)
  3. tests/mutation_check_v3.py --write              (every v3 mutant caught)
  4. tests/mutation_check_controls.py, tests/mutation_check_e1l.py  (the locked E1 / E1L mutation checks, re-run unchanged: rv2.controls is used for C1/C2)
  -> validation_logs/v3_validation_summary.json
  venv/Scripts/python.exe research_v2/tools/v3_cpu_validation.py
"""
import json
import subprocess
import sys
import time
from pathlib import Path

RV = Path(__file__).resolve().parents[1]
PY = sys.executable
STEPS = [("v3 CPU suite (tiny + real + static)", [str(RV / "tests" / "run_v3_tests.py"), "--static", "--real", "--write"]),
         ("v3 analysis tests (simulated truth)", [str(RV / "tests" / "test_analyze_v3.py")]),
         ("v3 CPU harness of the driver's GPU code paths", [str(RV / "tests" / "cpu_harness_v3_paths.py"), "--write"]),
         ("v3 repair-script tests (amendment 1)", [str(RV / "tests" / "test_v3_repair.py")]),
         ("v3 crash-injection tests (amendment 1)", [str(RV / "tests" / "test_v3_resume.py"), "--write"]),
         ("v3 mutation check", [str(RV / "tests" / "mutation_check_v3.py"), "--write"]),
         ("E1 mutation check (unchanged)", [str(RV / "tests" / "mutation_check_controls.py")]),
         ("E1L mutation check (unchanged)", [str(RV / "tests" / "mutation_check_e1l.py")])]


def main():
    out = []
    for name, cmd in STEPS:
        t0 = time.time()
        r = subprocess.run([PY, *cmd], capture_output=True, text=True, encoding="utf-8", errors="replace")
        tail = [l for l in r.stdout.splitlines() if l.strip()][-6:]
        out.append({"step": name, "returncode": r.returncode, "passed": r.returncode == 0, "seconds": round(time.time() - t0, 1), "stdout_tail": tail})
        print(("PASS " if r.returncode == 0 else "FAIL ") + name, f"({out[-1]['seconds']} s)", *tail[-2:], sep="\n  ", flush=True)
    (RV / "validation_logs").mkdir(exist_ok=True)
    (RV / "validation_logs" / "v3_validation_summary.json").write_text(json.dumps({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                                                                 "all_passed": all(s["passed"] for s in out), "steps": out}, indent=1))
    sys.exit(0 if all(s["passed"] for s in out) else 1)


if __name__ == "__main__":
    main()
