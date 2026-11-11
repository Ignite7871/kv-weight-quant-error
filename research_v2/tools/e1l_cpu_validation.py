"""Run every Llama-E1 CPU check and record the outcome in validation_logs/e1l_cpu_validation.json (part of the lock).
  venv/Scripts/python.exe research_v2/tools/e1l_cpu_validation.py            (about 25 minutes on CPU; no GPU)"""

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

RV = Path(__file__).resolve().parents[1]
ROOT = RV.parent
PY = sys.executable
STEPS = [("unit_tests_tiny_llama_plus_real_llama_cpu", [PY, str(RV / "tests" / "run_e1l_tests.py"), "--real"]),
         ("mutation_check_19_mutants", [PY, str(RV / "tests" / "mutation_check_e1l.py")]),
         ("analysis_tests_locked_analyzer", [PY, str(RV / "tests" / "test_analyze_controls.py")]),
         ("plan_consistency", [PY, str(RV / "run_e1l.py"), "plan"]),
         ("cpu_harness_gpu_only_paths_real_llama", [PY, str(RV / "tests" / "cpu_harness_e1l_paths.py")])]


def main():
    out = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "python": platform.python_version(), "steps": {}}
    import bitsandbytes, torch, transformers
    out["versions"] = {"torch": torch.__version__, "transformers": transformers.__version__, "bitsandbytes": bitsandbytes.__version__}
    out["git_head_at_run"] = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    ok = True
    for name, cmd in STEPS:
        t = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        lines = [l for l in (r.stdout + r.stderr).splitlines() if l.strip() and "Warning" not in l and "_check_is_size" not in l and "Loading weights" not in l]
        out["steps"][name] = {"exit_code": r.returncode, "seconds": round(time.time() - t, 1), "tail": lines[-26:]}
        ok &= r.returncode == 0
        print(name, "exit", r.returncode, flush=True)
    out["all_passed"] = ok
    (RV / "validation_logs").mkdir(exist_ok=True)
    (RV / "validation_logs" / "e1l_cpu_validation.json").write_text(json.dumps(out, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
