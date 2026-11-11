"""Run every E1 CPU check and record the outcome in validation_logs/e1_cpu_validation.json (part of the lock).
  venv/Scripts/python.exe research_v2/tools/e1_cpu_validation.py            (about 12 minutes on CPU; no GPU)"""

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

RV = Path(__file__).resolve().parents[1]
ROOT = RV.parent
PY = sys.executable
STEPS = [("unit_tests_real_model_included", [PY, str(RV / "tests" / "run_controls_tests.py"), "--real"]),
         ("mutation_check", [PY, str(RV / "tests" / "mutation_check_controls.py")]),
         ("analysis_tests", [PY, str(RV / "tests" / "test_analyze_controls.py")]),
         ("plan_consistency", [PY, str(RV / "run_controls.py"), "plan"]),
         ("cpu_smoke_real_model_io", [PY, str(RV / "run_controls.py"), "cpu-smoke"])]


def main():
    out = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "python": platform.python_version(), "steps": {}}
    import torch, transformers, bitsandbytes
    out["versions"] = {"torch": torch.__version__, "transformers": transformers.__version__, "bitsandbytes": bitsandbytes.__version__}
    out["git_head_at_run"] = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    ok = True
    for name, cmd in STEPS:
        t = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        lines = [l for l in (r.stdout + r.stderr).splitlines() if l.strip() and "Warning" not in l and "_check_is_size" not in l]
        out["steps"][name] = {"exit_code": r.returncode, "seconds": round(time.time() - t, 1), "tail": lines[-18:]}
        ok &= r.returncode == 0
        print(name, "exit", r.returncode, flush=True)
    out["all_passed"] = ok
    (RV / "validation_logs").mkdir(exist_ok=True)
    (RV / "validation_logs" / "e1_cpu_validation.json").write_text(json.dumps(out, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
