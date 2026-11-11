"""v3 GPU mutation / correctness check (PROTOCOL_V3.md section 9): the CPU suite (tests/run_v3_tests.py) and every core mutant
(tests/mutation_check_v3.py) on CUDA, with a tiny model of each family and the real DynamicLinear / bitsandbytes CUDA NF4 kernels. Runs once,
before the first model's gates. Wall time is appended to the v3 ledger; amendment 1: a heartbeat (Stage 1 progress file, every 15 s) means a
hard kill is charged to the ledger by the next GPU process, and an unclosed earlier process is charged before this one starts.
  venv/Scripts/python.exe research_v2/tools/v3_gpu_mutation_check.py --authorized-gpu-run   -> v3_runs/gpu_mutation_check.json
"""
import json
import subprocess
import sys
from pathlib import Path

RV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RV))
sys.path.insert(0, str(RV / "tests"))
OUT = RV / "v3_runs"


def main():
    auth = OUT / "AUTHORIZED_BY_AUTHORS_V3.txt"
    lid = subprocess.run([sys.executable, str(RV / "tools" / "lock_v3.py"), "--lock-id"], capture_output=True, text=True).stdout.strip()
    txt = auth.read_text(encoding="utf-8", errors="replace") if auth.exists() else ""
    if "--authorized-gpu-run" not in sys.argv or not lid or "PROTOCOL_V3" not in txt or lid not in txt:
        sys.exit("Refusing: needs --authorized-gpu-run and v3_runs/AUTHORIZED_BY_AUTHORS_V3.txt containing 'PROTOCOL_V3' and the lock id.")
    if subprocess.run([sys.executable, str(RV / "tools" / "lock_v3.py"), "--verify"], capture_output=True).returncode != 0:
        sys.exit("Refusing: the v3 lock does not verify.")
    import mutation_check_v3 as MC
    import torch
    T = MC.T
    ok = False
    with T.R.ledger_heartbeat("gpu_mutation_check") as led:
        T.install("cuda")
        base = T.run_suite()
        res = []
        for mut in MC.CORE:
            MC.restore()
            T._BE.clear()
            try:
                mut()
                fails = T.run_suite()
            except Exception as e:  # noqa: BLE001
                fails = [("mutant raised", f"{type(e).__name__}: {e}")]
            finally:
                MC.restore()
            res.append({"mutant": mut.__name__, "caught": bool(fails), "first": [list(f) for f in fails[:2]]})
            print(("CAUGHT " if fails else "MISSED ") + mut.__name__, flush=True)
        ok = not base and all(r["caught"] for r in res)
        rep = {"device": torch.cuda.get_device_name(0), "suite_failures": [list(f) for f in base], "mutants": res,
               "caught": sum(r["caught"] for r in res), "total": len(res), "all_passed": ok}
        (OUT / "gpu_mutation_check.json").write_text(json.dumps(rep, indent=1))
        print(json.dumps({k: v for k, v in rep.items() if k != "mutants"}, indent=1))
        led["completed"] = ok
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
