"""GPU correctness + mutation check for the Llama E1 replication: the c01-c13 + l01-l04 suite and all 19 mutants on a tiny random Llama (GQA, head_dim 64) placed on
CUDA with the real DynamicLinear / bitsandbytes CUDA NF4 kernels. Wall time is appended to the Llama E1 ledger.
  venv/Scripts/python.exe research_v2/tools/e1l_gpu_mutation_check.py --authorized-gpu-run   -> e1l_runs/gates/gpu_mutation_check.json
(needs the same authorization file as the gates)"""
import json
import sys
import time
from pathlib import Path

RV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RV / "tests"))
sys.path.insert(0, str(RV))
import torch  # noqa: E402

import e1l_support as LS  # noqa: E402
import mutation_check_controls as M  # noqa: E402
import mutation_check_e1l as ME  # noqa: E402
import rv2.controls as C  # noqa: E402
import run_e1l_tests as E  # noqa: E402

OUT = RV / "e1l_runs"


def main():
    auth = OUT / "AUTHORIZED_BY_AUTHORS_LLAMA.txt"
    if "--authorized-gpu-run" not in sys.argv or not auth.exists() or "Llama" not in auth.read_text(encoding="utf-8", errors="replace"):
        sys.exit("Refusing: needs --authorized-gpu-run and e1l_runs/AUTHORIZED_BY_AUTHORS_LLAMA.txt containing 'Llama'.")
    t0 = time.time()
    E.install(LS.tiny_backend_gpu, "cuda")
    names = E.register_into_T()
    E.T.RESULTS.clear()
    for n in names:
        getattr(E.T, n)()
    suite = list(E.T.RESULTS)
    suite_ok = all(s == "PASS" for _, s, _ in suite)
    M.T, M.NAMES = E.T, names
    mutants = list(M.MUTANTS) + ME.extra
    caught, missed = [], []
    if suite_ok:
        for name, mut in mutants:
            for k, v in M.ORIG.items():
                setattr(C, k, v)
            mut()
            fails = M.run_suite()
            for k, v in M.ORIG.items():
                setattr(C, k, v)
            (caught if fails else missed).append(name)
            print("CAUGHT " if fails else "MISSED ", name, flush=True)
    rep = {"device": torch.cuda.get_device_name(0), "suite": suite, "mutants_caught": caught, "mutants_missed": missed,
           "passed": bool(suite_ok and not missed and len(caught) == len(mutants)), "seconds": time.time() - t0}
    (OUT / "gates").mkdir(parents=True, exist_ok=True)
    (OUT / "gates" / "gpu_mutation_check.json").write_text(json.dumps(rep, indent=1))
    with open(OUT / "gpu_ledger.jsonl", "a") as f:
        f.write(json.dumps({"start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)), "seconds": rep["seconds"], "block": "gpu_mutation_check", "completed": rep["passed"]}) + "\n")
    print(json.dumps({k: v for k, v in rep.items() if k != "suite"}, indent=1))
    sys.exit(0 if rep["passed"] else 1)


if __name__ == "__main__":
    main()
