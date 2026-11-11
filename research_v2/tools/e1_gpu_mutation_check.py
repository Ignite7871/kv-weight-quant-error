"""GPU correctness + mutation check for the E1 controls (authorization item: 'run GPU mutation and correctness checks').
Runs the SAME suites as tests/run_controls_tests.py and tests/mutation_check_controls.py, but on a tiny random Qwen2 placed on the CUDA device with the
real DynamicLinear / bitsandbytes CUDA NF4 kernels, so every semantic test (bitwise NAT == engine cache-only, alpha = 0 == FP16, shared token history,
earlier entries never modified, ...) and every injected bug is exercised on the real kernels. The wall time is appended to the E1 GPU ledger.
Does not modify any locked file (it only monkeypatches the test module's backend factory and prompt tensors at run time).

  venv/Scripts/python.exe research_v2/tools/e1_gpu_mutation_check.py     -> e1_runs/gates/gpu_mutation_check.json
"""

import json
import sys
import time
from pathlib import Path

RV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RV / "tests"))
sys.path.insert(0, str(RV))

from transformers import AutoTokenizer  # noqa: F401,E402

import torch  # noqa: E402

import rv2.controls as C  # noqa: E402
import run_controls_tests as T  # noqa: E402
import mutation_check_controls as M  # noqa: E402
from dynamic_linear import RoutingState  # noqa: E402
from dynamic_model import replace_linear_with_dynamic  # noqa: E402
from rv2.backends import tiny_qwen2  # noqa: E402
from rv2.engine import Backend  # noqa: E402

OUT = RV / "e1_runs"


def gpu_backend(seed=0):
    model = tiny_qwen2(seed).to(torch.float16).to("cuda").eval()
    state = RoutingState()
    assert replace_linear_with_dynamic(model, state), "no layers replaced"
    return Backend(model, state, "cuda")


def main():
    if not torch.cuda.is_available():
        sys.exit("CUDA not available")
    t0 = time.time()
    T.be = gpu_backend
    T.PROMPTS = [p.to("cuda") for p in T.PROMPTS]
    T.test_names = None
    rep = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "device": torch.cuda.get_device_name(0)}
    # 1. correctness suite on the GPU
    T.RESULTS.clear()
    names = [k for k in dir(T) if k.startswith("c") and k[1:3].isdigit() and callable(getattr(T, k))]
    for n in sorted(names):
        getattr(T, n)()
    rep["suite"] = [(n, s, m) for n, s, m in T.RESULTS]
    suite_ok = all(s == "PASS" for _, s, _ in T.RESULTS)
    print("GPU suite:", sum(s == "PASS" for _, s, _ in T.RESULTS), "/", len(T.RESULTS), flush=True)
    # 2. mutation check on the GPU
    M.T = T
    M.NAMES = names
    caught, missed = [], []
    if suite_ok:
        for name, mut in M.MUTANTS:
            for k, v in M.ORIG.items():
                setattr(C, k, v)
            mut()
            fails = M.run_suite()
            for k, v in M.ORIG.items():
                setattr(C, k, v)
            (caught if fails else missed).append(name)
            print(("CAUGHT " if fails else "MISSED "), name, [n for n, _ in fails][:4], flush=True)
    rep["mutants_caught"], rep["mutants_missed"] = caught, missed
    rep["passed"] = bool(suite_ok and not missed and len(caught) == len(M.MUTANTS))
    secs = time.time() - t0
    rep["seconds"] = secs
    OUT.mkdir(exist_ok=True)
    (OUT / "gates").mkdir(exist_ok=True)
    (OUT / "gates" / "gpu_mutation_check.json").write_text(json.dumps(rep, indent=1))
    with open(OUT / "gpu_ledger.jsonl", "a") as f:
        f.write(json.dumps({"start_utc": rep["started_utc"], "seconds": secs, "block": "gpu_mutation_check", "completed": rep["passed"]}) + "\n")
    print(json.dumps({k: v for k, v in rep.items() if k != "suite"}, indent=1))
    sys.exit(0 if rep["passed"] else 1)


if __name__ == "__main__":
    main()
