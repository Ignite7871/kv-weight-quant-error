"""GM (PROTOCOL_V5.md section 5): the CUDA check on the cloud GPU, once, before U0. Run through `run_v5.py gm --authorized-gpu-run` (which checks
AUTHORIZED_V5_GM.txt, the lock, the SKU, the deterministic settings, the image record, the no-token rule and the ledgers). Writes v5_runs/gm.json;
then `tools/lock_v5.py --append-gm`.

  1. the v5 core test suite (v5/tests/test_rv5.py) on CUDA: real bitsandbytes NF4 and HQQ CUDA paths, FP16 and BF16, the new geometries;
  2. the full v5 mutation check (v5/tests/mutation_check_v5.py) on CUDA: every mutant must be caught;
  3. the environment: driver version, CUDA version and ECC mode from nvidia-smi, torch's CUDA runtime, the deterministic settings as seen by a CUDA
     process, and the first-boot image digest record. GM passes only if all of them are on record and the settings are active.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

RV = Path(__file__).resolve().parents[1]
V5 = RV / "v5"
OUT = Path(os.environ.get("V5_RUNS", RV / "v5_runs"))


def verdict(r1, r2, t, m, env_complete, determinism_ok, image):
    """GM's pass rule: both suites clean, every test passed, every mutant caught, the environment record complete, an image record."""
    return bool(r1 == 0 and r2 == 0 and t.get("n") and t.get("passed") == t.get("n") and m.get("all_caught") and env_complete and determinism_ok
                and image)


def main():
    if "--authorized-gpu-run" not in sys.argv:
        sys.exit("GM runs only through run_v5.py gm --authorized-gpu-run")
    sys.path.insert(0, str(V5))
    import v5_ops as O
    lid = subprocess.run([sys.executable, str(RV / "tools" / "lock_v5.py"), "--lock-id"], capture_output=True, **O.TEXT).stdout.strip() or None
    sys.path.insert(0, str(RV / "tools"))
    import lock_v5 as LK
    dev = LK.active_deviation()
    sfx = f"_{dev}" if dev else ""
    ok, why = O.check_auth(OUT, "GM", lid)
    if not ok:
        sys.exit(f"Refusing: {why}")
    env = O.gpu_environment()
    image = O.image_record(OUT)
    rep = {"started_utc": O.utc(time.time()), "gpu": O.gpu_name(), "gpu_environment": env, "image": image}
    r1 = subprocess.run([sys.executable, str(V5 / "tests" / "test_rv5.py"), "--device", "cuda", "--write", str(OUT / f"gm{sfx}_test_rv5.json")])
    r2 = subprocess.run([sys.executable, str(V5 / "tests" / "mutation_check_v5.py"), "--device", "cuda", "--write", str(OUT / f"gm{sfx}_mutation.json")])
    code = """
import sys, json
sys.path.insert(0, r'%s'); sys.path.insert(0, r'%s'); sys.path.insert(0, r'%s')
import rv4, torch
det = rv4.apply_determinism()
print(json.dumps({'determinism': det, 'torch_cuda': torch.version.cuda}))
""" % (RV, RV / "v4", V5)
    r3 = subprocess.run([sys.executable, "-c", code], capture_output=True, **O.TEXT)
    res = json.loads(r3.stdout.strip().splitlines()[-1]) if r3.returncode == 0 else {"error": r3.stderr[-2000:]}
    det = res.get("determinism", {})
    det_ok = det.get("cublas_workspace_config") == ":4096:8" and det.get("deterministic_algorithms") is True
    t = json.loads((OUT / f"gm{sfx}_test_rv5.json").read_text()) if (OUT / f"gm{sfx}_test_rv5.json").exists() else {}
    m = json.loads((OUT / f"gm{sfx}_mutation.json").read_text()) if (OUT / f"gm{sfx}_mutation.json").exists() else {}
    rep.update(tests_passed=f"{t.get('passed')}/{t.get('n')}", mutants_caught=f"{m.get('caught')}/{m.get('n')}", determinism=det,
               torch_cuda=res.get("torch_cuda"), all_passed=verdict(r1.returncode, r2.returncode, t, m, O.gpu_environment_complete(env), det_ok, image),
               finished_utc=O.utc(time.time()))
    (OUT / f"gm{sfx}.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))
    return 0 if rep["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
