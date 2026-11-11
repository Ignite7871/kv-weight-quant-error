"""GM (PROTOCOL_V4.md section 5): the CUDA check on the cloud GPU, once, before Step 0. Run through `run_v4.py gm --authorized-gpu-run`
(which checks AUTHORIZED_V4_GM.txt, the lock, the SKU, the deterministic settings, the image record and the ledgers). Writes v4_runs/gm.json;
then `tools/lock_v4.py --append-gm`.

  1. the v4 core test suite (tests/test_rv4.py) on CUDA: real bitsandbytes NF4 CUDA kernels, FP16 and BF16;
  2. the full mutation check (tests/mutation_check_v4.py) on CUDA: every mutant must be caught;
  3. r4: the environment. Driver version, CUDA version and ECC mode from nvidia-smi, torch's CUDA runtime, the deterministic settings as seen by a
     CUDA process, and the first-boot image digest record. GM passes only if all of them are on record and the settings are active.
  0. r8: before the suites, the locked v3 records that the Step 0 tests need (tools/v4_v3_records.py verify: every file of
     v4/v3_records_manifest.json with its hash, each calibration.json equal to the V3 lock). If they do not verify, the suites do not run and
     GM fails (the Step 0 tests would otherwise have nothing to check).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

RV = Path(__file__).resolve().parents[1]
V4 = RV / "v4"
OUT = Path(os.environ.get("V4_RUNS", RV / "v4_runs"))


def verdict(r1, r2, t, m, env_complete, determinism_ok, image, v3_records_ok):
    """GM's pass rule: both suites clean, every test passed, every mutant caught, the r4 environment record complete, and (r8) the v3 records
    verified."""
    return bool(r1 == 0 and r2 == 0 and t.get("passed") == t.get("n") and m.get("all_caught") and env_complete and determinism_ok and image
                and v3_records_ok is True)


def main():
    if "--authorized-gpu-run" not in sys.argv:
        sys.exit("GM runs only through run_v4.py gm --authorized-gpu-run")
    sys.path.insert(0, str(V4))
    import v4_ops as O
    lid = subprocess.run([sys.executable, str(RV / "tools" / "lock_v4.py"), "--lock-id"], capture_output=True, **O.TEXT).stdout.strip() or None
    sys.path.insert(0, str(RV / "tools"))
    import lock_v4 as LK
    dev = LK.active_deviation()                                       # V4-D1: GM after a deviation writes gm_<ID>*.json; the first GM record stays
    sfx = f"_{dev}" if dev else ""
    ok, why = O.check_auth(OUT, "GM", lid)
    if not ok:
        sys.exit(f"Refusing: {why}")
    env = O.gpu_environment()
    image = O.image_record(OUT)
    rep = {"started_utc": O.utc(time.time()), "gpu": O.gpu_name(), "gpu_environment": env, "image": image}
    sys.path.insert(0, str(RV / "tools"))
    import v4_v3_records as V3R
    v3_ok, v3_probs = V3R.verify(RV)
    rep["v3_records"] = {"ok": v3_ok, "problems": v3_probs[:30]}
    if not v3_ok:
        rep.update(all_passed=False, finished_utc=O.utc(time.time()))
        (OUT / f"gm{sfx}.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
        print(json.dumps(rep, indent=1))
        print("GM refused to run the suites: the v3 records do not verify (tools/v4_v3_records.py)")
        return 1
    r1 = subprocess.run([sys.executable, str(V4 / "tests" / "test_rv4.py"), "--device", "cuda", "--write", str(OUT / f"gm{sfx}_test_rv4.json")])
    r2 = subprocess.run([sys.executable, str(V4 / "tests" / "mutation_check_v4.py"), "--device", "cuda", "--write", str(OUT / f"gm{sfx}_mutation.json")])
    code = """
import sys, json
sys.path.insert(0, r'%s'); sys.path.insert(0, r'%s'); sys.path.insert(0, r'%s')
import rv4, torch
det = rv4.apply_determinism()
print(json.dumps({'determinism': det, 'torch_cuda': torch.version.cuda}))
""" % (RV, V4, V4 / "tests")
    r3 = subprocess.run([sys.executable, "-c", code], capture_output=True, **O.TEXT)
    res = json.loads(r3.stdout.strip().splitlines()[-1]) if r3.returncode == 0 else {"error": r3.stderr[-2000:]}
    det = res.get("determinism", {})
    det_ok = det.get("cublas_workspace_config") == ":4096:8" and det.get("deterministic_algorithms") is True
    t = json.loads((OUT / f"gm{sfx}_test_rv4.json").read_text()) if (OUT / f"gm{sfx}_test_rv4.json").exists() else {}
    m = json.loads((OUT / f"gm{sfx}_mutation.json").read_text()) if (OUT / f"gm{sfx}_mutation.json").exists() else {}
    rep.update(tests_passed=f"{t.get('passed')}/{t.get('n')}", mutants_caught=f"{m.get('caught')}/{m.get('n')}",
               determinism=det, torch_cuda=res.get("torch_cuda"),
               all_passed=verdict(r1.returncode, r2.returncode, t, m, O.gpu_environment_complete(env), det_ok, image, v3_ok),
               finished_utc=O.utc(time.time()))
    (OUT / f"gm{sfx}.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))
    return 0 if rep["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
