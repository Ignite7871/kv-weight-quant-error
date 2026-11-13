"""V5-D3: the restore set for the U1 re-execution on a new Standard volume (the first volume was deleted with U1's completed checkpoint unfetched,
V5-A10). Built on the laptop from verified records only; it copies files and edits none.

  python research_v2/tools/v5_d3_prepare.py --out <dir> [--pod-id P --volume-id V --volume-created-utc YYYY-MM-DDTHH:MM:SSZ --volume-gb N]

What goes into <dir>/d3_restore.tar (with SHA256SUMS inside and d3_restore.tar.sha256 beside it), to be extracted into /workspace/v5_runs
BEFORE cloud/pod_setup_v5.sh runs (so the write-once image record is the first volume's and a changed digest is logged, not silently replaced):
  - every run file the lock's appended lines name, each checked against its line (GM, ANCHOR, DESIGN, every CALIBRATION, HQQ U4/U6); the one
    exception is gm_D2.json, which was never checkpointed and is declared lost in V5-D3's record;
  - what the locked order rules read for U1: U0/test_complete.json; and U2..U6/test_complete.json so that no rule can mark them not run;
  - U6/d1_stop.json, the probes, DESIGN/, gm_test_rv5.json, gm_mutation.json, image_digest.json, and the cost and GPU ledgers (with the final
    billing line);
  - U1/d3_reference/: the test records of the 13 problems that survive in U1's first checkpoint (U1_20261112T065200Z.tar, verified against its
    .sha256 and its SHA256SUMS), with d3_reference.json (ids in the locked order, source tar and hashes). Nothing else of U1 except its locked
    calibration.json: download, gates, calibration and the test run again from scratch;
  - cost_config.json for the new volume, when its ids are given (the first volume's config stays on the laptop).
Not restored: checkpoints/, _downloads/ (the model is downloaded and verified again), status.json, pod_ssh.json, known_hosts, authorization files.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

RV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RV / "v5"))
sys.path.insert(0, str(RV / "tools"))
import v5_ops as O  # noqa: E402
import lock_v5 as LK  # noqa: E402

RUNS = RV / "v5_runs"
REF_TAR = "U1_20261112T065200Z.tar"
REF_SHA = "db2c16ebc03c453b41fe8d1829431e711b84d89a005ca6e3b8be06f85a3359c1"
LOST = {"gm_D2.json"}
EXTRA = ["gm_test_rv5.json", "gm_mutation.json", "image_digest.json", "cost_ledger.jsonl", "gpu_ledger.jsonl", "U0/test_complete.json",
         "U6/d1_stop.json"] + [f"U{k}/test_complete.json" for k in range(2, 7)]


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--pod-id")
    ap.add_argument("--volume-id")
    ap.add_argument("--volume-created-utc")
    ap.add_argument("--volume-gb", type=int, default=50)
    a = ap.parse_args(argv)
    out = Path(a.out)
    stage = out / "stage"
    if stage.exists():
        sys.exit(f"{stage} exists; use a new directory")
    stage.mkdir(parents=True)

    files = {}
    lines = [l for l in LK.LOCK.read_text(encoding="utf-8").splitlines() if l.startswith(LK.APPENDIX) and not l.startswith("DEVIATION ")]
    for l in lines:
        p = LK.appendix_file(l)
        rel = p.relative_to(LK.OUT).as_posix()
        h = l.split(": ", 1)[1].split()[0]
        if rel in LOST:
            if p.exists():
                sys.exit(f"{rel} exists here although V5-D3 declares it lost")
            continue
        src = RUNS / rel
        if not src.exists() or LK.file_sha(src) != h:
            sys.exit(f"{rel} is missing here or differs from its lock line")
        files[rel] = src
    for rel in EXTRA:
        src = RUNS / rel
        if not src.exists():
            sys.exit(f"{rel} is missing here")
        files[rel] = src
    for src in sorted((RUNS / "probes").glob("*.json")) + sorted(q for q in (RUNS / "DESIGN").rglob("*") if q.is_file()):
        files[src.relative_to(RUNS).as_posix()] = src
    for rel, src in files.items():
        dst = stage / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)

    tar = RUNS / "checkpoints" / REF_TAR
    if O.sha256_file(tar) != REF_SHA or not O.verify_sha256_file(Path(str(tar) + ".sha256")):
        sys.exit(f"{REF_TAR} does not match its recorded SHA-256")
    with tempfile.TemporaryDirectory() as td:
        with tarfile.open(tar) as tf:
            for m in tf.getmembers():
                if m.issym() or m.islnk() or ".." in Path(m.name).parts:
                    sys.exit(f"unsafe member {m.name}")
            tf.extractall(td)
        if not O.verify_sha256_file(Path(td) / "SHA256SUMS"):
            sys.exit("a member of the reference checkpoint does not match its SHA256SUMS")
        src = Path(td) / "U1" / "test"
        ref = stage / "U1" / "d3_reference"
        shutil.copytree(src, ref / "test")
        if (Path(td) / "U1" / "calibration.json").read_bytes() != (RUNS / "U1" / "calibration.json").read_bytes():
            sys.exit("the reference checkpoint's U1 calibration differs from the laptop's")
    ids = [json.loads(l)["index"] for l in (ref / "test" / "NAT.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    hashes = {q.relative_to(ref).as_posix(): sha(q.read_bytes()) for q in sorted((ref / "test").rglob("*")) if q.is_file()}
    (ref / "d3_reference.json").write_text(json.dumps({"deviation": "V5-D3", "ids": ids, "source_tar": REF_TAR, "source_tar_sha256": REF_SHA,
                                                        "files": hashes}, indent=1), encoding="utf-8")

    if a.pod_id or a.volume_id or a.volume_created_utc:
        if not (a.pod_id and a.volume_id and a.volume_created_utc):
            sys.exit("--pod-id, --volume-id and --volume-created-utc go together")
        O.parse_utc(a.volume_created_utc)
        cfg = json.loads((RUNS / "cost_config.json").read_text(encoding="utf-8"))
        cfg = {k: v for k, v in cfg.items() if k != "volume_deleted_utc"}
        cfg.update(pod_id=a.pod_id, volume_id=a.volume_id, volume_created_utc=a.volume_created_utc, volume_gb=a.volume_gb, volume_tier="standard",
                   storage_usd_per_gb_month=0.07, datacenter="EU-SE-1")
        (stage / "cost_config.json").write_text(json.dumps(cfg), encoding="utf-8")

    sums = [f"{sha(q.read_bytes())}  {q.relative_to(stage).as_posix()}" for q in sorted(stage.rglob("*")) if q.is_file()]
    t = out / "d3_restore.tar"
    with tarfile.open(t, "w") as tf:
        for q in sorted(stage.rglob("*")):
            if q.is_file():
                tf.add(q, arcname=q.relative_to(stage).as_posix())
        b = ("\n".join(sums) + "\n").encode()
        ti = tarfile.TarInfo("SHA256SUMS")
        ti.size = len(b)
        tf.addfile(ti, io.BytesIO(b))
    Path(str(t) + ".sha256").write_bytes(f"{O.sha256_file(t)}  {t.name}\n".encode())
    print(json.dumps({"tar": str(t), "sha256": O.sha256_file(t), "n_files": len(sums), "reference_ids": ids,
                      "cost_config": (stage / "cost_config.json").exists()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
