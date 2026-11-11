"""r8 (PROTOCOL_V4 section 15): the locked v3 records that the Step 0 tests (test_analyze_v4 a04, a16) need on the pod.

research_v2/v3_runs is not tracked in git, so it is not in the bundle that v4_remote pushes. r8 copies exactly the files listed in
v4/v3_records_manifest.json (for each Step 0 label: tierA/*, calib/*, calibration.json) into research_v2/v3_runs on the pod, and GM verifies them
before its suites run:
  - every listed file exists and has the listed SHA-256 (raw bytes; scp and tar keep the bytes);
  - each calibration.json also has the hash appended to V3_PROTOCOL_LOCK.txt ("CALIBRATION <label>: ... <sha256>").
The V3 lock hashes only calibration.json; the other files are bound by the manifest, which is a committed file under the v4 lock.

  python research_v2/tools/v4_v3_records.py write              (laptop) write v4/v3_records_manifest.json from research_v2/v3_runs
  python research_v2/tools/v4_v3_records.py verify [--root D]  check D/v3_runs (default research_v2) against the manifest and the V3 lock
  python research_v2/tools/v4_v3_records.py pack <out.tar>     (laptop) tar of exactly the manifest files, paths relative to research_v2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tarfile
from pathlib import Path

RV = Path(__file__).resolve().parents[1]
MANIFEST = RV / "v4" / "v3_records_manifest.json"
V3_LOCK = RV / "V3_PROTOCOL_LOCK.txt"
LABELS = ("qwen-base", "llama-instruct", "llama-base")          # analyze_v4.STEP0 values (a04 uses qwen-base and llama-base)
SUBDIRS = ("tierA", "calib")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def listed_files(root: Path) -> list[str]:
    out = []
    for lab in LABELS:
        base = root / "v3_runs" / lab
        for sd in SUBDIRS:
            d = base / sd
            if not d.is_dir():
                raise FileNotFoundError(f"{d} is missing")
            out += [p.relative_to(root).as_posix() for p in sorted(d.iterdir()) if p.is_file()]
        if not (base / "calibration.json").is_file():
            raise FileNotFoundError(f"{base / 'calibration.json'} is missing")
        out.append(f"v3_runs/{lab}/calibration.json")
    return out


def v3_lock_calibration(lock_text: str) -> dict:
    """label -> sha256 from the 'CALIBRATION <label>:' lines of the V3 lock."""
    res = {}
    for line in lock_text.splitlines():
        m = re.match(r"CALIBRATION (\S+): .*?\b([0-9a-f]{64})\b", line)
        if m:
            res[m.group(1)] = m.group(2)
    return res


def write(root: Path = RV, manifest: Path = MANIFEST) -> dict:
    files = {f: sha(root / f) for f in listed_files(root)}
    cal = v3_lock_calibration(V3_LOCK.read_text(encoding="utf-8"))
    for lab in LABELS:
        if files[f"v3_runs/{lab}/calibration.json"] != cal.get(lab):
            raise SystemExit(f"{lab}/calibration.json does not match the V3 lock; manifest not written")
    d = {"note": "r8: locked v3 records copied to the pod for the Step 0 tests; SHA-256 of raw bytes", "labels": list(LABELS), "files": files}
    manifest.write_text(json.dumps(d, indent=1) + "\n", encoding="utf-8", newline="\n")
    return d


def verify(root: Path = RV, manifest: Path = MANIFEST, lock_text: str | None = None) -> tuple[bool, list[str]]:
    """(ok, problems). Every manifest file present with its hash, every calibration.json equal to the V3 lock's hash."""
    probs = []
    if not manifest.is_file():
        return False, [f"{manifest} is missing"]
    files = json.loads(manifest.read_text(encoding="utf-8"))["files"]
    if not files:
        return False, ["the manifest lists no files"]
    for f, h in files.items():
        p = root / f
        if not p.is_file():
            probs.append(f"missing: {f}")
        elif sha(p) != h:
            probs.append(f"hash differs: {f}")
    cal = v3_lock_calibration(lock_text if lock_text is not None else V3_LOCK.read_text(encoding="utf-8"))
    for lab in LABELS:
        f = f"v3_runs/{lab}/calibration.json"
        if f not in files:
            probs.append(f"not in the manifest: {f}")
        elif cal.get(lab) is None:
            probs.append(f"no CALIBRATION line for {lab} in the V3 lock")
        elif files[f] != cal[lab]:
            probs.append(f"manifest hash of {f} differs from the V3 lock")
        elif (root / f).is_file() and sha(root / f) != cal[lab]:
            probs.append(f"{f} differs from the V3 lock")
    return not probs, probs


def pack(out: Path, root: Path = RV, manifest: Path = MANIFEST):
    files = json.loads(manifest.read_text(encoding="utf-8"))["files"]
    with tarfile.open(out, "w") as tf:
        for f in files:
            tf.add(root / f, arcname=f, recursive=False)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["write", "verify", "pack"])
    ap.add_argument("out", nargs="?")
    ap.add_argument("--root", default=str(RV))
    a = ap.parse_args(argv)
    if a.cmd == "write":
        d = write()
        print(f"manifest written: {len(d['files'])} files")
    elif a.cmd == "pack":
        if not a.out:
            sys.exit("pack <out.tar>")
        pack(Path(a.out))
        print(f"packed {a.out}")
    else:
        ok, probs = verify(Path(a.root))
        n = len(json.loads(MANIFEST.read_text(encoding="utf-8"))["files"]) if MANIFEST.is_file() else 0
        print(f"v3 records verify: {n} files, calibration.json equal to the V3 lock" if ok else "v3 records DO NOT verify:\n  " + "\n  ".join(probs[:30]))
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
