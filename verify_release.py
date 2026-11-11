"""Standalone check of this release against the protocol locks (no git history needed). Python 3 standard library only.

For research_v2/V4_PROTOCOL_LOCK.txt and research_v2/V3_PROTOCOL_LOCK.txt:
  - every "<sha256>  <path>" line: the file's SHA-256, raw or with CRLF normalized to LF, must equal the locked hash. A file that
    the post-data deviation D1 changed is checked against the hash in research_v2/v4/deviations/V4-D1.json instead;
  - every PREVIOUS LOCK line: the earlier lock file's hash (LF-normalized);
  - every appended run line (GM, CALIBRATION, STEP0, DESIGN, DEVIATION, GM D1): the run file's hash.
Then research_v2/v4_runs/v4_report.json is checked against the hash the manuscript generator expects.
Usage:  python verify_release.py
"""
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RV = ROOT / "research_v2"
REPORT_SHA = "df57a31483160e62351378576d14123893081249e06525ed5be5f5099758dccb"


def h(b):
    return hashlib.sha256(b).hexdigest()


def hashes(p):
    b = p.read_bytes()
    return {h(b), h(b.replace(b"\r\n", b"\n"))}


def find(rel, bases):
    for base in bases:
        p = base / rel
        if p.is_file():
            return p
    return None


def check_lock(lock, run_dirs, overrides):
    ok, bad, missing, lines = 0, [], [], lock.read_text(encoding="utf-8").splitlines()
    for line in lines:
        m = re.match(r"^([0-9a-f]{64})  (\S+)$", line)
        if m:
            want, rel = m.groups()
            p = find(rel, [ROOT])
            if p is None:
                missing.append(rel)
                continue
            got = hashes(p)
            if want in got or overrides.get(rel) in got:
                ok += 1
            else:
                bad.append(rel)
            continue
        m = re.match(r"^PREVIOUS LOCK: ([0-9a-f]{64})  (\S+)$", line)
        if m:
            want, rel = m.groups()
            p = find(rel, [ROOT])
            (missing.append(rel) if p is None else (ok := ok + 1) if want in hashes(p) else bad.append(rel))
            continue
        m = re.match(r"^(GM|GM D1|STEP0|DESIGN|DEVIATION D1|CALIBRATION [^:]+): ([0-9a-f]{64})  (\S+)", line)
        if m:
            _, want, rel = m.groups()
            p = find(rel, [ROOT] + run_dirs)
            (missing.append(rel) if p is None else (ok := ok + 1) if want in hashes(p) else bad.append(rel))
    return ok, bad, missing


def main():
    d1 = json.loads((RV / "v4" / "deviations" / "V4-D1.json").read_text(encoding="utf-8"))["files"]
    r4 = check_lock(RV / "V4_PROTOCOL_LOCK.txt", [RV / "v4_runs"], d1)
    r3 = check_lock(RV / "V3_PROTOCOL_LOCK.txt", [RV / "v3_runs"], {})
    rep = h((RV / "v4_runs" / "v4_report.json").read_bytes()) == REPORT_SHA
    for name, (ok, bad, missing) in (("V4 lock", r4), ("V3 lock", r3)):
        print(f"{name}: {ok} hashes match; {len(bad)} differ {bad}; {len(missing)} not in this release {missing}")
    print(f"v4_report.json matches the manuscript's hash: {rep}")
    sys.exit(0 if not (r4[1] or r3[1]) and rep else 1)


if __name__ == "__main__":
    main()
