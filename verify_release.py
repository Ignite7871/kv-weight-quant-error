"""Standalone check of this release against the protocol locks (no git history needed). Python 3 standard library only.

For research_v2/V5_PROTOCOL_LOCK.txt, V4_PROTOCOL_LOCK.txt and V3_PROTOCOL_LOCK.txt:
  - every "<sha256>  <path>" line: the file's SHA-256, raw or with CRLF normalized to LF, must equal the locked hash. A file that
    a post-data deviation changed is checked against the hash in its deviation record instead (v4: V4-D1.json; v5: the latest
    of V5-D1.json, V5-D2.json, V5-D3.json that lists it);
  - every PREVIOUS LOCK line, and every AMENDMENT line's preserved lock: the earlier lock file's hash (LF-normalized). The V5
    lock hashed V4_PROTOCOL_LOCK.txt as it stood at the v4 lock (b0b191df...); this release ships the final v4 lock with its
    appended run lines, so that line matches when the leading whole lines of the shipped file have the recorded hash;
  - every appended run line (GM, GM D<n>, CALIBRATION, STEP0, ANCHOR, DESIGN, HQQ, DEVIATION D<n>): the run file's hash. A v5
    run file that V5-D3 declares lost (gm_D2.json) is reported as recorded lost, not as missing.
Then v4_runs/v4_report.json and v5_runs/v5_report.json are checked against the hashes the manuscript generators expect.
Usage:  python verify_release.py
"""
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RV = ROOT / "research_v2"
REPORT_SHA = {"v4_runs/v4_report.json": "df57a31483160e62351378576d14123893081249e06525ed5be5f5099758dccb",
              "v5_runs/v5_report.json": "2f00f5113aec05f67990fac55e9150340780f31273a18080d2e276db0f493706"}
RUN_RX = re.compile(r"^(GM|GM D\d+|STEP0|ANCHOR|DESIGN|DEVIATION D\d+|CALIBRATION [^:]+|HQQ [^:]+): ([0-9a-f]{64})  (\S+)")
AMEND_RX = re.compile(r"^AMENDMENT \S+: .*old lock sha256 ([0-9a-f]{64}) preserved at (\S+);")


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


def prefix_match(p, want):
    """A PREVIOUS LOCK hash taken before that lock gained its appended run lines: some leading run of whole lines of the
    shipped (final) lock file, LF-normalized, must hash to it."""
    b = p.read_bytes().replace(b"\r\n", b"\n")
    ends = [i + 1 for i, c in enumerate(b) if c == 0x0A]
    return any(h(b[:e]) == want for e in ends)


def check_lock(lock, run_dirs, overrides, lost):
    ok, bad, missing, rec_lost = 0, [], [], []
    global prefixed
    prefixed = []
    for line in lock.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([0-9a-f]{64})  (\S+)$", line)
        if m:
            want, rel = m.groups()
            p = find(rel, [ROOT])
            if p is None:
                missing.append(rel)
            elif want in hashes(p) or overrides.get(rel) in hashes(p):
                ok += 1
            else:
                bad.append(rel)
            continue
        m = re.match(r"^PREVIOUS LOCK: ([0-9a-f]{64})  (\S+)$", line) or AMEND_RX.match(line)
        if m:
            want, rel = m.groups()
            p = find(rel, [ROOT])
            if p is None:
                missing.append(rel)
            elif want in hashes(p):
                ok += 1
            elif prefix_match(p, want):
                ok += 1
                prefixed.append(rel)
            else:
                bad.append(rel)
            continue
        m = RUN_RX.match(line)
        if m:
            _, want, rel = m.groups()
            p = find(rel, [ROOT] + run_dirs)
            if p is None and lost.get(rel) == want:
                rec_lost.append(rel)
            elif p is None:
                missing.append(rel)
            elif want in hashes(p):
                ok += 1
            else:
                bad.append(rel)
    return ok, bad, missing, rec_lost, list(prefixed)


prefixed = []


def main():
    d1 = json.loads((RV / "v4" / "deviations" / "V4-D1.json").read_text(encoding="utf-8"))["files"]
    v5o, v5lost = {}, {}
    for d in ("V5-D1", "V5-D2", "V5-D3"):
        rec = json.loads((RV / "v5" / "deviations" / f"{d}.json").read_text(encoding="utf-8"))
        v5o.update(rec["files"])
        v5lost.update(rec.get("lost_appendix_files") or {})
    results = {"V5 lock": check_lock(RV / "V5_PROTOCOL_LOCK.txt", [RV / "v5_runs"], v5o, v5lost),
               "V4 lock": check_lock(RV / "V4_PROTOCOL_LOCK.txt", [RV / "v4_runs"], d1, {}),
               "V3 lock": check_lock(RV / "V3_PROTOCOL_LOCK.txt", [RV / "v3_runs"], {}, {})}
    for name, (ok, bad, missing, rec_lost, pre) in results.items():
        print(f"{name}: {ok} hashes match; {len(bad)} differ {bad}; {len(missing)} not in this release {missing}"
              + (f"; recorded lost by a deviation record: {rec_lost}" if rec_lost else "")
              + (f"; previous lock matched as the leading lines of the shipped final lock (before its appended run lines): {pre}"
                 if pre else ""))
    reps = {}
    for rel, want in REPORT_SHA.items():
        p = RV / rel
        reps[rel] = p.is_file() and h(p.read_bytes()) == want
        print(f"{rel} matches the manuscript's hash: {reps[rel]}")
    sys.exit(0 if not any(r[1] for r in results.values()) and all(reps.values()) else 1)


if __name__ == "__main__":
    main()
