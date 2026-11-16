"""Write / verify the dated SHA-256 lock of v6 (PROTOCOL_V6.md section 11). Private dated hash; NOT a public pre-registration.

  python research_v2/tools/lock_v6.py            create V6_PROTOCOL_LOCK.txt from COMMITTED content (every locked file committed and unmodified)
  python research_v2/tools/lock_v6.py --verify   exit 0 iff every locked file in the working tree matches its locked hash
  python research_v2/tools/lock_v6.py --lock-id  the lock id: first 16 hex of the SHA-256 of the lock text
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

RV = Path(__file__).resolve().parents[1]
ROOT = RV.parent
LOCK = RV / "V6_PROTOCOL_LOCK.txt"
FILES = ["v6/PROTOCOL_V6.md", "v6/THEORY_PLAN_V6.md", "v6/rv6.py", "v6/run_v6.py", "v6/analyze_v6.py", "v6/analyze_r_v6.py",
         "v6/tests/test_rv6.py", "v6/tests/test_analyze_v6.py", "v6/v6_sample.json", "v4/V4_MODEL_MANIFEST.json", "v4/rv4.py", "v4/run_v4.py",
         "rv3/core.py", "analyze_v3.py", "v3/phase8/analyze_x13.py", "tools/lock_v6.py"]


def git(*a):
    return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, check=True).stdout


def sha_lf(b: bytes) -> str:
    return hashlib.sha256(b.replace(b"\r\n", b"\n")).hexdigest()


def lock_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def create():
    if LOCK.exists():
        sys.exit("V6_PROTOCOL_LOCK.txt exists; a lock is never rewritten")
    dirty = git("status", "--porcelain", "--", *[f"research_v2/{f}" for f in FILES]).decode().strip()
    if dirty:
        sys.exit(f"refused: locked files not committed or modified:\n{dirty}")
    head = git("rev-parse", "HEAD").decode().strip()
    lines = ["V6 PROTOCOL LOCK (PROTOCOL_V6.md section 11). Private dated hash; not a public pre-registration.",
             f"created_utc: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}", f"commit: {head}"]
    for f in FILES:
        lines.append(f"{sha_lf(git('show', f'HEAD:research_v2/{f}'))}  research_v2/{f}")
    text = "\n".join(lines) + "\n"
    LOCK.write_text(text, encoding="utf-8", newline="\n")
    print("lock id", lock_id(text))


def verify() -> bool:
    text = LOCK.read_text(encoding="utf-8")
    ok = True
    for line in text.splitlines()[3:]:
        h, p = line.split("  ", 1)
        cur = sha_lf((ROOT / p).read_bytes())
        if cur != h:
            print("MODIFIED", p)
            ok = False
    print("lock id", lock_id(text), "verify", ok)
    return ok


if __name__ == "__main__":
    if "--verify" in sys.argv:
        sys.exit(0 if verify() else 1)
    elif "--lock-id" in sys.argv:
        print(lock_id(LOCK.read_text(encoding="utf-8")))
    else:
        create()
