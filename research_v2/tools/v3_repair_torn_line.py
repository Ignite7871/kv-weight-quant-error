"""v3 amendment 1: the only permitted repair of a record file (PROTOCOL_V3.md section 15). Removes an INCOMPLETE TRAILING LINE (bytes after the
last newline, or a final line that is not valid JSON) from one v3 record file, and nothing else. Every repair is logged and recorded as a deviation;
the affected problem is then rerun by the driver (its already-written arms are kept, the missing arm is recomputed). Records are never edited by hand.

  venv/Scripts/python.exe research_v2/tools/v3_repair_torn_line.py <path to .jsonl under research_v2/v3_runs> --reason "<what happened>"

Refuses (exit 2, file untouched) if: the file is outside v3_runs (unless --root is given, for tests); the file is intact; any line other than the last is
invalid; or the removed part would contain a newline-terminated valid record.
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

RV = Path(__file__).resolve().parents[1]


def sha(b):
    return hashlib.sha256(b).hexdigest()


def plan_repair(b: bytes):
    """Returns (kept_bytes, removed_bytes) or raises ValueError with the reason the file must not be repaired."""
    if not b:
        raise ValueError("empty file: nothing to repair")
    if b.endswith(b"\n"):
        body, tail = b[:-1], b""
        lines = body.split(b"\n")
        last = lines[-1]
        try:
            json.loads(last.decode("utf-8"))
            last_ok = True
        except (ValueError, UnicodeDecodeError):
            last_ok = False
        if last_ok:
            prefix_lines = lines[:-1]
            for ln in prefix_lines:
                if ln.strip():
                    json.loads(ln.decode("utf-8"))
            raise ValueError("file is intact: nothing to repair")
        kept = b"\n".join(lines[:-1]) + (b"\n" if len(lines) > 1 else b"")
        removed = last + b"\n"
    else:
        cut = b.rfind(b"\n")
        kept, removed = (b[:cut + 1], b[cut + 1:]) if cut >= 0 else (b"", b)
    for ln in kept.split(b"\n"):
        if ln.strip():
            try:
                json.loads(ln.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                raise ValueError("an earlier line is invalid: not a torn trailing line; stop and report") from None
    return kept, removed


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--reason", required=True)
    ap.add_argument("--root", default=str(RV / "v3_runs"), help="records root (tests only)")
    ap.add_argument("--deviation-log", default=str(RV / "v3" / "DEVIATION_LOG_V3.md"))
    a = ap.parse_args(argv)
    p, root = Path(a.path).resolve(), Path(a.root).resolve()
    if root not in p.parents or p.suffix != ".jsonl":
        print(f"refused: {p} is not a .jsonl record file under {root}")
        return 2
    b = p.read_bytes()
    try:
        kept, removed = plan_repair(b)
    except ValueError as e:
        print(f"refused: {e}")
        return 2
    utc = datetime.now(timezone.utc).isoformat()
    bak = root / "repairs" / f"{p.parent.parent.name}_{p.parent.name}_{p.stem}_{utc.replace(':', '').replace('.', '')}.removed"
    bak.parent.mkdir(parents=True, exist_ok=True)
    bak.write_bytes(removed)
    tmp = p.with_suffix(".jsonl.tmp")
    tmp.write_bytes(kept)
    tmp.replace(p)
    entry = {"utc": utc, "file": p.relative_to(root).as_posix(), "sha256_before": sha(b), "sha256_after": sha(kept), "bytes_removed": len(removed),
             "removed_bytes_saved_to": bak.relative_to(root).as_posix(), "reason": a.reason}
    with open(root / "repair_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    with open(a.deviation_log, "a", encoding="utf-8") as f:
        f.write(f"| V3-R{utc[:19]} | {utc[:10]} | 4 | Records are append-only | Removed an incomplete trailing line ({len(removed)} bytes) from "
                f"`{entry['file']}` with tools/v3_repair_torn_line.py; sha256 {entry['sha256_before'][:12]}… → {entry['sha256_after'][:12]}…; removed bytes kept in "
                f"`{entry['removed_bytes_saved_to']}`; the problem is rerun by the driver | {a.reason} | None expected: the incomplete record was never valid and is recomputed |\n")
    print(json.dumps(entry, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
