"""Secret scan (PROTOCOL_V4.md section 11, download gate). Scans files for HF token patterns (hf_...), the RunPod API-key format (rpa_...) and the
literal values of the secret environment variables (HF_TOKEN, HUGGING_FACE_HUB_TOKEN, RUNPOD_API_KEY). Archives are opened recursively: tar
members, gzip streams and zip members (including .npz). A hit is reported by file, member and pattern, NEVER with the value, and the exit status
is 1, which blocks the download (pod) or the commit (laptop).

  python research_v2/tools/v4_secret_scan.py <file or directory> [...]
"""

from __future__ import annotations

import gzip
import io
import sys
import tarfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v4"))
import v4_ops as O  # noqa: E402

MAX_DEPTH = 4


def scan_bytes(b: bytes, where: str, literals, depth=0) -> list:
    hits = [(where, name, off) for name, off in O.find_secrets(b, literals)]
    if depth >= MAX_DEPTH:
        return hits
    if b[:2] == b"\x1f\x8b":                                            # gzip
        try:
            hits += scan_bytes(gzip.decompress(b), where + "!gunzip", literals, depth + 1)
        except (OSError, EOFError):
            pass
    if b[:4] == b"PK\x03\x04":                                          # zip / npz
        try:
            with zipfile.ZipFile(io.BytesIO(b)) as z:
                for n in z.namelist():
                    hits += scan_bytes(z.read(n), f"{where}!{n}", literals, depth + 1)
        except zipfile.BadZipFile:
            pass
    if len(b) > 262 and b[257:262] == b"ustar":                          # tar
        try:
            with tarfile.open(fileobj=io.BytesIO(b)) as tf:
                for m in tf.getmembers():
                    if m.isfile():
                        hits += scan_bytes(tf.extractfile(m).read(), f"{where}!{m.name}", literals, depth + 1)
        except tarfile.TarError:
            pass
    return hits


def scan_path(p: Path, literals=None) -> list:
    literals = O.secret_literals() if literals is None else literals
    files = [p] if p.is_file() else sorted(x for x in p.rglob("*") if x.is_file())
    hits = []
    for f in files:
        hits += scan_bytes(f.read_bytes(), str(f), literals)
    return hits


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if not args:
        sys.exit(__doc__)
    hits = []
    for a in args:
        hits += scan_path(Path(a))
    for where, name, off in hits:
        print(f"SECRET SCAN HIT: {where}: {name} at byte {off} (value withheld)")
    print("secret scan: clean" if not hits else f"secret scan: {len(hits)} hit(s); BLOCKED")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
