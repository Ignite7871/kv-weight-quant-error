"""Release check for the WikiText-2 hash-only rule (PROTOCOL_V5 §2, D-V5-5): no released file may hold WikiText-2 text or token
ids. CPU only, read-only.

The pinned parquet splits (test, train) are read from the local Hugging Face cache, every paragraph of the v5 sample (100 test,
50 calibration, 3 timing) is rebuilt and checked against its hash in v5/v5_wikitext_sample.json, and every 32-character window
of every sample paragraph is indexed. Then, for every file under each path given:
  - any file but .npz: every maximal run of 32 or more non-control characters (decoded as UTF-8, errors ignored) is slid over,
    and no 32-character window of it may be a window of a sample paragraph (an exact search for any copied text);
  - JSON / JSONL records under U3/ or U5/ (the WikiText units): no key named top, ids, history_ids, prompt_ids or text, and no
    list of 20 or more integers outside the two construction-diagnostic fields rdev_hist and rdev_n_gt5 (DIAG_KEYS);
  - .npz archives: every array is numeric (no string or object arrays).
Exit 1 on any finding.   python research_v2/tools/v5_wikitext_release_scan.py <dir-or-file> [...]
"""
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

RV = Path(__file__).resolve().parent.parent
SNAP = Path.home() / ".cache/huggingface/hub/datasets--Salesforce--wikitext/snapshots/b08601e04326c79dfdd32d625aee71d232d685c3"
BAD_KEYS = {"top", "ids", "history_ids", "prompt_ids", "text"}
# Integer lists that are not token ids: the locked v4 construction diagnostics written by rv4.py into every record, rdev_hist (a
# histogram of slot-norm deviations over the fixed RDEV_EDGES bins) and rdev_n_gt5 (per-step counts of slots deviating > 5 %).
DIAG_KEYS = {"rdev_hist", "rdev_n_gt5"}
RUN = re.compile(r"[^\x00-\x1f\"\\]{32,}")
K = 32


def windows():
    sample = json.loads((RV / "v5/v5_wikitext_sample.json").read_text(encoding="utf-8"))
    rows = {s: pq.read_table(SNAP / f"wikitext-2-raw-v1/{s}-00000-of-00001.parquet").column("text").to_pylist() for s in ("test", "train")}
    ents = [("test", e) for e in sample["test"]] + [("train", e) for e in sample["calibration"] + sample["timing"]]
    W = set()
    for split, e in ents:
        p = rows[split][e["index"]].strip()
        assert hashlib.sha256(p.encode("utf-8")).hexdigest() == e["text_sha256"], (split, e["index"])
        W |= {p[i:i + K] for i in range(len(p) - K + 1)}
    return len(ents), W


def int_lists(o, n=20):
    if isinstance(o, list):
        if len(o) >= n and all(isinstance(x, int) and not isinstance(x, bool) for x in o):
            return True
        return any(int_lists(x, n) for x in o)
    if isinstance(o, dict):
        return any(int_lists(v, n) for k, v in o.items() if k not in DIAG_KEYS)
    return False


def bad_keys(o):
    if isinstance(o, dict):
        return [k for k in o if k in BAD_KEYS] + [k for v in o.values() for k in bad_keys(v)]
    if isinstance(o, list):
        return [k for v in o for k in bad_keys(v)]
    return []


def main(paths):
    npara, W = windows()
    files = [f for p in map(Path, paths) for f in ([p] if p.is_file() else sorted(p.rglob("*"))) if f.is_file()]
    find, nnpz, nrec, nruns = [], 0, 0, 0
    for f in files:
        if f.suffix == ".npz":
            nnpz += 1
            with np.load(f, allow_pickle=False) as z:
                for k in z.files:
                    if z[k].dtype.kind not in "biufc":
                        find.append(f"{f}: npz array {k} has dtype {z[k].dtype}")
            continue
        t = f.read_bytes().decode("utf-8", errors="ignore")
        for m in RUN.finditer(t):
            r = m.group(0)
            nruns += 1
            hit = next((r[i:i + K] for i in range(len(r) - K + 1) if r[i:i + K] in W), None)
            if hit:
                find.append(f"{f}: contains WikiText-2 text {hit!r}")
                break
        if any(part in ("U3", "U5") for part in f.parts) and f.suffix in (".json", ".jsonl"):
            for line in (t.splitlines() if f.suffix == ".jsonl" else [t]):
                if not line.strip():
                    continue
                nrec += 1
                o = json.loads(line)
                if bad_keys(o):
                    find.append(f"{f}: WikiText unit record has key(s) {sorted(set(bad_keys(o)))}")
                if int_lists(o):
                    find.append(f"{f}: WikiText unit record has a list of >= 20 integers")
    print(f"sample paragraphs rebuilt and hash-checked: {npara}; indexed {K}-character windows: {len(W)}")
    print(f"files scanned: {len(files)} ({nnpz} npz); character runs checked: {nruns}; WikiText-unit records checked: {nrec}; "
          f"findings: {len(find)}")
    for x in find[:50]:
        print("  FINDING:", x)
    sys.exit(1 if find else 0)


if __name__ == "__main__":
    main(sys.argv[1:] or [str(RV / "v5_runs")])
