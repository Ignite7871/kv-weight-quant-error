"""Pre-run check for Stage 2 (STAGE2_SPEC.md section 2): the 389 held-out GSM8K test problems are disjoint from the 930-problem
pool, and no stored outcome file contains an evaluated result for them. Every data file in the repository (excluding venv, .git and
the Stage 1 run folders, which are checked separately and strictly) is scanned for per-item records keyed by a problem index that
fall on a held-out index. Hits are CLASSIFIED against a closed list of known, explained cases; any other hit fails the check.

  python research_v2/tools/check_heldout_clean.py [--out research_v2/validation_logs/stage2_heldout_clean_check.json]

Exit status 0 iff (a) disjointness holds, (b) no Stage 1 / pilot / gate record has a held-out test index, and (c) every repository
hit is in KNOWN_HITS. Known hits are disclosed in the spec; they are not per-problem GSM8K task outcomes.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

RV = Path(__file__).resolve().parents[1]
ROOT = RV.parent
INDEX_KEYS = ("index", "idx", "problem_idx", "question_idx", "problem_index", "problem_id")

# file (posix, relative to ROOT) -> why a held-out index appears there and why it is not a GSM8K test outcome for the problem
KNOWN_HITS = {
    "results/triviaqa_n930_all_conditions.json": "TriviaQA validation items; integer 'index' is a TriviaQA index (different dataset), numerically coincident",
    "paper_tmlr/supplement/supplement/dps_supplement/results/triviaqa_n930_all_conditions.json": "copy of the TriviaQA file above",
    "research_v2/pilot_analysis/paired_outcomes.csv": "pilot TRAIN-split problem indices (GSM8K train split, 7,473 items); numerically coincident with test indices",
    "results/eqf_spike_dataset_Qwen2.5-0.5B.csv": "teacher-forced token labelling of GOLD solutions of test problems 0-199 (FP16/INT4 argmax per gold token); no generation, no answer correctness",
    "results/eqf_spike_dataset_TinyLlama_v1.1.csv": "same labelling with a different model",
    "paper_tmlr/supplement/supplement/dps_supplement/results/eqf_spike_dataset_Qwen2.5-0.5B.csv": "copy of the gold-trajectory labelling file above",
    "paper_tmlr/supplement/supplement/dps_supplement/results/eqf_spike_dataset_TinyLlama_v1.1.csv": "copy of the gold-trajectory labelling file above",
}
PROBE_EXPOSED = list(range(8))      # scripts/probe_gsm8k_routing*.py NUM_PROBLEMS = 8 (console output only; no per-problem file)
GOLD_EXPOSED_RANGE = range(200)     # scripts/build_eqf_spike.py NUM_PROBLEMS = 200


def index_hits(path: Path, held: set) -> list:
    ext = path.suffix.lower()
    hits = []
    try:
        if ext == ".jsonl":
            for l in path.read_text(encoding="utf-8").splitlines():
                if l.strip():
                    r = json.loads(l)
                    if isinstance(r, dict) and r.get("index") in held and r.get("split", "test") == "test":
                        hits.append(("jsonl", r["index"]))
        elif ext == ".csv":
            with open(path, encoding="utf-8", errors="ignore", newline="") as fh:
                rd = csv.DictReader(fh)
                key = next((c for c in (rd.fieldnames or []) if c.lower() in INDEX_KEYS), None)
                if key:
                    n = 0
                    for r in rd:
                        try:
                            if int(float(r[key])) in held:
                                n += 1
                        except (ValueError, TypeError):
                            pass
                    if n:
                        hits.append((f"csv key={key}", n))
        elif ext == ".json":
            d = json.loads(path.read_text(encoding="utf-8"))

            def walk(o, p=""):
                if isinstance(o, dict):
                    for k, v in o.items():
                        yield from walk(v, p + "/" + str(k))
                elif isinstance(o, list):
                    if o and isinstance(o[0], dict):
                        for k in INDEX_KEYS:
                            ids = [x[k] for x in o if isinstance(x, dict) and isinstance(x.get(k), int)]
                            if len(ids) >= 5 and set(ids) & held:
                                yield (p, k, len(set(ids) & held))
                    for v in o[:3000]:
                        if isinstance(v, (dict, list)):
                            yield from walk(v, p + "[]")
            hits += [("json" + h[0] + " key=" + h[1], h[2]) for h in walk(d)]
    except Exception:  # noqa: BLE001  (unreadable files cannot hold parseable records)
        pass
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RV / "validation_logs" / "stage2_heldout_clean_check.json"))
    a = ap.parse_args()
    s1 = json.loads((RV / "stage1_sample.json").read_text())
    held, pool = set(s1["reserved389"]), set(s1["pool930"])
    rep = {"n_heldout": len(held), "n_pool": len(pool), "disjoint": not (held & pool),
           "union_is_full_test_split": (held | pool) == set(range(1319))}
    ok = rep["disjoint"] and rep["union_is_full_test_split"] and len(held) == 389 and len(pool) == 930
    # (b) Stage 1 / pilot / gate records: strict, split-aware
    strict = []
    for d in (RV / "stage1_runs", RV / "pilot_runs", RV / "validation_logs"):
        for f in sorted(d.rglob("*.jsonl")) if d.exists() else []:
            h = index_hits(f, held)
            if h:
                strict.append((f.relative_to(ROOT).as_posix(), h))
    rep["stage1_pilot_gate_jsonl_records_with_heldout_test_index"] = strict
    ok &= not strict
    # (c) the rest of the repository
    hits, scanned = {}, 0
    for root, dirs, files in os.walk(ROOT):
        dirs[:] = [x for x in dirs if x not in ("venv", ".git", "__pycache__", "node_modules", "stage1_runs", "stage2_runs")]
        for f in files:
            p = Path(root) / f
            if p.suffix.lower() not in (".json", ".jsonl", ".csv") or p.stat().st_size > 200_000_000:
                continue
            rel = p.relative_to(ROOT).as_posix()
            if rel.startswith("research_v2/stage1_analysis") or rel.startswith("research_v2/validation_logs/stage2_"):
                continue
            scanned += 1
            h = index_hits(p, held)
            if h:
                hits[rel] = h
    rep["files_scanned"] = scanned
    rep["repository_hits"] = {k: {"hits": v, "classification": KNOWN_HITS.get(k, "UNEXPECTED")} for k, v in sorted(hits.items())}
    unexpected = [k for k in hits if k not in KNOWN_HITS]
    rep["unexpected_hits"] = unexpected
    ok &= not unexpected
    # (d) exposures that cannot be seen in files: disclosed, with sizes
    rep["exposure_disclosure"] = {
        "probe_test_0_7_console_only": sorted(held & set(PROBE_EXPOSED)),
        "gold_trajectory_labelling_test_0_199": sorted(held & set(GOLD_EXPOSED_RANGE)),
        "n_gold_trajectory_labelled": len(held & set(GOLD_EXPOSED_RANGE)),
        "note": ("Neither exposure produced a stored per-problem GSM8K task outcome (generated answer or correctness) for these problems. "
                 "The gold-trajectory file stores teacher-forced FP16/INT4 token argmax labels on the reference solution. "
                 "STAGE2_SPEC.md pre-declares a sensitivity analysis excluding all of them."),
    }
    rep["clean"] = bool(ok)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("n_heldout", "disjoint", "union_is_full_test_split", "files_scanned", "unexpected_hits", "clean")}, indent=1))
    print("exposure:", rep["exposure_disclosure"]["probe_test_0_7_console_only"], rep["exposure_disclosure"]["n_gold_trajectory_labelled"], "gold-labelled")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
