"""Verify the Stage 1 checkpoint (amendment A2): previously completed records unchanged, order = seeded run order, no repeats or skips.

  python tools/verify_checkpoint.py [--snapshot stage1_runs/checkpoint_155_integrity.json]

Exit status 0 iff every check passes. Reads only; prints structural facts, never accuracies.
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from stage1_runtime import ARMS, verify_checkpoint  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=str(HERE / "stage1_runs" / "checkpoint_155_integrity.json"))
    ap.add_argument("--eval-dir", default=str(HERE / "stage1_runs" / "eval"))
    a = ap.parse_args()
    s = json.loads((HERE / "stage1_sample.json").read_text())
    rep = verify_checkpoint(Path(a.eval_dir), Path(a.snapshot), s["run_order"], ARMS)
    n = rep["n_complete_all_arms"]
    rep["next_problem_order_position"] = n
    rep["next_problem_index"] = s["run_order"][n] if n < len(s["run_order"]) else None
    print(json.dumps(rep, indent=1))
    sys.exit(0 if rep["ok"] else 1)


if __name__ == "__main__":
    main()
