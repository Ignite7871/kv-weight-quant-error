"""Mutation check for the amendment-A2 runtime layer (stage1_runtime.py): inject known bugs and confirm the A2 tests (s12-s17) fail.

Run: venv/Scripts/python.exe research_v2/tests/mutation_check_a2.py
"""

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_stage1 as RS
import run_stage1_tests as T
import stage1_runtime as R

A2_TESTS = [k for k in dir(T) if k[:3] in ("s12", "s13", "s14", "s15", "s16", "s17") and callable(getattr(T, k))]


def swap(name, fn):
    orig = getattr(R, name)

    def apply():
        setattr(R, name, fn)
        if hasattr(RS, name):
            setattr(RS, name, fn)

    def revert():
        setattr(R, name, orig)
        if hasattr(RS, name):
            setattr(RS, name, orig)
    return apply, revert


def wrap_integrity(filter_fn):
    orig = R.problem_integrity_errors
    return lambda Rec, max_mem_gb=R.MAX_MEM_GB: [e for e in orig(Rec, max_mem_gb) if filter_fn(e)]


def wrap_verify(post):
    orig = R.verify_checkpoint

    def f(*a, **k):
        rep = orig(*a, **k)
        return post(rep)
    return f


def run_named(names):
    T.RESULTS.clear()
    for n in names:
        getattr(T, n)()
    return {n: s for n, s, _ in T.RESULTS}


def main():
    orig_wd = R.Watchdog

    class NeverFires(orig_wd):
        def run(self):
            return

    plan = [
        ("stop rule without the safety margin", *swap("stop_decision", lambda u, w, c, m=R.HARD_MARGIN_S: "cap" if u + w > c else None)),
        ("stop rule ignores worst-case problem duration", *swap("stop_decision", lambda u, w, c, m=R.HARD_MARGIN_S: "cap" if u + m > c else None)),
        ("stop rule never stops", *swap("stop_decision", lambda u, w, c, m=R.HARD_MARGIN_S: None)),
        ("worst-case factor 1.0 instead of 1.25", *swap("worst_problem_seconds", lambda t, o, factor=1.0: factor * max(t) * o if t else 0.0)),
        ("watchdog never fires", *swap("Watchdog", NeverFires)),
        ("pooled pace uses only the current session", *swap("pace_stats", (lambda orig: lambda ta, ts, o, u, r, c: orig(ts or ta, ts, o, u, r, c))(R.pace_stats))),
        ("temperature stop disabled", *swap("hardware_stop", lambda tele, limit_c=R.TEMP_STOP_C: None)),
        ("integrity: token-source check dropped", *swap("problem_integrity_errors", wrap_integrity(lambda e: "declared" not in e))),
        ("integrity: KV-letter check dropped", *swap("problem_integrity_errors", wrap_integrity(lambda e: "KV letters" not in e))),
        ("integrity: step-1 label check dropped", *swap("problem_integrity_errors", wrap_integrity(lambda e: "step-1" not in e))),
        ("integrity: flip-oracle prefix check dropped", *swap("problem_integrity_errors", wrap_integrity(lambda e: "first flip" not in e))),
        ("integrity: identical-history outcome check dropped", *swap("problem_integrity_errors", wrap_integrity(lambda e: "different outcome" not in e))),
        ("integrity: JS range check dropped", *swap("problem_integrity_errors", wrap_integrity(lambda e: "JS divergence" not in e))),
        ("checkpoint: hash comparison dropped", *swap("verify_checkpoint", wrap_verify(lambda r: {**r, "ok": not [p for p in r["problems"] if "snapshot" not in p]}))),
        ("checkpoint: order/repeat/skip check dropped", *swap("verify_checkpoint", wrap_verify(lambda r: {**r, "ok": not [p for p in r["problems"] if "run-order" not in p and "duplicate" not in p]}))),
    ]
    killed = 0
    for name, apply, revert in plan:
        apply()
        try:
            res = run_named(A2_TESTS)
        finally:
            revert()
        failed = [n for n, s in res.items() if s == "FAIL"]
        k = bool(failed)
        killed += k
        print(f"{'KILLED  ' if k else 'SURVIVED'}  {name:56s}  failed by: {[f.split('_')[0] for f in failed]}")
    clean = run_named(A2_TESTS)
    clean_ok = all(s == "PASS" for s in clean.values())
    print(f"\nclean code passes all {len(clean)} A2 tests: {clean_ok}")
    print(f"{killed}/{len(plan)} mutants killed")
    sys.exit(0 if killed == len(plan) and clean_ok else 1)


if __name__ == "__main__":
    main()
