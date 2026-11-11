"""Mutation check for the Stage 1 interventions: inject known bugs into the semantic helpers and confirm the
Stage 1 CPU suite fails for each. A suite that passes on correct code proves little unless it fails on wrong code.

Run: venv/Scripts/python.exe research_v2/tests/mutation_check_stage1.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_stage1_tests as T
import rv2.engine as E
import rv2.stage1 as S

ORIG = {k: getattr(S, k) for k in ("_unkept_precision", "_restore_cache", "_token_logits", "_branch_logits")}
ALL = [k for k in dir(T) if k.startswith("s") and k[1:2].isdigit() and callable(getattr(T, k))]
_state = {"second": False}


def mut_token_source_swapped(ts, kv, lg_kept, lg_unkept):       # emits the OTHER branch's token
    return lg_unkept if ts == kv else lg_kept


def mut_same_precision_twice(kv):                                 # the "other" pass uses the kept precision
    return kv


def mut_restore_skipped(cache, L):                                # second pass sees the first pass's entry too
    pass


def mut_restore_too_far(cache, L):                                # drops one genuine past entry
    cache.crop(L - 1)


def mut_restore_marks_second():                                   # for the kept-pass-wrong-precision mutant
    orig = ORIG["_restore_cache"]

    def f(cache, L):
        orig(cache, L)
        _state["second"] = True
    return f


def mut_branch_labels_swapped(kv, lg_kept, lg_unkept):           # logs tF/tN under the wrong names
    return (lg_unkept, lg_kept) if kv == "F" else (lg_kept, lg_unkept)


def kept_pass_wrong_precision():
    orig_run = E.Backend.run

    def run(self, ids, cache, use_4bit):
        if cache is not None and _state["second"]:
            _state["second"] = False
            use_4bit = not use_4bit
        return orig_run(self, ids, cache, use_4bit)
    return orig_run, run


def setp(name, fn):
    return lambda: setattr(S, name, fn)


def undo(name):
    return lambda: setattr(S, name, ORIG[name])


def run_named(names):
    T.RESULTS.clear()
    for n in names:
        getattr(T, n)()
    return {n: s for n, s, _ in T.RESULTS}


def main():
    orig_run, bad_run = kept_pass_wrong_precision()
    plan = [
        ("token source swapped", setp("_token_logits", mut_token_source_swapped), undo("_token_logits")),
        ("unkept pass uses kept precision (no real second branch)", setp("_unkept_precision", mut_same_precision_twice), undo("_unkept_precision")),
        ("cache not restored between the two passes", setp("_restore_cache", mut_restore_skipped), undo("_restore_cache")),
        ("cache restored one entry too far", setp("_restore_cache", mut_restore_too_far), undo("_restore_cache")),
        ("tF/tN branch labels swapped in the log", setp("_branch_logits", mut_branch_labels_swapped), undo("_branch_logits")),
        ("kept pass runs in the wrong precision",
         lambda: (setattr(S, "_restore_cache", mut_restore_marks_second()), setattr(E.Backend, "run", bad_run), _state.update(second=False)),
         lambda: (setattr(S, "_restore_cache", ORIG["_restore_cache"]), setattr(E.Backend, "run", orig_run), _state.update(second=False))),
    ]
    killed = 0
    for name, apply, revert in plan:
        apply()
        try:
            res = run_named(ALL)
        finally:
            revert()
        failed = [n for n, s in res.items() if s == "FAIL"]
        k = bool(failed)
        killed += k
        print(f"{'KILLED  ' if k else 'SURVIVED'}  {name:58s}  failed by: {[f.split('_')[0] for f in failed]}")
    clean = run_named(ALL)
    clean_ok = all(s == "PASS" for s in clean.values())
    print(f"\nclean code passes all {len(clean)} tests: {clean_ok}")
    print(f"{killed}/{len(plan)} mutants killed")
    sys.exit(0 if killed == len(plan) and clean_ok else 1)


if __name__ == "__main__":
    main()
