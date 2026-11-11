"""Mutation check: inject known routing/cache bugs and confirm the CPU suite catches each one.

A test suite that passes on correct code proves little unless it fails on wrong code. Each mutant
monkeypatches one piece of the engine; the listed tests must FAIL for the mutant to count as killed.

Run: venv/Scripts/python.exe research_v2/tests/mutation_check.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

import run_cpu_tests as T
import rv2.engine as E
import rv2.policies as P
from routing import calculate_entropy

ORIG = {"restore": E._restore, "snapshot": E._snapshot_last, "lag_decide": P.LagEntropy.decide,
        "generate": E.generate}


def mut_restore_skipped(cache, L, saved):          # oracle keeps FP16 KV even when NF4 is chosen
    pass


def mut_snapshot_wrong_position(cache, L):          # snapshots the previous token's KV instead
    return [(lay.keys[..., L - 1:L, :].clone(), lay.values[..., L - 1:L, :].clone()) for lay in cache.layers]


_lag_state = {}


def mut_lag_off_by_one(self, view):                 # uses the logits from TWO steps back
    prev = _lag_state.get("older", view.prev_logits)
    _lag_state["older"] = view.prev_logits
    return bool((calculate_entropy(prev) > self.tau).item())


def mut_routing_flag_inverted():
    orig_run = E.Backend.run

    def run(self, ids, cache, use_4bit):
        return orig_run(self, ids, cache, (not use_4bit) if cache is not None else use_4bit)
    return orig_run, run


MUTANTS = [
    ("oracle restore skipped", lambda: setattr(E, "_restore", mut_restore_skipped),
     lambda: setattr(E, "_restore", ORIG["restore"]),
     ["t2b_oracle_extremes_equal_static_bitwise_including_cache", "t2c_mixed_routes_equal_single_pass_replay_bitwise"]),
    ("oracle snapshot of wrong position", lambda: setattr(E, "_snapshot_last", mut_snapshot_wrong_position),
     lambda: setattr(E, "_snapshot_last", ORIG["snapshot"]),
     ["t2b_oracle_extremes_equal_static_bitwise_including_cache"]),
    ("lag-entropy off by one step", lambda: (setattr(P.LagEntropy, "decide", mut_lag_off_by_one), _lag_state.clear()),
     lambda: setattr(P.LagEntropy, "decide", ORIG["lag_decide"]),
     ["t1_lag_entropy_matches_historical_DynamicQwen_generate"]),
]


def run_named(names):
    T.RESULTS.clear()
    for n in names:
        getattr(T, n)()
    return {n: s for n, s, _ in T.RESULTS}


def main():
    killed = 0
    total = 0
    # inverted routing flag mutant (backend-level)
    orig_run, bad_run = mut_routing_flag_inverted()
    from transformers import DynamicCache
    orig_crop = DynamicCache.crop
    plan = MUTANTS + [("routing flag inverted in backend", lambda: setattr(E.Backend, "run", bad_run),
                       lambda: setattr(E.Backend, "run", orig_run),
                       ["t1_lag_entropy_matches_historical_DynamicQwen_generate",
                        "t1e_static_fp16_equals_unwrapped_model_absolute_reference"]),
                      ("cascade rollback (cache crop) disabled", lambda: setattr(DynamicCache, "crop", lambda self, n: None),
                       lambda: setattr(DynamicCache, "crop", orig_crop),
                       ["t2_cascade_extremes_equal_static_bitwise_including_cache",
                        "t3d_cascade_decisions_invariant_to_fp16_weights_until_first_escalation"])]
    for name, apply, undo, tests in plan:
        total += 1
        apply()
        try:
            res = run_named(tests)
        finally:
            undo()
        k = any(s == "FAIL" for s in res.values())
        killed += k
        print(f"{'KILLED ' if k else 'SURVIVED'}  {name:40s}  {res}")
    # sanity: unmutated code passes the same tests
    clean = run_named(sorted({t for *_, ts in plan for t in ts}))
    print("clean run:", clean)
    ok = killed == total and all(s == "PASS" for s in clean.values())
    print(f"\n{killed}/{total} mutants killed; clean code passes: {all(s == 'PASS' for s in clean.values())}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
