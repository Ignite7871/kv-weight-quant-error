"""Mutation check for the Stage 2 analysis and driver helpers: inject known bugs, confirm the Stage 2 tests fail for each.

Run: venv/Scripts/python.exe research_v2/tests/mutation_check_stage2.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import analyze_stage2 as B
import run_stage2 as RS2
import run_stage2_tests as G
import test_analyze_stage2 as H

TESTS = [(G, k) for k in dir(G) if k[:1] == "g" and k[1:2].isdigit()] + [(H, k) for k in dir(H) if k[:1] == "b" and k[1:2].isdigit()]


def run_all():
    G.RESULTS.clear()
    H.RESULTS.clear()
    for mod, k in TESTS:
        getattr(mod, k)()
    return {n: s for n, s, _ in G.RESULTS + H.RESULTS}


def setter(mod, name, fn):
    orig = getattr(mod, name)
    return (lambda: setattr(mod, name, fn)), (lambda: setattr(mod, name, orig))


def main():
    orig_rc, orig_het, orig_pc, orig_merge = B.replication_category, B.heterogeneity, B.pattern_counts, B.merge_samples
    orig_assert = RS2.assert_heldout_indices

    def rc_ignores_holm(e1, e2, p_raw, p_holm, ci):
        return orig_rc(e1, e2, p_raw, p_raw, ci)

    def rc_sign_flip_always_uncertainty(e1, e2, p_raw, p_holm, ci):
        r = orig_rc(e1, e2, p_raw, p_holm, ci)
        return "opposite or zero point estimate, but compatible with Stage 1 (difference CI includes 0): statistical uncertainty" if "disagreement" in r else r

    def het_wrong_sign(d1, d2, rng, B_=B.B_HET):
        r = orig_het(d1, d2, rng, B_)
        return {**r, "diff_pp": -r["diff_pp"]}

    def merge_no_check(R1, R2):
        R = {a: {**R1[a], **R2[a]} for a in B.ARMS}
        return R, sorted(set.intersection(*[set(R[a]) for a in B.ARMS]))

    def pc_wrong_order(R, common):
        r = orig_pc(R, common)
        r["patterns"] = {k[::-1]: v for k, v in r["patterns"].items()}
        return r

    def assert_accepts_all(indices, s):
        return None

    def eq_bound_5(ci90, bound_pp=5.0):
        lo, hi = ci90
        return "equivalence replicated (90% CI within +-3 pp)" if lo > -bound_pp and hi < bound_pp else "equivalence not demonstrated (inconclusive)"

    plan = [
        ("Holm adjustment ignored in the replication category", *setter(B, "replication_category", rc_ignores_holm)),
        ("sign flips never reported as directional disagreement", *setter(B, "replication_category", rc_sign_flip_always_uncertainty)),
        ("heterogeneity difference has the wrong sign", *setter(B, "heterogeneity", het_wrong_sign)),
        ("stage merge without the overlap check", *setter(B, "merge_samples", merge_no_check)),
        ("pattern strings in the wrong arm order", *setter(B, "pattern_counts", pc_wrong_order)),
        ("driver accepts any held-out index (pool included)", *setter(RS2, "assert_heldout_indices", assert_accepts_all)),
        ("equivalence bound widened to +-5 pp", *setter(B, "equivalence_category", eq_bound_5)),
    ]
    killed = 0
    for name, apply, revert in plan:
        apply()
        try:
            res = run_all()
        finally:
            revert()
        failed = [n.split("_")[0] for n, s in res.items() if s == "FAIL"]
        killed += bool(failed)
        print(f"{'KILLED  ' if failed else 'SURVIVED'}  {name:58s}  failed by: {failed}")
    clean = run_all()
    ok = all(s == "PASS" for s in clean.values())
    print(f"\nclean code passes all {len(clean)} tests: {ok}")
    print(f"{killed}/{len(plan)} mutants killed")
    sys.exit(0 if killed == len(plan) and ok else 1)


if __name__ == "__main__":
    main()
