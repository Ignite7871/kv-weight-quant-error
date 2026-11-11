"""Mutation check for X9 (v3/phase7/FOLLOWUP_X9.md section 5): every injected fault must make at least one test of test_x9.py fail.

  venv/Scripts/python.exe research_v2/v3/phase7/mutation_check_x9.py [--write]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

P7 = Path(__file__).resolve().parent
sys.path.insert(0, str(P7))

import test_x9 as TX                             # noqa: E402  (imports run_v3_tests, rv3x9, run_x9, analyze_x9)

import torch                                     # noqa: E402

X, RX, AX, V = TX.X, TX.RX, TX.AX, TX.V
SNAP = {"X": dict(vars(X)), "RX": dict(vars(RX)), "AX": dict(vars(AX)), "PS": dict(vars(X.PerLayerPredictorState))}


def restore():
    X.uninstall()
    for name, mod in (("X", X), ("RX", RX), ("AX", AX)):
        for k, v in SNAP[name].items():
            if not k.startswith("__"):
                setattr(mod, k, v)
    for k in ("evaluate", "record_write"):
        setattr(X.PerLayerPredictorState, k, SNAP["PS"][k])


O = SNAP["X"]


def _l0nat_row(row):
    def f(arm, alpha, kF, kM, gen, vgen, ctx):
        if arm != X.X9_ARM:
            return O["_LOCKED"]["stored_entry_v3"](arm, alpha, kF, kM, gen, vgen, ctx)
        out, info = O["_LOCKED"]["stored_entry_v3"]("NAT-K*", alpha, kF, kM, gen, vgen, ctx)
        if alpha == 0.0:
            return out, info
        out, pre, tgt = out.clone(), info["pre"].clone(), info["target"].clone()
        out[row] = kM[row]
        pre[row] = kM[row].double()
        tgt[row] = (kM[row].double() - kF[row].double()).norm(dim=-1, keepdim=True)
        if row % 2 == 1:
            info["exact_rows"] = info["exact_rows"].clone()
            info["exact_rows"][row] = False
        info["pre"], info["target"] = pre, tgt
        return out, info
    return f


def m_l0nat_wrong_layer():
    X.stored_entry_x9 = _l0nat_row(2)


def m_l0nat_on_values():
    X.stored_entry_x9 = _l0nat_row(1)


def m_projector_wrong_layer():
    X.project = lambda U, x: O["project"](torch.roll(U, 1, dims=0), x)


def m_projector_wrong_head():
    X.project = lambda U, x: O["project"](torch.roll(U, 1, dims=1), x)


def m_summed_predictor_other_order():
    ev = SNAP["PS"]["evaluate"]

    def f(self, cache, queries):
        out = ev(self, cache, queries)
        if self.n:
            out["P_logit"] = sum(self.layer_terms[-1]["PL_logit"])          # sum of (term / denom) instead of (sum of terms) / denom
        return out
    X.PerLayerPredictorState.evaluate = f


def m_pl_norm_wrong_axis():
    ev = SNAP["PS"]["evaluate"]

    def f(self, cache, queries):
        out = ev(self, cache, queries)
        if self.n:
            rms = (self.cum2 / self.n).sqrt()
            self.layer_terms[-1]["PL_norm"] = [float(x) for x in rms.t().reshape(self.n_layers, 2 * self.n_kv).mean(-1)]
        return out
    X.PerLayerPredictorState.evaluate = f


def m_comparator_tolerance():
    same = SNAP["RX"]["same"]

    def f(a, b):
        if isinstance(a, float) and isinstance(b, float) and a == a and b == b:
            return abs(a - b) <= 1e-12 * max(abs(b), 1e-300)
        return same(a, b)
    RX.same = f


def m_comparator_nan_blind():
    same = SNAP["RX"]["same"]
    RX.same = lambda a, b: True if (isinstance(a, float) and isinstance(b, float) and (a != a or b != b)) else same(a, b)


def m_auth_ignores_manifest_id():
    RX.auth_ok = lambda text, mid: RX.AUTH_TEXT in text


def m_energies_of_stored_entry():
    def f(self, stored, kF):
        X._LOCKED["PredictorState"].record_write(self, stored, kF)
        e = X.slot_energies(stored.double(), kF.double(), X.SUBSPACE["U"])
        for k in X.SLOT_KEYS:
            self.slots[k].append(e[k])
    X.PerLayerPredictorState.record_write = f


def m_share_swapped():
    g = SNAP["AX"]["share_G"]

    def f(a, b, c, rng):
        G, ci = g(a, b, c, rng)
        return (1.0 / G if G else float("inf")), [1.0 / ci[1] if ci[1] else float("inf"), 1.0 / ci[0] if ci[0] else float("inf")]
    AX.share_G = f


def m_history_always_matches():
    RX.history_matches = lambda regen, locked: True


def m_subset_seed_changed():
    RX.SUBSET_SEED = 20261208


MUTANTS = [m_l0nat_wrong_layer, m_l0nat_on_values, m_projector_wrong_layer, m_projector_wrong_head, m_summed_predictor_other_order,
           m_pl_norm_wrong_axis, m_comparator_tolerance, m_comparator_nan_blind, m_auth_ignores_manifest_id, m_energies_of_stored_entry,
           m_share_swapped, m_history_always_matches, m_subset_seed_changed]


def main():
    restore()
    base = TX.run_suite()
    if base:
        print("baseline fails; fix the suite first:", base)
        sys.exit(1)
    rep, survived = {}, []
    for m in MUTANTS:
        t0 = time.time()
        try:
            m()
            fails = TX.run_suite()
        finally:
            restore()
        rep[m.__name__] = {"caught": bool(fails), "failing_tests": sorted({f[0] for f in fails})[:6], "seconds": round(time.time() - t0, 1)}
        print(m.__name__, "CAUGHT" if fails else "SURVIVED", rep[m.__name__]["failing_tests"][:3], flush=True)
        if not fails:
            survived.append(m.__name__)
    print(f"{len(MUTANTS) - len(survived)}/{len(MUTANTS)} mutants caught")
    if "--write" in sys.argv:
        (P7 / "x9_mutation_check.json").write_text(json.dumps({"mutants": rep, "caught": len(MUTANTS) - len(survived), "total": len(MUTANTS)}, indent=1))
    sys.exit(0 if not survived else 1)


if __name__ == "__main__":
    main()
