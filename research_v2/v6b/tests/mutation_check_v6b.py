"""Mutation check for v6b (FOLLOWUP_V6B s6, G0): every injected fault must make at least one of the CPU tests fail.

Each mutant edits the source text of ONE function (or one constant) of rv6b / run_v6b / analyze_v6b / recompute_v6b_independent in-process,
re-executes it in that module's namespace, runs the test suite covering that module (test_rv6b: rv6b, run_v6b, launcher;
test_analyze_v6b: analysis, recomputation), and restores the original object before the
next mutant. A mutant that no test detects is a SURVIVOR and fails the check.

  python research_v2/v6b/tests/mutation_check_v6b.py [--write <json>]
"""
from __future__ import annotations

import argparse
import contextlib
import inspect
import io
import json
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import test_rv6b as TR  # noqa: E402
import test_analyze_v6b as TA  # noqa: E402
import rv6b  # noqa: E402
import run_v6b as RB  # noqa: E402
import analyze_v6b as AN  # noqa: E402
import recompute_v6b_independent as RC  # noqa: E402
import launch_test_v6b as LT  # noqa: E402

# (name, module, function or None for a constant, old text, new text)
MUTANTS = [
    # arm constructions
    ("A re-normed to ||Dn||", rv6b, "new_entry", "        d = c\n", "        d = nD * c / nc.clamp_min(1e-300)\n"),
    ("A target ||Dn||", rv6b, "new_entry", "        d = c\n        target[rows] = nc", "        d = c\n        target[rows] = nD"),
    ("A not centered", rv6b, "new_entry", "        d = c\n", "        d = Dn\n"),
    ("C at natural norm", rv6b, "new_entry", "        iso = nc * g", "        iso = nD * g"),
    ("B drops mu", rv6b, "new_entry", "            d = mu + iso", "            d = iso + 0 * mu"),
    ("B mu doubled", rv6b, "new_entry", "            d = mu + iso", "            d = 2 * mu + iso"),
    ("B and C draw from vgen", rv6b, "stored_entry", "new_entry(arm, kF, kM, gen, ctx)", "new_entry(arm, kF, kM, vgen, ctx)"),
    ("draw not unit", rv6b, "new_entry", "g = V._unit_safe(V._gauss(Dn.shape, gen, dev).double())", "g = V._gauss(Dn.shape, gen, dev).double()"),
    ("key rows not exact", rv6b, "new_entry", "    exact[other] = True\n", "    exact[other] = False\n"),
    ("mu from kM - kF", rv6b, "new_entry", "    c = Dn - mu\n", "    c = Dn - 0.5 * mu\n"),
    ("B dev check off", rv6b, "new_entry", "State.b_dev.append(float(dev_[pos].max()) if bool(pos.any()) else 0.0)", "State.b_dev.append(0.5)"),
    ("B ratio uses ||c||", rv6b, "new_entry", "(e.norm(dim=-1)[posD] / nD.squeeze(-1)[posD])", "(e.norm(dim=-1)[posD] / nc.squeeze(-1)[posD])"),
    ("dispatcher changes v6 arms", rv6b, "stored_entry", "out, info = _ORIG(arm, kF, kM, gen, vgen, ctx)", "out, info = _ORIG(arm, kF, kM, vgen, gen, ctx)"),
    ("mu sums every arm", rv6b, "State.reset", 'cls.sums = rv4.SlotSums() if arm == "NAT-V" else None', "cls.sums = rv4.SlotSums()"),
    ("mu records kM", rv6b, "stored_entry", "State.sums.record(out, kF)", "State.sums.record(out, out)"),
    ("degenerate not counted", rv6b, "new_entry", "degen = int(((nc < DEGEN_TOL * nD) & (nD > 0)).sum())", "degen = 0"),
    ("B seed own stream", rv6b, None, "SEED_ARM", {"NAT-V-cen0": "NAT-V-cen0", "C1-V+mu": "C1-V+mu", "C1-V@cen": "C1-V"}),
    ("arm order", rv6b, None, "ARMS", ["NAT-V", "NAT-V-cen", "C1-V", "NAT-V-cen0", "C1-V+mu", "C1-V@cen"]),
    # driver
    ("G1 ignores tolerance", RB, "g1", "p = same and d <= G1_TOL", "p = same"),
    ("G1 ignores S", RB, "g1", "p = same and d <= G1_TOL", "p = d <= G1_TOL"),
    ("G1 missing record passes", RB, "g1", '"reason": "no v6 record"}\n            ok = False', '"reason": "no v6 record"}\n            ok = True'),
    ("integrity skips b flag", RB, None, "FLAGS", RB.rv6.FLAGS),
    ("cut 1 without bitwise", RB, "rerun_for", "and first_bitwise(d, ids)", ""),
    ("cut 1 from the start", RB, "rerun_for", "k >= N_BITWISE", "k >= 0"),
    ("bitwise check one arm", RB, "first_bitwise", "for a in rv6b.REPRO_ARMS)", 'for a in ["NAT-V"])'),
    ("cap 16", RB, None, "CAP_H", 16.0),
    ("G1 tol 0.5", RB, None, "G1_TOL", 0.5),
    ("tier test", RB, None, "TIER", {"test": "V6B", "train": "V6reh"}),
    ("no refusal", RB, "rehearse", 'sys.exit("refused: v6b_runs/AUTHORIZED_V6B_REHEARSAL.txt missing")', "pass"),
    # launcher
    ("scan allows duplicates", LT, "scan", 'stop(f"{m}/{a}: duplicate problem records")', "pass"),
    ("scan allows partial", LT, "scan", 'stop(f"{m}: partial problem(s) {sorted(union - full)}; author decision needed")', "pass"),
    ("scan ignores g1 log", LT, "scan", "if len(g) != len(set(g)) or set(g) != full:", "if False:"),
    ("scan allows stray reruns", LT, "scan", "extra = (per[\"C1-V\"] | per[\"NAT-V-cen\"]) - full", "extra = set()"),
    ("cut ignores cap", LT, None, "CAP_H", 16.0),
    ("cut 1 no saving", LT, "project_and_cut", "f = (4 / 6) if CUT1 in d[\"cuts\"] else 1.0", "f = 1.0"),
    ("no guard factor", LT, None, "GUARD", 1.0),
    # analysis
    ("threshold 0.4", AN, None, "THRESH", 0.4),
    ("reject on lower bound", AN, "verdict", "rej = sorted(k for k, v in hi.items() if v < THRESH)", "rej = sorted(k for k, v in lo.items() if v < THRESH)"),
    ("fixed norm checks U", AN, "verdict", 'for p in ("R1", "R2"))', 'for p in ("U1", "U2"))'),
    ("keep checks R1 only", AN, "verdict", "elif all(x > THRESH for x in lo.values()):", "elif all(lo[(m, 'R1')] > THRESH for m in R):"),
    ("R2 path via cen", AN, None, "PATHS", {**AN.PATHS, "R2": (("NAT-V", "NAT-V-cen"), ("NAT-V-cen", "C1-V"), ("NAT-V", "C1-V"))}),
    ("U total vs C1-V", AN, None, "PATHS", {**AN.PATHS, "U1": (("NAT-V-cen0", "C1-V"), ("NAT-V", "NAT-V-cen0"), ("NAT-V", "C1-V"))}),
    ("no v6 fill", AN, "recs", "                A[a][i] = v6[a][i]\n                n6 += 1", "                pass"),
    ("duplicates allowed", AN, "read", 'raise SystemExit(f"duplicate records in {f}")', "pass"),
    ("interaction U sign", AN, "per_model", 't("NAT-V", "C1-V+mu") - t("NAT-V-cen0", "C1-V@cen"), m, "IU"', 't("NAT-V-cen0", "C1-V@cen") - t("NAT-V-cen0", "C1-V@cen"), m, "IU"'),
    ("renorm effect order", AN, "per_model", 't("NAT-V-cen", "NAT-V-cen0")', 't("NAT-V-cen0", "NAT-V-cen")'),
    ("status not required", AN, "main", 'raise SystemExit("refused: not every model has a status file (complete or stopped)")', "pass"),
    ("flags not counted", AN, "per_model", "if a in A and i in A[a] and not all(", "if False and not all("),
    # recomputation
    ("recompute decide on lo", RC, "decide", "if any(hi[c] < 0.5 for c in cells):", "if any(lo[c] < 0.5 for c in cells):"),
    ("recompute no fill", RC, "model", "                A[a][i] = v6[a][i]\n                n6 += 1", "                pass"),
    ("recompute CI tolerance off", RC, "compare", "for x, y in zip(a, b))", "for x, y in zip(a, b)) or True"),
]


def run_tests(mods=None):
    fails = []
    for mod in mods or (TR, TA):
        for t in mod.TESTS:
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    t()
            except BaseException as e:  # noqa: BLE001
                fails.append(f"{mod.__name__}.{t.__name__}")
    return fails


def apply(mod, fn, old, new):
    """Returns a restore function."""
    if fn is None:
        prev = getattr(mod, old)
        setattr(mod, old, new)
        return lambda: setattr(mod, old, prev)
    owner, name = (getattr(mod, fn.split(".")[0]), fn.split(".")[1]) if "." in fn else (mod, fn)
    desc = owner.__dict__[name]                                   # the raw descriptor (classmethod object for State.reset)
    raw = desc.__func__ if isinstance(desc, classmethod) else desc
    src = textwrap.dedent(inspect.getsource(raw))
    if src.count(old) != 1:
        raise RuntimeError(f"mutant text not found once in {fn}: {old!r}")
    src = src.replace("@classmethod\n", "")
    ns = {}
    exec(compile(src.replace(old, new), f"<mutant {fn}>", "exec"), mod.__dict__, ns)   # live module globals (tests re-point OUT etc.)
    new_obj = ns[name]
    setattr(owner, name, classmethod(new_obj) if isinstance(desc, classmethod) else new_obj)
    return lambda: setattr(owner, name, desc)


SUITE = {"rv6b": (TR,), "run_v6b": (TR,), "launch_test_v6b": (TR,), "analyze_v6b": (TA,), "recompute_v6b_independent": (TA,)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write")
    a = ap.parse_args()
    base = run_tests()
    if base:
        sys.exit(f"baseline failures: {base}")
    out, surv = [], []
    for name, mod, fn, old, new in MUTANTS:
        restore = apply(mod, fn, old, new)
        try:
            f = run_tests(SUITE[mod.__name__])            # the suite that covers the mutated module (baseline and restore: both)
        finally:
            restore()
            rv6b.uninstall()
        killed = bool(f)
        out.append({"mutant": name, "module": mod.__name__, "killed": killed, "by": f[:4]})
        print(("KILLED  " if killed else "SURVIVED") + f" {name}  <- {f[:2]}", flush=True)
        if not killed:
            surv.append(name)
    after = run_tests()
    res = {"mutants": len(MUTANTS), "killed": len(MUTANTS) - len(surv), "survivors": surv, "baseline_after_restore_failures": after, "detail": out}
    print(f"mutation check: {res['killed']}/{res['mutants']} killed; survivors {surv}; clean after restore: {not after}")
    if a.write:
        Path(a.write).write_text(json.dumps(res, indent=1), encoding="utf-8")
    sys.exit(0 if not surv and not after else 1)


if __name__ == "__main__":
    main()
