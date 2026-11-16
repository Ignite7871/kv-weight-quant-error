"""Independent recomputation of the v6b analysis (FOLLOWUP_V6B s4-s5; tolerances in RECOMPUTATION_TOLERANCES_V6B.md).

Imports numpy and the standard library only; no project module. Written from FOLLOWUP_V6B, not from analyze_v6b.py.
Reads the raw v6b records (and, for problems where cut 1 skipped the rerun, the v6 records of C1-V and NAT-V-cen), re-checks the flags and
that no (model, arm, problem) appears twice, recomputes every reported quantity with its own methods, and compares with v6b_report.json.

  python research_v2/v6b/recompute_v6b_independent.py     -> v6b_runs/v6b_recomputation.json
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RV = HERE.parent
OUT = Path(os.environ.get("V6B_RUNS", RV / "v6b_runs"))
V6 = Path(os.environ.get("V6_RUNS", RV / "v6_runs"))
MODELS = ["qwen2.5-0.5b", "llama3.2-1b", "qwen2.5-1.5b"]
ARMS = ["NAT-V", "C1-V", "NAT-V-cen", "NAT-V-cen0", "C1-V+mu", "C1-V@cen"]
FLAGS = ["exact_ok", "rounding_ok", "finite_ok", "routing_reset", "natural_ok", "construction_ok", "b_construction_ok"]
NB = 10_000
SEED_MEAN, SEED_SHARE = 6262161, 6262162


def lines(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def table(d):
    out = {}
    for arm in ARMS:
        rows = lines(d / f"{arm}.jsonl")
        ids = [r["index"] for r in rows]
        assert len(ids) == len(set(ids)), f"duplicate records {d}/{arm}"
        out[arm] = {r["index"]: r for r in rows}
    return out


def lnm(r):
    s = 0.0
    for x in r["js"]:
        s += float(x)
    return math.log(max(s / len(r["js"]), 1e-9))


def counts_boot(n, seed):
    """multinomial resampling counts (NB x n)."""
    g = np.random.default_rng(seed)
    return g.multinomial(n, np.full(n, 1.0 / n), size=NB)


def mean_ci(x, seed):
    x = np.asarray(x, float)
    w = counts_boot(len(x), seed)
    bm = (w @ x) / len(x)
    return {"estimate": float(np.mean(x)), "ci95": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))]}


def share_ci(num, den, seed):
    num, den = np.asarray(num, float), np.asarray(den, float)
    w = counts_boot(len(num), seed)
    bs = (w @ num) / (w @ den)
    return {"share": float(num.sum() / den.sum()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}


def model(m, k):
    A = table(OUT / "test" / m)
    v6 = {a: {r["index"]: r for r in lines(V6 / "test" / m / f"{a}.jsonl")} for a in ("C1-V", "NAT-V-cen")}
    ids = sorted(A["C1-V@cen"])
    n6 = 0
    for a in ("C1-V", "NAT-V-cen"):
        for i in ids:
            if i not in A[a]:
                A[a][i] = v6[a][i]
                n6 += 1
    for a in ARMS:
        assert all(i in A[a] for i in ids), f"{m}: {a} incomplete"
    flags_bad = sum(1 for a in ARMS for i in ids if not all(A[a][i].get(f, True) for f in FLAGS))
    L = {a: np.array([lnm(A[a][i]) for i in ids]) for a in ARMS}
    th = lambda a, b: L[a] - L[b]
    paths = {"R1": (th("NAT-V-cen", "C1-V"), th("NAT-V", "C1-V")), "R2": (th("NAT-V", "C1-V+mu"), th("NAT-V", "C1-V")),
             "U1": (th("NAT-V-cen0", "C1-V@cen"), th("NAT-V", "C1-V@cen")), "U2": (th("NAT-V", "C1-V+mu"), th("NAT-V", "C1-V@cen"))}
    res = {"n": len(ids), "records_from_v6": n6, "records_with_a_failed_flag": flags_bad,
           "theta_V": mean_ci(th("NAT-V", "C1-V"), SEED_MEAN + 100 * k), "theta_N": mean_ci(th("C1-V@cen", "C1-V"), SEED_MEAN + 100 * k + 1),
           "interaction_R": mean_ci(th("NAT-V", "C1-V+mu") - th("NAT-V-cen", "C1-V"), SEED_MEAN + 100 * k + 2),
           "interaction_U": mean_ci(th("NAT-V", "C1-V+mu") - th("NAT-V-cen0", "C1-V@cen"), SEED_MEAN + 100 * k + 3),
           "renorm_effect": mean_ci(th("NAT-V-cen", "NAT-V-cen0"), SEED_MEAN + 100 * k + 4), "paths": {}}
    for j, (p, (d, t)) in enumerate(paths.items()):
        res["paths"][p] = {"dir": mean_ci(d, SEED_MEAN + 100 * k + 10 + j), "total": mean_ci(t, SEED_MEAN + 100 * k + 20 + j),
                           "share_dir": share_ci(d, t, SEED_SHARE + 100 * k + j)}
    return res


def decide(R):
    """FOLLOWUP_V6B s5, written from the protocol table."""
    cells = [(m, p) for m in R for p in ("R1", "R2", "U1", "U2")]
    lo = {c: R[c[0]]["paths"][c[1]]["share_dir"]["ci95"][0] for c in cells}
    hi = {c: R[c[0]]["paths"][c[1]]["share_dir"]["ci95"][1] for c in cells}
    if any(R[m]["paths"][p]["total"]["estimate"] <= 0 for m, p in cells):
        return "undecided"
    if any(hi[c] < 0.5 for c in cells):
        return "reject"
    if all(lo[c] > 0.5 for c in cells):
        return "keep, unrestricted"
    if all(lo[(m, p)] > 0.5 for m in R for p in ("R1", "R2")):
        return "keep at fixed norm"
    return "undecided"


def compare(rep, R, verdict):
    out = []

    def det(name, a, b):
        out.append({"q": name, "locked": a, "recomputed": b, "ok": abs(a - b) <= 1e-9 * max(1.0, abs(a))})

    def ci(name, a, b):
        w = abs(a[1] - a[0])
        out.append({"q": name, "locked": a, "recomputed": b, "ok": all(abs(x - y) <= max(0.05 * w, 1e-9) for x, y in zip(a, b))})

    for m in MODELS:
        L, X = rep["models"][m], R[m]
        out.append({"q": f"{m}:n", "locked": L["n"], "recomputed": X["n"], "ok": L["n"] == X["n"]})
        out.append({"q": f"{m}:flags", "locked": L["records_with_a_failed_flag"], "recomputed": X["records_with_a_failed_flag"],
                    "ok": L["records_with_a_failed_flag"] == X["records_with_a_failed_flag"]})
        for q in ("theta_V", "theta_N", "interaction_R", "interaction_U", "renorm_effect"):
            det(f"{m}:{q}", L[q]["estimate"], X[q]["estimate"])
            ci(f"{m}:{q}:ci95", L[q]["ci95"], X[q]["ci95"])
        for p in ("R1", "R2", "U1", "U2"):
            for q in ("dir", "total"):
                det(f"{m}:{p}:{q}", L["paths"][p][q]["estimate"], X["paths"][p][q]["estimate"])
            det(f"{m}:{p}:share", L["paths"][p]["share_dir"]["share"], X["paths"][p]["share_dir"]["share"])
            ci(f"{m}:{p}:share:ci95", L["paths"][p]["share_dir"]["ci95"], X["paths"][p]["share_dir"]["ci95"])
    out.append({"q": "verdict H*", "locked": rep["verdict"]["H_star"], "recomputed": verdict, "ok": rep["verdict"]["H_star"] == verdict})
    return out


def main():
    R = {m: model(m, k) for k, m in enumerate(MODELS)}
    v = decide(R)
    rep = json.loads((OUT / "v6b_report.json").read_text(encoding="utf-8"))
    cmp_ = compare(rep, R, v)
    bad = [c for c in cmp_ if not c["ok"]]
    res = {"what": "independent recomputation of v6b (numpy + stdlib)", "verdict": v, "compared": len(cmp_), "outside_tolerance": len(bad),
           "disagreements": bad, "models": R}
    (OUT / "v6b_recomputation.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"compared {len(cmp_)}, outside tolerance {len(bad)}, verdict {v}")
    for c in bad:
        print("  DISAGREE", c)


if __name__ == "__main__":
    main()
