"""v6b analysis (FOLLOWUP_V6B.md sections 4 and 5): the split of the value-path excess along four paths, the norm term, the interactions,
the re-norming effect, descriptives, and the pre-stated verdict on H*.

  python research_v2/v6b/analyze_v6b.py      -> v6b_runs/v6b_report.json (refuses unless every model is complete or stopped)
  env V6B_RUNS / V6_RUNS relocate the run directories (the synthetic test uses them)

CPU only, read-only. analyze_v3.summarize and analyze_v3.logm are imported unchanged. ratio_ci has the code of analyze_v6.ratio_ci with the
v6b seed (20261116). If cut 1 was applied, C1-V and NAT-V-cen of the problems after the first 10 come from the v6 records of the same
problem (which the first 10 reproduced bitwise).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RV = HERE.parent
sys.path.insert(0, str(RV))
import analyze_v3 as L  # noqa: E402

OUT = Path(os.environ.get("V6B_RUNS", RV / "v6b_runs"))
V6 = Path(os.environ.get("V6_RUNS", RV / "v6_runs"))
MODELS = ["qwen2.5-0.5b", "llama3.2-1b", "qwen2.5-1.5b"]
ARMS = ["NAT-V", "C1-V", "NAT-V-cen", "NAT-V-cen0", "C1-V+mu", "C1-V@cen"]
FROM_V6 = ["C1-V", "NAT-V-cen"]
FLAGS = ["exact_ok", "rounding_ok", "finite_ok", "routing_reset", "natural_ok", "construction_ok", "b_construction_ok"]
SEED, B = 20261116, 10_000
THRESH = 0.5
# path -> (direction-part contrast, common-mode-part contrast, total contrast); each contrast is (a, b) for theta(a, b)
PATHS = {"R1": (("NAT-V-cen", "C1-V"), ("NAT-V", "NAT-V-cen"), ("NAT-V", "C1-V")),
         "R2": (("NAT-V", "C1-V+mu"), ("C1-V+mu", "C1-V"), ("NAT-V", "C1-V")),
         "U1": (("NAT-V-cen0", "C1-V@cen"), ("NAT-V", "NAT-V-cen0"), ("NAT-V", "C1-V@cen")),
         "U2": (("NAT-V", "C1-V+mu"), ("C1-V+mu", "C1-V@cen"), ("NAT-V", "C1-V@cen"))}


def key(k):
    return int.from_bytes(hashlib.sha256(str(k).encode()).digest()[:4], "big")


def rng(*k):
    return np.random.default_rng([SEED] + [key(x) for x in k])


def summ(d, *k):
    return L.summarize(np.asarray(d, float), rng(*k), None, True, B)


def ratio_ci(num, den, *k):
    """Share mean(num)/mean(den) with a problem-bootstrap percentile CI (the code of analyze_v6.ratio_ci; v6b seed)."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    g = rng(*k)
    j = g.integers(0, len(num), size=(B, len(num)))
    bs = num[j].mean(1) / den[j].mean(1)
    return {"share": float(num.mean() / den.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}


def read(d):
    out = {}
    if d.exists():
        for f in d.glob("*.jsonl"):
            rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
            ids = [r["index"] for r in rows]
            if len(ids) != len(set(ids)):
                raise SystemExit(f"duplicate records in {f}")
            out[f.stem] = {r["index"]: r for r in rows}
    return out


def recs(m):
    """v6b records of model m, with C1-V / NAT-V-cen filled from v6 for problems where cut 1 skipped the rerun. Returns (A, ids, n_from_v6)."""
    A = read(OUT / "test" / m)
    ids = sorted(A.get("C1-V@cen", {}), key=lambda i: SAMPLE_POS[i])
    v6 = read(V6 / "test" / m)
    n6 = 0
    for a in FROM_V6:
        A.setdefault(a, {})
        for i in ids:
            if i not in A[a]:
                A[a][i] = v6[a][i]
                n6 += 1
    for a in ARMS:
        missing = [i for i in ids if i not in A.get(a, {})]
        if missing:
            raise SystemExit(f"{m}: arm {a} missing problems {missing[:5]}")
    return A, ids, n6


def theta(A, a, b, ids):
    return np.array([L.logm(A[a][i]) - L.logm(A[b][i]) for i in ids])


def per_model(m):
    A, ids, n6 = recs(m)
    t = lambda a, b: theta(A, a, b, ids)
    res = {"n": len(ids), "records_from_v6": n6}
    bad = sum(1 for a in ARMS for i in ids if a in A and i in A[a] and not all(A[a][i].get(k, True) for k in FLAGS))
    res["records_with_a_failed_flag"] = bad
    tV, tU, tN = t("NAT-V", "C1-V"), t("NAT-V", "C1-V@cen"), t("C1-V@cen", "C1-V")
    assert np.allclose(tV, tU + tN, atol=1e-12)
    res["theta_V"], res["theta_U"], res["theta_N"] = summ(tV, m, "V"), summ(tU, m, "U"), summ(tN, m, "N")
    res["paths"] = {}
    for p, (dr, cm, tot) in PATHS.items():
        xd, xc, xt = t(*dr), t(*cm), t(*tot)
        assert np.allclose(xt, xd + xc, atol=1e-12), p
        res["paths"][p] = {"dir": summ(xd, m, p, "dir"), "cm": summ(xc, m, p, "cm"), "total": summ(xt, m, p, "total"),
                           "share_dir": ratio_ci(xd, xt, m, p, "share")}
    res["interaction_R"] = summ(t("NAT-V", "C1-V+mu") - t("NAT-V-cen", "C1-V"), m, "IR")
    res["interaction_U"] = summ(t("NAT-V", "C1-V+mu") - t("NAT-V-cen0", "C1-V@cen"), m, "IU")
    res["renorm_effect"] = summ(t("NAT-V-cen", "NAT-V-cen0"), m, "renorm")
    br = np.concatenate([np.asarray(A["C1-V+mu"][i]["b_norm_ratio"], float) for i in ids])
    res["descriptive"] = {"B_norm_ratio_median": float(np.median(br)), "B_norm_ratio_p05_p95": [float(np.percentile(br, 5)), float(np.percentile(br, 95))],
                          "degenerate_steps": {a: int(sum(sum(A[a][i]["degenerate"]) for i in ids)) for a in ("NAT-V-cen0", "C1-V+mu", "C1-V@cen")},
                          "stored_over_natural_rel_V": {a: float(np.median([np.mean(A[a][i]["stored_rel_V"]) / np.mean(A[a][i]["rel_V"]) for i in ids]))
                                                        for a in ARMS}}
    g1 = OUT / "test" / m / "g1.jsonl"
    if g1.exists():
        G = [json.loads(l)["G1"] for l in g1.read_text(encoding="utf-8").splitlines() if l.strip()]
        res["G1"] = {"problems": len(G), "all_pass": all(x["pass"] for g in G for x in g.values()),
                     "all_bitwise": all(x.get("bitwise_js", False) for g in G for x in g.values()),
                     "max_abs_dlnm": max(x["abs_dlnm"] for g in G for x in g.values())}
    return res


def verdict(R):
    """FOLLOWUP_V6B s5. lo/hi = 95% CI of s_dir per (model, path)."""
    lo = {(m, p): R[m]["paths"][p]["share_dir"]["ci95"][0] for m in R for p in PATHS}
    hi = {(m, p): R[m]["paths"][p]["share_dir"]["ci95"][1] for m in R for p in PATHS}
    neg_total = [(m, p) for m in R for p in PATHS if R[m]["paths"][p]["total"]["estimate"] <= 0]
    rej = sorted(k for k, v in hi.items() if v < THRESH)
    if neg_total:
        v = "undecided"
    elif rej:
        v = "reject"
    elif all(x > THRESH for x in lo.values()):
        v = "keep, unrestricted"
    elif all(lo[(m, p)] > THRESH for m in R for p in ("R1", "R2")):
        v = "keep at fixed norm"
    else:
        v = "undecided"
    order = sorted(f"{m}:{w}" for m in R for w in ("interaction_R", "interaction_U")
                   if R[m][w]["ci95"][0] > 0 or R[m][w]["ci95"][1] < 0)
    return {"H_star": v, "rejecting_paths": [f"{m}:{p}" for m, p in rej], "nonpositive_totals": [f"{m}:{p}" for m, p in neg_total],
            "order_dependent": order, "threshold": THRESH}


def load_sample():
    global SAMPLE_POS
    s = json.loads((RV / "v6" / "v6_sample.json").read_text(encoding="utf-8"))
    SAMPLE_POS = {i: k for k, i in enumerate(s["test_ids"])}


SAMPLE_POS = {}


def main():
    load_sample()
    st = {m: (OUT / f"status_{m}.json") for m in MODELS}
    if not all(p.exists() for p in st.values()):
        raise SystemExit("refused: not every model has a status file (complete or stopped)")
    R = {m: per_model(m) for m in MODELS}
    rep = {"what": "v6b locked analysis (FOLLOWUP_V6B s4-s5)", "models": R, "verdict": verdict(R),
           "status": {m: json.loads(p.read_text(encoding="utf-8")) for m, p in st.items()}}
    (OUT / "v6b_report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep["verdict"], indent=1))


if __name__ == "__main__":
    main()
