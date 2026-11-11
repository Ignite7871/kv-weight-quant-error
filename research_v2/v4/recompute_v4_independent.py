"""Independent recomputation of the v4 locked analysis (PROTOCOL_V4.md section 8; tolerances in RECOMPUTATION_TOLERANCES_V4.md).
numpy / json / math / itertools / pathlib only: no project import. Written from the protocol text, not from analyze_v4.py.

  python research_v2/v4/recompute_v4_independent.py [--runs <v4_runs>]   -> <v4_runs>/v4_recomputation.json (agreement table)
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SEED_BOOT, SEED_NULL = 4242041, 4242042
B_BOOT, B_FLIP, N_NULL = 10_000, 100_000, 200
FLOOR = 1e-9
QWEN = ["qwen2.5-0.5b", "qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b"]
LLAMA = ["llama3.2-1b", "llama3.2-3b", "llama3.1-8b"]
LADDER = QWEN + LLAMA + ["mistral-7b-v0.3"]
PAIRS = {"H1": ("NAT-K", "C1-K"), "H2": ("NAT-V", "C1-V"), "H3": ("NAT-K", "NAT-V"), "NAT-C1": ("NAT", "C1")}
SIGN = {"H1": -1, "H2": 1, "H3": -1}
FLAGS = ("exact_ok", "natural_ok", "routing_reset", "construction_ok", "rounding_ok", "finite_ok")


def records(p: Path):
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            assert r["index"] not in out, f"duplicate record {r['index']} in {p}"
            assert all(r.get(k, False) for k in FLAGS) and r["js"][0] == 0.0, f"integrity flag false in {p} problem {r['index']}"
            out[r["index"]] = r
    return out


def ell(r):
    return math.log(max(float(np.mean(r["js"])), FLOOR))


def stats(d, key, margin=None, informative=True):
    d = np.asarray(d, float)
    n = len(d)
    rng = np.random.default_rng([SEED_BOOT, key])
    w = rng.multinomial(n, np.full(n, 1.0 / n), size=B_BOOT)
    bm = (w @ d) / n
    s = rng.choice([-1.0, 1.0], size=(B_FLIP, n))
    obs = abs(d.mean())
    p = (1 + int((np.abs((s * d).mean(1)) >= obs - 1e-15).sum())) / (B_FLIP + 1)
    out = {"estimate": float(d.mean()), "ci95": np.percentile(bm, [2.5, 97.5]).tolist(), "ci90": np.percentile(bm, [5, 95]).tolist(), "p": p, "n": n}
    lo, hi = out["ci95"]
    if lo > 0 or hi < 0:
        out["verdict"] = "difference"
    elif margin is not None and informative and -margin < out["ci90"][0] and out["ci90"][1] < margin:
        out["verdict"] = "equivalent"
    else:
        out["verdict"] = "inconclusive"
    return out


def holm(ps):
    ps = np.asarray(ps, float)
    adj = np.empty_like(ps)
    run = 0.0
    for rank, i in enumerate(np.argsort(ps)):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def read_dir(s, hp, sign):
    lo, hi = s["ci95"]
    if hp < 0.05 and (hi < 0 if sign < 0 else lo > 0):
        return "supported"
    return "falsified" if ((lo > 0 if sign < 0 else hi < 0) or s["verdict"] == "equivalent") else "inconclusive"


def overall(rd):
    v = list(rd.values())
    return "supported in every model" if v and all(x == "supported" for x in v) else ("falsified" if "falsified" in v else "not established in all models")


def cm_x(norms, S1, hd, rng):
    n = norms.shape[0]
    share = lambda s1: float(((s1 / n) ** 2).sum() / ((norms ** 2).sum(0) / n).sum())  # noqa: E731
    nulls = []
    for _ in range(N_NULL):
        g = rng.standard_normal(norms.shape + (hd,))
        nulls.append(math.log(n * share((g / np.linalg.norm(g, axis=-1, keepdims=True) * norms[..., None]).sum(0))))
    return math.log(n * share(S1)) - float(np.mean(nulls))


def model(runs: Path, tag, hd, key):
    A = {a: records(runs / tag / "test" / f"{a}.jsonl") for a in ("NAT", "C1", "NAT-K", "C1-K", "NAT-V", "C1-V")}
    idx = sorted(set.intersection(*[set(v) for v in A.values()]))
    cal = json.loads((runs / tag / "calibration.json").read_text(encoding="utf-8"))
    out = {"n": len(idx), "contrasts": {}, "H5": {}}
    for k, (h, (a, b)) in enumerate(PAIRS.items()):
        out["contrasts"][h] = stats([ell(A[a][i]) - ell(A[b][i]) for i in idx], key * 10 + k, cal["margin_log_js"], cal["dose_response_informative"])
    for k, p in enumerate(("K", "V")):
        rng = np.random.default_rng([SEED_NULL, key, k])
        xs = []
        for i in idx:
            with np.load(runs / tag / "test" / "slots" / f"NAT_{i}.npz") as z:
                xs.append(cm_x(z[f"norms{p}"], z[f"S1{p}"], hd, rng))
        out["H5"][p] = stats(xs, key * 10 + 5 + k)
    return out


def families(ms, E, excluded=()):
    res = {}
    for h in ("H1", "H2", "H3"):
        hp = holm([E[m]["contrasts"][h]["p"] for m in ms]) if ms else []
        rd = {m: read_dir(E[m]["contrasts"][h], q, SIGN[h]) for m, q in zip(ms, hp)}
        res[{"H1": "F1 H1", "H2": "F2 H2", "H3": "F3 H3"}[h]] = {"readings": rd, "result": overall(rd)}
    keys = [(m, p) for m in ms for p in ("K", "V")]
    hp = holm([E[m]["H5"][p]["p"] for m, p in keys]) if keys else []
    rd = {}
    for (m, p), q in zip(keys, hp):
        s = E[m]["H5"][p]
        rd[f"{m}/{p}"] = "supported" if (s["ci95"][0] > math.log(4) and q < 0.05) else ("falsified" if s["ci95"][1] < math.log(2) else "inconclusive")
    res["F5 H5"] = {"readings": rd, "result": overall(rd)}
    q = [m for m in QWEN if m in ms]
    l_ = [m for m in LLAMA if m in ms]
    if set(q + l_) != {m for m in QWEN + LLAMA if m not in excluded}:                # H4 needs all seven (minus the flagged set in the without-flagged version)
        res["H4"] = {"result": "not estimable"}
        return res
    th = {m: E[m]["contrasts"]["NAT-C1"]["estimate"] for m in q + l_}
    allm = q + l_
    tobs = np.mean([th[m] for m in l_]) - np.mean([th[m] for m in q])
    ts = [np.mean([th[allm[j]] for j in c]) - np.mean([th[allm[j]] for j in range(len(allm)) if j not in c])
          for c in itertools.combinations(range(len(allm)), len(l_))]
    p = sum(t >= tobs - 1e-12 for t in ts) / len(ts)
    s = {m: E[m]["contrasts"]["NAT-C1"] for m in allm}
    signs = all(s[m]["ci95"][1] < 0 for m in q) and not any(s[m]["ci95"][1] < 0 for m in l_)
    fal = any(s[m]["ci95"][0] > 0 or s[m]["verdict"] == "equivalent" for m in q) or any(s[m]["ci95"][1] < 0 for m in l_)
    res["H4"] = {"T": float(tobs), "p": float(p), "result": "supported" if (p <= 0.05 and signs) else ("falsified" if fal else "inconclusive")}
    return res


def close(a, b, tol):
    return abs(a - b) <= tol


def compare(rep, E, R, flagged):
    rows = []

    def add(what, ok, locked, recomputed):
        rows.append({"quantity": what, "agree": bool(ok), "locked": locked, "recomputed": recomputed})
    for m, e in E.items():
        L_ = rep["per_model"][m]
        for h, s in e["contrasts"].items():
            ls = L_["contrasts"][h]
            add(f"{m} {h} estimate", close(ls["estimate"], s["estimate"], 1e-9 * max(1, abs(s["estimate"]))), ls["estimate"], s["estimate"])
            w95, w90 = ls["ci95"][1] - ls["ci95"][0], ls["ci90"][1] - ls["ci90"][0]
            add(f"{m} {h} ci95", all(close(x, y, max(0.05 * w95, 1e-9)) for x, y in zip(ls["ci95"], s["ci95"])), ls["ci95"], s["ci95"])
            add(f"{m} {h} ci90", all(close(x, y, max(0.05 * w90, 1e-9)) for x, y in zip(ls["ci90"], s["ci90"])), ls["ci90"], s["ci90"])
            add(f"{m} {h} p", close(ls["p"], s["p"], 0.01 + 0.1 * ls["p"]), ls["p"], s["p"])
            add(f"{m} {h} verdict", ls["verdict"] == s["verdict"], ls["verdict"], s["verdict"])
        for p, s in e["H5"].items():
            ls = L_["H5"][p]
            w = ls["ci95"][1] - ls["ci95"][0]
            add(f"{m} L^{p}", close(ls["estimate"], s["estimate"], 0.02 + 0.05 * w), ls["estimate"], s["estimate"])
            add(f"{m} L^{p} ci95", all(close(x, y, 0.02 + 0.1 * w) for x, y in zip(ls["ci95"], s["ci95"])), ls["ci95"], s["ci95"])
    for part, fam in (("all_models", R["all"]), ("without_flagged", R["without"])):
        for k, v in fam.items():
            lv = rep[part][k]
            add(f"{part} {k} result", lv["result"] == v["result"], lv["result"], v["result"])
            if "readings" in v:
                add(f"{part} {k} readings", lv["readings"] == v["readings"], lv["readings"], v["readings"])
            if k == "H4" and "T" in v:
                add(f"{part} H4 T", close(lv["permutation"]["T"], v["T"], 1e-9), lv["permutation"]["T"], v["T"])
                add(f"{part} H4 p", close(lv["permutation"]["p_one_sided"], v["p"], 1e-12), lv["permutation"]["p_one_sided"], v["p"])
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(HERE.parent / "v4_runs"))
    a = ap.parse_args(argv)
    runs = Path(a.runs)
    rep = json.loads((runs / "v4_report.json").read_text(encoding="utf-8"))
    man = json.loads((HERE / "V4_MODEL_MANIFEST.json").read_text(encoding="utf-8"))["models"]
    ms = [m for m in LADDER if m in rep["per_model"]]
    flagged = [m for m in LADDER if (runs / m / "precision.json").exists()]
    E = {m: model(runs, m, man[m]["geometry"]["head_dim"], k) for k, m in enumerate(LADDER) if m in ms}
    R = {"all": families(ms, E), "without": families([m for m in ms if m not in flagged], E, flagged)}
    rows = compare(rep, E, R, flagged)
    bad = [r for r in rows if not r["agree"]]
    out = {"n_compared": len(rows), "n_disagree": len(bad), "disagreements": bad, "rows": rows,
           "unresolved_verdicts": [r["quantity"] for r in bad if any(w in r["quantity"] for w in ("verdict", "result", "readings"))]}
    (runs / "v4_recomputation.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(f"compared {len(rows)} quantities; {len(bad)} outside tolerance")
    for r in bad:
        print("DISAGREE:", r["quantity"], r["locked"], r["recomputed"])
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
