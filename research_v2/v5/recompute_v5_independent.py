"""Independent recomputation of the v5 locked analysis (PROTOCOL_V5.md section 8; tolerances in RECOMPUTATION_TOLERANCES_V5.md).
numpy / json / math / hashlib / pathlib only: no project import. Written from the protocol text, not from analyze_v5.py.

  python research_v2/v5/recompute_v5_independent.py [--runs <v5_runs>] [--v4-runs <v4_runs>]   -> <v5_runs>/v5_recomputation.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SEED_BOOT, SEED_NULL, SEED_AX = 5252051, 5252052, 5252053
B_BOOT, B_FLIP, N_NULL = 10_000, 100_000, 200
FLOOR = 1e-9
V4_REPORT_SHA256 = "df57a31483160e62351378576d14123893081249e06525ed5be5f5099758dccb"
UNITS = ["U0", "U1", "U2", "U3", "U4", "U5", "U6"]
MODEL = {"U0": "qwen2.5-0.5b", "U1": "qwen3-0.6b-base", "U2": "smollm2-1.7b", "U3": "qwen2.5-0.5b", "U4": "qwen2.5-0.5b", "U5": "mistral-7b-v0.3",
         "U6": "mistral-7b-v0.3"}
PAIRS = {"H1": ("NAT-K", "C1-K"), "H2": ("NAT-V", "C1-V"), "H3": ("NAT-K", "NAT-V"), "NAT-C1": ("NAT", "C1")}
FAMS = {"A-H1": (["U1", "U2"], "H1", -1), "A-H2": (["U1", "U2"], "H2", 1), "A-H3": (["U1", "U2"], "H3", -1), "A-H5": (["U1", "U2"], "H5", 0),
        "B-H2": (["U4", "U6"], "H2", 1), "B-H3": (["U4", "U6"], "H3", -1), "B-H5": (["U4", "U6"], "H5", 0),
        "D-H2": (["U3", "U5"], "H2", 1), "D-H5": (["U3", "U5"], "H5", 0)}
REFS = {"U1": {"size": ["qwen2.5-0.5b"], "hd": ["qwen2.5-1.5b"]},
        "U2": {"size": ["qwen2.5-1.5b", "qwen2.5-3b", "llama3.2-3b"], "hd": ["qwen2.5-0.5b", "llama3.2-1b"]}}
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
    p = (1 + int((np.abs((s * d).mean(1)) >= abs(d.mean()) - 1e-15).sum())) / (B_FLIP + 1)
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


def unit(runs: Path, u, hd, key):
    A = {a: records(runs / u / "test" / f"{a}.jsonl") for a in ("NAT", "C1", "NAT-K", "C1-K", "NAT-V", "C1-V")}
    idx = sorted(set.intersection(*[set(v) for v in A.values()]))
    cal = json.loads((runs / u / "calibration.json").read_text(encoding="utf-8"))
    out = {"n": len(idx), "contrasts": {}, "H5": {}, "theta": {}}
    for k, (h, (a, b)) in enumerate(PAIRS.items()):
        th = [ell(A[a][i]) - ell(A[b][i]) for i in idx]
        out["theta"][h] = th
        out["contrasts"][h] = stats(th, key * 10 + k, cal["margin_log_js"], cal["dose_response_informative"])
    for k, p in enumerate(("K", "V")):
        rng = np.random.default_rng([SEED_NULL, key, k])
        xs = []
        for i in idx:
            with np.load(runs / u / "test" / "slots" / f"NAT_{i}.npz") as z:
                xs.append(cm_x(z[f"norms{p}"], z[f"S1{p}"], hd, rng))
        out["H5"][p] = stats(xs, key * 10 + 5 + k)
    return out


def ax(E, rep4):
    res = {}
    for c in ("NAT-C1", "H1"):
        us = [u for u in ("U1", "U2") if u in E]
        rows, ps = {}, []
        for j, u in enumerate(us):
            size = float(np.mean([rep4["per_model"][m]["contrasts"][c]["estimate"] for m in REFS[u]["size"]]))
            hd = float(np.mean([rep4["per_model"][m]["contrasts"][c]["estimate"] for m in REFS[u]["hd"]]))
            b = 0.5 * (size + hd)
            d = np.asarray(E[u]["theta"][c]) - b
            sgn = 1.0 if d.mean() >= 0 else -1.0
            rng = np.random.default_rng([SEED_AX, j, len(c)])
            f = rng.choice([-1.0, 1.0], size=(B_FLIP, d.size))
            p = (1 + int(((f * d).mean(1) * sgn >= sgn * d.mean() - 1e-15).sum())) / (B_FLIP + 1)
            rows[u] = {"boundary": b, "size_pred": size, "hd_pred": hd, "p": p}
            ps.append(p)
        for u, hp in zip(us, holm(ps)):
            lo, hi = E[u]["contrasts"][c]["ci95"]
            b, up = rows[u]["boundary"], rows[u]["hd_pred"] > rows[u]["boundary"]
            if hp < 0.05 and ((lo > b) if up else (hi < b)):
                cl = "head-dimension pattern"
            elif hp < 0.05 and ((hi < b) if up else (lo > b)):
                cl = "size pattern"
            else:
                cl = "unresolved"
            rows[u].update(holm_p=float(hp), classification=cl)
        res[c] = rows
    cls = [res["NAT-C1"][u]["classification"] for u in ("U1", "U2") if u in res["NAT-C1"]]
    joint = ("in these checkpoints the v4 size trend tracks head dimension" if len(cls) == 2 and set(cls) == {"head-dimension pattern"} else
             "in these checkpoints the v4 size trend tracks size" if len(cls) == 2 and set(cls) == {"size pattern"} else
             "neither factor alone accounts for it in these checkpoints")
    res["joint"] = joint
    return res


def families(E, units):
    res = {}
    for name, (us, c, sign) in FAMS.items():
        us = [u for u in us if u in units]
        if c == "H5":
            keys = [(u, p) for u in us for p in ("K", "V")]
            hp = holm([E[u]["H5"][p]["p"] for u, p in keys]) if keys else []
            rd = {}
            for (u, p), q in zip(keys, hp):
                s = E[u]["H5"][p]
                rd[f"{u}/{p}"] = "supported" if (s["ci95"][0] > math.log(4) and q < 0.05) else ("falsified" if s["ci95"][1] < math.log(2) else "inconclusive")
        else:
            hp = holm([E[u]["contrasts"][c]["p"] for u in us]) if us else []
            rd = {u: read_dir(E[u]["contrasts"][c], q, sign) for u, q in zip(us, hp)}
        res[name] = {"readings": rd, "result": overall(rd)}
    return res


def close(a, b, tol):
    return abs(a - b) <= tol


def compare(rep, E, F, AXr, part="all_units"):
    rows = []

    def add(what, ok, locked, recomputed):
        rows.append({"quantity": what, "agree": bool(ok), "locked": locked, "recomputed": recomputed})
    for u, e in E.items():
        L_ = rep["per_unit"][u]
        for h, s in e["contrasts"].items():
            ls = L_["contrasts"][h]
            add(f"{u} {h} estimate", close(ls["estimate"], s["estimate"], 1e-9 * max(1, abs(s["estimate"]))), ls["estimate"], s["estimate"])
            w95, w90 = ls["ci95"][1] - ls["ci95"][0], ls["ci90"][1] - ls["ci90"][0]
            add(f"{u} {h} ci95", all(close(x, y, max(0.05 * w95, 1e-9)) for x, y in zip(ls["ci95"], s["ci95"])), ls["ci95"], s["ci95"])
            add(f"{u} {h} ci90", all(close(x, y, max(0.05 * w90, 1e-9)) for x, y in zip(ls["ci90"], s["ci90"])), ls["ci90"], s["ci90"])
            add(f"{u} {h} p", close(ls["p"], s["p"], 0.01 + 0.1 * ls["p"]), ls["p"], s["p"])
            add(f"{u} {h} verdict", ls["verdict"] == s["verdict"], ls["verdict"], s["verdict"])
        for p, s in e["H5"].items():
            ls = L_["H5"][p]
            w = ls["ci95"][1] - ls["ci95"][0]
            add(f"{u} L^{p}", close(ls["estimate"], s["estimate"], 0.02 + 0.05 * w), ls["estimate"], s["estimate"])
            add(f"{u} L^{p} ci95", all(close(x, y, 0.02 + 0.1 * w) for x, y in zip(ls["ci95"], s["ci95"])), ls["ci95"], s["ci95"])
    for k, v in F.items():
        lv = rep[part][k]
        add(f"{part} {k} result", lv["result"] == v["result"], lv["result"], v["result"])
        add(f"{part} {k} readings", lv["readings"] == v["readings"], lv["readings"], v["readings"])
    lax = rep[part]["A-X"]
    for c in ("NAT-C1", "H1"):
        for u, r in AXr[c].items():
            lr = lax[c]["units"][u]
            add(f"{part} A-X {c} {u} boundary", close(lr["boundary"], r["boundary"], 1e-12), lr["boundary"], r["boundary"])
            add(f"{part} A-X {c} {u} p", close(lr["p_one_sided"], r["p"], 0.01 + 0.1 * lr["p_one_sided"]), lr["p_one_sided"], r["p"])
            add(f"{part} A-X {c} {u} classification", lr["classification"] == r["classification"], lr["classification"], r["classification"])
    add(f"{part} A-X joint reading", lax["joint_reading_NAT-C1"]["reading"] == AXr["joint"], lax["joint_reading_NAT-C1"]["reading"], AXr["joint"])
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(HERE.parent / "v5_runs"))
    ap.add_argument("--v4-runs", default=str(HERE.parent / "v4_runs"))
    a = ap.parse_args(argv)
    runs = Path(a.runs)
    rep = json.loads((runs / "v5_report.json").read_text(encoding="utf-8"))
    b4 = (Path(a.v4_runs) / "v4_report.json").read_bytes()
    assert hashlib.sha256(b4).hexdigest() == V4_REPORT_SHA256, "not the locked v4 report"
    rep4 = json.loads(b4)
    man = json.loads((HERE / "V5_MODEL_MANIFEST.json").read_text(encoding="utf-8"))["models"]
    us = [u for u in UNITS if u in rep["per_unit"]]
    E = {u: unit(runs, u, man[MODEL[u]]["geometry"]["head_dim"], k) for k, u in enumerate(UNITS) if u in us}
    flagged = [u for u in UNITS if (runs / u / "precision.json").exists()]
    rows = compare(rep, E, families(E, us), ax(E, rep4), "all_units")
    rest = [u for u in us if u not in flagged]
    rows += compare(rep, {}, families(E, rest), ax({u: E[u] for u in rest}, rep4), "without_flagged")
    bad = [r for r in rows if not r["agree"]]
    out = {"n_compared": len(rows), "n_disagree": len(bad), "disagreements": bad, "rows": rows,
           "unresolved_verdicts": [r["quantity"] for r in bad if any(w in r["quantity"] for w in ("verdict", "result", "readings", "classification", "joint"))]}
    (runs / "v5_recomputation.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(f"compared {len(rows)} quantities; {len(bad)} outside tolerance")
    for r in bad:
        print("DISAGREE:", r["quantity"], r["locked"], r["recomputed"])
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
