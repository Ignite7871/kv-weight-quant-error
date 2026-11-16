"""v6 test-phase analysis (PROTOCOL_V6 section 7): counted tests L1, L2, L5, L6, L7, L8a/b, L9a/b; estimation targets L3, L4; descriptives.

  python research_v2/v6/analyze_v6.py            -> v6_runs/v6_report.json (refuses unless every model is complete or stopped)
  env V6_RUNS relocates v6_runs (the synthetic test uses it)

CPU only, read-only. Locked primitives (analyze_v3.summarize, logm; analyze_x13.cm_share) imported unchanged.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RV = HERE.parent
sys.path.insert(0, str(RV))
sys.path.insert(0, str(RV / "v3" / "phase8"))
sys.path.insert(0, str(HERE))
import analyze_v3 as L       # noqa: E402
import analyze_x13 as X13    # noqa: E402
import analyze_r_v6 as RV6   # noqa: E402  (inv_freq, dirichlet, spearman, holm, UNITS: unchanged)

OUT = Path(os.environ.get("V6_RUNS", RV / "v6_runs"))
MODELS = ["qwen2.5-0.5b", "llama3.2-1b", "qwen2.5-1.5b"]
SEED, B = 20261115, 10_000
V_ONLY = ["NAT-V", "C1-V", "NAT-V-cen", "ST-V", "ST-V@nat", "ST-V@nat-cen"]
K_ONLY = ["NAT-K", "C1-K", "NAT-K-cen", "ST-K", "ST-K@nat", "ST-Kc", "ST-Kc@nat"]


def key(k):
    return int.from_bytes(hashlib.sha256(str(k).encode()).digest()[:4], "big")


def rng(*k):
    return np.random.default_rng([SEED] + [key(x) for x in k])


def summ(d, *k, margin=None):
    return L.summarize(np.asarray(d, float), rng(*k), margin, True, B)


def recs(m):
    d = OUT / "test" / m
    out = {}
    for f in d.glob("*.jsonl"):
        out[f.stem] = {json.loads(l)["index"]: json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()}
    return out


def arr(m, arm, i):
    with np.load(OUT / "test" / m / "arrays" / f"{arm}_{i}.npz") as z:
        return {k: z[k] for k in z.files}


def theta_i(A, a, b, idx):
    return np.array([L.logm(A[a][i]) - L.logm(A[b][i]) for i in idx])


def spearman_test(x, y, *k):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rho = RV6.spearman(x, y)
    g = rng(*k, "boot")
    bs = [RV6.spearman(x[j], y[j]) for j in g.integers(0, len(x), size=(B, len(x)))]
    gp = rng(*k, "perm")
    p = (1 + sum(abs(RV6.spearman(x, gp.permutation(y))) >= abs(rho) - 1e-15 for _ in range(B))) / (B + 1)
    return {"rho": rho, "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "p": p}


def ratio_ci(num, den, *k):
    """Share mean(num)/mean(den) with a problem-bootstrap percentile CI."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    g = rng(*k)
    j = g.integers(0, len(num), size=(B, len(num)))
    bs = num[j].mean(1) / den[j].mean(1)
    return {"share": float(num.mean() / den.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}


def mean_ln(a, keyname):
    return float(np.nanmean(np.log(a[keyname])))


def per_model(m, margin):
    A = recs(m)
    arms = [a for a in A if not a.endswith("nolog")]
    idx = sorted(set.intersection(*[set(A[a]) for a in arms]))
    res = {"n": len(idx), "arms": sorted(arms)}
    # L3: outcome decomposition of H2 (exact per problem)
    tV = theta_i(A, "NAT-V", "C1-V", idx)
    tcm = theta_i(A, "NAT-V", "NAT-V-cen", idx)
    tdir = theta_i(A, "NAT-V-cen", "C1-V", idx)
    assert np.allclose(tV, tcm + tdir, atol=1e-12)
    res["L3"] = {"theta_V": summ(tV, m, "L3V"), "theta_cm": summ(tcm, m, "L3cm"), "theta_dir": summ(tdir, m, "L3dir"),
                 "share_cm": ratio_ci(tcm, tV, m, "L3share")}
    if "NAT-K-cen" in A:
        res["L3_K_descriptive"] = {"theta_cm_K": summ(theta_i(A, "NAT-K", "NAT-K-cen", idx), m, "L3Kcm"),
                                   "theta_dir_K": summ(theta_i(A, "NAT-K-cen", "C1-K", idx), m, "L3Kdir")}
    # per-problem logged sums
    S = {a: {i: arr(m, a, i) for i in idx} for a in arms}
    tot = lambda a, i, k: float(np.nansum(S[a][i][k]))
    # L4: head-output decomposition
    a4 = np.array([math.log(tot("NAT-V", i, "meas") / tot("NAT-V-cen", i, "meas")) for i in idx])
    b4 = np.array([math.log(tot("NAT-V-cen", i, "meas") / tot("C1-V", i, "meas")) for i in idx])
    res["L4"] = {"cm": summ(a4, m, "L4cm"), "dir": summ(b4, m, "L4dir"), "share_cm": ratio_ci(a4, a4 + b4, m, "L4share")}
    # L1
    def l1(arms_, num):
        v = []
        for a in arms_:
            if a not in S:
                continue
            for i in idx:
                me, pr = S[a][i]["meas"], S[a][i][num]
                ok = np.isfinite(me) & np.isfinite(pr) & (me > 0) & (pr > 0)
                v.append(np.abs(np.log(pr[ok] / me[ok])))
        v = np.concatenate(v)
        return {"median_abs_ln": float(np.median(v)), "n_slot_steps": int(v.size)}
    res["L1"] = {"V_mod_vs_meas": l1(V_ONLY, "mod_V"), "K_lin_vs_meas": l1(K_ONLY, "lin_K")}
    # L2, L6
    x2 = [math.log(tot("NAT-V", i, "meas") / tot("C1-V", i, "meas")) for i in idx]
    res["L2"] = spearman_test(x2, tV, m, "L2")
    tK = theta_i(A, "NAT-K", "C1-K", idx)
    x6 = [math.log(tot("NAT-K", i, "lin_K") / tot("C1-K", i, "lin_K")) for i in idx]
    model_ln = math.log(sum(tot("NAT-K", i, "lin_K") for i in idx) / sum(tot("C1-K", i, "lin_K") for i in idx))
    res["L6"] = {"spearman": spearman_test(x6, tK, m, "L6"), "model_ln_ratio_linK": model_ln, "theta_H1": summ(tK, m, "L6H1")}
    # L5 (r3: predicted c_post = A_p c_pre + (1 - c_pre)/n)
    hd, theta, sc = RV6.UNITS[m][1], RV6.UNITS[m][2], RV6.UNITS[m][3]
    w = RV6.inv_freq(hd, theta, sc)
    h = hd // 2
    lr = []
    cpre_layer_K, cV_layer = [], []
    for i in idx:
        a = S["NAT-K"][i]
        n = a["normsK"].shape[0]
        Ap = RV6.dirichlet(n, w)
        post = (a["S1K"][..., :h] ** 2 + a["S1K"][..., h:] ** 2) / n ** 2
        pre = (a["S1K_pre"][..., :h] ** 2 + a["S1K_pre"][..., h:] ** 2) / n ** 2
        c_post = post / (a["S2K_pair"] / n)
        c_pre = pre / (a["S2K_pre_pair"] / n)
        pred = Ap * c_pre + (1 - c_pre) / n
        ok = (Ap >= 0.05) & (a["S2K_pair"] > 0) & (pred > 0) & (c_post > 0)
        lr.append(np.abs(np.log(c_post[ok] / pred[ok])))
        cpre_layer_K.append((pre.sum(-1) / (a["S2K_pre_pair"].sum(-1) / n)).mean(-1))
        v = S["NAT-V"][i]
        nv = v["normsV"].shape[0]
        cV_layer.append(((v["S1V"] / nv) ** 2).sum(-1).mean(-1) / ((v["normsV"] ** 2).sum(0) / nv).mean(-1))
    lr = np.concatenate(lr)
    res["L5"] = {"median_abs_ln_ratio": float(np.median(lr)), "n": int(lr.size),
                 "descriptive_pre_c_K_vs_c_V_per_layer": {"c_pre_K": np.mean(cpre_layer_K, 0).tolist(), "c_V": np.mean(cV_layer, 0).tolist()}}
    # L7
    nC = [n_ * X13.cm_share(S["ST-V"][i]["normsV"], S["ST-V"][i]["S1V"]) for i in idx for n_ in [S["ST-V"][i]["normsV"].shape[0]]]
    rhoK = float(np.median([np.median(A["ST-K"][i]["stored_rel_K"]) for i in idx]))
    rhoV = float(np.median([np.median(A["ST-V"][i]["stored_rel_V"]) for i in idx]))
    res["L7"] = {"i_nC_V_ST": float(np.mean(nC)),
                 "ii_STK@nat_vs_C1K": summ(theta_i(A, "ST-K@nat", "C1-K", idx), m, "L7iiK", margin=margin),
                 "ii_STV@nat_vs_C1V": summ(theta_i(A, "ST-V@nat", "C1-V", idx), m, "L7iiV", margin=margin),
                 "iii_rho_ratio_K_over_V": rhoK / rhoV, "iii_STK_vs_STV": summ(theta_i(A, "ST-K", "ST-V", idx), m, "L7iii"),
                 "H3_replication": summ(theta_i(A, "NAT-K", "NAT-V", idx), m, "H3"), "H2_this_run": summ(tV, m, "H2"), "margin": margin}
    # L8, L9
    d8a = [mean_ln(S["NAT-V"][i], "A_cov") - mean_ln(S["C1-V"][i], "A_cov") for i in idx]
    d8b = [mean_ln(S["NAT-V"][i], "A_o") - mean_ln(S["C1-V"][i], "A_o") for i in idx]
    d9b = [mean_ln(S["NAT-V"][i], "A_cov") - mean_ln(S["ST-V@nat"][i], "A_cov") for i in idx]
    res["L8a"], res["L8b"] = summ(d8a, m, "L8a"), summ(d8b, m, "L8b")
    res["L9a"] = summ(theta_i(A, "NAT-V-cen", "ST-V@nat-cen", idx), m, "L9a")
    res["L9b"] = summ(d9b, m, "L9b")
    # descriptives
    desc = {"H1": summ(tK, m, "dH1"), "H2": summ(tV, m, "dH2"), "H3": res["L7"]["H3_replication"]}
    if "ST-Kc" in A:
        desc["ST-Kc_vs_ST-V"] = summ(theta_i(A, "ST-Kc", "ST-V", idx), m, "dKc")
        desc["ST-Kc@nat_vs_C1-K"] = summ(theta_i(A, "ST-Kc@nat", "C1-K", idx), m, "dKcnat")
        desc["rho_Kc"] = float(np.median([np.median(A["ST-Kc"][i]["stored_rel_K"]) for i in idx]))
    lay = {}
    for a in arms:
        if a in V_ONLY:
            lay[a] = {k: np.nanmedian(np.concatenate([S[a][i][k] for i in idx], axis=0), axis=(0, 2)).tolist() for k in ("A_o", "A_cov")}
    desc["per_layer_alignment_median"] = lay
    desc["per_layer_m_ND_NAT-V"] = {k: np.nanmean(np.concatenate([S["NAT-V"][i][k][1:] for i in idx]), axis=(0, 2)).tolist() for k in ("m", "N_D")}
    desc["a_in_per_layer"] = np.mean([(S["NAT-V"][i]["S1x"] ** 2).sum(-1) / S["NAT-V"][i]["normsV"].shape[0] / S["NAT-V"][i]["S2x"] for i in idx], 0).tolist()
    res["descriptive"] = desc
    return res


def verdicts(R):
    ps = {}
    for m, r in R.items():
        ps[f"L2:{m}"], ps[f"L6:{m}"] = r["L2"]["p"], r["L6"]["spearman"]["p"]
        ps[f"L7iii:{m}"] = r["L7"]["iii_STK_vs_STV"]["p"]
        for t in ("L8a", "L8b", "L9a", "L9b"):
            ps[f"{t}:{m}"] = r[t]["p"]
    hp = RV6.holm(ps)
    pos = lambda s, name: s["ci95"][0] > 0 and hp[name] < 0.05
    neg = lambda s: s["ci95"][1] < 0
    v = {}
    v["L1"] = ("agreement" if all(r["L1"]["V_mod_vs_meas"]["median_abs_ln"] <= math.log(1.25) and r["L1"]["K_lin_vs_meas"]["median_abs_ln"] <= math.log(1.5) for r in R.values())
               else "disagreement" if any(max(r["L1"]["V_mod_vs_meas"]["median_abs_ln"], r["L1"]["K_lin_vs_meas"]["median_abs_ln"]) > math.log(2) for r in R.values()) else "neither")
    v["L2"] = ("agreement" if all(pos({"ci95": r["L2"]["ci95"]}, f"L2:{m}") for m, r in R.items())
               else "disagreement" if any(r["L2"]["ci95"][1] <= 0 for r in R.values()) else "neither")
    v["L5"] = ("agreement" if all(r["L5"]["median_abs_ln_ratio"] <= math.log(1.5) for r in R.values())
               else "disagreement" if any(r["L5"]["median_abs_ln_ratio"] > math.log(2) for r in R.values()) else "neither")
    signs = all(np.sign(r["L6"]["model_ln_ratio_linK"]) == np.sign(r["L6"]["theta_H1"]["estimate"]) for r in R.values())
    v["L6"] = ("agreement" if signs and sum(pos({"ci95": r["L6"]["spearman"]["ci95"]}, f"L6:{m}") for m, r in R.items()) >= 2
               else "disagreement" if not signs else "neither")
    l7i = all(r["L7"]["i_nC_V_ST"] <= 2 for r in R.values())
    inside = lambda s, mg: s["ci90"][0] > -mg and s["ci90"][1] < mg
    l7ii = all(inside(r["L7"]["ii_STK@nat_vs_C1K"], r["L7"]["margin"]) and inside(r["L7"]["ii_STV@nat_vs_C1V"], r["L7"]["margin"]) for r in R.values())
    l7iii = all(r["L7"]["iii_rho_ratio_K_over_V"] >= 1 and pos(r["L7"]["iii_STK_vs_STV"], f"L7iii:{m}") and neg(r["L7"]["H3_replication"]) for m, r in R.items())
    dis_ii = any((lambda s, h2: (s["ci95"][0] > 0 or s["ci95"][1] < 0) and abs(s["estimate"]) >= h2)(r["L7"]["ii_STV@nat_vs_C1V"], r["L7"]["H2_this_run"]["estimate"]) for r in R.values())
    dis_iii = any(r["L7"]["iii_rho_ratio_K_over_V"] < 1 and neg(r["L7"]["iii_STK_vs_STV"]) for r in R.values())
    v["L7"] = {"i": l7i, "ii": l7ii, "iii": l7iii, "verdict": "agreement" if (l7i and l7ii and l7iii) else "disagreement" if (dis_ii or dis_iii) else "neither"}
    for t in ("L8a", "L8b"):
        v[t] = ("agreement" if all(pos(r[t], f"{t}:{m}") for m, r in R.items()) else "disagreement" if any(neg(r[t]) for r in R.values()) else "neither")
    for t in ("L9a", "L9b"):
        npos = sum(pos(r[t], f"{t}:{m}") for m, r in R.items())
        v[t] = ("disagreement" if (any(neg(r[t]) for r in R.values()) or sum(r[t]["estimate"] <= 0 for r in R.values()) >= 2)
                else "agreement" if npos >= 2 else "neither")
    return v, hp


def main():
    margins = {}
    rep4 = RV / "v4_runs" / "v4_report.json"
    if rep4.exists():
        pm = json.loads(rep4.read_text(encoding="utf-8"))["per_model"]
        margins = {m: pm[m]["margin"] for m in MODELS if m in pm}
    models = [m for m in MODELS if (OUT / "test" / m).exists()]
    if os.environ.get("V6_ALLOW_PARTIAL") != "1":
        incomplete = [m for m in MODELS if m not in models]
        if incomplete:
            sys.exit(f"refused: models without records {incomplete}")
    R = {m: per_model(m, margins.get(m, 0.25)) for m in models}
    v, hp = verdicts(R)
    out = {"what": "v6 test-phase report (PROTOCOL_V6 section 7)", "per_model": R, "verdicts": v, "holm_p": hp}
    (OUT / "v6_report.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(v, indent=1))


if __name__ == "__main__":
    main()
