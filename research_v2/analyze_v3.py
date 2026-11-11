"""v3 analysis (PROTOCOL_V3.md sections 5, 7, 8). CPU only; reads records written by run_v3.py; never modifies them.

  venv/Scripts/python.exe research_v2/analyze_v3.py calibrate --model M     -> v3_runs/M/calibration.json   (calibration problems only)
  venv/Scripts/python.exe research_v2/analyze_v3.py analyze                 -> v3_runs/v3_report.json / .md  (refuses before all runs finish)

Unit: the problem. Every interval resamples problems (10,000 percentile-bootstrap resamples, seed 20261103); p-values are two-sided paired sign-flip
permutation tests (10,000), except the H-C shrinkage (bootstrap p). Holm within the declared families: P (H-A x 4 models), F_m (per model), X (H-F).
Amendment 2 (PROTOCOL_V3 section 16): the realized post-rounding per-slot norm deviation is reported per arm and model (mean, p99, max, dominating
slots), and an exploratory sensitivity re-estimates the primary and secondary contrasts on steps not reached by any write with a slot above 5 %.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "v3_runs"
B_BOOT, B_PERM = 10_000, 10_000
SEED = 20261103
FLOOR = 1e-9                     # per-problem mean JS floor (E1)
FLOOR_STEP_JS, FLOOR_P = 1e-12, 1e-30
DOSES = (0.5, 1.0, 2.0)
C_LO, C_HI = 0.10, 0.40
MIN_SLOPE = 0.25
MIN_STEPS_HA = 8
SCALE_RANGE = (0.5, 2.0)
ASSUMED_SD = 0.497               # common SD of per-problem theta (E1/E1L), budget_projection.json
SD_GUARD = 1.25
A6 = ["NAT", "C1", "C2", "QA", "QO", "CV"]
RDEV_FLAG = 0.05                 # amendment 2 (rv3.core.RDEV_FLAG)
RDEV_EDGES = np.logspace(-8, 1, 361)   # amendment 2: bin edges of the per-record histogram (rv3.core.RDEV_EDGES; bin k = (edge[k-1], edge[k]])
SENS_MIN_STEPS = 8               # amendment 2 sensitivity: problems with fewer retained steps are dropped (and counted)
SENS_SEED = SEED + 2             # separate stream: the exploratory sensitivity never changes a primary or secondary estimate

ORDER = ["qwen-base", "llama-instruct", "qwen-instruct", "llama-base"]
CELLS = {"a": "qwen-base", "b": "qwen-instruct", "c": "llama-base", "d": "llama-instruct"}


# ----------------------------------------------------------------------------- primitives
def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = run
    return adj


def boot_idx(n, rng, B=None):
    B = B or B_BOOT
    return rng.integers(0, n, size=(B, n))


def signflip_p(d, rng, B=None):
    B = B or B_PERM
    d = np.asarray(d, float)
    obs = abs(d.mean())
    signs = rng.choice([-1.0, 1.0], size=(B, len(d)))
    perm = np.abs((signs * d).mean(axis=1))
    return (1 + int((perm >= obs - 1e-15).sum())) / (B + 1)


def summarize(d, rng, margin=None, informative=True, B=None):
    B = B or B_BOOT
    d = np.asarray(d, float)
    bm = d[boot_idx(len(d), rng, B)].mean(axis=1)
    out = {"n": int(len(d)), "estimate": float(d.mean()), "ci95": [float(x) for x in np.percentile(bm, [2.5, 97.5])],
           "ci90": [float(x) for x in np.percentile(bm, [5, 95])], "p": signflip_p(d, rng, min(B, B_PERM))}
    out["verdict"] = verdict(out, margin, informative)
    return out


def verdict(s, margin=None, informative=True):
    lo, hi = s["ci95"]
    if lo > 0 or hi < 0:
        return "difference"
    if margin is not None and informative and s["ci90"][0] > -margin and s["ci90"][1] < margin:
        return "equivalent"
    return "inconclusive"


def rankdata(x):
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    xs = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def logm(rec, lo=1, hi=None, stat="mean"):
    js = np.asarray(rec["js"], float)[lo - 1:(hi if hi else None)]
    v = float(np.median(js) if stat == "median" else js.mean())
    return math.log(max(v, FLOOR))


def load_jsonl(path: Path):
    """Amendment 1: every (model, arm, problem) must appear exactly once; a duplicate index fails loudly."""
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r["index"] in out:
                    raise ValueError(f"duplicate record for problem {r['index']} in {path}: each (model, arm, problem) must appear exactly once")
                out[r["index"]] = r
    return out


# ----------------------------------------------------------------------------- calibration
def calibrate_from(recs: dict, aux: dict) -> dict:
    """recs: tag -> {index: record} for NAT and C1_a{0.5,1,2}; aux: the accumulator state (numpy arrays)."""
    import torch
    import rv3.core as V
    idx = sorted(set.intersection(*[set(v) for v in recs.values()]))
    slopes = []
    for i in idx:
        y = [logm(recs[f"C1_a{a:g}"][i]) for a in DOSES]
        x = np.log(DOSES) - np.log(DOSES).mean()
        slopes.append(float((x * (np.asarray(y) - np.mean(y))).sum() / (x * x).sum()))
    b_hat = float(np.mean(slopes))
    cvs = []
    for i in idx:
        t = np.asarray(recs["NAT"][i]["target_norm_mean"], float)
        if len(t) > 3 and t.mean() > 0:
            cvs.append(t.std() / t.mean())
    c_raw = float(np.median(cvs))
    c = float(min(max(c_raw, C_LO), C_HI))
    informative = bool(b_hat >= MIN_SLOPE)
    margin = abs(b_hat) * math.log(1 + c)
    U8, E8, r80 = V.subspace_from_M(torch.as_tensor(np.asarray(aux["M"])))
    sigma2 = np.asarray(aux["d2"]) / aux["n_meas"]
    absF = np.asarray(aux["absF"]) / aux["n_meas"]
    outliers = np.argsort(-absF, axis=-1, kind="stable")[..., :4]
    rho_K, rho_V = float(np.median(aux["rhoK"])), float(np.median(aux["rhoV"]))
    rho_star = math.sqrt(rho_K * rho_V)
    s_K, s_V = rho_star / rho_K, rho_star / rho_V
    hd_desc = not (SCALE_RANGE[0] <= s_K <= SCALE_RANGE[1] and SCALE_RANGE[0] <= s_V <= SCALE_RANGE[1])
    E8n, r80n = E8.numpy(), r80.numpy()
    return {"n_calibration_problems": len(idx), "dose_slope_b_hat": b_hat, "dose_slope_problem_sd": float(np.std(slopes)),
            "dose_response_informative": informative, "cv_raw_median": c_raw, "c_used": c, "margin_log_js": margin,
            "rho_K": rho_K, "rho_V": rho_V, "rho_star": rho_star, "scale_factor_K": s_K, "scale_factor_V": s_V, "HD_descriptive": hd_desc,
            "HD_rule": f"H-D descriptive if s_K or s_V outside {list(SCALE_RANGE)}",
            "E8": {"median": float(np.median(E8n)), "min": float(E8n.min()), "max": float(E8n.max())},
            "r80": {"median": float(np.median(r80n)), "min": int(r80n.min()), "max": int(r80n.max()), "share_le_8": float((r80n <= 8).mean())},
            "E8_per_slot": E8n.tolist(), "r80_per_slot": r80n.tolist(),
            "U8": U8.numpy().tolist(), "sigma2": sigma2.astype(np.float32).tolist(), "outliers": outliers.tolist(),
            "n_query_steps": int(aux["n_q"]), "n_measure_steps": int(aux["n_meas"])}


# ----------------------------------------------------------------------------- H-C sufficient statistics
def hc_stats(nat: dict, c1: dict, idx):
    """per problem: [n, sum y, sum x, sum xy, sum xx] over steps 2..S (y = log JS^NAT - log JS^C1, x = log P_logit^NAT - log P_logit^C1)."""
    st = []
    for i in idx:
        a, b = nat[i], c1[i]
        y = np.log(np.maximum(np.asarray(a["js"][1:], float), FLOOR_STEP_JS)) - np.log(np.maximum(np.asarray(b["js"][1:], float), FLOOR_STEP_JS))
        x = np.log(np.maximum(np.asarray(a["P_logit"][1:], float), FLOOR_P)) - np.log(np.maximum(np.asarray(b["P_logit"][1:], float), FLOOR_P))
        st.append([len(y), y.sum(), x.sum(), (x * y).sum(), (x * x).sum()])
    return np.asarray(st, float)


def hc_betas(S):
    """S [..., 5] summed statistics -> (beta0, beta1). beta1 = intercept of pooled OLS y = b1 + g x."""
    n, sy, sx, sxy, sxx = (S[..., k] for k in range(5))
    b0 = sy / n
    var = sxx - sx * sx / n
    g = np.where(var > 0, (sxy - sx * sy / n) / np.where(var > 0, var, 1.0), 0.0)
    b1 = b0 - g * sx / n
    return b0, b1


def band(f_hat, ci):
    if f_hat >= 0.5 and ci[0] > 0:
        return "substantial"
    if ci[0] > 0 and f_hat < 0.5:
        return "partial"
    if ci[1] < 0.25:
        return "no"
    return "inconclusive"


def hc_model(stats, rng, B=None):
    B = B or B_BOOT
    W = np.zeros((B, len(stats)))
    bi = boot_idx(len(stats), rng, B)
    np.add.at(W, (np.repeat(np.arange(B), len(stats)), bi.ravel()), 1.0)
    Sb = W @ stats
    b0, b1 = hc_betas(stats.sum(0))
    B0, B1 = hc_betas(Sb)
    shrink = np.abs(B0) - np.abs(B1)
    frac = 1 - np.abs(B1) / np.where(B0 != 0, np.abs(B0), np.nan)
    s_hat = abs(b0) - abs(b1)
    f_hat = 1 - abs(b1) / abs(b0) if b0 != 0 else float("nan")
    p = float(min(1.0, max(1.0 / B, 2 * min((shrink <= 0).mean(), (shrink >= 0).mean()))))
    ci_b0 = [float(x) for x in np.percentile(B0, [2.5, 97.5])]
    ci_f = [float(x) for x in np.nanpercentile(frac, [2.5, 97.5])]
    applicable = bool(ci_b0[0] > 0 or ci_b0[1] < 0)
    return {"beta0": float(b0), "beta1": float(b1), "beta0_ci95": ci_b0, "shrinkage": float(s_hat), "shrinkage_ci95": [float(x) for x in np.percentile(shrink, [2.5, 97.5])],
            "fraction": float(f_hat), "fraction_ci95": ci_f, "p": p, "applicable": applicable, "band": band(f_hat, ci_f) if applicable else "not applicable"}, W


def hc_cross(stA, stB, rng, B=None):
    B = B or B_BOOT
    """Same problems resampled for both models (stA, stB aligned by problem)."""
    W = np.zeros((B, len(stA)))
    bi = boot_idx(len(stA), rng, B)
    np.add.at(W, (np.repeat(np.arange(B), len(stA)), bi.ravel()), 1.0)
    a0, a1 = hc_betas(W @ stA)
    c0, c1 = hc_betas(W @ stB)
    g0b, g1b = a0 - c0, a1 - c1
    A0, A1 = hc_betas(stA.sum(0))
    C0, C1 = hc_betas(stB.sum(0))
    g0, g1 = A0 - C0, A1 - C1
    F = 1 - abs(g1) / abs(g0) if g0 != 0 else float("nan")
    Fb = 1 - np.abs(g1b) / np.where(g0b != 0, np.abs(g0b), np.nan)
    ci_g0 = [float(x) for x in np.percentile(g0b, [2.5, 97.5])]
    ci_F = [float(x) for x in np.nanpercentile(Fb, [2.5, 97.5])]
    applicable = bool(ci_g0[0] > 0 or ci_g0[1] < 0)
    out = {"g0": float(g0), "g0_ci95": ci_g0, "g1": float(g1), "g1_ci95": [float(x) for x in np.percentile(g1b, [2.5, 97.5])],
           "F": float(F), "F_ci95": ci_F, "precondition_met": applicable, "band": band(F, ci_F) if applicable else "not applicable"}
    if not applicable:
        out["finding"] = ("The Qwen-base vs Llama-instruct difference in the NAT-C1 contrast seen across sessions in E1/E1L did not replicate when both "
                          "models were rerun in the same session (95% CI of the unadjusted gap includes 0).")
    return out


# ----------------------------------------------------------------------------- per-model analysis
def theta(A, a, b, idx, **kw):
    return np.array([logm(A[a][i], **kw) - logm(A[b][i], **kw) for i in idx])


def ha_D(nat: dict, idx, pred="P_logit", base="P_norm"):
    D, excl = [], 0
    for i in idx:
        r = nat[i]
        if r["S"] < MIN_STEPS_HA:
            excl += 1
            continue
        js = r["js"][1:]
        d = spearman(r[pred][1:], js) - spearman(r[base][1:], js)
        if d != d:
            excl += 1
            continue
        D.append((i, d))
    return D, excl


def ha_pooled_arms(A, idx):
    D = []
    for i in idx:
        if A["NAT"][i]["S"] < MIN_STEPS_HA:
            continue
        P, N, J = [], [], []
        for a in A6:
            if a in A:
                r = A[a][i]
                P += r["P_logit"][1:]
                N += r["P_norm"][1:]
                J += r["js"][1:]
        d = spearman(P, J) - spearman(N, J)
        if d == d:
            D.append((i, d))
    return D


# ----------------------------------------------------------------------------- amendment 2: realized deviation (reported) and sensitivity
def rdev_report(A: dict, idx):
    """Per arm: realized post-rounding relative per-slot norm deviation |r - t| / t over every perturbed slot (t > 0) of every step and problem."""
    out = {}
    for a, recs in A.items():
        rs = [recs[i] for i in idx if i in recs]
        n = int(sum(r["rdev_count"] for r in rs))
        if n == 0:
            out[a] = {"n_slots": 0, "note": "no constructed slots (natural or exact arm)"}
            continue
        hist = np.sum([r["rdev_hist"] for r in rs], axis=0)
        k = int(np.searchsorted(np.cumsum(hist) / hist.sum(), 0.99))
        steps = [(float(r["rdev_max"][s]), r["index"], s + 1, r["rdev_arg"][s]) for r in rs for s in range(len(r["rdev_max"]))]
        top = sorted(steps, key=lambda x: -x[0])[:5]
        fl = [tuple(x[3]) for x in steps if x[0] > RDEV_FLAG and x[3] is not None]
        out[a] = {"n_slots": n, "mean": float(sum(r["rdev_sum"] for r in rs) / n),
                  "p99_at_most": float(RDEV_EDGES[k]) if k < len(RDEV_EDGES) else float("inf"), "max": top[0][0],
                  "largest": [{"deviation": d, "problem": i, "step": st, "slot": f"L{sl[0]} {sl[1]} h{sl[2]}" if sl else None} for d, i, st, sl in top],
                  "n_steps": len(steps), "steps_with_a_slot_above_5pct": len(fl),
                  "slots_dominating_those_steps": [{"slot": f"L{l} {pth} h{h}", "share": c / len(fl)} for (l, pth, h), c in Counter(fl).most_common(5)]}
    return out


def first_unaffected(rec):
    """Leading steps whose JS no flagged write can reach: the entry written at step s enters the logits from step s + 1 on, so with the first
    write that has a slot above 5 % at step f, steps 1..f are kept; without one, all S steps."""
    for s, n in enumerate(rec["rdev_n_gt5"]):
        if n > 0:
            return s + 1
    return len(rec["js"])


def truncated(rec, k):
    S = len(rec["js"])
    out = {key: (v[:k] if isinstance(v, list) and len(v) == S else v) for key, v in rec.items()}
    out["S"] = k
    return out


def sens_data(A, arms, idx):
    """Pairs every arm of a contrast on the same retained steps (the minimum over the arms) and drops problems with < SENS_MIN_STEPS."""
    Ad, kept, dropped, frac = {a: {} for a in arms}, [], 0, []
    for i in idx:
        k = min(first_unaffected(A[a][i]) for a in arms)
        if k < SENS_MIN_STEPS:
            dropped += 1
            continue
        kept.append(i)
        frac.append(k / len(A[arms[0]][i]["js"]))
        for a in arms:
            Ad[a][i] = truncated(A[a][i], k)
    return Ad, kept, dropped, (float(np.median(frac)) if frac else None)


def rounding_sensitivity(A: dict, cal: dict, idx):
    rng = np.random.default_rng(SENS_SEED)
    margin, inf = cal["margin_log_js"], cal["dose_response_informative"]
    out = {"label": "EXPLORATORY (amendment 2): steps reached by a write with a slot whose realized post-rounding norm deviation exceeds 5 % "
                    "are excluded (truncation at the first such write; arms of a contrast share the retained steps)", "contrasts": {}}

    def put(name, arms, fn):
        if not all(a in A for a in arms):
            return
        Ad, kept, dropped, frac = sens_data(A, arms, idx)
        v = fn(Ad, kept) if len(kept) >= 2 else None
        out["contrasts"][name] = {"n_kept": len(kept), "n_dropped": dropped, "median_retained_fraction": frac, "result": v}
    put("HA (NAT, primary)", ("NAT",), lambda Ad, k: summarize([d for _, d in ha_D(Ad["NAT"], k)[0]], rng))
    for name, a, b in (("NAT-C1", "NAT", "C1"), ("HB QA-QO", "QA", "QO"), ("HD K*-V*", "NAT-K*", "NAT-V*"), ("Prov-K NAT-K-C1-K", "NAT-K", "C1-K"),
                       ("Prov-V NAT-V-C1-V", "NAT-V", "C1-V"), ("HE INH-LOC", "INH", "LOC"), ("NAT-C2", "NAT", "C2"), ("C2-C1", "C2", "C1"),
                       ("NAT-CV", "NAT", "CV"), ("NAT-K-NAT-V", "NAT-K", "NAT-V")):
        put(name, (a, b), lambda Ad, k, a=a, b=b: summarize(theta(Ad, a, b, k), rng, margin, inf))
    put("HC shrinkage", ("NAT", "C1"), lambda Ad, k: {x: y for x, y in hc_model(hc_stats(Ad["NAT"], Ad["C1"], k), rng)[0].items()
                                                      if x in ("applicable", "shrinkage", "shrinkage_ci95", "fraction", "fraction_ci95", "band")})
    six = tuple(a for a in A6 if a in A)
    put("HA pooled norm-matched arms (secondary)", six, lambda Ad, k: summarize([d for _, d in ha_pooled_arms(Ad, k)], rng))
    Ad, kept, _, _ = sens_data(A, ("NAT", "C1"), idx)
    out["theta_NAT_C1_per_problem"] = dict(zip(kept, theta(Ad, "NAT", "C1", kept).tolist())) if kept else {}
    return out


def analyze_model(A: dict, cal: dict, idx, rng, exposed=()):
    margin, inf = cal["margin_log_js"], cal["dose_response_informative"]
    res = {"n_problems": len(idx), "margin": margin, "dose_informative": inf}
    D, excl = ha_D(A["NAT"], idx)
    res["HA"] = {**summarize([d for _, d in D], rng), "excluded_problems": excl, "per_problem": dict(D)}
    res["HA"]["verdict"] = "difference" if res["HA"]["verdict"] == "difference" else "inconclusive"
    fam = {}
    fam["NAT-C1"] = summarize(theta(A, "NAT", "C1", idx), rng, margin, inf)
    fam["HB QA-QO"] = summarize(theta(A, "QA", "QO", idx), rng, margin, inf)
    hc, _ = hc_model(hc_stats(A["NAT"], A["C1"], idx), rng)
    res["HC"] = hc
    if hc["applicable"]:
        fam["HC shrinkage"] = {"estimate": hc["shrinkage"], "ci95": hc["shrinkage_ci95"], "p": hc["p"], "verdict": hc["band"]}
    hd = summarize(theta(A, "NAT-K*", "NAT-V*", idx), rng, margin, inf)
    if cal["HD_descriptive"]:
        res["HD_descriptive"] = {**hd, "reason": "scale factor outside [0.5, 2]: extrapolation beyond the measured dose range"}
    else:
        fam["HD K*-V*"] = hd
    fam["Prov-K NAT-K-C1-K"] = summarize(theta(A, "NAT-K", "C1-K", idx), rng, margin, inf)
    fam["Prov-V NAT-V-C1-V"] = summarize(theta(A, "NAT-V", "C1-V", idx), rng, margin, inf)
    fam["HE INH-LOC"] = summarize(theta(A, "INH", "LOC", idx), rng, margin, inf)
    ph = holm([v["p"] for v in fam.values()])
    for (k, v), h in zip(fam.items(), ph):
        v["holm_p"] = float(h)
    res["family"] = fam
    # readings for the directional / competing hypotheses
    hdv = fam.get("HD K*-V*", res.get("HD_descriptive"))
    res["HD_reading"] = ("K more disruptive: supports the storage-level account" if hdv["verdict"] == "difference" and hdv["estimate"] > 0 else
                         "V more disruptive: supports the weight-projection account" if hdv["verdict"] == "difference" else
                         "neither path dominates (equivalent)" if hdv["verdict"] == "equivalent" else "inconclusive")
    he = fam["HE INH-LOC"]
    res["HE_reading"] = ("INH > LOC: consistent with prior work" if he["verdict"] == "difference" and he["estimate"] > 0 else
                         "SURPRISE: LOC >= INH" if (he["verdict"] == "difference" and he["estimate"] < 0) or he["verdict"] == "equivalent" else "inconclusive")
    hb = fam["HB QA-QO"]
    res["HB_falsified"] = bool((hb["verdict"] == "difference" and hb["estimate"] < 0) or hb["verdict"] == "equivalent")
    # descriptive
    desc = {}
    for a, b in (("NAT", "C2"), ("C2", "C1"), ("NAT", "CV"), ("NAT-K", "NAT-V")):
        if a in A and b in A:
            desc[f"{a}-{b}"] = summarize(theta(A, a, b, idx), rng, margin, inf)
    ratio = [math.exp(logm(A["NAT"][i])) / (math.exp(logm(A["LOC"][i])) + math.exp(logm(A["INH"][i]))) for i in idx]
    bm = np.asarray(ratio)[boot_idx(len(ratio), rng)]
    desc["nonadditivity m_NAT/(m_LOC+m_INH)"] = {"median": float(np.median(ratio)), "ci95_of_median": [float(x) for x in np.percentile(np.median(bm, axis=1), [2.5, 97.5])]}
    for a in ("NAT", "LOC", "INH"):
        desc[f"mean ||Delta||^2 {a} (K, V)"] = [float(np.mean([np.mean(A[a][i]["dn2_K"]) for i in idx])), float(np.mean([np.mean(A[a][i]["dn2_V"]) for i in idx]))]
    for k in ("P_val", "P_anormK", "P_out"):
        Dk, _ = ha_D(A["NAT"], idx, pred=k)
        desc[f"HA-type {k} vs P_norm (NAT)"] = summarize([d for _, d in Dk], rng) if Dk else None
    Dp = ha_pooled_arms(A, idx)
    desc["HA pooled norm-matched arms (secondary)"] = summarize([d for _, d in Dp], rng) if Dp else None
    desc["rel norm K / V (NAT, mean)"] = [float(np.mean([np.mean(A["NAT"][i]["rel_K"]) for i in idx])), float(np.mean([np.mean(A["NAT"][i]["rel_V"]) for i in idx]))]
    desc["calibration diagnostics"] = {k: cal[k] for k in ("E8", "r80", "rho_K", "rho_V", "rho_star", "scale_factor_K", "scale_factor_V", "HD_descriptive")}
    res["descriptive"] = desc
    # sensitivity (none replaces a primary)
    sens = {}
    keep = [i for i in idx if i not in set(exposed)]
    long_ = [i for i in idx if A["NAT"][i]["S"] >= 32]
    for name, a, b in (("NAT-C1", "NAT", "C1"), ("HB", "QA", "QO"), ("HD", "NAT-K*", "NAT-V*"), ("HE", "INH", "LOC")):
        sens[name] = {"steps 2-8": summarize(theta(A, a, b, idx, lo=2, hi=8), rng)["estimate"],
                      "steps 9-64": summarize(theta(A, a, b, [i for i in idx if A[a][i]["S"] >= 9], lo=9), rng)["estimate"],
                      "per-problem median": summarize(theta(A, a, b, idx, stat="median"), rng)["estimate"],
                      "excluding exposed": summarize(theta(A, a, b, keep), rng)["estimate"],
                      "S >= 32": summarize(theta(A, a, b, long_), rng)["estimate"] if long_ else None}
    res["sensitivity"] = sens
    res["theta_NAT_C1_per_problem"] = dict(zip(idx, theta(A, "NAT", "C1", idx).tolist()))
    # amendment 2: reported, and exploratory (own random stream; nothing above depends on it)
    res["realized_norm_deviation"] = rdev_report(A, idx)
    res["sensitivity_rounding"] = rounding_sensitivity(A, cal, idx)
    return res


def analyze_all(data: dict, cals: dict, exposed=()) -> dict:
    """data: model -> arm -> {index: record}; cals: model -> calibration dict."""
    rng = np.random.default_rng(SEED)
    models = [m for m in ORDER if m in data]
    rep = {"models": models, "per_model": {}}
    idx_m = {}
    for m in models:
        A = data[m]
        idx = sorted(set.intersection(*[set(v) for v in A.values()]))
        idx_m[m] = idx
        rep["per_model"][m] = analyze_model(A, cals[m], idx, rng, exposed)
    # primary family P: H-A per model
    pa = holm([rep["per_model"][m]["HA"]["p"] for m in models])
    for m, h in zip(models, pa):
        H = rep["per_model"][m]["HA"]
        H["holm_p"] = float(h)
        H["supported"] = bool(h < 0.05 and H["ci95"][0] > 0)
        H["falsified"] = bool(H["ci95"][1] < 0)
    shared = sorted(set.intersection(*[set(idx_m[m]) for m in models]))
    D = [np.mean([rep["per_model"][m]["HA"]["per_problem"][i] for m in models]) for i in shared
         if all(i in rep["per_model"][m]["HA"]["per_problem"] for m in models)]
    rep["HA_pooled_secondary"] = summarize(D, rng) if D else None
    # H-C cross-model decision: Qwen-base vs Llama-instruct; other pairs descriptive
    rep["HC_cross"] = {}
    for A_, B_ in [(a, b) for k, a in enumerate(models) for b in models[k + 1:]]:
        sh = sorted(set(idx_m[A_]) & set(idx_m[B_]))
        r = hc_cross(hc_stats(data[A_]["NAT"], data[A_]["C1"], sh), hc_stats(data[B_]["NAT"], data[B_]["C1"], sh), rng)
        r["role"] = "decision rule" if {A_, B_} == {"qwen-base", "llama-instruct"} else "descriptive"
        if {A_, B_} == {"qwen-base", "llama-instruct"} and A_ != "qwen-base":
            r["note"] = "gap oriented as first-listed minus second-listed model"
        rep["HC_cross"][f"{A_} vs {B_}"] = r
    # H-F (family X)
    if all(CELLS[k] in data for k in CELLS):
        T = {k: rep["per_model"][CELLS[k]]["theta_NAT_C1_per_problem"] for k in CELLS}
        sh = sorted(set.intersection(*[set(v) for v in T.values()]))
        a, b, c, d = (np.array([T[k][i] for i in sh]) for k in "abcd")
        sds = {CELLS[k]: float(np.std(v, ddof=1)) for k, v in zip("abcd", (a, b, c, d))}
        desc = any(s > ASSUMED_SD * SD_GUARD for s in sds.values())
        margins = [cals[CELLS[k]]["margin_log_js"] for k in CELLS]
        inf = all(cals[CELLS[k]]["dose_response_informative"] for k in CELLS)
        mg = float(np.mean(margins))
        X = {"family (Llama - Qwen)": summarize(0.5 * ((c + d) - (a + b)), rng, mg, inf),
             "tuning (instruct - base)": summarize(0.5 * ((b + d) - (a + c)), rng, mg, inf),
             "interaction": summarize(0.5 * ((a - b) - (c - d)), rng, mg, inf)}
        hx = holm([v["p"] for v in X.values()])
        for (k, v), h in zip(X.items(), hx):
            v["holm_p"] = float(h)
        rep["HF"] = {"label": "descriptive (25% SD guard triggered)" if desc else "test across these four models (one model per cell)",
                     "sd_per_model": sds, "assumed_sd": ASSUMED_SD, "contrasts": X, "n_shared": len(sh),
                     "readings": {"family account": _acct(X["family (Llama - Qwen)"]), "tuning account": _acct(X["tuning (instruct - base)"]),
                                  "additivity": "falsified" if X["interaction"]["verdict"] == "difference" else "not falsified"}}
    else:
        rep["HF"] = {"label": "not estimable (not all four models completed)"}
    if all(CELLS[k] in data for k in CELLS):                     # amendment 2: exploratory H-F on the truncated per-problem theta
        srng = np.random.default_rng(SENS_SEED + 1)
        T = {k: rep["per_model"][CELLS[k]]["sensitivity_rounding"]["theta_NAT_C1_per_problem"] for k in CELLS}
        sh = sorted(set.intersection(*[set(v) for v in T.values()]))
        if len(sh) >= 2:
            a, b, c, d = (np.array([T[k][i] for i in sh]) for k in "abcd")
            mg = float(np.mean([cals[CELLS[k]]["margin_log_js"] for k in CELLS]))
            inf = all(cals[CELLS[k]]["dose_response_informative"] for k in CELLS)
            rep["HF_sensitivity_rounding"] = {"label": "EXPLORATORY (amendment 2)", "n_shared": len(sh),
                                              "family (Llama - Qwen)": summarize(0.5 * ((c + d) - (a + b)), srng, mg, inf),
                                              "tuning (instruct - base)": summarize(0.5 * ((b + d) - (a + c)), srng, mg, inf),
                                              "interaction": summarize(0.5 * ((a - b) - (c - d)), srng, mg, inf)}
        else:
            rep["HF_sensitivity_rounding"] = {"label": "EXPLORATORY (amendment 2)", "n_shared": len(sh), "note": "fewer than 2 shared problems retained"}
    return rep


def _acct(s):
    if s["verdict"] == "difference":
        return "supported" if s["estimate"] > 0 else "falsified"
    return "falsified" if s["verdict"] == "equivalent" else "inconclusive"


def to_markdown(rep):
    L = ["# v3 results (generated by analyze_v3.py)", ""]
    L += ["## Primary family P: H-A per model", "", "| Model | D | 95% CI | p | Holm p | supported | falsified | excluded |", "|---|---|---|---|---|---|---|---|"]
    for m in rep["models"]:
        h = rep["per_model"][m]["HA"]
        L.append(f"| {m} | {h['estimate']:+.4f} | [{h['ci95'][0]:+.4f}, {h['ci95'][1]:+.4f}] | {h['p']:.4f} | {h['holm_p']:.4f} | {h['supported']} | {h['falsified']} | {h['excluded_problems']} |")
    for m in rep["models"]:
        r = rep["per_model"][m]
        L += ["", f"## Family F_m: {m} (margin {r['margin']:.3f}, dose informative {r['dose_informative']})", "",
              "| Test | Estimate | 95% CI | p | Holm p | Verdict |", "|---|---|---|---|---|---|"]
        for k, v in r["family"].items():
            ci = v.get("ci95")
            L.append(f"| {k} | {v['estimate']:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | {v['p']:.4f} | {v['holm_p']:.4f} | {v['verdict']} |")
        L += ["", f"H-C: {r['HC']['band']} (fraction {r['HC']['fraction']:+.3f}, CI {r['HC']['fraction_ci95']}); H-D: {r['HD_reading']}; H-E: {r['HE_reading']}"]
    L += ["", "## H-C cross-model", ""]
    for k, v in rep["HC_cross"].items():
        L.append(f"- {k} ({v['role']}): g0 {v['g0']:+.4f} {v['g0_ci95']}, F {v['F']:+.3f} {v['F_ci95']}, band {v['band']}" + (f" — {v['finding']}" if "finding" in v else ""))
    L += ["", "## H-F", "", f"Label: {rep['HF'].get('label')}"]
    for k, v in rep["HF"].get("contrasts", {}).items():
        L.append(f"- {k}: {v['estimate']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}], Holm p {v['holm_p']:.4f}, {v['verdict']}")
    L += ["", "## Realized post-rounding per-slot norm deviation (reported; amendment 2)", "",
          "| Model | Arm | slots | mean | p99 (at most) | max | steps with a slot > 5 % | dominating slots |", "|---|---|---|---|---|---|---|---|"]
    for m in rep["models"]:
        for a, v in rep["per_model"][m]["realized_norm_deviation"].items():
            if v["n_slots"]:
                dom = "; ".join(f"{x['slot']} {x['share']:.0%}" for x in v["slots_dominating_those_steps"][:3])
                L.append(f"| {m} | {a} | {v['n_slots']} | {v['mean']:.2e} | {v['p99_at_most']:.2e} | {v['max']:.3f} | "
                         f"{v['steps_with_a_slot_above_5pct']}/{v['n_steps']} | {dom} |")
    L += ["", "## Sensitivity: excluding steps reached by a slot above 5 % (EXPLORATORY; amendment 2)", ""]
    for m in rep["models"]:
        for k, v in rep["per_model"][m]["sensitivity_rounding"]["contrasts"].items():
            r = v["result"] or {}
            est = r.get("estimate", r.get("shrinkage"))
            ci = r.get("ci95", r.get("shrinkage_ci95"))
            L.append(f"- {m} {k}: kept {v['n_kept']}, dropped {v['n_dropped']}, median retained {v['median_retained_fraction']}; "
                     + (f"{est:+.4f} {ci}" if est is not None else "not estimable"))
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["calibrate", "analyze"])
    ap.add_argument("--model", choices=ORDER)
    ap.add_argument("--dir", default=str(OUT))
    ap.add_argument("--final-truncated", action="store_true", help="analyze completed models after a ceiling truncation (ledger at the cap)")
    a = ap.parse_args()
    root = Path(a.dir)
    sys.path.insert(0, str(HERE))
    if a.cmd == "calibrate":
        import torch
        d = root / a.model / "calib"
        recs = {t: load_jsonl(d / f"{t}.jsonl") for t in ("NAT", "C1_a0.5", "C1_a1", "C1_a2")}
        aux = {k: (v.numpy() if hasattr(v, "numpy") else v) for k, v in torch.load(d / "aux_state.pt").items()}
        out = calibrate_from(recs, aux)
        p = root / a.model / "calibration.json"
        if p.exists():
            sys.exit("calibration.json exists already (frozen); refusing to overwrite")
        p.write_text(json.dumps(out))
        print({k: v for k, v in out.items() if k not in ("U8", "sigma2", "outliers", "E8_per_slot", "r80_per_slot")})
        return
    done = [m for m in ORDER if (root / m / "tierA_complete.json").exists()]
    if len(done) < 4 and not a.final_truncated:
        sys.exit(f"Refusing to analyze before all runs finish (complete: {done}).")
    sample = json.loads((HERE / "v3_sample.json").read_text())
    design = json.loads((root / "design.json").read_text())
    data, cals = {}, {}
    from rv3.core import TEST_ARMS
    for m in ORDER:
        d = root / m / "tierA"
        if not d.exists():
            continue
        arms = [x for x in TEST_ARMS if x not in design.get("dropped_arms", [])]
        from rv3.core import file_tag
        data[m] = {t: load_jsonl(d / f"{file_tag(t)}.jsonl") for t in arms}
        cals[m] = json.loads((root / m / "calibration.json").read_text())
    rep = analyze_all(data, cals, set(sample["exposed_test_ids_in_sample"]))
    rep["design"] = design
    (root / "v3_report.json").write_text(json.dumps(rep, indent=1, default=float))
    (root / "v3_report.md").write_text(to_markdown(rep), encoding="utf-8")
    print(to_markdown(rep))


if __name__ == "__main__":
    main()
