"""Phase 5: independent recomputation of the v3 analysis from the raw Tier A records (PROTOCOL_V3 sections 5-8, 16). Not locked.

Written from the protocol text. Imports no project module (numpy, json, pathlib only). Methods differ from analyze_v3.py on purpose:
multinomial-count bootstrap (10,000, seed 4242031), 100,000 sign flips, own ranking / Holm / OLS, H-C point estimates refit by least squares on
the step-level data. Tolerances: v3/phase5/RECOMPUTATION_TOLERANCES.md (declared before this was run).

  venv/Scripts/python.exe research_v2/v3/phase5/recompute_v3_independent.py
    -> v3/phase5/recompute_v3.json (own results) and v3/phase5/recompute_vs_locked.json / .md (comparison with v3_runs/v3_report.json)
"""
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

RV = Path(__file__).resolve().parents[2]
RUNS = RV / "v3_runs"
HERE = Path(__file__).resolve().parent

MODELS = ["qwen-base", "llama-instruct", "qwen-instruct", "llama-base"]
ARMS = ["NAT", "C1", "C2", "QA", "QO", "CV", "NAT-K", "NAT-V", "NAT-K*", "NAT-V*", "C1-K", "C1-V", "LOC", "INH", "FP16_history"]
NB, NP = 10_000, 100_000
SEED, SEED_SENS, SEED_SENS_HF = 4242031, 4242032, 4242033
M_FLOOR, JS_FLOOR, P_FLOOR = 1e-9, 1e-12, 1e-30
SIX = ["NAT", "C1", "C2", "QA", "QO", "CV"]
SD_ASSUMED = 0.497
FLAG = 0.05
EDGES = 10.0 ** np.linspace(-8.0, 1.0, 361)


# ------------------------------------------------------------------ records
def load():
    sample = json.loads((RV / "v3_sample.json").read_text())
    data = {}
    for m in MODELS:
        data[m] = {}
        for a in ARMS:
            recs = {}
            for ln in (RUNS / m / "tierA" / (a.replace("*", "star") + ".jsonl")).read_text(encoding="utf-8").splitlines():
                if not ln.strip():
                    continue
                r = json.loads(ln)
                assert r["index"] not in recs, (m, a, r["index"])
                recs[r["index"]] = r
            data[m][a] = recs
    return data, sample


def check_integrity(data):
    """Independent re-check of what every record claims (section 9 per-problem integrity, section 16 construction/rounding)."""
    bad = []
    for m, A in data.items():
        ids = set(A["NAT"])
        for a, recs in A.items():
            if set(recs) != ids:
                bad.append((m, a, "index set differs from NAT"))
            if a == "FP16_history":
                continue
            for i, r in recs.items():
                js = np.asarray(r["js"], float)
                if r.get("split") != "test" or len(js) != r["S"] or not np.all(np.isfinite(js)) or js.min() < 0 or js[0] != 0.0:
                    bad.append((m, a, i, "js/S/split"))
                for k in ("exact_ok", "natural_ok", "routing_reset", "construction_ok", "rounding_ok"):
                    if r.get(k) is not True:
                        bad.append((m, a, i, k))
                for k in ("P_norm", "P_logit", "P_val", "P_anormK", "P_out"):
                    if len(r[k]) != r["S"]:
                        bad.append((m, a, i, k + " length"))
        S_by = {a: {i: A[a][i]["S"] for i in ids} for a in A if a != "FP16_history"}
        if any(S_by[a] != S_by["NAT"] for a in S_by):
            bad.append((m, "S differs between arms"))
    return bad


# ------------------------------------------------------------------ primitives (own implementations)
def avg_ranks(x):
    x = np.asarray(x, float)
    uniq, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    start = np.concatenate([[0], np.cumsum(cnt)[:-1]])          # 0-based first position of each tied group in sorted order
    return (start + (cnt + 1) / 2.0)[inv]


def rho_s(x, y):
    rx, ry = avg_ranks(x), avg_ranks(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    den = math.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def holm_adjust(p):
    p = np.asarray(p, float)
    k = len(p)
    o = np.argsort(p, kind="stable")
    stepped = np.minimum(1.0, np.maximum.accumulate((k - np.arange(k)) * p[o]))
    out = np.empty(k)
    out[o] = stepped
    return out


class Boot:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.cache = {}

    def counts(self, n):
        return self.rng.multinomial(n, np.full(n, 1.0 / n), size=NB).astype(float)

    def mean_dist(self, d):
        d = np.asarray(d, float)
        return self.counts(len(d)) @ d / len(d)

    def flip_p(self, d):
        d = np.asarray(d, float)
        obs = abs(d.mean())
        hits, done = 0, 0
        while done < NP:
            b = min(20_000, NP - done)
            s = self.rng.integers(0, 2, size=(b, len(d))) * 2.0 - 1.0
            hits += int((np.abs(s @ d / len(d)) >= obs * (1 - 1e-12)).sum())
            done += b
        return (hits + 1) / (NP + 1)


def pct(x, q):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return [float(v) for v in np.percentile(x, q)]


def decide(est_ci95, ci90, margin, informative, allow_equiv=True):
    lo, hi = est_ci95
    if lo > 0 or hi < 0:
        return "difference"
    if allow_equiv and margin is not None and informative and -margin < ci90[0] and ci90[1] < margin:
        return "equivalent"
    return "inconclusive"


def paired(d, bt, margin=None, informative=True, allow_equiv=True):
    d = np.asarray(d, float)
    dist = bt.mean_dist(d)
    ci95, ci90 = pct(dist, [2.5, 97.5]), pct(dist, [5, 95])
    return {"n": int(len(d)), "estimate": float(d.mean()), "ci95": ci95, "ci90": ci90, "p": bt.flip_p(d),
            "verdict": decide(ci95, ci90, margin, informative, allow_equiv)}


# ------------------------------------------------------------------ estimands
def mlog(r, a=1, b=None, med=False):
    js = np.asarray(r["js"], float)[a - 1:b]
    v = np.median(js) if med else js.mean()
    return math.log(max(float(v), M_FLOOR))


def theta_vec(A, x, y, ids, **kw):
    return np.array([mlog(A[x][i], **kw) - mlog(A[y][i], **kw) for i in ids])


def D_vec(nat, ids, pred="P_logit"):
    out, excl = {}, 0
    for i in ids:
        r = nat[i]
        if r["S"] < 8:
            excl += 1
            continue
        js = r["js"][1:r["S"]]
        v = rho_s(r[pred][1:r["S"]], js) - rho_s(r["P_norm"][1:r["S"]], js)
        if math.isnan(v):
            excl += 1
            continue
        out[i] = v
    return out, excl


def D_pooled_arms(A, ids):
    out = {}
    for i in ids:
        if A["NAT"][i]["S"] < 8:
            continue
        P = np.concatenate([A[a][i]["P_logit"][1:A[a][i]["S"]] for a in SIX])
        N = np.concatenate([A[a][i]["P_norm"][1:A[a][i]["S"]] for a in SIX])
        J = np.concatenate([A[a][i]["js"][1:A[a][i]["S"]] for a in SIX])
        v = rho_s(P, J) - rho_s(N, J)
        if not math.isnan(v):
            out[i] = v
    return out


def hc_steps(nat, c1, ids):
    """per problem (y, x) arrays over steps 2..S."""
    out = []
    for i in ids:
        a, b = nat[i], c1[i]
        S = a["S"]
        y = np.log(np.maximum(np.asarray(a["js"][1:S], float), JS_FLOOR)) - np.log(np.maximum(np.asarray(b["js"][1:S], float), JS_FLOOR))
        x = np.log(np.maximum(np.asarray(a["P_logit"][1:S], float), P_FLOOR)) - np.log(np.maximum(np.asarray(b["P_logit"][1:S], float), P_FLOOR))
        out.append((y, x))
    return out


def ols_betas(steps):
    y = np.concatenate([s[0] for s in steps])
    x = np.concatenate([s[1] for s in steps])
    b0 = float(y.mean())
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b0, float(coef[0])


def moments(steps):
    return np.array([[len(y), y.sum(), x.sum(), (x * y).sum(), (x * x).sum()] for y, x in steps])


def betas_from(M):
    n, sy, sx, sxy, sxx = M.T
    b0 = sy / n
    sxx_c = sxx - sx ** 2 / n
    slope = np.divide(sxy - sx * sy / n, sxx_c, out=np.zeros_like(sxx_c), where=sxx_c > 0)
    return b0, b0 - slope * sx / n


def band(f, ci):
    if f >= 0.5 and ci[0] > 0:
        return "substantial"
    if ci[0] > 0:
        return "partial"
    if ci[1] < 0.25:
        return "no"
    return "inconclusive"


def hc_one(steps, bt):
    b0, b1 = ols_betas(steps)
    W = bt.counts(len(steps))
    B0, B1 = betas_from(W @ moments(steps))
    sh = np.abs(B0) - np.abs(B1)
    with np.errstate(divide="ignore", invalid="ignore"):
        fr = 1 - np.abs(B1) / np.abs(B0)
    ci_b0 = pct(B0, [2.5, 97.5])
    f = 1 - abs(b1) / abs(b0)
    ci_f = pct(fr, [2.5, 97.5])
    app = ci_b0[0] > 0 or ci_b0[1] < 0
    return {"beta0": b0, "beta1": b1, "beta0_ci95": ci_b0, "beta1_ci95": pct(B1, [2.5, 97.5]), "shrinkage": abs(b0) - abs(b1),
            "shrinkage_ci95": pct(sh, [2.5, 97.5]), "fraction": f, "fraction_ci95": ci_f,
            "p": float(2 * min((sh <= 0).mean(), (sh >= 0).mean())), "applicable": bool(app), "band": band(f, ci_f) if app else "not applicable"}


def hc_pair(stA, stB, bt):
    a0, a1 = ols_betas(stA)
    c0, c1 = ols_betas(stB)
    g0, g1 = a0 - c0, a1 - c1
    W = bt.counts(len(stA))                                          # the same problems for both models
    A0, A1 = betas_from(W @ moments(stA))
    C0, C1 = betas_from(W @ moments(stB))
    G0, G1 = A0 - C0, A1 - C1
    with np.errstate(divide="ignore", invalid="ignore"):
        Fb = 1 - np.abs(G1) / np.abs(G0)
    ci_g0 = pct(G0, [2.5, 97.5])
    F = 1 - abs(g1) / abs(g0)
    ci_F = pct(Fb, [2.5, 97.5])
    ok = ci_g0[0] > 0 or ci_g0[1] < 0
    return {"g0": g0, "g0_ci95": ci_g0, "g1": g1, "g1_ci95": pct(G1, [2.5, 97.5]), "F": F, "F_ci95": ci_F,
            "precondition_met": bool(ok), "band": band(F, ci_F) if ok else "not applicable"}


# ------------------------------------------------------------------ amendment 2
def rdev(A, ids):
    out = {}
    for a in ARMS:
        if a == "FP16_history":
            continue
        rs = [A[a][i] for i in ids]
        n = sum(r["rdev_count"] for r in rs)
        if n == 0:
            out[a] = {"n_slots": 0}
            continue
        h = np.sum([r["rdev_hist"] for r in rs], axis=0).astype(float)
        cum = np.cumsum(h)
        k = int(np.argmax(cum >= 0.99 * cum[-1]))
        flagged_steps = [(r["index"], s) for r in rs for s in range(r["S"]) if r["rdev_n_gt5"][s] > 0]
        alt = sum(1 for r in rs for s in range(r["S"]) if r["rdev_max"][s] > FLAG)
        args = Counter(tuple(A[a][i]["rdev_arg"][s]) for i, s in flagged_steps)
        out[a] = {"n_slots": int(n), "mean": float(sum(r["rdev_sum"] for r in rs) / n),
                  "p99_at_most": float(EDGES[k]) if k < len(EDGES) else float("inf"),
                  "max": float(max(max(r["rdev_max"]) for r in rs)), "n_steps": int(sum(r["S"] for r in rs)),
                  "steps_with_a_slot_above_5pct": len(flagged_steps), "steps_rdev_max_gt5_crosscheck": alt,
                  "n_slots_above_5pct": int(sum(sum(r["rdev_n_gt5"]) for r in rs)),
                  "slots_dominating_those_steps": [{"slot": f"L{l} {p} h{hh}", "share": c / len(flagged_steps)} for (l, p, hh), c in args.most_common(5)]}
    return out


def keep_len(r):
    for s, n in enumerate(r["rdev_n_gt5"][:r["S"]]):
        if n > 0:
            return s + 1                                             # the write at step s+1 first affects step s+2: keep 1..s+1
    return r["S"]


def cut(r, k):
    o = dict(r)
    for key in ("js", "P_norm", "P_logit", "P_val", "P_anormK", "P_out"):
        o[key] = r[key][:k]
    o["S"] = k
    return o


def trunc(A, arms, ids):
    T, kept, dropped, fr = {a: {} for a in arms}, [], 0, []
    for i in ids:
        k = min(keep_len(A[a][i]) for a in arms)
        if k < 8:
            dropped += 1
            continue
        kept.append(i)
        fr.append(k / A[arms[0]][i]["S"])
        for a in arms:
            T[a][i] = cut(A[a][i], k)
    return T, kept, dropped, (float(np.median(fr)) if fr else None)


# ------------------------------------------------------------------ analysis
FAMILY = [("NAT-C1", "NAT", "C1"), ("HB QA-QO", "QA", "QO"), ("HD K*-V*", "NAT-K*", "NAT-V*"), ("Prov-K NAT-K-C1-K", "NAT-K", "C1-K"),
          ("Prov-V NAT-V-C1-V", "NAT-V", "C1-V"), ("HE INH-LOC", "INH", "LOC")]
DESCR = [("NAT-C2", "NAT", "C2"), ("C2-C1", "C2", "C1"), ("NAT-CV", "NAT", "CV"), ("NAT-K-NAT-V", "NAT-K", "NAT-V")]


def per_model(m, A, cal, ids, exposed, bt):
    mg, inf = cal["margin_log_js"], cal["dose_slope_b_hat"] >= 0.25
    s_out = not (0.5 <= cal["scale_factor_K"] <= 2 and 0.5 <= cal["scale_factor_V"] <= 2)
    R = {"n_problems": len(ids), "margin": mg, "informative": bool(inf), "HD_descriptive_rule": s_out}
    D, excl = D_vec(A["NAT"], ids)
    R["HA"] = {**paired(list(D.values()), bt, allow_equiv=False), "excluded_problems": excl}
    R["_HA_per_problem"] = D
    fam = {}
    for name, x, y in FAMILY:
        if name == "HD K*-V*" and s_out:
            continue
        fam[name] = paired(theta_vec(A, x, y, ids), bt, mg, inf)
    hc = hc_one(hc_steps(A["NAT"], A["C1"], ids), bt)
    R["HC"] = hc
    if hc["applicable"]:
        fam["HC shrinkage"] = {"estimate": hc["shrinkage"], "ci95": hc["shrinkage_ci95"], "p": hc["p"], "verdict": hc["band"]}
    hp = holm_adjust([v["p"] for v in fam.values()])
    for v, h in zip(fam.values(), hp):
        v["holm_p"] = float(h)
    R["family"] = fam
    if s_out:
        R["HD_descriptive"] = paired(theta_vec(A, "NAT-K*", "NAT-V*", ids), bt, mg, inf)
    hd = fam.get("HD K*-V*", R.get("HD_descriptive"))
    R["HD_reading"] = ("K more disruptive: supports the storage-level account" if hd["verdict"] == "difference" and hd["estimate"] > 0 else
                       "V more disruptive: supports the weight-projection account" if hd["verdict"] == "difference" else
                       "neither path dominates (equivalent)" if hd["verdict"] == "equivalent" else "inconclusive")
    he = fam["HE INH-LOC"]
    R["HE_reading"] = ("INH > LOC: consistent with prior work" if he["verdict"] == "difference" and he["estimate"] > 0 else
                       "SURPRISE: LOC >= INH" if he["verdict"] in ("difference", "equivalent") else "inconclusive")
    hb = fam["HB QA-QO"]
    R["HB_falsified"] = bool((hb["verdict"] == "difference" and hb["estimate"] < 0) or hb["verdict"] == "equivalent")
    desc = {}
    for name, x, y in DESCR:
        desc[name] = paired(theta_vec(A, x, y, ids), bt, mg, inf)
    m_ = lambda a, i: max(float(np.mean(A[a][i]["js"][:A[a][i]["S"]])), M_FLOOR)
    ratio = np.array([m_("NAT", i) / (m_("LOC", i) + m_("INH", i)) for i in ids])
    W = bt.counts(len(ratio)).astype(int)
    meds = [float(np.median(np.repeat(ratio, w))) for w in W]
    desc["nonadditivity"] = {"median": float(np.median(ratio)), "ci95_of_median": pct(meds, [2.5, 97.5])}
    for a in ("NAT", "LOC", "INH"):
        desc[f"dn2 {a} (K, V)"] = [float(np.mean([np.mean(A[a][i][f"dn2_{p}"]) for i in ids])) for p in "KV"]
    for k in ("P_val", "P_anormK", "P_out"):
        Dk, _ = D_vec(A["NAT"], ids, pred=k)
        desc[f"HA-type {k}"] = paired(list(Dk.values()), bt, allow_equiv=False)
    desc["HA pooled norm-matched arms"] = paired(list(D_pooled_arms(A, ids).values()), bt, allow_equiv=False)
    desc["rel norm K/V (NAT)"] = [float(np.mean([np.mean(A["NAT"][i][f"rel_{p}"]) for i in ids])) for p in "KV"]
    R["descriptive"] = desc
    sens = {}
    nonexp = [i for i in ids if i not in exposed]
    long_ = [i for i in ids if A["NAT"][i]["S"] >= 32]
    for name, x, y in FAMILY + DESCR:
        sens[name] = {"steps 2-8": float(theta_vec(A, x, y, ids, a=2, b=8).mean()),
                      "steps 9-64": float(theta_vec(A, x, y, [i for i in ids if A[x][i]["S"] >= 9], a=9).mean()),
                      "per-problem median": float(theta_vec(A, x, y, ids, med=True).mean()),
                      "excluding exposed": float(theta_vec(A, x, y, nonexp).mean()), "n excluding exposed": len(nonexp),
                      "S >= 32": float(theta_vec(A, x, y, long_).mean()) if long_ else None, "n S >= 32": len(long_)}
    R["sensitivity"] = sens
    R["_theta_NAT_C1"] = dict(zip(ids, theta_vec(A, "NAT", "C1", ids)))
    R["realized_norm_deviation"] = rdev(A, ids)
    return R


def sensitivity_rounding(A, cal, ids, bs):
    mg, inf = cal["margin_log_js"], cal["dose_slope_b_hat"] >= 0.25
    out = {}

    def put(name, arms, f):
        T, kept, dropped, fr = trunc(A, arms, ids)
        out[name] = {"n_kept": len(kept), "n_dropped": dropped, "median_retained_fraction": fr, "result": f(T, kept) if len(kept) >= 2 else None}
    put("HA (NAT, primary)", ("NAT",), lambda T, k: paired(list(D_vec(T["NAT"], k)[0].values()), bs, allow_equiv=False))
    for name, x, y in FAMILY + DESCR:
        put(name, (x, y), lambda T, k, x=x, y=y: paired(theta_vec(T, x, y, k), bs, mg, inf))
    put("HC shrinkage", ("NAT", "C1"), lambda T, k: hc_one(hc_steps(T["NAT"], T["C1"], k), bs))
    put("HA pooled norm-matched arms", tuple(SIX), lambda T, k: paired(list(D_pooled_arms(T, k).values()), bs, allow_equiv=False))
    T, kept, _, _ = trunc(A, ("NAT", "C1"), ids)
    return out, dict(zip(kept, theta_vec(T, "NAT", "C1", kept)))


def hf(T, cals, bt):
    sh = sorted(set.intersection(*[set(T[m]) for m in MODELS]))
    a, b, c, d = (np.array([T[m][i] for i in sh]) for m in ("qwen-base", "qwen-instruct", "llama-base", "llama-instruct"))
    mg = float(np.mean([cals[m]["margin_log_js"] for m in MODELS]))
    inf = all(cals[m]["dose_slope_b_hat"] >= 0.25 for m in MODELS)
    X = {"family (Llama - Qwen)": paired(0.5 * ((c + d) - (a + b)), bt, mg, inf),
         "tuning (instruct - base)": paired(0.5 * ((b + d) - (a + c)), bt, mg, inf),
         "interaction": paired(0.5 * ((a - b) - (c - d)), bt, mg, inf)}
    for v, h in zip(X.values(), holm_adjust([v["p"] for v in X.values()])):
        v["holm_p"] = float(h)
    sds = {m: float(np.std(v, ddof=1)) for m, v in zip(("qwen-base", "qwen-instruct", "llama-base", "llama-instruct"), (a, b, c, d))}
    return sh, mg, sds, X


def account(s):
    if s["verdict"] == "difference":
        return "supported" if s["estimate"] > 0 else "falsified"
    return "falsified" if s["verdict"] == "equivalent" else "inconclusive"


def run():
    data, sample = load()
    integ = check_integrity(data)
    cals = {m: json.loads((RUNS / m / "calibration.json").read_text()) for m in MODELS}
    exposed = set(sample["exposed_test_ids_in_sample"])
    bt = Boot(SEED)
    out = {"integrity_problems": integ, "per_model": {}}
    ids_m = {}
    for m in MODELS:
        A = data[m]
        ids = sorted(set.intersection(*[set(A[a]) for a in ARMS]))
        ids_m[m] = ids
        out["per_model"][m] = per_model(m, A, cals[m], ids, exposed, bt)
    hp = holm_adjust([out["per_model"][m]["HA"]["p"] for m in MODELS])
    for m, h in zip(MODELS, hp):
        H = out["per_model"][m]["HA"]
        H["holm_p"] = float(h)
        H["supported"] = bool(h < 0.05 and H["ci95"][0] > 0)
        H["falsified"] = bool(H["ci95"][1] < 0)
    shared = [i for i in sorted(set.intersection(*[set(out["per_model"][m]["_HA_per_problem"]) for m in MODELS]))]
    out["HA_pooled_secondary"] = paired([np.mean([out["per_model"][m]["_HA_per_problem"][i] for m in MODELS]) for i in shared], bt, allow_equiv=False)
    out["HC_cross"] = {}
    for k, A_ in enumerate(MODELS):
        for B_ in MODELS[k + 1:]:
            sh = sorted(set(ids_m[A_]) & set(ids_m[B_]))
            r = hc_pair(hc_steps(data[A_]["NAT"], data[A_]["C1"], sh), hc_steps(data[B_]["NAT"], data[B_]["C1"], sh), bt)
            r["role"] = "decision rule" if (A_, B_) == ("qwen-base", "llama-instruct") else "descriptive"
            out["HC_cross"][f"{A_} vs {B_}"] = r
    T = {m: out["per_model"][m]["_theta_NAT_C1"] for m in MODELS}
    sh, mg, sds, X = hf(T, cals, bt)
    guard = any(s > 1.25 * SD_ASSUMED for s in sds.values())
    out["HF"] = {"label": "descriptive (25% SD guard triggered)" if guard else "test across these four models (one model per cell)",
                 "sd_per_model": sds, "guard_limit": 1.25 * SD_ASSUMED, "n_shared": len(sh), "mean_margin": mg, "contrasts": X,
                 "readings": {"family account": account(X["family (Llama - Qwen)"]), "tuning account": account(X["tuning (instruct - base)"]),
                              "additivity": "falsified" if X["interaction"]["verdict"] == "difference" else "not falsified"}}
    # exploratory (amendment 2), own streams
    bs = Boot(SEED_SENS)
    Tt = {}
    for m in MODELS:
        s, Tt[m] = sensitivity_rounding(data[m], cals[m], ids_m[m], bs)
        out["per_model"][m]["sensitivity_rounding"] = s
    sh2, _, _, X2 = hf(Tt, cals, Boot(SEED_SENS_HF))
    out["HF_sensitivity_rounding"] = {"n_shared": len(sh2), **{k: {kk: vv for kk, vv in v.items() if kk != "holm_p"} for k, v in X2.items()}}
    for m in MODELS:
        out["per_model"][m]["theta_NAT_C1_problem_mean"] = float(np.mean(list(T[m].values())))
        out["per_model"][m]["_HA_per_problem"] = {str(k): v for k, v in out["per_model"][m]["_HA_per_problem"].items()}
        out["per_model"][m]["_theta_NAT_C1"] = {str(k): float(v) for k, v in out["per_model"][m]["_theta_NAT_C1"].items()}
    return out


# ------------------------------------------------------------------ comparison with the locked report
def compare(mine, rep):
    rows = []

    def det(path, a, b):
        if a is None and b is None:
            return
        ok = a is not None and b is not None and abs(a - b) <= 1e-9 * max(1.0, abs(b))
        rows.append({"what": path, "kind": "deterministic", "mine": a, "locked": b, "ok": bool(ok)})

    def ci(path, a, b):
        w = abs(b[1] - b[0])
        tol = max(0.05 * w, 1e-9)
        ok = all(abs(x - y) <= tol for x, y in zip(a, b))
        rows.append({"what": path, "kind": "CI", "mine": a, "locked": b, "tol": tol, "ok": bool(ok)})

    def pv(path, a, b):
        ok = abs(a - b) <= 0.01 + 0.10 * b
        rows.append({"what": path, "kind": "p", "mine": a, "locked": b, "ok": bool(ok)})

    def dec(path, a, b):
        rows.append({"what": path, "kind": "decision", "mine": a, "locked": b, "ok": a == b})

    def summ(path, s, r, holm=True):
        det(path + " estimate", s["estimate"], r["estimate"])
        if "n" in r:
            det(path + " n", s["n"], r["n"])
        ci(path + " ci95", s["ci95"], r["ci95"])
        if "ci90" in r:
            ci(path + " ci90", s["ci90"], r["ci90"])
        pv(path + " p", s["p"], r["p"])
        if holm and "holm_p" in r:
            pv(path + " holm_p", s["holm_p"], r["holm_p"])
            dec(path + " Holm-significant", s["holm_p"] < 0.05, r["holm_p"] < 0.05)
        dec(path + " verdict", s["verdict"], r["verdict"])

    for m in MODELS:
        S, L = mine["per_model"][m], rep["per_model"][m]
        summ(f"{m} HA", S["HA"], L["HA"])
        det(f"{m} HA excluded", S["HA"]["excluded_problems"], L["HA"]["excluded_problems"])
        dec(f"{m} HA supported", S["HA"]["supported"], L["HA"]["supported"])
        dec(f"{m} HA falsified", S["HA"]["falsified"], L["HA"]["falsified"])
        dec(f"{m} F_m membership", sorted(S["family"]), sorted(L["family"]))
        for k in set(S["family"]) & set(L["family"]):
            if k == "HC shrinkage":
                det(f"{m} {k} estimate", S["family"][k]["estimate"], L["family"][k]["estimate"])
                ci(f"{m} {k} ci95", S["family"][k]["ci95"], L["family"][k]["ci95"])
                pv(f"{m} {k} p", S["family"][k]["p"], L["family"][k]["p"])
                pv(f"{m} {k} holm_p", S["family"][k]["holm_p"], L["family"][k]["holm_p"])
                dec(f"{m} {k} Holm-significant", S["family"][k]["holm_p"] < 0.05, L["family"][k]["holm_p"] < 0.05)
                dec(f"{m} {k} band", S["family"][k]["verdict"], L["family"][k]["verdict"])
            else:
                summ(f"{m} {k}", S["family"][k], L["family"][k])
        h, l = S["HC"], L["HC"]
        for k in ("beta0", "beta1", "shrinkage", "fraction"):
            det(f"{m} HC {k}", h[k], l[k])
        for k in ("beta0_ci95", "shrinkage_ci95", "fraction_ci95"):
            ci(f"{m} HC {k}", h[k], l[k])
        pv(f"{m} HC p", h["p"], l["p"])
        dec(f"{m} HC applicable", h["applicable"], l["applicable"])
        dec(f"{m} HC band", h["band"], l["band"])
        dec(f"{m} HD descriptive rule", S["HD_descriptive_rule"], "HD_descriptive" in L)
        dec(f"{m} HD reading", S["HD_reading"], L["HD_reading"])
        dec(f"{m} HE reading", S["HE_reading"], L["HE_reading"])
        dec(f"{m} HB falsified", S["HB_falsified"], L["HB_falsified"])
        Ld = L["descriptive"]
        for k in ("NAT-C2", "C2-C1", "NAT-CV", "NAT-K-NAT-V"):
            summ(f"{m} descriptive {k}", S["descriptive"][k], Ld[k], holm=False)
        det(f"{m} nonadditivity median", S["descriptive"]["nonadditivity"]["median"], Ld["nonadditivity m_NAT/(m_LOC+m_INH)"]["median"])
        ci(f"{m} nonadditivity ci95", S["descriptive"]["nonadditivity"]["ci95_of_median"], Ld["nonadditivity m_NAT/(m_LOC+m_INH)"]["ci95_of_median"])
        for a in ("NAT", "LOC", "INH"):
            for j in (0, 1):
                det(f"{m} dn2 {a} {'KV'[j]}", S["descriptive"][f"dn2 {a} (K, V)"][j], Ld[f"mean ||Delta||^2 {a} (K, V)"][j])
        for k in ("P_val", "P_anormK", "P_out"):
            summ(f"{m} HA-type {k}", S["descriptive"][f"HA-type {k}"], Ld[f"HA-type {k} vs P_norm (NAT)"], holm=False)
        summ(f"{m} HA pooled arms", S["descriptive"]["HA pooled norm-matched arms"], Ld["HA pooled norm-matched arms (secondary)"], holm=False)
        for j in (0, 1):
            det(f"{m} rel norm {'KV'[j]}", S["descriptive"]["rel norm K/V (NAT)"][j], Ld["rel norm K / V (NAT, mean)"][j])
        for lk, mk in (("NAT-C1", "NAT-C1"), ("HB", "HB QA-QO"), ("HD", "HD K*-V*"), ("HE", "HE INH-LOC")):
            for k in ("steps 2-8", "steps 9-64", "per-problem median", "excluding exposed", "S >= 32"):
                det(f"{m} sensitivity {lk} {k}", S["sensitivity"][mk][k], L["sensitivity"][lk][k])
        for a, v in S["realized_norm_deviation"].items():
            lv = L["realized_norm_deviation"].get(a, {"n_slots": None})
            det(f"{m} rdev {a} n_slots", v["n_slots"], lv["n_slots"])
            if v["n_slots"]:
                for k in ("mean", "p99_at_most", "max", "n_steps", "steps_with_a_slot_above_5pct"):
                    det(f"{m} rdev {a} {k}", v[k], lv[k])
                dec(f"{m} rdev {a} dominating slots", [x["slot"] for x in v["slots_dominating_those_steps"]],
                    [x["slot"] for x in lv["slots_dominating_those_steps"]])
        Ls = L["sensitivity_rounding"]["contrasts"]
        name_map = {"HA pooled norm-matched arms": "HA pooled norm-matched arms (secondary)"}
        for k, v in S["sensitivity_rounding"].items():
            lv = Ls.get(name_map.get(k, k))
            if lv is None:
                continue
            det(f"{m} trunc-sens {k} kept", v["n_kept"], lv["n_kept"])
            det(f"{m} trunc-sens {k} dropped", v["n_dropped"], lv["n_dropped"])
            det(f"{m} trunc-sens {k} median retained", v["median_retained_fraction"], lv["median_retained_fraction"])
            r, lr = v["result"], lv["result"]
            if r and lr:
                if k == "HC shrinkage":
                    det(f"{m} trunc-sens {k} shrinkage", r["shrinkage"], lr["shrinkage"])
                    ci(f"{m} trunc-sens {k} ci95", r["shrinkage_ci95"], lr["shrinkage_ci95"])
                    dec(f"{m} trunc-sens {k} band", r["band"], lr["band"])
                else:
                    summ(f"{m} trunc-sens {k}", r, lr, holm=False)
    summ("HA pooled (secondary)", mine["HA_pooled_secondary"], rep["HA_pooled_secondary"], holm=False)
    for k, v in mine["HC_cross"].items():
        lv = rep["HC_cross"][k]
        for q in ("g0", "g1", "F"):
            det(f"HC cross {k} {q}", v[q], lv[q])
        for q in ("g0_ci95", "g1_ci95", "F_ci95"):
            ci(f"HC cross {k} {q}", v[q], lv[q])
        dec(f"HC cross {k} precondition", v["precondition_met"], lv["precondition_met"])
        dec(f"HC cross {k} band", v["band"], lv["band"])
        dec(f"HC cross {k} role", v["role"], lv["role"])
    H, LH = mine["HF"], rep["HF"]
    dec("HF label (25% guard)", H["label"], LH["label"])
    for m in MODELS:
        det(f"HF SD {m}", H["sd_per_model"][m], LH["sd_per_model"][m])
    det("HF n_shared", H["n_shared"], LH["n_shared"])
    for k in H["contrasts"]:
        summ(f"HF {k}", H["contrasts"][k], LH["contrasts"][k])
    for k in ("family account", "tuning account", "additivity"):
        dec(f"HF reading {k}", H["readings"][k], LH["readings"][k])
    LS = rep.get("HF_sensitivity_rounding", {})
    det("HF trunc-sens n_shared", mine["HF_sensitivity_rounding"]["n_shared"], LS.get("n_shared"))
    for k in ("family (Llama - Qwen)", "tuning (instruct - base)", "interaction"):
        if k in LS:
            summ(f"HF trunc-sens {k}", mine["HF_sensitivity_rounding"][k], LS[k], holm=False)
    # per-problem vectors
    for m in MODELS:
        a = mine["per_model"][m]["_theta_NAT_C1"]
        b = rep["per_model"][m]["theta_NAT_C1_per_problem"]
        det(f"{m} per-problem theta(NAT,C1) max |diff|", max(abs(a[k] - b[k]) for k in b), 0.0)
        a = mine["per_model"][m]["_HA_per_problem"]
        b = rep["per_model"][m]["HA"]["per_problem"]
        dec(f"{m} per-problem D index set", sorted(a), sorted(b))
        det(f"{m} per-problem D max |diff|", max(abs(a[k] - b[k]) for k in b), 0.0)
    return rows


def main():
    mine = run()
    (HERE / "recompute_v3.json").write_text(json.dumps(mine, indent=1, default=float))
    rep = json.loads((RUNS / "v3_report.json").read_text())
    rows = compare(mine, rep)
    bad = [r for r in rows if not r["ok"]]
    summary = {"n_compared": len(rows), "n_disagreements": len(bad), "by_kind": dict(Counter(r["kind"] for r in rows)),
               "integrity_problems": mine["integrity_problems"], "disagreements": bad}
    (HERE / "recompute_vs_locked.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1, default=float))
    print(json.dumps({k: v for k, v in summary.items() if k != "disagreements"}, default=float))
    for r in bad:
        print("DISAGREEMENT", json.dumps(r, default=float))


if __name__ == "__main__":
    main()
