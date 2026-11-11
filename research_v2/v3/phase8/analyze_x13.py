"""X13 analysis (v3/phase8/FOLLOWUP_X13.md section 8). EXPLORATORY. CPU only; reads v3_runs_x13 (X13 records), v3_runs (locked records)
and v3_runs_x9 (X9 slot energies for the X9a comparison), all read only.

  venv/Scripts/python.exe research_v2/v3/phase8/analyze_x13.py      -> v3/phase8/x13_results.json, X13_RESULTS.md

Only models with x13_complete.json and no REPRODUCTION_MISMATCH.json are analysed; the others are listed with their status.
(a), (b), (d) are exploratory tests with the pre-stated rules; (c) is descriptive.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

P8 = Path(__file__).resolve().parent
RV = P8.parents[1]
sys.path.insert(0, str(RV))
import analyze_v3 as L                          # noqa: E402  (locked primitives, imported unchanged)

OUT = RV / "v3_runs_x13"
LOCKED_RUNS = RV / "v3_runs"
X9_RUNS = RV / "v3_runs_x9"
REPORT = RV / "v3" / "phase5" / "v3_report_locked.json"
ORDER = ["qwen-base", "qwen-instruct", "llama-base", "llama-instruct"]
SEED = 20261209                                 # D7: bootstrap / sign-flip stream
NULL_SEED = [20261209, 13]                      # D7: isotropic null of (a)
N_NULL = 200
LN4, LN2 = math.log(4.0), math.log(2.0)
ISO8 = 8 / 64
LABEL = "EXPLORATORY (FOLLOWUP_X13.md): (a), (b), (d) exploratory tests; (c) descriptive"
PATHS = ("K", "V")
CEN = {"K": "NAT-K-cen", "V": "NAT-V-cen"}
LOCKED_NAT = {"K": "NAT-K", "V": "NAT-V"}
LOCKED_C1 = {"K": "C1-K", "V": "C1-V"}
SIGN = {"K": 1.0, "V": -1.0}                    # predicted sign of theta_cen
D_CONTRASTS = {"NAT-C1": ("C1", "C1-nc", "NAT-C1"), "NAT-K-C1-K": ("C1-K", "C1-K-nc", "Prov-K NAT-K-C1-K")}


def family(m):
    return m.split("-")[0]


# ----------------------------------------------------------------------------- loading
def load_slots(m, tag, ids):
    out = {}
    for i in ids:
        with np.load(OUT / m / "slots" / f"{tag}_{i}.npz") as z:
            out[i] = {k: z[k] for k in z.files}
    return out


def ell(recs, ids):
    return np.array([L.logm(recs[i]) for i in ids])


# ----------------------------------------------------------------------------- (a) common mode
def cm_share(norms, S1):
    """C = sum_slots ||S1 / n||^2 / sum_slots (S2 / n), S2 = sum_j ||delta_j||^2. norms [n, L, n_kv], S1 [L, n_kv, hd]."""
    n = norms.shape[0]
    S2 = (norms ** 2).sum(0)
    return float(((S1 / n) ** 2).sum() / (S2 / n).sum())


def null_lognc(norms, rng, draws=N_NULL, hd=64):
    """Mean over draws of ln(n C) when every entry keeps its norm and gets an independent uniform direction in R^hd."""
    n = norms.shape[0]
    out = []
    for _ in range(draws):
        g = rng.standard_normal(norms.shape + (hd,))
        u = g / np.linalg.norm(g, axis=-1, keepdims=True)
        S1 = (u * norms[..., None]).sum(0)
        out.append(math.log(n * cm_share(norms, S1)))
    return float(np.mean(out))


def cm_stat(slots, path, rng_null, null_cache=None, key=None):
    """x_i = ln(n_i C_i) - nu_i over problems (dict idx -> x). null_cache: idx -> nu (reused when the norms are the same)."""
    xs, nus = {}, {}
    for i, s in slots.items():
        norms, S1 = s[f"norms{path}"], s[f"S1{path}"]
        n = norms.shape[0]
        nu = null_lognc(norms, rng_null)
        nus[i] = nu
        xs[i] = math.log(n * cm_share(norms, S1)) - nu
    return xs, nus


def verdict_a(s, holm_p):
    if s["ci95"][0] > LN4 and holm_p < 0.05:
        return "supported"
    if s["ci95"][1] < LN2:
        return "contradicted"
    return "inconclusive"


def cm_per_layer(slots, path):
    """Pooled over problems: per layer, sum ||S1/n||^2 / sum (S2/n), and the energy share of the mean component sum n||S1/n||^2 / sum S2."""
    num = den = None
    for s in slots.values():
        norms, S1 = s[f"norms{path}"], s[f"S1{path}"]
        n = norms.shape[0]
        a = ((S1 / n) ** 2).sum(axis=(1, 2))
        b = ((norms ** 2).sum(0) / n).sum(axis=1)
        num, den = (a, b) if num is None else (num + a, den + b)
    return [float(x) for x in num / den]


# ----------------------------------------------------------------------------- (b) centred arms
def gap_closed(l_cen, l_nat, l_c1, rng):
    """G = mean(l_cen - l_nat) / mean(l_c1 - l_nat); problem-bootstrap percentile CI of the ratio."""
    num, den = np.asarray(l_cen) - np.asarray(l_nat), np.asarray(l_c1) - np.asarray(l_nat)
    G = float(num.mean() / den.mean())
    bi = L.boot_idx(len(num), rng)
    b = num[bi].mean(1) / den[bi].mean(1)
    return G, [float(x) for x in np.nanpercentile(b, [2.5, 97.5])]


def verdict_direction(s, holm_p, sign):
    lo, hi = s["ci95"]
    if sign > 0:
        if s["estimate"] > 0 and lo > 0 and holm_p < 0.05:
            return "supported"
        return "contradicted" if hi < 0 else "inconclusive"
    if s["estimate"] < 0 and hi < 0 and holm_p < 0.05:
        return "supported"
    return "contradicted" if lo > 0 else "inconclusive"


def size_band(G, ci):
    if G >= 0.5 and ci[0] > 0.2:
        band = "most of the gap"
    elif ci[1] < 0.25:
        band = "small part"
    else:
        band = "partial"
    return band + ("; overshoot" if ci[0] > 1 else "")


def reading(per_model, path):
    """Cross-model reading per path (section 8b)."""
    rows = [r["b"][path] for r in per_model.values() if path in r.get("b", {})]
    counted = [x for x in rows if x["manipulation_ok"]]
    sup = [x for x in counted if x["direction"] == "supported"]
    if sum(x["direction"] == "contradicted" for x in counted) >= 2:
        return "contradicted"
    if len(sup) >= 3:
        if sum(x.get("band", "").startswith("most of the gap") for x in sup) >= 3:
            return "supported; accounts for most of the provenance gap"
        return "supported"
    return "mixed"


def truncated_theta(cen, nat, ids, rng):
    """theta_cen on the steps before the first centred-arm write with a slot above 5 % (locked first_unaffected), both arms on the same steps;
    problems with fewer than 8 such steps dropped. Descriptive."""
    d = []
    for i in ids:
        k = L.first_unaffected(cen[i])
        if k >= 8:
            d.append(L.logm(cen[i], hi=k) - L.logm(nat[i], hi=k))
    return {"n_kept": len(d), **({k: v for k, v in L.summarize(np.array(d), rng).items() if k in ("estimate", "ci95")} if d else {})}


# ----------------------------------------------------------------------------- (c) descriptive
def sp_series(recs, key):
    out = {}
    for i, r in recs.items():
        if r["S"] < L.MIN_STEPS_HA:
            continue
        v = L.spearman(r[key][1:], r["js"][1:])
        if v == v:
            out[i] = v
    return out


def ci_mean(v, rng):
    v = np.asarray(v, float)
    if len(v) == 0:
        return None
    b = v[L.boot_idx(len(v), rng)].mean(1)
    return {"mean": float(v.mean()), "ci95": [float(x) for x in np.percentile(b, [2.5, 97.5])], "n": int(len(v))}


def x9_A(m, ids):
    """X9a's ln(phi_i / 0.125) on the same problems, from the X9 NAT slot energies (pK / eK)."""
    out = {}
    for i in ids:
        p = X9_RUNS / m / "slots" / f"NAT_{i}.npz"
        if p.exists():
            with np.load(p) as z:
                out[i] = math.log(float(z["pK"].sum() / z["eK"].sum()) / ISO8)
    return out


# ----------------------------------------------------------------------------- (d)
def half_width(ci):
    return (ci[1] - ci[0]) / 2.0


def verdict_d(ci, h):
    if -h < ci[0] and ci[1] < h:
        return "supported"
    if ci[0] > h or ci[1] < -h:
        return "contradicted"
    return "inconclusive (does not dismiss the rounding confound)"


def sdev_summary(rec, prefix=""):
    n = sum(rec[f"{prefix}sdev_n"])
    return {"mean": (sum(rec[f"{prefix}sdev_sum"]) / n) if n else 0.0, "share_above_5pct": (sum(rec[f"{prefix}sdev_pos5"]) / n) if n else 0.0,
            "share_below_minus_5pct": (sum(rec[f"{prefix}sdev_neg5"]) / n) if n else 0.0, "min": min(rec[f"{prefix}sdev_min"]),
            "max": max(rec[f"{prefix}sdev_max"]), "n_slot_steps": n}


# ----------------------------------------------------------------------------- per model
def status(m):
    if (OUT / m / "REPRODUCTION_MISMATCH.json").exists():
        return "reproduction mismatch or integrity error (not analysed)"
    if not (OUT / m / "x13_complete.json").exists():
        return "not complete (not analysed)"
    return "complete"


def analyse_model(m, rng, rng_null, rep_locked):
    meta = json.loads((OUT / m / "x13_complete.json").read_text())
    nat = L.load_jsonl(OUT / m / "NAT.jsonl")
    ids = sorted(nat)
    res = {"n": len(ids), "arms": meta["arms"], "a": {}, "b": {}}
    s_nat = load_slots(m, "NAT", ids)
    locked = {t: L.load_jsonl(LOCKED_RUNS / m / "tierA" / f"{t}.jsonl") for t in ("NAT-K", "NAT-V", "C1-K", "C1-V", "C1", "NAT")}
    for P in PATHS:
        xs, _ = cm_stat(s_nat, P, rng_null)
        res["a"][P] = {"test": L.summarize(np.array([xs[i] for i in ids]), rng), "per_layer_share": cm_per_layer(s_nat, P),
                       "mean_ln_nC_minus_null": float(np.mean([xs[i] for i in ids]))}
        cen = L.load_jsonl(OUT / m / f"{CEN[P]}.jsonl")
        s_cen = load_slots(m, CEN[P], ids)
        xc, _ = cm_stat(s_cen, P, rng_null)
        man = L.summarize(np.array([xc[i] for i in ids]), rng)
        l_cen, l_nat, l_c1 = ell(cen, ids), ell(locked[LOCKED_NAT[P]], ids), ell(locked[LOCKED_C1[P]], ids)
        th = L.summarize(l_cen - l_nat, rng)
        G, Gci = gap_closed(l_cen, l_nat, l_c1, rng)
        b = {"theta_cen": th, "G": G, "G_ci95": Gci, "manipulation": man, "manipulation_ok": bool(man["ci95"][1] < LN2),
             "locked_gap_on_subset": float((l_c1 - l_nat).mean()), "truncated_theta_cen": truncated_theta(cen, locked[LOCKED_NAT[P]], ids, rng),
             "degenerate_slots": int(sum(sum(cen[i].get("degenerate", [])) for i in ids)),
             "signed_deviation": sdev_summary({k: [x for i in ids for x in cen[i][k]] for k in
                                               ("sdev_sum", "sdev_n", "sdev_pos5", "sdev_neg5", "sdev_min", "sdev_max")})}
        if P == "K" and family(m) == "qwen":
            c1knc = L.load_jsonl(OUT / m / "C1-K-nc.jsonl")
            Gn, Gnci = gap_closed(l_cen, l_nat, ell(c1knc, ids), rng)
            b["G_with_C1-K-nc"] = {"G": Gn, "ci95": Gnci}
        res["b"][P] = b
    # (c)
    sc, sl = sp_series(nat, "P_logit_c"), sp_series(nat, "P_logit")
    both = sorted(set(sc) & set(sl))
    lnR = {i: math.log(float(s_nat[i]["RN"].sum() / s_nat[i]["RD"].sum())) for i in ids}
    RNl = np.stack([s_nat[i]["RN"].sum(1) for i in ids])
    RDl = np.stack([s_nat[i]["RD"].sum(1) for i in ids])
    A = x9_A(m, ids)
    th_k = {i: L.logm(locked["NAT-K"][i]) - L.logm(locked["C1-K"][i]) for i in ids}
    common = [i for i in ids if i in A]
    res["c"] = {"Sp_P_logit_c": ci_mean([sc[i] for i in both], rng), "Sp_P_logit": ci_mean([sl[i] for i in both], rng),
                "D_c": ci_mean([sc[i] - sl[i] for i in both], rng), "mean_ln_R": ci_mean([lnR[i] for i in ids], rng),
                "ln_R_per_layer": [float(x) for x in np.log(RNl.sum(0) / RDl.sum(0))],
                "ln_R_no_layer0": ci_mean([math.log(float(s_nat[i]["RN"][1:].sum() / s_nat[i]["RD"][1:].sum())) for i in ids], rng),
                "X9a_A_same_problems": ci_mean([A[i] for i in common], rng),
                "spearman_lnR_vs_locked_theta_NATK_C1K": L.spearman(np.array([lnR[i] for i in ids]), np.array([th_k[i] for i in ids]))}
    # (d)
    if family(m) == "qwen":
        res["d"] = {}
        for name, (larm, ncarm, fam_key) in D_CONTRASTS.items():
            nc = L.load_jsonl(OUT / m / f"{ncarm}.jsonl")
            l_lock, l_nc = ell(locked[larm], ids), ell(nc, ids)
            dth = L.summarize(l_lock - l_nc, rng)
            h = half_width(rep_locked["per_model"][m]["family"][fam_key]["ci95"])
            l_natpath = ell(locked["NAT"], ids) if larm == "C1" else ell(locked["NAT-K"], ids)
            res["d"][name] = {"delta_theta": dth, "h": h, "verdict": verdict_d(dth["ci95"], h),
                              "theta_sub_nc": float((l_natpath - l_nc).mean()), "theta_sub_locked": float((l_natpath - l_lock).mean()),
                              "theta_full_locked": rep_locked["per_model"][m]["family"][fam_key]["estimate"],
                              "signed_after": sdev_summary({k: [x for i in ids for x in nc[i][k]] for k in
                                                            ("sdev_sum", "sdev_n", "sdev_pos5", "sdev_neg5", "sdev_min", "sdev_max")}),
                              "signed_nearest": sdev_summary({k: [x for i in ids for x in nc[i][k]] for k in
                                                              ("nn_sdev_sum", "nn_sdev_n", "nn_sdev_pos5", "nn_sdev_neg5", "nn_sdev_min", "nn_sdev_max")},
                                                             "nn_"),
                              "lambda_range": [min(x for i in ids for x in nc[i]["lam_min"]), max(x for i in ids for x in nc[i]["lam_max"])],
                              "moves": int(sum(sum(nc[i]["moves"]) for i in ids)), "cos_min": min(x for i in ids for x in nc[i]["cos_min"]),
                              "steps_gt5_nc": int(sum(sum(1 for x in nc[i]["rdev_n_gt5"] if x > 0) for i in ids)),
                              "steps_gt5_locked": int(sum(sum(1 for x in locked[larm][i]["rdev_n_gt5"] if x > 0) for i in ids))}
    return res


def analyse():
    rng = np.random.default_rng(SEED)
    rng_null = np.random.default_rng(NULL_SEED)
    rep_locked = json.loads(REPORT.read_text())
    out = {"label": LABEL, "status": {m: status(m) for m in ORDER}, "per_model": {}}
    for m in ORDER:
        if out["status"][m] == "complete":
            out["per_model"][m] = analyse_model(m, rng, rng_null, rep_locked)
    ms = list(out["per_model"])
    for P in PATHS:
        if not ms:
            continue
        for m, h in zip(ms, L.holm([out["per_model"][m]["a"][P]["test"]["p"] for m in ms])):
            a = out["per_model"][m]["a"][P]
            a["holm_p"], a["verdict"] = float(h), verdict_a(a["test"], h)
        for m, h in zip(ms, L.holm([out["per_model"][m]["b"][P]["theta_cen"]["p"] for m in ms])):
            b = out["per_model"][m]["b"][P]
            b["holm_p"] = float(h)
            b["direction"] = verdict_direction(b["theta_cen"], h, SIGN[P])
            if b["direction"] == "supported":
                b["band"] = size_band(b["G"], b["G_ci95"])
            if not b["manipulation_ok"]:
                b["note"] = "manipulation check failed: reported descriptively only"
        out[f"reading_b_{P}"] = reading(out["per_model"], P)
    return out


def _ci(c):
    return f"[{c[0]:+.3f}, {c[1]:+.3f}]"


def to_md(o):
    Lm = ["# X13 results (EXPLORATORY; generated by analyze_x13.py)", "", f"Status: {o['status']}", ""]
    for P in PATHS:
        if f"reading_b_{P}" in o:
            Lm.append(f"- (b) reading across models, path {P}: **{o[f'reading_b_{P}']}**")
    Lm.append("")
    for m, r in o["per_model"].items():
        Lm += [f"## {m} (n = {r['n']})", ""]
        for P in PATHS:
            a, b = r["a"][P], r["b"][P]
            Lm.append(f"- (a) {P}: L {a['test']['estimate']:+.3f} [{a['test']['ci95'][0]:+.3f}, {a['test']['ci95'][1]:+.3f}], "
                      f"Holm p {a['holm_p']:.4f}: **{a['verdict']}**")
            Lm.append(f"- (b) {P}: theta_cen {b['theta_cen']['estimate']:+.3f} {_ci(b['theta_cen']['ci95'])}, Holm p {b['holm_p']:.4f}: "
                      f"**{b['direction']}**{'; ' + b['band'] if 'band' in b else ''}; G {b['G']:+.2f} {_ci(b['G_ci95'])}; manipulation L "
                      f"{b['manipulation']['estimate']:+.3f} (ok: {b['manipulation_ok']})")
        c = r["c"]
        Lm.append(f"- (c) Sp(P_logit^c) {c['Sp_P_logit_c']['mean']:+.3f}, Sp(P_logit) {c['Sp_P_logit']['mean']:+.3f}, D_c {c['D_c']['mean']:+.3f} "
                  f"{_ci(c['D_c']['ci95'])}; mean ln R {c['mean_ln_R']['mean']:+.3f} vs X9a A "
                  f"{c['X9a_A_same_problems']['mean'] if c['X9a_A_same_problems'] else float('nan'):+.3f}")
        for name, d in r.get("d", {}).items():
            Lm.append(f"- (d) {name}: delta theta {d['delta_theta']['estimate']:+.3f} {_ci(d['delta_theta']['ci95'])}, h {d['h']:.3f}: "
                      f"**{d['verdict']}**")
        Lm.append("")
    return "\n".join(Lm) + "\n"


def main():
    o = analyse()
    (P8 / "x13_results.json").write_text(json.dumps(o, indent=1, default=float))
    md = to_md(o)
    (P8 / "X13_RESULTS.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
