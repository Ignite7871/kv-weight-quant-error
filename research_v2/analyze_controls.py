"""E1 analysis (E1_SPEC.md sections 4-7). CPU only; reads per-problem JSONL records written by run_controls.py; never modifies them.

  venv/Scripts/python.exe research_v2/analyze_controls.py calibrate [--dir research_v2/e1_runs]     -> e1_runs/e1_calibration.json
  venv/Scripts/python.exe research_v2/analyze_controls.py analyze   [--dir research_v2/e1_runs]     -> e1_runs/e1_report.json / .md

Unit of analysis: the problem. Steps are averaged within a problem before any inference (no step-level pseudo-replication); all intervals resample
problems; paired contrasts use the same problems in every arm. Primary and secondary estimands are those of the spec; everything else is labelled
sensitivity or descriptive.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
B_BOOT, B_PERM = 10_000, 10_000
SEED = 20261101
FLOOR = 1e-9                     # floor on a per-problem mean JS before taking logs (float64 JS is exactly 0 only for identical distributions)
TAU = 64
DOSES = (0.5, 1.0, 2.0)
ALPHA_GRID = (0.5, 1.0, 2.0, 4.0)
C_LO, C_HI = 0.10, 0.40
MIN_SLOPE = 0.25                 # below this the dose-response is too flat for the matched size to be informative
EPS_STEPS_ALPHA_STAR = (2, 16)   # steps used to match effect size for C1*


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


def boot_means(d, rng, B=B_BOOT):
    d = np.asarray(d, float)
    idx = rng.integers(0, len(d), size=(B, len(d)))
    return d[idx].mean(axis=1)


def signflip_p(d, rng, B=B_PERM):
    d = np.asarray(d, float)
    n = len(d)
    obs = abs(d.mean())
    signs = rng.choice([-1.0, 1.0], size=(B, n))
    perm = np.abs((signs * d).mean(axis=1))
    return (1 + int((perm >= obs - 1e-15).sum())) / (B + 1)


def summarize(d, rng, margin=None, B=B_BOOT):
    """Paired differences d_i -> estimate, 95% percentile CI, sign-flip p, and (if a margin is given) the 90% CI and TOST-style equivalence verdict."""
    d = np.asarray(d, float)
    bm = boot_means(d, rng, B)
    out = {"n": int(len(d)), "estimate": float(d.mean()), "ci95": [float(x) for x in np.percentile(bm, [2.5, 97.5])],
           "ci90": [float(x) for x in np.percentile(bm, [5, 95])], "p_signflip": signflip_p(d, rng, min(B, B_PERM))}
    if margin is not None:
        out["margin"] = float(margin)
        out["equivalent_within_margin"] = bool(out["ci90"][0] > -margin and out["ci90"][1] < margin)
    return out


def first_div(a, b):
    n = min(len(a), len(b))
    for k in range(n):
        if a[k] != b[k]:
            return k
    return None if len(a) == len(b) else n


def restricted_first_div(ids, ref, tau=TAU):
    """min(D, tau); problems identical to the reference through the shared generation get tau."""
    D = first_div(ids, ref)
    return float(tau if D is None else min(D, tau))


def per_problem_mean_js(rec, lo=1, hi=None, stat="mean"):
    js = np.asarray(rec["js"], float)
    seg = js[lo - 1:(hi if hi else len(js))]
    v = float(np.median(seg) if stat == "median" else seg.mean())
    return max(v, FLOOR)


def per_problem_flip_rate(rec):
    return float(np.mean(rec["flip"]))


def dose_slope_one(logm, logalpha):
    x = np.asarray(logalpha, float)
    y = np.asarray(logm, float)
    x = x - x.mean()
    return float((x * (y - y.mean())).sum() / (x * x).sum())


# ----------------------------------------------------------------------------- loading
def load_jsonl(path: Path):
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["index"]] = r
    return out


def tier_a_arms(d: Path):
    names = {"NAT": "NAT", "C1_a1": "C1_a1.0", "C2_a1": "C2_a1.0", "C1_a0.5": "C1_a0.5", "C1_a2": "C1_a2.0", "C1star": "C1star"}
    return {k: load_jsonl(d / "tierA" / f"{v}.jsonl") for k, v in names.items()}


# ----------------------------------------------------------------------------- calibration (train problems only; no test data)
def calibrate(calib: dict) -> dict:
    """calib: arm-name -> {index: record} for the calibration problems; arms NAT and C1_a{0.5,1,2,4}. Returns the frozen constants."""
    arms = {a: calib[a] for a in calib}
    idx = sorted(set.intersection(*[set(v) for v in arms.values()]))
    lo, hi = EPS_STEPS_ALPHA_STAR

    def mean_over_problems(arm, lo_=lo, hi_=hi):
        return float(np.mean([per_problem_mean_js(arms[arm][i], lo_, hi_) for i in idx]))
    m_nat = mean_over_problems("NAT")
    grid = [(a, mean_over_problems(f"C1_a{a:g}")) for a in ALPHA_GRID]
    la, lm = np.log([g[0] for g in grid]), np.log([g[1] for g in grid])
    flag = None
    target = math.log(m_nat)
    if target <= lm.min():
        alpha_star, flag = float(ALPHA_GRID[0]), "NAT effect below the smallest C1 dose: alpha* clamped to the grid minimum"
    elif target >= lm.max():
        alpha_star, flag = float(ALPHA_GRID[-1]), "NAT effect above the largest C1 dose: alpha* clamped to the grid maximum"
    else:
        alpha_star = float(math.exp(np.interp(target, lm, la)))               # lm increasing in la for a monotone dose response
    # dose-response slope on log-JS, pooled over problems, doses 0.5/1/2 over the full 1..S window
    slopes = []
    for i in idx:
        y = [math.log(per_problem_mean_js(arms[f"C1_a{a:g}"][i])) for a in DOSES]
        slopes.append(dose_slope_one(y, np.log(DOSES)))
    b_hat = float(np.mean(slopes))
    # natural step-to-step relative variability of the NF4 perturbation size (NAT arm, per problem, then median over problems)
    cvs = []
    for i in idx:
        t = np.asarray(arms["NAT"][i]["target_norm_mean"], float)
        if len(t) > 3 and t.mean() > 0:
            cvs.append(t.std() / t.mean())
    c_raw = float(np.median(cvs))
    c = float(min(max(c_raw, C_LO), C_HI))
    informative = bool(b_hat >= MIN_SLOPE)
    margin_a = abs(b_hat) * math.log(1 + c) if informative else None
    return {"n_calibration_problems": len(idx), "alpha_star": alpha_star, "alpha_star_flag": flag, "m_NAT_steps_2_16": m_nat,
            "m_C1_by_alpha_steps_2_16": {f"{a:g}": m for a, m in grid}, "dose_slope_b_hat": b_hat,
            "dose_slope_problem_sd": float(np.std(slopes)), "dose_response_informative": informative,
            "cv_raw_median": c_raw, "c_used": c, "margin_tierA_log_js": margin_a,
            "margin_rule": "margin_A = |b_hat| * ln(1 + c); c = median over problems of the step-to-step CV of the mean NF4 perturbation norm, clamped to [0.10, 0.40]; "
                           "equivalence is claimable only if b_hat >= 0.25",
            "tierB_relative_margin_c": c}


# ----------------------------------------------------------------------------- confirmatory analysis
def analyze(tierA: dict, tierB: dict, calib: dict, exposed: set | None = None, B: int = B_BOOT) -> dict:
    rng = np.random.default_rng(SEED)
    rep = {"seed": SEED, "B": B, "calibration": calib, "tierA": {}, "tierB": {}, "holm_family": {}, "sensitivity": {}}
    margin_a = calib.get("margin_tierA_log_js")
    # ---------------- Tier A
    idxA = sorted(set.intersection(*[set(tierA[a]) for a in ("NAT", "C1_a1", "C2_a1")])) if tierA.get("NAT") else []
    if idxA:
        m = lambda arm, **kw: np.array([per_problem_mean_js(tierA[arm][i], **kw) for i in idxA])          # noqa: E731
        fl = lambda arm: np.array([per_problem_flip_rate(tierA[arm][i]) for i in idxA])                   # noqa: E731
        lg = np.log
        prim = summarize(lg(m("NAT")) - lg(m("C1_a1")), rng, margin_a, B)
        prim["ratio_NAT_over_C1"] = float(math.exp(prim["estimate"]))
        verdict = ("difference" if (prim["ci95"][0] > 0 or prim["ci95"][1] < 0) else
                   "equivalent" if prim.get("equivalent_within_margin") and calib.get("dose_response_informative") else "inconclusive")
        prim["verdict"] = verdict
        rep["tierA"]["primary_log_JS_NAT_minus_C1"] = prim
        fam = {}
        fam["A: NAT - C2 (log JS)"] = summarize(lg(m("NAT")) - lg(m("C2_a1")), rng, margin_a, B)
        fam["A: C2 - C1 (log JS)"] = summarize(lg(m("C2_a1")) - lg(m("C1_a1")), rng, margin_a, B)
        have_dose = all(tierA.get(k) for k in ("C1_a0.5", "C1_a1", "C1_a2"))
        if have_dose:
            idxD = sorted(set.intersection(*[set(tierA[k]) for k in ("C1_a0.5", "C1_a1", "C1_a2")]))
            sl = np.array([dose_slope_one([math.log(per_problem_mean_js(tierA[f"C1_a{a:g}"][i])) for a in DOSES], np.log(DOSES)) for i in idxD])
            fam["A: dose slope of C1 (log JS per log alpha)"] = summarize(sl, rng, None, B)
        fam["A: top-1 flip rate NAT - C1"] = summarize(fl("NAT") - fl("C1_a1"), rng, None, B)
        if tierA.get("C1star"):
            idxS = sorted(set(idxA) & set(tierA["C1star"]))
            fam["A: NAT - C1* (effect-matched, log JS)"] = summarize(
                np.array([math.log(per_problem_mean_js(tierA["NAT"][i])) - math.log(per_problem_mean_js(tierA["C1star"][i])) for i in idxS]), rng, margin_a, B)
        rep["tierA"]["secondary"] = fam
        # sensitivity (descriptive; none replaces the primary)
        sens = {}
        sens["steps 1-32"] = summarize(lg(m("NAT", hi=32)) - lg(m("C1_a1", hi=32)), rng, None, B)
        sens["steps 9-64"] = summarize(lg(m("NAT", lo=9)) - lg(m("C1_a1", lo=9)), rng, None, B)
        sens["per-problem median JS"] = summarize(lg(m("NAT", stat="median")) - lg(m("C1_a1", stat="median")), rng, None, B)
        keep = [k for k, i in enumerate(idxA) if len(tierA["NAT"][i]["js"]) >= 32]
        if len(keep) >= 10:
            sens["problems with >= 32 steps"] = summarize((lg(m("NAT")) - lg(m("C1_a1")))[keep], rng, None, B)
        sens["relative difference (NAT-C1)/C1"] = summarize((m("NAT") - m("C1_a1")) / m("C1_a1"), rng, None, B)
        if exposed:
            keep = [k for k, i in enumerate(idxA) if i not in exposed]
            if len(keep) >= 10:
                sens["excluding exposed test problems"] = summarize((lg(m("NAT")) - lg(m("C1_a1")))[keep], rng, None, B)
        # step-level cluster bootstrap: resample problems, pool steps (long problems weigh more)
        steps = [(np.log(np.maximum(np.asarray(tierA["NAT"][i]["js"], float), FLOOR)), np.log(np.maximum(np.asarray(tierA["C1_a1"][i]["js"], float), FLOOR))) for i in idxA]
        per = np.array([ (a - b).sum() for a, b in steps]); cnt = np.array([len(a) for a, _ in steps], float)
        bi = rng.integers(0, len(idxA), size=(B, len(idxA)))
        sens["pooled steps, problem-clustered (mean of log JS ratio)"] = {"estimate": float(per.sum() / cnt.sum()),
            "ci95": [float(x) for x in np.percentile(per[bi].sum(1) / cnt[bi].sum(1), [2.5, 97.5])]}
        # effective-dose check in the first steps (few perturbed entries; accumulation cannot explain differences)
        sens["steps 2-8 only"] = summarize(lg(m("NAT", lo=2, hi=8)) - lg(m("C1_a1", lo=2, hi=8)), rng, None, B)
        rep["sensitivity"]["tierA"] = sens
        # diagnostics
        diag = {}
        for arm in ("NAT", "C1_a1", "C2_a1", "C1_a0.5", "C1_a2", "C1star"):
            if tierA.get(arm):
                recs = tierA[arm].values()
                diag[arm] = {"mean_norm_dev_mean": float(np.mean([np.mean(r["norm_dev_mean"]) for r in recs])),
                             "max_norm_dev_max": float(np.max([np.max(r["norm_dev_max"]) for r in recs])),
                             "mean_rel_scale": float(np.mean([np.mean(r["rel_scale_mean"]) for r in recs]))}
        rep["tierA"]["diagnostics"] = diag
    # ---------------- Tier B
    if tierB.get("NAT") and tierB.get("C1") and tierB.get("FP16"):
        idxB = sorted(set(tierB["FP16"]) & set(tierB["NAT"]) & set(tierB["C1"]))
        Y = {a: np.array([restricted_first_div(tierB[a][i]["ids"], tierB["FP16"][i]["ids"]) for i in idxB]) for a in ("NAT", "C1") + (("C2",) if tierB.get("C2") else ())}
        rel_margin = calib.get("tierB_relative_margin_c")
        mB = rel_margin * float(Y["NAT"].mean()) if rel_margin else None
        rep["tierB"]["primary_restricted_first_divergence_NAT_minus_C1"] = summarize(Y["NAT"] - Y["C1"], rng, mB, B)
        rep["tierB"]["margin_note"] = ("relative margin c x mean(NAT restricted first divergence); supportive only (no dose arm exists in Tier B)")
        famB = {}
        if "C2" in Y:
            idxB2 = sorted(set(idxB) & set(tierB["C2"]))
            Y["C2"] = np.array([restricted_first_div(tierB["C2"][i]["ids"], tierB["FP16"][i]["ids"]) for i in idxB2])
            famB["B: NAT - C2 (restricted first divergence)"] = summarize(
                np.array([restricted_first_div(tierB["NAT"][i]["ids"], tierB["FP16"][i]["ids"]) - restricted_first_div(tierB["C2"][i]["ids"], tierB["FP16"][i]["ids"]) for i in idxB2]), rng, None, B)
        rep["tierB"]["secondary"] = famB
        sensB = {}
        for tau in (32, 128):
            dd = np.array([restricted_first_div(tierB["NAT"][i]["ids"], tierB["FP16"][i]["ids"], tau) - restricted_first_div(tierB["C1"][i]["ids"], tierB["FP16"][i]["ids"], tau) for i in idxB])
            sensB[f"tau={tau}"] = summarize(dd, rng, None, B)
        acc = np.array([(tierB["NAT"][i]["status"] == "SOLVED") - (tierB["C1"][i]["status"] == "SOLVED") for i in idxB], float)
        sensB["accuracy NAT - C1 (paired, descriptive; underpowered)"] = summarize(acc, rng, None, B)
        accf = np.array([(tierB["NAT"][i]["status"] == "SOLVED") - (tierB["FP16"][i]["status"] == "SOLVED") for i in idxB], float)
        sensB["accuracy NAT - FP16 (paired, descriptive)"] = summarize(accf, rng, None, B)
        accg = np.array([(tierB["C1"][i]["status"] == "SOLVED") - (tierB["FP16"][i]["status"] == "SOLVED") for i in idxB], float)
        sensB["accuracy C1 - FP16 (paired, descriptive)"] = summarize(accg, rng, None, B)
        if exposed:
            keep = [k for k, i in enumerate(idxB) if i not in exposed]
            if len(keep) >= 10:
                sensB["excluding exposed test problems"] = summarize((Y["NAT"] - Y["C1"])[keep], rng, None, B)
        rep["sensitivity"]["tierB"] = sensB
        rep["tierB"]["mean_restricted_first_divergence"] = {a: float(v.mean()) for a, v in Y.items()}
    # ---------------- Holm over the secondary family (primary is tested separately at full alpha)
    allsec = {**rep["tierA"].get("secondary", {}), **rep["tierB"].get("secondary", {})}
    if rep["tierB"].get("primary_restricted_first_divergence_NAT_minus_C1"):
        allsec["B: NAT - C1 (restricted first divergence)"] = rep["tierB"]["primary_restricted_first_divergence_NAT_minus_C1"]
    if allsec:
        names = list(allsec)
        adj = holm([allsec[k]["p_signflip"] for k in names])
        rep["holm_family"] = {k: {"raw_p": allsec[k]["p_signflip"], "holm_p": float(a), "estimate": allsec[k]["estimate"], "ci95": allsec[k]["ci95"]} for k, a in zip(names, adj)}
    return rep


def to_markdown(rep: dict) -> str:
    L = ["# E1 report (generated; see E1_SPEC.md for definitions)", ""]
    c = rep["calibration"]
    L += [f"Calibration (train problems only): alpha* = {c.get('alpha_star'):.3f}; dose slope b = {c.get('dose_slope_b_hat'):.3f}; c = {c.get('c_used'):.3f}; "
          f"Tier A margin (log JS) = {c.get('margin_tierA_log_js')}", ""]
    p = rep["tierA"].get("primary_log_JS_NAT_minus_C1")
    if p:
        L += ["## Tier A primary: log mean-JS, NAT minus C1 (paired by problem)",
              f"estimate {p['estimate']:+.4f} (ratio {p['ratio_NAT_over_C1']:.3f}); 95% CI [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}]; raw p {p['p_signflip']:.4f}; "
              f"90% CI [{p['ci90'][0]:+.4f}, {p['ci90'][1]:+.4f}]; margin {p.get('margin')}; verdict: **{p['verdict']}**", ""]
    pb = rep["tierB"].get("primary_restricted_first_divergence_NAT_minus_C1")
    if pb:
        L += ["## Tier B: restricted first divergence (tokens), NAT minus C1",
              f"estimate {pb['estimate']:+.2f}; 95% CI [{pb['ci95'][0]:+.2f}, {pb['ci95'][1]:+.2f}]; raw p {pb['p_signflip']:.4f}", ""]
    if rep["holm_family"]:
        L += ["## Secondary family (Holm)", "| Test | estimate | 95% CI | raw p | Holm p |", "|---|---|---|---|---|"]
        for k, v in rep["holm_family"].items():
            L.append(f"| {k} | {v['estimate']:+.4f} | [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] | {v['raw_p']:.4f} | {v['holm_p']:.4f} |")
        L.append("")
    for tier, d in rep["sensitivity"].items():
        L += [f"## Sensitivity ({tier}; descriptive)", "| Analysis | estimate | 95% CI |", "|---|---|---|"]
        for k, v in d.items():
            L.append(f"| {k} | {v['estimate']:+.4f} | [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] |")
        L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["calibrate", "analyze"])
    ap.add_argument("--dir", default=str(HERE / "e1_runs"))
    a = ap.parse_args()
    d = Path(a.dir)
    if a.cmd == "calibrate":
        calib = {k: load_jsonl(d / "calib" / f"{k}.jsonl") for k in ("NAT", "C1_a0.5", "C1_a1", "C1_a2", "C1_a4")}
        res = calibrate(calib)
        (d / "e1_calibration.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))
        return
    calib = json.loads((d / "e1_calibration.json").read_text())
    sample = json.loads((HERE / "e1_sample.json").read_text())
    exposed = set(sample.get("exposed_test_ids_in_sample", []))
    tierB = {k: load_jsonl(d / "tierB" / f"{k}.jsonl") for k in ("FP16", "NAT", "C1", "C2")}
    rep = analyze(tier_a_arms(d), tierB, calib, exposed)
    (d / "e1_report.json").write_text(json.dumps(rep, indent=1))
    (d / "e1_report.md").write_text(to_markdown(rep), encoding="utf-8")
    print(to_markdown(rep))


if __name__ == "__main__":
    main()
