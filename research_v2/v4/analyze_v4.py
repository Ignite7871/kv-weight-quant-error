"""v4 analysis (PROTOCOL_V4.md sections 3, 4, 6-9). CPU only; reads v4_runs (and, for Step 0, the locked v3 records and report) read-only.

  python research_v2/v4/analyze_v4.py calibrate --model M           -> v4_runs/<model>/calibration.json (calibration problems only)
  python research_v2/v4/analyze_v4.py step0                         -> v4_runs/step0.json (Step 0 criteria 1-3, reported items; decision)
  python research_v2/v4/analyze_v4.py step0 --author-decision proceed|stop
                                                                    record the authors' decision after a Step 0 failure (needs
                                                                    v4_runs/STEP0_AUTHOR_DECISION.txt with PROTOCOL_V4, the lock id and the word)
  python research_v2/v4/analyze_v4.py analyze                       -> v4_runs/v4_report.json (refuses before every model is complete or stopped)

Statistics: the analyze_v3 primitives, imported unchanged (percentile problem bootstrap, two-sided paired sign-flip test, Holm, the verdict rule,
log mean-JS with the 1e-9 floor). Each summary draws from its own stream numpy.random.default_rng([20261105, key...]) so that every estimate is
reproducible on its own. X13a uses the X13 functions with the model's own head_dim and the null stream default_rng([20261105, 13, model]).
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
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
import analyze_v3 as L                          # noqa: E402  (locked primitives, unchanged)
import analyze_x13 as X13                       # noqa: E402  (X13a functions, unchanged; head_dim passed explicitly)

OUT = Path(os.environ.get("V4_RUNS", RV / "v4_runs"))
V3_RUNS = RV / "v3_runs"
V3_REPORT = RV / "v3" / "phase5" / "v3_report_locked.json"
X13_RESULTS = RV / "v3" / "phase8" / "x13_results.json"
MANIFEST = HERE / "V4_MODEL_MANIFEST.json"
SAMPLE = HERE / "v4_sample.json"
SEED = 20261105                                  # locked-analysis seed (section 7)
N_NULL = 200
LN4, LN2 = math.log(4.0), math.log(2.0)
MIN_N = 100                                      # section 10: fewer test problems -> unresolved (r5: the full design, so a partial model is unresolved)
STOP_FILES = ("ceiling_stop.json", "drift_stop.json", "integrity_failure.json", "g8_failure.json")
QWEN = ["qwen2.5-0.5b", "qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b"]
LLAMA = ["llama3.2-1b", "llama3.2-3b", "llama3.1-8b"]
MISTRAL = ["mistral-7b-v0.3"]
LADDER = QWEN + LLAMA + MISTRAL
STEP0 = {"qwen2.5-0.5b": "qwen-base", "llama3.2-1b-instruct": "llama-instruct", "llama3.2-1b": "llama-base"}
CONTRASTS = {"H1": ("NAT-K", "C1-K"), "H2": ("NAT-V", "C1-V"), "H3": ("NAT-K", "NAT-V"), "NAT-C1": ("NAT", "C1")}
V3_KEYS = {"NAT-C1": ("family", "NAT-C1"), "H1": ("family", "Prov-K NAT-K-C1-K"), "H2": ("family", "Prov-V NAT-V-C1-V"),
           "H3": ("descriptive", "NAT-K-NAT-V")}
PREDICTED = {"H1": -1, "H2": +1, "H3": -1}
STEP0_D_MAX, STEP0_VERDICT_MIN = 0.10, 0.2
CAL_REL_TOL, BHAT_TOL = 0.05, 0.25


def manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))["models"]


def key_int(k) -> int:
    return k if isinstance(k, int) else int.from_bytes(hashlib.sha256(str(k).encode()).digest()[:4], "big")


def rng_for(*keys):
    return np.random.default_rng([SEED] + [key_int(k) for k in keys])


def summarize(d, keys, margin=None, informative=True, B=None):
    return L.summarize(d, rng_for(*keys), margin, informative, B)


def load_arm(tag, split, arm):
    return L.load_jsonl(OUT / tag / split / f"{arm}.jsonl")


def load_test(tag, arms=("NAT", "C1", "NAT-K", "C1-K", "NAT-V", "C1-V")):
    return {a: load_arm(tag, "test", a) for a in arms}


# ----------------------------------------------------------------------------- calibration
def calibrate_core(recs: dict, rhoK, rhoV) -> dict:
    """The E1/v3 rules (analyze_v3.calibrate_from, restricted to the v4 outputs): b_hat = mean per-problem OLS slope of log m on log alpha over
    alpha in {0.5, 1, 2}; c = median per-problem CV of the NAT per-step target norm, clamped to [0.10, 0.40]; margin = |b_hat| ln(1 + c);
    informative iff b_hat >= 0.25; rho_K, rho_V = medians of the per-slot relative natural error."""
    idx = sorted(set.intersection(*[set(v) for v in recs.values()]))
    x = np.log(L.DOSES) - np.log(L.DOSES).mean()
    slopes = []
    for i in idx:
        y = np.asarray([L.logm(recs[f"C1_a{a:g}"][i]) for a in L.DOSES])
        slopes.append(float((x * (y - y.mean())).sum() / (x * x).sum()))
    b_hat = float(np.mean(slopes))
    cvs = []
    for i in idx:
        t = np.asarray(recs["NAT"][i]["target_norm_mean"], float)
        if len(t) > 3 and t.mean() > 0:
            cvs.append(t.std() / t.mean())
    c_raw = float(np.median(cvs))
    c = float(min(max(c_raw, L.C_LO), L.C_HI))
    rho_K, rho_V = float(np.median(rhoK)), float(np.median(rhoV))
    return {"n_calibration_problems": len(idx), "dose_slope_b_hat": b_hat, "dose_slope_problem_sd": float(np.std(slopes)),
            "dose_response_informative": bool(b_hat >= L.MIN_SLOPE), "cv_raw_median": c_raw, "c_used": c, "margin_log_js": abs(b_hat) * math.log(1 + c),
            "rho_K": rho_K, "rho_V": rho_V, "rho_V_over_rho_K": rho_V / rho_K}


def calibrate(tag):
    import torch
    d = OUT / tag / "calib"
    recs = {t: L.load_jsonl(d / f"{t}.jsonl") for t in ("NAT", "C1_a0.5", "C1_a1", "C1_a2")}
    aux = torch.load(d / "aux_state.pt")
    cal = calibrate_core(recs, aux["rhoK"].numpy(), aux["rhoV"].numpy())
    cal.update(tag=tag, n_measure_steps=int(aux["n_meas"]), included=len(aux["included"]))
    p = OUT / tag / "calibration.json"
    if p.exists():
        sys.exit(f"{p} exists; a frozen calibration is never rewritten")
    p.write_text(json.dumps(cal, indent=1), encoding="utf-8")
    print(json.dumps(cal, indent=1))
    print(f"next: tools/lock_v4.py --append-calibration {tag}")
    return cal


# ----------------------------------------------------------------------------- per-model estimates
def theta(A, a, b, idx, **kw):
    return L.theta(A, a, b, idx, **kw)


def x13a(tag, idx, hd, model_key):
    """{path: (x_i list)} for NAT; x_i = ln(n_i C_i) - nu_i with the isotropic null in R^head_dim."""
    out = {}
    for path in ("K", "V"):
        rng_null = np.random.default_rng([SEED, 13, key_int(model_key), key_int(path)])
        xs = []
        for i in idx:
            with np.load(OUT / tag / "test" / "slots" / f"NAT_{i}.npz") as z:
                norms, S1 = z[f"norms{path}"], z[f"S1{path}"]
            n = norms.shape[0]
            nu = X13.null_lognc(norms, rng_null, N_NULL, hd)
            xs.append(math.log(n * X13.cm_share(norms, S1)) - nu)
        out[path] = xs
    return out


def c_per_layer(tag, idx):
    """Section 7.5 descriptive: C^P per layer, pooled over problems (X13 cm_per_layer)."""
    slots = {}
    for i in idx:
        with np.load(OUT / tag / "test" / "slots" / f"NAT_{i}.npz") as z:
            slots[i] = {k: z[k] for k in z.files}
    return {p: X13.cm_per_layer(slots, p) for p in ("K", "V")}


def rdev_truncated_theta(A, a, b, idx):
    """Amendment 2 sensitivity (exploratory): steps not reached by any write with a slot above 5 % in either arm; problems with fewer than 8
    retained steps are dropped."""
    vals, dropped = [], 0
    for i in idx:
        k = min(L.first_unaffected(A[a][i]), L.first_unaffected(A[b][i]))
        if k < L.SENS_MIN_STEPS:
            dropped += 1
            continue
        vals.append(L.logm(L.truncated(A[a][i], k)) - L.logm(L.truncated(A[b][i], k)))
    return vals, dropped


def per_model(tag, model, cal, idx, exposed, hd):
    A = load_test(tag)
    margin, inf = cal["margin_log_js"], cal["dose_response_informative"]
    res = {"tag": tag, "model": model, "n_problems": len(idx), "head_dim": hd, "margin": margin, "dose_informative": inf,
           "rho_K": cal["rho_K"], "rho_V": cal["rho_V"], "rho_V_over_rho_K": cal["rho_V_over_rho_K"], "contrasts": {}, "sensitivity": {}}
    keep = [i for i in idx if i not in set(exposed)]
    for h, (a, b) in CONTRASTS.items():
        res["contrasts"][h] = summarize(theta(A, a, b, idx), (tag, h), margin, inf)
        sens = {"excluding exposed": summarize(theta(A, a, b, keep), (tag, h, "exposed"))["estimate"],
                "steps 2-8": summarize(theta(A, a, b, idx, lo=2, hi=8), (tag, h, "2-8"))["estimate"],
                "per-problem median": summarize(theta(A, a, b, idx, stat="median"), (tag, h, "median"))["estimate"]}
        long_ = [i for i in idx if A[a][i]["S"] >= 9 and A[b][i]["S"] >= 9]
        sens["steps 9-64"] = summarize(theta(A, a, b, long_, lo=9), (tag, h, "9-64"))["estimate"] if long_ else None
        tv, dropped = rdev_truncated_theta(A, a, b, idx)
        sens["rounding truncation (>5 %)"] = {"estimate": summarize(tv, (tag, h, "rdev"))["estimate"] if tv else None, "dropped": dropped}
        res["sensitivity"][h] = sens
    res["descriptive"] = {"rel_K_mean": float(np.mean([np.mean(A["NAT"][i]["rel_K"]) for i in idx])),
                          "rel_V_mean": float(np.mean([np.mean(A["NAT"][i]["rel_V"]) for i in idx]))}
    return res, A


# ----------------------------------------------------------------------------- readings
def reading_directional(s, holm_p, sign):
    """H1 / H3 (sign -1) and H2 (sign +1): supported / falsified / inconclusive (section 7.1-7.3)."""
    lo, hi = s["ci95"]
    sup = (hi < 0) if sign < 0 else (lo > 0)
    fal = ((lo > 0) if sign < 0 else (hi < 0)) or s["verdict"] == "equivalent"
    if holm_p < 0.05 and sup:
        return "supported"
    return "falsified" if fal else "inconclusive"


def reading_h5(s, holm_p):
    if s["ci95"][0] > LN4 and holm_p < 0.05:
        return "supported"
    return "falsified" if s["ci95"][1] < LN2 else "inconclusive"


def universal(readings: dict) -> str:
    v = list(readings.values())
    if v and all(x == "supported" for x in v):
        return "supported in every model"
    if any(x == "falsified" for x in v):
        return "falsified"
    return "not established in all models"


def family_tests(models, est, key, sign):
    ps = [est[m]["contrasts"][key]["p"] for m in models]
    ph = L.holm(ps) if ps else []
    rd = {m: reading_directional(est[m]["contrasts"][key], float(h), sign) for m, h in zip(models, ph)}
    return {"models": models, "holm_p": dict(zip(models, map(float, ph))), "readings": rd, "result": universal(rd)}


def h5_family(models, x13s):
    keys = [(m, p) for m in models for p in ("K", "V")]
    ss = {k: summarize(x13s[k[0]][k[1]], (k[0], "H5", k[1])) for k in keys}
    ph = L.holm([ss[k]["p"] for k in keys]) if keys else []
    rd = {f"{m}/{p}": reading_h5(ss[(m, p)], float(h)) for (m, p), h in zip(keys, ph)}
    return {"summaries": {f"{m}/{p}": ss[(m, p)] for m, p in keys}, "holm_p": {f"{m}/{p}": float(h) for (m, p), h in zip(keys, ph)},
            "readings": rd, "result": universal(rd)}


def h4_permutation(theta_hat: dict, qwen, llama):
    """Exact one-sided permutation over all C(n, |llama|) family labellings; T = mean(Llama) - mean(Qwen); models are the unit."""
    ms = list(qwen) + list(llama)
    vals = np.array([theta_hat[m] for m in ms])
    k = len(llama)
    t_obs = vals[len(qwen):].mean() - vals[:len(qwen)].mean()
    ts = []
    for comb in itertools.combinations(range(len(ms)), k):
        mask = np.zeros(len(ms), bool)
        mask[list(comb)] = True
        ts.append(vals[mask].mean() - vals[~mask].mean())
    ts = np.array(ts)
    return {"T": float(t_obs), "n_labellings": len(ts), "p_one_sided": float((ts >= t_obs - 1e-12).sum() / len(ts)), "min_attainable_p": 1.0 / len(ts),
            "complete_separation": bool(max(theta_hat[m] for m in qwen) < min(theta_hat[m] for m in llama))}


def h4(models, est, excluded=()):
    """r5: H4 needs all seven Qwen/Llama models. `excluded` is the BF16-flagged set of the without-flagged version only (section 5), whose smaller
    permutation set and smallest attainable p are reported."""
    qwen = [m for m in QWEN if m in models]
    llama = [m for m in LLAMA if m in models]
    if set(qwen + llama) != {m for m in QWEN + LLAMA if m not in excluded}:
        return {"result": "not estimable", "qwen": qwen, "llama": llama, "reason": "H4 needs all seven Qwen/Llama models (section 10)"}
    th = {m: est[m]["contrasts"]["NAT-C1"]["estimate"] for m in qwen + llama}
    perm = h4_permutation(th, qwen, llama)
    s = {m: est[m]["contrasts"]["NAT-C1"] for m in qwen + llama}
    signs_ok = all(s[m]["ci95"][1] < 0 for m in qwen) and not any(s[m]["ci95"][1] < 0 for m in llama)
    falsified = (any(s[m]["ci95"][0] > 0 or s[m]["verdict"] == "equivalent" for m in qwen) or any(s[m]["ci95"][1] < 0 for m in llama))
    result = "supported" if (perm["p_one_sided"] <= 0.05 and signs_ok) else ("falsified" if falsified else "inconclusive")
    out = {"qwen": qwen, "llama": llama, "theta_hat": th, "permutation": perm, "per_model_signs_ok": signs_ok, "result": result}
    if "mistral-7b-v0.3" in est:
        out["mistral_out_of_sample"] = {k: est["mistral-7b-v0.3"]["contrasts"]["NAT-C1"][k] for k in ("estimate", "ci95")}
    params = {m: manifest()[m]["shards"] for m in qwen + llama}
    sizes = {m: sum(v["size"] for v in params[m].values()) for m in params}
    out["spearman_theta_vs_size"] = {"qwen": L.spearman([th[m] for m in qwen], [math.log(sizes[m]) for m in qwen]),
                                     "llama": L.spearman([th[m] for m in llama], [math.log(sizes[m]) for m in llama]),
                                     "note": "descriptive, no test; size = total bytes of the pinned shards"}
    return out


def hypotheses(models, est, x13s, excluded=()):
    return {"F1 H1": family_tests(models, est, "H1", PREDICTED["H1"]), "F2 H2": family_tests(models, est, "H2", PREDICTED["H2"]),
            "F3 H3": family_tests(models, est, "H3", PREDICTED["H3"]), "F5 H5": h5_family(models, x13s), "H4": h4(models, est, excluded)}


# ----------------------------------------------------------------------------- analyze
def model_status(tag):
    if (OUT / tag / "not_run.json").exists():
        return "not run"
    if not (OUT / tag / "calibration.json").exists():
        return "missing"
    n = len(load_arm(tag, "test", "NAT"))
    if (OUT / tag / "test_complete.json").exists():
        return "complete"
    if any((OUT / tag / f).exists() for f in STOP_FILES):
        return "stopped" if n >= MIN_N else "unresolved"
    return "running"


def analyze():
    s = json.loads(SAMPLE.read_text(encoding="utf-8"))
    man = manifest()
    status = {m: model_status(m) for m in LADDER}
    if any(v in ("running", "missing") for v in status.values()) and not (OUT / "STUDY_STOPPED.json").exists():
        sys.exit(f"Refusing: not every model is complete or stopped: {status}")
    flagged = [m for m in LADDER if (OUT / m / "precision.json").exists()]
    usable = [m for m in LADDER if status[m] in ("complete", "stopped")]
    est, x13s = {}, {}
    for m in usable:
        cal = json.loads((OUT / m / "calibration.json").read_text(encoding="utf-8"))
        A = load_test(m)
        idx = sorted(set.intersection(*[set(v) for v in A.values()]))
        hd = man[m]["geometry"]["head_dim"]
        est[m], _ = per_model(m, m, cal, idx, s["exposed_test_ids_in_sample"], hd)
        x13s[m] = x13a(m, idx, hd, m)
        est[m]["H5"] = {p: summarize(x13s[m][p], (m, "H5", p)) for p in ("K", "V")}
        est[m]["H5_C_per_layer_descriptive"] = c_per_layer(m, idx)
    rep = {"seed": SEED, "status": status, "flagged_bf16": flagged, "per_model": est}
    rep["all_models"] = hypotheses(usable, est, x13s)
    rep["without_flagged"] = hypotheses([m for m in usable if m not in flagged], est, x13s, excluded=flagged)
    disagree = {k: (rep["all_models"][k]["result"], rep["without_flagged"][k]["result"]) for k in rep["all_models"]
                if rep["all_models"][k]["result"] != rep["without_flagged"][k]["result"]}
    rep["abstract_level"] = {"source": "without_flagged" if flagged else "all_models", "disagreements": disagree}
    (OUT / "v4_report.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: rep["all_models"][k]["result"] for k in rep["all_models"]}, indent=1))
    return rep


# ----------------------------------------------------------------------------- Step 0
def v3_theta(label, a, b, idx):
    A = {x: L.load_jsonl(V3_RUNS / label / "tierA" / f"{x}.jsonl") for x in (a, b)}
    return L.theta(A, a, b, idx)


def category(s):
    return ("difference " + ("+" if s["estimate"] > 0 else "-")) if s["verdict"] == "difference" else s["verdict"]


def v3_calibration_on(label):
    """r5: v3's calibration on the same problems as the cloud's (the first 50 of the fixed order), by the same rules, so that criterion 3 compares like with
    like: the locked v3 file used 100. The per-slot ratios are concatenated per problem in the order of `included`."""
    import torch
    ids = json.loads(SAMPLE.read_text(encoding="utf-8"))["calibration_train_ids"]
    d = V3_RUNS / label / "calib"
    recs = {t: L.load_jsonl(d / f"{t}.jsonl") for t in ("NAT", "C1_a0.5", "C1_a1", "C1_a2")}
    aux = torch.load(d / "aux_state.pt")
    inc = list(aux["included"])
    if inc[:len(ids)] != list(ids):
        raise ValueError(f"{label}: the v3 calibration order is not the sample's")
    steps = [len(recs["NAT"][i]["js"]) for i in inc]
    per_step = aux["rhoK"].shape[0] // sum(steps)
    if per_step * sum(steps) != aux["rhoK"].shape[0]:
        raise ValueError(f"{label}: the ratios do not divide into steps")
    n = per_step * sum(steps[:len(ids)])
    return calibrate_core({t: {i: r[i] for i in ids} for t, r in recs.items()}, aux["rhoK"][:n].numpy(), aux["rhoV"][:n].numpy())


def step0_model(m, label, v3rep):
    A = load_test(m)
    cal = json.loads((OUT / m / "calibration.json").read_text(encoding="utf-8"))
    cal3 = json.loads((V3_RUNS / label / "calibration.json").read_text(encoding="utf-8"))
    margin3 = cal3["margin_log_js"]
    ids3 = set(L.load_jsonl(V3_RUNS / label / "tierA" / "NAT.jsonl"))
    idx = sorted(set.intersection(*[set(v) for v in A.values()]) & ids3)
    out = {"v3_label": label, "n_paired": len(idx), "v3_margin": margin3, "contrasts": {}}
    ok = True
    for h, (a, b) in CONTRASTS.items():
        tc, t3 = theta(A, a, b, idx), v3_theta(label, a, b, idx)
        dl = summarize(tc - t3, (m, "step0", h))
        c1 = bool(-margin3 < dl["ci90"][0] and dl["ci90"][1] < margin3 and abs(dl["estimate"]) <= STEP0_D_MAX)
        sec, key = V3_KEYS[h]
        s3 = v3rep["per_model"][label][sec][key]
        row = {"delta": dl, "criterion_1": c1, "theta_v3": s3["estimate"], "v3_category": category(s3)}
        if abs(s3["estimate"]) >= STEP0_VERDICT_MIN:
            sc = summarize(tc, (m, "step0-cloud", h), cal["margin_log_js"], cal["dose_response_informative"])
            row.update(cloud_category=category(sc), criterion_2=category(sc) == category(s3))
        else:
            row["criterion_2"] = "not applicable (|theta_v3| < 0.2)"
        ok = ok and c1 and row["criterion_2"] is not False
        out["contrasts"][h] = row
    cal3s = v3_calibration_on(label)
    c3 = {"rho_K": abs(cal["rho_K"] / cal3s["rho_K"] - 1) <= CAL_REL_TOL, "rho_V": abs(cal["rho_V"] / cal3s["rho_V"] - 1) <= CAL_REL_TOL,
          "b_hat": abs(cal["dose_slope_b_hat"] - cal3s["dose_slope_b_hat"]) <= BHAT_TOL}
    out["criterion_3"] = {**c3, "v4": {k: cal[k] for k in ("rho_K", "rho_V", "dose_slope_b_hat")},
                          "v3": {k: cal3s[k] for k in ("rho_K", "rho_V", "dose_slope_b_hat")}, "v3_n_problems": cal3s["n_calibration_problems"],
                          "v3_locked_all_100": {k: cal3[k] for k in ("rho_K", "rho_V", "dose_slope_b_hat")}}
    ok = ok and all(c3.values())
    h4 = L.load_jsonl(OUT / m / "test" / "FP16_history.jsonl")
    h3 = L.load_jsonl(V3_RUNS / label / "tierA" / "FP16_history.jsonl")
    same = [h4[i]["ids"] == h3[i]["ids"] for i in idx if i in h4 and i in h3]
    js4 = np.concatenate([np.asarray(A["NAT"][i]["js"], float) for i in idx])
    A3 = L.load_jsonl(V3_RUNS / label / "tierA" / "NAT.jsonl")
    pairs = [(np.asarray(A["NAT"][i]["js"], float), np.asarray(A3[i]["js"], float)) for i in idx]
    pairs = [(x, y) for x, y in pairs if len(x) == len(y)]
    r = float(np.corrcoef(np.concatenate([x for x, _ in pairs]), np.concatenate([y for _, y in pairs]))[0, 1]) if pairs else None
    hd = manifest()[m]["geometry"]["head_dim"]
    xs = x13a(m, idx, hd, m)
    x13ref = json.loads(X13_RESULTS.read_text(encoding="utf-8"))["per_model"].get(label, {}).get("a", {})
    out["reported"] = {"history_token_identical_share": float(np.mean(same)) if same else None, "nat_step_js_correlation_with_v3": r,
                       "nat_js_pairs_same_length": len(pairs), "nat_js_total_steps": int(js4.size),
                       "X13a_v4": {p: float(np.mean(xs[p])) for p in xs},
                       "X13a_X13": {p: x13ref.get(p, {}).get("mean_ln_nC_minus_null") for p in ("K", "V")}}
    out["pass"] = bool(ok)
    return out


def step0(author_decision=None):
    p = OUT / "step0.json"
    if author_decision:
        d = json.loads(p.read_text(encoding="utf-8"))
        if d["decision"] == "pass":
            sys.exit("Step 0 passed; no author decision is needed")
        f = OUT / "STEP0_AUTHOR_DECISION.txt"
        txt = f.read_text(encoding="utf-8") if f.exists() else ""
        import subprocess
        lid = subprocess.run([sys.executable, str(RV / "tools" / "lock_v4.py"), "--lock-id"], capture_output=True, text=True, encoding="utf-8", errors="replace").stdout.strip()
        if not ("PROTOCOL_V4" in txt and lid and lid in txt and author_decision in txt.split()):
            sys.exit("STEP0_AUTHOR_DECISION.txt (written by the corresponding author) must contain PROTOCOL_V4, the lock id and the decision word")
        d["decision"] = f"author: {author_decision}"
        d["author_decision_file_sha256"] = hashlib.sha256(f.read_bytes()).hexdigest()
        if author_decision == "proceed":
            d["label_for_all_v3_comparisons"] = "cross-hardware, not agreed"
        p.write_text(json.dumps(d, indent=1), encoding="utf-8")
        print(d["decision"])
        return d
    if p.exists():
        sys.exit("step0.json exists; it is never recomputed")
    v3rep = json.loads(V3_REPORT.read_text(encoding="utf-8"))
    res = {m: step0_model(m, lab, v3rep) for m, lab in STEP0.items()}
    allp = all(r["pass"] for r in res.values())
    d = {"models": res, "pass": allp, "decision": "pass" if allp else "fail: stop and report; the authors decide (section 4)"}
    p.write_text(json.dumps(d, indent=1, default=float), encoding="utf-8")
    print(json.dumps({m: r["pass"] for m, r in res.items()}, indent=1), "\ndecision:", d["decision"])
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["calibrate", "step0", "analyze"])
    ap.add_argument("--model")
    ap.add_argument("--author-decision", choices=["proceed", "stop"])
    a = ap.parse_args()
    if a.cmd == "calibrate":
        calibrate(a.model)
    elif a.cmd == "step0":
        step0(a.author_decision)
    else:
        analyze()


if __name__ == "__main__":
    main()
