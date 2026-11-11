"""Generates every v3 number macro, table and figure of main_v3.tex from saved results. CPU only; reads, never writes, research outputs.

Sources (all read-only):
  research_v2/v3/phase5/v3_report_locked.json   committed copy of the locked analysis output (sha256 checked below)
  research_v2/v3/phase5/recompute_v3.json, recompute_vs_locked.json   independent recomputation
  research_v2/v3/phase5/exploratory_v3.json     exploratory X1/X2 (EXPLORATORY_PLAN_V3.md)
  research_v2/v3/phase5/exploratory_v3_x4x8.json, exploratory_v3_x10.json, exploratory_v3_x11.json, exploratory_v3_x12.json   exploratory X4-X8, X10-X12
  research_v2/v3/phase7/x9_results.json         exploratory X9 (FOLLOWUP_X9.md); v3_runs_x9/gpu_ledger.jsonl, v3_runs_x9/<model>/reproduction.jsonl
  research_v2/e1_runs/tierA, e1l_runs/tierA     earlier E1/E1L records (flip rates and mean JS, descriptive, as in E1_REPORT.md / E1L_REPORT.md)
  research_v2/v3_runs/<model>/calibration.json, tierA/FP16_history.jsonl, tierA/NAT.jsonl, gates/gates.json, g6.json; v3_runs/gpu_ledger.jsonl, design.json
  ~/.cache/huggingface/hub/<model>/snapshots/<rev>/config.json   at the revision and sha256 in research_v2/V3_PROTOCOL_LOCK.txt (LOC weight share)
Outputs: paper_journal/tables_v3/*.tex (numbers_v3.tex holds the \\vn{key} macros), paper_journal/figures_v3/*.pdf,
         paper_journal/tables_v3/numbers_v3_sources.json (key -> value, source) for the number audit.

  venv/Scripts/python.exe research_v2/paper_journal/make_v3.py
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RV = HERE.parent
P5 = RV / "v3" / "phase5"
P7 = RV / "v3" / "phase7"
RUNS = RV / "v3_runs"
RUNS9 = RV / "v3_runs_x9"
TAB = HERE / "tables_v3"
FIG = HERE / "figures_v3"
REPORT_SHA = "f9ce364a8392bd42a279041a122ca1951c3f4645523b8af7f1dd6123d8f6078b"

MODELS = ["qwen-base", "qwen-instruct", "llama-base", "llama-instruct"]          # 2x2 order: a, b, c, d
SHORT = {"qwen-base": "qb", "qwen-instruct": "qi", "llama-base": "lb", "llama-instruct": "li"}
NAME = {"qwen-base": "Qwen-base", "qwen-instruct": "Qwen-instruct", "llama-base": "Llama-base", "llama-instruct": "Llama-instruct"}
FAM = [("natc1", "NAT-C1"), ("hb", "HB QA-QO"), ("hd", "HD K*-V*"), ("provk", "Prov-K NAT-K-C1-K"), ("provv", "Prov-V NAT-V-C1-V"),
       ("he", "HE INH-LOC")]

FMLAB = {"NAT-C1": "NAT$-$C1", "HB QA-QO": "H-B: QA$-$QO", "HD K*-V*": "H-D: REL-K$-$REL-V",
         "Prov-K NAT-K-C1-K": "NAT-K$-$C1-K", "Prov-V NAT-V-C1-V": "NAT-V$-$C1-V", "HE INH-LOC": "H-E: INH$-$LOC"}
ARMNAME = {"NAT-K*": "REL-K", "NAT-V*": "REL-V"}                 # paper names of the protocol arms NAT-K*, NAT-V* (records keep the protocol names)
V, SRC = {}, {}


# ------------------------------------------------------------------ formatting
def sg(x, d=3):
    return f"{x:+.{d}f}"


def sci(x):
    m, e = f"{x:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def num(x, d=3):
    return f"${sg(x, d)}$"


def ci(c, d=3):
    return f"$[{sg(c[0], d)}, {sg(c[1], d)}]$"


def pv(p):
    return "$<0.001$" if p < 0.001 else f"${p:.3f}$"


def un(x, d=3):
    return f"${x:.{d}f}$"


def sci3(x):
    m, e = f"{x:.2e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def put(key, val, src):
    V[key] = val
    SRC[key] = src


def rng_(key, vals, fmt, src):
    """Range over models (min, max by value), so that prose never orders endpoints by hand."""
    lo, hi = min(vals.values()), max(vals.values())
    put(f"{key}.min", fmt(lo), f"min over {sorted(vals)} of {src}")
    put(f"{key}.max", fmt(hi), f"max over {sorted(vals)} of {src}")


# ------------------------------------------------------------------ load
def load():
    raw = (P5 / "v3_report_locked.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != REPORT_SHA:
        sys.exit("v3_report_locked.json does not match the locked analysis output (sha256)")
    rep = json.loads(raw)
    exp = json.loads((P5 / "exploratory_v3.json").read_text())
    rec = json.loads((P5 / "recompute_v3.json").read_text())
    cmp_ = json.loads((P5 / "recompute_vs_locked.json").read_text())["summary"]
    cal = {m: json.loads((RUNS / m / "calibration.json").read_text()) for m in MODELS}
    return rep, exp, rec, cmp_, cal


def verdict_word(v):
    return {"difference": "difference", "equivalent": "equivalent", "inconclusive": "inconclusive"}.get(v, v)


def e1l_category(s):
    """E1L_SPEC section 3 categories for the primary NAT - C1 (stated relative to the Qwen E1 direction, theta < 0)."""
    if s["verdict"] == "equivalent":
        return "equivalent to C1"
    if s["estimate"] < 0 and s["ci95"][1] < 0:
        return "replicated"
    if s["estimate"] < 0:
        return "same direction, unresolved"
    return "not replicated"


# ------------------------------------------------------------------ numbers
def numbers(rep, exp, rec, cmp_, cal):
    R = "v3_report_locked.json"
    for m in MODELS:
        k, P = SHORT[m], rep["per_model"][m]
        H = P["HA"]
        put(f"ha.{k}.est", num(H["estimate"]), f"{R}:per_model.{m}.HA.estimate")
        put(f"ha.{k}.ci", ci(H["ci95"]), f"{R}:per_model.{m}.HA.ci95")
        put(f"ha.{k}.p", pv(H["p"]), f"{R}:per_model.{m}.HA.p")
        put(f"ha.{k}.holm", pv(H["holm_p"]), f"{R}:per_model.{m}.HA.holm_p")
        put(f"ha.{k}.word", "supported" if H["supported"] else "contradicted" if H["falsified"] else "inconclusive",
            f"{R}:per_model.{m}.HA.supported/falsified (paper word for 'falsified': contradicted)")
        for key, name in FAM:
            s = P["family"][name]
            put(f"{key}.{k}.est", num(s["estimate"]), f"{R}:per_model.{m}.family.{name}.estimate")
            put(f"{key}.{k}.ci", ci(s["ci95"]), f"{R}:per_model.{m}.family.{name}.ci95")
            put(f"{key}.{k}.ci90", ci(s["ci90"]), f"{R}:per_model.{m}.family.{name}.ci90")
            put(f"{key}.{k}.p", pv(s["p"]), f"{R}:per_model.{m}.family.{name}.p")
            put(f"{key}.{k}.holm", pv(s["holm_p"]), f"{R}:per_model.{m}.family.{name}.holm_p")
            put(f"{key}.{k}.verdict", verdict_word(s["verdict"]), f"{R}:per_model.{m}.family.{name}.verdict")
            put(f"{key}.{k}.ratio", un(float(np.exp(s["estimate"])), 2), f"exp({R}:per_model.{m}.family.{name}.estimate)")
        hc = P["HC"]
        put(f"hc.{k}.b0", num(hc["beta0"]), f"{R}:per_model.{m}.HC.beta0")
        put(f"hc.{k}.b0ci", ci(hc["beta0_ci95"]), f"{R}:per_model.{m}.HC.beta0_ci95")
        put(f"hc.{k}.b1", num(hc["beta1"]), f"{R}:per_model.{m}.HC.beta1")
        put(f"hc.{k}.S", num(hc["shrinkage"]), f"{R}:per_model.{m}.HC.shrinkage")
        put(f"hc.{k}.Sci", ci(hc["shrinkage_ci95"]), f"{R}:per_model.{m}.HC.shrinkage_ci95")
        put(f"hc.{k}.f", num(hc["fraction"], 2), f"{R}:per_model.{m}.HC.fraction")
        put(f"hc.{k}.fci", ci(hc["fraction_ci95"], 2), f"{R}:per_model.{m}.HC.fraction_ci95")
        put(f"hc.{k}.p", pv(hc["p"]), f"{R}:per_model.{m}.HC.p")
        put(f"hc.{k}.band", hc["band"], f"{R}:per_model.{m}.HC.band")
        if "HC shrinkage" in P["family"]:
            put(f"hc.{k}.holm", pv(P["family"]["HC shrinkage"]["holm_p"]), f"{R}:per_model.{m}.family.HC shrinkage.holm_p")
        else:
            put(f"hc.{k}.holm", "---", f"{R}:per_model.{m}.family (HC removed: not applicable)")
        put(f"fam.{k}.size", str(len(P["family"])), f"{R}:per_model.{m}.family (count)")
        put(f"natc1.{k}.cat", e1l_category(P["family"]["NAT-C1"]), f"E1L_SPEC §3 rule applied to {R}:per_model.{m}.family.NAT-C1")
        d = P["descriptive"]
        for key, name in (("natc2", "NAT-C2"), ("c2c1", "C2-C1"), ("natcv", "NAT-CV"), ("nkv", "NAT-K-NAT-V")):
            put(f"{key}.{k}.est", num(d[name]["estimate"]), f"{R}:per_model.{m}.descriptive.{name}.estimate")
            put(f"{key}.{k}.ci", ci(d[name]["ci95"]), f"{R}:per_model.{m}.descriptive.{name}.ci95")
            put(f"{key}.{k}.verdict", verdict_word(d[name]["verdict"]), f"{R}:per_model.{m}.descriptive.{name}.verdict")
        for key, name in (("pval", "HA-type P_val vs P_norm (NAT)"), ("panormk", "HA-type P_anormK vs P_norm (NAT)"),
                          ("pout", "HA-type P_out vs P_norm (NAT)"), ("hapool", "HA pooled norm-matched arms (secondary)")):
            put(f"{key}.{k}.est", num(d[name]["estimate"]), f"{R}:per_model.{m}.descriptive.{name}.estimate")
            put(f"{key}.{k}.ci", ci(d[name]["ci95"]), f"{R}:per_model.{m}.descriptive.{name}.ci95")
        na = d["nonadditivity m_NAT/(m_LOC+m_INH)"]
        put(f"nonadd.{k}.med", un(na["median"]), f"{R}:per_model.{m}.descriptive.nonadditivity.median")
        put(f"nonadd.{k}.ci", "$[" + f"{na['ci95_of_median'][0]:.3f}, {na['ci95_of_median'][1]:.3f}" + "]$",
            f"{R}:per_model.{m}.descriptive.nonadditivity.ci95_of_median")
        rk, rv_ = d["rel norm K / V (NAT, mean)"]
        put(f"rel.{k}.K", f"{100 * rk:.1f}", f"{R}:per_model.{m}.descriptive.rel norm K / V (NAT, mean)[0] x100")
        put(f"rel.{k}.V", f"{100 * rv_:.1f}", f"{R}:per_model.{m}.descriptive.rel norm K / V (NAT, mean)[1] x100")
        sens = P["sensitivity"]["NAT-C1"]
        for key, name in (("s28", "steps 2-8"), ("s964", "steps 9-64"), ("smed", "per-problem median"), ("sexp", "excluding exposed"),
                          ("s32", "S >= 32")):
            put(f"sens.{k}.{key}", num(sens[name]), f"{R}:per_model.{m}.sensitivity.NAT-C1.{name}")
        # calibration
        c = cal[m]
        put(f"cal.{k}.rhoK", un(c["rho_K"], 3), f"{m}/calibration.json:rho_K")
        put(f"cal.{k}.rhoV", un(c["rho_V"], 3), f"{m}/calibration.json:rho_V")
        put(f"cal.{k}.rhostar", un(c["rho_star"], 4), f"{m}/calibration.json:rho_star")
        put(f"cal.{k}.sK", un(c["scale_factor_K"], 2), f"{m}/calibration.json:scale_factor_K")
        put(f"cal.{k}.sV", un(c["scale_factor_V"], 2), f"{m}/calibration.json:scale_factor_V")
        put(f"cal.{k}.b", un(c["dose_slope_b_hat"], 2), f"{m}/calibration.json:dose_slope_b_hat")
        put(f"cal.{k}.c", un(c["c_used"], 2), f"{m}/calibration.json:c_used")
        put(f"cal.{k}.craw", un(c["cv_raw_median"], 3), f"{m}/calibration.json:cv_raw_median (before the clamp)")
        put(f"cal.{k}.margin", un(c["margin_log_js"], 3), f"{m}/calibration.json:margin_log_js")
        put(f"cal.{k}.E8", un(c["E8"]["median"], 2), f"{m}/calibration.json:E8.median")
        put(f"cal.{k}.r80", f"{c['r80']['median']:g}", f"{m}/calibration.json:r80.median")
        put(f"cal.{k}.r80share", f"{100 * c['r80']['share_le_8']:.0f}", f"{m}/calibration.json:r80.share_le_8 x100")
    put("ha.pooled.est", num(rep["HA_pooled_secondary"]["estimate"], 4), f"{R}:HA_pooled_secondary.estimate")
    put("ha.pooled.ci", ci(rep["HA_pooled_secondary"]["ci95"]), f"{R}:HA_pooled_secondary.ci95")
    put("ha.pooled.p", pv(rep["HA_pooled_secondary"]["p"]), f"{R}:HA_pooled_secondary.p")
    x = rep["HC_cross"]["qwen-base vs llama-instruct"]
    for key, name in (("g0", "g0"), ("g1", "g1")):
        put(f"hcx.{key}", num(x[name]), f"{R}:HC_cross.qwen-base vs llama-instruct.{name}")
        put(f"hcx.{key}ci", ci(x[name + "_ci95"]), f"{R}:HC_cross.qwen-base vs llama-instruct.{name}_ci95")
    put("hcx.F", num(x["F"], 2), f"{R}:HC_cross.qwen-base vs llama-instruct.F")
    put("hcx.Fci", ci(x["F_ci95"], 2), f"{R}:HC_cross.qwen-base vs llama-instruct.F_ci95")
    put("hcx.band", x["band"], f"{R}:HC_cross.qwen-base vs llama-instruct.band")
    # the same locked gap with the sign convention of the paper's cross-model comparisons (Llama-instruct minus Qwen-base)
    put("hcx.g0rev", num(-x["g0"]), f"-1 x {R}:HC_cross.qwen-base vs llama-instruct.g0")
    put("hcx.g0revci", ci([-x["g0_ci95"][1], -x["g0_ci95"][0]]), f"-1 x {R}:HC_cross.qwen-base vs llama-instruct.g0_ci95 (endpoints swapped)")
    # problem-mean gap, descriptive (recomputation method, seed 4242034, as in PHASE5_REPORT.md section a)
    sys.path.insert(0, str(P5))
    import recompute_v3_independent as I
    a = rep["per_model"]["llama-instruct"]["theta_NAT_C1_per_problem"]
    b = rep["per_model"]["qwen-base"]["theta_NAT_C1_per_problem"]
    g = I.paired(np.array([a[i] - b[i] for i in sorted(a)]), I.Boot(4242034), allow_equiv=False)
    put("gap.theta", num(g["estimate"]), "recompute_v3_independent.paired(theta_LI - theta_QB), seed 4242034")
    put("gap.thetaci", ci(g["ci95"]), "recompute_v3_independent.paired(theta_LI - theta_QB), seed 4242034")
    HF = rep["HF"]
    for key, name in (("fam", "family (Llama - Qwen)"), ("tun", "tuning (instruct - base)"), ("int", "interaction")):
        s = HF["contrasts"][name]
        put(f"hf.{key}.est", num(s["estimate"]), f"{R}:HF.contrasts.{name}.estimate")
        put(f"hf.{key}.ci", ci(s["ci95"]), f"{R}:HF.contrasts.{name}.ci95")
        put(f"hf.{key}.ci90", ci(s["ci90"]), f"{R}:HF.contrasts.{name}.ci90")
        put(f"hf.{key}.p", pv(s["p"]), f"{R}:HF.contrasts.{name}.p")
        put(f"hf.{key}.holm", pv(s["holm_p"]), f"{R}:HF.contrasts.{name}.holm_p")
        put(f"hf.{key}.verdict", verdict_word(s["verdict"]), f"{R}:HF.contrasts.{name}.verdict")
    put("hf.int.holmrecomp", pv(rec["HF"]["contrasts"]["interaction"]["holm_p"]), "recompute_v3.json:HF.contrasts.interaction.holm_p")
    for m in MODELS:
        put(f"hf.sd.{SHORT[m]}", un(HF["sd_per_model"][m]), f"{R}:HF.sd_per_model.{m}")
    put("hf.sdmax", un(max(HF["sd_per_model"].values())), f"max {R}:HF.sd_per_model")
    put("hf.guard", un(HF["assumed_sd"] * 1.25), f"1.25 x {R}:HF.assumed_sd")
    put("hf.margin", un(float(np.mean([cal[m]["margin_log_js"] for m in MODELS]))), "mean of calibration.json:margin_log_js")
    for key, name in (("fam", "family account"), ("tun", "tuning account"), ("add", "additivity")):
        put(f"hf.read.{key}", HF["readings"][name], f"{R}:HF.readings.{name}")
        word = {"supported": "supported", "falsified": "contradicted"}.get(HF["readings"][name], HF["readings"][name])
        if key == "add":
            word = {"falsified": "rejected", "supported": "not rejected"}.get(HF["readings"][name], HF["readings"][name])
        put(f"hf.word.{key}", word, f"{R}:HF.readings.{name} (paper wording)")
    # recomputation
    put("recomp.n", f"{cmp_['n_compared']:,}".replace(",", "{,}"), "recompute_vs_locked.json:summary.n_compared")
    put("recomp.ndis", str(cmp_["n_disagreements"]), "recompute_vs_locked.json:summary.n_disagreements")
    # exploratory
    for m in MODELS:
        k, e = SHORT[m], exp["X1"]["per_model"][m]
        put(f"x1.{k}.wK", un(e["mean_wK"], 2), f"exploratory_v3.json:X1.per_model.{m}.mean_wK")
        put(f"x1.{k}.pred", num(e["pred"]["estimate"]), f"exploratory_v3.json:X1.per_model.{m}.pred.estimate")
        put(f"x1.{k}.predci", ci(e["pred"]["ci95"]), f"exploratory_v3.json:X1.per_model.{m}.pred.ci95")
        put(f"x1.{k}.obs", num(e["obs"]["estimate"]), f"exploratory_v3.json:X1.per_model.{m}.obs.estimate")
        put(f"x1.{k}.d", num(e["obs_minus_pred"]["estimate"]), f"exploratory_v3.json:X1.per_model.{m}.obs_minus_pred.estimate")
        put(f"x1.{k}.dci", ci(e["obs_minus_pred"]["ci95"]), f"exploratory_v3.json:X1.per_model.{m}.obs_minus_pred.ci95")
        put(f"x1.{k}.holm", pv(e["obs_minus_pred"]["holm_p"]), f"exploratory_v3.json:X1.per_model.{m}.obs_minus_pred.holm_p")
        put(f"x1.{k}.sign", "yes" if e["sign_predicted"] else "no", f"exploratory_v3.json:X1.per_model.{m}.sign_predicted")
        put(f"x1.{k}.rho", num(e["spearman_pred_obs"]["rho"], 2), f"exploratory_v3.json:X1.per_model.{m}.spearman_pred_obs.rho")
        put(f"x1.{k}.rhoci", ci(e["spearman_pred_obs"]["ci95"], 2), f"exploratory_v3.json:X1.per_model.{m}.spearman_pred_obs.ci95")
        for key, name in (("x2a", "X2a P_val vs P_norm"), ("x2b", "X2b P_comb vs P_norm"), ("x2c", "X2c D_comb - D_logit")):
            s = exp["X2"][name]["per_model"][m]
            put(f"{key}.{k}.est", num(s["estimate"]), f"exploratory_v3.json:X2.{name}.per_model.{m}.estimate")
            put(f"{key}.{k}.ci", ci(s["ci95"]), f"exploratory_v3.json:X2.{name}.per_model.{m}.ci95")
            put(f"{key}.{k}.holm", pv(s["holm_p"]), f"exploratory_v3.json:X2.{name}.per_model.{m}.holm_p")
            put(f"{key}.{k}.verdict", s["verdict"], f"exploratory_v3.json:X2.{name}.per_model.{m}.verdict")
    for key, name in (("x2a", "X2a P_val vs P_norm"), ("x2b", "X2b P_comb vs P_norm"), ("x2c", "X2c D_comb - D_logit")):
        s = exp["X2"][name]["pooled_mean_over_models"]
        put(f"{key}.pooled.est", num(s["estimate"]), f"exploratory_v3.json:X2.{name}.pooled_mean_over_models.estimate")
        put(f"{key}.pooled.ci", ci(s["ci95"]), f"exploratory_v3.json:X2.{name}.pooled_mean_over_models.ci95")
    # runs: GPU hours, histories, gates, G6
    led = [json.loads(x) for x in (RUNS / "gpu_ledger.jsonl").read_text().splitlines() if x.strip()]
    tot = sum(x["seconds"] for x in led) / 3600
    put("gpu.total", f"{tot:.2f}", "sum gpu_ledger.jsonl seconds / 3600")
    put("gpu.left", f"{25.0 - tot:.2f}", "25.0 - sum gpu_ledger.jsonl")
    for m in MODELS:
        h = sum(x["seconds"] for x in led if x["block"].endswith(":" + m)) / 3600
        put(f"gpu.{SHORT[m]}", f"{h:.2f}", f"sum gpu_ledger.jsonl blocks *:{m}")
    put("gpu.mut", f"{sum(x['seconds'] for x in led if x['block'] == 'gpu_mutation_check') / 3600:.2f}", "gpu_ledger.jsonl gpu_mutation_check")
    g6 = json.loads((RUNS / "llama-base" / "g6.json").read_text())
    put("g6.lb.drift", f"${100 * g6['drift']:+.1f}$", "llama-base/g6.json:drift")
    put("g6.lb.t", f"{g6['t_arm_problem_s']:.2f}", "llama-base/g6.json:t_arm_problem_s")
    put("g6.lb.exp", f"{g6['expected_t_arm_problem_s']:.2f}", "llama-base/g6.json:expected_t_arm_problem_s")
    for m in MODELS:
        hist = [json.loads(x) for x in (RUNS / m / "tierA" / "FP16_history.jsonl").read_text().splitlines() if x.strip()]
        nat = [json.loads(x) for x in (RUNS / m / "tierA" / "NAT.jsonl").read_text().splitlines() if x.strip()]
        S = np.array([r["S"] for r in nat])
        put(f"hist.{SHORT[m]}.Smean", f"{S.mean():.1f}", f"{m}/tierA/NAT.jsonl mean S")
        put(f"hist.{SHORT[m]}.S64", f"{100 * (S == 64).mean():.0f}", f"{m}/tierA/NAT.jsonl share S == 64 x100")
        if m == "qwen-base":
            put("hist.qb.e1same", str(sum(1 for r in hist if r.get("equals_stored_e1_history"))), f"{m}/tierA/FP16_history.jsonl equals_stored_e1_history")
    # amendment-2 deviation and truncation summaries
    for m in MODELS:
        k, rd = SHORT[m], rep["per_model"][m]["realized_norm_deviation"]
        cons = {a: v for a, v in rd.items() if v["n_slots"]}
        put(f"rdev.{k}.max", f"{max(v['max'] for v in cons.values()):.3f}", f"max {R}:per_model.{m}.realized_norm_deviation.*.max")
        put(f"rdev.{k}.meanmax", sci(max(v['mean'] for v in cons.values())),
            f"max {R}:per_model.{m}.realized_norm_deviation.*.mean")
        sr = rep["per_model"][m]["sensitivity_rounding"]["contrasts"]
        for key, name in (("natc1", "NAT-C1"), ("hb", "HB QA-QO"), ("provk", "Prov-K NAT-K-C1-K"), ("natcv", "NAT-CV")):
            put(f"trunc.{k}.{key}.kept", str(sr[name]["n_kept"]), f"{R}:per_model.{m}.sensitivity_rounding.contrasts.{name}.n_kept")
            mr = sr[name]["median_retained_fraction"]
            put(f"trunc.{k}.{key}.frac", f"{mr:.2f}", f"{R}:per_model.{m}.sensitivity_rounding.contrasts.{name}.median_retained_fraction")
    put("trunc.hf.n", str(rep["HF_sensitivity_rounding"]["n_shared"]), f"{R}:HF_sensitivity_rounding.n_shared")
    for m in MODELS:                                  # QA/QO realized target norms (each matched to its own arm's natural error)
        qa = {r["index"]: r for r in map(json.loads, (RUNS / m / "tierA" / "QA.jsonl").read_text().splitlines())}
        qo = {r["index"]: r for r in map(json.loads, (RUNS / m / "tierA" / "QO.jsonl").read_text().splitlines())}
        lr = np.mean([np.log(np.mean(qa[i]["target_norm_mean"]) / np.mean(qo[i]["target_norm_mean"])) for i in qa])
        put(f"qaqo.{SHORT[m]}.lognorm", num(float(lr)), f"{m}/tierA/QA.jsonl, QO.jsonl: mean log(mean target_norm_mean QA / QO)")
    # ranges over the four models (min, max by value), so that prose never orders endpoints by hand
    P = rep["per_model"]
    rng_("rng.sK", {m: cal[m]["scale_factor_K"] for m in MODELS}, lambda x: un(x, 2), "calibration.json:scale_factor_K")
    rng_("rng.margin", {m: cal[m]["margin_log_js"] for m in MODELS}, lambda x: un(x, 3), "calibration.json:margin_log_js")
    rng_("rng.E8", {m: cal[m]["E8"]["median"] for m in MODELS}, lambda x: un(x, 2), "calibration.json:E8.median")
    rng_("rng.rhoVK", {m: cal[m]["rho_V"] / cal[m]["rho_K"] for m in MODELS}, lambda x: f"{x:.1f}", "calibration.json:rho_V/rho_K")
    rng_("rng.b", {m: cal[m]["dose_slope_b_hat"] for m in MODELS}, lambda x: un(x, 2), "calibration.json:dose_slope_b_hat")
    rng_("rng.craw", {m: cal[m]["cv_raw_median"] for m in MODELS}, lambda x: un(x, 3), "calibration.json:cv_raw_median")
    for key, name in (("provk", "Prov-K NAT-K-C1-K"), ("provv", "Prov-V NAT-V-C1-V"), ("hb", "HB QA-QO"), ("hd", "HD K*-V*"), ("he", "HE INH-LOC")):
        rng_(f"rng.{key}", {m: P[m]["family"][name]["estimate"] for m in MODELS}, num, f"{R}:per_model.*.family.{name}.estimate")
    rng_("rng.hbratio", {m: float(np.exp(P[m]["family"]["HB QA-QO"]["estimate"])) for m in MODELS}, lambda x: un(x, 0),
         f"exp of {R}:per_model.*.family.HB QA-QO.estimate")
    rng_("rng.nkv", {m: P[m]["descriptive"]["NAT-K-NAT-V"]["estimate"] for m in MODELS}, num, f"{R}:per_model.*.descriptive.NAT-K-NAT-V.estimate")
    rng_("rng.c2c1", {m: P[m]["descriptive"]["C2-C1"]["estimate"] for m in MODELS}, num, f"{R}:per_model.*.descriptive.C2-C1.estimate")
    rng_("rng.nonadd", {m: P[m]["descriptive"]["nonadditivity m_NAT/(m_LOC+m_INH)"]["median"] for m in MODELS}, un,
         f"{R}:per_model.*.descriptive.nonadditivity.median")
    E = exp["X1"]["per_model"]
    rng_("rng.x1wK", {m: E[m]["mean_wK"] for m in MODELS}, lambda x: un(x, 2), "exploratory_v3.json:X1.per_model.*.mean_wK")
    rng_("rng.x1d", {m: E[m]["obs_minus_pred"]["estimate"] for m in MODELS}, num, "exploratory_v3.json:X1.per_model.*.obs_minus_pred.estimate")
    rng_("rng.x1rho", {m: E[m]["spearman_pred_obs"]["rho"] for m in MODELS}, lambda x: num(x, 2), "exploratory_v3.json:X1.per_model.*.spearman_pred_obs.rho")
    rng_("rng.x2c", {m: exp["X2"]["X2c D_comb - D_logit"]["per_model"][m]["estimate"] for m in MODELS}, num,
         "exploratory_v3.json:X2.X2c.per_model.*.estimate")
    rng_("rng.qaqo", {m: float(V[f"qaqo.{SHORT[m]}.lognorm"].strip("$")) for m in MODELS}, num, "qaqo.*.lognorm (rounded)")
    rng_("rng.rdevmaxllama", {m: float(max(v["max"] for v in P[m]["realized_norm_deviation"].values() if v["n_slots"]))
                              for m in ("llama-base", "llama-instruct")}, lambda x: f"{x:.3f}", f"{R}:per_model.llama-*.realized_norm_deviation.*.max")
    rng_("rng.nkvratio", {m: float(np.exp(-P[m]["descriptive"]["NAT-K-NAT-V"]["estimate"])) for m in MODELS}, lambda x: f"{x:.1f}",
         f"exp(-{R}:per_model.*.descriptive.NAT-K-NAT-V.estimate)")
    rng_("rng.provkratio", {m: float(np.exp(-P[m]["family"]["Prov-K NAT-K-C1-K"]["estimate"])) for m in MODELS}, lambda x: f"{x:.1f}",
         f"exp(-{R}:per_model.*.family.Prov-K NAT-K-C1-K.estimate)")
    return V


QWEN, LLAMA = ("qwen-base", "qwen-instruct"), ("llama-base", "llama-instruct")


def linear_weight_shares(m):
    """Share of the NF4-quantized linear weights (q, k, v, o, gate, up, down; embeddings, norms, head and biases stay FP16) that sit
    in k_proj and v_proj, i.e. the weights LOC quantizes. Read from the model's config.json in the local Hugging Face cache, at the
    revision named in V3_PROTOCOL_LOCK.txt, whose sha256 must match the lock."""
    hf = Path.home() / ".cache" / "huggingface" / "hub"
    for line in (RV / "V3_PROTOCOL_LOCK.txt").read_text().splitlines():
        if line.startswith(f"MODEL {m} ") and line.endswith("config.json"):
            repo, rev = line.split()[2].rstrip(":").split("@")
            sha = line.split()[3]
            break
    else:
        sys.exit(f"V3_PROTOCOL_LOCK.txt: no config.json line for {m}")
    raw = (hf / ("models--" + repo.replace("/", "--")) / "snapshots" / rev / "config.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha:
        sys.exit(f"config.json of {m} does not match the lock")
    c = json.loads(raw)
    h, nq, nkv, n_i = c["hidden_size"], c["num_attention_heads"], c["num_key_value_heads"], c["intermediate_size"]
    hd = c.get("head_dim") or h // nq
    kv = 2 * h * nkv * hd
    total = 2 * h * nq * hd + kv + 3 * h * n_i                                                  # per layer; the layer count cancels
    return {"loc": kv / total, "inh": 1 - kv / total}
X6ARMS = [("NAT", "nat"), ("C1", "c1"), ("C2", "c2"), ("QA", "qa"), ("QO", "qo"), ("CV", "cv"), ("NAT-K", "natk"), ("NAT-V", "natv"),
          ("NAT-K*", "relk"), ("NAT-V*", "relv"), ("C1-K", "c1k"), ("C1-V", "c1v"), ("LOC", "loc"), ("INH", "inh")]
X4PRED = [("P_norm", "norm"), ("P_logit", "logit"), ("P_val", "val"), ("P_anormK", "anormk")]


def numbers_x(cal):
    """Exploratory X4-X12 (plan EXPLORATORY_PLAN_V3.md) and X9 (FOLLOWUP_X9.md), the X9 run summary, and the earlier E1/E1L flip rates."""
    S48, S10, S11, S12, S9 = "exploratory_v3_x4x8.json", "exploratory_v3_x10.json", "exploratory_v3_x11.json", "exploratory_v3_x12.json", "x9_results.json"
    X = json.loads((P5 / S48).read_text())
    X10 = json.loads((P5 / S10).read_text())["per_model"]
    X11 = json.loads((P5 / S11).read_text())["per_model"]
    X12 = json.loads((P5 / S12).read_text())
    X9 = json.loads((P7 / S9).read_text())
    if any(v != "complete" for v in X9["status"].values()):
        sys.exit("x9_results.json: not every model is complete")
    X9 = X9["per_model"]
    for m in MODELS:
        k = SHORT[m]
        for p, pk in X4PRED:                                                                      # X4 Spearman levels
            v = X["X4"]["per_model"][m][p]
            put(f"x4.{k}.{pk}", num(v["mean"]), f"{S48}:X4.per_model.{m}.{p}.mean")
            put(f"x4.{k}.{pk}.ci", ci(v["ci95"]), f"{S48}:X4.per_model.{m}.{p}.ci95")
        v = X["X5"]["per_model"][m]                                                               # X5 P_logit - P_anormK
        put(f"x5.{k}.est", num(v["estimate"]), f"{S48}:X5.per_model.{m}.estimate")
        put(f"x5.{k}.ci", ci(v["ci95"]), f"{S48}:X5.per_model.{m}.ci95")
        put(f"x5.{k}.holm", pv(v["holm_p"]), f"{S48}:X5.per_model.{m}.holm_p")
        for arm, ak in X6ARMS:                                                                    # X6 absolute JS
            v = X["X6"]["per_model"][m][arm]
            put(f"x6.{k}.{ak}.geo", sci3(v["geo_mean"]), f"{S48}:X6.per_model.{m}.{arm}.geo_mean")
            put(f"x6.{k}.{ak}.arith", sci3(v["arith_mean"]), f"{S48}:X6.per_model.{m}.{arm}.arith_mean")
        a = X["X7"]["X7a diagnostics (locked H-C, P_logit)"][m]                                  # X7 H-C diagnostics
        put(f"x7a.{k}.b1ci", ci(a["beta1_ci95"]), f"{S48}:X7.X7a.{m}.beta1_ci95")
        put(f"x7a.{k}.xbar", num(a["x_bar"]), f"{S48}:X7.X7a.{m}.x_bar")
        put(f"x7a.{k}.xbarci", ci(a["x_bar_ci95"]), f"{S48}:X7.X7a.{m}.x_bar_ci95")
        put(f"x7a.{k}.gamma", num(a["gamma"]), f"{S48}:X7.X7a.{m}.gamma")
        put(f"x7a.{k}.gammaci", ci(a["gamma_ci95"]), f"{S48}:X7.X7a.{m}.gamma_ci95")
        put(f"x7a.{k}.neg", un(a["share_steps_x_negative"], 4), f"{S48}:X7.X7a.{m}.share_steps_x_negative")
        put(f"x7a.{k}.nneg", f"{round(a['share_steps_x_negative'] * a['n_steps']):,}".replace(",", "{,}"),
            f"{S48}:X7.X7a.{m}.share_steps_x_negative x n_steps")
        put(f"x7a.{k}.nsteps", f"{a['n_steps']:,}".replace(",", "{,}"), f"{S48}:X7.X7a.{m}.n_steps")
        put(f"x7a.{k}.xval", num(a["x_bar_val"]), f"{S48}:X7.X7a.{m}.x_bar_val")
        for part, key in (("b", "X7b joint rank-based (P_logit, P_val)"), ("c", "X7c rank-based P_logit only (control)")):
            b = X["X7"][key][m]
            put(f"x7{part}.{k}.b1", num(b["beta1"]), f"{S48}:X7.{key}.{m}.beta1")
            put(f"x7{part}.{k}.b1ci", ci(b["beta1_ci95"]), f"{S48}:X7.{key}.{m}.beta1_ci95")
            put(f"x7{part}.{k}.f", num(b["fraction"], 2), f"{S48}:X7.{key}.{m}.fraction")
            put(f"x7{part}.{k}.fci", ci(b["fraction_ci95"], 2), f"{S48}:X7.{key}.{m}.fraction_ci95")
            put(f"x7{part}.{k}.band", b["band"], f"{S48}:X7.{key}.{m}.band")
        v = X["X8"]["per_model"][m]                                                               # X8 origin shares
        for arm in ("INH", "LOC"):
            put(f"x8.{k}.{arm.lower()}", f"{100 * v[arm]['geo_ratio']:.0f}", f"{S48}:X8.per_model.{m}.{arm}.geo_ratio x100")
            put(f"x8.{k}.{arm.lower()}ci", "$[" + ", ".join(f"{100 * x:.0f}" for x in v[arm]["geo_ratio_ci95"]) + "]$",
                f"{S48}:X8.per_model.{m}.{arm}.geo_ratio_ci95 x100")
        put(f"x8.{k}.share", f"{100 * v['INH_share_of_INH_plus_LOC']:.0f}", f"{S48}:X8.per_model.{m}.INH_share_of_INH_plus_LOC x100")
        lw = linear_weight_shares(m)                                                              # LOC's share of the NF4 linear weights
        put(f"arch.{k}.locpct", f"{100 * lw['loc']:.1f}", f"locked config.json of {m}: (k_proj + v_proj weights) / all NF4 linear weights x100")
        put(f"arch.{k}.potency", un(v["LOC"]["geo_ratio"] / lw["loc"], 0),
            f"{S48}:X8.per_model.{m}.LOC.geo_ratio / LOC share of NF4 linear weights")
        for p in ("K", "V"):                                                                      # X10 path scaling
            r = X10[m][p]
            put(f"x10.{k}.{p}.robs", un(r["R_obs"], 1 if r["R_obs"] > 10 else 2), f"{S10}:per_model.{m}.{p}.R_obs")
            put(f"x10.{k}.{p}.rpred", un(r["R_pred"], 2), f"{S10}:per_model.{m}.{p}.R_pred")
            put(f"x10.{k}.{p}.E", num(r["excess_E"], 2), f"{S10}:per_model.{m}.{p}.excess_E")
            put(f"x10.{k}.{p}.Eci", ci(r["excess_E_ci95"], 2), f"{S10}:per_model.{m}.{p}.excess_E_ci95")
            put(f"x10.{k}.{p}.ratio", un(r["R_obs_over_R_pred"], 2 if r["R_obs_over_R_pred"] < 10 else 1), f"{S10}:per_model.{m}.{p}.R_obs_over_R_pred")
        r = X11[m]                                                                                # X11 H-B benchmark
        put(f"x11.{k}.B", un(r["B"], 1), f"{S11}:per_model.{m}.B")
        put(f"x11.{k}.Bq", un(r["B_q"], 1), f"{S11}:per_model.{m}.B_q")
        put(f"x11.{k}.B1", un(r["B_1"], 1), f"{S11}:per_model.{m}.B_1")
        put(f"x11.{k}.dil", un(r["dilution_gC1V_over_gQO"], 2), f"{S11}:per_model.{m}.dilution_gC1V_over_gQO")
        put(f"x11.{k}.dilci", "$[" + ", ".join(f"{x:.2f}" for x in r["dilution_ci95"]) + "]$", f"{S11}:per_model.{m}.dilution_ci95")
        r = X9[m]                                                                                 # X9 (a)-(d)
        a = r["a"]
        put(f"x9a.{k}.A", num(a["test"]["estimate"]), f"{S9}:per_model.{m}.a.test.estimate")
        put(f"x9a.{k}.Aci", ci(a["test"]["ci95"]), f"{S9}:per_model.{m}.a.test.ci95")
        put(f"x9a.{k}.holm", pv(a["holm_p"]), f"{S9}:per_model.{m}.a.holm_p")
        put(f"x9a.{k}.verdict", a["verdict"], f"{S9}:per_model.{m}.a.verdict")
        put(f"x9a.{k}.F", un(a["F_mean"], 3), f"{S9}:per_model.{m}.a.F_mean")
        put(f"x9a.{k}.F0", un(a["F_per_layer"][0]["F"], 3), f"{S9}:per_model.{m}.a.F_per_layer[0].F")
        sp = a["spearman_lnF_vs_locked_theta_NATK_C1K"]
        put(f"x9a.{k}.rho", num(sp["rho"], 2), f"{S9}:per_model.{m}.a.spearman_lnF_vs_locked_theta_NATK_C1K.rho")
        put(f"x9a.{k}.rhoci", ci(sp["ci95"], 2), f"{S9}:per_model.{m}.a.spearman_lnF_vs_locked_theta_NATK_C1K.ci95")
        b = r["b"]
        put(f"x9b.{k}.G", num(b["G"], 2), f"{S9}:per_model.{m}.b.G")
        put(f"x9b.{k}.Gci", ci(b["G_ci95"], 2), f"{S9}:per_model.{m}.b.G_ci95")
        put(f"x9b.{k}.verdict", b["verdict"], f"{S9}:per_model.{m}.b.verdict")
        put(f"x9b.{k}.EK", num(b["E_Kstar"], 2), f"{S9}:per_model.{m}.b.E_Kstar")
        put(f"x9b.{k}.EL0", num(b["E_L0"], 2), f"{S9}:per_model.{m}.b.E_L0")
        put(f"x9b.{k}.EL0ci", ci(b["E_L0_ci95"], 2), f"{S9}:per_model.{m}.b.E_L0_ci95")
        c = r["c"]["P_logit"]
        put(f"x9c.{k}.drho", num(c["delta_rho"]["mean"], 4), f"{S9}:per_model.{m}.c.P_logit.delta_rho.mean")
        put(f"x9c.{k}.drhoci", ci(c["delta_rho"]["ci95"], 4), f"{S9}:per_model.{m}.c.P_logit.delta_rho.ci95")
        put(f"x9c.{k}.D", num(c["D"]["mean"]), f"{S9}:per_model.{m}.c.P_logit.D.mean")
        put(f"x9c.{k}.D0", num(c["D_no_l0"]["mean"]), f"{S9}:per_model.{m}.c.P_logit.D_no_l0.mean")
        d = [x["median"] for x in r["d"]["per_layer"]]
        put(f"x9d.{k}.L0", un(d[0], 0 if d[0] >= 10 else 1), f"{S9}:per_model.{m}.d.per_layer[0].median")
        put(f"x9d.{k}.restmax", un(max(d[1:]), 1), f"max over layers >= 1 of {S9}:per_model.{m}.d.per_layer.median")
        put(f"x9d.{k}.restmed", un(float(np.median(d[1:])), 2), f"median over layers >= 1 of {S9}:per_model.{m}.d.per_layer.median")
        big = [str(l) for l, x in enumerate(d) if l >= 1 and x > 5]
        put(f"x9d.{k}.biglayers", (", ".join(big[:-1]) + " and " + big[-1]) if len(big) > 1 else (big[0] if big else "none"),
            f"layers >= 1 with {S9}:per_model.{m}.d.per_layer.median > 5")
        put(f"x9d.{k}.sK", un(r["d"]["s_K"], 2), f"{S9}:per_model.{m}.d.s_K")
    put("x5.pooled.est", num(X["X5"]["pooled_mean_over_models"]["estimate"]), f"{S48}:X5.pooled_mean_over_models.estimate")
    put("x5.pooled.ci", ci(X["X5"]["pooled_mean_over_models"]["ci95"]), f"{S48}:X5.pooled_mean_over_models.ci95")
    d = X["X7"]["X7d decision pair, joint"]
    put("x7d.FJ", num(d["F_J"], 2), f"{S48}:X7.X7d.F_J")
    put("x7d.FJci", ci(d["F_J_ci95"], 2), f"{S48}:X7.X7d.F_J_ci95")
    put("x7d.band", d["band"], f"{S48}:X7.X7d.band")
    for key, name in (("q", "Qwen tuning effect (instruct - base)"), ("l", "Llama tuning effect (instruct - base)")):
        v = X12["effects"][name]
        put(f"x12.{key}.est", num(v["estimate"]), f"{S12}:effects.{name}.estimate")
        put(f"x12.{key}.ci", ci(v["ci95"]), f"{S12}:effects.{name}.ci95")
        put(f"x12.{key}.holm", pv(v["holm_p"]), f"{S12}:effects.{name}.holm_p")
    XP = X["X4"]["per_model"]
    rng_("rng.x4logitqwen", {m: XP[m]["P_logit"]["mean"] for m in QWEN}, num, f"{S48}:X4.per_model.qwen-*.P_logit.mean")
    rng_("rng.x4logitllama", {m: XP[m]["P_logit"]["mean"] for m in LLAMA}, num, f"{S48}:X4.per_model.llama-*.P_logit.mean")
    rng_("rng.x4norm", {m: XP[m]["P_norm"]["mean"] for m in MODELS}, num, f"{S48}:X4.per_model.*.P_norm.mean")
    rng_("rng.x4all", {f"{m}/{p}": XP[m][p]["mean"] for m in MODELS for p, _ in X4PRED}, num, f"{S48}:X4.per_model.*.*.mean")
    rng_("rng.x5", {m: X["X5"]["per_model"][m]["estimate"] for m in MODELS}, num, f"{S48}:X5.per_model.*.estimate")
    rng_("rng.x6nat", {m: X["X6"]["per_model"][m]["NAT"]["geo_mean"] for m in MODELS}, sci3, f"{S48}:X6.per_model.*.NAT.geo_mean")
    rng_("rng.x7xbar", {m: X["X7"]["X7a diagnostics (locked H-C, P_logit)"][m]["x_bar"] for m in MODELS}, num, f"{S48}:X7.X7a.*.x_bar")
    rng_("rng.x7neg", {m: X["X7"]["X7a diagnostics (locked H-C, P_logit)"][m]["share_steps_x_negative"] for m in MODELS}, lambda x: un(x, 4),
         f"{S48}:X7.X7a.*.share_steps_x_negative")
    rng_("rng.x8inh", {m: X["X8"]["per_model"][m]["INH"]["geo_ratio"] for m in MODELS}, lambda x: f"{100 * x:.0f}", f"{S48}:X8.per_model.*.INH.geo_ratio x100")
    rng_("rng.x8loc", {m: X["X8"]["per_model"][m]["LOC"]["geo_ratio"] for m in MODELS}, lambda x: f"{100 * x:.0f}", f"{S48}:X8.per_model.*.LOC.geo_ratio x100")
    rng_("rng.x8share", {m: X["X8"]["per_model"][m]["INH_share_of_INH_plus_LOC"] for m in MODELS}, lambda x: f"{100 * x:.0f}",
         f"{S48}:X8.per_model.*.INH_share_of_INH_plus_LOC x100")
    rng_("rng.archloc", {m: linear_weight_shares(m)["loc"] for m in MODELS}, lambda x: f"{100 * x:.1f}",
         "locked config.json: (k_proj + v_proj weights) / all NF4 linear weights x100")
    rng_("rng.archpotency", {m: X["X8"]["per_model"][m]["LOC"]["geo_ratio"] / linear_weight_shares(m)["loc"] for m in MODELS},
         lambda x: un(x, 0), f"{S48}:X8.per_model.*.LOC.geo_ratio / LOC share of NF4 linear weights")
    rng_("rng.x10Kqwen", {m: X10[m]["K"]["R_obs_over_R_pred"] for m in QWEN}, lambda x: un(x, 0), f"{S10}:per_model.qwen-*.K.R_obs_over_R_pred")
    rng_("rng.x10Krobsqwen", {m: X10[m]["K"]["R_obs"] for m in QWEN}, lambda x: un(x, 0), f"{S10}:per_model.qwen-*.K.R_obs")
    rng_("rng.x10Kllama", {m: X10[m]["K"]["R_obs_over_R_pred"] for m in LLAMA}, lambda x: un(x, 2), f"{S10}:per_model.llama-*.K.R_obs_over_R_pred")
    rng_("rng.x10V", {m: X10[m]["V"]["R_obs_over_R_pred"] for m in MODELS}, lambda x: un(x, 2), f"{S10}:per_model.*.V.R_obs_over_R_pred")
    rng_("rng.x10pred", {f"{m}/K": X10[m]["K"]["R_pred"] for m in MODELS}, lambda x: un(x, 1), f"{S10}:per_model.*.K.R_pred")
    rng_("rng.x11B", {m: X11[m]["B"] for m in MODELS}, lambda x: un(x, 0), f"{S11}:per_model.*.B")
    rng_("rng.x11dil", {m: X11[m]["dilution_gC1V_over_gQO"] for m in MODELS}, lambda x: un(x, 2), f"{S11}:per_model.*.dilution_gC1V_over_gQO")
    rng_("rng.x9aF", {m: X9[m]["a"]["F_mean"] for m in MODELS}, lambda x: un(x, 3), f"{S9}:per_model.*.a.F_mean")
    rng_("rng.x9aF0qwen", {m: X9[m]["a"]["F_per_layer"][0]["F"] for m in QWEN}, lambda x: un(x, 3), f"{S9}:per_model.qwen-*.a.F_per_layer[0].F")
    rng_("rng.x9aA", {m: X9[m]["a"]["test"]["estimate"] for m in MODELS}, num, f"{S9}:per_model.*.a.test.estimate")
    rng_("rng.x9adeficit", {m: 1 - X9[m]["a"]["F_mean"] / 0.125 for m in MODELS}, lambda x: f"{100 * x:.1f}",
         f"1 - {S9}:per_model.*.a.F_mean / 0.125, x100")
    rng_("rng.x9arho", {m: X9[m]["a"]["spearman_lnF_vs_locked_theta_NATK_C1K"]["rho"] for m in MODELS}, lambda x: num(x, 2),
         f"{S9}:per_model.*.a.spearman.rho")
    rng_("rng.x9bGqwen", {m: X9[m]["b"]["G"] for m in QWEN}, lambda x: num(x, 2), f"{S9}:per_model.qwen-*.b.G")
    rng_("rng.x9bEL0qwen", {m: X9[m]["b"]["E_L0"] for m in QWEN}, lambda x: num(x, 2), f"{S9}:per_model.qwen-*.b.E_L0")
    rng_("rng.x9bEL0qwenratio", {m: float(np.exp(X9[m]["b"]["E_L0"])) for m in QWEN}, lambda x: un(x, 1), f"exp {S9}:per_model.qwen-*.b.E_L0")
    rng_("rng.x9cabs", {f"{m}/{p}": abs(X9[m]["c"][p]["delta_rho"]["mean"]) for m in MODELS for p in ("P_logit", "P_val", "P_anormK")},
         lambda x: un(x, 3), f"|{S9}:per_model.*.c.*.delta_rho.mean|")
    rng_("rng.x9dL0qwen", {m: X9[m]["d"]["per_layer"][0]["median"] for m in QWEN}, lambda x: un(x, 0), f"{S9}:per_model.qwen-*.d.per_layer[0].median")
    rng_("rng.x9dL0llama", {m: X9[m]["d"]["per_layer"][0]["median"] for m in LLAMA}, lambda x: un(x, 1), f"{S9}:per_model.llama-*.d.per_layer[0].median")
    rng_("rng.x9dbigqwen", {f"{m}/{l}": x["median"] for m in QWEN for l, x in enumerate(X9[m]["d"]["per_layer"]) if l >= 1 and x["median"] > 5},
         lambda x: un(x, 1), f"{S9}:per_model.qwen-*.d.per_layer[l>=1].median > 5")
    # the X9 run: GPU time (own ledger) and the bitwise reproduction gate
    led = [json.loads(x) for x in (RUNS9 / "gpu_ledger.jsonl").read_text().splitlines() if x.strip()]
    for m in MODELS:
        put(f"x9.gpu.{SHORT[m]}", f"{sum(x['seconds'] for x in led if x['block'] == 'x9:' + m) / 3600:.3f}", f"v3_runs_x9/gpu_ledger.jsonl blocks x9:{m}")
    tot9 = sum(x["seconds"] for x in led) / 3600
    put("x9.gpu.total", f"{tot9:.3f}", "sum v3_runs_x9/gpu_ledger.jsonl seconds / 3600")
    v3tot = sum(json.loads(x)["seconds"] for x in (RUNS / "gpu_ledger.jsonl").read_text().splitlines() if x.strip()) / 3600
    put("gpu.v3x9", f"{v3tot + tot9:.2f}", "v3 ledger + X9 ledger, hours")
    rep = [json.loads(x) for m in MODELS for x in (RUNS9 / m / "reproduction.jsonl").read_text().splitlines() if x.strip()]
    put("x9.repro.n", str(len(rep)), "v3_runs_x9/*/reproduction.jsonl count")
    mc = json.loads((P7 / "x9_mutation_check.json").read_text())
    put("x9.mut.caught", str(mc["caught"]), "v3/phase7/x9_mutation_check.json:caught")
    put("x9.mut.total", str(mc["total"]), "v3/phase7/x9_mutation_check.json:total")
    cv = json.loads((P7 / "x9_cpu_validation.json").read_text())
    put("x9.cpu.pass", str(cv["tiny_tests_total"] - len(cv["tiny_tests_failed"])), "v3/phase7/x9_cpu_validation.json: tiny tests passed")
    put("x9.cpu.total", str(cv["tiny_tests_total"]), "v3/phase7/x9_cpu_validation.json:tiny_tests_total")
    put("x9.repro.ok", str(sum(1 for r in rep if r["ok"] and r["history_matches_locked"] and all(r["bitwise_equal"].values()))),
        "v3_runs_x9/*/reproduction.jsonl: ok, history match, bitwise NAT and NAT-K*")
    numbers_x13(v3tot, tot9)
    # earlier studies: top-1 flip rate and mean JS of the teacher-forced arms (as in E1_REPORT.md / E1L_REPORT.md)
    for lab, d in (("e1", RV / "e1_runs" / "tierA"), ("e1l", RV / "e1l_runs" / "tierA")):
        for arm, ak in (("NAT", "nat"), ("C1_a1.0", "c1"), ("C1_a0.5", "c1half"), ("C1_a2.0", "c1two")):
            rs = [json.loads(x) for x in (d / f"{arm}.jsonl").read_text().splitlines() if x.strip()]
            put(f"{lab}.flip.{ak}", f"{100 * np.mean([np.mean(r['flip']) for r in rs]):.2f}", f"{d.parent.name}/tierA/{arm}.jsonl mean per-problem flip rate x100")
            put(f"{lab}.js.{ak}", sci3(float(np.mean([np.mean(r["js"]) for r in rs]))), f"{d.parent.name}/tierA/{arm}.jsonl mean per-problem mean JS")


def numbers_x13(v3tot, tot9):
    """X13 (FOLLOWUP_X13.md): run summary and results from v3/phase8/x13_results.json, the X13 ledger and records."""
    P8, RUNS13, S13 = RV / "v3" / "phase8", RV / "v3_runs_x13", "x13_results.json"
    O = json.loads((P8 / S13).read_text())
    if any(v != "complete" for v in O["status"].values()):
        sys.exit("x13_results.json: not every model is complete")
    X13 = O["per_model"]
    led = [json.loads(x) for x in (RUNS13 / "gpu_ledger.jsonl").read_text().splitlines() if x.strip()]
    for m in MODELS:
        put(f"x13.gpu.{SHORT[m]}", f"{sum(x['seconds'] for x in led if x['block'] == 'x13:' + m) / 3600:.3f}", f"v3_runs_x13/gpu_ledger.jsonl blocks x13:{m}")
    t13 = sum(x["seconds"] for x in led) / 3600
    put("x13.gpu.total", f"{t13:.3f}", "sum v3_runs_x13/gpu_ledger.jsonl seconds / 3600")
    put("gpu.grand", f"{v3tot + tot9 + t13:.2f}", "v3 ledger + X9 ledger + X13 ledger, hours")
    des = json.loads((RUNS13 / "design_x13.json").read_text())
    put("x13.n", str(des["n"]["qwen-base"]), "v3_runs_x13/design_x13.json:n (all models)")
    put("x13.proj.full", f"{des['projection_steps'][0]['projected_h']:.2f}", "design_x13.json:projection_steps[0]")
    put("x13.proj.final", f"{des['projection_steps'][-1]['projected_h']:.2f}", "design_x13.json:projection_steps[-1]")
    rep = [json.loads(x) for m in MODELS for x in (RUNS13 / m / "reproduction.jsonl").read_text().splitlines() if x.strip()]
    put("x13.repro.n", str(len(rep)), "v3_runs_x13/*/reproduction.jsonl count")
    put("x13.repro.ok", str(sum(1 for r in rep if r["ok"] and r["history_matches_locked"] and r["bitwise_equal"]["NAT"])),
        "v3_runs_x13/*/reproduction.jsonl: ok, history, NAT bitwise")
    mc = json.loads((P8 / "x13_mutation_check.json").read_text())
    put("x13.mut.caught", str(mc["caught"]), "v3/phase8/x13_mutation_check.json:caught")
    put("x13.mut.total", str(mc["total"]), "v3/phase8/x13_mutation_check.json:total")
    cv = json.loads((P8 / "x13_cpu_validation.json").read_text())
    put("x13.cpu.pass", str(cv["tiny_tests_total"] - len(cv["tiny_tests_failed"])), "x13_cpu_validation.json tiny passed")
    put("x13.cpu.total", str(cv["tiny_tests_total"]), "x13_cpu_validation.json:tiny_tests_total")
    nc_steps = nc_bad = 0
    for m in QWEN:
        for a in ("C1-nc", "C1-K-nc"):
            for x in (RUNS13 / m / f"{a}.jsonl").read_text().splitlines():
                if x.strip():
                    r = json.loads(x)
                    nc_steps += len(r["nc_ok"])
                    nc_bad += sum(1 for v in r["nc_ok"] if not v)
    put("x13.nc.steps", f"{nc_steps:,}".replace(",", "{,}"), "v3_runs_x13/qwen-*/C1*-nc.jsonl nc_ok count")
    put("x13.nc.bad", str(nc_bad), "v3_runs_x13/qwen-*/C1*-nc.jsonl nc_ok false count")
    for m in MODELS:
        k, r = SHORT[m], X13[m]
        for P in ("K", "V"):
            a, b = r["a"][P], r["b"][P]
            put(f"x13a.{k}.{P}", num(a["test"]["estimate"]), f"{S13}:per_model.{m}.a.{P}.test.estimate")
            put(f"x13a.{k}.{P}.ci", ci(a["test"]["ci95"]), f"{S13}:per_model.{m}.a.{P}.test.ci95")
            put(f"x13a.{k}.{P}.verdict", a["verdict"], f"{S13}:per_model.{m}.a.{P}.verdict")
            put(f"x13a.{k}.{P}.fac", un(float(np.exp(a["test"]["estimate"])), 1), f"exp {S13}:per_model.{m}.a.{P}.test.estimate")
            put(f"x13b.{k}.{P}", num(b["theta_cen"]["estimate"]), f"{S13}:per_model.{m}.b.{P}.theta_cen.estimate")
            put(f"x13b.{k}.{P}.ci", ci(b["theta_cen"]["ci95"]), f"{S13}:per_model.{m}.b.{P}.theta_cen.ci95")
            put(f"x13b.{k}.{P}.holm", pv(b["holm_p"]), f"{S13}:per_model.{m}.b.{P}.holm_p")
            put(f"x13b.{k}.{P}.dir", b["direction"], f"{S13}:per_model.{m}.b.{P}.direction")
            put(f"x13b.{k}.{P}.G", num(b["G"], 2), f"{S13}:per_model.{m}.b.{P}.G")
            put(f"x13b.{k}.{P}.Gci", ci(b["G_ci95"], 2), f"{S13}:per_model.{m}.b.{P}.G_ci95")
            put(f"x13b.{k}.{P}.band", b.get("band", "---"), f"{S13}:per_model.{m}.b.{P}.band")
            put(f"x13b.{k}.{P}.man", num(b["manipulation"]["estimate"], 2), f"{S13}:per_model.{m}.b.{P}.manipulation.estimate")
            put(f"x13b.{k}.{P}.manok", "yes" if b["manipulation_ok"] else "no", f"{S13}:per_model.{m}.b.{P}.manipulation_ok")
        c = r["c"]
        put(f"x13c.{k}.Dc", num(c["D_c"]["mean"]), f"{S13}:per_model.{m}.c.D_c.mean")
        put(f"x13c.{k}.Dcci", ci(c["D_c"]["ci95"]), f"{S13}:per_model.{m}.c.D_c.ci95")
        put(f"x13c.{k}.spc", num(c["Sp_P_logit_c"]["mean"]), f"{S13}:per_model.{m}.c.Sp_P_logit_c.mean")
        put(f"x13c.{k}.lnR", num(c["mean_ln_R"]["mean"], 2), f"{S13}:per_model.{m}.c.mean_ln_R.mean")
        put(f"x13c.{k}.R", un(float(np.exp(c["mean_ln_R"]["mean"])), 2), f"exp {S13}:per_model.{m}.c.mean_ln_R.mean")
        put(f"x13c.{k}.A", num(c["X9a_A_same_problems"]["mean"]), f"{S13}:per_model.{m}.c.X9a_A_same_problems.mean")
        put(f"x13c.{k}.rho", num(c["spearman_lnR_vs_locked_theta_NATK_C1K"], 2), f"{S13}:per_model.{m}.c.spearman_lnR_vs_locked_theta")
        put(f"x13c.{k}.lnR0", num(c["ln_R_per_layer"][0], 2), f"{S13}:per_model.{m}.c.ln_R_per_layer[0]")
    for m in QWEN:
        k = SHORT[m]
        for key, name in (("c1", "NAT-C1"), ("c1k", "NAT-K-C1-K")):
            d = X13[m]["d"][name]
            put(f"x13d.{k}.{key}", num(d["delta_theta"]["estimate"]), f"{S13}:per_model.{m}.d.{name}.delta_theta.estimate")
            put(f"x13d.{k}.{key}.ci", ci(d["delta_theta"]["ci95"]), f"{S13}:per_model.{m}.d.{name}.delta_theta.ci95")
            put(f"x13d.{k}.{key}.h", un(d["h"]), f"{S13}:per_model.{m}.d.{name}.h")
            put(f"x13d.{k}.{key}.verdict", d["verdict"], f"{S13}:per_model.{m}.d.{name}.verdict")
            put(f"x13d.{k}.{key}.gt5nc", str(d["steps_gt5_nc"]), f"{S13}:per_model.{m}.d.{name}.steps_gt5_nc")
            put(f"x13d.{k}.{key}.gt5lock", str(d["steps_gt5_locked"]), f"{S13}:per_model.{m}.d.{name}.steps_gt5_locked")
    Q = lambda f: {m: f(m) for m in QWEN}
    A_ = lambda f: {m: f(m) for m in MODELS}
    rng_("rng.x13fac", {f"{m}/{P}": float(np.exp(X13[m]["a"][P]["test"]["estimate"])) for m in MODELS for P in ("K", "V")}, lambda x: f"{x:.0f}",
         f"exp {S13}:per_model.*.a.*.test.estimate")
    rng_("rng.x13aL", {f"{m}/{P}": X13[m]["a"][P]["test"]["estimate"] for m in MODELS for P in ("K", "V")}, num, f"{S13}:per_model.*.a.*.test.estimate")
    rng_("rng.x13bV", A_(lambda m: X13[m]["b"]["V"]["theta_cen"]["estimate"]), num, f"{S13}:per_model.*.b.V.theta_cen.estimate")
    rng_("rng.x13GV", A_(lambda m: X13[m]["b"]["V"]["G"]), lambda x: un(x, 2), f"{S13}:per_model.*.b.V.G")
    rng_("rng.x13bKq", Q(lambda m: X13[m]["b"]["K"]["theta_cen"]["estimate"]), num, f"{S13}:per_model.qwen-*.b.K.theta_cen.estimate")
    rng_("rng.x13bKl", {m: X13[m]["b"]["K"]["theta_cen"]["estimate"] for m in LLAMA}, num, f"{S13}:per_model.llama-*.b.K.theta_cen.estimate")
    rng_("rng.x13holmKl", {m: X13[m]["b"]["K"]["holm_p"] for m in LLAMA}, lambda x: un(x, 3), f"{S13}:per_model.llama-*.b.K.holm_p")
    rng_("rng.x13man", {f"{m}/{P}": X13[m]["b"][P]["manipulation"]["estimate"] for m in MODELS for P in ("K", "V")}, lambda x: num(x, 2),
         f"{S13}:per_model.*.b.*.manipulation.estimate")
    rng_("rng.x13Dc", A_(lambda m: X13[m]["c"]["D_c"]["mean"]), num, f"{S13}:per_model.*.c.D_c.mean")
    rng_("rng.x13spcq", Q(lambda m: X13[m]["c"]["Sp_P_logit_c"]["mean"]), num, f"{S13}:per_model.qwen-*.c.Sp_P_logit_c.mean")
    rng_("rng.x13R", A_(lambda m: float(np.exp(X13[m]["c"]["mean_ln_R"]["mean"]))), lambda x: un(x, 2), f"exp {S13}:per_model.*.c.mean_ln_R.mean")
    rng_("rng.x13lnR", A_(lambda m: X13[m]["c"]["mean_ln_R"]["mean"]), lambda x: num(x, 2), f"{S13}:per_model.*.c.mean_ln_R.mean")
    rng_("rng.x13A", A_(lambda m: X13[m]["c"]["X9a_A_same_problems"]["mean"]), num, f"{S13}:per_model.*.c.X9a_A_same_problems.mean")
    rng_("rng.x13rho", A_(lambda m: X13[m]["c"]["spearman_lnR_vs_locked_theta_NATK_C1K"]), lambda x: num(x, 2), f"{S13}:per_model.*.c.spearman")
    rng_("rng.x13dabs", {f"{m}/{n}": abs(X13[m]["d"][n]["delta_theta"]["estimate"]) for m in QWEN for n in ("NAT-C1", "NAT-K-C1-K")},
         lambda x: un(x, 3), f"|{S13}:per_model.qwen-*.d.*.delta_theta.estimate|")
    rng_("rng.x13dciabs", {f"{m}/{n}/{j}": abs(X13[m]["d"][n]["delta_theta"]["ci95"][j]) for m in QWEN for n in ("NAT-C1", "NAT-K-C1-K") for j in (0, 1)},
         lambda x: un(x, 3), f"|{S13}:per_model.qwen-*.d.*.delta_theta.ci95|")
    rng_("rng.x13gt5lock", {f"{m}/{n}": X13[m]["d"][n]["steps_gt5_locked"] for m in QWEN for n in ("NAT-C1", "NAT-K-C1-K")}, str,
         f"{S13}:per_model.qwen-*.d.*.steps_gt5_locked")
    rng_("rng.x13nnpos", {f"{m}/{n}": 100 * X13[m]["d"][n]["signed_nearest"]["share_above_5pct"] for m in QWEN for n in ("NAT-C1", "NAT-K-C1-K")},
         lambda x: f"{x:.2f}", f"{S13}:per_model.qwen-*.d.*.signed_nearest.share_above_5pct x100")
    rng_("rng.x13nnneg", {f"{m}/{n}": 100 * X13[m]["d"][n]["signed_nearest"]["share_below_minus_5pct"] for m in QWEN for n in ("NAT-C1", "NAT-K-C1-K")},
         lambda x: f"{x:.2f}", f"{S13}:per_model.qwen-*.d.*.signed_nearest.share_below_minus_5pct x100")
    put("x13.readK", O["reading_b_K"], f"{S13}:reading_b_K")
    put("x13.readV", O["reading_b_V"], f"{S13}:reading_b_V")
    put("x13.ncgt5", str(sum(X13[m]["d"][n]["steps_gt5_nc"] for m in QWEN for n in ("NAT-C1", "NAT-K-C1-K"))), f"sum {S13}:per_model.qwen-*.d.*.steps_gt5_nc")


def table_x13():
    _xtable("x13.tex", "tab:x13",
            "X13 \\tagE{} (FOLLOWUP\\_X13; $n=$\\ \\vn{x13.n} problems per model after the pre-declared runtime reductions). (a) $L$: log of the "
            "natural common-mode share over the isotropic share (null-corrected); (b) $\\theta_{\\mathrm{cen}}$: centred arm minus the locked natural "
            "single-path arm, with the share $G$ of the locked provenance gap it closes, and the manipulation check (the same statistic $L$ for the "
            "centred arm, below $\\ln 2$ required); (c) $D_c=\\mathrm{Sp}(P_{\\mathrm{logit}}^c)-\\mathrm{Sp}(P_{\\mathrm{logit}})$ and the energy-weighted "
            "alignment $R$ (isotropic 1), descriptive; (d) the change of the Qwen isotropic contrasts under norm-corrected rounding, against the "
            "locked half-width $h$. Holm over the four models within each path. 95\\% intervals.",
            [("(a) Common mode", [
                ("$L$, keys; verdict", lambda k: f"\\vn{{x13a.{k}.K}} \\vn{{x13a.{k}.K.ci}}\\newline \\vn{{x13a.{k}.K.verdict}}"),
                ("$L$, values; verdict", lambda k: f"\\vn{{x13a.{k}.V}} \\vn{{x13a.{k}.V.ci}}\\newline \\vn{{x13a.{k}.V.verdict}}")]),
             ("(b) Common mode removed", [
                ("Values: $\\theta_{\\mathrm{cen}}$; Holm; direction", lambda k: f"\\vn{{x13b.{k}.V}} \\vn{{x13b.{k}.V.ci}}\\newline Holm \\vn{{x13b.{k}.V.holm}}; \\vn{{x13b.{k}.V.dir}}"),
                ("Values: $G$; band", lambda k: f"\\vn{{x13b.{k}.V.G}} \\vn{{x13b.{k}.V.Gci}}\\newline \\vn{{x13b.{k}.V.band}}"),
                ("Keys: $\\theta_{\\mathrm{cen}}$; Holm; direction", lambda k: f"\\vn{{x13b.{k}.K}} \\vn{{x13b.{k}.K.ci}}\\newline Holm \\vn{{x13b.{k}.K.holm}}; \\vn{{x13b.{k}.K.dir}}"),
                ("Manipulation check $L$ (K; V); passed", lambda k: f"\\vn{{x13b.{k}.K.man}}; \\vn{{x13b.{k}.V.man}}; \\vn{{x13b.{k}.K.manok}}/\\vn{{x13b.{k}.V.manok}}")]),
             ("(c) Descriptive", [
                ("$D_c$", lambda k: f"\\vn{{x13c.{k}.Dc}} \\vn{{x13c.{k}.Dcci}}"),
                ("$\\mathrm{Sp}(P_{\\mathrm{logit}}^c)$", lambda k: f"\\vn{{x13c.{k}.spc}}"),
                ("$R$ ($\\ln R$); X9a $A$ on the same problems", lambda k: f"\\vn{{x13c.{k}.R}} (\\vn{{x13c.{k}.lnR}}); \\vn{{x13c.{k}.A}}"),
                ("Spearman of $\\ln R_i$ with $\\theta_i$(NAT-K, C1-K)", lambda k: f"\\vn{{x13c.{k}.rho}}")]),
             ("(d) Norm-corrected rounding (Qwen)", [
                ("NAT$-$C1: $\\Delta\\theta$; $h$; verdict", lambda k: f"\\vn{{x13d.{k}.c1}} \\vn{{x13d.{k}.c1.ci}}; \\vn{{x13d.{k}.c1.h}}\\newline \\vn{{x13d.{k}.c1.verdict}}" if k in ("qb", "qi") else "---"),
                ("NAT-K$-$C1-K: $\\Delta\\theta$; $h$; verdict", lambda k: f"\\vn{{x13d.{k}.c1k}} \\vn{{x13d.{k}.c1k.ci}}; \\vn{{x13d.{k}.c1k.h}}\\newline \\vn{{x13d.{k}.c1k.verdict}}" if k in ("qb", "qi") else "---"),
                ("Steps with a slot $>5\\%$: locked C1 / C1-nc", lambda k: f"\\vn{{x13d.{k}.c1.gt5lock}} / \\vn{{x13d.{k}.c1.gt5nc}}" if k in ("qb", "qi") else "---")])])


# ------------------------------------------------------------------ tables
def boxed(path):
    """Wraps every tabular of a generated table file in a max-width adjustbox (no table may exceed the text width)."""
    s = path.read_text(encoding="utf-8")
    if "adjustbox" not in s:
        s = s.replace(r"\begin{tabular}", r"\begin{adjustbox}{max width=\linewidth}\begin{tabular}").replace(
            r"\end{tabular}", r"\end{tabular}\end{adjustbox}")
        path.write_text(s, encoding="utf-8")


def tex_numbers():
    L = ["% generated by make_v3.py; do not edit", "\\makeatletter",
         "\\providecommand{\\vn}[1]{\\ifcsname vn@#1\\endcsname\\csname vn@#1\\endcsname\\else\\textbf{\\textcolor{red}{??#1}}\\fi}"]
    for k, v in V.items():
        L.append(f"\\expandafter\\def\\csname vn@{k}\\endcsname{{{v}}}")
    L.append("\\makeatother")
    (TAB / "numbers_v3.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    (TAB / "numbers_v3_sources.json").write_text(json.dumps({k: {"value": V[k], "source": SRC[k]} for k in V}, indent=1), encoding="utf-8")


def cell(key, k, holm=True, verdict=True):
    s = f"\\vn{{{key}.{k}.est}} \\vn{{{key}.{k}.ci}}"
    tail = []
    if holm:
        tail.append(f"Holm \\vn{{{key}.{k}.holm}}")
    if verdict:
        tail.append(f"\\vn{{{key}.{k}.verdict}}")
    return s + ("\\newline " + "; ".join(tail) if tail else "")


def table_main():
    ks = [SHORT[m] for m in MODELS]
    hdr = " & ".join(NAME[m] for m in MODELS)
    rows = []
    rows.append("\\multicolumn{5}{@{}l}{\\emph{Key--value asymmetry (Section~\\ref{sec:res:kv})}} \\\\")
    rows.append("Provenance K: NAT-K$-$C1-K & " + " & ".join(cell("provk", k) for k in ks) + " \\\\")
    rows.append("Provenance V: NAT-V$-$C1-V & " + " & ".join(cell("provv", k) for k in ks) + " \\\\")
    rows.append("H-D: REL-K$-$REL-V & " + " & ".join(cell("hd", k) for k in ks) + " \\\\")
    rows.append("\\midrule\\multicolumn{5}{@{}l}{\\emph{Four case studies (Section~\\ref{sec:res:cases}; the 2$\\times$2 is in Table~\\ref{tab:hf})}} \\\\")
    rows.append("NAT$-$C1 & " + " & ".join(cell("natc1", k) for k in ks) + " \\\\")
    rows.append("\\midrule\\multicolumn{5}{@{}l}{\\emph{Origin (Section~\\ref{sec:res:origin})}} \\\\")
    rows.append("H-E: INH$-$LOC & " + " & ".join(cell("he", k) for k in ks) + " \\\\")
    rows.append("\\midrule\\multicolumn{5}{@{}l}{\\emph{Storage-derived predictors and controls (Section~\\ref{sec:res:pred})}} \\\\")
    rows.append("H-A: $D_m$ (Holm over the four models) & " + " & ".join(
        f"\\vn{{ha.{k}.est}} \\vn{{ha.{k}.ci}}\\newline Holm \\vn{{ha.{k}.holm}}; \\vn{{ha.{k}.word}}" for k in ks) + " \\\\")
    rows.append("H-B: QA$-$QO (positive control) & " + " & ".join(cell("hb", k) for k in ks) + " \\\\")
    t = ["\\begin{table}[!htbp]", "\\centering\\scriptsize", "\\setlength{\\tabcolsep}{3pt}",
         "\\begin{tabular}{@{}p{2.9cm}p{2.75cm}p{2.75cm}p{2.75cm}p{2.75cm}@{}}", "\\toprule",
         f"Test & {hdr} \\\\", "\\midrule", *rows, "\\bottomrule", "\\end{tabular}",
         "\\caption{Locked tests per model \\tagC{}, in the order of Table~\\ref{tab:hyp} (generated by \\texttt{make\\_v3.py} from the locked "
         "analysis output). Log mean-JS contrasts unless stated (positive: first arm more disruptive); 95\\% problem-clustered bootstrap intervals. "
         "Verdict words (Section~\\ref{sec:rules}) use the unadjusted 95\\% interval; ``Holm'' is the $p$ adjusted within the model's family "
         "$\\mathcal{F}_m$ (H-A: within the family $\\mathcal{F}_{\\mathrm{A}}$ of the four models). $n=200$ problems per model. The locked "
         "adjustment test H-C is not identified and is reported only in Table~\\ref{tab:hc} (Section~\\ref{sec:res:hc}).}",
         "\\label{tab:v3main}", "\\end{table}"]
    (TAB / "main.tex").write_text("\n".join(t) + "\n", encoding="utf-8")


def table_hf():
    t = ["\\begin{table}[!htbp]", "\\centering\\small", "\\begin{tabular}{@{}l l l l l l l@{}}", "\\toprule",
         "Contrast & Estimate & 95\\% CI & 90\\% CI & Holm $p$ & Verdict & Pre-declared prediction \\\\", "\\midrule"]
    for key, name, pred in (("fam", "Family: $\\tfrac12[(c+d)-(a+b)]$", "$>0$: \\vn{hf.word.fam}"),
                            ("tun", "Tuning: $\\tfrac12[(b+d)-(a+c)]$", "$>0$: \\vn{hf.word.tun}"),
                            ("int", "Interaction: $\\tfrac12[(a-b)-(c-d)]$", "two-sided: additivity \\vn{hf.word.add}")):
        t.append(f"{name} & \\vn{{hf.{key}.est}} & \\vn{{hf.{key}.ci}} & \\vn{{hf.{key}.ci90}} & \\vn{{hf.{key}.holm}} & \\vn{{hf.{key}.verdict}} & {pred} \\\\")
    t += ["\\midrule",
          "\\multicolumn{7}{@{}l}{SD of $\\theta_{m,i}$: Qwen-base \\vn{hf.sd.qb}, Qwen-instruct \\vn{hf.sd.qi}, Llama-base \\vn{hf.sd.lb}, "
          "Llama-instruct \\vn{hf.sd.li} (guard limit \\vn{hf.guard})} \\\\",
          "\\bottomrule", "\\end{tabular}",
          "\\caption{H-F \\tagC{}: family $\\mathcal{F}_{2\\times2}$ over the per-problem $\\theta_{m,i}=\\log m_i^{\\mathrm{NAT}}-\\log m_i^{\\mathrm{C1}}$, with "
          "$a=$ Qwen-base, $b=$ Qwen-instruct, $c=$ Llama-base, $d=$ Llama-instruct, paired over the same 200 problems; Holm over the three "
          "contrasts; equivalence margin $\\pm$\\vn{hf.margin} (mean of the four calibrated margins). One model per cell: the contrasts compare "
          "four specific models.}", "\\label{tab:hf}", "\\end{table}"]
    (TAB / "hf.tex").write_text("\n".join(t) + "\n", encoding="utf-8")


def table_hc(rep):
    t = ["\\begin{table}[!htbp]", "\\centering\\scriptsize", "\\begin{tabular}{@{}l l l l l l@{}}", "\\toprule",
         "Model & $\\beta_0$ [95\\% CI] & $\\beta_1$ & $S_m$ [95\\% CI] & $f_m$ [95\\% CI] & Band \\\\", "\\midrule"]
    for m in MODELS:
        k = SHORT[m]
        t.append(f"{NAME[m]} & \\vn{{hc.{k}.b0}} \\vn{{hc.{k}.b0ci}} & \\vn{{hc.{k}.b1}} & \\vn{{hc.{k}.S}} \\vn{{hc.{k}.Sci}} & "
                 f"\\vn{{hc.{k}.f}} \\vn{{hc.{k}.fci}} & \\vn{{hc.{k}.band}} \\\\")
    t += ["\\midrule", "Pair (first minus second) & $\\Gamma_0$ [95\\% CI] & $\\Gamma_1$ [95\\% CI] & & $f_\\times$ [95\\% CI] & Band \\\\", "\\midrule"]
    x = rep["HC_cross"]
    for pair, v in x.items():
        a, b = pair.split(" vs ")
        role = " (decision rule)" if v["role"] == "decision rule" else ""
        Fs = f"{ci(v['F_ci95'], 2)}" if v["precondition_met"] else "---"
        Fv = num(v["F"], 2) if v["precondition_met"] else ""
        t.append(f"{NAME[a]} vs {NAME[b]}{role} & {num(v['g0'])} {ci(v['g0_ci95'])} & {num(v['g1'])} {ci(v['g1_ci95'])} & & {Fv} {Fs} & {v['band']} \\\\")
        SRC[f"table hc_cross {pair}"] = f"v3_report_locked.json:HC_cross.{pair}"
    t += ["\\bottomrule", "\\end{tabular}",
          "\\caption{H-C \\tagC{}, reported for completeness; the adjustment is not identified (Section~\\ref{sec:res:hc}): $x$ is negative at "
          "every scored step, so $\\beta_1$, the intercept at $x=0$, lies outside the observed range, and the bands and $p$-values below carry no "
          "conclusion. Top: per-model regression adjustment "
          "(step-level $y=\\log\\mathrm{JS}^{\\mathrm{NAT}}-\\log\\mathrm{JS}^{\\mathrm{C1}}$ "
          "on $x=\\log P_{\\mathrm{logit}}^{\\mathrm{NAT}}-\\log P_{\\mathrm{logit}}^{\\mathrm{C1}}$, steps $2\\ldots T_i$); $\\beta_0=\\bar y$ is the "
          "step-level mean log ratio, not the problem-level contrast $\\theta(\\mathrm{NAT},\\mathrm{C1})$; $S_m=|\\beta_0|-|\\beta_1|$, "
          "$f_m=1-|\\beta_1|/|\\beta_0|$. Bottom: cross-model gaps $\\Gamma_0$ (unadjusted) and $\\Gamma_1$ (adjusted), $f_\\times=1-|\\Gamma_1|/|\\Gamma_0|$, "
          "same problems resampled for both models; only Qwen-base vs Llama-instruct is a decision rule, the other pairs are descriptive. "
          "``---'': precondition not met. H-C bands and $p$-values for this table only: Qwen-base Holm \\vn{hc.qb.holm}, Qwen-instruct Holm "
          "\\vn{hc.qi.holm}, Llama-instruct Holm \\vn{hc.li.holm} (within $\\mathcal{F}_m$).}",
          "\\label{tab:hc}", "\\end{table}"]
    (TAB / "hc.tex").write_text("\n".join(t) + "\n", encoding="utf-8")


def table_cal():
    rows = [("$\\rho_K$ / $\\rho_V$ (median $\\|\\Delta\\|/\\|\\kappa^{\\FF}\\|$)", lambda k: f"\\vn{{cal.{k}.rhoK}} / \\vn{{cal.{k}.rhoV}}"),
            ("$\\rho^\\ast$; $s_K$ / $s_V$", lambda k: f"\\vn{{cal.{k}.rhostar}}; \\vn{{cal.{k}.sK}} / \\vn{{cal.{k}.sV}}"),
            ("$\\hat b$; $c$ (before clamp); margin", lambda k: f"\\vn{{cal.{k}.b}}; \\vn{{cal.{k}.c}} (\\vn{{cal.{k}.craw}}); $\\pm$\\vn{{cal.{k}.margin}}"),
            ("$E_8$ median; $r_{80}$ median; share $r_{80}\\le8$", lambda k: f"\\vn{{cal.{k}.E8}}; \\vn{{cal.{k}.r80}}; \\vn{{cal.{k}.r80share}}\\%"),
            ("Mean scored steps; share with 64", lambda k: f"\\vn{{hist.{k}.Smean}}; \\vn{{hist.{k}.S64}}\\%"),
            ("GPU hours (gates, calibration, runtime gate, test run)", lambda k: f"\\vn{{gpu.{k}}}")]
    t = ["\\begin{table}[!htbp]", "\\centering\\scriptsize", "\\begin{tabular}{@{}l l l l l@{}}", "\\toprule",
         "Quantity & " + " & ".join(NAME[m] for m in MODELS) + " \\\\", "\\midrule"]
    for lab, f in rows:
        t.append(lab + " & " + " & ".join(f(SHORT[m]) for m in MODELS) + " \\\\")
    t += ["\\bottomrule", "\\end{tabular}",
          "\\caption{Calibration (100 GSM8K training problems per model; frozen and hashed into the lock before the model's first test problem) and "
          "run summary. H-D is descriptive if $s_K$ or $s_V$ lies outside $[0.5,2]$; equivalence verdicts require $\\hat b\\ge0.25$. "
          "$c$ is clamped to $[0.10,0.40]$; the unclamped value is in parentheses.}",
          "\\label{tab:cal}", "\\end{table}"]
    (TAB / "cal.tex").write_text("\n".join(t) + "\n", encoding="utf-8")


def table_desc():
    t = ["\\begin{table}[!htbp]", "\\centering\\scriptsize", "\\setlength{\\tabcolsep}{3pt}",
         "\\begin{tabular}{@{}p{3.1cm}p{2.7cm}p{2.7cm}p{2.7cm}p{2.7cm}@{}}", "\\toprule",
         "Quantity & " + " & ".join(NAME[m] for m in MODELS) + " \\\\", "\\midrule"]
    for key, lab in (("natc2", "NAT$-$C2"), ("c2c1", "C2$-$C1"), ("natcv", "NAT$-$CV"), ("nkv", "NAT-K$-$NAT-V (natural sizes)")):
        t.append(lab + " & " + " & ".join(f"\\vn{{{key}.{SHORT[m]}.est}} \\vn{{{key}.{SHORT[m]}.ci}}\\newline \\vn{{{key}.{SHORT[m]}.verdict}}"
                                          for m in MODELS) + " \\\\")
    for key, lab in (("pval", "$D$ with $P_{\\mathrm{val}}$"), ("panormk", "$D$ with $P_{\\mathrm{anormK}}$"), ("pout", "$D$ with $P_{\\mathrm{out}}$"),
                     ("hapool", "$D$ pooled over NAT, C1, C2, QA, QO, CV")):
        t.append(lab + " & " + " & ".join(f"\\vn{{{key}.{SHORT[m]}.est}} \\vn{{{key}.{SHORT[m]}.ci}}" for m in MODELS) + " \\\\")
    t.append("$m^{\\mathrm{NAT}}/(m^{\\mathrm{LOC}}+m^{\\mathrm{INH}})$, median & " + " & ".join(
        f"\\vn{{nonadd.{SHORT[m]}.med}} \\vn{{nonadd.{SHORT[m]}.ci}}" for m in MODELS) + " \\\\")
    t.append("$\\|\\Delta\\|/\\|\\kappa^{\\FF}\\|$ in NAT, K / V (\\%) & " + " & ".join(
        f"\\vn{{rel.{SHORT[m]}.K}} / \\vn{{rel.{SHORT[m]}.V}}" for m in MODELS) + " \\\\")
    t += ["\\bottomrule", "\\end{tabular}",
          "\\caption{Descriptive and secondary quantities \\tagD{} (pre-declared, not tested; 95\\% bootstrap intervals). The $D$ rows are the H-A "
          "estimand with another predictor or over the six norm-matched arms (\\S8.1 secondary of the protocol).}",
          "\\label{tab:desc}", "\\end{table}"]
    (TAB / "desc.tex").write_text("\n".join(t) + "\n", encoding="utf-8")


def table_fm_full(rep):
    t = ["\\begin{table}[p]", "\\centering\\scriptsize", "\\begin{tabular}{@{}l l l l l l l@{}}", "\\toprule",
         "Model & Test & Estimate & 95\\% CI & 90\\% CI & $p$ & Holm $p$ \\\\", "\\midrule"]
    for m in MODELS:
        fam = rep["per_model"][m]["family"]
        for name, s in fam.items():
            if name == "HC shrinkage":
                t.append(f"{NAME[m]} & H-C shrinkage & {num(s['estimate'])} & {ci(s['ci95'])} & --- & {pv(s['p'])} & {pv(s['holm_p'])} \\\\")
            else:
                t.append(f"{NAME[m]} & {FMLAB[name]} & {num(s['estimate'])} & {ci(s['ci95'])} & {ci(s['ci90'])} & {pv(s['p'])} & {pv(s['holm_p'])} \\\\")
        t.append("\\midrule")
    t[-1] = "\\bottomrule"
    t += ["\\end{tabular}", "\\caption{All tests of the per-model families $F_m$ \\tagC{} with 90\\% intervals (used by the equivalence rule) and raw and "
          "Holm-adjusted $p$. $p$ below $0.001$ is at or near the permutation floor $1/10{,}001$. Llama-base has six tests (H-C not applicable).}",
          "\\label{tab:fmfull}", "\\end{table}"]
    for m in MODELS:
        SRC[f"table fmfull {m}"] = f"v3_report_locked.json:per_model.{m}.family"
    (TAB / "fm_full.tex").write_text("\n".join(t) + "\n", encoding="utf-8")


def table_rdev(rep):
    t = ["\\begin{table}[!htbp]", "\\centering\\scriptsize", "\\begin{tabular}{@{}l l r l l l l@{}}", "\\toprule",
         "Model & Arm & Steps & Mean & p99 $\\le$ & Max & Steps with a slot $>5\\%$ \\\\", "\\midrule"]
    for m in MODELS:
        for a, v in rep["per_model"][m]["realized_norm_deviation"].items():
            if not v["n_slots"]:
                continue
            t.append(f"{NAME[m]} & {ARMNAME.get(a, a)} & {v['n_steps']} & {sci(v['mean'])} & {sci(v['p99_at_most'])} & {v['max']:.3f} & "
                     f"{v['steps_with_a_slot_above_5pct']} \\\\")
        t.append("\\midrule")
        SRC[f"table rdev {m}"] = f"v3_report_locked.json:per_model.{m}.realized_norm_deviation"
    t[-1] = "\\bottomrule"
    t += ["\\end{tabular}", "\\caption{Realized post-rounding relative per-slot norm deviation $|r-t|/t$ of the constructed arms (amendment~2; reported, "
          "not gated). p99 is an upper bin edge (40 bins per decade). Natural arms (NAT, NAT-K, NAT-V, LOC, INH) have no constructed slots.}",
          "\\label{tab:rdev}", "\\end{table}"]
    (TAB / "rdev.tex").write_text("\n".join(t) + "\n", encoding="utf-8")


def table_trunc(rep):
    names = [("NAT-C1", "NAT$-$C1"), ("HB QA-QO", "H-B"), ("HD K*-V*", "H-D"), ("Prov-K NAT-K-C1-K", "Prov-K"), ("Prov-V NAT-V-C1-V", "Prov-V"),
             ("HE INH-LOC", "H-E"), ("C2-C1", "C2$-$C1"), ("NAT-CV", "NAT$-$CV"), ("HC shrinkage", "H-C shrinkage"),
             ("HA (NAT, primary)", "H-A (NAT)"), ("HA pooled norm-matched arms (secondary)", "H-A pooled arms")]
    t = ["\\begin{table}[!htbp]", "\\centering\\scriptsize", "\\begin{tabular}{@{}l l r r l l@{}}", "\\toprule",
         "Model & Contrast & Kept & Dropped & Median retained & Estimate [95\\% CI], verdict \\\\", "\\midrule"]
    for m in ("qwen-base", "qwen-instruct"):
        sr = rep["per_model"][m]["sensitivity_rounding"]["contrasts"]
        for name, lab in names:
            v = sr[name]
            r = v["result"]
            if r is None:
                res = "not estimable"
            elif "estimate" in r:
                res = f"{num(r['estimate'])} {ci(r['ci95'])}, {r['verdict']}"
            else:
                res = f"$S$ {num(r['shrinkage'])} {ci(r['shrinkage_ci95'])}, {r['band']}"
            fr = "---" if v["median_retained_fraction"] is None else f"{v['median_retained_fraction']:.2f}"
            t.append(f"{NAME[m]} & {lab} & {v['n_kept']} & {v['n_dropped']} & {fr} & {res} \\\\")
        t.append("\\midrule")
        SRC[f"table trunc {m}"] = f"v3_report_locked.json:per_model.{m}.sensitivity_rounding"
    t[-1] = "\\bottomrule"
    t += ["\\end{tabular}", "\\caption{Exploratory truncation sensitivity \\tagE{} (amendment~2) for the two Qwen models: per problem and contrast, steps after "
          "the first write with a slot above 5\\% deviation are dropped (arms of a contrast share the retained steps); problems with fewer than 8 retained steps "
          "are dropped. In both Llama models no slot exceeds 5\\%, all 200 problems are kept and the estimates equal the primaries.}",
          "\\label{tab:trunc}", "\\end{table}"]
    (TAB / "trunc.tex").write_text("\n".join(t) + "\n", encoding="utf-8")


def table_sens(rep):
    t = ["\\begin{table}[!htbp]", "\\centering\\scriptsize", "\\begin{tabular}{@{}l l l l l l@{}}", "\\toprule",
         "Contrast & Variant & " + " & ".join(NAME[m] for m in MODELS) + " \\\\", "\\midrule"]
    for key, lab in (("NAT-C1", "NAT$-$C1"), ("HB", "H-B"), ("HD", "H-D"), ("HE", "H-E")):
        for var in ("steps 2-8", "steps 9-64", "per-problem median", "excluding exposed", "S >= 32"):
            vals = [rep["per_model"][m]["sensitivity"][key][var] for m in MODELS]
            t.append(f"{lab} & {var.replace('S >= 32', '$T_i\\ge32$').replace('2-8', '2--8').replace('9-64', '9--64')} & " +
                     " & ".join(num(v) for v in vals) + " \\\\")
        t.append("\\midrule")
    for m in MODELS:
        SRC[f"table sens {m}"] = f"v3_report_locked.json:per_model.{m}.sensitivity"
    t[-1] = "\\bottomrule"
    t += ["\\end{tabular}", "\\caption{Pre-declared sensitivity variants \\tagE{} (point estimates, as computed by the locked analysis); none replaces a "
          "primary. ``Excluding exposed'' removes the 4 test problems whose gold solutions were token-labelled in earlier work "
          "(Appendix~\\ref{app:protocols}).}", "\\label{tab:sensv3}", "\\end{table}"]
    (TAB / "sens.tex").write_text("\n".join(t) + "\n", encoding="utf-8")


def table_exp():
    t = ["\\begin{table}[!htbp]", "\\centering\\scriptsize", "\\setlength{\\tabcolsep}{3pt}",
         "\\begin{tabular}{@{}p{3.3cm}p{2.65cm}p{2.65cm}p{2.65cm}p{2.65cm}@{}}", "\\toprule",
         "Quantity & " + " & ".join(NAME[m] for m in MODELS) + " \\\\", "\\midrule",
         "\\multicolumn{5}{@{}l}{\\emph{X1: path-weighted provenance as a predictor of NAT$-$C1}} \\\\"]
    rows = [("Mean natural K share $w_K$", lambda k: f"\\vn{{x1.{k}.wK}}"),
            ("Predicted (weighted)", lambda k: f"\\vn{{x1.{k}.pred}} \\vn{{x1.{k}.predci}}"),
            ("Observed NAT$-$C1", lambda k: f"\\vn{{x1.{k}.obs}}"),
            ("Observed $-$ predicted; Holm", lambda k: f"\\vn{{x1.{k}.d}} \\vn{{x1.{k}.dci}}\\newline Holm \\vn{{x1.{k}.holm}}"),
            ("Sign predicted", lambda k: f"\\vn{{x1.{k}.sign}}"),
            ("Spearman(pred, obs) over problems", lambda k: f"\\vn{{x1.{k}.rho}} \\vn{{x1.{k}.rhoci}}")]
    for lab, f in rows:
        t.append(lab + " & " + " & ".join(f(SHORT[m]) for m in MODELS) + " \\\\")
    t.append("\\midrule\\multicolumn{5}{@{}l}{\\emph{X2: H-A estimand $D$ with other predictors (NAT, steps $2\\ldots T_i$)}} \\\\")
    for key, lab in (("x2a", "X2a: $P_{\\mathrm{val}}$ vs $P_{\\mathrm{norm}}$"), ("x2b", "X2b: rank-sum $P_{\\mathrm{comb}}$ vs $P_{\\mathrm{norm}}$"),
                     ("x2c", "X2c: $D_{\\mathrm{comb}}-D_{\\mathrm{logit}}$")):
        t.append(lab + " & " + " & ".join(f"\\vn{{{key}.{SHORT[m]}.est}} \\vn{{{key}.{SHORT[m]}.ci}}\\newline Holm \\vn{{{key}.{SHORT[m]}.holm}}; "
                                          f"\\vn{{{key}.{SHORT[m]}.verdict}}" for m in MODELS) + " \\\\")
    t += ["\\bottomrule", "\\end{tabular}",
          "\\caption{Exploratory X1 and X2 \\tagE{} (plan entry \\texttt{4b1ca9d}, committed after the locked results were known and before these "
          "were computed; not blind). Holm over the four models within each row, labelled exploratory. ``Sign predicted'' (fixed in the plan): "
          "\\emph{yes} if the mean predicted and mean observed contrasts have the same sign and both 95\\% intervals exclude 0; Llama-base is "
          "\\emph{no} because its observed interval includes 0. X2a reproduces a pre-declared descriptive quantity of the locked analysis.}",
          "\\label{tab:exp}", "\\end{table}"]
    (TAB / "exp.tex").write_text("\n".join(t) + "\n", encoding="utf-8")


def _xtable(name, label, caption, groups):
    t = ["\\begin{table}[!htbp]", "\\centering\\scriptsize", "\\setlength{\\tabcolsep}{3pt}",
         "\\begin{tabular}{@{}p{4.1cm}p{2.55cm}p{2.55cm}p{2.55cm}p{2.55cm}@{}}", "\\toprule",
         "Quantity & " + " & ".join(NAME[m] for m in MODELS) + " \\\\"]
    for title, rows in groups:
        t.append(f"\\midrule\\multicolumn{{5}}{{@{{}}l}}{{\\emph{{{title}}}}} \\\\")
        for lab, f in rows:
            t.append(lab + " & " + " & ".join(f(SHORT[m]) for m in MODELS) + " \\\\")
    t += ["\\bottomrule", "\\end{tabular}", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\end{table}"]
    (TAB / name).write_text("\n".join(t) + "\n", encoding="utf-8")


def table_xa():
    _xtable("xa.tex", "tab:xa",
            "Exploratory analyses \\tagE{}, part 1: absolute divergence, the key-error geometry and origin shares, and the predictors. "
            "Each row names its analysis (X$n$); plan entries and blind status are in Table~\\ref{tab:xplan}. X6, X8 and X4 are descriptive; "
            "X9a is an exploratory test (Holm over the four models); X5 is exploratory with Holm over the four models; X9c is descriptive. "
            "JS in nats; intervals are 95\\% problem-bootstrap intervals; X9 uses 50 of the 200 problems.",
            [("Absolute divergence (X6)", [
                ("NAT: geometric-mean JS", lambda k: f"\\vn{{x6.{k}.nat.geo}}"),
                ("NAT: arithmetic-mean JS", lambda k: f"\\vn{{x6.{k}.nat.arith}}")]),
             ("Key error in the query subspace (X9a)", [
                ("Share $\\phi$ of $\\|\\Delta_K\\|^2$ in the top-8 subspace (isotropic 0.125)", lambda k: f"\\vn{{x9a.{k}.F}}"),
                ("$A=\\mathrm{mean}\\,\\ln(\\phi_i/0.125)$; Holm; verdict", lambda k: f"\\vn{{x9a.{k}.A}} \\vn{{x9a.{k}.Aci}}\\newline Holm \\vn{{x9a.{k}.holm}}; \\vn{{x9a.{k}.verdict}}"),
                ("$\\phi$ in layer 0", lambda k: f"\\vn{{x9a.{k}.F0}}"),
                ("Spearman of $\\ln \\phi_i$ with $\\theta_i$(NAT-K, C1-K)", lambda k: f"\\vn{{x9a.{k}.rho}} \\vn{{x9a.{k}.rhoci}}")]),
             ("Origin shares of NAT's geometric-mean JS (X8)", [
                ("INH / NAT (\\%)", lambda k: f"\\vn{{x8.{k}.inh}} \\vn{{x8.{k}.inhci}}"),
                ("LOC / NAT (\\%)", lambda k: f"\\vn{{x8.{k}.loc}} \\vn{{x8.{k}.locci}}")]),
             ("Spearman of each predictor with per-step JS, NAT (X4)", [
                ("$P_{\\mathrm{norm}}$", lambda k: f"\\vn{{x4.{k}.norm}} \\vn{{x4.{k}.norm.ci}}"),
                ("$P_{\\mathrm{logit}}$", lambda k: f"\\vn{{x4.{k}.logit}} \\vn{{x4.{k}.logit.ci}}"),
                ("$P_{\\mathrm{val}}$", lambda k: f"\\vn{{x4.{k}.val}} \\vn{{x4.{k}.val.ci}}"),
                ("$P_{\\mathrm{anormK}}$", lambda k: f"\\vn{{x4.{k}.anormk}} \\vn{{x4.{k}.anormk.ci}}")]),
             ("Query direction and layer 0 (X5, X9c)", [
                ("X5: $\\mathrm{Sp}(P_{\\mathrm{logit}})-\\mathrm{Sp}(P_{\\mathrm{anormK}})$; Holm", lambda k: f"\\vn{{x5.{k}.est}} \\vn{{x5.{k}.ci}}\\newline Holm \\vn{{x5.{k}.holm}}"),
                ("X9c: change of $\\mathrm{Sp}(P_{\\mathrm{logit}})$ without layer 0", lambda k: f"\\vn{{x9c.{k}.drho}} \\vn{{x9c.{k}.drhoci}}"),
                ("X9c: H-A $D$, all layers $\\to$ without layer 0", lambda k: f"\\vn{{x9c.{k}.D}} $\\to$ \\vn{{x9c.{k}.D0}}")])])


def table_xb():
    _xtable("xb.tex", "tab:xb",
            "Exploratory analyses \\tagE{}, part 2: equal-relative-error matching and the H-B positive control. X10, X9d and X11 are descriptive; "
            "X9b is an exploratory test in the two Qwen models (rule of FOLLOWUP\\_X9; no adjustment across the two) and a descriptive control "
            "in the Llama models. $R_{\\mathrm{obs}}=e^{\\theta(\\text{REL-}P,\\,\\text{NAT-}P)}$; benchmark $s_P^{\\hat b}$ from the calibrated "
            "dose slope; ratios above 1 mean REL-$P$ is more disruptive than uniform scaling at the slope's rate would predict. "
            "GM: geometric mean over problems of the per-problem mean JS.",
            [("REL arms against the dose-slope benchmark (X10)", [
                ("K: $R_{\\mathrm{obs}}$ vs $s_K^{\\hat b}$", lambda k: f"\\vn{{x10.{k}.K.robs}} vs \\vn{{x10.{k}.K.rpred}}"),
                ("K: $R_{\\mathrm{obs}}/s_K^{\\hat b}$", lambda k: f"\\vn{{x10.{k}.K.ratio}}"),
                ("V: $R_{\\mathrm{obs}}/s_V^{\\hat b}$", lambda k: f"\\vn{{x10.{k}.V.ratio}}")]),
             ("Effective REL-K scale per layer, $\\rho^\\ast/\\rho_{\\mathrm{slot}}$ (X9d)", [
                ("Nominal $s_K$", lambda k: f"\\vn{{x9d.{k}.sK}}"),
                ("Layer 0 (median)", lambda k: f"\\vn{{x9d.{k}.L0}}"),
                ("Largest other layer; median of other layers", lambda k: f"\\vn{{x9d.{k}.restmax}}; \\vn{{x9d.{k}.restmed}}")]),
             ("Layer 0 left natural (X9b)", [
                ("Share $G$ of the REL-K excess removed; verdict", lambda k: f"\\vn{{x9b.{k}.G}} \\vn{{x9b.{k}.Gci}}\\newline \\vn{{x9b.{k}.verdict}}"),
                ("Excess over benchmark: REL-K $\\to$ layer 0 natural", lambda k: f"\\vn{{x9b.{k}.EK}} $\\to$ \\vn{{x9b.{k}.EL0}}")]),
             ("H-B positive control (X11)", [
                ("First-order benchmark $B$", lambda k: f"\\vn{{x11.{k}.B}}"),
                ("Observed $e^{\\theta(\\mathrm{QA},\\mathrm{QO})}$ (locked)", lambda k: f"\\vn{{hb.{k}.ratio}}"),
                ("Shared-V dilution $\\mathrm{GM}(\\text{C1-V})/\\mathrm{GM}(\\mathrm{QO})$", lambda k: f"\\vn{{x11.{k}.dil}} \\vn{{x11.{k}.dilci}}")])])


def table_x6full():
    rows = [(f"{ARMNAME.get(a, a)}", lambda k, ak=ak: f"\\vn{{x6.{k}.{ak}.geo}}") for a, ak in X6ARMS]
    _xtable("x6full.tex", "tab:x6full",
            "X6 \\tagD{}: geometric-mean over problems of the per-problem mean JS (nats, steps $1\\ldots T_i$) for every arm. Plan entry "
            "\\texttt{21e16a5}; descriptive. The arithmetic means and intervals are in \\texttt{EXPLORATORY\\_RESULTS\\_X4X8.md}.",
            [("Geometric-mean JS", rows)])


def table_x7():
    _xtable("x7.tex", "tab:x7",
            "X7 \\tagE{}: why H-C is not identified (plan entry \\texttt{21e16a5}). X7a: the locked covariate $x$ and fit, over all scored "
            "steps $2\\ldots T_i$ of the 200 problems. X7b/X7c re-fit the adjustment on within-problem rank differences of the predictors "
            "(X7b: $P_{\\mathrm{logit}}$ and $P_{\\mathrm{val}}$ jointly; X7c: $P_{\\mathrm{logit}}$ only), whose zero point is also never "
            "reached. They are shown only to illustrate how the extrapolated intercept depends on the covariate scale; they are not results.",
            [("X7a: locked H-C covariate and fit", [
                ("$\\bar x$", lambda k: f"\\vn{{x7a.{k}.xbar}} \\vn{{x7a.{k}.xbarci}}"),
                ("Steps with $x<0$ / scored steps", lambda k: f"\\vn{{x7a.{k}.nneg}} / \\vn{{x7a.{k}.nsteps}}"),
                ("Slope $\\gamma$", lambda k: f"\\vn{{x7a.{k}.gamma}} \\vn{{x7a.{k}.gammaci}}"),
                ("$\\beta_1$ (95\\% CI)", lambda k: f"\\vn{{hc.{k}.b1}} \\vn{{x7a.{k}.b1ci}}"),
                ("$\\bar x$ for $P_{\\mathrm{val}}$", lambda k: f"\\vn{{x7a.{k}.xval}}")]),
             ("X7b: joint rank-based adjustment (illustration)", [
                ("$\\beta_1$; $f$; band", lambda k: f"\\vn{{x7b.{k}.b1}} \\vn{{x7b.{k}.b1ci}}\\newline $f$ \\vn{{x7b.{k}.f}} \\vn{{x7b.{k}.fci}}; \\vn{{x7b.{k}.band}}")]),
             ("X7c: rank-based, $P_{\\mathrm{logit}}$ only (illustration)", [
                ("$\\beta_1$; $f$; band", lambda k: f"\\vn{{x7c.{k}.b1}} \\vn{{x7c.{k}.b1ci}}\\newline $f$ \\vn{{x7c.{k}.f}} \\vn{{x7c.{k}.fci}}; \\vn{{x7c.{k}.band}}")])])


def fig_x9():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    X9 = json.loads((P7 / "x9_results.json").read_text())["per_model"]
    cols = {"qwen-base": "#1f5fa8", "qwen-instruct": "#6fa8dc", "llama-base": "#b45309", "llama-instruct": "#f2a65a"}
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for m in MODELS:
        F = [x["F"] for x in X9[m]["a"]["F_per_layer"]]
        ax[0].plot(range(len(F)), F, "o-", ms=2.5, lw=0.9, color=cols[m], label=NAME[m])
        d = [x["median"] for x in X9[m]["d"]["per_layer"]]
        ax[1].semilogy(range(len(d)), d, "o-", ms=2.5, lw=0.9, color=cols[m], label=NAME[m])
        ax[1].axhline(X9[m]["d"]["s_K"], color=cols[m], lw=0.6, ls=":")
    ax[0].axhline(8 / 64, color="k", lw=0.6, ls="--")
    ax[0].set_xlabel("layer")
    ax[0].set_ylabel("share of $\\|\\Delta_K\\|^2$ in top-8 query subspace")
    ax[0].text(0.02, 0.97, "dashed: isotropic 8/64", transform=ax[0].transAxes, ha="left", va="top", fontsize=6)
    ax[1].set_xlabel("layer")
    ax[1].set_ylabel("effective REL-K scale $\\rho^*/\\rho_{slot}$ (median)")
    ax[1].text(0.98, 0.92, "dotted: nominal $s_K$", transform=ax[1].transAxes, ha="right", fontsize=6)
    ax[0].set_ylim(0, None)
    ax[0].legend(fontsize=6, frameon=False, loc="lower center", ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "fig_x9.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ figures
def figures(rep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42})
    cols = {"qwen-base": "#1f5fa8", "qwen-instruct": "#6fa8dc", "llama-base": "#b45309", "llama-instruct": "#f2a65a"}
    tests = [("Prov-K NAT-K-C1-K", "NAT-K − C1-K"), ("Prov-V NAT-V-C1-V", "NAT-V − C1-V"), ("HD K*-V*", "H-D: REL-K − REL-V"),
             ("NAT-C1", "NAT − C1"), ("HE INH-LOC", "H-E: INH − LOC"), ("HB QA-QO", "H-B: QA − QO")]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw={"width_ratios": [2.3, 1]})
    for j, (name, lab) in enumerate(tests):
        for q, m in enumerate(MODELS):
            s = rep["per_model"][m]["family"][name]
            y = -(j * 5 + q)
            ax[0].errorbar(s["estimate"], y, xerr=[[s["estimate"] - s["ci95"][0]], [s["ci95"][1] - s["estimate"]]], fmt="o", ms=3,
                           color=cols[m], label=NAME[m] if j == 0 else None)
    ax[0].axvline(0, color="k", lw=0.6)
    ax[0].set_yticks([-(j * 5 + 1.5) for j in range(len(tests))])
    ax[0].set_yticklabels([lab for _, lab in tests])
    ax[0].set_xlabel("log mean-JS contrast (95% CI)")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=7, frameon=False)   # outside the axes: no marker hidden
    for q, m in enumerate(MODELS):
        s = rep["per_model"][m]["HA"]
        ax[1].errorbar(s["estimate"], -q, xerr=[[s["estimate"] - s["ci95"][0]], [s["ci95"][1] - s["estimate"]]], fmt="o", ms=3, color=cols[m])
    s = rep["HA_pooled_secondary"]
    ax[1].errorbar(s["estimate"], -4.5, xerr=[[s["estimate"] - s["ci95"][0]], [s["ci95"][1] - s["estimate"]]], fmt="D", ms=3, color="k")
    ax[1].axvline(0, color="k", lw=0.6)
    ax[1].set_yticks([0, -1, -2, -3, -4.5])
    ax[1].set_yticklabels([NAME[m] for m in MODELS] + ["pooled (secondary)"])
    ax[1].set_xlabel("H-A: $D_m$ (95% CI)")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIG / "fig_contrasts.pdf")
    plt.close(fig)
    # 2x2: mean theta(NAT, C1) per model with CI
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for fam, ms_, xs in (("Qwen", ("qwen-base", "qwen-instruct"), 0), ("Llama", ("llama-base", "llama-instruct"), 1)):
        ests = [rep["per_model"][m]["family"]["NAT-C1"] for m in ms_]
        ax.plot([0, 1], [e["estimate"] for e in ests], "-", color="#1f5fa8" if fam == "Qwen" else "#b45309", lw=1)
        for t_, (m, e) in enumerate(zip(ms_, ests)):
            ax.errorbar(t_ + (0.03 if fam == "Llama" else -0.03), e["estimate"], yerr=[[e["estimate"] - e["ci95"][0]], [e["ci95"][1] - e["estimate"]]],
                        fmt="o", ms=3, color="#1f5fa8" if fam == "Qwen" else "#b45309", label=fam if t_ == 0 else None)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["base", "instruct"])
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylabel("NAT − C1, log mean-JS (95% CI)")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_2x2.pdf")
    plt.close(fig)


def main():
    TAB.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    rep, exp, rec, cmp_, cal = load()
    numbers(rep, exp, rec, cmp_, cal)
    numbers_x(cal)
    tex_numbers()
    table_main()
    table_hf()
    table_hc(rep)
    table_cal()
    table_desc()
    table_fm_full(rep)
    table_rdev(rep)
    table_trunc(rep)
    table_sens(rep)
    table_exp()
    table_xa()
    table_xb()
    table_x6full()
    table_x7()
    table_x13()
    figures(rep)
    fig_x9()
    for f in TAB.glob("*.tex"):
        if f.name != "numbers_v3.tex":
            boxed(f)
    tex_numbers()                                         # again: tables add table-level sources
    print(f"{len(V)} macros; tables in {TAB}; figures in {FIG}")


if __name__ == "__main__":
    main()
