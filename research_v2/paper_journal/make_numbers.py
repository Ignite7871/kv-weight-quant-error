"""Generate tables/numbers.tex (LaTeX macros) from the LOCKED report JSONs so that no E1/E1L number in the manuscript is typed by hand.
The independent audit (audit/audit_journal.py) re-derives the same numbers from the raw records.
  venv/Scripts/python.exe research_v2/paper_journal/make_numbers.py"""
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RV = HERE.parent
q = json.loads((RV / "e1_runs" / "e1_report.json").read_text())
l = json.loads((RV / "e1l_runs" / "e1l_report.json").read_text())
aud = json.loads((HERE / "audit" / "JOURNAL_AUDIT_RESULTS.json").read_text())
tb = json.loads((RV / "e1_runs" / "tierB_norm_tolerance_assessment.json").read_text())
M = {}


def f(x, d=3, plus=False):
    s = f"{x:+.{d}f}" if plus else f"{x:.{d}f}"
    return "$" + s.replace("-", "{-}") + "$" if s[0] in "+-" else s


def put(name, val):
    assert re.fullmatch(r"[A-Za-z]+", name), name
    M[name] = val


def est(prefix, v, d=3):
    put(prefix, f(v["estimate"], d, True))
    put(prefix + "CI", "$[" + f"{v['ci95'][0]:+.{d}f}" .replace("-", "{-}") + "," + f"{v['ci95'][1]:+.{d}f}".replace("-", "{-}") + "]$")


for tag, rep in (("Q", q), ("L", l)):
    P = rep["tierA"]["primary_log_JS_NAT_minus_C1"]
    est(tag + "ThetaA", P)
    put(tag + "RatioA", f(P["ratio_NAT_over_C1"], 3))
    put(tag + "PA", f"{P['p_signflip']:.4f}" if P["p_signflip"] >= 1e-3 else "$<0.001$")
    put(tag + "CIninety", "$[" + f"{P['ci90'][0]:+.3f}".replace("-", "{-}") + "," + f"{P['ci90'][1]:+.3f}".replace("-", "{-}") + "]$")
    put(tag + "Margin", f(rep["calibration"]["margin_tierA_log_js"], 3))
    put(tag + "AlphaStar", f(rep["calibration"]["alpha_star"], 3))
    put(tag + "DoseB", f(rep["calibration"]["dose_slope_b_hat"], 2))
    put(tag + "CUsed", f(rep["calibration"]["c_used"], 2))
    put(tag + "CRaw", f(rep["calibration"]["cv_raw_median"], 3))
    S = rep["tierA"]["secondary"]
    est(tag + "NatTwo", S["A: NAT - C2 (log JS)"])
    est(tag + "TwoOne", S["A: C2 - C1 (log JS)"])
    est(tag + "Dose", S["A: dose slope of C1 (log JS per log alpha)"], 2)
    est(tag + "Star", S["A: NAT - C1* (effect-matched, log JS)"])
    est(tag + "Flip", S["A: top-1 flip rate NAT - C1"], 4)
    for k, nm in (("A: NAT - C2 (log JS)", "NatTwo"), ("A: C2 - C1 (log JS)", "TwoOne"), ("A: dose slope of C1 (log JS per log alpha)", "Dose"),
                  ("A: NAT - C1* (effect-matched, log JS)", "Star"), ("A: top-1 flip rate NAT - C1", "Flip")):
        h = rep["holm_family"][k]["holm_p"]
        put(tag + nm + "Holm", f"{h:.4f}" if h >= 1e-3 else "$<0.001$")
    put(tag + "NatTwoEq", "yes" if S["A: NAT - C2 (log JS)"].get("equivalent_within_margin") else "no")
    D = rep["tierA"]["diagnostics"]
    put(tag + "RelScale", f(100 * D["NAT"]["mean_rel_scale"], 1))
    put(tag + "DevMaxCone", f(100 * D["C1_a1"]["max_norm_dev_max"], 1))
    put(tag + "DevMeanCone", f(100 * D["C1_a1"]["mean_norm_dev_mean"], 3))
    sens = rep["sensitivity"]["tierA"]
    for k, nm in (("steps 1-32", "SensEarly"), ("steps 9-64", "SensLate"), ("per-problem median JS", "SensMed"), ("steps 2-8 only", "SensTwoEight"),
                  ("pooled steps, problem-clustered (mean of log JS ratio)", "SensPooled"), ("excluding exposed test problems", "SensExp"), ("problems with >= 32 steps", "SensLong")):
        est(tag + nm, sens[k])
    for a, nm in (("NAT", "Nat"), ("C1_a1.0", "One"), ("C2_a1.0", "Two"), ("C1_a0.5", "Half"), ("C1_a2.0", "Double"), ("C1star", "StarJS")):
        pass
for tag, root in (("Q", "e1_runs"), ("L", "e1l_runs")):
    R = {t: {json.loads(x)["index"]: json.loads(x) for x in open(RV / root / "tierA" / f"{t}.jsonl", encoding="utf-8") if x.strip()}
         for t in ("NAT", "C1_a1.0", "C2_a1.0", "C1_a0.5", "C1_a2.0", "C1star")}
    for t, nm in (("NAT", "Nat"), ("C1_a1.0", "One"), ("C2_a1.0", "Two"), ("C1_a0.5", "Half"), ("C1_a2.0", "Double"), ("C1star", "StarJS")):
        put(tag + "Js" + nm, f(np.mean([np.mean(R[t][i]["js"]) for i in R[t]]), 5))
        put(tag + "Fl" + nm, f(100 * np.mean([np.mean(R[t][i]["flip"]) for i in R[t]]), 2))
rh = aud["reference_history"]
for tag, k in (("Q", "qwen"), ("L", "llama")):
    put(tag + "SMean", f(rh[k]["S_mean"], 1))
    put(tag + "ShareSSixtyFour", f(100 * rh[k]["share_S64"], 0))
    put(tag + "GenMean", f(rh[k]["full_generation_mean_tokens"], 0))
P = q["tierB"]["primary_restricted_first_divergence_NAT_minus_C1"]
put("BThetaB", f(P["estimate"], 2, True))
put("BThetaBCI", "$[" + f"{P['ci95'][0]:+.2f}" + "," + f"{P['ci95'][1]:+.2f}" + "]$")
put("BThetaBHolm", f"{q['holm_family']['B: NAT - C1 (restricted first divergence)']['holm_p']:.3f}")
S2 = q["tierB"]["secondary"]["B: NAT - C2 (restricted first divergence)"]
put("BNatTwo", f(S2["estimate"], 2, True))
put("BNatTwoCI", "$[" + f"{S2['ci95'][0]:+.2f}".replace("-", "{-}") + "," + f"{S2['ci95'][1]:+.2f}" + "]$")
for a in ("NAT", "C1", "C2"):
    put("BMean" + {"NAT": "Nat", "C1": "Cone", "C2": "Ctwo"}[a], f(q["tierB"]["mean_restricted_first_divergence"][a], 1))
put("BExclEst", f(tb["sensitivity_tierB_restricted_first_divergence"]["NAT - C1"]["excluding_affected"]["estimate"], 2, True))
put("BNormSteps", f"{tb['prevalence']['C1']['steps_above']}")
put("BNormStepsAll", f"{tb['prevalence']['C1']['steps']:,}".replace(",", "\\,"))
put("BNormMax", f(100 * tb["prevalence"]["C1"]["max_deviation"], 1))
put("BAccFP", f(100 * tb["accuracy"]["FP16"], 1))
put("BAccNat", f(100 * tb["accuracy"]["NAT"], 1))
put("BAccOne", f(100 * tb["accuracy"]["C1"], 1))
put("BAccTwo", f(100 * tb["accuracy"]["C2"], 1))
diff = l["tierA"]["primary_log_JS_NAT_minus_C1"]["estimate"] - q["tierA"]["primary_log_JS_NAT_minus_C1"]["estimate"]
sq = (q["tierA"]["primary_log_JS_NAT_minus_C1"]["ci95"][1] - q["tierA"]["primary_log_JS_NAT_minus_C1"]["ci95"][0]) / 3.92
sl = (l["tierA"]["primary_log_JS_NAT_minus_C1"]["ci95"][1] - l["tierA"]["primary_log_JS_NAT_minus_C1"]["ci95"][0]) / 3.92
se = (sq ** 2 + sl ** 2) ** 0.5
put("DiffTheta", f(diff, 3, True))
put("DiffThetaCI", "$[" + f"{diff - 1.96 * se:+.3f}" + "," + f"{diff + 1.96 * se:+.3f}" + "]$")
out = ["% generated by make_numbers.py from the locked report JSONs; do not edit"] + [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(M.items())]
(HERE / "tables" / "numbers.tex").write_text("\n".join(out) + "\n", encoding="utf-8")
print(len(M), "macros")
