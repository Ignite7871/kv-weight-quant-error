"""Side-by-side Qwen2.5-0.5B (E1 Tier A, locked) vs Llama-3.2-1B-Instruct (E1L Tier A, locked) tables, generated from the two report JSONs and the raw Tier A records.
Descriptive comparison only; replication categories are those pre-declared in E1L_SPEC.md section 3.
  venv/Scripts/python.exe research_v2/followup/compare_qwen_llama.py   -> e1l_runs/qwen_llama_comparison.md"""
import json
import math
import sys
from pathlib import Path

import numpy as np

RV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RV))
import analyze_controls as A  # noqa: E402

Q, L = json.loads((RV / "e1_runs" / "e1_report.json").read_text()), json.loads((RV / "e1l_runs" / "e1l_report.json").read_text())
QC, LC = Q["calibration"], L["calibration"]
load = lambda p: A.load_jsonl(p)   # noqa: E731
arms = {"NAT": "NAT", "C1 (a=1)": "C1_a1.0", "C2 (a=1)": "C2_a1.0", "C1 (a=0.5)": "C1_a0.5", "C1 (a=2)": "C1_a2.0", "C1* (effect-matched)": "C1star"}


def stats(root):
    out = {}
    for lab, tag in arms.items():
        r = load(root / "tierA" / f"{tag}.jsonl")
        idx = sorted(r)
        out[lab] = (np.mean([np.mean(r[i]["js"]) for i in idx]), np.mean([np.mean(r[i]["flip"]) for i in idx]),
                    np.mean([np.mean(r[i]["rel_scale_mean"]) for i in idx]), np.max([np.max(r[i]["norm_dev_max"]) for i in idx]), len(idx))
    return out


SQ, SL = stats(RV / "e1_runs"), stats(RV / "e1l_runs")


def row(name, a, b):
    return f"| {name} | {a} | {b} |"


def ci(v):
    return f"{v['estimate']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]"


pq, pl = Q["tierA"]["primary_log_JS_NAT_minus_C1"], L["tierA"]["primary_log_JS_NAT_minus_C1"]
lines = ["# Qwen2.5-0.5B vs Llama-3.2-1B-Instruct: E1 Tier A side by side (generated)", "",
         "| | Qwen2.5-0.5B (locked E1) | Llama-3.2-1B-Instruct (locked E1L) |", "|---|---|---|",
         row("Test problems (Tier A)", pq["n"], pl["n"]),
         row("Calibration: alpha* (effect-matched scale)", f"{QC['alpha_star']:.3f}", f"{LC['alpha_star']:.3f}"),
         row("Calibration: dose slope b (log JS per log alpha)", f"{QC['dose_slope_b_hat']:.3f}", f"{LC['dose_slope_b_hat']:.3f}"),
         row("Calibration: c (clamped) / margin (log JS)", f"{QC['c_used']:.2f} / {QC['margin_tierA_log_js']:.3f}", f"{LC['c_used']:.2f} / {LC['margin_tierA_log_js'] if LC['margin_tierA_log_js'] else 'n/a'}"),
         row("**Primary theta_A = log JS(NAT) - log JS(C1)** [95% CI]", ci(pq), ci(pl)),
         row("Ratio NAT/C1 (geometric)", f"{pq['ratio_NAT_over_C1']:.3f}", f"{pl['ratio_NAT_over_C1']:.3f}"),
         row("Sign-flip p / verdict", f"{pq['p_signflip']:.4f} / {pq['verdict']}", f"{pl['p_signflip']:.4f} / {pl['verdict']}")]
for k in sorted(set(Q["tierA"]["secondary"]) & set(L["tierA"]["secondary"])):
    lines.append(row(k, ci(Q["tierA"]["secondary"][k]), ci(L["tierA"]["secondary"][k])))
lines += ["", "Holm-adjusted p (Qwen: family of 7 incl. two Tier B tests; Llama: family of 5, Tier A only)", "| Test | Qwen raw / Holm | Llama raw / Holm |", "|---|---|---|"]
for k in sorted(set(Q["holm_family"]) & set(L["holm_family"])):
    a, b = Q["holm_family"][k], L["holm_family"][k]
    lines.append(row(k, f"{a['raw_p']:.4f} / {a['holm_p']:.4f}", f"{b['raw_p']:.4f} / {b['holm_p']:.4f}"))
lines += ["", "Descriptive per-arm means over test problems (JS to the pure-FP16 reference; top-1 flip rate; |Delta|/|kappa^F|; max per-slot norm deviation)",
          "| Arm | Qwen JS / flip / rel.scale / max dev | Llama JS / flip / rel.scale / max dev |", "|---|---|---|"]
for lab in arms:
    a, b = SQ[lab], SL[lab]
    lines.append(row(lab, f"{a[0]:.5f} / {100 * a[1]:.2f}% / {a[2]:.3f} / {a[3]:.3f}", f"{b[0]:.5f} / {100 * b[1]:.2f}% / {b[2]:.3f} / {b[3]:.3f}"))
dq, dl = pq["estimate"], pl["estimate"]
sq, sl = (pq["ci95"][1] - pq["ci95"][0]) / 3.92, (pl["ci95"][1] - pl["ci95"][0]) / 3.92
d = dl - dq
se = math.sqrt(sq ** 2 + sl ** 2)
lines += ["", "Between-model difference in the primary effect (Llama - Qwen), independent samples, normal approximation from the two bootstrap SEs (descriptive): "
          f"{d:+.3f} (95% interval {d - 1.96 * se:+.3f} to {d + 1.96 * se:+.3f}).",
          "", "Sensitivity (log JS, NAT - C1)", "| Analysis | Qwen | Llama |", "|---|---|---|"]
for k in Q["sensitivity"]["tierA"]:
    if k in L["sensitivity"]["tierA"]:
        lines.append(row(k, ci(Q["sensitivity"]["tierA"][k]), ci(L["sensitivity"]["tierA"][k])))
(RV / "e1l_runs" / "qwen_llama_comparison.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
