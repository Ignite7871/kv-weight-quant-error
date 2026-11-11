"""Number audit for main_v3.tex (v3 Phase 6). CPU only; read-only.

1. Every \\vn{key} used in the v3 sources is defined by tables_v3/numbers_v3.tex (an undefined key would print ??key).
2. Every v3 result macro taken from the locked analysis is re-checked against the INDEPENDENT recomputation (v3/phase5/recompute_v3.json,
   no shared code with analyze_v3.py): point estimates must agree to the printed precision; 95 % CI endpoints within 5 % of the CI width
   (the declared Monte-Carlo tolerance) plus half a unit of the printed precision; verdict words identical.
3. Every literal number in the v3 main-text sources (outside \\vn{}, \\ref, \\cite, \\label, macros) is listed with its context and a
   classification; unclassified literals are flagged.
4. The E1/E1L numbers cited in the main text through the earlier macros are compared with the E1/E1L reports.
Writes audit/V3_NUMBER_AUDIT.md and audit/V3_NUMBER_AUDIT.json.
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
PJ = HERE.parent
RV = PJ.parent
MAIN = ["00_abstract", "01_introduction", "02_related", "03_methods", "04_results", "05_exploratory", "06_discussion", "07_limitations",
        "08_conclusion", "H_v3extra", "I_xdetails", "G_background", "J_aiuse"]
RDEV = {m: v["realized_norm_deviation"] for m, v in json.loads(
    (Path(__file__).resolve().parent.parent.parent / "v3" / "phase5" / "v3_report_locked.json").read_text())["per_model"].items()}
MODELS ={"qb": "qwen-base", "qi": "qwen-instruct", "lb": "llama-base", "li": "llama-instruct"}
FAMKEY = {"natc1": "NAT-C1", "hb": "HB QA-QO", "hd": "HD K*-V*", "provk": "Prov-K NAT-K-C1-K", "provv": "Prov-V NAT-V-C1-V", "he": "HE INH-LOC"}
DESCKEY = {"natc2": "NAT-C2", "c2c1": "C2-C1", "natcv": "NAT-CV", "nkv": "NAT-K-NAT-V"}

# design constants and labels that may appear as literals (value -> where it is fixed)
CONST = {
    "200": "test problems per model (PROTOCOL_V3 §3)", "100": "calibration problems (§5)", "64": "Tier A steps / head dimension (§3, §2)",
    "65": "history cap (§3)", "8": "top-r subspace, S_i<8 exclusion, kept-step minimum (§4, §8.1, §16)", "10{,}000": "bootstrap / permutation count (§3)",
    "100{,}000": "recomputation sign flips (RECOMPUTATION_TOLERANCES.md)", "20261103": "locked bootstrap seed (§3)", "25": "GPU-hour ceiling (§9)",
    "0.25": "b-hat threshold / band (§5, §8.3)", "0.5": "band / scale range / dose (§5, §8.3)", "2": "scale range / dose / count (§5)",
    "0.497": "assumed SD (§8.9)", "25\\%": "SD guard (§8.9)", "15\\%": "G6 drift limit (§9)", "5\\%": "amendment-2 flag (§16)",
    "0.05": "Holm significance level (§8)", "95\\%": "CI level", "90\\%": "CI level", "80\\%": "r80 definition (§5)", "1": "step 1 / count",
    "10^{-4}": "construction tolerance (§16)", "10^{-9}": "JS floor (§3)", "24": "Qwen layers (§2)", "16": "Llama layers (§2)",
    "14": "Qwen query heads", "32": "Llama query heads", "4": "outlier channels / exposed problems / four models", "0": "zero",
    "3": "3-shot / count", "8\\,GB": "GPU memory", "4060": "GPU model", "0.5B": "model size", "1B": "model size",
    "2.5": "model name Qwen2.5", "3.2": "model name Llama-3.2", "2026-10-29": "date (AI-use log)", "2026-10-31": "date (AI-use log)",
    "2\\times2": "design", "2$\\times$2": "design", "9": "steps 9--64", "2--8": "window", "9--64": "window", "2^{-10}": "E1 tolerance wording (V3-D05)",
    "1.5": "x", "3": "count",
    "930": "Stage 1 sample size (earlier study; STAGE1_REPORT, carried over)", "389": "Stage 2 held-out sample size (earlier study; STAGE2_REPORT)",
    "2026": "date (search period of the related-work section)", "0.10,0.40": "clamp of c (analyze_v3.C_LO, C_HI; PROTOCOL_V3 §5)",
    "50": "X9 problems per model (FOLLOWUP_X9 §2, decision D5)", "0.125": "isotropic subspace share 8/64 (FOLLOWUP_X9 §8a)",
    "0.2": "X9b decision threshold (FOLLOWUP_X9 §8b, decision D1)",
    "100\\%": "the whole (shares need not add to 100%)", "0.10": "lower clamp of c (analyze_v3.C_LO; PROTOCOL_V3 §5)",
    "47": "Stage 2 gold-token-labelled problems (E1_SPEC §2; earlier study)", "2026-11-01": "date (AI-use log)",
    "22": "date: 22 July 2026, earliest ChatGPT project discussion (supplied by the corresponding author, AI_USE_LOG_V3.md)",
    "6": "model name GPT-6 (supplied by the corresponding author, AI_USE_LOG_V3.md)",
}


def macros():
    t = (PJ / "tables_v3" / "numbers_v3.tex").read_text(encoding="utf-8")
    return dict(re.findall(r"\\csname vn@([^\\]+)\\endcsname\{(.*)\}$", t, flags=re.M))


def nums(s):
    return [float(x) for x in re.findall(r"[+-]?\d+\.\d+|[+-]?\d+", s.replace("$", "").replace("\\times10^{", "e").replace("}", ""))]


def check_used(V):
    used, out = set(), []
    for f in list((PJ / "sections_v3").glob("*.tex")) + list((PJ / "tables_v3").glob("*.tex")):
        for k in re.findall(r"\\vn\{([^}]+)\}", f.read_text(encoding="utf-8")):
            used.add(k)
            if k not in V:
                out.append({"file": f.name, "key": k})
    return used, out


def within(printed, true, decimals):
    return abs(printed - true) <= 0.5 * 10 ** (-decimals) + 1e-12


def check_recompute(V):
    R = json.loads((RV / "v3" / "phase5" / "recompute_v3.json").read_text())
    rows = []

    def cmp_est(key, true):
        if key not in V:
            return
        s = V[key].replace("$", "")
        d = len(s.split(".")[1]) if "." in s else 0
        rows.append({"macro": key, "printed": V[key], "recomputed": true, "ok": within(float(s), true, d), "kind": "estimate"})

    def cmp_ci(key, ci):
        if key not in V:
            return
        a, b = nums(V[key])
        s = V[key].replace("$", "").split(",")[0]
        d = len(s.split(".")[1]) if "." in s else 0
        tol = 0.05 * abs(ci[1] - ci[0]) + 0.5 * 10 ** (-d)
        rows.append({"macro": key, "printed": V[key], "recomputed": ci, "ok": abs(a - ci[0]) <= tol and abs(b - ci[1]) <= tol, "kind": "CI"})

    def cmp_word(key, word):
        if key in V:
            rows.append({"macro": key, "printed": V[key], "recomputed": word, "ok": V[key] == word, "kind": "decision"})

    for k, m in MODELS.items():
        P = R["per_model"][m]
        cmp_est(f"ha.{k}.est", P["HA"]["estimate"]); cmp_ci(f"ha.{k}.ci", P["HA"]["ci95"])
        cmp_word(f"ha.{k}.word", "supported" if P["HA"]["supported"] else "contradicted" if P["HA"]["falsified"] else "inconclusive")  # paper word for 'falsified'
        for key, name in FAMKEY.items():
            s = P["family"][name]
            cmp_est(f"{key}.{k}.est", s["estimate"]); cmp_ci(f"{key}.{k}.ci", s["ci95"]); cmp_ci(f"{key}.{k}.ci90", s["ci90"])
            cmp_word(f"{key}.{k}.verdict", s["verdict"])
        hc = P["HC"]
        for key, name in (("b0", "beta0"), ("b1", "beta1"), ("S", "shrinkage"), ("f", "fraction")):
            cmp_est(f"hc.{k}.{key}", hc[name])
        for key, name in (("b0ci", "beta0_ci95"), ("Sci", "shrinkage_ci95"), ("fci", "fraction_ci95")):
            cmp_ci(f"hc.{k}.{key}", hc[name])
        cmp_word(f"hc.{k}.band", hc["band"])
        for key, name in DESCKEY.items():
            s = P["descriptive"][name]
            cmp_est(f"{key}.{k}.est", s["estimate"]); cmp_ci(f"{key}.{k}.ci", s["ci95"]); cmp_word(f"{key}.{k}.verdict", s["verdict"])
        for key, name in (("pval", "HA-type P_val"), ("panormk", "HA-type P_anormK"), ("pout", "HA-type P_out"), ("hapool", "HA pooled norm-matched arms")):
            cmp_est(f"{key}.{k}.est", P["descriptive"][name]["estimate"]); cmp_ci(f"{key}.{k}.ci", P["descriptive"][name]["ci95"])
        cmp_est(f"nonadd.{k}.med", P["descriptive"]["nonadditivity"]["median"])
        cmp_ci(f"nonadd.{k}.ci", P["descriptive"]["nonadditivity"]["ci95_of_median"])
        for key, name in (("s28", "steps 2-8"), ("s964", "steps 9-64"), ("smed", "per-problem median"), ("sexp", "excluding exposed"), ("s32", "S >= 32")):
            cmp_est(f"sens.{k}.{key}", P["sensitivity"]["NAT-C1"][name])
        for key, a in (("K", 0), ("V", 1)):
            cmp_est(f"rel.{k}.{key}", 100 * P["descriptive"]["rel norm K/V (NAT)"][a])
    cmp_est("ha.pooled.est", R["HA_pooled_secondary"]["estimate"]); cmp_ci("ha.pooled.ci", R["HA_pooled_secondary"]["ci95"])
    x = R["HC_cross"]["qwen-base vs llama-instruct"]
    for key in ("g0", "g1", "F"):
        cmp_est(f"hcx.{key}", x[key])
    for key in ("g0ci", "g1ci", "Fci"):
        cmp_ci(f"hcx.{key}", x[key.replace("ci", "") + "_ci95"])
    cmp_word("hcx.band", x["band"])
    for key, name in (("fam", "family (Llama - Qwen)"), ("tun", "tuning (instruct - base)"), ("int", "interaction")):
        s = R["HF"]["contrasts"][name]
        cmp_est(f"hf.{key}.est", s["estimate"]); cmp_ci(f"hf.{key}.ci", s["ci95"]); cmp_ci(f"hf.{key}.ci90", s["ci90"]); cmp_word(f"hf.{key}.verdict", s["verdict"])
    for k, m in MODELS.items():
        cmp_est(f"hf.sd.{k}", R["HF"]["sd_per_model"][m])
    return rows


def literals():
    out = []
    for name in MAIN:
        s = (PJ / "sections_v3" / f"{name}.tex").read_text(encoding="utf-8")
        for ln, line in enumerate(s.splitlines(), 1):
            if line.strip().startswith("%"):
                continue
            clean = re.sub(r"\\(vn|ref|label|cite|citep|input|includegraphics|texttt|url)\{[^}]*\}", " ", line)
            clean = re.sub(r"\\[A-Za-z]+\*?", " ", clean)                         # control words (incl. macro names)
            clean = re.sub(r"\{(?:@\{\}|[lrcp]|p\{[\d.]+cm\})+\}", " ", clean)  # tabular specs
            clean = re.sub(r"p\{[\d.]+cm\}|[\d.]+cm|[\d.]+pt", " ", clean)
            clean = re.sub(r"Qwen2\.5-0\.5B(-Instruct)?|Llama-3\.2-1B(-Instruct)?|Qwen2\.5|Llama-3\.2|NF4|FP16|FP32|BF16|GSM8K|RTX 4060|SHA-256|"
                           r"V3-[AD]\d+|E1L?|E[123][abc]?|H-[A-F]|C[12](-[KV])?|INT[48]|RoPE|P_8|U_8|E_8|r_\{?80\}?|\\mathbb\{R\}\^\{64\}|"
                           r"k_\{[^}]*\}|[a-z]_\{?\d\}?|\^\{?\d\}?|_\{?\d+\}?|\d{4}[a-z]+\d{4}[a-z]+|claude-opus-5-5|Opus 5\.5|Sonnet 5(\.5)?|"
                           r"[0-9a-f]{16}|Tables?~|Section~|Figure~", " ", clean)
            for mm in re.finditer(r"(?<![\w.])(\d[\d{},.]*(?:\\%|\\,GB|\{,\}\d+)?)", clean):
                tok = mm.group(1).rstrip(".,")
                if not tok or tok in ("0", "1"):
                    continue
                cls = CONST.get(tok)
                if cls is None and ("{" in tok or "}" in tok or "tfrac" in line or "10^{" in line):
                    cls = "formula fragment (LaTeX math, not a result)"
                if cls is None and ("2026-10" in line or "October 2026" in line):
                    cls = "date"
                if cls is None and tok in ("0.5,1,2", "0.5,2"):
                    cls = "design constant: dose set / scale range (PROTOCOL_V3 §5)"
                if cls is None and tok == "0.001":
                    cls = "summary threshold; verified by the '<0.001 claims' check"
                if cls is None and "includegraphics" in line:
                    cls = "figure width"
                if cls is None and tok == "5" and "Phase 5" in line:
                    cls = "phase label"
                out.append({"file": name, "line": ln, "literal": tok, "class": cls or "UNCLASSIFIED", "context": line.strip()[:140]})
    return out


def e1_macros():
    t = (PJ / "tables" / "numbers.tex").read_text(encoding="utf-8")
    d = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}$", t, flags=re.M))
    want = {"QThetaA": "$-0.253$", "QThetaACI": "$[-0.325,-0.179]$", "LThetaA": "$+0.058$", "LThetaACI": "$[-0.007,+0.121]$",
            "DiffTheta": "$+0.311$", "DiffThetaCI": "$[+0.214,+0.407]$"}
    src = "E1_REPORT.md line 30 (-0.253 [-0.325, -0.179]); E1L_REPORT.md lines 17 and 34 (+0.058 [-0.007, +0.121]; +0.311 [+0.214, +0.407])"
    norm = lambda x: x.replace(" ", "").replace("{-}", "-")
    return [{"macro": k, "value": d.get(k), "expected": v, "ok": norm(d.get(k, "")) == norm(v), "source": src} for k, v in want.items()]


def check_claims(V):
    """Prose statements that summarize several macros ('every/all Holm p < 0.001', 'in all four models')."""
    ks = list(MODELS)
    claims = {
        "H-B: every Holm p < 0.001": all(V[f"hb.{k}.holm"] == "$<0.001$" for k in ks),
        "H-B: all four 'difference' with estimate > 0": all(V[f"hb.{k}.verdict"] == "difference" and V[f"hb.{k}.est"].startswith("$+") for k in ks),
        "H-D: all four 'difference', K more disruptive (estimate > 0), Holm < 0.001": all(V[f"hd.{k}.verdict"] == "difference" and V[f"hd.{k}.est"].startswith("$+") and V[f"hd.{k}.holm"] == "$<0.001$" for k in ks),
        "Provenance: all Holm p < 0.001, K negative, V positive": all(V[f"provk.{k}.holm"] == "$<0.001$" and V[f"provv.{k}.holm"] == "$<0.001$" and V[f"provk.{k}.est"].startswith("$-") and V[f"provv.{k}.est"].startswith("$+") for k in ks),
        "NAT-K - NAT-V: all four negative, CI excludes 0": all(V[f"nkv.{k}.est"].startswith("$-") and V[f"nkv.{k}.verdict"] == "difference" for k in ks),
        "H-E: all Holm p < 0.001, INH > LOC": all(V[f"he.{k}.holm"] == "$<0.001$" and V[f"he.{k}.est"].startswith("$+") for k in ks),
        "X1: obs - pred Holm < 0.001 in each model, negative": all(V[f"x1.{k}.holm"] == "$<0.001$" and V[f"x1.{k}.d"].startswith("$-") for k in ks),
        "X1: sign predicted only in Llama-instruct": [V[f"x1.{k}.sign"] for k in ks] == ["no", "no", "no", "yes"],
        "X1: predicted positive in every model; observed opposite sign in both Qwen": all(V[f"x1.{k}.pred"].startswith("$+") for k in ks) and all(V[f"x1.{k}.obs"].startswith("$-") for k in ("qb", "qi")),
        "X1 caption: Llama-base 'no' because its observed (NAT-C1) CI includes 0": V["x1.lb.sign"] == "no" and V["natc1.lb.ci"].startswith("$[-") and "+" in V["natc1.lb.ci"].split(",")[1],
        "Locked tests cited in abstract/conclusions have Holm < 0.05 (provenance, H-E, NAT-C1 QB/QI/LI, family)": all(V[f"{t}.{k}.holm"] == "$<0.001$" for t in ("provk", "provv", "he") for k in ks) and all(float(V[f"natc1.{k}.holm"].replace("$<", "").strip("$")) < 0.05 for k in ("qb", "qi", "li")) and V["hf.fam.holm"] == "$<0.001$",
        "Interaction Holm < 0.05 (both analyses)": float(V["hf.int.holm"].strip("$")) < 0.05 and float(V["hf.int.holmrecomp"].strip("$")) < 0.05,
        "X4: Qwen P_logit Spearman CIs include 0 ('near zero'); P_norm higher than P_logit in Qwen": all(V[f"x4.{k}.logit.ci"].startswith("$[-") and "+" in V[f"x4.{k}.logit.ci"].split(",")[1] for k in ("qb", "qi")) and all(float(V[f"x4.{k}.norm"].strip("$")) > float(V[f"x4.{k}.logit"].strip("$")) for k in ("qb", "qi")),
        "X4: P_anormK follows the family pattern (negative in Qwen, positive in Llama)": all(V[f"x4.{k}.anormk"].startswith("$-") for k in ("qb", "qi")) and all(V[f"x4.{k}.anormk"].startswith("$+") for k in ("lb", "li")),
        "X5: positive in all four, Holm < 0.001": all(V[f"x5.{k}.est"].startswith("$+") and V[f"x5.{k}.holm"] == "$<0.001$" for k in ks),
        "X7: x < 0 at every scored step in every model": all(V[f"x7a.{k}.nneg"] == V[f"x7a.{k}.nsteps"] for k in ks),
        "X7: x-bar for P_val positive in every model": all(V[f"x7a.{k}.xval"].startswith("$+") for k in ks),
        "X9a: supported in all four; largest Holm p is Llama-base's": all(V[f"x9a.{k}.verdict"] == "supported" for k in ks) and V["x9a.lb.holm"] not in ("$<0.001$",) and all(V[f"x9a.{k}.holm"] == "$<0.001$" for k in ("qb", "qi", "li")),
        "X9a: every Spearman(lnF, theta) CI includes 0": all(V[f"x9a.{k}.rhoci"].startswith("$[-") and "+" in V[f"x9a.{k}.rhoci"].split(",")[1] for k in ks),
        "X9b: inconclusive in both Qwen models; Llama descriptive": [V[f"x9b.{k}.verdict"] for k in ks] == ["inconclusive", "inconclusive", "descriptive", "descriptive"],
        "X9c: Qwen D negative with and without layer 0": all(V[f"x9c.{k}.{d}"].startswith("$-") for k in ("qb", "qi") for d in ("D", "D0")),
        "X9d: layers 1, 2 and 8 inflated in both Qwen models; none in Llama": V["x9d.qb.biglayers"] == V["x9d.qi.biglayers"] == "1, 2 and 8" and V["x9d.lb.biglayers"] == V["x9d.li.biglayers"] == "none",
        "X10: Llama K below benchmark (ratio < 1); Qwen K > 10x": all(float(V[f"x10.{k}.K.ratio"].strip("$")) < 1 for k in ("lb", "li")) and all(float(V[f"x10.{k}.K.ratio"].strip("$")) > 10 for k in ("qb", "qi")),
        "X11: observed H-B ratio below benchmark B in every model": all(float(V[f"hb.{k}.ratio"].strip("$")) < float(V[f"x11.{k}.B"].strip("$")) for k in ks),
        "X12: neither tuning effect's CI excludes 0; opposite signs": all(V[f"x12.{k}.ci"].startswith("$[-") and "+" in V[f"x12.{k}.ci"].split(",")[1] for k in ("q", "l")) and V["x12.q.est"].startswith("$-") and V["x12.l.est"].startswith("$+"),
        "X9 reproduction: all problems bitwise": V["x9.repro.ok"] == V["x9.repro.n"],
        "NAT arith JS equals the earlier studies' value (QB = E1, LI = E1L)": V["x6.qb.nat.arith"] == V["e1.js.nat"] and V["x6.li.nat.arith"] == V["e1l.js.nat"],
        "H-A: supported in both Llama, contradicted in both Qwen": [V[f"ha.{k}.word"] for k in ks] == ["contradicted", "contradicted", "supported", "supported"],
        "Phase 8a: H-F readings family supported, tuning contradicted, additivity rejected": (V["hf.word.fam"], V["hf.word.tun"], V["hf.word.add"]) == ("supported", "contradicted", "rejected") and V["hf.tun.verdict"] == "equivalent",
        "Phase 8a: unclamped c below the 0.10 clamp in all four (margins set by the clamp)": all(float(V[f"cal.{k}.craw"].strip("$")) < 0.10 and V[f"cal.{k}.c"] == "$0.10$" for k in ks),
        "Phase 8a: Llama REL arms within [x10Kllama.min, x10V.max] of the benchmark, and x10V.max is a Llama value": all(float(V["rng.x10Kllama.min"].strip("$")) <= float(V[f"x10.{k}.{p}.ratio"].strip("$")) <= float(V["rng.x10V.max"].strip("$")) for k in ("lb", "li") for p in ("K", "V")) and V["rng.x10V.max"] in (V["x10.lb.V.ratio"], V["x10.li.V.ratio"]),
        "Phase 8a: C1-V within rng.rdevmaxllama.max in every model (from the locked report)": all(
            RDEV[m]["C1-V"]["max"] <= float(V["rng.rdevmaxllama.max"]) for m in MODELS.values()),
        "Phase 8a: LOC potency > 1 in every model (LOC share of NAT divergence exceeds its weight share)": all(float(V[f"arch.{k}.potency"].strip("$")) > 1 for k in ks),
        "Phase 8a: hcx.g0rev = -hcx.g0": abs(float(V["hcx.g0rev"].strip("$")) + float(V["hcx.g0"].strip("$"))) < 1e-9,
        "Phase 8a: Llama-base is the H-C 'not applicable' model and QB/LI share the band": V["hc.lb.band"] == "not applicable" and V["hc.qb.band"] == V["hc.li.band"],
        "Phase 8c: X13a supported on both paths in all four": all(V[f"x13a.{k}.{P}.verdict"] == "supported" for k in ks for P in "KV"),
        "Phase 8c: X13b V direction supported, band partial, in all four; manipulation passed everywhere":
            all(V[f"x13b.{k}.V.dir"] == "supported" and V[f"x13b.{k}.V.band"] == "partial" for k in ks)
            and all(V[f"x13b.{k}.{P}.manok"] == "yes" for k in ks for P in "KV"),
        "Phase 8c: X13b K inconclusive in all four; Qwen K negative, Llama K positive; readings K mixed, V supported":
            all(V[f"x13b.{k}.K.dir"] == "inconclusive" for k in ks) and all(V[f"x13b.{k}.K"].startswith("$-") for k in ("qb", "qi"))
            and all(V[f"x13b.{k}.K"].startswith("$+") for k in ("lb", "li")) and (V["x13.readK"], V["x13.readV"]) == ("mixed", "supported"),
        "Phase 8c: X13b Llama K CIs exclude 0 but Holm p > 0.05": all(V[f"x13b.{k}.K.ci"].startswith("$[+") for k in ("lb", "li"))
            and all(float(V[f"x13b.{k}.K.holm"].strip("$")) > 0.05 for k in ("lb", "li")),
        "Phase 8c: X13d supported in both Qwen models and contrasts; no >5% step after correction":
            all(V[f"x13d.{k}.{c}.verdict"] == "supported" for k in ("qb", "qi") for c in ("c1", "c1k")) and V["x13.ncgt5"] == "0",
        "Phase 8c: X13c D_c positive with CI > 0 in all four": all(V[f"x13c.{k}.Dc"].startswith("$+") and V[f"x13c.{k}.Dcci"].startswith("$[+") for k in ks),
        "Phase 8c: X13 NAT reproduction complete and nc_ok never false": V["x13.repro.ok"] == V["x13.repro.n"] and V["x13.nc.bad"] == "0",
        "Phase 8c: GPU grand total = v3 + X9 + X13 (2 dp)": abs(float(V["gpu.grand"]) - (float(V["gpu.total"]) + float(V["x9.gpu.total"]) + float(V["x13.gpu.total"]))) <= 0.011,  # printed parts carry 0.005 + 0.0005 + 0.0005 rounding, the total 0.005
        "Phase 8a: r80 median below 8 in every model": all(float(V[f"cal.{k}.r80"]) < 8 for k in ks),
        "H-C: |beta1| > |beta0| in all four": all(abs(float(V[f"hc.{k}.b1"].strip("$"))) > abs(float(V[f"hc.{k}.b0"].strip("$"))) for k in ks),
        "H-C bands: no (QB, LI), inconclusive (QI), not applicable (LB)": [V[f"hc.{k}.band"] for k in ks] == ["no", "inconclusive", "not applicable", "no"],
        "C2 - C1 negative with CI excluding 0 in all four": all(V[f"c2c1.{k}.est"].startswith("$-") and V[f"c2c1.{k}.verdict"] == "difference" for k in ks),
        "Pooled six-arm H-A positive in all four (CI lower bound > 0)": all(V[f"hapool.{k}.ci"].startswith("$[+") for k in ks),
        "X2c negative in every model": all(V[f"x2c.{k}.est"].startswith("$-") for k in ks),
        "X2a/X2b: Qwen negative, Llama-base positive, Llama-instruct inconclusive": all(V[f"{e}.{k}.est"].startswith("$-") for e in ("x2a", "x2b") for k in ("qb", "qi")) and all(V[f"{e}.lb.est"].startswith("$+") for e in ("x2a", "x2b")) and all(V[f"{e}.li.verdict"] == "inconclusive" for e in ("x2a", "x2b")),
        "Tuning step opposite in the two families (QI < QB, LI > LB in NAT-C1)": float(V["natc1.qi.est"].strip("$")) < float(V["natc1.qb.est"].strip("$")) and float(V["natc1.li.est"].strip("$")) > float(V["natc1.lb.est"].strip("$")),
        "H-F guard not triggered (max SD < limit)": float(V["hf.sdmax"].strip("$")) < float(V["hf.guard"].strip("$")),
        "Llama-base NAT-C1 equivalent; others difference": [V[f"natc1.{k}.verdict"] for k in ks] == ["difference", "difference", "equivalent", "difference"],
        "H-C cross precondition met (g0 CI excludes 0, negative)": V["hcx.g0ci"].startswith("$[-") and "-" in V["hcx.g0ci"].split(",")[1],
    }
    return [{"claim": c, "ok": bool(v)} for c, v in claims.items()]


def main():
    V = macros()
    used, undefined = check_used(V)
    rec = check_recompute(V)
    lit = literals()
    e1 = e1_macros()
    cl = check_claims(V)
    src = json.loads((PJ / "tables_v3" / "numbers_v3_sources.json").read_text(encoding="utf-8"))
    res = {"n_macros_defined": len(V), "n_macros_used": len(used), "undefined_used": undefined,
           "unused_macros": sorted(set(V) - used), "recompute_checks": len(rec), "recompute_failures": [r for r in rec if not r["ok"]],
           "literals": lit, "unclassified_literals": [x for x in lit if x["class"] == "UNCLASSIFIED"], "e1_macros": e1, "summary_claims": cl}
    (HERE / "V3_NUMBER_AUDIT.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    L = ["# Number audit of main_v3.tex (generated by audit/audit_v3_numbers.py)", "",
         f"- v3 macros defined: {len(V)}; used in the text and tables: {len(used)}; used but undefined: {len(undefined)}",
         f"- macro values re-checked against the independent recomputation: {len(rec)}; failures: {len(res['recompute_failures'])}",
         f"- E1/E1L macros cited in the main text checked against the E1/E1L reports: {sum(x['ok'] for x in e1)}/{len(e1)} agree",
         f"- literal numbers in the v3 main text: {len(lit)}; unclassified: {len(res['unclassified_literals'])}",
         f"- prose summary claims checked against the macros: {sum(c['ok'] for c in cl)}/{len(cl)} hold", "",
         "## Summary claims", ""] + [f"- [{'ok' if c['ok'] else 'FAILS'}] {c['claim']}" for c in cl] + ["",
         "## Recomputation failures", ""]
    L += [f"- `{r['macro']}` printed {r['printed']} vs recomputed {r['recomputed']}" for r in res["recompute_failures"]] or ["none"]
    L += ["", "## Literal numbers in the main text", "", "| File:line | Literal | Classification | Context |", "|---|---|---|---|"]
    for x in lit:
        L.append(f"| {x['file']}:{x['line']} | `{x['literal']}` | {x['class']} | {x['context'].replace('|', '/')} |")
    L += ["", "## Every v3 macro and its source", "", "| Macro | Value | Source |", "|---|---|---|"]
    for k in sorted(V):
        L.append(f"| `{k}` | {V[k].replace('|', '/')} | {src.get(k, {}).get('source', '?')} |")
    (HERE / "V3_NUMBER_AUDIT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print({k: (v if not isinstance(v, list) else len(v)) for k, v in res.items()})


if __name__ == "__main__":
    main()
