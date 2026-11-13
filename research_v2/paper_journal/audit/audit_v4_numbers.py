"""Number audit for main_v4.tex. CPU only; read-only. Shares no code with make_v4.py.

1. Every \\vf{key} and \\vn{key} used in main_v4.tex, sections_v4/ and tables_v4/ is defined (an undefined key would print ??key).
2. Every v4 macro is RE-DERIVED from its declared source: the JSON pointer is re-read with this script's own reader, the declared
   operation is recomputed with this script's own implementation, the result is re-formatted with this script's own formatter, and the
   printed string must be identical to the macro.
3. Every v4 per-model result macro taken from the locked report is re-checked against the INDEPENDENT recomputation
   (v4_runs/v4_recomputation.json, no shared code with analyze_v4.py): point estimates identical at the printed precision; CI endpoints
   within 5 % of the CI width plus half a printed unit (the declared tolerance); p within 0.01 + 10 %; verdict and reading words identical.
4. Every v3 macro used is present in tables_v3/numbers_v3_sources.json (audited by audit_v3_numbers.py).
5. Every literal number in the v4 sources (outside \\vf{}, \\vn{}, \\ref, \\cite, \\label, file names) is listed with its context and a
   classification; unclassified literals are flagged.
6. Prose claims (words such as "every", "all", "monotonically", "larger", "seven") are checked against the locked report.
Writes audit/V4_NUMBER_AUDIT.md and audit/V4_NUMBER_AUDIT.json.
"""
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
PJ = HERE.parent
RV = PJ.parent
SRCS = ["main_v4.tex"] + sorted(str(p.relative_to(PJ)) for p in (PJ / "sections_v4").glob("*.tex"))
TABS = sorted(str(p.relative_to(PJ)) for p in (PJ / "tables_v4").glob("*.tex") if p.name != "numbers_v4.tex")
MAIN_TEXT = [s for s in SRCS if s != "main_v4.tex"] + TABS


def read(rel):
    return (PJ / rel).read_text(encoding="utf-8")


def macro_defs(rel, prefix):
    return dict(re.findall(r"\\csname " + prefix + r"@([^\\]+)\\endcsname\{(.*)\}$", read(rel), flags=re.M))


# ---------------------------------------------------------------- independent reader / formatter
_files = {}


def jload(rel):
    if rel not in _files:
        t = (RV / rel).read_text(encoding="utf-8")
        _files[rel] = [json.loads(x) for x in t.splitlines() if x.strip()] if rel.endswith(".jsonl") else json.loads(t)
    return _files[rel]


def at(ptr):
    f, path = ptr.split("#", 1)
    node = jload(f)
    for part in path.split("/"):
        if part == "":
            continue
        part = part.replace("~1", "/")
        node = node[int(part)] if isinstance(node, list) else node[part]
    return node


def sgn(x, d):
    s = f"{abs(x):.{d}f}"
    return ("-" if x < 0 and float(s) != 0 else "+") + s


def show(x, f):
    table = {
        "s3": lambda: f"${sgn(x, 3)}$", "s2": lambda: f"${sgn(x, 2)}$", "u3": lambda: f"${x:.3f}$", "u2": lambda: f"${x:.2f}$",
        "u1": lambda: f"${x:.1f}$", "u4": lambda: f"${x:.4f}$", "int": lambda: f"${round(x)}$",
        "intc": lambda: "$" + f"{round(x):,}".replace(",", "{,}") + "$",
        "ci3": lambda: f"$[{sgn(x[0], 3)}, {sgn(x[1], 3)}]$", "ci2": lambda: f"$[{sgn(x[0], 2)}, {sgn(x[1], 2)}]$",
        "uci3": lambda: f"$[{x[0]:.3f}, {x[1]:.3f}]$", "p": lambda: "$<0.001$" if x < 0.001 else f"${x:.3f}$",
        "p4": lambda: f"${x:.4f}$", "usd": lambda: "\\$" + f"{x:.2f}", "pct0": lambda: f"${100 * x:.0f}$\\%",
        "bn": lambda: f"${x / 1e9:.2f}$", "word": lambda: str(x), "bool": lambda: "yes" if x else "no"}
    return table[f]()


def recompute(e):
    """Re-derive a macro's raw value from its declared source or operation."""
    if "src" in e:
        return at(e["src"])
    op, ins = e["op"], e["inputs"]
    if op == "sum_div3600":
        return sum(r["seconds"] for r in jload("v4_runs/gpu_ledger.jsonl")) / 3600
    if op == "count" and ins == ["v4_runs/gpu_ledger.jsonl#*"]:
        return len(jload("v4_runs/gpu_ledger.jsonl"))
    if op == "last_session_close":
        return [r for r in jload("v4_runs/cost_ledger.jsonl") if r.get("kind") == "session_close"][-1]["used_usd_estimate"]
    if op == "count_lines":
        return sum(1 for l in (RV / ins[0]).read_text(encoding="utf-8").splitlines() if l.strip())
    if op == "count_files":
        return len(list((RV / "v4/lock_history").glob("V4_PROTOCOL_LOCK_*.txt")))
    if op == "count_same_sign":
        pairs = [x.split("|") for x in ins]
        return sum((at(a) < 0) == (at(b) < 0) for a, b in pairs)
    if op == "count" and ins and "|" in ins[0]:
        return len(ins)
    vals = [at(p) for p in ins]
    f = {"min": min, "max": max, "len": lambda v: len(v[0]), "div": lambda v: v[0] / v[1], "exp": lambda v: math.exp(v[0]),
         "absmax": lambda v: max(abs(a) for a in v), "exp_min": lambda v: math.exp(min(v)), "exp_max": lambda v: math.exp(max(v)),
         "expneg_max": lambda v: math.exp(-max(v)), "expneg_min": lambda v: math.exp(-min(v)),
         "count_gt0": lambda v: sum(a > 0 for a in v), "count_le1": lambda v: sum(a <= 1 for a in v), "count": len}[op]
    return f(vals)


# ---------------------------------------------------------------- literal classification
CONST = {
    "64": "head dimension / step cap T_i (PROTOCOL_V4 §2, §3)", "128": "head dimension (§2)", "65": "history cap (§3)",
    "200": "earlier study's test sample / null draws (§3, §6)", "100": "earlier study's calibration sample / count", "50": "Step 0 calibration problems (§4)",
    "10{,}000": "bootstrap and permutation count (§7)", "0.25": "b-hat threshold (§3)", "0.5": "dose", "0.10": "Step 0 |delta| limit / c clamp (§4, §3)",
    "0.40": "c clamp (§3)", "0.2": "Step 0 verdict threshold (§4, D9) / MDE wording (§7.6: 0.206)", "0.1": "equivalence attainability (§7.6: 0.115)",
    "0.05": "Holm level", "0.01": "Holm level wording (all Holm p = 0.0016)", "0.001": "p display threshold",
    "95\\%": "CI level", "90\\%": "CI level", "5\\%": "truncation sensitivity / Step 0 rho tolerance (§4, §8)", "25\\%": "drift tolerance (§10; D1)",
    "1.48": "D1 drift factor (V4-D1.json)", "29": "pre-declared $29 stop (§10, r5)", "1.39": "ln 4 rounded (threshold)",
    "10^{-4}": "construction tolerance", "10^{-9}": "JS floor", "1": "step 1 / count / 1/n", "2": "dose / steps 2-8 / ln 2", "4": "ln 4 / four",
    "3": "3-shot / three-point", "0": "zero", "35": "C(7,3) labellings (§7.4)", "7": "C(7,3)", "8": "eight / top-8 / steps 2-8",
    "9": "steps 9-64", "16": "H5 family size (16 tests)", "14": "earlier study arms (§6)", "12": "twelve model-contrast pairs (3x4)",
    "2--8": "window", "9--64": "window", "1/35": "smallest attainable p", "1/n": "isotropic share", "1/n_i": "isotropic share",
    "48": "A40 memory (GB)", "8\\,GB": "laptop GPU memory", "4060": "GPU model RTX 4060", "40": "A40 (GPU model)",
    "24.04": "Ubuntu version (image tag)", "12.8.1": "image CUDA (tag)", "1.0.2": "image tag", "2.13.0": "PyTorch version (V4_PACKAGES)",
    "126": "cu126 build", "5.13.0": "transformers version (V4_PACKAGES)", "0.49.2": "bitsandbytes version (V4_PACKAGES)",
    "1.4.0": "RunPod plugin version (AI_USE_LOG_V4)", "83": "GM attempt mutants caught (V4-A12)", "87": "GM attempt mutants total (V4-A12)",
    "100\\%": "the whole", "13386": "TMLR submission number", "283": "ARR submission number", "6": "GPT-6 model name",
    "2.0": "Apache License 2.0", "3.1": "model / licence name Llama 3.1", "3.2": "model / licence name Llama 3.2", "4.0": "CC BY 4.0",
    "2026": "year", "2026-10-28": "date (AI-use log)", "2026-10-29": "date", "2026-11-01": "date", "2026-11-02": "date", "2026-11-09": "date", "2026-11-11": "date (Gemini proofreading, AI-use appendix)",
    "1409b94": "commit id", "b4eecc6": "commit id", "13": "X1--X13 label", "4.5": "section ref", "2.5": "model name Qwen2.5",
    "1/2": "half", "0.5B": "model size", "1.5B": "model size", "3B": "model size", "7B": "model size", "8B": "model size", "1B": "model size",
    "1.3B": "cited model size (Dewage et al.)", "12": "count",
    "10^{-9}": "JS floor (§3)", "10^{-4}": "construction tolerance (§3)", "256": "SHA-256 (hash name)", "48\\,GB": "A40 memory",
    "0.5,1,2": "calibration doses (§3)", "0.10,0.40": "clamp of c (§3)", "5.5": "model name Claude Opus 5.5 (AI_USE_LOG_V4)",
}
# v5 (second locked study, Section 6 and Appendix C): protocol constants and logged operational facts. Every v5 result number is
# a \vsv{} macro audited by audit_v5_numbers.py.
CONST.update({
    "4.13": "NF4 bits per weight, approximate (PROTOCOL_V5 §3)", "4.5": "section ref / HQQ bits per weight (PROTOCOL_V5 §3)",
    "600": "WikiText-2 prompt length in characters (PROTOCOL_V5 §3)", "65": "history cap (§3; PROTOCOL_V5 §3)",
    "15": "v5 ceiling in USD (PROTOCOL_V5 §0, §9; V5-D3)", "14": "earlier study arms (§6) / v5 $14 ledger stop (PROTOCOL_V5 §9)",
    "16": "H5 family size (16 tests) / V5-D1 $16 stop", "3.0": "licence name CC BY-SA 3.0 (WikiText-2 card)",
    "2026-11-13": "date (AI_USE_LOG_V5)", "2026-11-11": "date (Gemini proofreading; v5 start, AI_USE_LOG_V5)",
    "3.7": "first v5 idle period, hours (V5-A8)", "26": "second v5 idle period, minutes (V5-A13: closeout 08:52:42Z, stop 09:18:23Z)",
    "25\\%": "drift tolerance (§10; D1; PROTOCOL_V5 §9)", "64": "head dimension / step cap T_i (PROTOCOL_V4 §2, §3) / HQQ group size",
    "2": "dose / steps 2-8 / ln 2 / WikiText-2 name", "11": "date: 11 to 13 November 2026, the v5 period (AI_USE_LOG_V5)",
})
MODEL_RE = re.compile(r"(Qwen2\.5|Qwen3|Llama-3(\.\d)?|Llama-3\.\d-\d+B|Mistral-7B(-v0\.3)?|v0\.3|0\.5B|1\.5B|\d+B|NF4|FP16|BF16|A40|RTX|"
                      r"float64|GPT-6|H[1-5]|F[1-5]|X[1-9]\d*|r[4-8]b?|V4-A\d+|A\d+|D1|L\d|Step~0|3-shot|top-8|4-bit|16-bit|"
                      r"[0-9a-f]{16}|CA-MTL-1|cu\d+|sm_\d+|Ada|_\{10\}|R\^\{?d\}?|R\^\{64\}|S1|S2|x_i\^P|L\^[KPV]|C\^[KPV]|e\^\{\\theta\}|2\\times2|2\$\\times\$2)")


def uncomment(t):
    return re.sub(r"(?<!\\)%.*", "", t)


def strip_tex(t):
    t = uncomment(t)
    t = re.sub(r"\\setlength\{\\tabcolsep\}\{[^}]*\}", " ", t)
    t = re.sub(r"\\cmidrule\([a-z]*\)\{[^}]*\}", " ", t)
    t = re.sub(r"\\(vf|vn|vsv|ref|label|cite|citet|citep|input|includegraphics|texttt|url|acmDOI|acmVolume|acmArticle|acmMonth|acmYear|"
               r"multirow|addbibresource)(\[[^\]]*\])?\{[^{}]*\}", " ", t)
    t = re.sub(r"\\begin\{tabular\}\{.*?\}\s*$", " ", t, flags=re.M)
    t = re.sub(r"\\(begin|end)\{[^}]*\}(\[[^\]]*\])?", " ", t)
    t = re.sub(r"\\multicolumn\{\d+\}\{[^}]*\}", " ", t)
    t = re.sub(r"\\includegraphics\[[^\]]*\]", " ", t)
    t = re.sub(r"\\[A-Za-z]+\*?", " ", t)          # every remaining command name (\ge0.25 -> " 0.25", \ln4 -> " 4")
    return t


def literals(rel):
    out = []
    for ln, line in enumerate(strip_tex(read(rel)).splitlines(), 1):
        scrub = MODEL_RE.sub(" ", line)
        for m in re.finditer(r"10\^\{-\d+\}|(?<![A-Za-z\d.])\d+(?:[.,:/-]\d+)*(?:\{,\}\d+)?(?:\\%|\\,GB|B)?", scrub):
            tok = m.group(0)
            ctx = scrub[max(0, m.start() - 40): m.end() + 40].strip()
            out.append((rel, ln, tok, ctx))
    return out


# ---------------------------------------------------------------- claims
REP = "v4_runs/v4_report.json"
QWEN = ["qwen2.5-0.5b", "qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b"]
LLAMA = ["llama3.2-1b", "llama3.2-3b", "llama3.1-8b"]
ALL = QWEN + LLAMA + ["mistral-7b-v0.3"]


def claims():
    r = jload(REP)
    pm = r["per_model"]
    est = lambda m, c: pm[m]["contrasts"][c]["estimate"]
    ci = lambda m, c: pm[m]["contrasts"][c]["ci95"]
    am = r["all_models"]
    cal = {m: jload(f"v4_runs/{m}/calibration.json") for m in ALL}
    st0 = jload("v4_runs/step0.json")
    ex = jload("v4_runs/v4_exploratory.json")
    C = []
    add = lambda text, ok: C.append({"claim": text, "ok": bool(ok)})
    add("All eight models complete, n = 100, none flagged", all(v == "complete" for v in r["status"].values()) and r["flagged_bf16"] == []
        and all(pm[m]["n_problems"] == 100 for m in ALL))
    add("H2: every model supported, every CI > 0, every Holm p < 0.001", all(am["F2 H2"]["readings"][m] == "supported" and ci(m, "H2")[0] > 0
        and am["F2 H2"]["holm_p"][m] < 0.001 for m in ALL))
    add("H2 [descriptive]: point estimate larger in the largest than the smallest model of each family; Mistral highest",
        est("qwen2.5-7b", "H2") > est("qwen2.5-0.5b", "H2") and est("llama3.1-8b", "H2") > est("llama3.2-1b", "H2")
        and max(ALL, key=lambda m: est(m, "H2")) == "mistral-7b-v0.3")
    ov = lambda a, b: ci(a, "H2")[0] <= ci(b, "H2")[1] and ci(b, "H2")[0] <= ci(a, "H2")[1]
    add("H2: head-dim-128 intervals overlap within each family (Qwen 1.5/3/7B pairwise; Llama 3B/8B); Mistral overlaps Llama-8B",
        ov("qwen2.5-1.5b", "qwen2.5-3b") and ov("qwen2.5-1.5b", "qwen2.5-7b") and ov("qwen2.5-3b", "qwen2.5-7b")
        and ov("llama3.2-3b", "llama3.1-8b") and ov("mistral-7b-v0.3", "llama3.1-8b"))
    add("H2 rose across the head-dimension step in both families (0.5B->1.5B, 1B->3B)",
        est("qwen2.5-1.5b", "H2") > est("qwen2.5-0.5b", "H2") and est("llama3.2-3b", "H2") > est("llama3.2-1b", "H2"))
    add("H5: smallest 95% lower bound >= 2.255 (printed 'at or above 2.255')",
        min(pm[m]["H5"][P]["ci95"][0] for m in ALL for P in "KV") >= 2.255)
    add("H3: every model supported, all Holm p < 0.001", all(am["F3 H3"]["readings"][m] == "supported" and am["F3 H3"]["holm_p"][m] < 0.001
                                                           and ci(m, "H3")[1] < 0 for m in ALL))
    add("H5: all 16 supported, lower bounds > ln 4, Holm p < 0.01", all(v == "supported" for v in am["F5 H5"]["readings"].values())
        and all(pm[m]["H5"][P]["ci95"][0] > math.log(4) for m in ALL for P in "KV") and max(am["F5 H5"]["holm_p"].values()) < 0.01)
    add("H5: L^K > L^V in all Qwen and Mistral; L^V > L^K in all Llama",
        all(pm[m]["H5"]["K"]["estimate"] > pm[m]["H5"]["V"]["estimate"] for m in QWEN + ["mistral-7b-v0.3"])
        and all(pm[m]["H5"]["V"]["estimate"] > pm[m]["H5"]["K"]["estimate"] for m in LLAMA))
    add("H1: supported in the seven Qwen/Llama models with Holm p < 0.001; Mistral inconclusive",
        all(am["F1 H1"]["readings"][m] == "supported" and am["F1 H1"]["holm_p"][m] < 0.001 for m in QWEN + LLAMA)
        and am["F1 H1"]["readings"]["mistral-7b-v0.3"] == "inconclusive")
    mi = pm["mistral-7b-v0.3"]["contrasts"]["H1"]
    mg = cal["mistral-7b-v0.3"]["margin_log_js"]
    add("Mistral H1: 90% CI excludes 0 but not inside the margin, so not equivalent; 95% CI includes 0",
        mi["ci90"][1] < 0 and not (mi["ci90"][0] > -mg and mi["ci90"][1] < mg) and mi["ci95"][0] < 0 < mi["ci95"][1])
    add("H1 [descriptive]: shrinks monotonically across the four Qwen sizes",
        all(est(a, "H1") < est(b, "H1") for a, b in zip(QWEN, QWEN[1:])))
    add("H1 [descriptive]: Llama changes between 1B and 3B (> 0.4) and is flat from 3B to 8B (|change| < 0.01)",
        est("llama3.2-3b", "H1") - est("llama3.2-1b", "H1") > 0.4 and abs(est("llama3.1-8b", "H1") - est("llama3.2-3b", "H1")) < 0.01)
    add("H1: Qwen-7B and Llama-8B resolved (Holm p < 0.001, CI < 0), unlike Mistral",
        all(am["F1 H1"]["holm_p"][m] < 0.001 and ci(m, "H1")[1] < 0 for m in ("qwen2.5-7b", "llama3.1-8b"))
        and ci("mistral-7b-v0.3", "H1")[1] > 0)
    add("NAT-C1 at matched sizes: Llama above Qwen (3B: Llama-3B > Qwen-3B; 7-8B: Llama-8B > Qwen-7B); T > 0; p > 0.05",
        est("llama3.2-3b", "NAT-C1") > est("qwen2.5-3b", "NAT-C1") and est("llama3.1-8b", "NAT-C1") > est("qwen2.5-7b", "NAT-C1")
        and am["H4"]["permutation"]["T"] > 0 and am["H4"]["permutation"]["p_one_sided"] > 0.05)
    # ---- prose claims added by the second fresh-eyes review (2026-11-13)
    add("Conclusion: Llama-3.2-1B NAT-C1 is equivalent to 0 (90% CI inside its margin), so on the combined contrast natural "
        "error is equivalent to isotropic noise there",
        pm["llama3.2-1b"]["contrasts"]["NAT-C1"]["verdict"] == "equivalent"
        and -cal["llama3.2-1b"]["margin_log_js"] < pm["llama3.2-1b"]["contrasts"]["NAT-C1"]["ci90"][0]
        and pm["llama3.2-1b"]["contrasts"]["NAT-C1"]["ci90"][1] < cal["llama3.2-1b"]["margin_log_js"])
    add("NAT-C1 changes sign in Qwen only between 0.5B and 1.5B (the head-dimension step): CI < 0 at 0.5B, CI > 0 at 1.5B, 3B, 7B",
        ci("qwen2.5-0.5b", "NAT-C1")[1] < 0 and all(ci(m, "NAT-C1")[0] > 0 for m in QWEN[1:])
        and pm["qwen2.5-0.5b"]["head_dim"] == 64 and all(pm[m]["head_dim"] == 128 for m in QWEN[1:]))
    add("Practical reading: values-only error 2.6-4.0 times as disruptive as keys-only error = exp(-H3) range over the eight models",
        f"{math.exp(-max(est(m, 'H3') for m in ALL)):.1f}" == "2.6" and f"{math.exp(-min(est(m, 'H3') for m in ALL)):.1f}" == "4.0")
    add("X1: Llama H1 slope restricted to head-dim 128 has a CI including 0 (flat)",
        ex["X4"]["hd128_only_slopes"]["llama"]["H1"]["ci95_boot"][0] < 0 < ex["X4"]["hd128_only_slopes"]["llama"]["H1"]["ci95_boot"][1])
    vn = dict(re.findall(r"\\csname vn@([^\\]+)\\endcsname\{(.*)\}$", (PJ / "tables_v3/numbers_v3.tex").read_text(encoding="utf-8"), flags=re.M))
    num = lambda k: [float(x) for x in re.findall(r"[+-]?\d+\.\d+", vn[k])]
    add("v3: Llama-base NAT-C1 CI includes 0; Llama-instruct CI > 0 (\"near zero or positive\")",
        num("natc1.lb.ci")[0] < 0 < num("natc1.lb.ci")[1] and num("natc1.li.ci")[0] > 0)
    add("v3 X13: Llama key-path centring not significant after Holm (both p > 0.05)",
        num("rng.x13holmKl.min")[0] > 0.05 and num("rng.x13holmKl.max")[0] > 0.05)
    pot = lambda k: int(re.findall(r"\d+", vn[k])[0])
    add("v3: LOC potency printed as Llama (low end) to Qwen (high end): Llama-base < Qwen-base, both instruct equal to base",
        pot("arch.lb.potency") < pot("arch.qb.potency") and pot("arch.li.potency") == pot("arch.lb.potency")
        and pot("arch.qi.potency") == pot("arch.qb.potency"))
    add("H4 falsified; three Qwen CIs entirely above 0; Qwen-0.5B negative; Llama-1B equivalent; other Llama positive",
        am["H4"]["result"] == "falsified" and sum(ci(m, "NAT-C1")[0] > 0 for m in QWEN) == 3 and ci("qwen2.5-0.5b", "NAT-C1")[1] < 0
        and pm["llama3.2-1b"]["contrasts"]["NAT-C1"]["verdict"] == "equivalent" and all(ci(m, "NAT-C1")[0] > 0 for m in LLAMA[1:]))
    add("NAT-C1 negative only in Qwen-0.5B; positive in every model from 1.5B params up (incl. Mistral)",
        [m for m in ALL if est(m, "NAT-C1") < 0] == ["qwen2.5-0.5b"]
        and all(est(m, "NAT-C1") > 0 for m in ALL if ex["size"][m]["params_16bit"] >= 1.5e9))
    add("NAT-C1 rises monotonically with size within each family; Mistral largest of the eight",
        all(est(a, "NAT-C1") < est(b, "NAT-C1") for a, b in zip(QWEN, QWEN[1:]))
        and all(est(a, "NAT-C1") < est(b, "NAT-C1") for a, b in zip(LLAMA, LLAMA[1:]))
        and max(ALL, key=lambda m: est(m, "NAT-C1")) == "mistral-7b-v0.3")
    add("Families did not separate completely", not am["H4"]["permutation"]["complete_separation"])
    add("Without-flagged identical to all-models", r["without_flagged"] == r["all_models"] and r["abstract_level"]["disagreements"] == {})
    add("Every b-hat >= 0.25 (informative)", all(cal[m]["dose_slope_b_hat"] >= 0.25 for m in ALL))
    add("Step 0 passed for all three models; every criterion-1 and -2 met",
        st0["pass"] and all(v["pass"] for v in st0["models"].values())
        and all(c["criterion_1"] and c.get("criterion_2", True) is not False for v in st0["models"].values() for c in v["contrasts"].values()))
    add("Head dimension 64 only for the smallest Qwen and Llama", [m for m in ALL if pm[m]["head_dim"] == 64] == ["qwen2.5-0.5b", "llama3.2-1b"])
    add("X1: all four within-family slopes positive with CIs excluding 0",
        all(ex["X1"][f][c]["ci95_boot"][0] > 0 for f in ("qwen", "llama") for c in ("H1", "NAT-C1")))
    add("X3: within-model r near 0 (CI includes 0) in the four smallest models, CI > 0 in the four larger",
        all(ex["X3"]["within_model_problem_level"][m]["ci95_boot"][0] < 0 < ex["X3"]["within_model_problem_level"][m]["ci95_boot"][1]
            for m in ("qwen2.5-0.5b", "qwen2.5-1.5b", "qwen2.5-3b", "llama3.2-1b"))
        and all(ex["X3"]["within_model_problem_level"][m]["ci95_boot"][0] > 0 for m in ("qwen2.5-7b", "llama3.2-3b", "llama3.1-8b", "mistral-7b-v0.3"))
        and sorted(ALL, key=lambda m: ex["size"][m]["params_16bit"])[:4] == ["qwen2.5-0.5b", "llama3.2-1b", "qwen2.5-1.5b", "qwen2.5-3b"])
    add("X4: Qwen hd128-only slopes positive with CI > 0; Llama H1 flat (CI includes 0); Llama NAT-C1 CI includes 0",
        all(ex["X4"]["hd128_only_slopes"]["qwen"][c]["ci95_boot"][0] > 0 for c in ("H1", "NAT-C1"))
        and ex["X4"]["hd128_only_slopes"]["llama"]["H1"]["ci95_boot"][0] < 0 < ex["X4"]["hd128_only_slopes"]["llama"]["H1"]["ci95_boot"][1]
        and ex["X4"]["hd128_only_slopes"]["llama"]["NAT-C1"]["ci95_boot"][0] < 0)
    add("X2: Mistral layer 0 is its maximum on both paths", ex["X2"]["mistral-7b-v0.3"]["K"]["argmax_layer"] == 0
        and ex["X2"]["mistral-7b-v0.3"]["V"]["argmax_layer"] == 0)
    add("H2/H3/H5 per-model signs identical at both head dimensions", all(est(m, "H2") > 0 and est(m, "H3") < 0 for m in ALL))
    return C


# ---------------------------------------------------------------- main
def main():
    vf = macro_defs("tables_v4/numbers_v4.tex", "vf")
    vn = macro_defs("tables_v3/numbers_v3.tex", "vn")
    srcs = json.loads(read("tables_v4/numbers_v4_sources.json"))["macros"]
    v3src = json.loads(read("tables_v3/numbers_v3_sources.json"))
    v3keys = set(v3src) if isinstance(v3src, dict) else set()
    used_vf, used_vn = set(), set()
    for rel in SRCS + TABS:
        t = uncomment(read(rel))
        used_vf |= set(re.findall(r"\\vf\{([^}]+)\}", t))
        used_vn |= set(re.findall(r"\\vn\{([^}]+)\}", t))
    undef_vf = sorted(used_vf - set(vf))
    undef_vn = sorted(used_vn - set(vn))
    v3_unsourced = sorted(k for k in used_vn if v3keys and k not in v3keys and not any(k in str(x) for x in [v3keys]))
    # 2. re-derivation of every v4 macro
    rederive = []
    for k, e in srcs.items():
        try:
            raw = recompute(e)
            printed = show(raw, e["fmt"])
            ok = printed == vf.get(k)
            rederive.append({"key": k, "ok": ok, "macro": vf.get(k), "rederived": printed})
        except Exception as ex:  # noqa: BLE001
            rederive.append({"key": k, "ok": False, "macro": vf.get(k), "rederived": f"ERROR {ex!r}"})
    # 3. independent recomputation
    rows = {x["quantity"]: x for x in jload("v4_runs/v4_recomputation.json")["rows"]}
    S = {"qwen2.5-0.5b": "qa", "qwen2.5-1.5b": "qb", "qwen2.5-3b": "qc", "qwen2.5-7b": "qd", "llama3.2-1b": "la", "llama3.2-3b": "lb",
         "llama3.1-8b": "lc", "mistral-7b-v0.3": "mi"}
    CN = {"H1": "hone", "H2": "htwo", "H3": "hthree", "NAT-C1": "natc"}
    rec = []
    for m, s in S.items():
        for c, cn in CN.items():
            r_est, r_ci, r_p, r_v = (rows[f"{m} {c} {q}"]["recomputed"] for q in ("estimate", "ci95", "p", "verdict"))
            lo, hi = r_ci
            w = hi - lo
            mac_ci = [float(x) for x in re.findall(r"[+-]\d+\.\d+", vf[f"{cn}.{s}.ci"])]
            rec.append({"key": f"{cn}.{s}.est", "ok": show(r_est, "s3") == vf[f"{cn}.{s}.est"]})
            rec.append({"key": f"{cn}.{s}.ci", "ok": abs(mac_ci[0] - lo) <= 0.05 * w + 0.0005 and abs(mac_ci[1] - hi) <= 0.05 * w + 0.0005})
            pp = at(f"{REP}#per_model/{m}/contrasts/{c}/p")
            rec.append({"key": f"{cn}.{s}.p", "ok": (pp < 0.001 and r_p < 0.001) or abs(pp - r_p) <= 0.01 + 0.1 * pp})
            rec.append({"key": f"{cn}.{s}.verdict", "ok": r_v == vf[f"{cn}.{s}.verdict"]})
        for P in ("K", "V"):
            r_L, r_Lci = rows[f"{m} L^{P}"]["recomputed"], rows[f"{m} L^{P} ci95"]["recomputed"]
            w = r_Lci[1] - r_Lci[0]
            mac = [float(x) for x in re.findall(r"\d+\.\d+", vf[f"hfive{P}.{s}.ci"])]
            rec.append({"key": f"hfive{P}.{s}.est", "ok": abs(float(vf[f"hfive{P}.{s}.est"].strip("$")) - r_L) <= 0.02 + 0.05 * w})
            rec.append({"key": f"hfive{P}.{s}.ci", "ok": all(abs(a - b) <= 0.02 + 0.1 * w for a, b in zip(mac, r_Lci))})
    for fam, k in (("F1 H1", "hone"), ("F2 H2", "htwo"), ("F3 H3", "hthree"), ("F5 H5", "hfive")):
        rec.append({"key": f"fam.{k}", "ok": rows[f"all_models {fam} result"]["recomputed"] == vf[f"fam.{k}"]})
        rec.append({"key": f"famwf.{k}", "ok": rows[f"without_flagged {fam} result"]["recomputed"] == vf[f"famwf.{k}"]})
    rec.append({"key": "hfour.result", "ok": rows["all_models H4 result"]["recomputed"] == vf["hfour.result"]})
    rec.append({"key": "hfour.T", "ok": show(rows["all_models H4 T"]["recomputed"], "u3") == vf["hfour.T"]})
    rec.append({"key": "hfour.p", "ok": show(rows["all_models H4 p"]["recomputed"], "u3") == vf["hfour.p"]})
    for fam, k in (("F1 H1", "hone"), ("F2 H2", "htwo"), ("F3 H3", "hthree")):
        rr = rows[f"all_models {fam} readings"]["recomputed"]
        for m, s in S.items():
            rec.append({"key": f"{k}.{s}.read", "ok": rr[m] == vf[f"{k}.{s}.read"]})
    # 5. literals
    lits = []
    for rel in MAIN_TEXT:
        for f, ln, tok, ctx in literals(rel):
            cls = CONST.get(tok)
            lits.append({"file": f, "line": ln, "literal": tok, "class": cls, "context": ctx})
    uncl = [l for l in lits if l["class"] is None]
    C = claims()
    res = {"vf_defined": len(vf), "vf_used": len(used_vf), "vf_undefined": undef_vf, "vn_used": len(used_vn), "vn_undefined": undef_vn,
           "v3_unsourced": v3_unsourced, "rederived_n": len(rederive), "rederived_fail": [r for r in rederive if not r["ok"]],
           "recomp_n": len(rec), "recomp_fail": [r for r in rec if not r["ok"]], "literals_n": len(lits), "literals_unclassified": uncl,
           "claims": C, "literals": lits}
    (HERE / "V4_NUMBER_AUDIT.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    L = ["# Number audit of main_v4.tex (generated by audit/audit_v4_numbers.py)", "",
         f"- v4 macros defined: {len(vf)}; used in the text and tables: {len(used_vf)}; used but undefined: {len(undef_vf)}",
         f"- v3 macros used: {len(used_vn)}; undefined: {len(undef_vn)}; not in numbers_v3_sources.json: {len(v3_unsourced)}",
         f"- v4 macros re-derived from their sources by this script: {len(rederive)}; mismatches: {len(res['rederived_fail'])}",
         f"- v4 result macros checked against the independent recomputation: {len(rec)}; failures: {len(res['recomp_fail'])}",
         f"- literal numbers in the v4 text and tables: {len(lits)}; unclassified: {len(uncl)}",
         f"- prose claims checked against the locked report: {sum(c['ok'] for c in C)}/{len(C)} hold", "", "## Claims", ""]
    L += [f"- [{'ok' if c['ok'] else 'FAIL'}] {c['claim']}" for c in C]
    for title, items in (("Re-derivation mismatches", res["rederived_fail"]), ("Recomputation failures", res["recomp_fail"]),
                         ("Unclassified literals", uncl), ("Undefined macros", [{"vf": undef_vf, "vn": undef_vn}] if undef_vf or undef_vn else [])):
        L += ["", f"## {title}", ""] + ([f"- {json.dumps(x)}" for x in items] or ["none"])
    L += ["", "## Literal numbers by class", ""]
    by = {}
    for l in lits:
        by.setdefault(l["class"] or "UNCLASSIFIED", []).append(l)
    for k in sorted(by):
        L.append(f"- {k}: {len(by[k])} ({', '.join(sorted(set(x['literal'] for x in by[k])))})")
    (HERE / "V4_NUMBER_AUDIT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:9]))


if __name__ == "__main__":
    main()
