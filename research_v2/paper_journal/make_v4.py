"""Generates every v4 number macro, table and figure of main_v4.tex from saved results. CPU only; reads, never writes, research outputs.

Sources (all read-only; paths relative to research_v2/):
  v4_runs/v4_report.json              locked analysis output (sha256 checked below)
  v4_runs/v4_recomputation.json       independent recomputation
  v4_runs/v4_exploratory.json         exploratory X1-X4 (v4/EXPLORATORY_PLAN_V4.md, commit 1409b94)
  v4_runs/step0.json                  Step 0 cross-hardware check
  v4_runs/<model>/calibration.json    frozen calibrations
  v4_runs/gpu_ledger.jsonl, cost_ledger.jsonl, gm.json, gm_D1.json, design.json
  v4/V4_MODEL_MANIFEST.json, v4/deviations/V4-D1.json, v4/closeout/billing_snapshot_2026-11-09.json
Outputs: paper_journal/tables_v4/*.tex (numbers_v4.tex holds the \\vf{key} macros), paper_journal/figures_v4/*.pdf,
         paper_journal/tables_v4/numbers_v4_sources.json: key -> {val, raw, fmt, src or op+inputs}, re-derived by audit_v4.py.

  venv/Scripts/python.exe research_v2/paper_journal/make_v4.py
"""
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
RV = HERE.parent
TAB, FIG = HERE / "tables_v4", HERE / "figures_v4"
REPORT_SHA = "df57a31483160e62351378576d14123893081249e06525ed5be5f5099758dccb"

QWEN = ["qwen2.5-0.5b", "qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b"]
LLAMA = ["llama3.2-1b", "llama3.2-3b", "llama3.1-8b"]
MISTRAL = ["mistral-7b-v0.3"]
LADDER = QWEN + LLAMA + MISTRAL
S = {"qwen2.5-0.5b": "qa", "qwen2.5-1.5b": "qb", "qwen2.5-3b": "qc", "qwen2.5-7b": "qd",
     "llama3.2-1b": "la", "llama3.2-3b": "lb", "llama3.1-8b": "lc", "mistral-7b-v0.3": "mi", "llama3.2-1b-instruct": "li"}
NAME = {"qwen2.5-0.5b": "Qwen2.5-0.5B", "qwen2.5-1.5b": "Qwen2.5-1.5B", "qwen2.5-3b": "Qwen2.5-3B", "qwen2.5-7b": "Qwen2.5-7B",
        "llama3.2-1b": "Llama-3.2-1B", "llama3.2-3b": "Llama-3.2-3B", "llama3.1-8b": "Llama-3.1-8B", "mistral-7b-v0.3": "Mistral-7B-v0.3",
        "llama3.2-1b-instruct": "Llama-3.2-1B-Instruct"}
CON = {"H1": "hone", "H2": "htwo", "H3": "hthree", "NAT-C1": "natc"}
V, SRC = {}, {}
_cache = {}


# ------------------------------------------------------------------ sources
def load(rel):
    if rel not in _cache:
        p = RV / rel
        if rel.endswith(".jsonl"):
            _cache[rel] = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        else:
            _cache[rel] = json.loads(p.read_text(encoding="utf-8"))
    return _cache[rel]


def get(ptr):
    """'file#a/b/0/c' -> value."""
    f, _, path = ptr.partition("#")
    o = load(f)
    for k in [k.replace("~1", "/") for k in path.split("/") if k != ""]:
        o = o[int(k)] if isinstance(o, list) else o[k]
    return o


# ------------------------------------------------------------------ formatting (mirrored in audit_v4.py)
def fmt(x, f):
    if f == "s3": return f"${x:+.3f}$".replace("+-", "-")
    if f == "s2": return f"${x:+.2f}$"
    if f == "u3": return f"${x:.3f}$"
    if f == "u2": return f"${x:.2f}$"
    if f == "u1": return f"${x:.1f}$"
    if f == "u4": return f"${x:.4f}$"
    if f == "int": return f"${int(round(x))}$"
    if f == "nword": return ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"][int(round(x))]
    if f == "intc": return f"${int(round(x)):,}$".replace(",", "{,}")
    if f == "ci3": return f"$[{x[0]:+.3f}, {x[1]:+.3f}]$"
    if f == "ci2": return f"$[{x[0]:+.2f}, {x[1]:+.2f}]$"
    if f == "uci3": return f"$[{x[0]:.3f}, {x[1]:.3f}]$"
    if f == "p": return "$<0.001$" if x < 0.001 else f"${x:.3f}$"
    if f == "p4": return f"${x:.4f}$"
    if f == "usd": return f"\\${x:.2f}"
    if f == "pct0": return f"${100 * x:.0f}$\\%"
    if f == "bn": return f"${x / 1e9:.2f}$"
    if f == "word": return str(x)
    if f == "bool": return "yes" if x else "no"
    raise ValueError(f)


def put(key, raw, f, src=None, op=None, inputs=None):
    assert key not in V, key
    V[key] = fmt(raw, f)
    SRC[key] = {"val": V[key], "raw": raw, "fmt": f, **({"src": src} if src else {"op": op, "inputs": inputs})}


def src(key, ptr, f):
    put(key, get(ptr), f, src=ptr)


OPS = {"min": min, "max": max, "sum": sum, "id": lambda v: v[0], "div": lambda v: v[0] / v[1], "sub": lambda v: v[0] - v[1],
       "abs": lambda v: abs(v[0]), "exp": lambda v: math.exp(v[0]), "neg": lambda v: -v[0], "absmax": lambda v: max(abs(x) for x in v),
       "absmin": lambda v: min(abs(x) for x in v), "count_true": lambda v: sum(bool(x) for x in v), "len": lambda v: len(v[0]),
       "sum_div3600": lambda v: sum(v) / 3600, "pair": lambda v: [v[0], v[1]], "ratio_pct": lambda v: v[0] / v[1]}


def der(key, op, ptrs, f):
    vals = [get(p) for p in ptrs]
    put(key, OPS[op](vals), f, op=op, inputs=ptrs)


# ------------------------------------------------------------------ macros
REP, REC, EXP, ST0 = "v4_runs/v4_report.json", "v4_runs/v4_recomputation.json", "v4_runs/v4_exploratory.json", "v4_runs/step0.json"
MAN = "v4/V4_MODEL_MANIFEST.json"


def pm(m, *k):
    return f"{REP}#per_model/{m}/" + "/".join(k)


def build_models():
    for m in LADDER:
        s = S[m]
        for c, cn in CON.items():
            src(f"{cn}.{s}.est", pm(m, "contrasts", c, "estimate"), "s3")
            src(f"{cn}.{s}.ci", pm(m, "contrasts", c, "ci95"), "ci3")
            src(f"{cn}.{s}.cininety", pm(m, "contrasts", c, "ci90"), "ci3")
            src(f"{cn}.{s}.p", pm(m, "contrasts", c, "p"), "p")
            src(f"{cn}.{s}.verdict", pm(m, "contrasts", c, "verdict"), "word")
        for fam, c in (("F1 H1", "hone"), ("F2 H2", "htwo"), ("F3 H3", "hthree")):
            src(f"{c}.{s}.holm", f"{REP}#all_models/{fam}/holm_p/{m}", "p")
            src(f"{c}.{s}.read", f"{REP}#all_models/{fam}/readings/{m}", "word")
        for P in ("K", "V"):
            src(f"hfive{P}.{s}.est", pm(m, "H5", P, "estimate"), "u3")
            src(f"hfive{P}.{s}.ci", pm(m, "H5", P, "ci95"), "uci3")
            src(f"hfive{P}.{s}.holm", f"{REP}#all_models/F5 H5/holm_p/{m}~1{P}", "p")
            src(f"hfive{P}.{s}.read", f"{REP}#all_models/F5 H5/readings/{m}~1{P}", "word")
            der(f"hfive{P}.{s}.fac", "exp", [pm(m, "H5", P, "estimate")], "u1")
        cal = f"v4_runs/{m}/calibration.json"
        src(f"cal.{s}.b", f"{cal}#dose_slope_b_hat", "u2")
        src(f"cal.{s}.craw", f"{cal}#cv_raw_median", "u3")
        src(f"cal.{s}.c", f"{cal}#c_used", "u3")
        src(f"cal.{s}.margin", f"{cal}#margin_log_js", "u3")
        src(f"cal.{s}.rhoK", f"{cal}#rho_K", "u3")
        src(f"cal.{s}.rhoV", f"{cal}#rho_V", "u3")
        src(f"cal.{s}.rhoVK", f"{cal}#rho_V_over_rho_K", "u2")
        src(f"cal.{s}.ncal", f"{cal}#n_calibration_problems", "int")
        g = f"{MAN}#models/{m}/geometry/"
        for k in ("layers", "q_heads", "kv_heads", "head_dim"):
            src(f"geo.{s}.{k.replace('_', '')}", g + k, "int")
        src(f"geo.{s}.rope", g + "rope_theta", "intc")
        src(f"geo.{s}.vocab", g + "vocab", "intc")
        src(f"n.{s}", pm(m, "n_problems"), "int")
        src(f"x.{s}.params", f"{EXP}#size/{m}/params_16bit", "bn")
        src(f"x.{s}.log", f"{EXP}#size/{m}/log10_params", "u2")
        src(f"rep.{s}.hd", pm(m, "head_dim"), "int")


def rng(key, field, models, f, *, op_lo="min", op_hi="max"):
    ptrs = [field(m) for m in models]
    der(f"{key}.min", op_lo, ptrs, f)
    der(f"{key}.max", op_hi, ptrs, f)


def build_ranges():
    rng("rng.htwo", lambda m: pm(m, "contrasts", "H2", "estimate"), LADDER, "s3")
    rng("rng.hthree", lambda m: pm(m, "contrasts", "H3", "estimate"), LADDER, "s3")
    rng("rng.hone7", lambda m: pm(m, "contrasts", "H1", "estimate"), QWEN + LLAMA, "s3")
    rng("rng.natc.big", lambda m: pm(m, "contrasts", "NAT-C1", "estimate"), ["qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b", "llama3.2-3b", "llama3.1-8b"], "s3")
    for P in ("K", "V"):
        rng(f"rng.hfive{P}", lambda m, P=P: pm(m, "H5", P, "estimate"), LADDER, "u2")
        rng(f"rng.hfive{P}lo", lambda m, P=P: pm(m, "H5", P, "ci95", "0"), LADDER, "u2")
        rng(f"rng.fac{P}", lambda m, P=P: pm(m, "H5", P, "estimate"), LADDER, "u1")
        # the factor range is exp of the estimate range
        for side, op in (("min", "min"), ("max", "max")):
            k = f"rng.fac{P}.{side}"
            vals = [get(pm(m, "H5", P, "estimate")) for m in LADDER]
            del V[k], SRC[k]
            put(k, math.exp(OPS[op](vals)), "u1", op="exp_" + op, inputs=[pm(m, "H5", P, "estimate") for m in LADDER])
    rng("rng.rhoVK", lambda m: f"v4_runs/{m}/calibration.json#rho_V_over_rho_K", LADDER, "u2")
    rng("rng.margin", lambda m: f"v4_runs/{m}/calibration.json#margin_log_js", LADDER, "u3")
    rng("rng.b", lambda m: f"v4_runs/{m}/calibration.json#dose_slope_b_hat", LADDER, "u2")
    rng("rng.craw", lambda m: f"v4_runs/{m}/calibration.json#cv_raw_median", LADDER, "u3")
    rng("rng.hthreeratio", lambda m: pm(m, "contrasts", "H3", "estimate"), LADDER, "u1")
    for side, op in (("min", "min"), ("max", "max")):          # e^{-theta3}: factor by which NAT-V exceeds NAT-K
        k = f"rng.hthreeratio.{side}"
        vals = [-get(pm(m, "contrasts", "H3", "estimate")) for m in LADDER]
        del V[k], SRC[k]
        put(k, math.exp(OPS[op](vals)), "u1", op="expneg_" + ("max" if op == "min" else "min"), inputs=[pm(m, "contrasts", "H3", "estimate") for m in LADDER])


def build_families():
    for fam, k in (("F1 H1", "hone"), ("F2 H2", "htwo"), ("F3 H3", "hthree"), ("F5 H5", "hfive")):
        src(f"fam.{k}", f"{REP}#all_models/{fam}/result", "word")
        src(f"famwf.{k}", f"{REP}#without_flagged/{fam}/result", "word")
    h4 = f"{REP}#all_models/H4/"
    src("hfour.result", h4 + "result", "word")
    src("hfourwf.result", f"{REP}#without_flagged/H4/result", "word")
    src("hfour.T", h4 + "permutation/T", "u3")
    src("hfour.nlab", h4 + "permutation/n_labellings", "int")
    src("hfour.p", h4 + "permutation/p_one_sided", "u3")
    src("hfour.minp", h4 + "permutation/min_attainable_p", "u3")
    src("hfour.sep", h4 + "permutation/complete_separation", "bool")
    src("hfour.signs", h4 + "per_model_signs_ok", "bool")
    src("hfour.mi.est", h4 + "mistral_out_of_sample/estimate", "s3")
    src("hfour.mi.ci", h4 + "mistral_out_of_sample/ci95", "ci3")
    src("hfour.sp.q", h4 + "spearman_theta_vs_size/qwen", "u2")
    src("hfour.sp.l", h4 + "spearman_theta_vs_size/llama", "u2")
    der("nflag", "len", [f"{REP}#flagged_bf16"], "int")
    der("ndisagree", "len", [f"{REP}#abstract_level/disagreements"], "int")
    der("nmodels", "len", [f"{REP}#per_model"], "int")
    # how many Qwen models have NAT-C1 CI entirely above 0
    ptrs = [pm(m, "contrasts", "NAT-C1", "ci95", "0") for m in QWEN]
    put("hfour.nqwenpos", sum(get(p) > 0 for p in ptrs), "nword", op="count_gt0", inputs=ptrs)


def build_recomp_exp():
    src("recomp.n", f"{REC}#n_compared", "int")
    src("recomp.ndis", f"{REC}#n_disagree", "int")
    der("recomp.nunres", "len", [f"{REC}#unresolved_verdicts"], "int")
    X = EXP + "#"
    for fam, f in (("qwen", "q"), ("llama", "l")):
        for c, cn in (("H1", "hone"), ("NAT-C1", "natc")):
            src(f"xone.{f}.{cn}.slope", f"{X}X1/{fam}/{c}/slope_per_decade", "s2")
            src(f"xone.{f}.{cn}.ci", f"{X}X1/{fam}/{c}/ci95_boot", "ci2")
            src(f"xone.{f}.{cn}.r", f"{X}X1/{fam}/{c}/pearson_x_y", "s2")
            src(f"xfour.{f}.{cn}.slope", f"{X}X4/hd128_only_slopes/{fam}/{c}/slope_per_decade", "s2")
            src(f"xfour.{f}.{cn}.ci", f"{X}X4/hd128_only_slopes/{fam}/{c}/ci95_boot", "ci2")
        for c, cn in CON.items():
            src(f"xfour.{f}.step.{cn}", f"{X}X4/step_hd64_to_next/{fam}/{c}/delta", "s3")
            src(f"xfour.{f}.step.{cn}.ci", f"{X}X4/step_hd64_to_next/{fam}/{c}/ci95_boot", "ci3")
        src(f"xfour.{f}.dlog", f"{X}X4/step_hd64_to_next/{fam}/delta_log10_params", "u2")
    src("xthree.all.r", f"{X}X3/across_models_all8/pearson", "s2")
    src("xthree.all.rho", f"{X}X3/across_models_all8/spearman", "s2")
    src("xthree.seven.r", f"{X}X3/across_models_7_no_mistral/pearson", "s2")
    src("xthree.seven.rho", f"{X}X3/across_models_7_no_mistral/spearman", "s2")
    for m in LADDER:
        src(f"xthree.{S[m]}.r", f"{X}X3/within_model_problem_level/{m}/pearson", "s2")
        src(f"xthree.{S[m]}.ci", f"{X}X3/within_model_problem_level/{m}/ci95_boot", "ci2")
        for P in ("K", "V"):
            b = f"{X}X2/{m}/{P}/"
            src(f"xtwo.{S[m]}.{P}.mean", b + "mean", "u3")
            src(f"xtwo.{S[m]}.{P}.max", b + "max", "u3")
            src(f"xtwo.{S[m]}.{P}.argmax", b + "argmax_layer", "int")
            src(f"xtwo.{S[m]}.{P}.L", b + "L", "int")
            src(f"xtwo.{S[m]}.{P}.lzero", b + "layer0", "u3")
            src(f"xtwo.{S[m]}.{P}.rest", b + "mean_layers_1_on", "u3")
            src(f"xtwo.{S[m]}.{P}.sp", b + "spearman_vs_depth", "s2")
        src(f"xtwo.{S[m]}.fracVK", f"{X}X2/{m}/frac_layers_V_gt_K", "u2")
    src("x.seed", X + "seed", "int")
    src("x.B", X + "B", "intc")
    rng("rng.xthree.within.big", lambda m: f"{X}X3/within_model_problem_level/{m}/pearson",
        ["qwen2.5-7b", "llama3.2-3b", "llama3.1-8b", "mistral-7b-v0.3"], "s2")
    rng("rng.xthree.within.small", lambda m: f"{X}X3/within_model_problem_level/{m}/pearson",
        ["qwen2.5-0.5b", "qwen2.5-1.5b", "qwen2.5-3b", "llama3.2-1b"], "s2")
    # X2: share of models whose largest C^P is in layer 0 or 1 (both paths)
    ptrs = [f"{X}X2/{m}/{P}/argmax_layer" for m in LADDER for P in ("K", "V")]
    put("xtwo.nearly", sum(get(p) <= 1 for p in ptrs), "int", op="count_le1", inputs=ptrs)
    put("xtwo.nslots", len(ptrs), "int", op="count", inputs=ptrs)


def build_step0():
    for m in ("qwen2.5-0.5b", "llama3.2-1b-instruct", "llama3.2-1b"):
        s, b = S[m], f"{ST0}#models/{m}/"
        src(f"st.{s}.margin", b + "v3_margin", "u3")
        for c, cn in CON.items():
            src(f"st.{s}.{cn}.d", b + f"contrasts/{c}/delta/estimate", "s3")
            src(f"st.{s}.{cn}.ci", b + f"contrasts/{c}/delta/ci90", "ci3")
            src(f"st.{s}.{cn}.vthree", b + f"contrasts/{c}/theta_v3", "s3")
            src(f"st.{s}.{cn}.ok", b + f"contrasts/{c}/criterion_1", "bool")
        src(f"st.{s}.hist", b + "reported/history_token_identical_share", "pct0")
        src(f"st.{s}.jsr", b + "reported/nat_step_js_correlation_with_v3", "u3")
        for k in ("rho_K", "rho_V", "dose_slope_b_hat"):
            kk = k.replace("_", "").replace("doseslopebhat", "b")
            src(f"st.{s}.{kk}.vfour", b + f"criterion_3/v4/{k}", "u3")
            src(f"st.{s}.{kk}.vthree", b + f"criterion_3/v3/{k}", "u3")
        src(f"st.{s}.pass", b + "pass", "bool")
    src("st.pass", ST0 + "#pass", "bool")
    ptrs = [f"{ST0}#models/{m}/contrasts/{c}/delta/estimate" for m in ("qwen2.5-0.5b", "llama3.2-1b-instruct", "llama3.2-1b") for c in CON]
    der("st.dabsmax", "absmax", ptrs, "u3")
    rng("rng.st.hist", lambda m: f"{ST0}#models/{m}/reported/history_token_identical_share", ("qwen2.5-0.5b", "llama3.2-1b-instruct", "llama3.2-1b"), "pct0")
    rng("rng.st.jsr", lambda m: f"{ST0}#models/{m}/reported/nat_step_js_correlation_with_v3", ("qwen2.5-0.5b", "llama3.2-1b-instruct", "llama3.2-1b"), "u3")


def build_ops():
    led = load("v4_runs/gpu_ledger.jsonl")
    secs = [r["seconds"] for r in led]
    put("gpu.hours", sum(secs) / 3600, "u1", op="sum_div3600", inputs=["v4_runs/gpu_ledger.jsonl#*/seconds"])
    put("gpu.nblocks", len(led), "int", op="count", inputs=["v4_runs/gpu_ledger.jsonl#*"])
    cl = load("v4_runs/cost_ledger.jsonl")
    last = [r for r in cl if r.get("kind") == "session_close"][-1]
    put("cost.inpod", last["used_usd_estimate"], "usd", op="last_session_close", inputs=["v4_runs/cost_ledger.jsonl#session_close/-1/used_usd_estimate"])
    B = "v4/closeout/billing_snapshot_2026-11-09.json#"
    src("cost.total", B + "totals/totalAmount", "usd")
    src("cost.gpu", B + "totals/podGpuAmount", "usd")
    src("cost.disk", B + "totals/podDiskAmount", "usd")
    src("cost.storage", B + "totals/storageHighPerformanceAmount", "usd")
    src("cost.ceiling", B + "prepaid_ceiling_usd", "int")
    src("cost.rate", "v4_runs/cost_config.json#rate_usd_per_h", "usd")
    src("cost.volgb", "v4_runs/cost_config.json#volume_gb", "int")
    src("gm.tests", "v4_runs/gm_D1.json#tests_passed", "word")
    src("gm.mutants", "v4_runs/gm_D1.json#mutants_caught", "word")
    src("gm.driver", "v4_runs/gm_D1.json#gpu_environment/driver_version", "word")
    src("gm.hostcuda", "v4_runs/gm_D1.json#gpu_environment/cuda_version", "word")
    src("gm.torchcuda", "v4_runs/gm_D1.json#torch_cuda", "word")
    src("gm.ecc", "v4_runs/gm_D1.json#gpu_environment/ecc_mode", "word")
    src("gm.cublas", "v4_runs/gm_D1.json#determinism/cublas_workspace_config", "word")
    src("design.ncal", "v4_runs/design.json#n_cal", "int")
    src("design.ntest", "v4_runs/design.json#n_test", "int")
    src("design.used", "v4_runs/design.json#used_usd_at_design", "usd")
    src("seed.locked", REP + "#seed", "int")
    lines = [l for l in (RV / "v4/closeout/volume_sha256sums.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    put("closeout.nfiles", len(lines), "intc", op="count_lines", inputs=["v4/closeout/volume_sha256sums.txt"])
    locks = sorted(p.name for p in (RV / "v4/lock_history").glob("V4_PROTOCOL_LOCK_*.txt"))
    put("lock.nprev", len(locks), "int", op="count_files", inputs=["v4/lock_history/V4_PROTOCOL_LOCK_*.txt"])
    # common-mode factor over both paths
    ptrs = [pm(m, "H5", P, "estimate") for m in LADDER for P in ("K", "V")]
    put("rng.facKV.min", math.exp(min(get(p) for p in ptrs)), "u1", op="exp_min", inputs=ptrs)
    put("rng.facKV.max", math.exp(max(get(p) for p in ptrs)), "u1", op="exp_max", inputs=ptrs)
    der("x.ratio.laqa", "div", [f"{EXP}#size/llama3.2-1b/params_16bit", f"{EXP}#size/qwen2.5-0.5b/params_16bit"], "u1")
    ptrs = [pm(m, "H5", P, "ci95", "0") for m in LADDER for P in ("K", "V")]
    put("rng.hfivelo.min", min(get(p) for p in ptrs), "u3", op="min", inputs=ptrs)


def write_macros():
    TAB.mkdir(exist_ok=True)
    L = ["% generated by make_v4.py; do not edit", "\\makeatletter",
         "\\providecommand{\\vf}[1]{\\ifcsname vf@#1\\endcsname\\csname vf@#1\\endcsname\\else\\textbf{\\textcolor{red}{??#1}}\\fi}"]
    for k, v in V.items():
        L.append(f"\\expandafter\\def\\csname vf@{k}\\endcsname{{{v}}}")
    L.append("\\makeatother")
    (TAB / "numbers_v4.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    (TAB / "numbers_v4_sources.json").write_text(json.dumps({"report_sha256": REPORT_SHA, "macros": SRC}, indent=1, default=float), encoding="utf-8")


# ------------------------------------------------------------------ tables
def vf(k):
    return f"\\vf{{{k}}}"


def tab_models():
    rows = []
    for m in LADDER:
        s = S[m]
        rows.append(f"{NAME[m]} & {vf(f'x.{s}.params')} & {vf(f'geo.{s}.layers')} & {vf(f'geo.{s}.qheads')}/{vf(f'geo.{s}.kvheads')} & "
                    f"{vf(f'geo.{s}.headdim')} & {vf(f'cal.{s}.b')} & {vf(f'cal.{s}.margin')} & {vf(f'cal.{s}.rhoK')} & {vf(f'cal.{s}.rhoV')} & {vf(f'cal.{s}.rhoVK')} \\\\")
    t = ["\\begin{table}[tbp]", "\\centering\\small", "\\caption{Models and frozen calibrations. Parameters are the 16-bit count of the stored "
         "checkpoint (billions). $\\hat b$: dose slope of C1; margin $=|\\hat b|\\ln(1+c)$; $\\rho_K$, $\\rho_V$: median natural relative "
         "error of keys and values. Calibration used " + vf("design.ncal") + " training problems per model.}", "\\label{tab:models}",
         "\\begin{tabular}{@{}lrrrrrrrrr@{}}", "\\toprule",
         "Model & Params & Layers & Q/KV & Head dim. & $\\hat b$ & Margin & $\\rho_K$ & $\\rho_V$ & $\\rho_V/\\rho_K$ \\\\", "\\midrule"]
    t += rows[:4] + ["\\midrule"] + rows[4:7] + ["\\midrule"] + rows[7:] + ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TAB / "models.tex").write_text("\n".join(t) + "\n", encoding="utf-8")


def tab_main():
    t = ["\\begin{table}[tbp]", "\\centering\\footnotesize",
         "\\caption{Locked tests H1--H3 per model: estimate, 95\\% problem-clustered bootstrap interval and Holm-adjusted $p$ (Holm over the "
         "eight models of each family). Contrasts are differences of log mean JS divergence; negative means the first arm is milder. "
         "Reading: S, supported; I, inconclusive. hd: head dimension. $n=$\\,\\vf{n.qa} problems per model.}", "\\label{tab:main}",
         "\\setlength{\\tabcolsep}{2.5pt}",
         "\\begin{tabular}{@{}lr lll lll lll@{}}", "\\toprule",
         " & & \\multicolumn{3}{c}{H1: NAT-K$-$C1-K} & \\multicolumn{3}{c}{H2: NAT-V$-$C1-V} & \\multicolumn{3}{c}{H3: NAT-K$-$NAT-V} \\\\",
         "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\\cmidrule(lr){9-11}",
         "Model & hd & Est. & 95\\% CI & Holm & Est. & 95\\% CI & Holm & Est. & 95\\% CI & Holm \\\\", "\\midrule"]
    for i, m in enumerate(LADDER):
        s = S[m]
        if i in (4, 7):
            t.append("\\midrule")
        cells = []
        for cn in ("hone", "htwo", "hthree"):
            r = "S" if get(f"{REP}#all_models/F{ {'hone': 1, 'htwo': 2, 'hthree': 3}[cn] } H{ {'hone': 1, 'htwo': 2, 'hthree': 3}[cn] }/readings/{m}") == "supported" else "I"
            cells.append(f"{vf(f'{cn}.{s}.est')} & {vf(f'{cn}.{s}.ci')} & {vf(f'{cn}.{s}.holm')}\\,{r}")
        t.append(f"{NAME[m]} & {vf(f'geo.{s}.headdim')} & " + " & ".join(cells) + " \\\\")
    t += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TAB / "main.tex").write_text("\n".join(t) + "\n", encoding="utf-8")


def tab_h5_h4():
    t = ["\\begin{table}[tbp]", "\\centering\\small",
         "\\caption{H5 (common mode) and the NAT$-$C1 contrast entering H4. $L^P$ is the null-corrected log ratio of the cross-position "
         "common-mode share to its isotropic value; support requires a 95\\% lower bound above $\\ln 4$ and Holm $p<0.05$ (Holm over the "
         "16 tests). NAT$-$C1 is shown with its 95\\% interval and verdict; Mistral-7B-v0.3 is out of sample in H4. hd: head dimension.}", "\\label{tab:h5h4}",
         "\\setlength{\\tabcolsep}{4pt}",
         "\\begin{tabular}{@{}lr ll ll ll@{}}", "\\toprule",
         " & & \\multicolumn{2}{c}{H5, $L^K$} & \\multicolumn{2}{c}{H5, $L^V$} & \\multicolumn{2}{c}{NAT$-$C1 (H4)} \\\\",
         "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}",
         "Model & hd & Est. & 95\\% CI & Est. & 95\\% CI & Est. & 95\\% CI \\\\", "\\midrule"]
    for i, m in enumerate(LADDER):
        s = S[m]
        if i in (4, 7):
            t.append("\\midrule")
        t.append(f"{NAME[m]} & {vf(f'geo.{s}.headdim')} & {vf(f'hfiveK.{s}.est')} & {vf(f'hfiveK.{s}.ci')} & {vf(f'hfiveV.{s}.est')} & "
                 f"{vf(f'hfiveV.{s}.ci')} & {vf(f'natc.{s}.est')} & {vf(f'natc.{s}.ci')} \\\\")
    t += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TAB / "h5h4.tex").write_text("\n".join(t) + "\n", encoding="utf-8")


def tab_step0():
    t = ["\\begin{table}[tbp]", "\\centering\\small",
         "\\caption{Step 0, cross-hardware check on the A40 against the earlier laptop-GPU records: $\\delta=\\theta_{\\mathrm{A40}}-"
         "\\theta_{\\mathrm{earlier}}$ per contrast, paired by problem, with its 90\\% interval. Criterion 1 requires the interval inside "
         "$\\pm$ the earlier margin and $|\\hat\\delta|\\le0.10$.}", "\\label{tab:step0}",
         "\\setlength{\\tabcolsep}{4pt}",
         "\\begin{tabular}{@{}llrll@{}}", "\\toprule", "Model (margin) & Contrast & Earlier $\\theta$ & $\\hat\\delta$ & 90\\% CI of $\\delta$ \\\\", "\\midrule"]
    lab = {"natc": "NAT$-$C1", "hone": "NAT-K$-$C1-K", "htwo": "NAT-V$-$C1-V", "hthree": "NAT-K$-$NAT-V"}
    for j, m in enumerate(("qwen2.5-0.5b", "llama3.2-1b-instruct", "llama3.2-1b")):
        s = S[m]
        if j:
            t.append("\\midrule")
        for i, cn in enumerate(("hone", "htwo", "hthree", "natc")):
            head = f"{NAME[m]} ({vf(f'st.{s}.margin')})" if i == 0 else ""
            t.append(f"{head} & {lab[cn]} & {vf(f'st.{s}.{cn}.vthree')} & {vf(f'st.{s}.{cn}.d')} & {vf(f'st.{s}.{cn}.ci')} \\\\")
    t += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TAB / "step0.tex").write_text("\n".join(t) + "\n", encoding="utf-8")


def tab_x():
    t = ["\\begin{table}[tbp]", "\\centering\\small",
         "\\caption{Exploratory X1 and X4: OLS slope of the per-model estimate on $\\log_{10}$ parameters within each family, per decade, "
         "with a 95\\% bootstrap interval over problems (" + vf("x.B") + " resamples, models fixed). Llama's head-dimension-128 ``slope'' "
         "joins two points. Not blind; no test.}", "\\label{tab:x1}",
         "\\begin{tabular}{@{}llll@{}}", "\\toprule", "Family & Contrast & All sizes & Head dim.\\ 128 only \\\\", "\\midrule"]
    for f, fam in (("q", "Qwen"), ("l", "Llama")):
        for cn, lab in (("hone", "H1"), ("natc", "NAT$-$C1")):
            t.append(f"{fam} & {lab} & {vf(f'xone.{f}.{cn}.slope')} {vf(f'xone.{f}.{cn}.ci')} & {vf(f'xfour.{f}.{cn}.slope')} {vf(f'xfour.{f}.{cn}.ci')} \\\\")
    t += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TAB / "x1.tex").write_text("\n".join(t) + "\n", encoding="utf-8")
    # X2 (supplementary-style appendix table)
    t = ["\\begin{table}[tbp]", "\\centering\\small",
         "\\caption{Exploratory X2: per-layer common-mode share $C^P$ from the locked report, pooled over problems (no interval, not "
         "head-dimension-adjusted). Layer of the maximum (0-based), layer 0 against the mean of the other layers, and the fraction of layers "
         "with $C^V>C^K$.}", "\\label{tab:x2}", "\\setlength{\\tabcolsep}{3.5pt}",
         "\\begin{tabular}{@{}lr rrrr rrrr r@{}}", "\\toprule",
         " & & \\multicolumn{4}{c}{K} & \\multicolumn{4}{c}{V} & \\\\", "\\cmidrule(lr){3-6}\\cmidrule(lr){7-10}",
         "Model & $L$ & mean & argmax & layer 0 & rest & mean & argmax & layer 0 & rest & $C^V>C^K$ \\\\", "\\midrule"]
    for m in LADDER:
        s = S[m]
        c = [f"{vf(f'xtwo.{s}.{P}.mean')} & {vf(f'xtwo.{s}.{P}.argmax')} & {vf(f'xtwo.{s}.{P}.lzero')} & {vf(f'xtwo.{s}.{P}.rest')}" for P in ("K", "V")]
        t.append(f"{NAME[m]} & {vf(f'xtwo.{s}.K.L')} & {c[0]} & {c[1]} & {vf(f'xtwo.{s}.fracVK')} \\\\")
    t += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TAB / "x2.tex").write_text("\n".join(t) + "\n", encoding="utf-8")


SENS = [("excluding exposed", "exp"), ("steps 2-8", "early"), ("steps 9-64", "late"), ("per-problem median", "med")]


def build_sens():
    sign_ok, ptrs_all = 0, []
    for m in LADDER:
        for c, cn in CON.items():
            for k, kk in SENS:
                src(f"sens.{S[m]}.{cn}.{kk}", pm(m, "sensitivity", c, k), "s3")
            src(f"sens.{S[m]}.{cn}.rdev", pm(m, "sensitivity", c, "rounding truncation (>5 %)", "estimate"), "s3")
            src(f"sens.{S[m]}.{cn}.rdevdrop", pm(m, "sensitivity", c, "rounding truncation (>5 %)", "dropped"), "int")
            if c in ("H1", "H2", "H3"):
                prim = get(pm(m, "contrasts", c, "estimate"))
                for k, kk in SENS:
                    ptrs_all.append((pm(m, "contrasts", c, "estimate"), pm(m, "sensitivity", c, k)))
                ptrs_all.append((pm(m, "contrasts", c, "estimate"), pm(m, "sensitivity", c, "rounding truncation (>5 %)", "estimate")))
    same = [math.copysign(1, get(a)) == math.copysign(1, get(b)) for a, b in ptrs_all]
    put("sens.nsame", sum(same), "int", op="count_same_sign", inputs=[f"{a}|{b}" for a, b in ptrs_all])
    put("sens.ntotal", len(same), "int", op="count", inputs=[f"{a}|{b}" for a, b in ptrs_all])


def tab_sens():
    lab = {"hone": "H1", "htwo": "H2", "hthree": "H3", "natc": "NAT$-$C1"}
    t = [r"\begin{table}[tbp]", r"\centering\footnotesize",
         r"\caption{Locked sensitivity analyses (none replaces a primary estimate): excluding the exposed test problems, scoring steps 2--8 "
         r"or 9--64 only, per-problem medians instead of means, and truncating each problem before the first step written with a slot "
         r"deviating by more than 5\% from its target norm (problems dropped in parentheses).}", r"\label{tab:sens}",
         r"\setlength{\tabcolsep}{3.5pt}", r"\begin{tabular}{@{}ll rrrrrr@{}}", r"\toprule",
         r"Model & Contrast & Primary & Excl.\ exposed & Steps 2--8 & Steps 9--64 & Median & Truncated \\", r"\midrule"]
    for i, m in enumerate(LADDER):
        s = S[m]
        if i:
            t.append(r"\midrule")
        for j, cn in enumerate(CON.values()):
            head = NAME[m] if j == 0 else ""
            t.append(f"{head} & {lab[cn]} & {vf(f'{cn}.{s}.est')} & {vf(f'sens.{s}.{cn}.exp')} & {vf(f'sens.{s}.{cn}.early')} & "
                     f"{vf(f'sens.{s}.{cn}.late')} & {vf(f'sens.{s}.{cn}.med')} & {vf(f'sens.{s}.{cn}.rdev')} ({vf(f'sens.{s}.{cn}.rdevdrop')}) \\\\")
    t += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB / "sens.tex").write_text("\n".join(t) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ figures
def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "font.family": "serif"})
    FIG.mkdir(exist_ok=True)
    rep = load(REP)
    col = {m: ("#1f4e79" if m in QWEN else "#b5651d" if m in LLAMA else "#3d7a3d") for m in LADDER}
    mk = {m: ("o" if rep["per_model"][m]["head_dim"] == 128 else "s") for m in LADDER}
    # Figure 1: forest plot H1, H2, H3
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.8), sharey=True)
    ys = list(range(len(LADDER)))[::-1]
    for ax, (c, lab) in zip(axs, (("H1", "H1: NAT-K − C1-K"), ("H2", "H2: NAT-V − C1-V"), ("H3", "H3: NAT-K − NAT-V"))):
        for y, m in zip(ys, LADDER):
            s = rep["per_model"][m]["contrasts"][c]
            ax.plot(s["ci95"], [y, y], color=col[m], lw=1.2)
            ax.plot(s["estimate"], y, mk[m], color=col[m], ms=4, mfc="white" if mk[m] == "s" else col[m])
        ax.axvline(0, color="0.5", lw=0.6, ls="--")
        ax.set_title(lab, fontsize=8)
        ax.set_xlabel("log mean-JS contrast")
    axs[0].set_yticks(ys)
    axs[0].set_yticklabels([NAME[m] for m in LADDER])
    fig.tight_layout()
    fig.savefig(FIG / "fig_forest.pdf")
    plt.close(fig)
    # Figure 2: NAT-C1 and H1 against log10 params
    ex = load(EXP)
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for ax, (c, lab) in zip(axs, (("NAT-C1", "NAT − C1 (H4 contrast)"), ("H1", "H1: NAT-K − C1-K"))):
        for fam, ms in ((QWEN, "Qwen2.5"), (LLAMA, "Llama-3.x"), (MISTRAL, "Mistral")):
            xs = [ex["size"][m]["log10_params"] for m in fam]
            yv = [rep["per_model"][m]["contrasts"][c]["estimate"] for m in fam]
            if len(fam) > 1:
                ax.plot(xs, yv, "-", color=col[fam[0]], lw=0.8)
            for m, x, y in zip(fam, xs, yv):
                lo, hi = rep["per_model"][m]["contrasts"][c]["ci95"]
                ax.plot([x, x], [lo, hi], color=col[m], lw=1.0)
                ax.plot(x, y, mk[m], color=col[m], ms=4.5, mfc="white" if mk[m] == "s" else col[m], label=ms if m == fam[0] else None)
        ax.axhline(0, color="0.5", lw=0.6, ls="--")
        ax.set_xlabel("log$_{10}$ parameters")
        ax.set_title(lab, fontsize=8)
    axs[0].legend(fontsize=7, frameon=False, loc="upper left")
    axs[1].text(0.02, 0.04, "open squares: head dimension 64", transform=axs[1].transAxes, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig_size.pdf")
    plt.close(fig)


def main():
    h = hashlib.sha256((RV / REP.split("#")[0]).read_bytes()).hexdigest()
    assert h == REPORT_SHA, f"v4_report.json changed: {h}"
    build_models()
    build_ranges()
    build_families()
    build_recomp_exp()
    build_step0()
    build_ops()
    build_sens()
    write_macros()
    tab_models(); tab_main(); tab_h5_h4(); tab_step0(); tab_x(); tab_sens()
    figures()
    print(f"{len(V)} macros; tables in {TAB.name}/, figures in {FIG.name}/")


if __name__ == "__main__":
    main()
