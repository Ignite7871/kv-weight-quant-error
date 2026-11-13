"""Generates every v5 number macro and table of main_v4.tex (the second locked study, Section 6) from saved results. CPU only; reads,
never writes, research outputs. Shares no code with audit/audit_v5_numbers.py.

Sources (read-only; paths relative to research_v2/):
  v5_runs/v5_report.json            locked analysis output, rerun with U1 under V5-D3 (sha256 checked below)
  v5_runs/v5_recomputation.json     independent recomputation (230 quantities)
  v5_runs/anchor.json               U0 against the locked v4 records
  v5_runs/design.json               design point (reduction, per-unit projections)
  v5_runs/U<k>/calibration.json     frozen calibrations
  v5_runs/U1/drift_stop.json        the U1 drift stop behind V5-D2
  v5_runs/U1/d3_calibration_check.json, d3_reference_check.jsonl   the two V5-D3 checks
  v5_runs/gm.json, gm_D3.json       gate checks (GM; GM D2's record is declared lost in V5-D3)
  v5_runs/gpu_ledger.jsonl, cost_ledger.jsonl (last billing line)
  v5/V5_MODEL_MANIFEST.json         geometry, parameters, QK-norm tensors
  v4_runs/gpu_ledger.jsonl, v4/closeout/billing_snapshot_2026-11-09.json   (v4 totals for the combined compute summary)
Outputs: paper_journal/tables_v5/numbers_v5.tex (\\vsv{key}), numbers_v5_sources.json, v5main.tex, v5ax.tex, v5h5.tex.

  venv/Scripts/python.exe research_v2/paper_journal/make_v5.py
"""
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
RV = HERE.parent
TAB = HERE / "tables_v5"
REPORT_SHA = "2f00f5113aec05f67990fac55e9150340780f31273a18080d2e276db0f493706"
REP, REC, ANC, DES = "v5_runs/v5_report.json", "v5_runs/v5_recomputation.json", "v5_runs/anchor.json", "v5_runs/design.json"
MAN = "v5/V5_MODEL_MANIFEST.json"
UNITS = ["U0", "U1", "U2", "U3", "U4", "U5", "U6"]
CON = {"H1": "hone", "H2": "htwo", "H3": "hthree", "NAT-C1": "natc"}
LABEL = {"U0": ("Qwen2.5-0.5B", "NF4", "GSM8K"), "U1": ("Qwen3-0.6B", "NF4", "GSM8K"), "U2": ("SmolLM2-1.7B", "NF4", "GSM8K"),
         "U3": ("Qwen2.5-0.5B", "NF4", "WT-2"), "U4": ("Qwen2.5-0.5B", "HQQ", "GSM8K"),
         "U5": ("Mistral-7B", "NF4", "WT-2"), "U6": ("Mistral-7B", "HQQ", "GSM8K")}
FAMS = ["A-H1", "A-H2", "A-H3", "A-H5", "B-H2", "B-H3", "B-H5", "D-H2", "D-H5"]
V, SRC, _cache = {}, {}, {}


def load(rel):
    if rel not in _cache:
        t = (RV / rel).read_text(encoding="utf-8")
        _cache[rel] = [json.loads(l) for l in t.splitlines() if l.strip()] if rel.endswith(".jsonl") else json.loads(t)
    return _cache[rel]


def get(ptr):
    f, _, path = ptr.partition("#")
    o = load(f)
    for k in [k.replace("~1", "/") for k in path.split("/") if k != ""]:
        o = o[int(k)] if isinstance(o, list) else o[k]
    return o


def fmt(x, f):
    if f == "s3": return f"${x:+.3f}$".replace("+-", "-")
    if f == "s2": return f"${x:+.2f}$"
    if f == "u3": return f"${x:.3f}$"
    if f == "u2": return f"${x:.2f}$"
    if f == "u1": return f"${x:.1f}$"
    if f == "int": return f"${int(round(x))}$"
    if f == "intc": return f"${int(round(x)):,}$".replace(",", "{,}")
    if f == "ci3": return f"$[{x[0]:+.3f}, {x[1]:+.3f}]$"
    if f == "uci3": return f"$[{x[0]:.3f}, {x[1]:.3f}]$"
    if f == "p": return "$<0.001$" if x < 0.001 else f"${x:.3f}$"
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


def pu(u, *k):
    return f"{REP}#per_unit/{u}/" + "/".join(k)


def ledger_hours(rel, since=None):
    return sum(r["seconds"] for r in load(rel) if since is None or r["start_utc"] >= since) / 3600


# ------------------------------------------------------------------ macros
def build_units():
    for u in UNITS:
        s = u.lower()
        for c, cn in CON.items():
            src(f"{cn}.{s}.est", pu(u, "contrasts", c, "estimate"), "s3")
            src(f"{cn}.{s}.ci", pu(u, "contrasts", c, "ci95"), "ci3")
            src(f"{cn}.{s}.p", pu(u, "contrasts", c, "p"), "p")
            src(f"{cn}.{s}.verdict", pu(u, "contrasts", c, "verdict"), "word")
        for P in ("K", "V"):
            src(f"hfive{P}.{s}.est", pu(u, "H5", P, "estimate"), "u3")
            src(f"hfive{P}.{s}.ci", pu(u, "H5", P, "ci95"), "uci3")
        src(f"n.{s}", pu(u, "n_problems"), "int")
        src(f"hd.{s}", pu(u, "head_dim"), "int")
        cal = f"v5_runs/{u}/calibration.json"
        src(f"cal.{s}.b", f"{cal}#dose_slope_b_hat", "u2")
        src(f"cal.{s}.margin", f"{cal}#margin_log_js", "u3")
        src(f"cal.{s}.rhoK", f"{cal}#rho_K", "u3")
        src(f"cal.{s}.rhoV", f"{cal}#rho_V", "u3")
        src(f"cal.{s}.rhoVK", f"{cal}#rho_V_over_rho_K", "u2")
    for m, s in (("qwen3-0.6b-base", "qt"), ("smollm2-1.7b", "sm")):
        g = f"{MAN}#models/{m}/"
        for k in ("layers", "q_heads", "kv_heads", "head_dim"):
            src(f"geo.{s}.{k.replace('_', '')}", g + "geometry/" + k, "int")
        src(f"geo.{s}.vocab", g + "geometry/vocab", "intc")
        src(f"geo.{s}.params", g + "params/total", "bn")
        src(f"geo.{s}.qknorm", g + "qk_norm_tensors/k_norm", "int")


def build_families():
    for fam in FAMS:
        k = fam.lower().replace("-", "")
        src(f"fam.{k}", f"{REP}#all_units/{fam}/result", "word")
        src(f"famwf.{k}", f"{REP}#without_flagged/{fam}/result", "word")
        for key, p in get(f"{REP}#all_units/{fam}/holm_p").items():
            kk = key.lower().replace("/", "")
            src(f"holm.{k}.{kk}", f"{REP}#all_units/{fam}/holm_p/{key.replace('/', '~1')}", "p")
            src(f"read.{k}.{kk}", f"{REP}#all_units/{fam}/readings/{key.replace('/', '~1')}", "word")
    der_len("nflag", f"{REP}#flagged_bf16")
    der_len("ndisagree", f"{REP}#abstract_level/disagreements")
    for c, cn in (("NAT-C1", "natc"), ("H1", "hone")):
        for u in ("U1", "U2"):
            b = f"{REP}#all_units/A-X/{c}/units/{u}/"
            s = u.lower()
            src(f"ax.{cn}.{s}.est", b + "estimate", "s3")
            src(f"ax.{cn}.{s}.ci", b + "ci95", "ci3")
            src(f"ax.{cn}.{s}.b", b + "boundary", "s3")
            src(f"ax.{cn}.{s}.size", b + "size_pred", "s3")
            src(f"ax.{cn}.{s}.hd", b + "hd_pred", "s3")
            src(f"ax.{cn}.{s}.holm", b + "holm_p", "p")
            src(f"ax.{cn}.{s}.class", b + "classification", "word")
    src("ax.joint", f"{REP}#all_units/A-X/joint_reading_NAT-C1/reading", "word")
    src("ax.jointwf", f"{REP}#without_flagged/A-X/joint_reading_NAT-C1/reading", "word")


def der_len(key, ptr):
    put(key, len(get(ptr)), "int", op="len", inputs=[ptr])


def build_anchor_recomp_design():
    src("anc.pass", ANC + "#pass", "bool")
    src("anc.req", ANC + "#reported_bitwise/records_equal", "int")
    src("anc.rtot", ANC + "#reported_bitwise/records_total", "int")
    src("anc.share", ANC + "#reported_bitwise/share", "pct0")
    src("anc.slots", ANC + "#reported_bitwise/nat_slots_equal", "int")
    src("anc.hist", ANC + "#reported_bitwise/histories_equal", "int")
    ptrs = [f"{ANC}#contrasts/{c}/delta/estimate" for c in CON]
    put("anc.dabsmax", max(abs(get(p)) for p in ptrs), "u3", op="absmax", inputs=ptrs)
    src("recomp.n", REC + "#n_compared", "int")
    src("recomp.ndis", REC + "#n_disagree", "int")
    der_len("recomp.nunres", REC + "#unresolved_verdicts")
    src("design.reduction", DES + "#reduction", "word")
    src("design.factor", DES + "#factor", "u2")
    src("design.ncal", DES + "#n_cal", "int")
    src("seed.locked", REP + "#seed", "int")
    put("dtwo.factor", get("v5_runs/U1/drift_stop.json#mean_problem_s") / get("v5_runs/U1/drift_stop.json#expected_problem_s"), "u2",
        op="div", inputs=["v5_runs/U1/drift_stop.json#mean_problem_s", "v5_runs/U1/drift_stop.json#expected_problem_s"])
    src("dthree.calequal", "v5_runs/U1/d3_calibration_check.json#equal", "bool")
    refs = load("v5_runs/U1/d3_reference_check.jsonl")
    put("dthree.nref", len(refs), "int", op="count_rows", inputs=["v5_runs/U1/d3_reference_check.jsonl"])
    put("dthree.nrefeq", sum(bool(r["bitwise_equal"]) for r in refs), "int", op="count_true_field",
        inputs=["v5_runs/U1/d3_reference_check.jsonl#*/bitwise_equal"])
    src("gm.tests", "v5_runs/gm.json#tests_passed", "word")
    src("gm.mutants", "v5_runs/gm.json#mutants_caught", "word")
    src("gmdthree.tests", "v5_runs/gm_D3.json#tests_passed", "word")
    src("gmdthree.mutants", "v5_runs/gm_D3.json#mutants_caught", "word")
    src("gm.driver", "v5_runs/gm_D3.json#gpu_environment/driver_version", "word")


def build_ops():
    put("gpu.hours", ledger_hours("v5_runs/gpu_ledger.jsonl"), "u1", op="ledger_hours", inputs=["v5_runs/gpu_ledger.jsonl"])
    put("gpu.dthree", ledger_hours("v5_runs/gpu_ledger.jsonl", "2026-11-13"), "u1", op="ledger_hours_since",
        inputs=["v5_runs/gpu_ledger.jsonl", "2026-11-13"])
    bills = [r for r in load("v5_runs/cost_ledger.jsonl") if r.get("kind") == "billing"]
    put("cost.total", bills[-1]["usd"], "usd", op="last_billing", inputs=["v5_runs/cost_ledger.jsonl"])
    v4c = "v4/closeout/billing_snapshot_2026-11-09.json#totals/totalAmount"
    put("both.cost", bills[-1]["usd"] + get(v4c), "usd", op="last_billing_plus", inputs=["v5_runs/cost_ledger.jsonl", v4c])
    put("both.hours", ledger_hours("v5_runs/gpu_ledger.jsonl") + ledger_hours("v4_runs/gpu_ledger.jsonl"), "u1", op="ledger_hours_sum",
        inputs=["v5_runs/gpu_ledger.jsonl", "v4_runs/gpu_ledger.jsonl"])
    src("cost.rate", "v5_runs/cost_config.json#rate_usd_per_h", "usd")


def build_ranges():
    for key, units, c in (("rng.btwo", ("U4", "U6"), "H2"), ("rng.bthree", ("U4", "U6"), "H3"), ("rng.dtwo", ("U3", "U5"), "H2")):
        ptrs = [pu(u, "contrasts", c, "estimate") for u in units]
        put(key + ".min", min(get(p) for p in ptrs), "s3", op="min", inputs=ptrs)
        put(key + ".max", max(get(p) for p in ptrs), "s3", op="max", inputs=ptrs)
    # lower bounds printed as "at or above x": rounded DOWN to 3 decimals, so the printed bound never exceeds the true one
    ptrs = [pu(u, "H5", P, "ci95", "0") for u in ("U3", "U4", "U5", "U6") for P in "KV"]
    put("rng.bdhfivelo.min", math.floor(1000 * min(get(p) for p in ptrs)) / 1000, "u3", op="floor3_min", inputs=ptrs)
    ptrs = [pu(u, "H5", P, "ci95", "0") for u in ("U1", "U2") for P in "KV"]
    put("rng.ahfivelo.min", math.floor(1000 * min(get(p) for p in ptrs)) / 1000, "u3", op="floor3_min", inputs=ptrs)


def write_macros():
    TAB.mkdir(exist_ok=True)
    L = ["% generated by make_v5.py; do not edit", "\\makeatletter",
         "\\newcommand{\\vsv}[1]{\\ifcsname vsv@#1\\endcsname\\csname vsv@#1\\endcsname\\else\\textbf{\\textcolor{red}{??#1}}\\fi}"]
    L += [f"\\expandafter\\def\\csname vsv@{k}\\endcsname{{{v}}}" for k, v in V.items()]
    L.append("\\makeatother")
    (TAB / "numbers_v5.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    (TAB / "numbers_v5_sources.json").write_text(json.dumps({"report_sha256": REPORT_SHA, "macros": SRC}, indent=1, default=float),
                                                encoding="utf-8")


# ------------------------------------------------------------------ tables
def vv(k):
    return f"\\vsv{{{k}}}"


def tab_main():
    t = [r"\begin{table}[tbp]", r"\centering\scriptsize",
         r"\caption{Second locked study (v5): the four contrasts per unit, estimate and 95\% problem-clustered bootstrap interval "
         r"($n=$\," + vv("n.u1") + r" problems per unit). Locked tests: A, H1--H3 in U1 and U2; B, H2 and H3 under HQQ; D, H2 on "
         r"WikiText-2. Every other cell is descriptive (no prediction). U0 is the anchor, a rerun of the v4 Qwen2.5-0.5B unit. "
         r"WT-2: WikiText-2; Qwen3-0.6B: Qwen3-0.6B-Base; Mistral-7B: Mistral-7B-v0.3. hd: head dimension. Holm-adjusted $p$ of every locked test: Table~\ref{tab:v5h5}.}", r"\label{tab:v5main}",
         r"\setlength{\tabcolsep}{2pt}", r"\begin{tabular}{@{}llllr ll ll ll ll@{}}", r"\toprule",
         r" & & & & & \multicolumn{2}{c}{H1: NAT-K$-$C1-K} & \multicolumn{2}{c}{H2: NAT-V$-$C1-V} & \multicolumn{2}{c}{H3: NAT-K$-$NAT-V}"
         r" & \multicolumn{2}{c}{NAT$-$C1} \\",
         r"\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}\cmidrule(lr){12-13}",
         r"Unit & Model & Quant. & Task & hd & Est. & 95\% CI & Est. & 95\% CI & Est. & 95\% CI & Est. & 95\% CI \\", r"\midrule"]
    for i, u in enumerate(["U0", "U1", "U2", "U4", "U6", "U3", "U5"]):
        s = u.lower()
        if i in (1, 3, 5):
            t.append(r"\midrule")
        m, q, task = LABEL[u]
        cells = " & ".join(f"{vv(f'{cn}.{s}.est')} & {vv(f'{cn}.{s}.ci')}" for cn in CON.values())
        t.append(f"{u} & {m} & {q} & {task} & {vv(f'hd.{s}')} & {cells} \\\\")
    t += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB / "v5main.tex").write_text("\n".join(t) + "\n", encoding="utf-8")


def tab_ax():
    t = [r"\begin{table}[tbp]", r"\centering\footnotesize",
         r"\caption{A-X, the crossed reading (locked). Predictions are locked v4 estimates of named source models (Qwen3: Qwen2.5-0.5B "
         r"for size, Qwen2.5-1.5B for head dimension; SmolLM2: the mean of the head-dimension-128 Qwen2.5-1.5B, Qwen2.5-3B and Llama-3.2-3B "
         r"for size, the mean of the two head-dimension-64 models for head dimension); $b$ is their midpoint. A classification needs the "
         r"95\% interval entirely on one side of $b$ and a one-sided Holm $p<0.05$ (Holm over the two models per contrast).}",
         r"\label{tab:v5ax}", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}lllrrrll@{}}", r"\toprule",
         r"Model & Contrast & Estimate [95\% CI] & Size pred. & hd pred. & $b$ & Holm $p$ & Classification \\", r"\midrule"]
    for j, (u, m) in enumerate((("U1", "Qwen3-0.6B (hd 128)"), ("U2", "SmolLM2-1.7B (hd 64)"))):
        s = u.lower()
        if j:
            t.append(r"\midrule")
        for i, (cn, lab) in enumerate((("natc", r"NAT$-$C1 (primary)"), ("hone", "H1 (secondary)"))):
            head = m if i == 0 else ""
            t.append(f"{head} & {lab} & {vv(f'ax.{cn}.{s}.est')} {vv(f'ax.{cn}.{s}.ci')} & {vv(f'ax.{cn}.{s}.size')} & "
                     f"{vv(f'ax.{cn}.{s}.hd')} & {vv(f'ax.{cn}.{s}.b')} & {vv(f'ax.{cn}.{s}.holm')} & {vv(f'ax.{cn}.{s}.class')} \\\\")
    t += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB / "v5ax.tex").write_text("\n".join(t) + "\n", encoding="utf-8")


def tab_h5():
    t = [r"\begin{table}[tbp]", r"\centering\footnotesize",
         r"\caption{v5 per unit: H5 ($L^K$, $L^V$ with 95\% intervals; support needs a lower bound above $\ln 4$), the frozen calibration "
         r"(margin, $\rho_K$, $\rho_V$, $\rho_V/\rho_K$) and the Holm-adjusted $p$ of each locked test of the unit's component (A: H1, H2, H3; "
         r"B: H2, H3; D: H2; H5 for A, B and D). Under HQQ, $\rho$ and the margin come from HQQ's own error.}", r"\label{tab:v5h5}",
         r"\setlength{\tabcolsep}{2.5pt}", r"\begin{tabular}{@{}l llll rrrr l@{}}", r"\toprule",
         r"Unit & $L^K$ & 95\% CI & $L^V$ & 95\% CI & Margin & $\rho_K$ & $\rho_V$ & $\rho_V/\rho_K$ & Holm $p$ (locked tests) \\",
         r"\midrule"]
    comp = {"U1": ("a", ["htwo", "hone", "hthree"]), "U2": ("a", ["htwo", "hone", "hthree"]), "U4": ("b", ["htwo", "hthree"]),
            "U6": ("b", ["htwo", "hthree"]), "U3": ("d", ["htwo"]), "U5": ("d", ["htwo"])}
    nm = {"hone": "H1", "htwo": "H2", "hthree": "H3"}
    for i, u in enumerate(["U0", "U1", "U2", "U4", "U6", "U3", "U5"]):
        s = u.lower()
        if i in (1, 3, 5):
            t.append(r"\midrule")
        if u in comp:
            c, hs = comp[u]
            ps = ", ".join(f"{nm[h]} {vv(f'holm.{c}h{nm[h][1]}.{s}')}" for h in hs) + f", H5 {vv(f'holm.{c}h5.{s}k')}"
        else:
            ps = "anchor (not tested)"
        t.append(f"{u} & {vv(f'hfiveK.{s}.est')} & {vv(f'hfiveK.{s}.ci')} & {vv(f'hfiveV.{s}.est')} & {vv(f'hfiveV.{s}.ci')} & "
                 f"{vv(f'cal.{s}.margin')} & {vv(f'cal.{s}.rhoK')} & {vv(f'cal.{s}.rhoV')} & {vv(f'cal.{s}.rhoVK')} & {ps} \\\\")
    t += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB / "v5h5.tex").write_text("\n".join(t) + "\n", encoding="utf-8")


def main():
    h = hashlib.sha256((RV / REP).read_bytes()).hexdigest()
    assert h == REPORT_SHA, f"v5_report.json changed: {h}"
    build_units(); build_families(); build_anchor_recomp_design(); build_ops(); build_ranges()
    write_macros()
    tab_main(); tab_ax(); tab_h5()
    print(f"{len(V)} v5 macros; tables in {TAB.name}/")


if __name__ == "__main__":
    main()
