"""Generates every v6 number macro (\\vsix{key}) and the v6 tables of main_jmlr.tex from saved results. CPU only; reads, never
writes, research outputs. Shares no code with audit/audit_v6_numbers.py.

Sources (read-only; paths relative to research_v2/):
  v6_runs/v6_report.json           locked analysis output (sha256 checked; = v6/results/V6_RESULTS_HASHES.txt)
  v6_runs/v6_recomputation.json    independent recomputation (sha256 checked)
  v6_runs/design.json              design point after V6-D1 (n, arms, cuts)
  v6_runs/gpu_ledger.jsonl         laptop GPU ledger
  v6/results/R_block_report.json   R block (pre-specified secondary analyses of the v4/v5 records)
Outputs: paper_journal/tables_v6/numbers_v6.tex, numbers_v6_sources.json, v6main.tex, v6tests.tex, rblock.tex.

  venv/Scripts/python.exe research_v2/paper_journal/make_v6.py
"""
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
RV = HERE.parent
TAB = HERE / "tables_v6"
REP, REC = "v6_runs/v6_report.json", "v6_runs/v6_recomputation.json"
RB = "v6/results/R_block_report.json"
SHA = {REP: "98626e3ec4a26da93fc6e45ac0a104264ad9fd914f8b14bb26b36c2ab246acc1",
       REC: "ec169a0d628cb2ea875406806afdc7d320333395ba08978f92a248ca804b5be6"}
MODELS = {"qwen2.5-0.5b": "qa", "llama3.2-1b": "la", "qwen2.5-1.5b": "qb"}
NAME = {"qa": "Qwen2.5-0.5B", "la": "Llama-3.2-1B", "qb": "Qwen2.5-1.5B"}
HD = {"qa": 64, "la": 64, "qb": 128}
RUNITS = {"qwen2.5-0.5b": "Qwen2.5-0.5B", "qwen2.5-1.5b": "Qwen2.5-1.5B", "qwen2.5-3b": "Qwen2.5-3B", "qwen2.5-7b": "Qwen2.5-7B",
          "llama3.2-1b": "Llama-3.2-1B", "llama3.2-3b": "Llama-3.2-3B", "llama3.1-8b": "Llama-3.1-8B",
          "mistral-7b-v0.3": "Mistral-7B-v0.3", "U1": "Qwen3-0.6B-Base", "U2": "SmolLM2-1.7B"}
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
    if f == "s4": return f"${x:+.4f}$".replace("+-", "-")
    if f == "s2": return f"${x:+.2f}$".replace("+-", "-")
    if f == "u3": return f"${x:.3f}$"
    if f == "u2": return f"${x:.2f}$"
    if f == "u1": return f"${x:.1f}$"
    if f == "int": return f"${int(round(x))}$"
    if f == "intc": return f"${int(round(x)):,}$".replace(",", "{,}")
    if f == "ci3": return f"$[{x[0]:+.3f}, {x[1]:+.3f}]$".replace("+-", "-")
    if f == "ci4": return f"$[{x[0]:+.4f}, {x[1]:+.4f}]$".replace("+-", "-")
    if f == "ci2": return f"$[{x[0]:+.2f}, {x[1]:+.2f}]$".replace("+-", "-")
    if f == "uci2": return f"$[{x[0]:.2f}, {x[1]:.2f}]$"
    if f == "p": return "$<0.001$" if x < 0.001 else f"${x:.3f}$"
    if f == "pct0": return f"${100 * x:.0f}$\\%"
    if f == "word": return str(x)
    if f == "tt": return f"\\texttt{{{x}}}"
    raise ValueError(f)


def put(key, raw, f, src=None, op=None, inputs=None):
    assert key not in V, key
    V[key] = fmt(raw, f)
    SRC[key] = {"val": V[key], "raw": raw, "fmt": f, **({"src": src} if src else {"op": op, "inputs": inputs})}


def src(key, ptr, f):
    put(key, get(ptr), f, src=ptr)


def pm(m, *k):
    return f"{REP}#per_model/{m}/" + "/".join(k)


def summ(key, m, path, f="s3", cif="ci3", ci90=False):
    src(f"{key}.est", pm(m, *path, "estimate"), f)
    src(f"{key}.ci", pm(m, *path, "ci95"), cif)
    if ci90:
        src(f"{key}.cininety", pm(m, *path, "ci90"), cif)


def rng(key, ptrs, f, op_min="min", op_max="max", tf=lambda x: x):
    vals = [tf(get(p)) for p in ptrs]
    put(key + ".min", min(vals), f, op=op_min, inputs=ptrs)
    put(key + ".max", max(vals), f, op=op_max, inputs=ptrs)


# ------------------------------------------------------------------ macros
def build_models():
    for m, s in MODELS.items():
        src(f"n.{s}", pm(m, "n"), "int")
        put(f"narms.{s}", len(get(pm(m, "arms"))), "int", op="len", inputs=[pm(m, "arms")])
        summ(f"tv.{s}", m, ["L3", "theta_V"])
        summ(f"tcm.{s}", m, ["L3", "theta_cm"])
        summ(f"tdir.{s}", m, ["L3", "theta_dir"])
        src(f"scm.{s}.est", pm(m, "L3", "share_cm", "share"), "u2")
        src(f"scm.{s}.ci", pm(m, "L3", "share_cm", "ci95"), "uci2")
        put(f"sdir.{s}.est", 1 - get(pm(m, "L3", "share_cm", "share")), "u2", op="one_minus", inputs=[pm(m, "L3", "share_cm", "share")])
        summ(f"lcm.{s}", m, ["L4", "cm"])
        summ(f"ldir.{s}", m, ["L4", "dir"])
        src(f"lsh.{s}.est", pm(m, "L4", "share_cm", "share"), "u2")
        src(f"lsh.{s}.ci", pm(m, "L4", "share_cm", "ci95"), "uci2")
        src(f"lone.V.{s}", pm(m, "L1", "V_mod_vs_meas", "median_abs_ln"), "u3")
        src(f"lone.K.{s}", pm(m, "L1", "K_lin_vs_meas", "median_abs_ln"), "u3")
        src(f"lone.nV.{s}", pm(m, "L1", "V_mod_vs_meas", "n_slot_steps"), "intc")
        src(f"lone.nK.{s}", pm(m, "L1", "K_lin_vs_meas", "n_slot_steps"), "intc")
        src(f"ltwo.{s}.rho", pm(m, "L2", "rho"), "s3")
        src(f"ltwo.{s}.ci", pm(m, "L2", "ci95"), "ci3")
        src(f"ltwo.{s}.holm", f"{REP}#holm_p/L2:{m}", "p")
        src(f"lfive.{s}", pm(m, "L5", "median_abs_ln_ratio"), "u3")
        src(f"lfive.n.{s}", pm(m, "L5", "n"), "intc")
        src(f"lsix.{s}.rho", pm(m, "L6", "spearman", "rho"), "s3")
        src(f"lsix.{s}.ci", pm(m, "L6", "spearman", "ci95"), "ci3")
        src(f"lsix.{s}.holm", f"{REP}#holm_p/L6:{m}", "p")
        src(f"lsix.{s}.lnr", pm(m, "L6", "model_ln_ratio_linK"), "s3")
        summ(f"hone.{s}", m, ["L6", "theta_H1"])
        src(f"lsev.i.{s}", pm(m, "L7", "i_nC_V_ST"), "u2")
        summ(f"lsev.K.{s}", m, ["L7", "ii_STK@nat_vs_C1K"], ci90=True)
        summ(f"lsev.V.{s}", m, ["L7", "ii_STV@nat_vs_C1V"], ci90=True)
        src(f"lsev.margin.{s}", pm(m, "L7", "margin"), "u3")
        src(f"lsev.ratio.{s}", pm(m, "L7", "iii_rho_ratio_K_over_V"), "u2")
        summ(f"lsev.st.{s}", m, ["L7", "iii_STK_vs_STV"])
        src(f"lsev.st.holm.{s}", f"{REP}#holm_p/L7iii:{m}", "p")
        summ(f"hthree.{s}", m, ["L7", "H3_replication"])
        for t, key in (("L8a", "leighta"), ("L8b", "leightb"), ("L9a", "lninea"), ("L9b", "lnineb")):
            fmt_e, fmt_c = ("s4", "ci4") if t == "L8b" else ("s3", "ci3")
            summ(f"{key}.{s}", m, [t], f=fmt_e, cif=fmt_c)
            src(f"{key}.{s}.holm", f"{REP}#holm_p/{t}:{m}", "p")
        # descriptive: median over layers of the per-layer median alignment ratio, per arm
        for arm, ak in (("NAT-V", "nat"), ("C1-V", "cone"), ("ST-V@nat", "st"), ("NAT-V-cen", "natcen")):
            for q, qk in (("A_cov", "acov"), ("A_o", "ao")):
                p = pm(m, "descriptive", "per_layer_alignment_median", arm, q)
                vals = sorted(get(p))
                med = vals[len(vals) // 2] if len(vals) % 2 else 0.5 * (vals[len(vals) // 2 - 1] + vals[len(vals) // 2])
                put(f"{qk}.{ak}.{s}", med, "u2", op="median_of_list", inputs=[p])
        p = pm(m, "descriptive", "per_layer_m_ND_NAT-V", "N_D")
        put(f"nd.{s}.mean", sum(get(p)) / len(get(p)), "u1", op="mean_of_list", inputs=[p])
        p = pm(m, "descriptive", "per_layer_m_ND_NAT-V", "m")
        put(f"mass.{s}.mean", sum(get(p)) / len(get(p)), "u2", op="mean_of_list", inputs=[p])
    # the per-channel key arms ran in Qwen2.5-0.5B only (first in the cut order)
    m = "qwen2.5-0.5b"
    summ("kc.st.qa", m, ["descriptive", "ST-Kc_vs_ST-V"])
    summ("kc.nat.qa", m, ["descriptive", "ST-Kc@nat_vs_C1-K"])
    summ("tcmK.qa", m, ["L3_K_descriptive", "theta_cm_K"])
    summ("tdirK.qa", m, ["L3_K_descriptive", "theta_dir_K"])


def build_verdicts_ranges():
    for t in ("L1", "L2", "L5", "L6", "L8a", "L8b", "L9a", "L9b"):
        src(f"verd.{t}", f"{REP}#verdicts/{t}", "word")
    src("verd.L7", f"{REP}#verdicts/L7/verdict", "word")
    for part in ("i", "ii", "iii"):
        put(f"verd.L7.{part}", "holds" if get(f"{REP}#verdicts/L7/{part}") else "fails", "word", op="bool_word",
            inputs=[f"{REP}#verdicts/L7/{part}"])
    ms = list(MODELS)
    P = lambda *k: [pm(m, *k) for m in ms]
    rng("rng.scm", P("L3", "share_cm", "share"), "u2")
    put("rng.sdir.min", 1 - max(get(p) for p in P("L3", "share_cm", "share")), "u2", op="one_minus_max", inputs=P("L3", "share_cm", "share"))
    put("rng.sdir.max", 1 - min(get(p) for p in P("L3", "share_cm", "share")), "u2", op="one_minus_min", inputs=P("L3", "share_cm", "share"))
    rng("rng.lsh", P("L4", "share_cm", "share"), "u2")
    rng("rng.tv", P("L3", "theta_V", "estimate"), "s3")
    rng("rng.tcm", P("L3", "theta_cm", "estimate"), "s3")
    rng("rng.tdir", P("L3", "theta_dir", "estimate"), "s3")
    rng("rng.loneV", P("L1", "V_mod_vs_meas", "median_abs_ln"), "u3")
    rng("rng.loneK", P("L1", "K_lin_vs_meas", "median_abs_ln"), "u3")
    rng("rng.lfive", P("L5", "median_abs_ln_ratio"), "u3")
    rng("rng.leighta", P("L8a", "estimate"), "s3")
    rng("rng.leightb", P("L8b", "estimate"), "s4")
    rng("rng.lninea", P("L9a", "estimate"), "s3")
    rng("rng.lnineb", P("L9b", "estimate"), "s3")
    rng("rng.ltwo", P("L2", "rho"), "s3")
    rng("rng.ltwohi", P("L2", "ci95", "1"), "s3")
    rng("rng.lsix", P("L6", "spearman", "rho"), "s3")
    rng("rng.lseviV", P("L7", "i_nC_V_ST"), "u2")
    rng("rng.lsevratio", P("L7", "iii_rho_ratio_K_over_V"), "u2")
    rng("rng.lsevst", P("L7", "iii_STK_vs_STV", "estimate"), "s3")
    rng("rng.hone", P("L6", "theta_H1", "estimate"), "s3")
    rng("rng.hthree", P("L7", "H3_replication", "estimate"), "s3")
    # exp of the alignment differences: geometric-mean alignment ratio of NAT-V to C1-V (L8a) and to ST-V@nat (L9b)
    rng("rng.eleighta", P("L8a", "estimate"), "u2", "exp_min", "exp_max", math.exp)
    rng("rng.elnineb", P("L9b", "estimate"), "u2", "exp_min", "exp_max", math.exp)
    # Holm p of every counted test (all-max printed once)
    hp = [f"{REP}#holm_p/{k}" for k in get(f"{REP}#holm_p")]
    put("holm.n", len(hp), "int", op="len", inputs=[f"{REP}#holm_p"])
    sig = [f"{REP}#holm_p/{t}:{m}" for t in ("L7iii", "L8a", "L8b", "L9a", "L9b") for m in ms]
    # printed as "Holm p <= x": rounded UP to 3 decimals so the printed bound holds
    put("holm.sig.max", math.ceil(1000 * max(get(p) for p in sig)) / 1000, "u3", op="ceil3_max", inputs=sig)
    # printed as "every Holm p above x" for L2: rounded DOWN to 1 decimal
    ns = [f"{REP}#holm_p/L2:{m}" for m in ms]
    put("holm.ltwo.floor", math.floor(10 * min(get(p) for p in ns)) / 10, "u1", op="floor1_min", inputs=ns)
    rng("rng.lonenV", P("L1", "V_mod_vs_meas", "n_slot_steps"), "intc")
    rng("rng.lonenK", P("L1", "K_lin_vs_meas", "n_slot_steps"), "intc")
    # per-layer median A_o, median over layers, over every logged value arm and model
    ao = []
    for m in ms:
        for arm in get(pm(m, "descriptive", "per_layer_alignment_median")):
            p = pm(m, "descriptive", "per_layer_alignment_median", arm, "A_o")
            vals = sorted(get(p))
            ao.append((p, vals[len(vals) // 2] if len(vals) % 2 else 0.5 * (vals[len(vals) // 2 - 1] + vals[len(vals) // 2])))
    put("rng.ao.min", min(v for _, v in ao), "u2", op="min_of_list_medians", inputs=[p for p, _ in ao])
    put("rng.ao.max", max(v for _, v in ao), "u2", op="max_of_list_medians", inputs=[p for p, _ in ao])
    for arm, k in (("NAT-V", "nat"), ("C1-V", "cone"), ("ST-V@nat", "st")):
        vals = []
        for m in ms:
            p = pm(m, "descriptive", "per_layer_alignment_median", arm, "A_cov")
            L = sorted(get(p))
            vals.append((p, L[len(L) // 2] if len(L) % 2 else 0.5 * (L[len(L) // 2 - 1] + L[len(L) // 2])))
        put(f"rng.acov.{k}.min", min(v for _, v in vals), "u2", op="min_of_list_medians", inputs=[p for p, _ in vals])
        put(f"rng.acov.{k}.max", max(v for _, v in vals), "u2", op="max_of_list_medians", inputs=[p for p, _ in vals])


def build_design_recomp():
    src("design.n", "v6_runs/design.json#n", "int")
    put("design.narms", len(get("v6_runs/design.json#arms")), "int", op="len", inputs=["v6_runs/design.json#arms"])
    led = load("v6_runs/gpu_ledger.jsonl")
    put("gpu.hours", sum(r["seconds"] for r in led) / 3600, "u1", op="ledger_hours", inputs=["v6_runs/gpu_ledger.jsonl"])
    put("gpu.test", sum(r["seconds"] for r in led if r["block"].startswith("test:")) / 3600, "u1", op="ledger_hours_prefix",
        inputs=["v6_runs/gpu_ledger.jsonl", "test:"])
    src("recomp.n", f"{REC}#comparison/n_compared", "int")
    src("recomp.ndis", f"{REC}#comparison/n_outside_tolerance", "int")
    vr, vc = get(f"{REP}#verdicts"), get(f"{REC}#verdicts")
    put("recomp.nverd", sum(vr[k] != vc[k] for k in vr), "int", op="count_verdict_diff", inputs=[f"{REP}#verdicts", f"{REC}#verdicts"])
    put("sha.report", SHA[REP][:8], "tt", op="sha256_prefix", inputs=[REP])
    put("sha.recomp", SHA[REC][:8], "tt", op="sha256_prefix", inputs=[REC])


def build_rblock():
    U = list(RUNITS)
    P = lambda *k: [f"{RB}#units/{u}/" + "/".join(k) for u in U]
    put("rb.nunits", len(U), "int", op="len", inputs=[f"{RB}#units"])
    rng("rb.rtwo", P("R2", "estimate"), "s3")
    rng("rb.rtwolo", P("R2", "ci95", "0"), "s3")
    rng("rb.rfourK", P("R4", "rho_K"), "u3")
    rng("rb.rfourV", P("R4", "rho_V_placebo"), "s3")
    rng("rb.rone", P("R1", "estimate"), "s3")
    put("rb.rone.npos", len(get(f"{RB}#verdicts/R1/positive_significant")), "int", op="len", inputs=[f"{RB}#verdicts/R1/positive_significant"])
    put("rb.rone.npt", len(get(f"{RB}#verdicts/R1/positive_point")), "int", op="len", inputs=[f"{RB}#verdicts/R1/positive_point"])
    put("rb.rthree.npos", len(get(f"{RB}#verdicts/R3/units_rho_positive")), "int", op="len", inputs=[f"{RB}#verdicts/R3/units_rho_positive"])
    src("rb.rthree.rho", f"{RB}#R3_pooled/rho", "s3")
    src("rb.rthree.ci", f"{RB}#R3_pooled/ci95", "ci3")
    for r in ("R1", "R3", "R4"):
        src(f"rb.verd.{r}", f"{RB}#verdicts/{r}/verdict", "word")
    for u in U:
        k = u.lower().replace(".", "").replace("-", "")
        for r, f in (("R1", "s3"), ("R2", "s3")):
            src(f"rb.{r.lower()}.{k}.est", f"{RB}#units/{u}/{r}/estimate", f)
            src(f"rb.{r.lower()}.{k}.ci", f"{RB}#units/{u}/{r}/ci95", "ci2")
        src(f"rb.r1.{k}.holm", f"{RB}#units/{u}/R1/holm_p", "p")
        src(f"rb.r3.{k}", f"{RB}#units/{u}/R3/rho", "s3")
        src(f"rb.r4K.{k}", f"{RB}#units/{u}/R4/rho_K", "u3")
        src(f"rb.r4V.{k}", f"{RB}#units/{u}/R4/rho_V_placebo", "s3")


def write_macros():
    TAB.mkdir(exist_ok=True)
    L = ["% generated by make_v6.py; do not edit", "\\makeatletter",
         "\\newcommand{\\vsix}[1]{\\ifcsname vsix@#1\\endcsname\\csname vsix@#1\\endcsname\\else\\textbf{\\textcolor{red}{??#1}}\\fi}"]
    L += [f"\\expandafter\\def\\csname vsix@{k}\\endcsname{{{v}}}" for k, v in V.items()]
    L.append("\\makeatother")
    (TAB / "numbers_v6.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    (TAB / "numbers_v6_sources.json").write_text(json.dumps({"sha256": SHA, "macros": SRC}, indent=1, default=float), encoding="utf-8")


# ------------------------------------------------------------------ tables
def vv(k):
    return f"\\vsix{{{k}}}"


def st(est, ci):
    """estimate over its interval, stacked in one cell (makecell)"""
    return f"\\makecell[l]{{{vv(est)}\\\\{vv(ci)}}}"


def tab_main():
    t = [r"\begin{table}[tbp]", r"\centering\footnotesize",
         r"\caption{Where the value-path excess comes from (v6, fresh problems). $\theta_V=\theta(\text{NAT-V},\text{C1-V})$ (H2) "
         r"splits exactly, per problem, into a common-mode part $\theta_{\mathrm{cm}}=\theta(\text{NAT-V},\text{NAT-V-cen})$ and a "
         r"direction part $\theta_{\mathrm{dir}}=\theta(\text{NAT-V-cen},\text{C1-V})$, the remainder after centering at fixed per-slot "
         r"norm (L3, an estimation target, not a test; one order of the split, Table~\ref{tab:v6bsplit} gives the others; Section~\ref{sec:model:direction} states what the remainder contains). The last column is the "
         r"same split of the measured local head-output energy (L4). Estimates with 95\% problem-bootstrap intervals below them.}",
         r"\label{tab:v6main}", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}lr lll ll@{}}", r"\toprule",
         r" & & \multicolumn{3}{c}{Outcome contrast (L3)} & \multicolumn{2}{c}{Common-mode share} \\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}",
         r"Model & $n$ & $\theta_V$ & $\theta_{\mathrm{cm}}$ & $\theta_{\mathrm{dir}}$ & of $\theta_V$ (L3) & of energy (L4) \\",
         r"\midrule"]
    for s in MODELS.values():
        t.append(f"{NAME[s]} & {vv(f'n.{s}')} & {st(f'tv.{s}.est', f'tv.{s}.ci')} & {st(f'tcm.{s}.est', f'tcm.{s}.ci')} & "
                 f"{st(f'tdir.{s}.est', f'tdir.{s}.ci')} & {st(f'scm.{s}.est', f'scm.{s}.ci')} & {st(f'lsh.{s}.est', f'lsh.{s}.ci')} \\\\")
    t += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB / "v6main.tex").write_text("\n".join(t) + "\n", encoding="utf-8")


def tab_tests():
    rows = [
        ("L1", r"median $|\ln(\mathrm{pred}/\mathrm{meas})|$, V; K", lambda s: f"{vv(f'lone.V.{s}')}; {vv(f'lone.K.{s}')}"),
        ("L5", r"median $|\ln(c_p^{\mathrm{post}}/\mathrm{pred})|$", lambda s: vv(f"lfive.{s}")),
        ("L8a", r"$\Delta\ln A_{\mathrm{cov}}$, NAT-V$-$C1-V", lambda s: st(f"leighta.{s}.est", f"leighta.{s}.ci")),
        ("L8b", r"$\Delta\ln A_o$, NAT-V$-$C1-V", lambda s: st(f"leightb.{s}.est", f"leightb.{s}.ci")),
        ("L9a", r"$\theta$(NAT-V-cen, ST-V@nat-cen)", lambda s: st(f"lninea.{s}.est", f"lninea.{s}.ci")),
        ("L9b", r"$\Delta\ln A_{\mathrm{cov}}$, NAT-V$-$ST-V@nat", lambda s: st(f"lnineb.{s}.est", f"lnineb.{s}.ci")),
        ("L2", r"Spearman $\rho$, energy ratio vs.\ $\theta_i$ (V)", lambda s: st(f"ltwo.{s}.rho", f"ltwo.{s}.ci")),
        ("L6", r"Spearman $\rho$, \texttt{lin\_K} ratio vs.\ $\theta_i$ (K)", lambda s: st(f"lsix.{s}.rho", f"lsix.{s}.ci")),
    ]
    t = [r"\begin{table}[tbp]", r"\centering\footnotesize",
         r"\caption{Counted tests of the v6 block on fresh problems, with the verdict under the criteria fixed before the rehearsal "
         r"(Section~\ref{sec:program:hyp}). Agreement needs: L1, medians $\le\ln1.25$ (V) and $\le\ln1.5$ (K) in every model; L5, "
         r"$\le\ln1.5$ in every model; L8a, L8b, L2, a 95\% interval above 0 and Holm $p<0.05$ in every model; L9a, L9b, the same in "
         r"at least two models and no interval below 0; L6, model-level signs agreeing with H1 and $\rho$ positive-significant in at "
         r"least two models. Holm runs over " + vv("holm.n") + r" model-level $p$-values. Alignment differences are per-problem "
         r"means of log ratios to the isotropic expectation. L7 is in Table~\ref{tab:v6st}. Estimates with 95\% intervals below them.}",
         r"\label{tab:v6tests}", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}l p{3.3cm} lll l@{}}", r"\toprule",
         r"Test & Statistic & " + " & ".join(NAME[s] for s in MODELS.values()) + r" & Verdict \\", r"\midrule"]
    for tid, stat, cell in rows:
        t.append(f"{tid} & {stat} & " + " & ".join(cell(s) for s in MODELS.values()) + f" & {vv('verd.' + tid)} \\\\")
    t += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB / "v6tests.tex").write_text("\n".join(t) + "\n", encoding="utf-8")


def tab_storage():
    t = [r"\begin{table}[tbp]", r"\centering\footnotesize",
         r"\caption{Storage error (per-token asymmetric INT4, groups of 32) against weight error, v6 (L7). (i) The common-mode share of "
         r"ST-V times $n$ (isotropic $\approx1$; agreement needs $\le2$). (ii) Storage-error directions re-normed to the natural NF4 "
         r"per-slot norm against isotropic noise of that norm, with 90\% intervals; equivalence needs the interval inside "
         r"$\pm$margin on both paths. (iii) The relative-size ratio $\rho_K^{\mathrm{st}}/\rho_V^{\mathrm{st}}$ and $\theta$(ST-K, ST-V) at "
         r"codec size (95\% interval), with H3, $\theta$(NAT-K, NAT-V), from the same runs. Verdict: (i) " + vv("verd.L7.i") + r", (ii) "
         + vv("verd.L7.ii") + r", (iii) " + vv("verd.L7.iii") + r"; L7 " + vv("verd.L7") + r".}", r"\label{tab:v6st}",
         r"\setlength{\tabcolsep}{2.5pt}", r"\begin{tabular}{@{}l l lll ll l@{}}", r"\toprule",
         r" & (i) & \multicolumn{3}{c}{(ii)} & \multicolumn{2}{c}{(iii)} & \\", r"\cmidrule(lr){2-2}\cmidrule(lr){3-5}\cmidrule(lr){6-7}",
         r"Model & $nC^V$ & \makecell[l]{ST-K@nat\\$-$C1-K} & \makecell[l]{ST-V@nat\\$-$C1-V} & margin & $\rho_K^{\mathrm{st}}/\rho_V^{\mathrm{st}}$ & \makecell[l]{ST-K\\$-$ST-V} & H3 \\",
         r"\midrule"]
    for s in MODELS.values():
        t.append(f"{NAME[s]} & {vv(f'lsev.i.{s}')} & {st(f'lsev.K.{s}.est', f'lsev.K.{s}.cininety')} & "
                 f"{st(f'lsev.V.{s}.est', f'lsev.V.{s}.cininety')} & $\\pm${vv(f'lsev.margin.{s}')} & {vv(f'lsev.ratio.{s}')} & "
                 f"{st(f'lsev.st.{s}.est', f'lsev.st.{s}.ci')} & {st(f'hthree.{s}.est', f'hthree.{s}.ci')} \\\\")
    t += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB / "v6st.tex").write_text("\n".join(t) + "\n", encoding="utf-8")


def tab_rblock():
    t = [r"\begin{table}[tbp]", r"\centering\footnotesize",
         r"\caption{R block: pre-specified secondary analyses of the v4 and v5 records (not blind; the aggregate results of these "
         r"records had been seen). R1: slope of the per-step log ratio NAT-V/C1-V on $\ln t$; R2: that log ratio at step 2, where "
         r"only one perturbed entry exists; R3: problem-level Spearman of the value common-mode statistic with H2; R4: Spearman of "
         r"the per-rotary-pair share of the key common mode with the RoPE attenuation factor, and the same on values as a placebo. "
         r"95\% intervals. Verdicts: R1 " + vv("rb.verd.R1") + ", R3 " + vv("rb.verd.R3") + ", R4 " + vv("rb.verd.R4") + r".}",
         r"\label{tab:rblock}", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}l ll ll r rr@{}}", r"\toprule",
         r"Model & R1 slope & 95\% CI & R2 $r_2$ & 95\% CI & R3 $\rho$ & R4 $\rho_K$ & placebo $\rho_V$ \\", r"\midrule"]
    for u, nm in RUNITS.items():
        k = u.lower().replace(".", "").replace("-", "")
        t.append(f"{nm} & {vv(f'rb.r1.{k}.est')} & {vv(f'rb.r1.{k}.ci')} & {vv(f'rb.r2.{k}.est')} & {vv(f'rb.r2.{k}.ci')} & "
                 f"{vv(f'rb.r3.{k}')} & {vv(f'rb.r4K.{k}')} & {vv(f'rb.r4V.{k}')} \\\\")
    t += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB / "rblock.tex").write_text("\n".join(t) + "\n", encoding="utf-8")


def figure():
    """Two panels: (a) the L3 split of theta_V per model; (b) common-mode share of the outcome (L3) vs the head-output energy (L4)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    BLUE, ORANGE, INK, GRID = "#2a78d6", "#eb6834", "#3a3a38", "#e4e3df"
    ms = list(MODELS)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.0, 2.6), gridspec_kw={"width_ratios": [1.35, 1]})
    comps = [("theta_V", r"$\theta_V$ (total, H2)", INK, "o"), ("theta_cm", r"$\theta_{\mathrm{cm}}$ (common mode)", BLUE, "s"),
             ("theta_dir", r"$\theta_{\mathrm{dir}}$ (direction)", ORANGE, "D")]
    for j, (k, lab, c, mk) in enumerate(comps):
        for i, m in enumerate(ms):
            s = get(pm(m, "L3", k))
            y = i + (j - 1) * 0.22
            a1.errorbar(s["estimate"], y, xerr=[[s["estimate"] - s["ci95"][0]], [s["ci95"][1] - s["estimate"]]], fmt=mk, ms=4.5,
                        color=c, ecolor=c, elinewidth=1.2, capsize=0, label=lab if i == 0 else None)
    a1.set_yticks(range(len(ms)), [NAME[MODELS[m]] for m in ms]); a1.invert_yaxis()
    a1.axvline(0, color=GRID, lw=1, zorder=0); a1.grid(axis="x", color=GRID, lw=0.6); a1.set_axisbelow(True)
    a1.set_xlabel("(a) split of the value-path excess:\ncontrast (difference of log mean JS)"); a1.set_xlim(0, 1.15)
    a1.legend(frameon=False, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, handletextpad=0.3)
    for j, (blk, lab, c, mk) in enumerate((("L3", "outcome (L3)", BLUE, "s"), ("L4", "head-output energy (L4)", ORANGE, "D"))):
        for i, m in enumerate(ms):
            s = get(pm(m, blk, "share_cm"))
            y = i + (j - 0.5) * 0.25
            a2.errorbar(s["share"], y, xerr=[[s["share"] - s["ci95"][0]], [s["ci95"][1] - s["share"]]], fmt=mk, ms=4.5, color=c,
                        ecolor=c, elinewidth=1.2, capsize=0, label=lab if i == 0 else None)
    a2.set_yticks(range(len(ms)), [""] * len(ms)); a2.invert_yaxis(); a2.set_xlim(0, 1)
    a2.grid(axis="x", color=GRID, lw=0.6); a2.set_axisbelow(True)
    a2.set_xlabel("(b) common-mode share\nof outcome and of head-output energy")
    a2.legend(frameon=False, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, handletextpad=0.3)
    fig.tight_layout()
    out = HERE / "figures_v6"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "fig_split.pdf")
    plt.close(fig)


def main():
    for rel, h in SHA.items():
        got = hashlib.sha256((RV / rel).read_bytes()).hexdigest()
        assert got == h, f"{rel} changed: {got}"
    build_models(); build_verdicts_ranges(); build_design_recomp(); build_rblock()
    write_macros()
    tab_main(); tab_tests(); tab_storage(); tab_rblock()
    figure()
    print(f"{len(V)} v6 macros; tables in {TAB.name}/")


if __name__ == "__main__":
    main()
