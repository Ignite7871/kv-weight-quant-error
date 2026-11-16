"""Generates every v6b number macro (\\vsixb{key}) and the v6b table of main_jmlr.tex from saved results. CPU only;
reads, never writes, research outputs. Shares no code with audit/audit_v6_numbers.py (which re-derives these macros too).

Sources (read-only; paths relative to research_v2/):
  v6b_runs/v6b_report.json         locked analysis output (sha256 checked; = v6b/results/V6B_RESULTS_HASHES.txt)
  v6b_runs/v6b_recomputation.json  independent recomputation (sha256 checked)
  v6b_runs/design.json             design point (n, cuts)
  v6b_runs/gpu_ledger.jsonl        laptop GPU ledger (rehearsal and test blocks)
Outputs: paper_journal/tables_v6b/numbers_v6b.tex, numbers_v6b_sources.json, v6bsplit.tex.

  venv/Scripts/python.exe research_v2/paper_journal/make_v6b.py
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RV = HERE.parent
TAB = HERE / "tables_v6b"
REP, REC = "v6b_runs/v6b_report.json", "v6b_runs/v6b_recomputation.json"
LED, DES = "v6b_runs/gpu_ledger.jsonl", "v6b_runs/design.json"
SHA = {REP: "a766874757f178caa46af4d28873c39b27b039f25e20330b16d321a14226d98f",
       REC: "2903910cb3cbb9d6048ca9583697725bbca811236c9118d85626268f5366e55f"}
MODELS = {"qwen2.5-0.5b": "qa", "llama3.2-1b": "la", "qwen2.5-1.5b": "qb"}
NAME = {"qa": "Qwen2.5-0.5B", "la": "Llama-3.2-1B", "qb": "Qwen2.5-1.5B"}
PATHS = ("R1", "R2", "U1", "U2")
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
    if f == "s2": return f"${x:+.2f}$".replace("+-", "-")
    if f == "u3": return f"${x:.3f}$"
    if f == "u2": return f"${x:.2f}$"
    if f == "u1": return f"${x:.1f}$"
    if f == "int": return f"${int(round(x))}$"
    if f == "ci3": return f"$[{x[0]:+.3f}, {x[1]:+.3f}]$".replace("+-", "-")
    if f == "ci2": return f"$[{x[0]:+.2f}, {x[1]:+.2f}]$".replace("+-", "-")
    if f == "uci2": return f"$[{x[0]:.2f}, {x[1]:.2f}]$"
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
    return f"{REP}#models/{m}/" + "/".join(k)


def summ(key, m, path, f="s3", cif="ci3"):
    src(f"{key}.est", pm(m, *path, "estimate"), f)
    src(f"{key}.ci", pm(m, *path, "ci95"), cif)


def rng(key, ptrs, f):
    vals = [get(p) for p in ptrs]
    put(key + ".min", min(vals), f, op="min", inputs=ptrs)
    put(key + ".max", max(vals), f, op="max", inputs=ptrs)


def build():
    ms = list(MODELS)
    for m, s in MODELS.items():
        src(f"n.{s}", pm(m, "n"), "int")
        summ(f"tv.{s}", m, ["theta_V"])
        summ(f"tu.{s}", m, ["theta_U"])
        summ(f"tn.{s}", m, ["theta_N"])
        summ(f"ir.{s}", m, ["interaction_R"])
        summ(f"iu.{s}", m, ["interaction_U"])
        summ(f"ren.{s}", m, ["renorm_effect"])
        for p in PATHS:
            k = p.lower()
            summ(f"{k}.dir.{s}", m, ["paths", p, "dir"])
            summ(f"{k}.cm.{s}", m, ["paths", p, "cm"])
            src(f"{k}.sh.{s}.est", pm(m, "paths", p, "share_dir", "share"), "u2")
            src(f"{k}.sh.{s}.ci", pm(m, "paths", p, "share_dir", "ci95"), "uci2")
        src(f"bratio.{s}", pm(m, "descriptive", "B_norm_ratio_median"), "u2")
        src(f"anorm.{s}", pm(m, "descriptive", "stored_over_natural_rel_V", "NAT-V-cen0"), "u2")
        src(f"g1.{s}", pm(m, "G1", "problems"), "int")
    P = lambda *k: [pm(m, *k) for m in ms]
    for p in PATHS:
        rng(f"rng.{p.lower()}.sh", P("paths", p, "share_dir", "share"), "u2")
        rng(f"rng.{p.lower()}.shhi", P("paths", p, "share_dir", "ci95", "1"), "u2")
    allsh = [pm(m, "paths", p, "share_dir", "share") for m in ms for p in PATHS]
    rng("rng.sh", allsh, "u2")
    rng("rng.ir", P("interaction_R", "estimate"), "s2")
    rng("rng.iu", P("interaction_U", "estimate"), "s2")
    irs = P("interaction_R", "estimate") + P("interaction_U", "estimate")
    rng("rng.inter", irs, "s2")
    rng("rng.ren", P("renorm_effect", "estimate"), "s2")
    rng("rng.tn", P("theta_N", "estimate"), "s2")
    rng("rng.tu", P("theta_U", "estimate"), "s2")
    rng("rng.anorm", P("descriptive", "stored_over_natural_rel_V", "NAT-V-cen0"), "u2")
    rng("rng.bratio", P("descriptive", "B_norm_ratio_median"), "u2")
    # the cm-first and dir-first common-mode shares of the re-normed paths (1 - s_dir), printed as ranges
    for p in ("R1", "R2"):
        ins = P("paths", p, "share_dir", "share")
        put(f"rng.{p.lower()}.shcm.min", 1 - max(get(x) for x in ins), "u2", op="one_minus_max", inputs=ins)
        put(f"rng.{p.lower()}.shcm.max", 1 - min(get(x) for x in ins), "u2", op="one_minus_min", inputs=ins)
    src("verd", f"{REP}#verdict/H_star", "word")
    put("nrej", len(get(f"{REP}#verdict/rejecting_paths")), "int", op="len", inputs=[f"{REP}#verdict/rejecting_paths"])
    put("norder", len(get(f"{REP}#verdict/order_dependent")), "int", op="len", inputs=[f"{REP}#verdict/order_dependent"])
    src("thresh", f"{REP}#verdict/threshold", "u1")
    src("design.n", f"{DES}#n", "int")
    put("design.ncuts", len(get(f"{DES}#cuts")), "int", op="len", inputs=[f"{DES}#cuts"])
    led = load(LED)
    put("gpu.hours", sum(r["seconds"] for r in led) / 3600, "u1", op="ledger_hours", inputs=[LED])
    put("gpu.test", sum(r["seconds"] for r in led if r["block"].startswith("test:")) / 3600, "u1", op="ledger_hours_prefix",
        inputs=[LED, "test:"])
    src("recomp.n", f"{REC}#compared", "int")
    src("recomp.ndis", f"{REC}#outside_tolerance", "int")
    src("recomp.verd", f"{REC}#verdict", "word")
    put("sha.report", SHA[REP][:8], "tt", op="sha256_prefix", inputs=[REP])
    put("sha.recomp", SHA[REC][:8], "tt", op="sha256_prefix", inputs=[REC])


def write_macros():
    TAB.mkdir(exist_ok=True)
    L = ["% generated by make_v6b.py; do not edit", "\\makeatletter",
         "\\newcommand{\\vsixb}[1]{\\ifcsname vsixb@#1\\endcsname\\csname vsixb@#1\\endcsname\\else\\textbf{\\textcolor{red}{??#1}}\\fi}"]
    L += [f"\\expandafter\\def\\csname vsixb@{k}\\endcsname{{{v}}}" for k, v in V.items()]
    L.append("\\makeatother")
    (TAB / "numbers_v6b.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    (TAB / "numbers_v6b_sources.json").write_text(json.dumps({"sha256": SHA, "macros": SRC}, indent=1, default=float),
                                                  encoding="utf-8")


def vv(k):
    return f"\\vsixb{{{k}}}"


def cell(k):
    return f"{vv(k + '.est')} {vv(k + '.ci')}"


def tab_split():
    rows = [(r"$\theta_V$ (total of R1, R2)", "tv"),
            (r"$\theta_U$ (total of U1, U2)", "tu"),
            (r"norm term $\theta_N$", "tn")]
    paths = [("R1", r"re-normed, common mode first (v6 L3): $\theta(\text{NAT-V},\text{NAT-V-cen})+\theta(\text{NAT-V-cen},\text{C1-V})$"),
             ("R2", r"re-normed, direction first: $\theta(\text{B},\text{C1-V})+\theta(\text{NAT-V},\text{B})$"),
             ("U1", r"not re-normed, common mode first: $\theta(\text{NAT-V},\text{A})+\theta(\text{A},\text{C})$"),
             ("U2", r"not re-normed, direction first: $\theta(\text{B},\text{C})+\theta(\text{NAT-V},\text{B})$")]
    ms = list(MODELS.values())
    t = [r"\begin{table}[tbp]", r"\centering\footnotesize",
         r"\caption{The split of the value-path excess along four paths (v6b, the first " + vv("design.n") + r" v6 test problems "
         r"of each model; estimation targets, not tests). A $=$ NAT-V-cen0 (centered error, not re-normed), B $=$ C1-V+$\mu$ "
         r"(natural common mode plus isotropic position-varying part), C $=$ C1-V@cen (isotropic at the centered norm). Each path "
         r"telescopes exactly per problem into a common-mode part and a direction part; $s_{\mathrm{dir}}$ is the direction part's "
         r"share of the total (ratio of means). The interactions $I_R=\theta(\text{NAT-V},\text{B})-\theta(\text{NAT-V-cen},\text{C1-V})$ "
         r"and $I_U=\theta(\text{NAT-V},\text{B})-\theta(\text{A},\text{C})$ are the differences between the two orders; the norm term is "
         r"$\theta_N=\theta(\text{C},\text{C1-V})$ and the re-norm effect $\theta(\text{NAT-V-cen},\text{A})$. Estimates "
         r"with 95\% problem-bootstrap intervals.}",
         r"\label{tab:v6bsplit}", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}l lll@{}}", r"\toprule",
         r" & " + " & ".join(NAME[s] for s in ms) + r" \\", r"\midrule"]
    for lab, k in rows:
        t.append(lab + " & " + " & ".join(cell(f"{k}.{s}") for s in ms) + r" \\")
    for p, desc in paths:
        k = p.lower()
        t.append(r"\midrule")
        t.append(r"\multicolumn{4}{@{}l}{\textit{" + p + ", " + desc + r"}} \\")
        t.append(r"\quad common-mode part & " + " & ".join(cell(f"{k}.cm.{s}") for s in ms) + r" \\")
        t.append(r"\quad direction part & " + " & ".join(cell(f"{k}.dir.{s}") for s in ms) + r" \\")
        t.append(r"\quad $s_{\mathrm{dir}}$ & " + " & ".join(f"{vv(f'{k}.sh.{s}.est')} {vv(f'{k}.sh.{s}.ci')}" for s in ms) + r" \\")
    t += [r"\midrule",
          r"interaction $I_R$ & " + " & ".join(cell(f"ir.{s}") for s in ms) + r" \\",
          r"interaction $I_U$ & " + " & ".join(cell(f"iu.{s}") for s in ms) + r" \\",
          r"re-norm effect & " + " & ".join(cell(f"ren.{s}") for s in ms) + r" \\",
          r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB / "v6bsplit.tex").write_text("\n".join(t) + "\n", encoding="utf-8")


def main():
    for rel, h in SHA.items():
        got = hashlib.sha256((RV / rel).read_bytes()).hexdigest()
        assert got == h, f"{rel} changed: {got}"
    build()
    write_macros()
    tab_split()
    print(f"{len(V)} v6b macros; table in {TAB.name}/")


if __name__ == "__main__":
    main()
