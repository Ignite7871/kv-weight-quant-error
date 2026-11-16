"""Number audit of the v6 and v6b material in main_jmlr.tex. CPU only; read-only. Shares no code with make_v6.py or make_v6b.py.

1. Every \\vsix{key} used in main_jmlr.tex, sections_jmlr/ and tables_v6/ is defined in tables_v6/numbers_v6.tex.
2. Every v6 macro is RE-DERIVED from the source or operation declared in numbers_v6_sources.json, with this script's own JSON
   reader, operations and formatter; the printed string must be identical. The two report hashes are re-checked.
3. Every v6 result macro that the independent recomputation (v6_runs/v6_recomputation.json) also holds is re-checked against it:
   estimates equal at the printed precision (or within the declared float32 tolerance), CI endpoints within 5 % of the CI width
   plus half a printed unit, Holm p within 0.01 + 10 %, verdict words identical.
4. Prose claims of the v6 text (and the v6 sentences of abstract, introduction, discussion, limitations, conclusion) are checked
   against the locked report, the R block report and the ledger.
5. The same for the v6b macros (\\vsixb, tables_v6b/numbers_v6b.tex, v6b_runs/ reports) and the v6b prose claims (FOLLOWUP_V6B).
Writes audit/V6_NUMBER_AUDIT.md and audit/V6_NUMBER_AUDIT.json.
"""
import hashlib
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
PJ = HERE.parent
RV = PJ.parent
REP, REC, RB = "v6_runs/v6_report.json", "v6_runs/v6_recomputation.json", "v6/results/R_block_report.json"
HASHES = {REP: "98626e3ec4a26da93fc6e45ac0a104264ad9fd914f8b14bb26b36c2ab246acc1",
          REC: "ec169a0d628cb2ea875406806afdc7d320333395ba08978f92a248ca804b5be6"}
M = ("qwen2.5-0.5b", "llama3.2-1b", "qwen2.5-1.5b")
_c = {}


def J(rel):
    if rel not in _c:
        raw = (RV / rel).read_text(encoding="utf-8")
        _c[rel] = [json.loads(x) for x in raw.splitlines() if x.strip()] if rel.endswith(".jsonl") else json.loads(raw)
    return _c[rel]


def at(ptr):
    f, path = ptr.split("#", 1)
    node = J(f)
    for part in filter(None, path.split("/")):
        part = part.replace("~1", "/")
        node = node[int(part)] if isinstance(node, list) else node[part]
    return node


def sgn(x, d):
    s = f"{abs(x):.{d}f}"
    return ("-" if x < 0 and float(s) != 0 else "+") + s


def show(x, f):
    if f in ("s2", "s3", "s4"):
        return f"${sgn(x, int(f[1]))}$"
    if f in ("u1", "u2", "u3"):
        return f"${x:.{int(f[1])}f}$"
    if f == "int":
        return f"${int(round(x))}$"
    if f == "intc":
        return "$" + f"{int(round(x)):,}".replace(",", "{,}") + "$"
    if f in ("ci2", "ci3", "ci4"):
        d = int(f[2])
        return f"$[{sgn(x[0], d)}, {sgn(x[1], d)}]$"
    if f == "uci2":
        return f"$[{x[0]:.2f}, {x[1]:.2f}]$"
    if f == "p":
        return "$<0.001$" if x < 0.001 else f"${x:.3f}$"
    if f == "word":
        return str(x)
    if f == "tt":
        return "\\texttt{" + str(x) + "}"
    raise ValueError(f)


def med(v):
    v = sorted(v)
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


def derive(e):
    if "src" in e:
        return at(e["src"])
    op, ins = e["op"], e["inputs"]
    vals = lambda: [at(p) for p in ins]
    table = {
        "len": lambda: len(at(ins[0])),
        "one_minus": lambda: 1 - at(ins[0]),
        "one_minus_max": lambda: 1 - max(vals()),
        "one_minus_min": lambda: 1 - min(vals()),
        "min": lambda: min(vals()), "max": lambda: max(vals()),
        "exp_min": lambda: math.exp(min(vals())), "exp_max": lambda: math.exp(max(vals())),
        "median_of_list": lambda: med(at(ins[0])),
        "mean_of_list": lambda: sum(at(ins[0])) / len(at(ins[0])),
        "min_of_list_medians": lambda: min(med(at(p)) for p in ins),
        "max_of_list_medians": lambda: max(med(at(p)) for p in ins),
        "ceil3_max": lambda: math.ceil(1000 * max(vals())) / 1000,
        "floor1_min": lambda: math.floor(10 * min(vals())) / 10,
        "bool_word": lambda: "holds" if at(ins[0]) else "fails",
        "ledger_hours": lambda: sum(r["seconds"] for r in J(ins[0])) / 3600,
        "ledger_hours_prefix": lambda: sum(r["seconds"] for r in J(ins[0]) if r["block"].startswith(ins[1])) / 3600,
        "count_verdict_diff": lambda: sum(at(ins[0])[k] != at(ins[1])[k] for k in at(ins[0])),
        "sha256_prefix": lambda: hashlib.sha256((RV / ins[0]).read_bytes()).hexdigest()[:8],
    }
    return table[op]()


RENAME = {"ii_STK@nat_vs_C1K": "ii_K", "ii_STV@nat_vs_C1V": "ii_V", "iii_rho_ratio_K_over_V": "iii_rho_ratio",
          "iii_STK_vs_STV": "iii_theta", "H3_replication": "H3", "H2_this_run": "H2"}


def rec_ptr(src):
    if not src.startswith(REP + "#"):
        return None
    path = src.split("#", 1)[1]
    for a, b in RENAME.items():
        path = path.replace("/" + a + "/", "/" + b + "/").replace("/" + a, "/" + b) if path.endswith(a) or ("/" + a + "/") in path else path
    try:
        at(REC + "#" + path)
        return REC + "#" + path
    except (KeyError, IndexError, TypeError):
        return None


def check_rec(key, e, printed):
    """returns None (not checkable), True or False"""
    rp = rec_ptr(e["src"])
    if rp is None:
        return None
    lk, rc = at(e["src"]), at(rp)
    f = e["fmt"]
    if f == "word":
        return lk == rc
    if f in ("ci2", "ci3", "ci4", "uci2"):
        d = {"ci2": 2, "ci3": 3, "ci4": 4, "uci2": 2}[f]
        w = abs(lk[1] - lk[0])
        return all(abs(a - b) <= 0.05 * w + 0.5 * 10 ** -d for a, b in zip(lk, rc))
    if f == "p":
        return abs(lk - rc) <= 0.01 + 0.1 * lk
    if isinstance(lk, (int, float)):
        return show(rc, f) == printed or abs(lk - rc) <= 1e-6 * max(1, abs(lk))
    return None


# ------------------------------------------------------------------ claims
def claims(V):
    out = []

    def C(text, ok):
        out.append((text, bool(ok)))

    R = {m: at(f"{REP}#per_model/{m}") for m in M}
    ci_out = lambda s: s["ci95"][0] > 0 or s["ci95"][1] < 0
    C("Report hashes equal the committed V6_RESULTS_HASHES.txt values",
      all(hashlib.sha256((RV / r).read_bytes()).hexdigest() == h for r, h in HASHES.items())
      and all(h in (RV / "v6/results/V6_RESULTS_HASHES.txt").read_text(encoding="utf-8") for h in HASHES.values()))
    C("n = 100 (Qwen2.5-0.5B, 13 arms) and 70 (Llama-3.2-1B, Qwen2.5-1.5B; 10 arms) = design n",
      R[M[0]]["n"] == 100 and R[M[1]]["n"] == R[M[2]]["n"] == at("v6_runs/design.json#n") == 70
      and len(R[M[0]]["arms"]) == 13 and len(R[M[1]]["arms"]) == len(R[M[2]]["arms"]) == 10)
    C("V6-D1 cuts: per-channel key arms and NAT-K-cen absent from the 10-arm models",
      all(a not in R[m]["arms"] for m in M[1:] for a in ("ST-Kc", "ST-Kc@nat", "NAT-K-cen")))
    C("Recomputation: 264 compared, 0 outside tolerance, 0 verdict disagreements",
      at(f"{REC}#comparison/n_compared") == 264 and at(f"{REC}#comparison/n_outside_tolerance") == 0
      and not at(f"{REC}#comparison/verdict_disagreements") and at(f"{REC}#verdicts") == at(f"{REP}#verdicts"))
    led = J("v6_runs/gpu_ledger.jsonl")
    tot = sum(r["seconds"] for r in led) / 3600
    C("Ledger total under the 16 h cap; test blocks are a subset", tot < 16 and sum(r["seconds"] for r in led if r["block"].startswith("test:")) / 3600 < tot)
    C("Replication: H1 < 0, H2 > 0, H3 < 0 with 95% CIs excluding 0 in all three models",
      all(R[m]["L6"]["theta_H1"]["ci95"][1] < 0 and R[m]["L3"]["theta_V"]["ci95"][0] > 0 and R[m]["L7"]["H3_replication"]["ci95"][1] < 0 for m in M))
    C("L3: theta_cm > 0 (CI excludes 0) and theta_dir > 0 (CI excludes 0) in every model ('milder in every model', 'remained much harsher')",
      all(R[m]["L3"]["theta_cm"]["ci95"][0] > 0 and R[m]["L3"]["theta_dir"]["ci95"][0] > 0 for m in M))
    C("L3: theta_cm + theta_dir = theta_V exactly (identity) in every model",
      all(abs(R[m]["L3"]["theta_cm"]["estimate"] + R[m]["L3"]["theta_dir"]["estimate"] - R[m]["L3"]["theta_V"]["estimate"]) < 1e-12 for m in M))
    C("L3 (one order, R1): remainder share 1 - share_cm > 0.5 with CI upper of share_cm < 0.5 in every model (printed as one order only)",
      all(R[m]["L3"]["share_cm"]["ci95"][1] < 0.5 for m in M))
    C("L4: common mode carries most of the head-output energy (share > 0.5, CI lower > 0.5) in every model",
      all(R[m]["L4"]["share_cm"]["ci95"][0] > 0.5 for m in M))
    C("Superseded band: Qwen2.5-1.5B share below 0.2; other two inside [0.2, 0.8]",
      R[M[2]]["L3"]["share_cm"]["share"] < 0.2 and all(0.2 <= R[m]["L3"]["share_cm"]["share"] <= 0.8 for m in M[:2]))
    pil = [float(x) for x in re.findall(r"\$([\d.]+)\$", V["_vn"]["rng.x13GV.min"] + V["_vn"]["rng.x13GV.max"])]
    C("Pilot centering closed 'about a third' (printed range inside [0.25, 0.40]); pilot centered arms re-normed, cm first (FOLLOWUP_X13: 're-norm to their own')",
      0.25 <= min(pil) and max(pil) <= 0.40 and "re-norm to their own" in (RV / "v3/phase8/FOLLOWUP_X13.md").read_text(encoding="utf-8"))
    C("L1 agreement thresholds: V <= ln1.25, K <= ln1.5 in every model",
      all(R[m]["L1"]["V_mod_vs_meas"]["median_abs_ln"] <= math.log(1.25) and R[m]["L1"]["K_lin_vs_meas"]["median_abs_ln"] <= math.log(1.5) for m in M))
    allL1 = [R[m]["L1"][k]["median_abs_ln"] for m in M for k in ("V_mod_vs_meas", "K_lin_vs_meas")]
    C("Abstract L1 range = overall min (V) to overall max (K) over both paths",
      min(allL1) == min(R[m]["L1"]["V_mod_vs_meas"]["median_abs_ln"] for m in M) and max(allL1) == max(R[m]["L1"]["K_lin_vs_meas"]["median_abs_ln"] for m in M))
    C("L5 <= ln 1.5 in every model", all(R[m]["L5"]["median_abs_ln_ratio"] <= math.log(1.5) for m in M))
    C("L2: every CI includes 0; none entirely <= 0 (neither, not disagreement); every Holm p above printed floor",
      all(R[m]["L2"]["ci95"][0] < 0 < R[m]["L2"]["ci95"][1] for m in M)
      and all(at(f"{REP}#holm_p/L2:{m}") > float(V["holm.ltwo.floor"].strip("$")) for m in M))
    C("L6: model-level lin_K log ratio has the sign of H1 in every model; every Spearman CI includes 0",
      all(math.copysign(1, R[m]["L6"]["model_ln_ratio_linK"]) == math.copysign(1, R[m]["L6"]["theta_H1"]["estimate"]) for m in M)
      and all(R[m]["L6"]["spearman"]["ci95"][0] < 0 < R[m]["L6"]["spearman"]["ci95"][1] for m in M))
    C("L8a: CI > 0 in every model and every Holm p <= printed bound",
      all(R[m]["L8a"]["ci95"][0] > 0 for m in M) and all(at(f"{REP}#holm_p/{t}:{m}") <= float(V["holm.sig.max"].strip("$"))
                                                       for t in ("L7iii", "L8a", "L8b", "L9a", "L9b") for m in M))
    C("L8b: estimate negative with CI entirely below 0 in every model (falsified)", all(R[m]["L8b"]["ci95"][1] < 0 for m in M))
    C("L9a and L9b: CI > 0 in every model", all(R[m][t]["ci95"][0] > 0 for t in ("L9a", "L9b") for m in M))
    L7 = {m: R[m]["L7"] for m in M}
    C("L7(i): n*C^V of ST-V <= 2 in every model", all(L7[m]["i_nC_V_ST"] <= 2 for m in M))
    C("L7(iii): rho ratio >= 1 and ST-K - ST-V CI > 0 in every model", all(L7[m]["iii_rho_ratio_K_over_V"] >= 1 and L7[m]["iii_STK_vs_STV"]["ci95"][0] > 0 for m in M))
    inside = lambda s, mg: -mg < s["ci90"][0] and s["ci90"][1] < mg
    C("L7(ii): ST-K@nat - C1-K 95% CI entirely below 0 in all three models", all(L7[m]["ii_STK@nat_vs_C1K"]["ci95"][1] < 0 for m in M))
    C("L7(ii): K 90% CI outside the margin in Qwen2.5-0.5B and Qwen2.5-1.5B, inside in Llama-3.2-1B",
      not inside(L7[M[0]]["ii_STK@nat_vs_C1K"], L7[M[0]]["margin"]) and not inside(L7[M[2]]["ii_STK@nat_vs_C1K"], L7[M[2]]["margin"])
      and inside(L7[M[1]]["ii_STK@nat_vs_C1K"], L7[M[1]]["margin"]))
    C("L7(ii): V equivalence (90% CI inside margin) in all three", all(inside(L7[m]["ii_STV@nat_vs_C1V"], L7[m]["margin"]) for m in M))
    C("L7 margin = the model's v4 locked margin (v4 report)",
      all(abs(L7[m]["margin"] - at(f"v4_runs/v4_report.json#per_model/{m}/margin")) < 1e-12 for m in M)
      if _v4ok() else False)
    vd = at(f"{REP}#verdicts")
    C("Exactly three 'neither' verdicts (L2, L6, L7) = 'three further tests were inconclusive'",
      sorted(k for k, v in vd.items() if (v["verdict"] if isinstance(v, dict) else v) == "neither") == ["L2", "L6", "L7"])
    C("Exactly one disagreement (L8b)", [k for k, v in vd.items() if (v["verdict"] if isinstance(v, dict) else v) == "disagreement"] == ["L8b"])
    d = {m: at(f"{REP}#per_model/{m}/descriptive/per_layer_alignment_median") for m in M}
    C("[descriptive] NAT-V-cen median A_cov within 0.05 of NAT-V in every model ('centering left it unchanged')",
      all(abs(med(d[m]["NAT-V-cen"]["A_cov"]) - med(d[m]["NAT-V"]["A_cov"])) <= 0.05 for m in M))
    C("[descriptive] ST-V@nat median A_cov below C1-V's in every model ('slightly below isotropic')",
      all(med(d[m]["ST-V@nat"]["A_cov"]) < med(d[m]["C1-V"]["A_cov"]) for m in M))
    C("[descriptive] NAT-V median A_cov above C1-V's in every model", all(med(d[m]["NAT-V"]["A_cov"]) > med(d[m]["C1-V"]["A_cov"]) for m in M))
    q = R[M[0]]
    C("[descriptive] per-channel key arm (Qwen2.5-0.5B only) lowers ST-K vs ST-V harm: theta(ST-Kc,ST-V) < theta(ST-K,ST-V)",
      q["descriptive"]["ST-Kc_vs_ST-V"]["estimate"] < q["L7"]["iii_STK_vs_STV"]["estimate"] and "ST-Kc" not in R[M[1]]["arms"])
    C("[descriptive] key centering: theta_cm^K CI includes 0; theta_dir^K within 0.1 of H1 ('essentially all direction')",
      q["L3_K_descriptive"]["theta_cm_K"]["ci95"][0] < 0 < q["L3_K_descriptive"]["theta_cm_K"]["ci95"][1]
      and abs(q["L3_K_descriptive"]["theta_dir_K"]["estimate"] - q["L6"]["theta_H1"]["estimate"]) < 0.1)
    U = at(f"{RB}#units")
    C("R block: 10 units (8 v4 + U1, U2)", len(U) == 10 and "U1" in U and "U2" in U)
    C("R2: positive with CI excluding 0 in all 10 units", all(U[u]["R2"]["ci95"][0] > 0 for u in U))
    C("R1: Qwen2.5-3B, Qwen2.5-7B, Llama-3.1-8B CIs include 0 ('flat'); 6 positive-significant",
      all(U[u]["R1"]["ci95"][0] < 0 < U[u]["R1"]["ci95"][1] for u in ("qwen2.5-3b", "qwen2.5-7b", "llama3.1-8b"))
      and len(at(f"{RB}#verdicts/R1/positive_significant")) == 6)
    C("R4: rho_K > 0.5 and > placebo in all 10 units", all(U[u]["R4"]["rho_K"] > 0.5 and U[u]["R4"]["rho_K"] > U[u]["R4"]["rho_V_placebo"] for u in U))
    C("R verdicts as printed: R1 neither, R3 agreement, R4 agreement",
      [at(f"{RB}#verdicts/{r}/verdict") for r in ("R1", "R3", "R4")] == ["neither", "agreement", "agreement"])
    return out


# ------------------------------------------------------------------ v6b (FOLLOWUP_V6B, lock cba09c444c6de782)
BREP, BREC = "v6b_runs/v6b_report.json", "v6b_runs/v6b_recomputation.json"
BHASHES = {BREP: "a766874757f178caa46af4d28873c39b27b039f25e20330b16d321a14226d98f",
           BREC: "2903910cb3cbb9d6048ca9583697725bbca811236c9118d85626268f5366e55f"}


def check_rec_b(e, printed):
    if not e["src"].startswith(BREP + "#"):
        return None
    rp = BREC + "#" + e["src"].split("#", 1)[1]
    try:
        rc = at(rp)
    except (KeyError, IndexError, TypeError):
        return None
    lk, f = at(e["src"]), e["fmt"]
    if f == "word":
        return lk == rc
    if f in ("ci2", "ci3", "uci2"):
        d = {"ci2": 2, "ci3": 3, "uci2": 2}[f]
        w = abs(lk[1] - lk[0])
        return all(abs(a - b) <= 0.05 * w + 0.5 * 10 ** -d for a, b in zip(lk, rc))
    if isinstance(lk, (int, float)):
        return show(rc, f) == printed or abs(lk - rc) <= 1e-6 * max(1, abs(lk))
    return None


def claims_v6b():
    out = []

    def C(text, ok):
        out.append(("[v6b] " + text, bool(ok)))

    B = {m: at(f"{BREP}#models/{m}") for m in M}
    A6 = {m: at(f"{REP}#per_model/{m}") for m in M}
    P = ("R1", "R2", "U1", "U2")
    sh = lambda m, p: B[m]["paths"][p]["share_dir"]
    est = lambda m, k: B[m][k]["estimate"]
    hl = (RV / "v6b/results/V6B_RESULTS_HASHES.txt").read_text(encoding="utf-8")
    C("Report hashes equal the committed V6B_RESULTS_HASHES.txt values",
      all(hashlib.sha256((RV / r).read_bytes()).hexdigest() == h and h in hl for r, h in BHASHES.items()))
    C("n = 70 in every model = design n, no cut, no record from v6, no failed flag",
      all(B[m]["n"] == 70 and B[m]["records_from_v6"] == 0 and B[m]["records_with_a_failed_flag"] == 0 for m in M)
      and at("v6b_runs/design.json#n") == 70 and at("v6b_runs/design.json#cuts") == [])
    C("G1: 70 problems, all pass, all bitwise, max |d ln m| = 0, in every model",
      all(B[m]["G1"]["problems"] == 70 and B[m]["G1"]["all_pass"] and B[m]["G1"]["all_bitwise"] and B[m]["G1"]["max_abs_dlnm"] == 0 for m in M))
    led = J("v6b_runs/gpu_ledger.jsonl")
    C("Ledger total under the 9 h cap; test blocks a subset", sum(r["seconds"] for r in led) / 3600 < 9
      and 0 < sum(r["seconds"] for r in led if r["block"].startswith("test:")) < sum(r["seconds"] for r in led))
    C("No deviation (DEVIATION_LOG_V6B.md has no entry row)",
      not [l for l in (RV / "v6b/DEVIATION_LOG_V6B.md").read_text(encoding="utf-8").splitlines() if l.startswith("| V6B-")])
    C("Recomputation: 85 compared, 0 outside tolerance, no disagreement, same verdict",
      at(f"{BREC}#compared") == 85 and at(f"{BREC}#outside_tolerance") == 0 and not at(f"{BREC}#disagreements")
      and at(f"{BREC}#verdict") == at(f"{BREP}#verdict/H_star"))
    C("Verdict reject; rejecting paths exactly U2 in Qwen2.5-0.5B and Llama-3.2-1B ('U2 the rejecting path in 2 of the three')",
      at(f"{BREP}#verdict/H_star") == "reject" and sorted(at(f"{BREP}#verdict/rejecting_paths")) == ["llama3.2-1b:U2", "qwen2.5-0.5b:U2"])
    C("U2 s_dir upper bound < 0.5 in Qwen2.5-0.5B and Llama-3.2-1B, not in Qwen2.5-1.5B; no other path's upper bound < 0.5",
      sh(M[0], "U2")["ci95"][1] < 0.5 and sh(M[1], "U2")["ci95"][1] < 0.5 and sh(M[2], "U2")["ci95"][1] >= 0.5
      and all(sh(m, p)["ci95"][1] >= 0.5 for m in M for p in P[:3]))
    C("Each path telescopes: cm + dir = total (per path) and theta_V = theta_U + theta_N, in every model",
      all(abs(B[m]["paths"][p]["cm"]["estimate"] + B[m]["paths"][p]["dir"]["estimate"] - B[m]["paths"][p]["total"]["estimate"]) < 1e-9
          for m in M for p in P)
      and all(abs(est(m, "theta_U") + est(m, "theta_N") - est(m, "theta_V")) < 1e-9 for m in M))
    C("Interactions I_R and I_U: every 95% CI below 0 in every model ('every interval excluding 0', order-dependent)",
      all(B[m][k]["ci95"][1] < 0 for m in M for k in ("interaction_R", "interaction_U"))
      and len(at(f"{BREP}#verdict/order_dependent")) == 6)
    C("Sub-additive: theta(NAT-V,cen)+theta(NAT-V,B) < theta_V and theta(NAT-V,A)+theta(NAT-V,B) < theta_U in every model",
      all(B[m]["paths"]["R1"]["cm"]["estimate"] + B[m]["paths"]["R2"]["dir"]["estimate"] < est(m, "theta_V")
          and B[m]["paths"]["U1"]["cm"]["estimate"] + B[m]["paths"]["U2"]["dir"]["estimate"] < est(m, "theta_U") for m in M))
    C("Common mode more harmful on top of isotropic: theta(B,C1-V) > theta(NAT-V,cen) and theta(B,C) > theta(NAT-V,A) in every model",
      all(B[m]["paths"]["R2"]["cm"]["estimate"] > B[m]["paths"]["R1"]["cm"]["estimate"]
          and B[m]["paths"]["U2"]["cm"]["estimate"] > B[m]["paths"]["U1"]["cm"]["estimate"] for m in M))
    C("R1 gives the common mode its smallest share (largest s_dir) of the four paths in every model (pilot sentence)",
      all(max(P, key=lambda p: sh(m, p)["share"]) == "R1" for m in M))
    C("Common-mode share larger without re-norming (U1) than with it (R1), in every model ('as the protocol had expected')",
      all(sh(m, "U1")["share"] < sh(m, "R1")["share"] for m in M))
    C("R2 common-mode share 'about half' (1 - s_dir inside [0.4, 0.6]) in every model",
      all(0.4 <= 1 - sh(m, "R2")["share"] <= 0.6 for m in M))
    C("Re-norm effect CI > 0 and theta_N CI < 0 in every model; |re-norm + theta_N| < 0.02 ('about as much as the norm term')",
      all(B[m]["renorm_effect"]["ci95"][0] > 0 and B[m]["theta_N"]["ci95"][1] < 0
          and abs(est(m, "renorm_effect") + est(m, "theta_N")) < 0.02 for m in M))
    C("U1 common-mode part theta(NAT-V,NAT-V-cen0) CI > 0 in every model ('made the error milder')",
      all(B[m]["paths"]["U1"]["cm"]["ci95"][0] > 0 for m in M))
    C("Every common-mode and direction part of every path has its 95% CI above 0 in every model ('both interventions reduce the excess')",
      all(B[m]["paths"][p][k]["ci95"][0] > 0 for m in M for p in P for k in ("cm", "dir")))
    C("B realized median norm ratio prints as 1.00 in every model ('1.00 in every model')",
      all(f"{B[m]['descriptive']['B_norm_ratio_median']:.2f}" == "1.00" for m in M))
    C("R1 reproduces v6 L3 exactly where the problems coincide (Llama-3.2-1B, Qwen2.5-1.5B: n = 70 in both; a subset in Qwen2.5-0.5B)",
      all(abs(sh(m, "R1")["share"] - (1 - A6[m]["L3"]["share_cm"]["share"])) < 1e-9 and A6[m]["n"] == 70 for m in M[1:])
      and A6[M[0]]["n"] == 100)
    C("B realized median norm ratio within 0.01 of 1 in every model", all(abs(B[m]["descriptive"]["B_norm_ratio_median"] - 1) < 0.01 for m in M))
    C("Threshold 0.5 in the report and in FOLLOWUP_V6B s5; cap 9 h in FOLLOWUP_V6B",
      at(f"{BREP}#verdict/threshold") == 0.5 and "9 h" in (RV / "v6b/FOLLOWUP_V6B.md").read_text(encoding="utf-8"))
    return out


def _v4ok():
    try:
        for m in M:
            at(f"v4_runs/v4_report.json#per_model/{m}/margin")
        return True
    except Exception:
        return False


def main():
    defs_txt = (PJ / "tables_v6/numbers_v6.tex").read_text(encoding="utf-8")
    defs = dict(re.findall(r"\\csname vsix@([^\\]+)\\endcsname\{(.*)\}$", defs_txt, flags=re.M))
    src = json.loads((PJ / "tables_v6/numbers_v6_sources.json").read_text(encoding="utf-8"))
    files = ["main_jmlr.tex"] + sorted(str(p.relative_to(PJ)) for p in (PJ / "sections_jmlr").glob("*.tex")) + \
            sorted(str(p.relative_to(PJ)) for p in (PJ / "tables_v6").glob("*.tex") if p.name != "numbers_v6.tex")
    used = set()
    for f in files:
        t = re.sub(r"(?<!\\)%.*", "", (PJ / f).read_text(encoding="utf-8"))
        used |= set(re.findall(r"\\vsix\{([^}]+)\}", t))
    undef = sorted(used - set(defs))
    mism, rec_ok, rec_bad, rec_na = [], 0, [], 0
    for k, e in src["macros"].items():
        got = show(derive(e), e["fmt"])
        if got != defs.get(k):
            mism.append((k, defs.get(k), got))
        if "src" in e:
            r = check_rec(k, e, defs.get(k))
            if r is None:
                rec_na += 1
            elif r:
                rec_ok += 1
            else:
                rec_bad.append(k)
    vn_txt = (PJ / "tables_v3/numbers_v3.tex").read_text(encoding="utf-8")
    V = dict(defs)
    V["_vn"] = dict(re.findall(r"\\csname vn@([^\\]+)\\endcsname\{(.*)\}$", vn_txt, flags=re.M))
    cl = claims(V)
    # v6b macros (\vsixb)
    bdefs = dict(re.findall(r"\\csname vsixb@([^\\]+)\\endcsname\{(.*)\}$", (PJ / "tables_v6b/numbers_v6b.tex").read_text(encoding="utf-8"), flags=re.M))
    bsrc = json.loads((PJ / "tables_v6b/numbers_v6b_sources.json").read_text(encoding="utf-8"))
    bfiles = files + sorted(str(p.relative_to(PJ)) for p in (PJ / "tables_v6b").glob("*.tex") if p.name != "numbers_v6b.tex")
    bused = set()
    for f in bfiles:
        bused |= set(re.findall(r"\\vsixb\{([^}]+)\}", re.sub(r"(?<!\\)%.*", "", (PJ / f).read_text(encoding="utf-8"))))
    bundef = sorted(bused - set(bdefs))
    for k, e in bsrc["macros"].items():
        got = show(derive(e), e["fmt"])
        if got != bdefs.get(k):
            mism.append(("v6b:" + k, bdefs.get(k), got))
        if "src" in e:
            r = check_rec_b(e, bdefs.get(k))
            if r is None:
                rec_na += 1
            elif r:
                rec_ok += 1
            else:
                rec_bad.append("v6b:" + k)
    undef += ["v6b:" + u for u in bundef]
    cl += claims_v6b()
    res = {"defined": len(defs), "used": len(used), "undefined": undef, "rederived": len(src["macros"]), "mismatches": mism,
           "v6b_defined": len(bdefs), "v6b_used": len(bused), "v6b_rederived": len(bsrc["macros"]),
           "recomputation_checked": rec_ok, "recomputation_failures": rec_bad, "recomputation_not_held": rec_na,
           "claims": [{"claim": c, "ok": o} for c, o in cl]}
    (HERE / "V6_NUMBER_AUDIT.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    L = ["# Number audit of the v6 and v6b material in main_jmlr.tex (generated by audit/audit_v6_numbers.py)", "",
         f"- v6 macros defined: {len(defs)}; used in the text and tables: {len(used)}; used but undefined: {len(undef)}",
         f"- v6b macros defined: {len(bdefs)}; used: {len(bused)}; used but undefined: {len(bundef)}",
         f"- v6 and v6b macros re-derived from their sources by this script: {len(src['macros'])} + {len(bsrc['macros'])}; mismatches: {len(mism)}",
         f"- v6 and v6b result macros checked against their independent recomputations: {rec_ok}; failures: {len(rec_bad)} "
         f"(source macros not held by the recomputation, e.g. descriptives and R block: {rec_na})",
         f"- prose claims: {sum(o for _, o in cl)}/{len(cl)} hold", "", "## Claims", ""]
    L += [f"- [{'ok' if o else 'FAIL'}] {c}" for c, o in cl]
    if mism or undef or rec_bad:
        L += ["", "## Problems", ""] + [f"- mismatch {m}" for m in mism] + [f"- undefined {u}" for u in undef] + [f"- recomputation {b}" for b in rec_bad]
    (HERE / "V6_NUMBER_AUDIT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:8]))
    bad = [c for c, o in cl if not o]
    if bad:
        print("FAILED CLAIMS:", *bad, sep="\n  ")


if __name__ == "__main__":
    main()
