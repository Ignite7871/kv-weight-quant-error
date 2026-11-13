"""Number audit of the v5 material in main_v4.tex (Section 6, Appendix C and the v5 sentences elsewhere). CPU only; read-only.
Shares no code with make_v5.py. Literal numbers outside macros are classified by audit_v4_numbers.py, which scans every section.

1. Every \\vsv{key} used in main_v4.tex, sections_v4/ and tables_v5/ is defined.
2. Every v5 macro is RE-DERIVED from its declared source with this script's own reader, operations and formatter; the printed
   string must be identical.
3. Every v5 per-unit result macro and every family result, reading and A-X classification is re-checked against the INDEPENDENT
   recomputation (v5_runs/v5_recomputation.json): estimates identical at the printed precision; CI endpoints within 5 % of the CI
   width plus half a printed unit; p within 0.01 + 10 %; L^P within 0.02 + 5 % of the CI width; words identical.
4. Prose claims of the v5 text (and of the v5 sentences in the abstract, introduction, discussion, limitations and conclusion)
   are checked against the locked report and the run records.
5. The abstract has at most 300 words (macros expanded, as printed) and every highlight in v4/MANUSCRIPT_PLAN_V4.md section 3 has
   at most 85 characters.
Writes audit/V5_NUMBER_AUDIT.md and audit/V5_NUMBER_AUDIT.json.
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
PJ = HERE.parent
RV = PJ.parent
SRCS = ["main_v4.tex"] + sorted(str(p.relative_to(PJ)) for p in (PJ / "sections_v4").glob("*.tex"))
TABS = sorted(str(p.relative_to(PJ)) for p in (PJ / "tables_v5").glob("*.tex") if p.name != "numbers_v5.tex")
REP = "v5_runs/v5_report.json"
_files = {}


def text(rel):
    return (PJ / rel).read_text(encoding="utf-8")


def uncomment(t):
    return re.sub(r"(?<!\\)%.*", "", t)


def defs(rel, prefix):
    return dict(re.findall(r"\\csname " + prefix + r"@([^\\]+)\\endcsname\{(.*)\}$", text(rel), flags=re.M))


def jl(rel):
    if rel not in _files:
        raw = (RV / rel).read_text(encoding="utf-8")
        _files[rel] = [json.loads(x) for x in raw.splitlines() if x.strip()] if rel.endswith(".jsonl") else json.loads(raw)
    return _files[rel]


def at(ptr):
    f, path = ptr.split("#", 1)
    node = jl(f)
    for part in filter(None, path.split("/")):
        part = part.replace("~1", "/")
        node = node[int(part)] if isinstance(node, list) else node[part]
    return node


def sg(x, d):
    s = f"{abs(x):.{d}f}"
    return ("-" if x < 0 and float(s) != 0 else "+") + s


def show(x, f):
    return {
        "s3": lambda: f"${sg(x, 3)}$", "s2": lambda: f"${sg(x, 2)}$", "u3": lambda: f"${x:.3f}$", "u2": lambda: f"${x:.2f}$",
        "u1": lambda: f"${x:.1f}$", "int": lambda: f"${round(x)}$", "intc": lambda: "$" + f"{round(x):,}".replace(",", "{,}") + "$", "ci3": lambda: f"$[{sg(x[0], 3)}, {sg(x[1], 3)}]$",
        "uci3": lambda: f"$[{x[0]:.3f}, {x[1]:.3f}]$", "p": lambda: "$<0.001$" if x < 0.001 else f"${x:.3f}$",
        "usd": lambda: "\\$" + f"{x:.2f}", "pct0": lambda: f"${100 * x:.0f}$\\%", "bn": lambda: f"${x / 1e9:.2f}$",
        "word": lambda: str(x), "bool": lambda: "yes" if x else "no"}[f]()


def hours(rel, since=None):
    return sum(r["seconds"] for r in jl(rel) if since is None or r["start_utc"] >= since) / 3600


def last_billing(rel):
    return [r for r in jl(rel) if r.get("kind") == "billing"][-1]["usd"]


def recompute(e):
    if "src" in e:
        return at(e["src"])
    op, ins = e["op"], e["inputs"]
    if op == "ledger_hours":
        return hours(ins[0])
    if op == "ledger_hours_since":
        return hours(ins[0], ins[1])
    if op == "ledger_hours_sum":
        return sum(hours(i) for i in ins)
    if op == "last_billing":
        return last_billing(ins[0])
    if op == "last_billing_plus":
        return last_billing(ins[0]) + at(ins[1])
    if op == "count_rows":
        return len(jl(ins[0]))
    if op == "count_true_field":
        f, field = ins[0].split("#*/")
        return sum(1 for r in jl(f) if r[field] is True)
    v = [at(p) for p in ins]
    if op == "floor3_min":
        return int(min(v) * 1000) / 1000 if min(v) >= 0 else -int(-min(v) * 1000 + 0.999999999) / 1000
    return {"len": lambda: len(v[0]), "absmax": lambda: max(abs(a) for a in v), "min": lambda: min(v), "max": lambda: max(v),
            "div": lambda: v[0] / v[1]}[op]()


# ---------------------------------------------------------------- claims
def claims(vsv):
    r = jl(REP)
    pu = r["per_unit"]
    au = r["all_units"]
    ax = au["A-X"]
    est = lambda u, c: pu[u]["contrasts"][c]["estimate"]
    ci = lambda u, c: pu[u]["contrasts"][c]["ci95"]
    lo = lambda u, P: pu[u]["H5"][P]["ci95"][0]
    num = lambda k: float(re.findall(r"[+-]?\d+\.\d+", vsv[k])[0])
    anc = jl("v5_runs/anchor.json")
    rec = jl("v5_runs/v5_recomputation.json")
    v4 = jl("v4_runs/v4_report.json")
    man = jl("v5/V5_MODEL_MANIFEST.json")["models"]
    C = []
    add = lambda t, ok: C.append({"claim": t, "ok": bool(ok)})
    add("All seven units complete, n = 100, no BF16 flag, design reduction 'none', no abstract-level disagreement",
        all(v == "complete" for v in r["status"].values()) and len(r["status"]) == 7 and r["flagged_bf16"] == []
        and all(pu[u]["n_problems"] == 100 for u in pu) and r["design"]["reduction"] == "none" and r["abstract_level"]["disagreements"] == {})
    add("Without-flagged families identical to all-units", r["without_flagged"] == r["all_units"])
    add("Anchor passed; every record, NAT slot set and history bitwise equal; every anchor delta exactly 0",
        anc["pass"] and anc["reported_bitwise"]["share"] == 1.0 and anc["reported_bitwise"]["nat_slots_equal"] == 100
        and anc["reported_bitwise"]["histories_equal"] == 100
        and all(c["delta"]["estimate"] == 0 and c["delta"]["ci95"] == [0.0, 0.0] for c in anc["contrasts"].values()))
    add("U0 reproduces the v4 Qwen2.5-0.5B estimates exactly (H1, H2, H3, NAT-C1)",
        all(est("U0", c) == v4["per_model"]["qwen2.5-0.5b"]["contrasts"][c]["estimate"] for c in ("H1", "H2", "H3", "NAT-C1")))
    add("U3 and U4 share U0's model; U3 differs only in task, U4 only in quantizer",
        pu["U0"]["model"] == pu["U3"]["model"] == pu["U4"]["model"] and pu["U3"]["quantizer"] == pu["U0"]["quantizer"]
        and pu["U3"]["task"] != pu["U0"]["task"] and pu["U4"]["task"] == pu["U0"]["task"] and pu["U4"]["quantizer"] != pu["U0"]["quantizer"])
    add("B and D run in Qwen2.5-0.5B and Mistral-7B-v0.3 (0.5B and 7B)",
        {pu[u]["model"] for u in ("U3", "U4")} == {"qwen2.5-0.5b"} and {pu[u]["model"] for u in ("U5", "U6")} == {"mistral-7b-v0.3"})
    add("Recomputation: 230 quantities, none outside tolerance, no unresolved verdict",
        rec["n_compared"] == 230 and rec["n_disagree"] == 0 and rec["unresolved_verdicts"] == [] and all(x["agree"] for x in rec["rows"]))
    add("A-H1, A-H2, A-H3, A-H5 each supported in every model, every Holm p < 0.001",
        all(au[f]["result"] == "supported in every model" and max(au[f]["holm_p"].values()) < 0.001 for f in ("A-H1", "A-H2", "A-H3", "A-H5")))
    add("A: H1 < 0 and H2 > 0 and H3 < 0 with CIs excluding 0 in U1 and U2",
        all(ci(u, "H1")[1] < 0 and ci(u, "H2")[0] > 0 and ci(u, "H3")[1] < 0 for u in ("U1", "U2")))
    add("A: every H5 lower bound at or above the printed rng.ahfivelo.min and above ln 4",
        all(lo(u, P) >= num("rng.ahfivelo.min") and lo(u, P) > 1.3863 for u in ("U1", "U2") for P in "KV"))
    add("A-X: H1 size pattern in both models; NAT-C1 size pattern in U1, unresolved in U2",
        ax["H1"]["units"]["U1"]["classification"] == ax["H1"]["units"]["U2"]["classification"] == "size pattern"
        and ax["NAT-C1"]["units"]["U1"]["classification"] == "size pattern" and ax["NAT-C1"]["units"]["U2"]["classification"] == "unresolved")
    u2 = ax["NAT-C1"]["units"]["U2"]
    add("A-X: SmolLM2's NAT-C1 interval contains its boundary b", u2["ci95"][0] <= u2["boundary"] <= u2["ci95"][1])
    add("A-X: every size-pattern classification has its CI entirely on the size side and Holm p < 0.05",
        all((x["ci95"][1] < x["boundary"]) == (x["size_pred"] < x["boundary"]) and (x["ci95"][0] > x["boundary"] or x["ci95"][1] < x["boundary"])
            and x["holm_p"] < 0.05 for c in ("H1", "NAT-C1") for x in ax[c]["units"].values() if x["classification"] == "size pattern"))
    add("A-X joint reading is exactly 'neither factor alone accounts for it in these checkpoints'",
        ax["joint_reading_NAT-C1"]["reading"] == "neither factor alone accounts for it in these checkpoints")
    u1 = ax["NAT-C1"]["units"]["U1"]
    add("[exploratory] Qwen3 NAT-C1: whole interval below the size prediction, which is below the head-dimension prediction",
        u1["ci95"][1] < u1["size_pred"] < u1["hd_pred"])
    add("SmolLM2's references are all from Qwen2.5 / Llama-3.2 models (no SmolLM2 in v4)",
        all(m.startswith(("qwen2.5", "llama3")) for c in ("H1", "NAT-C1") for side in ("size", "hd")
            for m in ax[c]["units"]["U2"]["sources"][side]))
    q3, sm = man["qwen3-0.6b-base"], man["smollm2-1.7b"]
    add("Qwen3: one key-norm tensor per layer (QK-norm); head dim 128; under 1B params",
        q3["qk_norm_tensors"]["k_norm"] == q3["geometry"]["layers"] and q3["geometry"]["head_dim"] == 128 and q3["params"]["total"] < 1e9)
    add("SmolLM2: KV heads = query heads (no GQA); head dim 64; 49,152-token vocabulary; no QK-norm",
        sm["geometry"]["kv_heads"] == sm["geometry"]["q_heads"] and sm["geometry"]["head_dim"] == 64 and sm["geometry"]["vocab"] == 49152
        and sm["qk_norm_tensors"]["k_norm"] == 0)
    add("v4 small models had head dim 64 and the larger ones 128 (the confound A addresses)",
        [m for m, d in v4["per_model"].items() if d["head_dim"] == 64] == ["qwen2.5-0.5b", "llama3.2-1b"])
    add("B-H2, B-H3, B-H5 and D-H2, D-H5 each supported in every model",
        all(au[f]["result"] == "supported in every model" for f in ("B-H2", "B-H3", "B-H5", "D-H2", "D-H5")))
    add("B and D: every H5 lower bound at or above the printed rng.bdhfivelo.min and above ln 4",
        all(lo(u, P) >= num("rng.bdhfivelo.min") and lo(u, P) > 1.3863 for u in ("U3", "U4", "U5", "U6") for P in "KV"))
    add("[descriptive] Mistral H1 under HQQ: CI below 0; v4 NF4 Mistral H1 inconclusive",
        ci("U6", "H1")[1] < 0 and v4["all_models"]["F1 H1"]["readings"]["mistral-7b-v0.3"] == "inconclusive")
    add("[descriptive] H3 negative (CI < 0) in both WikiText-2 units", all(ci(u, "H3")[1] < 0 for u in ("U3", "U5")))
    add("[descriptive] Mistral H1 small on WikiText-2 and in v4 (|estimate| < 0.15)",
        abs(est("U5", "H1")) < 0.15 and abs(v4["per_model"]["mistral-7b-v0.3"]["contrasts"]["H1"]["estimate"]) < 0.15)
    add("H1 and NAT-C1 under HQQ, and H1, H3, NAT-C1 on WikiText-2, carry no locked family",
        not any(f in au for f in ("B-H1", "B-NAT-C1", "D-H1", "D-H3", "D-NAT-C1")))
    d1 = jl("v5_runs/U6/d1_stop.json")
    add("D1 not triggered: U6 ran under the $14 stop", d1["triggered"] is False and d1["stop_usd"] == 14.0 and pu["U6"]["n_problems"] == 100)
    dev2 = jl("v5/deviations/V5-D2.json")["reason"]
    add("D2 factor printed (1.38) equals the deviation record's factor", "x 1.38" in dev2 and vsv["dtwo.factor"] == "$1.38$")
    refs = jl("v5_runs/U1/d3_reference_check.jsonl")
    add("D3: calibration recomputed equal to the locked hash; 13 of 13 surviving problems bitwise equal",
        jl("v5_runs/U1/d3_calibration_check.json")["equal"] is True and len(refs) == 13 and all(x["bitwise_equal"] for x in refs))
    add("D3 declares gm_D2.json lost", "gm_D2.json" in jl("v5/deviations/V5-D3.json").get("lost_appendix_files", {}))
    add("v5 billed cost under the $15 ceiling", last_billing("v5_runs/cost_ledger.jsonl") < 15)
    return C


def abstract_words(vf, vsv):
    t = uncomment(text("sections_v4/00_abstract.tex"))
    t = re.sub(r"\\vf\{([^}]+)\}", lambda m: vf[m.group(1)], t)
    t = re.sub(r"\\vsv\{([^}]+)\}", lambda m: vsv[m.group(1)], t)
    t = re.sub(r"\{\\bf ([^}]*)\}", r"\1", t)
    t = re.sub(r"\\[a-zA-Z]+", " ", t)
    t = re.sub(r"[$~{}\\]", " ", t)
    return len(t.split())


def highlights():
    plan = (RV / "v4/MANUSCRIPT_PLAN_V4.md").read_text(encoding="utf-8")
    sec = plan.split("## 3. Highlights", 1)[1].split("\n## ", 1)[0]
    return [h for h in re.findall(r"^\d+\. (.+)$", sec, flags=re.M)]


def main():
    vf = defs("tables_v4/numbers_v4.tex", "vf")
    vsv = defs("tables_v5/numbers_v5.tex", "vsv")
    srcs = json.loads(text("tables_v5/numbers_v5_sources.json"))["macros"]
    used = set()
    for rel in SRCS + TABS:
        used |= set(re.findall(r"\\vsv\{([^}]+)\}", uncomment(text(rel))))
    undef = sorted(used - set(vsv))
    der = []
    for k, e in srcs.items():
        try:
            pr = show(recompute(e), e["fmt"])
            der.append({"key": k, "ok": pr == vsv.get(k), "macro": vsv.get(k), "rederived": pr})
        except Exception as ex:  # noqa: BLE001
            der.append({"key": k, "ok": False, "macro": vsv.get(k), "rederived": f"ERROR {ex!r}"})
    rows = {x["quantity"]: x["recomputed"] for x in jl("v5_runs/v5_recomputation.json")["rows"]}
    CN = {"H1": "hone", "H2": "htwo", "H3": "hthree", "NAT-C1": "natc"}
    rc = []
    nums = lambda s: [float(x) for x in re.findall(r"[+-]?\d+\.\d+", s)]
    for u in ("U0", "U1", "U2", "U3", "U4", "U5", "U6"):
        s = u.lower()
        for c, cn in CN.items():
            e, (lo_, hi_), p, v = rows[f"{u} {c} estimate"], rows[f"{u} {c} ci95"], rows[f"{u} {c} p"], rows[f"{u} {c} verdict"]
            w = hi_ - lo_
            m = nums(vsv[f"{cn}.{s}.ci"])
            rc.append({"key": f"{cn}.{s}.est", "ok": show(e, "s3") == vsv[f"{cn}.{s}.est"]})
            rc.append({"key": f"{cn}.{s}.ci", "ok": abs(m[0] - lo_) <= 0.05 * w + 0.0005 and abs(m[1] - hi_) <= 0.05 * w + 0.0005})
            pl = at(f"{REP}#per_unit/{u}/contrasts/{c}/p")
            rc.append({"key": f"{cn}.{s}.p", "ok": (pl < 0.001 and p < 0.001) or abs(pl - p) <= 0.01 + 0.1 * pl})
            rc.append({"key": f"{cn}.{s}.verdict", "ok": v == vsv[f"{cn}.{s}.verdict"]})
        for P in "KV":
            L, Lci = rows[f"{u} L^{P}"], rows[f"{u} L^{P} ci95"]
            w = Lci[1] - Lci[0]
            rc.append({"key": f"hfive{P}.{s}.est", "ok": abs(nums(vsv[f"hfive{P}.{s}.est"])[0] - L) <= 0.02 + 0.05 * w})
            rc.append({"key": f"hfive{P}.{s}.ci", "ok": all(abs(a - b) <= 0.02 + 0.1 * w for a, b in zip(nums(vsv[f"hfive{P}.{s}.ci"]), Lci))})
    for fam in ("A-H1", "A-H2", "A-H3", "A-H5", "B-H2", "B-H3", "B-H5", "D-H2", "D-H5"):
        k = fam.lower().replace("-", "")
        rc.append({"key": f"fam.{k}", "ok": rows[f"all_units {fam} result"] == vsv[f"fam.{k}"]})
        rc.append({"key": f"famwf.{k}", "ok": rows[f"without_flagged {fam} result"] == vsv[f"famwf.{k}"]})
        for unit, word in rows[f"all_units {fam} readings"].items():
            rc.append({"key": f"read.{k}.{unit.lower().replace('/', '')}", "ok": word == vsv[f"read.{k}.{unit.lower().replace('/', '')}"]})
    for c, cn in (("NAT-C1", "natc"), ("H1", "hone")):
        for u in ("U1", "U2"):
            s = u.lower()
            rc.append({"key": f"ax.{cn}.{s}.class", "ok": rows[f"all_units A-X {c} {u} classification"] == vsv[f"ax.{cn}.{s}.class"]})
            rc.append({"key": f"ax.{cn}.{s}.b", "ok": show(rows[f"all_units A-X {c} {u} boundary"], "s3") == vsv[f"ax.{cn}.{s}.b"]})
            pl, pr = at(f"{REP}#all_units/A-X/{c}/units/{u}/p_one_sided"), rows[f"all_units A-X {c} {u} p"]
            rc.append({"key": f"ax.{cn}.{s}.p_one_sided", "ok": abs(pl - pr) <= 0.01 + 0.1 * pl})
    rc.append({"key": "ax.joint", "ok": rows["all_units A-X joint reading"] == vsv["ax.joint"]})
    rc.append({"key": "ax.jointwf", "ok": rows["without_flagged A-X joint reading"] == vsv["ax.jointwf"]})
    C = claims(vsv)
    nw = abstract_words(vf, vsv)
    H = highlights()
    C.append({"claim": f"Abstract has at most 300 words (counted: {nw})", "ok": nw <= 300})
    C.append({"claim": "Every highlight has at most 85 characters (" + ", ".join(str(len(h)) for h in H) + ")",
              "ok": len(H) >= 1 and all(len(h) <= 85 for h in H)})
    res = {"vsv_defined": len(vsv), "vsv_used": len(used), "vsv_undefined": undef, "rederived_n": len(der),
           "rederived_fail": [d for d in der if not d["ok"]], "recomp_n": len(rc), "recomp_fail": [x for x in rc if not x["ok"]],
           "claims": C, "abstract_words": nw, "highlights": [{"text": h, "chars": len(h)} for h in H]}
    (HERE / "V5_NUMBER_AUDIT.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    L = ["# Number audit of the v5 material in main_v4.tex (generated by audit/audit_v5_numbers.py)", "",
         f"- v5 macros defined: {len(vsv)}; used in the text and tables: {len(used)}; used but undefined: {len(undef)}",
         f"- v5 macros re-derived from their sources by this script: {len(der)}; mismatches: {len(res['rederived_fail'])}",
         f"- v5 result macros checked against the independent recomputation: {len(rc)}; failures: {len(res['recomp_fail'])}",
         f"- prose claims and length checks: {sum(c['ok'] for c in C)}/{len(C)} hold",
         f"- abstract words: {nw}; highlights: {len(H)}, longest {max(len(h) for h in H)} characters", "", "## Claims", ""]
    L += [f"- [{'ok' if c['ok'] else 'FAIL'}] {c['claim']}" for c in C]
    for title, items in (("Re-derivation mismatches", res["rederived_fail"]), ("Recomputation failures", res["recomp_fail"]),
                         ("Undefined macros", [undef] if undef else [])):
        L += ["", f"## {title}", ""] + ([f"- {json.dumps(x)}" for x in items] or ["none"])
    (HERE / "V5_NUMBER_AUDIT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:7]))
    for c in C:
        if not c["ok"]:
            print("FAIL:", c["claim"])


if __name__ == "__main__":
    main()
