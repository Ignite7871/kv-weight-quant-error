"""Pilot analysis (CPU only). Implements the report items of PILOT_SPEC.md sections 4-6.

Run: venv/Scripts/python.exe research_v2/analyze_pilot.py [--runs DIR] [--out DIR]
Writes, into --out (default research_v2/pilot_analysis):
  pilot_report.json        all numbers below
  paired_outcomes.csv      per-problem SOLVED/WRONG for every eval arm (rows = problems)
  boundary_audit.md        the 20 seeded records with text around each boundary, for manual audit
  criteria.txt             criteria table
Accuracy is descriptive only; McNemar p-values are descriptive statistics, not claims.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rv2  # noqa: F401
from rv2 import PROJECT_ROOT

TARGET = {"B10": 0.10, "B30": 0.30}
TOL = {"B10": 0.03, "B30": 0.05}
HIST = {  # gate0 arm -> stored per-problem outcomes
    "fp16": [("gsm8k_n280_condition_1_and_3_backfill.json", "condition_1_pure_fp16"), ("gsm8k_n650_all_conditions.json", "condition_1")],
    "nf4": [("gsm8k_n280_condition_2_and_4.json", "condition_2_pure_4bit"), ("gsm8k_n650_all_conditions.json", "condition_2")],
    "hist_lag_entropy_p50": [("gsm8k_n280_condition_1_and_3_backfill.json", "condition_3_entropy_only"), ("gsm8k_n650_all_conditions.json", "condition_3")],
    "hist_lag_entropy_p10": [("gsm8k_multibudget_p10_entropy_only.json", "results")],
    "hist_lag_entropy_p30": [("gsm8k_multibudget_p30_entropy_only.json", "results")],
}
HISTIMPL = {"hist_lag_entropy_p10": "histimpl_lag_entropy_p10", "hist_lag_entropy_p30": "histimpl_lag_entropy_p30",
            "hist_lag_entropy_p50": "histimpl_lag_entropy_p50", "fp16": "histimpl_fp16", "nf4": "histimpl_nf4"}
HIST_SHARE_4BIT_930 = {"hist_lag_entropy_p10": 41.0, "hist_lag_entropy_p30": 79.0, "hist_lag_entropy_p50": 93.5}


def load(p):
    p = Path(p)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else None


def hist_outcomes(arm):
    out = {}
    for fn, key in HIST[arm]:
        for r in json.load(open(PROJECT_ROOT / "results" / fn))[key]:
            out[r["index"]] = r["status"]
    return out


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def share(recs, which):
    num = sum(r["accounting"][f"fp16_steps_{which}"] for r in recs)
    den = sum(r["accounting"][f"routed_steps_{which}"] for r in recs)
    return num / den if den else None


def context(text, i, w=60):
    return text[max(0, i - w):i].replace("\n", "⏎") + " ‖ " + text[i:i + w].replace("\n", "⏎")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(HERE / "pilot_runs"))
    ap.add_argument("--out", default=str(HERE / "pilot_analysis"))
    a = ap.parse_args()
    runs, out = Path(a.runs), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep, crit = {"gate0": {}, "equiv": {}, "eval": {}, "comparisons": [], "calibration": {}}, []

    # ------------------------------------------------------------- 1. gate0: reproduction + implementation identity
    for arm in HIST:
        recs = load(runs / "gate0" / f"{arm}.jsonl")
        if not recs:
            continue
        hist = hist_outcomes(arm)
        dis = [r["index"] for r in recs if r["status"] != hist[r["index"]]]
        impl = load(runs / "gate0" / f"{HISTIMPL[arm]}.jsonl")
        impl_map = {r["index"]: r for r in impl} if impl else {}
        id_match = sum(r["ids"] == impl_map[r["index"]]["ids"] for r in recs if r["index"] in impl_map)
        route_match = sum(r["routes"] == impl_map[r["index"]]["routes"] for r in recs if r["index"] in impl_map)
        classified = []
        for i in dis:
            r = next(x for x in recs if x["index"] == i)
            if i in impl_map:
                cls = "historical-run drift (rv2 == untouched implementation != stored)" if impl_map[i]["ids"] == r["ids"] \
                    else "IMPLEMENTATION DEFECT (rv2 != untouched implementation)"
            else:
                cls = "UNCLASSIFIED: untouched implementation not run on this problem"
            classified.append({"index": i, "stored": hist[i], "rv2": r["status"], "class": cls})
        sh = 1 - share(recs, "full")
        rep["gate0"][arm] = {"n": len(recs), "agree_with_stored": len(recs) - len(dis), "disagreements": classified,
                             "histimpl_n": len(impl_map), "histimpl_token_identical": id_match,
                             "histimpl_routes_identical": route_match,
                             "realized_4bit_share_full_pct": 100 * sh,
                             "realized_4bit_share_answer_pct": 100 * (1 - share(recs, "answer")),
                             "historical_4bit_share_930_pct": HIST_SHARE_4BIT_930.get(arm),
                             "solved": sum(r["status"] == "SOLVED" for r in recs)}
        crit.append((f"G2 gate0 {arm}: exact reproduction of stored outcomes", len(dis) == 0, f"{len(recs)-len(dis)}/{len(recs)}"))
        if impl_map:
            crit.append((f"G2 gate0 {arm}: token-identical to untouched historical implementation", id_match == len(impl_map),
                         f"{id_match}/{len(impl_map)}"))
        if any(c["class"].startswith(("IMPLEMENTATION", "UNCLASSIFIED")) for c in classified):
            crit.append((f"G2 gate0 {arm}: all disagreements classified as non-defects", False, str(classified)))

    # ------------------------------------------------------------- 2. equivalence block
    ref = {k: load(runs / "equiv" / f"{k}.jsonl") for k in ("fp16", "nf4")}
    if ref["fp16"]:
        refm = {k: {r["index"]: r["ids"] for r in v} for k, v in ref.items()}
        for arm, tgt in (("equiv_cascade_always", "fp16"), ("equiv_cascade_never", "nf4"),
                         ("equiv_oracle_always_fp16", "fp16"), ("equiv_oracle_never_fp16", "nf4")):
            recs = load(runs / "equiv" / f"{arm}.jsonl") or []
            same = sum(r["ids"] == refm[tgt][r["index"]] for r in recs)
            rep["equiv"][arm] = {"target": tgt, "identical": same, "n": len(recs)}
            crit.append((f"G3 equiv {arm} == {tgt} (token ids)", same == len(recs) and len(recs) == 5, f"{same}/{len(recs)}"))

    # ------------------------------------------------------------- 3. eval arms
    arms = {}
    for blk in ("eval", "eval_random"):
        for p in sorted((runs / blk).glob("*.jsonl")) if (runs / blk).exists() else []:
            arms[p.stem] = load(p)
    plan = {r["name"]: r for r in json.load(open(HERE / "pilot_random_plan.json"))} if (HERE / "pilot_random_plan.json").exists() else {}
    if (runs / "_plan.json").exists():
        plan = {r["name"]: r for r in json.load(open(runs / "_plan.json"))}
    idx_all = sorted({r["index"] for recs in arms.values() for r in recs})
    for arm, recs in arms.items():
        acc = [r["accounting"] for r in recs]
        n_steps = sum(x["routed_steps_full"] for x in acc)
        wall = sum(r["wall_seconds"] for r in recs)
        rep["eval"][arm] = {
            "kind": recs[0]["kind"], "deployable": recs[0]["deployable"], "n": len(recs),
            "solved": sum(r["status"] == "SOLVED" for r in recs),
            "extraction_failed": sum(r["status"] == "EXTRACTION_FAILED" for r in recs),
            "fp16_share_full": share(recs, "full"), "fp16_share_answer": share(recs, "answer"),
            "fp16_share_decision": share(recs, "decision"),
            "answer_steps_fraction": sum(x["routed_steps_answer"] for x in acc) / n_steps,
            "decision_steps_fraction": sum(x["routed_steps_decision"] for x in acc) / n_steps,
            "per_problem_fp16_share_full_sd": st.pstdev([x["fp16_share_full"] for x in acc]),
            "cost_fpe_per_step": sum(x["cost_total_bytes"] for x in acc) / n_steps / (
                acc[0]["cost_total_bytes"] / acc[0]["cost_fpe"]),
            "cost_kv_and_ops_fraction": sum(x["cost_kv_bytes"] + x["cost_cacheop_bytes"] for x in acc) / sum(x["cost_total_bytes"] for x in acc),
            "nf4_passes": sum(x["nf4_passes"] for x in acc), "fp16_passes": sum(x["fp16_passes"] for x in acc),
            "recomputes": sum(x["recomputes"] for x in acc), "crops": sum(x["crops"] for x in acc),
            "restores": sum(x["restores"] for x in acc),
            "wall_s_total": wall, "wall_s_per_problem": wall / len(recs), "ms_per_step": 1e3 * wall / n_steps,
            "ms_per_nf4_pass": 1e3 * sum(x["time_nf4_pass"] for x in acc) / max(1, sum(x["nf4_passes"] for x in acc)),
            "ms_per_fp16_pass": 1e3 * sum(x["time_fp16_pass"] for x in acc) / max(1, sum(x["fp16_passes"] for x in acc)),
            "cache_ops_time_fraction": sum(x["time_cache_ops"] for x in acc) / sum(x["time_total"] for x in acc),
            "logging_time_fraction": sum(x["time_logging"] for x in acc) / sum(x["time_total"] for x in acc),
            "decide_time_fraction": sum(x["time_decide"] for x in acc) / sum(x["time_total"] for x in acc),
            "prefill_s_mean": st.mean(x["time_prefill"] for x in acc),
            "peak_mem_allocated_GB": max((r["peak_mem_allocated"] or 0) for r in recs) / 1e9,
            "peak_mem_reserved_GB": max((r["peak_mem_reserved"] or 0) for r in recs) / 1e9,
            "boundary_types": {t: sum(r["boundary"]["boundary_type"] == t for r in recs) for t in ("drift_marker", "eos", "cap")},
            "decision_types": {t: sum(r["boundary"]["decision_type"] == t for r in recs) for t in ("hash_number", "no_hash")},
            "offsets_inconsistent": [r["index"] for r in recs if not r["boundary"]["offsets_consistent"]],
            "truncated_extraction_mismatch": [r["index"] for r in recs if not r["truncated_extraction_identical"]],
            "decision_extraction_mismatch": [r["index"] for r in recs if not r["decision_extraction_identical"]],
            "matched_plan": plan.get(arm),
        }
        R = rep["eval"][arm]
        crit.append((f"G4 {arm}: truncated-text extraction identical", not R["truncated_extraction_mismatch"], str(R["truncated_extraction_mismatch"])))
        crit.append((f"G4 {arm}: offsets consistent", not R["offsets_inconsistent"], str(R["offsets_inconsistent"])))
        if arm in plan:
            crit.append((f"B3 {arm}: realized full share within 1.5 pp of p={plan[arm]['p_fp16']:.4f}",
                         abs(R["fp16_share_full"] - plan[arm]["p_fp16"]) <= 0.015, f"{R['fp16_share_full']:.4f}"))

    # ------------------------------------------------------------- 4. calibration consistency
    th = json.load(open(HERE / "pilot_thresholds.json")) if (HERE / "pilot_thresholds.json").exists() else {}
    if (runs / "_thresholds.json").exists():
        th = json.load(open(runs / "_thresholds.json"))
    for name, t in th.items():
        if name.startswith("_") or name not in rep["eval"]:
            continue
        ev = rep["eval"][name]["fp16_share_full"]
        b = name.rsplit("_", 1)[-1]
        rep["calibration"][name] = {**t, "eval_share_full": ev, "abs_error_vs_target": abs(ev - t["target"])}
        crit.append((f"B1 {name}: |eval share - target| <= {TOL.get(b, 0.03):.0%} (informs calibration design)",
                     abs(ev - t["target"]) <= TOL.get(b, 0.03), f"tau={t['tau']:.6g} calib={t['calib_share_on_source']:.4f} eval={ev:.4f}"))

    # ------------------------------------------------------------- 5. paired comparisons (descriptive)
    def paired(x, y):
        X = {r["index"]: r["status"] == "SOLVED" for r in arms[x]}
        Y = {r["index"]: r["status"] == "SOLVED" for r in arms[y]}
        common = sorted(set(X) & set(Y))
        b = sum(X[i] and not Y[i] for i in common); c = sum(Y[i] and not X[i] for i in common)
        return {"a": x, "b_arm": y, "n": len(common), "a_solved": sum(X[i] for i in common), "b_solved": sum(Y[i] for i in common),
                "a_only": b, "b_only": c, "mcnemar_exact_p_descriptive": mcnemar_exact(b, c)}
    for rname, r in plan.items():
        if rname not in arms:
            continue
        for tgt in r["matched_to"]:
            if tgt in arms:
                rep["comparisons"].append({**paired(tgt, rname), "matching_unit": r["unit"],
                                           "target_deployable": arms[tgt][0]["deployable"],
                                           "share_full": (rep["eval"][tgt]["fp16_share_full"], rep["eval"][rname]["fp16_share_full"]),
                                           "share_answer": (rep["eval"][tgt]["fp16_share_answer"], rep["eval"][rname]["fp16_share_answer"]),
                                           "cost_fpe_per_step": (rep["eval"][tgt]["cost_fpe_per_step"], rep["eval"][rname]["cost_fpe_per_step"])})
    for x, y in (("fp16", "nf4"),):
        if x in arms and y in arms:
            rep["comparisons"].append({**paired(x, y), "matching_unit": "anchors", "target_deployable": True})
    # B2: answer-only share gap between targeted policy and its full-matched random control
    for c in rep["comparisons"]:
        if c["matching_unit"] == "full":
            gap = c["share_answer"][0] - c["share_answer"][1]
            c["answer_share_gap"] = gap
            crit.append((f"B2 {c['a']} vs {c['b_arm']}: answer-only FP16 share gap at matched full share (finding)", None, f"{gap:+.4f}"))

    # ------------------------------------------------------------- 6. paired outcome table
    with open(out / "paired_outcomes.csv", "w", newline="") as f:
        w = csv.writer(f)
        names = sorted(arms)
        w.writerow(["index"] + names)
        for i in idx_all:
            row = [i]
            for nme in names:
                st_ = next((r["status"] for r in arms[nme] if r["index"] == i), "")
                row.append({"SOLVED": 1, "WRONG": 0, "EXTRACTION_FAILED": "X"}.get(st_, ""))
            w.writerow(row)

    # ------------------------------------------------------------- 7. boundary audit
    pool = [(nme, r["index"]) for nme, recs in sorted(arms.items()) for r in recs]
    audit = random.Random("rv2-boundary-audit").sample(pool, min(20, len(pool))) if pool else []
    lines = ["# Boundary audit (20 seeded records)", "",
             "Each entry: text around the answer boundary (‖ marks boundary_char; ⏎ = newline) and around the decision boundary.", ""]
    for nme, i in audit:
        r = next(x for x in arms[nme] if x["index"] == i)
        b = r["boundary"]
        lines += [f"## {nme} / problem {i}  [{b['boundary_type']}, decision={b['decision_type']}, status={r['status']}, "
                  f"candidate={r['candidate']}, gt={r['ground_truth']}]",
                  f"- answer boundary @ {b['boundary_char']}: `{context(r['text'], b['boundary_char'])}`",
                  f"- decision boundary @ {b['decision_char']}: `{context(r['text'], b['decision_char'])}`",
                  f"- answer-relevant steps {b['n_answer_tokens']} / decision {b['n_decision_tokens']} / generated {b['n_tokens']}", ""]
    (out / "boundary_audit.md").write_text("\n".join(lines), encoding="utf-8")
    rep["boundary_audit_sample"] = audit

    # ------------------------------------------------------------- 8. full-study projection from measured rates
    def rate(kind):
        xs = [v["wall_s_per_problem"] for v in rep["eval"].values() if v["kind"] == kind]
        return st.mean(xs) if xs else None
    rp, rc, ro = rate("pre"), rate("cascade"), rate("oracle")
    if rp and rc and ro:
        n = 1319
        design = {"anchors (fp16, nf4)": (2, rp), "lag entropy+margin x2 budgets": (4, rp),
                  "random controls (K=3 at B10, K=1 others, ~8 arms)": (8, rp), "cascade x2": (2, rc), "oracles x3": (3, ro)}
        hours = {k: m * r_ * n / 3600 for k, (m, r_) in design.items()}
        rep["projection_full_study_hours"] = {"per_component": hours, "total": sum(hours.values()),
                                              "rates_s_per_problem": {"pre": rp, "cascade": rc, "oracle": ro},
                                              "note": "n=1319 GSM8K test, both budgets; excludes calibration runs and model loads"}

    (out / "pilot_report.json").write_text(json.dumps(rep, indent=1, default=str))
    wcol = max((len(c[0]) for c in crit), default=10)
    txt = "\n".join(f"{n_:<{wcol}}  {'PASS' if ok else ('INFO' if ok is None else 'FAIL')}  {v}" for n_, ok, v in crit)
    (out / "criteria.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
