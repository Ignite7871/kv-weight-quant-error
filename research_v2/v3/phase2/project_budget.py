"""v3 Phase 2: GPU-hour projection and 2x2 power (MDE) from saved E1 (Qwen2.5-0.5B) and E1L (Llama-3.2-1B-Instruct) records. CPU only.
Reads records and ledgers; imports nothing from the analysis code. Output: budget_projection.json next to this file.

Basis
  * per arm-problem wall time at S<=64 teacher-forced steps (two passes per step): mean of `wall_seconds` of the E1/E1L Tier A arm records;
  * per-block overhead (reference logits, in-session history generation, model load) = ledger block seconds - sum of arm wall seconds;
  * calibration and gate blocks from the ledgers.
The v3 instruct/base sibling of each family is assumed to cost the same as the measured sibling (identical architecture and geometry).

  venv/Scripts/python.exe research_v2/v3/phase2/project_budget.py
"""

import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RV = HERE.parents[1]

V3_TEST_ARMS = ["NAT", "C1", "C2", "QA", "QO", "CV", "NAT-K", "NAT-V", "NAT-K*", "NAT-V*", "C1-K", "C1-V", "LOC", "INH"]
V3_CALIB_ARMS = ["NAT", "C1_a0.5", "C1_a1", "C1_a2"]     # see PROTOCOL_V3.md section 6
N_TEST, N_CALIB = 200, 100
PRED_OVERHEAD = 0.15        # assumed extra cost of the per-step predictor computation; replaced by the G6 measurement
GATE_HOURS_PER_MODEL = 0.35  # E1L measured gates + GPU mutants (see ledger); v3 has more arms/mutants
MARGIN = 0.05               # 5 % safety margin as in E1L G6


def read(p):
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def ledger(root):
    out = {}
    for r in read(RV / root / "gpu_ledger.jsonl"):
        out.setdefault(r["block"], 0.0)
        out[r["block"]] += r["seconds"]
    return out


def arm_seconds(root, sub, files):
    per = {}
    for f in files:
        rs = read(RV / root / sub / f)
        per[f] = {"n": len(rs), "mean_s": float(np.mean([r["wall_seconds"] for r in rs])), "sum_s": float(np.sum([r["wall_seconds"] for r in rs])),
                  "mean_steps": float(np.mean([r["S"] for r in rs]))}
    return per


def theta(root):
    nat = {r["index"]: r for r in read(RV / root / "tierA" / "NAT.jsonl")}
    c1 = {r["index"]: r for r in read(RV / root / "tierA" / "C1_a1.0.jsonl")}
    return {i: math.log(max(np.mean(nat[i]["js"]), 1e-9)) - math.log(max(np.mean(c1[i]["js"]), 1e-9)) for i in nat}


def main():
    res = {"assumptions": {"pred_overhead": PRED_OVERHEAD, "gate_hours_per_model": GATE_HOURS_PER_MODEL, "margin": MARGIN,
                           "n_test": N_TEST, "n_calib": N_CALIB, "test_arms": V3_TEST_ARMS, "calib_arms": V3_CALIB_ARMS}}
    fam = {}
    for name, root, tierA_files in (("qwen", "e1_runs", ["NAT.jsonl", "C1_a0.5.jsonl", "C1_a1.0.jsonl", "C1_a2.0.jsonl", "C2_a1.0.jsonl", "C1star.jsonl"]),
                                    ("llama", "e1l_runs", ["NAT.jsonl", "C1_a0.5.jsonl", "C1_a1.0.jsonl", "C1_a2.0.jsonl", "C2_a1.0.jsonl", "C1star.jsonl"])):
        L = ledger(root)
        A = arm_seconds(root, "tierA", tierA_files)
        C = arm_seconds(root, "calib", sorted(p.name for p in (RV / root / "calib").glob("*.jsonl")))
        arm_s = float(np.mean([v["mean_s"] for v in A.values()]))                  # per arm-problem, test
        tierA_over = L["tierA"] - sum(v["sum_s"] for v in A.values())             # refs + histories + load
        calib_arm_s = float(np.mean([v["mean_s"] for v in C.values()]))
        calib_over = L["calibrate"] - sum(v["sum_s"] for v in C.values())
        per_model = {
            "test_arms_h": len(V3_TEST_ARMS) * N_TEST * arm_s * (1 + PRED_OVERHEAD) / 3600,
            "test_overhead_h": tierA_over / 3600,
            "calib_h": (len(V3_CALIB_ARMS) * N_CALIB * calib_arm_s * (1 + PRED_OVERHEAD) + calib_over) / 3600,
            "gates_h": GATE_HOURS_PER_MODEL,
        }
        per_model["total_h"] = sum(per_model.values()) * (1 + MARGIN)
        per_arm_h = N_TEST * arm_s * (1 + PRED_OVERHEAD) / 3600
        fam[name] = {"measured": {"ledger_s": L, "tierA_arm_s": A, "calib_arm_s": C, "arm_problem_s_mean": arm_s,
                                  "tierA_overhead_s": tierA_over, "calib_overhead_s": calib_over},
                     "per_model_projection_h": per_model, "per_test_arm_h": per_arm_h}
    res["families"] = fam
    models = {"Qwen2.5-0.5B": "qwen", "Qwen2.5-0.5B-Instruct": "qwen", "Llama-3.2-1B": "llama", "Llama-3.2-1B-Instruct": "llama"}
    res["per_model_total_h"] = {m: fam[f]["per_model_projection_h"]["total_h"] for m, f in models.items()}
    res["grand_total_h"] = sum(res["per_model_total_h"].values())
    res["saving_if_drop_arm_all_models_h"] = sum(fam[f]["per_test_arm_h"] for f in models.values()) * (1 + MARGIN)

    # 2x2 power: per-problem theta = log m^NAT - log m^C1 from E1 (Qwen base) and E1L (Llama instruct), paired by problem
    tq, tl = theta("e1_runs"), theta("e1l_runs")
    idx = sorted(set(tq) & set(tl))
    q, l = np.array([tq[i] for i in idx]), np.array([tl[i] for i in idx])
    sq, sl, r = float(q.std(ddof=1)), float(l.std(ddof=1)), float(np.corrcoef(q, l)[0, 1])
    z = 1.959963984540054 + 0.8416212335729143
    # a main effect 0.5*[(a+b)-(c+d)] with per-model SD s and between-model correlations; assume SD = mean(sq, sl) for all four,
    # within-family correlation r_w and between-family r_b both set to the measured Qwen-Llama r (no within-family data exist)
    s = 0.5 * (sq + sl)
    # contrasts over a = Qwen-base, b = Qwen-instruct, c = Llama-base, d = Llama-instruct, paired by problem, common SD s;
    # r_b = between-family correlation (measured, Qwen-base vs Llama-instruct), r_w = within-family correlation (unknown -> grid)
    #   family  = 0.5[(c+d)-(a+b)]  Var = s^2 (1 + r_w - 2 r_b)
    #   tuning  = 0.5[(b+d)-(a+c)]  Var = s^2 (1 - r_w)
    #   interac = 0.5[(a-b)-(c-d)]  Var = s^2 (1 - r_w)
    grid = {}
    for rw in (r, 0.25, 0.5, 0.75, 0.9):
        grid[f"r_w={rw:.2f}"] = {"family": z * s * math.sqrt(max(1 + rw - 2 * r, 0) / N_TEST),
                                 "tuning": z * s * math.sqrt(max(1 - rw, 0) / N_TEST),
                                 "interaction": z * s * math.sqrt(max(1 - rw, 0) / N_TEST)}
    res["power_2x2"] = {"n_problems_paired": len(idx), "sd_theta_qwen": sq, "sd_theta_llama": sl, "r_between_family": r,
                        "theta_qwen_mean": float(q.mean()), "theta_llama_mean": float(l.mean()),
                        "observed_family_gap_E1_vs_E1L_log": float(l.mean() - q.mean()),
                        "mde_log_80pct_two_sided_by_r_within": grid,
                        "note": "common SD = mean of the two measured SDs; r_w unknown (no within-family pair has been run)"}
    (HERE / "budget_projection.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("per_model_total_h", "grand_total_h", "saving_if_drop_arm_all_models_h", "power_2x2")}, indent=1))
    for f in fam:
        print(f, json.dumps(fam[f]["per_model_projection_h"]), "arm-problem s", round(fam[f]["measured"]["arm_problem_s_mean"], 2),
              "calib arm s", round(float(np.mean([v["mean_s"] for v in fam[f]["measured"]["calib_arm_s"].values()])), 2))


if __name__ == "__main__":
    main()
