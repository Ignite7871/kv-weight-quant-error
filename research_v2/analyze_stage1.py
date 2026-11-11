"""Stage 1 analysis (CPU only). Implements STAGE1_SPEC.md section 7 exactly; written and tested before any data exist.

Usage: venv/Scripts/python.exe research_v2/analyze_stage1.py [--runs stage1_runs/eval] [--out stage1_analysis]

Outputs: stage1_report.json, paired_outcomes.csv, divergence_survival.csv, survival.png (+ printed summary).
Only problems completed for ALL five arms are analysed. Accuracy counts only status == SOLVED.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ARMS = ["fp16", "nf4", "cache_only", "token_only", "oracle_flip"]
B_BOOT, B_PERM, SEED = 10_000, 10_000, 20261023
TOST_BOUND = 0.03
BINS = [(1, 20), (21, 60), (61, 10_000)]
FAMILY = [  # name, coefficients over arms (sum of coef * Y)
    ("cache_only - fp16", {"cache_only": 1, "fp16": -1}),
    ("token_only - fp16", {"token_only": 1, "fp16": -1}),
    ("interaction (nf4 - cache_only - token_only + fp16)", {"nf4": 1, "cache_only": -1, "token_only": -1, "fp16": 1}),
    ("cache_only - nf4", {"cache_only": 1, "nf4": -1}),
    ("oracle_flip - cache_only", {"oracle_flip": 1, "cache_only": -1}),
]
SECONDARY = [("nf4 - fp16 (total gap)", {"nf4": 1, "fp16": -1}), ("token_only - nf4", {"token_only": 1, "nf4": -1}),
             ("oracle_flip - fp16", {"oracle_flip": 1, "fp16": -1}), ("oracle_flip - nf4", {"oracle_flip": 1, "nf4": -1})]


# ----------------------------------------------------------------------------- loading
def load_arm(d: Path, arm: str) -> dict:
    p = d / f"{arm}.jsonl"
    if not p.exists():
        return {}
    return {r["index"]: r for r in (json.loads(l) for l in p.read_text().splitlines() if l.strip())}


def load_all(d: Path, arms=ARMS):
    R = {a: load_arm(d, a) for a in arms}
    common = sorted(set.intersection(*[set(R[a]) for a in arms])) if all(R[a] for a in arms) else []
    return R, common


# ----------------------------------------------------------------------------- statistics
def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj, running = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def contrast(Y: dict, coef: dict, idx_boot: np.ndarray, rng: np.random.Generator) -> dict:
    d = sum(c * Y[a] for a, c in coef.items()).astype(float)        # per-problem contrast value
    n = len(d)
    boot = d[idx_boot].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    lo90, hi90 = np.percentile(boot, [5, 95])
    signs = rng.choice([-1.0, 1.0], size=(B_PERM, n))
    perm = np.abs((signs * d).mean(axis=1))
    obs = abs(d.mean())
    p = (1 + int((perm >= obs - 1e-15).sum())) / (B_PERM + 1)
    if lo90 > -TOST_BOUND and hi90 < TOST_BOUND:
        eq = "equivalent within +-3 pp"
    elif lo > 0 or hi < 0:
        eq = "differs from 0 (CI excludes 0)"
    else:
        eq = "inconclusive"
    return {"estimate_pp": 100 * d.mean(), "ci95_pp": [100 * lo, 100 * hi], "ci90_pp": [100 * lo90, 100 * hi90],
            "p_signflip": p, "tost_pm3pp": eq, "n": n, "n_positive": int((d > 0).sum()), "n_negative": int((d < 0).sum())}


def first_divergence(a, b):
    """Index of the first emitted token that differs, None if identical. A strict-prefix pair (different length, no
    differing token) cannot occur under the early-stop rule; it is reported as divergence at the shorter length and
    counted as an anomaly by the caller."""
    n = min(len(a), len(b))
    for k in range(n):
        if a[k] != b[k]:
            return k
    return None if len(a) == len(b) else n


def hazard_arrays(ids, ref, ref_mask):
    """Per-bin (risk, event) step counts for one problem: steps s >= 1 (token 0 is the shared prefill token) while the
    emitted history so far equals the reference's, restricted to answer-region steps of the reference."""
    D = first_divergence(ids, ref)
    L = min(len(ids), len(ref))
    risk, event = [0] * len(BINS), [0] * len(BINS)
    last = L - 1 if D is None else min(D, L - 1)
    for s in range(1, last + 1):
        if s < len(ref_mask) and ref_mask[s] != 1:
            continue
        for b, (lo, hi) in enumerate(BINS):
            if lo <= s <= hi:
                risk[b] += 1
                if D is not None and s == D:
                    event[b] += 1
                break
    return risk, event, D


def pooled_hazard(risk, event, idx):
    r, e = risk[idx].sum(axis=0), event[idx].sum(axis=0)
    return np.where(r > 0, e / np.maximum(r, 1), np.nan), r, e


# ----------------------------------------------------------------------------- the analysis
def historical_outcomes():
    """Stored historical per-problem outcomes of pure FP16 (condition 1) and pure 4-bit (condition 2)."""
    root = HERE.parent / "results"
    out = {1: {}, 2: {}}
    try:
        a = json.load(open(root / "gsm8k_n280_condition_1_and_3_backfill.json"))
        b = json.load(open(root / "gsm8k_n280_condition_2_and_4.json"))
        c = json.load(open(root / "gsm8k_n650_all_conditions.json"))
        for r in a["condition_1_pure_fp16"]:
            out[1][r["index"]] = r["status"]
        for r in b["condition_2_pure_4bit"]:
            out[2][r["index"]] = r["status"]
        for k in (1, 2):
            for r in c[f"condition_{k}"]:
                out[k][r["index"]] = r["status"]
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
    return out


def analyze(R: dict, common: list, out_dir: Path | None, pilot_dir: Path | None = None) -> dict:
    n = len(common)
    rng = np.random.default_rng(SEED)
    idx_boot = rng.integers(0, n, size=(B_BOOT, n))
    Y = {a: np.array([1 if R[a][i]["status"] == "SOLVED" else 0 for i in common]) for a in ARMS}
    rep = {"n_problems_complete_all_arms": n,
           "n_per_arm_any": {a: len(R[a]) for a in ARMS},
           "accuracy": {a: {"solved": int(Y[a].sum()), "of": n, "pct": 100 * Y[a].mean(),
                            "extraction_failed": sum(R[a][i]["status"] == "EXTRACTION_FAILED" for i in common)} for a in ARMS}}

    # ---- primary family (Holm) and secondary contrasts
    fam = {name: contrast(Y, coef, idx_boot, rng) for name, coef in FAMILY}
    adj = holm([v["p_signflip"] for v in fam.values()])
    for (name, v), p in zip(fam.items(), adj):
        v["p_holm"] = p
    rep["primary_family"] = fam
    rep["secondary_contrasts_exploratory"] = {name: contrast(Y, coef, idx_boot, rng) for name, coef in SECONDARY}
    total = Y["nf4"] - Y["fp16"]
    cache, token = Y["cache_only"] - Y["fp16"], Y["token_only"] - Y["fp16"]
    inter = Y["nf4"] - Y["cache_only"] - Y["token_only"] + Y["fp16"]
    assert (total == cache + token + inter).all()
    rep["identity_check_total_equals_cache_plus_token_plus_interaction"] = True
    rep["decomposition_pp_description_only"] = {"total_gap": 100 * total.mean(), "cache_path": 100 * cache.mean(),
                                                "token_path": 100 * token.mean(), "interaction": 100 * inter.mean(),
                                                "note": "an identity, not an attribution; no percentage-share claim is supported"}

    # ---- 2x2 paired outcome tables against FP16 and NF4
    tabs = {}
    for a in ARMS:
        for ref in ("fp16", "nf4"):
            if a == ref:
                continue
            tabs[f"{a}_vs_{ref}"] = {"both_solved": int(((Y[a] == 1) & (Y[ref] == 1)).sum()), f"{a}_only": int(((Y[a] == 1) & (Y[ref] == 0)).sum()),
                                     f"{ref}_only": int(((Y[a] == 0) & (Y[ref] == 1)).sum()), "neither": int(((Y[a] == 0) & (Y[ref] == 0)).sum())}
    rep["paired_tables"] = tabs

    # ---- divergence from the FP16 reference, hazards under identical history
    ref_ids = {i: R["fp16"][i]["ids"] for i in common}
    ref_mask = {i: R["fp16"][i]["answer_token_mask"] for i in common}
    div, haz, anomalies = {}, {}, {}
    for a in ARMS:
        if a == "fp16":
            continue
        Ds, risks, events, an = [], [], [], 0
        for i in common:
            r, e, D = hazard_arrays(R[a][i]["ids"], ref_ids[i], ref_mask[i])
            if D is not None and D == min(len(R[a][i]["ids"]), len(ref_ids[i])) and len(R[a][i]["ids"]) != len(ref_ids[i]):
                an += 1
            Ds.append(D); risks.append(r); events.append(e)
        risk, event = np.array(risks), np.array(events)
        div[a] = Ds
        anomalies[a] = an
        h, r, e = pooled_hazard(risk, event, np.arange(n))
        bh = np.array([pooled_hazard(risk, event, ib)[0] for ib in idx_boot[:2000]])
        haz[a] = {f"steps_{lo}-{min(hi, 256)}": {"risk_steps": int(r[b]), "events": int(e[b]), "hazard": None if np.isnan(h[b]) else float(h[b]),
                                                  "ci95": [None if np.isnan(x) else float(x) for x in np.nanpercentile(bh[:, b], [2.5, 97.5])] if np.isfinite(bh[:, b]).any() else None}
                  for b, (lo, hi) in enumerate(BINS)}
        haz[a]["pooled"] = {"risk_steps": int(r.sum()), "events": int(e.sum()), "hazard": float(e.sum() / max(1, r.sum()))}
    got = {a: [d for d in div[a] if d is not None] for a in div}
    rep["divergence_from_fp16"] = {a: {"n_diverged": len(got[a]), "of": n, "share_diverged": len(got[a]) / n,
                                       "first_divergence_token": {"median": statistics.median(got[a]) if got[a] else None,
                                                                  "q25": float(np.percentile(got[a], 25)) if got[a] else None,
                                                                  "q75": float(np.percentile(got[a], 75)) if got[a] else None,
                                                                  "min": min(got[a]) if got[a] else None, "max": max(got[a]) if got[a] else None},
                                       "prefix_length_anomalies": anomalies[a]} for a in div}
    rep["hazard_under_identical_history_vs_fp16"] = haz
    hs = {}
    for key in list(haz["cache_only"]):
        hc, ht, hn = haz["cache_only"][key]["hazard"], haz["token_only"][key]["hazard"], haz["nf4"][key]["hazard"]
        hs[key] = {"cache_only": hc, "token_only": ht, "sum_of_single_paths": None if hc is None or ht is None else hc + ht, "nf4_both": hn}
    rep["hazard_additivity_description_only"] = hs
    ks = [5, 10, 20, 40, 80, 120]
    surv = {a: {k: sum(1 for d in div[a] if d is None or d > k) / n for k in ks} for a in div}
    rep["survival_share_identical_to_fp16_through_token_k"] = surv

    # ---- non-diverged problems must have identical outcomes; transitions among diverged problems
    trans = {}
    for a in div:
        nd = [i for i, d in zip(common, div[a]) if d is None]
        bad = [i for i in nd if R[a][i]["status"] != R["fp16"][i]["status"]]
        dv = [(i, d) for i, d in zip(common, div[a]) if d is not None]
        t = {"non_diverged": len(nd), "non_diverged_with_different_outcome(must be 0)": len(bad), "diverged": len(dv)}
        for lo, hi, nm in ((1, 20, "diverged_at_1-20"), (21, 60, "diverged_at_21-60"), (61, 10_000, "diverged_at_61+")):
            sub = [i for i, d in dv if lo <= d <= hi]
            t[nm] = {"n": len(sub), "fp16_solved": sum(R["fp16"][i]["status"] == "SOLVED" for i in sub),
                     f"{a}_solved": sum(R[a][i]["status"] == "SOLVED" for i in sub),
                     "gained(arm solved, fp16 not)": sum(R[a][i]["status"] == "SOLVED" and R["fp16"][i]["status"] != "SOLVED" for i in sub),
                     "lost(fp16 solved, arm not)": sum(R[a][i]["status"] != "SOLVED" and R["fp16"][i]["status"] == "SOLVED" for i in sub)}
        trans[a] = t
    rep["outcome_transitions_vs_fp16"] = trans

    # ---- local branch disagreement in the hybrid cells
    loc = {}
    for a in ("cache_only", "token_only"):
        flips = steps = 0
        for i in common:
            tF, tN = R[a][i]["tF"], R[a][i]["tN"]
            flips += sum(x != y for x, y in zip(tF, tN)); steps += len(tF)
        loc[a] = {"steps": steps, "local_flips_tF_ne_tN": flips, "rate": flips / max(1, steps)}
    rep["local_branch_disagreement"] = loc

    # ---- surprising / contradictory cases (listed individually in the CSV; counts here)
    sc = {"cache_only_solved_fp16_wrong": [i for i in common if R["cache_only"][i]["status"] == "SOLVED" and R["fp16"][i]["status"] != "SOLVED"],
          "token_only_solved_fp16_wrong": [i for i in common if R["token_only"][i]["status"] == "SOLVED" and R["fp16"][i]["status"] != "SOLVED"],
          "oracle_flip_solved_fp16_wrong": [i for i in common if R["oracle_flip"][i]["status"] == "SOLVED" and R["fp16"][i]["status"] != "SOLVED"],
          "fp16_and_nf4_solved_but_cache_only_wrong": [i for i in common if R["fp16"][i]["status"] == "SOLVED" and R["nf4"][i]["status"] == "SOLVED" and R["cache_only"][i]["status"] != "SOLVED"],
          "fp16_and_nf4_solved_but_token_only_wrong": [i for i in common if R["fp16"][i]["status"] == "SOLVED" and R["nf4"][i]["status"] == "SOLVED" and R["token_only"][i]["status"] != "SOLVED"],
          "fp16_solved_nf4_wrong_but_cache_only_and_token_only_both_solved": [i for i in common if R["fp16"][i]["status"] == "SOLVED" and R["nf4"][i]["status"] != "SOLVED" and R["cache_only"][i]["status"] == "SOLVED" and R["token_only"][i]["status"] == "SOLVED"],
          "fp16_solved_nf4_wrong_but_cache_only_and_token_only_both_wrong": [i for i in common if R["fp16"][i]["status"] == "SOLVED" and R["nf4"][i]["status"] != "SOLVED" and R["cache_only"][i]["status"] != "SOLVED" and R["token_only"][i]["status"] != "SOLVED"],
          "cap_boundaries": {a: [i for i in common if R[a][i]["boundary"]["boundary_type"] == "cap"] for a in ARMS},
          "extraction_failed": {a: [i for i in common if R[a][i]["status"] == "EXTRACTION_FAILED"] for a in ARMS}}
    rep["surprising_or_contradictory_cases"] = {k: ({a: v for a, v in x.items()} if isinstance(x, dict) else {"n": len(x), "indices_first_30": x[:30]}) for k, x in sc.items()}

    # ---- historical continuity of the regenerated static arms
    hist = historical_outcomes()
    if "error" in hist:
        rep["historical_continuity"] = hist
    else:
        cont = {}
        for arm, k in (("fp16", 1), ("nf4", 2)):
            have = [i for i in common if i in hist[k]]
            dis = [i for i in have if (hist[k][i] == "SOLVED") != (R[arm][i]["status"] == "SOLVED")]
            cont[arm] = {"compared": len(have), "disagree": len(dis), "disagreeing_indices": dis,
                         "historical_solved": sum(hist[k][i] == "SOLVED" for i in have), "regenerated_solved": sum(R[arm][i]["status"] == "SOLVED" for i in have)}
        rep["historical_continuity"] = cont

    # ---- runtime and memory
    rt = {}
    for a in ARMS:
        recs = [R[a][i] for i in common]
        steps = sum(r["steps"] for r in recs)
        rt[a] = {"mean_wall_s_per_problem": float(np.mean([r["wall_seconds"] for r in recs])), "median_wall_s": float(np.median([r["wall_seconds"] for r in recs])),
                 "ms_per_step": 1000 * sum(r["gen_seconds"] for r in recs) / max(1, steps), "mean_steps": steps / n,
                 "total_gpu_hours_wall": sum(r["wall_seconds"] for r in recs) / 3600,
                 "peak_mem_alloc_gb_max": max(r["peak_mem_allocated"] or 0 for r in recs) / 1e9,
                 "cost_fp16_pass_equiv_per_step": float(np.mean([r["accounting"]["cost_fpe_per_step"] for r in recs])),
                 "early_stopped": sum(r["stopped_early"] for r in recs)}
    rep["runtime_and_memory"] = rt

    # ---- pilot comparison (flip oracle first divergence on the pilot's 50 train problems)
    if pilot_dir is not None:
        rep["pilot_flip_oracle_comparison"] = pilot_comparison(pilot_dir, div)

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "paired_outcomes.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["index"] + [f"{a}_status" for a in ARMS] + [f"{a}_steps" for a in ARMS] + [f"{a}_first_div_vs_fp16" for a in ARMS if a != "fp16"])
            for k, i in enumerate(common):
                w.writerow([i] + [R[a][i]["status"] for a in ARMS] + [R[a][i]["steps"] for a in ARMS] + ["" if div[a][k] is None else div[a][k] for a in ARMS if a != "fp16"])
        with open(out_dir / "divergence_survival.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["k"] + [f"{a}_share_identical_through_k" for a in div])
            for k in range(0, 160):
                w.writerow([k] + [sum(1 for d in div[a] if d is None or d > k) / n for a in div])
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6.4, 4.2))
            for a in div:
                ax.step(range(0, 160), [sum(1 for d in div[a] if d is None or d > k) / n for k in range(0, 160)], where="post", label=a)
            ax.set_xlabel("emitted token index k"); ax.set_ylabel("share of problems identical to FP16 through token k")
            ax.set_title(f"Divergence from the FP16 trajectory (n = {n}, GSM8K test pool)"); ax.legend(); ax.grid(alpha=.3)
            fig.tight_layout(); fig.savefig(out_dir / "survival.png", dpi=160)
        except Exception as e:  # noqa: BLE001
            rep["survival_plot_error"] = str(e)
        (out_dir / "stage1_report.json").write_text(json.dumps(rep, indent=1, default=float))
    return rep


def pilot_comparison(pilot_dir: Path, div_stage1: dict) -> dict:
    """Same first-divergence statistic recomputed on the pilot's stored full-length TRAIN trajectories (calib block:
    fp16, nf4, oracle_flip; 50 problems). Existing-data re-analysis only."""
    d = pilot_dir / "calib"
    if not all((d / f"{a}.jsonl").exists() for a in ("fp16", "nf4", "oracle_flip")):
        return {"error": "pilot calib records not found"}
    P = {a: {json.loads(l)["index"]: json.loads(l) for l in (d / f"{a}.jsonl").read_text().splitlines() if l.strip()} for a in ("fp16", "nf4", "oracle_flip")}
    common = sorted(set.intersection(*[set(v) for v in P.values()]))
    out = {"n_problems": len(common), "split": "train (pilot calibration block, full 256-token generations)"}
    for a in ("oracle_flip", "nf4"):
        ds = [first_divergence(P[a][i]["ids"], P["fp16"][i]["ids"]) for i in common]
        g = [x for x in ds if x is not None]
        out[f"{a}_vs_fp16"] = {"n_diverged": len(g), "of": len(common), "median": statistics.median(g) if g else None,
                               "min": min(g) if g else None, "max": max(g) if g else None}
    for a in ("oracle_flip", "cache_only", "token_only", "nf4"):
        if a in div_stage1:
            g = [x for x in div_stage1[a] if x is not None]
            out[f"stage1_{a}_vs_fp16"] = {"n_diverged": len(g), "of": len(div_stage1[a]), "median": statistics.median(g) if g else None,
                                          "min": min(g) if g else None, "max": max(g) if g else None}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(HERE / "stage1_runs" / "eval"))
    ap.add_argument("--out", default=str(HERE / "stage1_analysis"))
    ap.add_argument("--pilot-dir", default=str(HERE / "pilot_runs"))
    a = ap.parse_args()
    R, common = load_all(Path(a.runs))
    if not common:
        sys.exit("no problems completed for all five arms")
    rep = analyze(R, common, Path(a.out), Path(a.pilot_dir))
    print(json.dumps({k: rep[k] for k in ("n_problems_complete_all_arms", "accuracy", "primary_family", "decomposition_pp_description_only")}, indent=1, default=float))


if __name__ == "__main__":
    main()
