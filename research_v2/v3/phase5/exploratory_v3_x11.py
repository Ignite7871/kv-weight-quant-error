"""EXPLORATORY analysis X11 of v3/phase5/EXPLORATORY_PLAN_V3.md (dated entry 2026-10-31, committed 9706f11 before computing). Not locked.

First-order H-B (QA vs QO) benchmark from the frozen calibration state; descriptive. CPU only.
  venv/Scripts/python.exe research_v2/v3/phase5/exploratory_v3_x11.py   -> v3/phase5/exploratory_v3_x11.json, EXPLORATORY_RESULTS_X11.md
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

RV = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RV))
import analyze_v3 as L                           # noqa: E402
import rv3.core as V                             # noqa: E402

MODELS = L.ORDER
RUNS = RV / "v3_runs"
HERE = Path(__file__).resolve().parent
SEED_X11 = 20261212
R, D = 8, 64
LABEL = "EXPLORATORY, descriptive (EXPLORATORY_PLAN_V3.md, dated entry X11, 2026-10-31)"
LOCKED = json.loads((HERE / "v3_report_locked.json").read_text())


def load(m, arms):
    return {a: L.load_jsonl(RUNS / m / "tierA" / f"{a.replace('*', 'star')}.jsonl") for a in arms}


def ratio(w, e8):
    return float((w * e8 / R).sum() / (w * (1 - e8) / (D - R)).sum())


def x11():
    rng = np.random.default_rng(SEED_X11)
    out = {}
    for m in MODELS:
        aux = torch.load(RUNS / m / "calib" / "aux_state.pt")
        cal = json.loads((RUNS / m / "calibration.json").read_text())
        M = torch.as_tensor(aux["M"]).double()                          # [L, n_kv, d, d]
        _, E8t, _ = V.subspace_from_M(M, R)
        e8 = np.asarray(cal["E8_per_slot"], float)
        trM = np.einsum("lhdd->lh", M.numpy())
        dK = np.asarray(torch.as_tensor(aux["d2"]).double().numpy())[0::2].sum(-1)   # K rows: natural key-error energy per slot
        per_slot = (e8 / R) / ((1 - e8) / (D - R))
        hb = LOCKED["per_model"][m]["family"]["HB QA-QO"]
        A = load(m, ["QO", "C1-V"])
        idx = sorted(set(A["QO"]) & set(A["C1-V"]))
        s = L.summarize(L.theta(A, "C1-V", "QO", idx), rng)
        out[m] = {"B": ratio(trM * dK, e8), "B_q": ratio(trM, e8), "B_1": ratio(np.ones_like(e8), e8),
                  "per_slot_ratio_median": float(np.median(per_slot)), "per_slot_ratio_range": [float(per_slot.min()), float(per_slot.max())],
                  "observed_exp_theta_QA_QO": math.exp(hb["estimate"]), "observed_ci95": [math.exp(x) for x in hb["ci95"]],
                  "dilution_gC1V_over_gQO": math.exp(s["estimate"]), "dilution_ci95": [math.exp(x) for x in s["ci95"]],
                  "n_slots": int(e8.size),
                  "check: max |E8 recomputed from M - frozen E8_per_slot|": float(np.abs(E8t.numpy() - e8).max())}
    return out


def to_md(o):
    Lm = ["# EXPLORATORY results X11 — 2026-10-31", "",
          "**Exploratory and descriptive** (EXPLORATORY_PLAN_V3.md, dated entry committed 9706f11 before this script ran). "
          "Source: `exploratory_v3_x11.json`. B = Σ w E8/r ÷ Σ w (1−E8)/(d−r), with r = 8, d = 64 and w = tr(M)·Σ‖Δ_K‖² per (layer, KV head) "
          "from the frozen calibration state.", "",
          "First order only, and the observed ratio is diluted by the V draw that QA and QO share. g(C1-V)/g(QO) is a dilution "
          "indicator, not a correction.", "",
          "| Model | B (energy-weighted) | B_q (tr M) | B_1 (equal) | per-slot median [min, max] | observed e^θ(QA, QO) [95 % CI] | g(C1-V)/g(QO) [95 % CI] |",
          "|---|---|---|---|---|---|---|"]
    for m in MODELS:
        v = o[m]
        Lm.append(f"| {m} | {v['B']:.1f} | {v['B_q']:.1f} | {v['B_1']:.1f} | {v['per_slot_ratio_median']:.1f} "
                  f"[{v['per_slot_ratio_range'][0]:.1f}, {v['per_slot_ratio_range'][1]:.1f}] | "
                  f"{v['observed_exp_theta_QA_QO']:.2f} [{v['observed_ci95'][0]:.2f}, {v['observed_ci95'][1]:.2f}] | "
                  f"{v['dilution_gC1V_over_gQO']:.3f} [{v['dilution_ci95'][0]:.3f}, {v['dilution_ci95'][1]:.3f}] |")
    Lm += ["", "Checks (should be ~0): " + "; ".join(f"{m} {o[m]['check: max |E8 recomputed from M - frozen E8_per_slot|']:.1e}" for m in MODELS)]
    return "\n".join(Lm) + "\n"


def main():
    res = {"label": LABEL, "per_model": x11()}
    (HERE / "exploratory_v3_x11.json").write_text(json.dumps(res, indent=1, default=float))
    md = to_md(res["per_model"])
    (HERE / "EXPLORATORY_RESULTS_X11.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
