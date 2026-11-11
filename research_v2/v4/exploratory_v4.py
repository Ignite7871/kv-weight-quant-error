"""v4 exploratory analyses X1-X4 (EXPLORATORY_PLAN_V4.md, commit 1409b94). Descriptive only; not part of the locked analysis; not blind.

Reads the test records with the locked primitives (analyze_v4.load_test / theta / CONTRASTS) and the locked report; writes
v4_runs/v4_exploratory.json and v4/exploratory/X_RESULTS_V4.md. Seed 20261109 (not the locked seed).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_v4 as A4  # noqa: E402  (locked; imported read-only)

SEED, B = 20261109, 10_000
QWEN, LLAMA, MISTRAL = A4.QWEN, A4.LLAMA, A4.MISTRAL
FAMS = {"qwen": QWEN, "llama": LLAMA}
REPORT = A4.OUT / "v4_report.json"


def size_params():
    man = A4.manifest()
    return {m: sum(v["size"] for v in man[m]["shards"].values()) / 2 for m in A4.LADDER}, {m: man[m]["geometry"]["head_dim"] for m in A4.LADDER}


def slope(x, y):
    """OLS slope of y on x along the last axis (y: (n,) or (B, n))."""
    x = np.asarray(x, float); xc = x - x.mean()
    y = np.asarray(y, float)
    return (xc * (y - y.mean(axis=-1, keepdims=True))).sum(-1) / (xc * xc).sum()


def pct(a):
    return [float(v) for v in np.percentile(a, [2.5, 97.5])]


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    rep = json.loads(REPORT.read_text(encoding="utf-8"))
    P, HD = size_params()
    x = {m: math.log10(P[m]) for m in A4.LADDER}
    contrasts = ["H1", "H2", "H3", "NAT-C1"]
    th, check = {}, {}
    for m in A4.LADDER:
        A = A4.load_test(m)
        idx = sorted(set.intersection(*[set(v) for v in A.values()]))
        th[m] = {c: A4.theta(A, *A4.CONTRASTS[c], idx) for c in contrasts}
        check[m] = {c: abs(float(th[m][c].mean()) - rep["per_model"][m]["contrasts"][c]["estimate"]) for c in contrasts}
    max_dev = max(v for d in check.values() for v in d.values())
    assert max_dev < 1e-12, f"per-model means do not reproduce the locked estimates: {max_dev}"

    rng = np.random.default_rng(SEED)
    # one joint resample of problems per model per draw (shared across contrasts so X1/X3/X4 draws are coherent)
    boot = {}
    for m in A4.LADDER:
        n = len(th[m]["H1"])
        I = rng.integers(0, n, size=(B, n))
        boot[m] = {c: th[m][c][I].mean(1) for c in contrasts}
        boot[m]["_I"] = I
    y = {m: {c: float(th[m][c].mean()) for c in contrasts} for m in A4.LADDER}
    ci = {m: {c: rep["per_model"][m]["contrasts"][c]["ci95"] for c in contrasts} for m in A4.LADDER}

    def fam_slope(models, c, absval=False):
        xs = [x[m] for m in models]
        f = (lambda v: np.abs(v)) if absval else (lambda v: v)
        pt = float(slope(xs, np.array([f(y[m][c]) for m in models])))
        bs = slope(xs, np.stack([f(boot[m][c]) for m in models], axis=1))
        return {"models": models, "slope_per_decade": pt, "ci95_boot": pct(bs),
                "pearson_x_y": float(np.corrcoef(xs, [f(y[m][c]) for m in models])[0, 1])}

    out = {"plan_commit": "1409b94", "seed": SEED, "B": B, "not_blind": True, "status": "exploratory, descriptive only",
           "reproduces_locked_estimates_max_abs_dev": max_dev,
           "size": {m: {"params_16bit": P[m], "log10_params": x[m], "head_dim": HD[m]} for m in A4.LADDER}}

    # X1
    X1 = {}
    for fam, models in FAMS.items():
        X1[fam] = {"H1": fam_slope(models, "H1"), "H1_abs": fam_slope(models, "H1", True), "NAT-C1": fam_slope(models, "NAT-C1")}
    X1["per_model"] = {m: {c: {"estimate": y[m][c], "ci95_locked": ci[m][c]} for c in ("H1", "NAT-C1")} for m in A4.LADDER}
    out["X1"] = X1

    # X2
    X2 = {}
    for m in A4.LADDER:
        pl = rep["per_model"][m]["H5_C_per_layer_descriptive"]
        row = {}
        for p in ("K", "V"):
            v = np.asarray(pl[p], float); L = len(v); depth = np.arange(L) / (L - 1)
            row[p] = {"L": L, "mean": float(v.mean()), "median": float(np.median(v)), "min": float(v.min()), "max": float(v.max()),
                      "argmax_layer": int(v.argmax()), "argmax_rel_depth": float(v.argmax() / (L - 1)),
                      "layer0": float(v[0]), "mean_layers_1_on": float(v[1:].mean()), "spearman_vs_depth": spearman(v, depth)}
        vk, vv = np.asarray(pl["K"]), np.asarray(pl["V"])
        row["frac_layers_V_gt_K"] = float((vv > vk).mean())
        X2[m] = row
    out["X2"] = X2

    # X3
    h1 = [y[m]["H1"] for m in A4.LADDER]; nc = [y[m]["NAT-C1"] for m in A4.LADDER]
    no_mi = [m for m in A4.LADDER if m not in MISTRAL]
    X3 = {"across_models_all8": {"pearson": float(np.corrcoef(h1, nc)[0, 1]), "spearman": spearman(np.array(h1), np.array(nc))},
          "across_models_7_no_mistral": {"pearson": float(np.corrcoef([y[m]["H1"] for m in no_mi], [y[m]["NAT-C1"] for m in no_mi])[0, 1]),
                                         "spearman": spearman(np.array([y[m]["H1"] for m in no_mi]), np.array([y[m]["NAT-C1"] for m in no_mi]))},
          "within_model_problem_level": {}}
    for m in A4.LADDER:
        a, b = th[m]["H1"], th[m]["NAT-C1"]
        I = boot[m]["_I"]
        A_, B_ = a[I], b[I]
        Ac, Bc = A_ - A_.mean(1, keepdims=True), B_ - B_.mean(1, keepdims=True)
        rb = (Ac * Bc).sum(1) / np.sqrt((Ac * Ac).sum(1) * (Bc * Bc).sum(1))
        X3["within_model_problem_level"][m] = {"pearson": float(np.corrcoef(a, b)[0, 1]), "ci95_boot": pct(rb)}
    out["X3"] = X3

    # X4
    X4 = {"head_dim": HD, "hd128_only_slopes": {}, "step_hd64_to_next": {}}
    for fam, models in FAMS.items():
        m128 = [m for m in models if HD[m] == 128]
        X4["hd128_only_slopes"][fam] = {"H1": fam_slope(m128, "H1"), "NAT-C1": fam_slope(m128, "NAT-C1")}
        m64 = [m for m in models if HD[m] == 64]
        assert len(m64) == 1
        a, b = m64[0], m128[0]
        X4["step_hd64_to_next"][fam] = {"from": a, "to": b, "delta_log10_params": x[b] - x[a],
                                        **{c: {"delta": y[b][c] - y[a][c], "ci95_boot": pct(boot[b][c] - boot[a][c])} for c in contrasts}}
    out["X4"] = X4

    for m in boot:
        boot[m].pop("_I")
    (A4.OUT / "v4_exploratory.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    write_md(out)
    print(json.dumps({k: out[k] for k in ("reproduces_locked_estimates_max_abs_dev",)}, indent=1))


def f3(v):
    return f"{v:+.3f}"


def ci3(c):
    return f"[{c[0]:+.3f}, {c[1]:+.3f}]"


def write_md(o):
    L = ["# v4 exploratory results X1-X4 (EXPLORATORY, descriptive only, not blind)", "",
         f"Plan: `v4/EXPLORATORY_PLAN_V4.md` (commit {o['plan_commit']}, written 2026-11-09 after the locked results were read). "
         f"Seed {o['seed']}, B = {o['B']:,} problem resamples per model. Per-model means reproduce the locked estimates "
         f"(max |dev| = {o['reproduces_locked_estimates_max_abs_dev']:.1e}). No p-values; nothing here changes a locked verdict.", "",
         "## Sizes", "", "| Model | params (16-bit, stored) | log10 P | head_dim |", "|---|---|---|---|"]
    for m, s in o["size"].items():
        L.append(f"| {m} | {s['params_16bit']/1e9:.3f} B | {s['log10_params']:.3f} | {s['head_dim']} |")
    L += ["", "## X1: slope of estimate per decade of parameters (within family)", "",
          "| Family | contrast | slope / decade | 95 % bootstrap CI (problems only) | Pearson(x, y) |", "|---|---|---|---|---|"]
    for fam in ("qwen", "llama"):
        for c in ("H1", "H1_abs", "NAT-C1"):
            s = o["X1"][fam][c]
            L.append(f"| {fam} ({len(s['models'])} models) | {c.replace('H1_abs', 'magnitude of H1')} | {f3(s['slope_per_decade'])} | {ci3(s['ci95_boot'])} | {s['pearson_x_y']:+.3f} |")
    L += ["", "The CI covers problem sampling given these checkpoints only, not a population of models (Qwen 2 df, Llama 1 df).", "",
          "## X2: per-layer C^P (from the locked report; summary only, no CI, not head_dim-adjusted)", "",
          "| Model | path | L | mean | median | min | max | argmax layer (rel. depth) | layer 0 | mean layers 1+ | Spearman vs depth |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m, r in o["X2"].items():
        for p in ("K", "V"):
            s = r[p]
            L.append(f"| {m} | {p} | {s['L']} | {s['mean']:.3f} | {s['median']:.3f} | {s['min']:.3f} | {s['max']:.3f} | "
                     f"{s['argmax_layer']} ({s['argmax_rel_depth']:.2f}) | {s['layer0']:.3f} | {s['mean_layers_1_on']:.3f} | {s['spearman_vs_depth']:+.2f} |")
    L += ["", "| Model | fraction of layers with C^V > C^K |", "|---|---|"]
    for m, r in o["X2"].items():
        L.append(f"| {m} | {r['frac_layers_V_gt_K']:.2f} |")
    x3 = o["X3"]
    L += ["", "## X3: H1 against NAT-C1", "",
          f"Across models (point values, no CI): all 8 Pearson {x3['across_models_all8']['pearson']:+.3f}, Spearman {x3['across_models_all8']['spearman']:+.3f}; "
          f"7 without Mistral Pearson {x3['across_models_7_no_mistral']['pearson']:+.3f}, Spearman {x3['across_models_7_no_mistral']['spearman']:+.3f}.", "",
          "| Model | within-model problem-level Pearson(H1_i, NAT-C1_i) | 95 % bootstrap CI |", "|---|---|---|"]
    for m, r in x3["within_model_problem_level"].items():
        L.append(f"| {m} | {r['pearson']:+.3f} | {ci3(r['ci95_boot'])} |")
    L += ["", "Both contrasts share problem runs, so a positive within-model r is partly expected by construction.", "",
          "## X4: head_dim", "", "head_dim: " + ", ".join(f"{m} {h}" for m, h in o["X4"]["head_dim"].items()) + ".", "",
          "| Family | contrast | slope / decade, head_dim-128 models only | 95 % bootstrap CI |", "|---|---|---|---|"]
    for fam, r in o["X4"]["hd128_only_slopes"].items():
        for c in ("H1", "NAT-C1"):
            L.append(f"| {fam} ({len(r[c]['models'])} models) | {c} | {f3(r[c]['slope_per_decade'])} | {ci3(r[c]['ci95_boot'])} |")
    L += ["", "| Family | step | Δlog10 P | contrast | Δ estimate | 95 % bootstrap CI |", "|---|---|---|---|---|---|"]
    for fam, r in o["X4"]["step_hd64_to_next"].items():
        for c in ("H1", "H2", "H3", "NAT-C1"):
            L.append(f"| {fam} | {r['from']} (64) → {r['to']} (128) | {r['delta_log10_params']:.3f} | {c} | {f3(r[c]['delta'])} | {ci3(r[c]['ci95_boot'])} |")
    L += ["", "Size and head_dim change together in this step; the difference cannot be attributed to either.", ""]
    p = HERE / "exploratory" / "X_RESULTS_V4.md"
    p.parent.mkdir(exist_ok=True)
    p.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
