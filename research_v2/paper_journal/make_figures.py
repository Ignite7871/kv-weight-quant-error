"""Journal-manuscript figures from the stored records and the locked report JSONs (no GPU, no re-analysis).
  venv/Scripts/python.exe research_v2/paper_journal/make_figures.py
fig_e1_contrasts.pdf : Tier A paired log-JS contrasts, Qwen and Llama (locked reports), and Qwen Tier B (tokens)
fig_e1_dose.pdf      : dose-response of the isotropic control per model with the natural, sign-randomized and effect-matched arms marked
"""
import json
import shutil
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RV = HERE.parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e3e2de"
CQ, CL = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "savefig.dpi": 300, "legend.frameon": False})
q = json.loads((RV / "e1_runs" / "e1_report.json").read_text())
l = json.loads((RV / "e1l_runs" / "e1l_report.json").read_text())
load = lambda p: {json.loads(x)["index"]: json.loads(x) for x in open(p, encoding="utf-8") if x.strip()}   # noqa: E731

ROWS = [("natural vs random direction (C1)", "primary", "primary_log_JS_NAT_minus_C1"), ("natural vs sign-randomized (C2)", "sec", "A: NAT - C2 (log JS)"),
        ("sign-randomized vs random direction", "sec", "A: C2 - C1 (log JS)"), ("natural vs effect-matched (C1*)", "sec", "A: NAT - C1* (effect-matched, log JS)")]


def get(rep, kind, key):
    return rep["tierA"]["primary_log_JS_NAT_minus_C1"] if kind == "primary" else rep["tierA"]["secondary"][key]


def fig_contrasts():
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.6, 3.3), gridspec_kw={"width_ratios": [2.2, 1]}, layout="constrained")
    for i, (lab, kind, key) in enumerate(ROWS):
        y0 = len(ROWS) - 1 - i
        for rep, col, mk, off, name in ((q, CQ, "o", 0.14, "Qwen2.5-0.5B (n=200)"), (l, CL, "s", -0.14, "Llama-3.2-1B-Instruct (n=200)")):
            v = get(rep, kind, key)
            ax.errorbar(v["estimate"], y0 + off, xerr=[[v["estimate"] - v["ci95"][0]], [v["ci95"][1] - v["estimate"]]], fmt=mk, color=col, ms=5, lw=1.4, capsize=2.5,
                        label=name if i == 0 else None)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels([r[0] for r in ROWS][::-1])
    ax.set_xlabel("paired difference in log mean-JS, 95% problem-clustered CI\n(negative: first arm less disruptive)")
    ax.set_title("Tier A: identical token history", fontsize=8.5, pad=22)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7, handletextpad=0.3, columnspacing=1.0)
    P = q["tierB"]["primary_restricted_first_divergence_NAT_minus_C1"]
    S = q["tierB"]["secondary"]["B: NAT - C2 (restricted first divergence)"]
    for j, (lab, v) in enumerate((("natural vs C1", P), ("natural vs C2", S))):
        bx.errorbar(v["estimate"], 1 - j, xerr=[[v["estimate"] - v["ci95"][0]], [v["ci95"][1] - v["estimate"]]], fmt="o", color=CQ, ms=5, lw=1.4, capsize=2.5)
    bx.axvline(0, color=MUTED, lw=0.8)
    bx.set_yticks([1, 0])
    bx.set_yticklabels(["natural vs C1", "natural vs C2"])
    bx.set_xlabel("first-divergence token\ndifference (+: later)")
    bx.set_title("Tier B (Qwen, n=400)", fontsize=8.5, pad=22)
    bx.grid(axis="x", color=GRID, lw=0.6)
    fig.savefig(OUT / "fig_e1_contrasts.pdf", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def gm(R, tag):
    return float(np.exp(np.mean([np.log(max(np.mean(R[tag][i]["js"]), 1e-9)) for i in R[tag]])))


def boot_gm(R, tag, B=2000, seed=3):
    v = np.array([np.log(max(np.mean(R[tag][i]["js"]), 1e-9)) for i in sorted(R[tag])])
    r = np.random.default_rng(seed)
    m = v[r.integers(0, len(v), size=(B, len(v)))].mean(axis=1)
    return np.exp(np.percentile(m, [2.5, 97.5]))


def fig_dose():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=False, layout="constrained")
    for ax, (name, root, col, mk) in zip(axes, (("Qwen2.5-0.5B", "e1_runs", CQ, "o"), ("Llama-3.2-1B-Instruct", "e1l_runs", CL, "s"))):
        R = {t: load(RV / root / "tierA" / f"{t}.jsonl") for t in ("NAT", "C1_a0.5", "C1_a1.0", "C1_a2.0", "C2_a1.0", "C1star")}
        al = [0.5, 1.0, 2.0]
        ys = [gm(R, f"C1_a{a}") for a in al]
        ax.plot(al, ys, "-", color=col, lw=1.4)
        for a, t in zip(al, ("C1_a0.5", "C1_a1.0", "C1_a2.0")):
            lo, hi = boot_gm(R, t)
            ax.errorbar(a, gm(R, t), yerr=[[gm(R, t) - lo], [hi - gm(R, t)]], fmt=mk, color=col, ms=5, capsize=2)
        ax.axhline(gm(R, "NAT"), color=INK, lw=1.0, ls="--", label="natural NF4 difference (NAT)")
        ax.axhline(gm(R, "C2_a1.0"), color=MUTED, lw=1.0, ls=":", label="sign-randomized (C2)")
        a_star = json.loads((RV / root / ("e1_calibration.json" if root == "e1_runs" else "e1l_calibration.json")).read_text())["alpha_star"]
        ax.plot([a_star], [gm(R, "C1star")], "D", mfc="white", mec=col, ms=5, ls="none", label=f"effect-matched C1* (alpha*={a_star:.2f})")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(al)
        ax.set_xticklabels(["0.5", "1", "2"])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.legend(loc="upper left", fontsize=6.5, handlelength=2.0)
        ax.set_xlabel("perturbation scale alpha (isotropic C1)")
        ax.set_title(name, fontsize=8.5)
        ax.grid(color=GRID, lw=0.6)
    axes[0].set_ylabel("geometric mean over problems of\nper-problem mean JS to FP16")
    fig.savefig(OUT / "fig_e1_dose.pdf", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


if __name__ == "__main__":
    fig_contrasts()
    fig_dose()
    for f in ("fig_forest.pdf", "fig_interaction.pdf", "fig_accuracy.pdf", "fig_km.pdf"):
        shutil.copy(RV / "paper" / "figures" / f, OUT / f)
    print(sorted(p.name for p in OUT.glob("*.pdf")))
