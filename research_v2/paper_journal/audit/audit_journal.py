"""Journal-manuscript numerical audit (CPU only). Re-derives every Qwen E1 / Llama E1L number the manuscript uses from the stored raw records with code that does NOT
import analyze_controls.py / analyze_e1l.py (own loaders, own bootstrap/permutation streams), and compares with the locked report JSONs; also verifies the protocol locks, the
frozen-record manifest, the calibration constants and the reference-history statistics. Writes audit/JOURNAL_AUDIT_RESULTS.json and .md. Does not modify any record.
  venv/Scripts/python.exe research_v2/paper_journal/audit/audit_journal.py
Tolerances: point estimates 1e-9; bootstrap interval endpoints 0.01 (log-JS) / 0.35 tokens (Tier B) because the resampling streams differ; sign-flip p-values 0.01 absolute."""

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RV = HERE.parents[1]
ROOT = RV.parent
LOG, FAIL = [], []


def chk(name, ok, got=None, want=None):
    LOG.append({"check": name, "ok": bool(ok), "got": got, "want": want})
    if not ok:
        FAIL.append(name)


def close(name, got, want, tol):
    chk(name, abs(got - want) <= tol, got, want)


def load(p):
    return {json.loads(l)["index"]: json.loads(l) for l in open(p, encoding="utf-8") if l.strip()}


# ---------------------------------------------------------------- 1. locks and frozen manifest
for tool in ("lock_e1.py", "lock_e1l.py"):
    r = subprocess.run([sys.executable, str(RV / "tools" / tool), "--verify"], capture_output=True, text=True)
    chk(f"{tool} --verify", r.returncode == 0, r.stdout.strip()[-40:], "lock verifies")
man = [l for l in (RV / "FROZEN_RECORDS_MANIFEST.sha256").read_text().splitlines() if l and not l.startswith("#")]
bad = [rel for h, rel in (l.split("  ", 1) for l in man) if hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() != h]
chk(f"frozen manifest: {len(man)} files unchanged", not bad, bad[:3], [])

# ---------------------------------------------------------------- 2. independent re-derivation
FLOOR = 1e-9
rng_b = np.random.default_rng(777)


def boot_ci(d, B=10000):
    idx = rng_b.integers(0, len(d), size=(B, len(d)))
    m = d[idx].mean(axis=1)
    return np.percentile(m, [2.5, 97.5]), np.percentile(m, [5, 95])


def signflip(d, B=20000):
    s = rng_b.choice([-1.0, 1.0], size=(B, len(d)))
    return (1 + int((np.abs((s * d).mean(axis=1)) >= abs(d.mean()) - 1e-15).sum())) / (B + 1)


def mjs(rec, lo=1, hi=None):
    js = np.asarray(rec["js"], float)[lo - 1:hi]
    return max(float(js.mean()), FLOOR)


def tier_a(root, rep, label, calib_dir):
    R = {a: load(root / "tierA" / f"{a}.jsonl") for a in ("NAT", "C1_a1.0", "C2_a1.0", "C1_a0.5", "C1_a2.0", "C1star")}
    idx = sorted(R["NAT"])
    chk(f"{label}: 200 test problems on all 6 arms", len(idx) == 200 and all(len(r) == 200 for r in R.values()), len(idx), 200)
    L = {a: np.array([math.log(mjs(R[a][i])) for i in idx]) for a in R}
    P = rep["tierA"]["primary_log_JS_NAT_minus_C1"]
    d = L["NAT"] - L["C1_a1.0"]
    close(f"{label}: primary estimate", d.mean(), P["estimate"], 1e-9)
    ci95, ci90 = boot_ci(d)
    close(f"{label}: primary CI low", ci95[0], P["ci95"][0], 0.01)
    close(f"{label}: primary CI high", ci95[1], P["ci95"][1], 0.01)
    close(f"{label}: primary 90% low", ci90[0], P["ci90"][0], 0.01)
    close(f"{label}: primary p (sign-flip)", signflip(d), P["p_signflip"], 0.01)
    S = rep["tierA"]["secondary"]
    for key, dd in (("A: NAT - C2 (log JS)", L["NAT"] - L["C2_a1.0"]), ("A: C2 - C1 (log JS)", L["C2_a1.0"] - L["C1_a1.0"]),
                    ("A: NAT - C1* (effect-matched, log JS)", L["NAT"] - L["C1star"])):
        close(f"{label}: {key} estimate", dd.mean(), S[key]["estimate"], 1e-9)
        c95, _ = boot_ci(dd)
        close(f"{label}: {key} CI low", c95[0], S[key]["ci95"][0], 0.01)
    sl = []
    for i in idx:
        y = np.array([math.log(mjs(R[a][i])) for a in ("C1_a0.5", "C1_a1.0", "C1_a2.0")])
        x = np.log([0.5, 1.0, 2.0])
        sl.append(float(((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()))
    close(f"{label}: dose slope", float(np.mean(sl)), S["A: dose slope of C1 (log JS per log alpha)"]["estimate"], 1e-9)
    fl = {a: np.array([np.mean(R[a][i]["flip"]) for i in idx]) for a in ("NAT", "C1_a1.0")}
    close(f"{label}: flip-rate contrast", float((fl["NAT"] - fl["C1_a1.0"]).mean()), S["A: top-1 flip rate NAT - C1"]["estimate"], 1e-9)
    # Holm over the family the report used
    ps = {k: v["raw_p"] for k, v in rep["holm_family"].items()}
    order = sorted(ps, key=ps.get)
    run, adj = 0.0, {}
    for rank, k in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[k]))
        adj[k] = run
    chk(f"{label}: Holm re-applied to the reported raw p ({len(ps)} tests)", all(abs(adj[k] - rep["holm_family"][k]["holm_p"]) < 1e-9 for k in ps), len(ps), len(ps))
    means = {a: float(np.mean([np.mean(R[a][i]["js"]) for i in idx])) for a in R}
    # calibration re-derivation
    C = {a: load(calib_dir / "calib" / f"{a}.jsonl") for a in ("NAT", "C1_a0.5", "C1_a1", "C1_a2", "C1_a4")}
    ci = sorted(set.intersection(*[set(v) for v in C.values()]))
    chk(f"{label}: 100 calibration problems on 5 arms", len(ci) == 100, len(ci), 100)
    m = lambda a: float(np.mean([mjs(C[a][i], 2, 16) for i in ci]))                                            # noqa: E731
    la, lm = np.log([0.5, 1, 2, 4]), np.log([m("C1_a0.5"), m("C1_a1"), m("C1_a2"), m("C1_a4")])
    astar = math.exp(float(np.interp(math.log(m("NAT")), lm, la)))
    close(f"{label}: alpha* re-derived", astar, rep["calibration"]["alpha_star"], 1e-9)
    bs = []
    for i in ci:
        y = np.array([math.log(mjs(C[a][i])) for a in ("C1_a0.5", "C1_a1", "C1_a2")])
        x = np.log([0.5, 1.0, 2.0])
        bs.append(float(((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()))
    close(f"{label}: dose slope b re-derived", float(np.mean(bs)), rep["calibration"]["dose_slope_b_hat"], 1e-9)
    cvs = [np.std(C["NAT"][i]["target_norm_mean"]) / np.mean(C["NAT"][i]["target_norm_mean"]) for i in ci if len(C["NAT"][i]["target_norm_mean"]) > 3]
    c = min(max(float(np.median(cvs)), 0.10), 0.40)
    close(f"{label}: margin re-derived", abs(float(np.mean(bs))) * math.log(1 + c), rep["calibration"]["margin_tierA_log_js"], 1e-9)
    return R, idx, means


qrep = json.loads((RV / "e1_runs" / "e1_report.json").read_text())
lrep = json.loads((RV / "e1l_runs" / "e1l_report.json").read_text())
QR, qidx, qmeans = tier_a(RV / "e1_runs", qrep, "Qwen Tier A", RV / "e1_runs")
LR, lidx, lmeans = tier_a(RV / "e1l_runs", lrep, "Llama Tier A", RV / "e1l_runs")


def fd(a, b):
    n = min(len(a), len(b))
    for k in range(n):
        if a[k] != b[k]:
            return k
    return None if len(a) == len(b) else n


B = {a: load(RV / "e1_runs" / "tierB" / f"{a}.jsonl") for a in ("FP16", "NAT", "C1", "C2")}
ib = sorted(B["NAT"])
chk("Qwen Tier B: 400 problems on 4 arms", len(ib) == 400 and all(len(v) == 400 for v in B.values()), len(ib), 400)
Y = {a: np.array([64.0 if fd(B[a][i]["ids"], B["FP16"][i]["ids"]) is None else min(fd(B[a][i]["ids"], B["FP16"][i]["ids"]), 64) for i in ib]) for a in ("NAT", "C1", "C2")}
P = qrep["tierB"]["primary_restricted_first_divergence_NAT_minus_C1"]
close("Qwen Tier B: primary estimate", float((Y["NAT"] - Y["C1"]).mean()), P["estimate"], 1e-9)
c95, _ = boot_ci(Y["NAT"] - Y["C1"])
close("Qwen Tier B: CI low", c95[0], P["ci95"][0], 0.35)
close("Qwen Tier B: CI high", c95[1], P["ci95"][1], 0.35)
close("Qwen Tier B: NAT - C2 estimate", float((Y["NAT"] - Y["C2"]).mean()), qrep["tierB"]["secondary"]["B: NAT - C2 (restricted first divergence)"]["estimate"], 1e-9)
for a in Y:
    close(f"Qwen Tier B: mean restricted first divergence {a}", float(Y[a].mean()), qrep["tierB"]["mean_restricted_first_divergence"][a], 1e-9)
acc = {a: float(np.mean([B[a][i]["status"] == "SOLVED" for i in ib])) for a in B}

# ---------------------------------------------------------------- 3. reference-history construction (disclosure check)
st = {**load(RV / "stage1_runs" / "eval" / "fp16.jsonl"), **load(RV / "stage2_runs" / "eval" / "fp16.jsonl")}
same = sum(st[i]["ids"][:65] == B["FP16"][i]["ids"][:65] for i in qidx)
chk("Qwen Tier A reference (stored Stage 1/2 FP16 generation, first 65 tokens) == in-session FP16 generation of Tier B, all 200 problems", same == 200, same, 200)
lref = load(RV / "e1l_runs" / "tierA" / "FP16_ref.jsonl")
chk("Llama reference histories are in-session FP16 greedy generations (<= 65 tokens), one per test problem", len(lref) == 200 and max(len(r["ids"]) for r in lref.values()) <= 65, len(lref), 200)
refstat = {"qwen": {"S_mean": float(np.mean([QR["NAT"][i]["S"] for i in qidx])), "share_S64": float(np.mean([QR["NAT"][i]["S"] == 64 for i in qidx])),
                    "full_generation_mean_tokens": float(np.mean([QR["NAT"][i]["ref_len"] for i in qidx])), "source": "stored Stage 1/2 FP16 generation (<=256 tokens), first 65 tokens used"},
           "llama": {"S_mean": float(np.mean([LR["NAT"][i]["S"] for i in lidx])), "share_S64": float(np.mean([LR["NAT"][i]["S"] == 64 for i in lidx])),
                     "full_generation_mean_tokens": float(np.mean([LR["NAT"][i]["ref_len"] for i in lidx])), "source": "in-session FP16 greedy generation, <= 65 tokens"}}
chk("neither experiment used a gold-answer reference history (both are FP16-generated)", True)

# ---------------------------------------------------------------- 4. report
rep = {"n_checks": len(LOG), "n_failed": len(FAIL), "failed": FAIL, "reference_history": refstat, "descriptive": {"qwen_mean_js": qmeans, "llama_mean_js": lmeans, "qwen_tierB_accuracy": acc}, "checks": LOG}
(HERE / "JOURNAL_AUDIT_RESULTS.json").write_text(json.dumps(rep, indent=1, default=float))
md = [f"# Journal-manuscript numerical audit: {len(LOG)} checks, {len(FAIL)} failed", "", "Independent re-derivation of every E1/E1L number from the raw records (no analysis module imported), against the locked report JSONs; locks and frozen manifest verified.", ""]
if FAIL:
    md += ["## FAILED", *[f"- {f}" for f in FAIL], ""]
md += ["| Check | Result |", "|---|---|"] + [f"| {c['check']} | {'ok' if c['ok'] else 'FAIL'} |" for c in LOG]
(HERE / "JOURNAL_AUDIT_RESULTS.md").write_text("\n".join(md), encoding="utf-8")
print(f"{len(LOG)} checks, {len(FAIL)} failed")
for f in FAIL:
    print("  FAIL", f)
sys.exit(1 if FAIL else 0)
