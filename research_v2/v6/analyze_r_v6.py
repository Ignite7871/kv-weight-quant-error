"""v6 R block (R1-R4): pre-specified secondary analyses of previously seen v4/v5 records (R_ANALYSIS_PLAN_V6.md).

  python research_v2/v6/analyze_r_v6.py            -> v6/results/R_block_report.json and .md
  env V6_V4_RUNS / V6_V5_RUNS / V6_OUT override the record roots and the output folder (used by the synthetic test)

CPU only, read-only on the records. Locked primitives (analyze_v3, analyze_x13) are imported unchanged.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RV = HERE.parent
sys.path.insert(0, str(RV))
sys.path.insert(0, str(RV / "v3" / "phase8"))
import analyze_v3 as L        # noqa: E402
import analyze_x13 as X13     # noqa: E402

V4 = Path(os.environ.get("V6_V4_RUNS", RV / "v4_runs"))
V5 = Path(os.environ.get("V6_V5_RUNS", RV / "v5_runs"))
OUT = Path(os.environ.get("V6_OUT", HERE / "results"))
SEED_R, SEED_LOCKED, N_NULL, B = 20261114, 20261105, 200, 10_000
FLOOR = 1e-9

# unit -> (records root, head_dim, rope theta, llama3 scaling or None)
L3_32 = {"factor": 32.0, "low": 1.0, "high": 4.0, "orig": 8192}
L3_8 = {"factor": 8.0, "low": 1.0, "high": 4.0, "orig": 8192}
UNITS = {
    "qwen2.5-0.5b": (V4, 64, 1e6, None), "qwen2.5-1.5b": (V4, 128, 1e6, None), "qwen2.5-3b": (V4, 128, 1e6, None),
    "qwen2.5-7b": (V4, 128, 1e6, None), "llama3.2-1b": (V4, 64, 5e5, L3_32), "llama3.2-3b": (V4, 128, 5e5, L3_32),
    "llama3.1-8b": (V4, 128, 5e5, L3_8), "mistral-7b-v0.3": (V4, 128, 1e6, None),
    "U1": (V5, 128, 1e6, None), "U2": (V5, 64, 1.3e5, None),
}
ARMS = ("NAT-K", "C1-K", "NAT-V", "C1-V")


def key_int(k) -> int:
    return k if isinstance(k, int) else int.from_bytes(hashlib.sha256(str(k).encode()).digest()[:4], "big")


def rng(*keys):
    return np.random.default_rng([SEED_R] + [key_int(k) for k in keys])


def inv_freq(hd, theta, sc):
    f = theta ** (-np.arange(0, hd, 2, dtype=np.float64) / hd)
    if sc is None:
        return f
    lo_wl, hi_wl = sc["orig"] / sc["low"], sc["orig"] / sc["high"]
    wl = 2 * math.pi / f
    out = np.where(wl > lo_wl, f / sc["factor"], f)
    smooth = (sc["orig"] / wl - sc["low"]) / (sc["high"] - sc["low"])
    mid = (wl <= lo_wl) & (wl >= hi_wl)
    return np.where(mid, (1 - smooth) * f / sc["factor"] + smooth * f, out)


def dirichlet(n, w):
    s = np.sin(w / 2)
    return np.where(np.abs(s) < 1e-15, 1.0, (np.sin(n * w / 2) / (n * np.where(np.abs(s) < 1e-15, 1.0, s))) ** 2)


def ranks(x):
    x = np.asarray(x, float)
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[o] = np.arange(1, len(x) + 1)
    for v in np.unique(x):                       # average ranks for ties
        m = x == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    d = math.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / d) if d > 0 else 0.0


def perm_p(a, b, g, B_=B):
    obs = abs(spearman(a, b))
    cnt = sum(abs(spearman(a, g.permutation(b))) >= obs - 1e-15 for _ in range(B_))
    return (1 + cnt) / (B_ + 1)


def norm_p(z):
    return math.erfc(abs(z) / math.sqrt(2))


def holm(ps: dict) -> dict:
    items = sorted(ps.items(), key=lambda kv: kv[1])
    m, run, out = len(items), 0.0, {}
    for k, (name, p) in enumerate(items):
        run = max(run, min(1.0, (m - k) * p))
        out[name] = run
    return out


def load_unit(u):
    root = UNITS[u][0] / u / "test"
    A = {a: L.load_jsonl(root / f"{a}.jsonl") for a in ARMS}
    idx = sorted(set.intersection(*[set(v) for v in A.values()]))
    idx = [i for i in idx if (root / "slots" / f"NAT_{i}.npz").exists()]
    return A, idx, root / "slots"


def r1_r2(A, idx, u):
    slopes, r2, short = [], [], 0
    for i in idx:
        a, b = A["NAT-V"][i], A["C1-V"][i]
        S = min(a["S"], b["S"])
        if S < 4:
            short += 1
            continue
        t = np.arange(2, S + 1, dtype=float)
        r = np.log(np.maximum(np.asarray(a["js"][1:S], float), FLOOR)) - np.log(np.maximum(np.asarray(b["js"][1:S], float), FLOOR))
        x = np.log(t) - np.log(t).mean()
        slopes.append(float((x * (r - r.mean())).sum() / (x * x).sum()))
        r2.append(float(r[0]))
    return (L.summarize(slopes, rng(u, "R1"), B=B), L.summarize(r2, rng(u, "R2"), B=B), short)


def r3_unit(A, idx, slots, u, hd):
    g = np.random.default_rng([SEED_LOCKED, 13, key_int(u), key_int("V")])
    xs, th = [], []
    for i in idx:
        with np.load(slots / f"NAT_{i}.npz") as z:
            norms, S1 = z["normsV"], z["S1V"]
        n = norms.shape[0]
        xs.append(math.log(n * X13.cm_share(norms, S1)) - X13.null_lognc(norms, g, N_NULL, hd))
        th.append(L.logm(A["NAT-V"][i]) - L.logm(A["C1-V"][i]))
    rho = spearman(xs, th)
    return {"n": len(xs), "mean_x": float(np.mean(xs)), "rho": rho, "z": math.atanh(max(min(rho, 0.999999), -0.999999)), "var": 1.06 / (len(xs) - 3)}


def r4_unit(idx, slots, u, hd, theta, sc):
    w, w0 = inv_freq(hd, theta, sc), inv_freq(hd, theta, None)
    h = hd // 2
    EK, EV, Abar, Abar0 = np.zeros(h), np.zeros(h), np.zeros(h), np.zeros(h)
    for i in idx:
        with np.load(slots / f"NAT_{i}.npz") as z:
            n = z["normsK"].shape[0]
            K, V = z["S1K"] / n, z["S1V"] / n
        EK += (K[..., :h] ** 2 + K[..., h:] ** 2).reshape(-1, h).sum(0)
        EV += (V[..., :h] ** 2 + V[..., h:] ** 2).reshape(-1, h).sum(0)
        Abar += dirichlet(n, w)
        Abar0 += dirichlet(n, w0)
    Abar, Abar0 = Abar / len(idx), Abar0 / len(idx)
    sK, sV = EK / EK.sum(), EV / EV.sum()
    g = rng(u, "R4")
    return {"pairs": h, "rho_K": spearman(sK, Abar), "rho_V_placebo": spearman(sV, Abar), "p_K": perm_p(sK, Abar, g),
            "max_abs_dA_without_scaling": float(np.abs(Abar - Abar0).max()),
            "share_K_in_pairs_Abar_lt_0.05": float(sK[Abar < 0.05].sum()), "share_V_in_same_pairs": float(sV[Abar < 0.05].sum()),
            "n_pairs_Abar_lt_0.05": int((Abar < 0.05).sum())}


def run():
    rep = {"what": "v6 R block: pre-specified secondary analyses of previously seen data (R_ANALYSIS_PLAN_V6.md)", "units": {}}
    for u, (root, hd, theta, sc) in UNITS.items():
        A, idx, slots = load_unit(u)
        s1, s2, short = r1_r2(A, idx, u)
        rep["units"][u] = {"n": len(idx), "head_dim": hd, "R1": s1, "R1_dropped_short": short, "R2": s2,
                           "R3": r3_unit(A, idx, slots, u, hd), "R4": r4_unit(idx, slots, u, hd, theta, sc)}
        print(u, "done", flush=True)
    U = rep["units"]
    lr = V4 / "v4_report.json"                    # self-check: the locked L^V of the v4 units must be reproduced
    if lr.exists():
        locked = json.loads(lr.read_text(encoding="utf-8"))["per_model"]
        rep["selfcheck_LV_vs_locked"] = {u: [U[u]["R3"]["mean_x"], locked[u]["H5"]["V"]["estimate"]] for u in U if u in locked}
    w = np.array([1 / U[u]["R3"]["var"] for u in U])
    z = np.array([U[u]["R3"]["z"] for u in U])
    zp, se = float((w * z).sum() / w.sum()), float(1 / math.sqrt(w.sum()))
    pooled = {"rho": math.tanh(zp), "ci95": [math.tanh(zp - 1.96 * se), math.tanh(zp + 1.96 * se)], "p": norm_p(zp / se)}
    ps = {f"R1:{u}": U[u]["R1"]["p"] for u in U}
    ps["R3:pooled"] = pooled["p"]
    ps.update({f"R4:{u}": U[u]["R4"]["p_K"] for u in U})
    hp = holm(ps)
    for u in U:
        U[u]["R1"]["holm_p"], U[u]["R4"]["holm_p"] = hp[f"R1:{u}"], hp[f"R4:{u}"]
    pooled["holm_p"] = hp["R3:pooled"]
    rep["R3_pooled"] = pooled
    # verdicts (criteria from THEORY_PLAN_V6 section 7, operationalized in R_ANALYSIS_PLAN_V6)
    sig_pos = [u for u in U if U[u]["R1"]["ci95"][0] > 0 and U[u]["R1"]["holm_p"] < 0.05]
    neg = [u for u in U if U[u]["R1"]["ci95"][1] < 0]
    pos_pt = [u for u in U if U[u]["R1"]["estimate"] > 0]
    v1 = "disagreement" if (neg or len(pos_pt) < 5) else ("agreement" if len(sig_pos) >= 8 else "neither")
    r3pos = [u for u in U if U[u]["R3"]["rho"] > 0]
    if pooled["ci95"][1] <= 0:
        v3 = "disagreement"
    elif pooled["ci95"][0] > 0 and pooled["holm_p"] < 0.05 and len(r3pos) >= 7:
        v3 = "agreement"
    else:
        v3 = "neither"
    k5 = [u for u in U if U[u]["R4"]["rho_K"] > 0.5]
    kv = [u for u in U if U[u]["R4"]["rho_K"] > U[u]["R4"]["rho_V_placebo"]]
    kneg = [u for u in U if U[u]["R4"]["rho_K"] <= 0]
    v4 = "disagreement" if len(kneg) >= 5 else ("agreement" if len(k5) >= 8 and len(kv) >= 8 else "neither")
    rep["verdicts"] = {"R1": {"verdict": v1, "positive_significant": sig_pos, "ci_entirely_negative": neg, "positive_point": pos_pt},
                       "R3": {"verdict": v3, "units_rho_positive": r3pos},
                       "R4": {"verdict": v4, "rho_K_gt_0.5": k5, "rho_K_gt_placebo": kv, "rho_K_le_0": kneg}}
    rep["L_block_gate"] = "do not proceed (R1 or R3 disagreement)" if "disagreement" in (v1, v3) else "not contradicted by R1/R3"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "R_block_report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    return rep


if __name__ == "__main__":
    r = run()
    print(json.dumps(r["verdicts"], indent=1))
    print(r["L_block_gate"])
