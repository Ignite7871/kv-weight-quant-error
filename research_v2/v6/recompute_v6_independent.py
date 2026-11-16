"""Independent recomputation of the v6 test-phase analysis (PROTOCOL_V6.md section 7; tolerances in RECOMPUTATION_TOLERANCES_V6.md).
numpy / json / math / hashlib / argparse / pathlib only: no project import. Written from the protocol text, not from analyze_v6.py.

  python research_v2/v6/recompute_v6_independent.py [--runs <v6_runs>]   -> <v6_runs>/v6_recomputation.json (and the comparison, if
                                                                            <v6_runs>/v6_report.json exists)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RV = HERE.parent
MODELS = ["qwen2.5-0.5b", "llama3.2-1b", "qwen2.5-1.5b"]
FLAGS = ("exact_ok", "rounding_ok", "finite_ok", "routing_reset", "natural_ok", "construction_ok")
FLOOR = 1e-9
B_BOOT, B_FLIP, B_PERM = 10_000, 100_000, 100_000
S_BOOT, S_FLIP, S_PERM, S_RBOOT, S_SHARE = 6262061, 6262062, 6262063, 6262064, 6262065
V4_REPORT_SHA256 = "df57a31483160e62351378576d14123893081249e06525ed5be5f5099758dccb"
F32 = 1e-6                     # relative tolerance for quantities from float32-stored arrays
HF = Path.home() / ".cache" / "huggingface" / "hub"


def tag(*k):
    return [int.from_bytes(hashlib.sha256("|".join(map(str, k)).encode()).digest()[:4], "little")]


# ----------------------------------------------------------------------------- inputs
def read_records(p: Path):
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            assert r["index"] not in out, f"duplicate record {r['index']} in {p}"
            bad = [k for k in FLAGS if not r.get(k, False)]
            assert not bad, f"integrity flag(s) {bad} false in {p} problem {r['index']}"
            out[r["index"]] = r
    return out


def ln_m(r):
    """ln of the per-problem mean JS (PROTOCOL_V6 s2), floored at 1e-9."""
    return math.log(max(sum(r["js"]) / len(r["js"]), FLOOR))


def rope(m):
    """(head dim, per-pair angular frequencies) from the pinned config.json, git blob checked against the v4 manifest."""
    man = json.loads((RV / "v4" / "V4_MODEL_MANIFEST.json").read_text(encoding="utf-8"))
    man = man.get("models", man)[m]
    p = HF / ("models--" + man["hf_id"].replace("/", "--")) / "snapshots" / man["revision"] / "config.json"
    b = p.read_bytes()
    blob = hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()
    assert blob == man["small_files"]["config.json"]["git_blob"], f"{m}: config.json blob {blob} differs from the manifest"
    c = json.loads(b)
    hd = c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"]
    base = c["rope_theta"]
    w = np.array([base ** (-(2.0 * j) / hd) for j in range(hd // 2)])
    sc = c.get("rope_scaling")
    if sc and sc.get("rope_type", sc.get("type")) == "llama3":
        f, lo, hi, L0 = sc["factor"], sc["low_freq_factor"], sc["high_freq_factor"], sc["original_max_position_embeddings"]
        new = []
        for x in w:
            wav = 2 * math.pi / x
            if wav < L0 / hi:
                new.append(x)
            elif wav > L0 / lo:
                new.append(x / f)
            else:
                s = (L0 / wav - lo) / (hi - lo)
                new.append((1 - s) * x / f + s * x)
        w = np.array(new)
    return hd, w, blob


def dirichlet_factor(n, w):
    s = np.sin(w / 2.0)
    return np.array([1.0 if abs(si) < 1e-15 else (math.sin(n * wi / 2.0) / (n * si)) ** 2 for wi, si in zip(w, s)])


# ----------------------------------------------------------------------------- statistics (own methods)
def mean_stats(d, *k, margin=None):
    d = np.asarray(d, float)
    n = len(d)
    g = np.random.default_rng([S_BOOT] + tag(*k))
    w = g.multinomial(n, np.full(n, 1.0 / n), size=B_BOOT)
    bm = (w @ d) / n
    gf = np.random.default_rng([S_FLIP] + tag(*k))
    cnt = 0
    for c in range(0, B_FLIP, 10_000):
        s = gf.choice([-1.0, 1.0], size=(min(10_000, B_FLIP - c), n))
        cnt += int((np.abs((s * d).mean(1)) >= abs(d.mean()) - 1e-15).sum())
    return {"n": n, "estimate": float(d.mean()), "ci95": np.percentile(bm, [2.5, 97.5]).tolist(), "ci90": np.percentile(bm, [5, 95]).tolist(),
            "p": (1 + cnt) / (B_FLIP + 1), "margin": margin}


def avg_ranks(x):
    x = np.asarray(x, float)
    u, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    return (start + (cnt + 1) / 2.0)[inv]


def rho(a, b):
    ra, rb = avg_ranks(a), avg_ranks(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = math.sqrt(float((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / den) if den > 0 else 0.0


def spearman_stats(x, y, *k):
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    r0 = rho(x, y)
    rx = avg_ranks(x) - avg_ranks(x).mean()
    ry = avg_ranks(y) - avg_ranks(y).mean()
    den = math.sqrt(float((rx ** 2).sum() * (ry ** 2).sum()))
    g = np.random.default_rng([S_PERM] + tag(*k))
    cnt = 0
    for c in range(0, B_PERM, 5_000):
        P = np.argsort(g.random((min(5_000, B_PERM - c), n)), axis=1)
        rr = (ry[P] * rx).sum(1) / den
        cnt += int((np.abs(rr) >= abs(r0) - 1e-15).sum())
    gb = np.random.default_rng([S_RBOOT] + tag(*k))
    bs = [rho(x[j], y[j]) for j in gb.integers(0, n, size=(B_BOOT, n))]
    return {"rho": r0, "ci95": np.percentile(bs, [2.5, 97.5]).tolist(), "p": (1 + cnt) / (B_PERM + 1)}


def share_stats(num, den, *k):
    num, den = np.asarray(num, float), np.asarray(den, float)
    n = len(num)
    g = np.random.default_rng([S_SHARE] + tag(*k))
    w = g.multinomial(n, np.full(n, 1.0 / n), size=B_BOOT)
    bs = (w @ num) / (w @ den)
    return {"share": float(num.sum() / den.sum()), "ci95": np.percentile(bs, [2.5, 97.5]).tolist()}


def holm_adjust(ps: dict):
    names = sorted(ps, key=lambda k: (ps[k], k))
    out, run, M = {}, 0.0, len(names)
    for r, k in enumerate(names):
        run = max(run, min(1.0, (M - r) * ps[k]))
        out[k] = run
    return out


# ----------------------------------------------------------------------------- one model
def model(runs: Path, m, margin):
    d = runs / "test" / m
    A = {f.stem: read_records(f) for f in sorted(d.glob("*.jsonl")) if not f.stem.endswith("nolog")}
    idx = sorted(set.intersection(*[set(v) for v in A.values()]))

    def load(arm, i):
        with np.load(d / "arrays" / f"{arm}_{i}.npz") as z:
            return {k: np.array(z[k]) for k in z.files}

    Z = {a: {i: load(a, i) for i in idx} for a in A}
    th = lambda a, b: [ln_m(A[a][i]) - ln_m(A[b][i]) for i in idx]  # noqa: E731
    tot = lambda a, i, k: float(np.nansum(Z[a][i][k]))  # noqa: E731
    R = {"n": len(idx), "arms": sorted(A)}

    tV, tcm, tdir = np.array(th("NAT-V", "C1-V")), np.array(th("NAT-V", "NAT-V-cen")), np.array(th("NAT-V-cen", "C1-V"))
    R["L3"] = {"theta_V": mean_stats(tV, m, "L3V"), "theta_cm": mean_stats(tcm, m, "L3cm"), "theta_dir": mean_stats(tdir, m, "L3dir"),
               "share_cm": share_stats(tcm, tV, m, "L3share"), "identity_max_abs": float(np.max(np.abs(tV - tcm - tdir)))}
    a4 = np.array([math.log(tot("NAT-V", i, "meas") / tot("NAT-V-cen", i, "meas")) for i in idx])
    b4 = np.array([math.log(tot("NAT-V-cen", i, "meas") / tot("C1-V", i, "meas")) for i in idx])
    R["L4"] = {"cm": mean_stats(a4, m, "L4cm"), "dir": mean_stats(b4, m, "L4dir"), "share_cm": share_stats(a4, a4 + b4, m, "L4share")}

    def abs_ln(arms, pred):
        v = []
        for a in arms:
            for i in idx:
                me, pr = Z[a][i]["meas"].ravel().astype(np.float64), Z[a][i][pred].ravel().astype(np.float64)
                k = np.isfinite(me) & np.isfinite(pr) & (me > 0) & (pr > 0)
                v.append(np.abs(np.log(pr[k]) - np.log(me[k])))
        v = np.concatenate(v)
        return {"median_abs_ln": float(np.median(v)), "n_slot_steps": int(v.size)}
    v_arms = [a for a in A if "-V" in a]          # PROTOCOL_V6 s4: the value-path arms are those named ...-V...
    k_arms = [a for a in A if a not in v_arms]
    R["L1"] = {"V_mod_vs_meas": abs_ln(v_arms, "mod_V"), "K_lin_vs_meas": abs_ln(k_arms, "lin_K"), "V_arms": sorted(v_arms), "K_arms": sorted(k_arms)}

    R["L2"] = spearman_stats([math.log(tot("NAT-V", i, "meas") / tot("C1-V", i, "meas")) for i in idx], tV, m, "L2")
    tK = np.array(th("NAT-K", "C1-K"))
    R["L6"] = {"spearman": spearman_stats([math.log(tot("NAT-K", i, "lin_K") / tot("C1-K", i, "lin_K")) for i in idx], tK, m, "L6"),
               "model_ln_ratio_linK": math.log(sum(tot("NAT-K", i, "lin_K") for i in idx) / sum(tot("C1-K", i, "lin_K") for i in idx)),
               "theta_H1": mean_stats(tK, m, "H1")}

    hd, w, blob = rope(m)
    h = hd // 2
    vals = []
    for i in idx:
        z = Z["NAT-K"][i]
        n = z["normsK"].shape[0]
        Ap = dirichlet_factor(n, w)
        cpost = ((z["S1K"][..., :h] / n) ** 2 + (z["S1K"][..., h:] / n) ** 2) / (z["S2K_pair"] / n)
        cpre = ((z["S1K_pre"][..., :h] / n) ** 2 + (z["S1K_pre"][..., h:] / n) ** 2) / (z["S2K_pre_pair"] / n)
        pred = Ap * cpre + (1.0 - cpre) / n
        Ab = np.broadcast_to(Ap, cpost.shape)
        k = (Ab >= 0.05) & (z["S2K_pair"] > 0) & (pred > 0) & (cpost > 0) & np.isfinite(cpost) & np.isfinite(pred)
        vals.append(np.abs(np.log(cpost[k]) - np.log(pred[k])))
    vals = np.concatenate(vals)
    R["L5"] = {"median_abs_ln_ratio": float(np.median(vals)), "n": int(vals.size), "head_dim": hd, "config_blob": blob}

    nC = []
    for i in idx:
        z = Z["ST-V"][i]
        n = z["normsV"].shape[0]
        nC.append(n * float(((z["S1V"] / n) ** 2).sum() / ((z["normsV"] ** 2).sum(0) / n).sum()))
    rK = float(np.median([np.median(A["ST-K"][i]["stored_rel_K"]) for i in idx]))
    rV = float(np.median([np.median(A["ST-V"][i]["stored_rel_V"]) for i in idx]))
    R["L7"] = {"i_nC_V_ST": float(np.mean(nC)), "ii_K": mean_stats(th("ST-K@nat", "C1-K"), m, "L7iiK", margin=margin),
               "ii_V": mean_stats(th("ST-V@nat", "C1-V"), m, "L7iiV", margin=margin), "iii_rho_ratio": rK / rV,
               "iii_theta": mean_stats(th("ST-K", "ST-V"), m, "L7iii"), "H3": mean_stats(th("NAT-K", "NAT-V"), m, "H3"),
               "H2": mean_stats(tV, m, "H2"), "margin": margin}

    ml = lambda a, i, k: float(np.nanmean(np.log(Z[a][i][k])))  # noqa: E731
    R["L8a"] = mean_stats([ml("NAT-V", i, "A_cov") - ml("C1-V", i, "A_cov") for i in idx], m, "L8a")
    R["L8b"] = mean_stats([ml("NAT-V", i, "A_o") - ml("C1-V", i, "A_o") for i in idx], m, "L8b")
    R["L9a"] = mean_stats(th("NAT-V-cen", "ST-V@nat-cen"), m, "L9a")
    R["L9b"] = mean_stats([ml("NAT-V", i, "A_cov") - ml("ST-V@nat", i, "A_cov") for i in idx], m, "L9b")
    return R


def decide(R):
    ps = {}
    for m, r in R.items():
        ps[f"L2:{m}"], ps[f"L6:{m}"], ps[f"L7iii:{m}"] = r["L2"]["p"], r["L6"]["spearman"]["p"], r["L7"]["iii_theta"]["p"]
        for t in ("L8a", "L8b", "L9a", "L9b"):
            ps[f"{t}:{m}"] = r[t]["p"]
    assert len(ps) == 21
    H = holm_adjust(ps)
    psig = lambda ci, name: ci[0] > 0 and H[name] < 0.05  # noqa: E731
    below = lambda ci: ci[1] < 0  # noqa: E731
    V = {}
    l1 = [(r["L1"]["V_mod_vs_meas"]["median_abs_ln"], r["L1"]["K_lin_vs_meas"]["median_abs_ln"]) for r in R.values()]
    V["L1"] = "agreement" if all(a <= math.log(1.25) and b <= math.log(1.5) for a, b in l1) else ("disagreement" if any(max(a, b) > math.log(2) for a, b in l1) else "neither")
    V["L2"] = "agreement" if all(psig(r["L2"]["ci95"], f"L2:{m}") for m, r in R.items()) else ("disagreement" if any(r["L2"]["ci95"][1] <= 0 for r in R.values()) else "neither")
    l5 = [r["L5"]["median_abs_ln_ratio"] for r in R.values()]
    V["L5"] = "agreement" if all(x <= math.log(1.5) for x in l5) else ("disagreement" if any(x > math.log(2) for x in l5) else "neither")
    same = all((r["L6"]["model_ln_ratio_linK"] > 0) == (r["L6"]["theta_H1"]["estimate"] > 0) for r in R.values())
    nsig = sum(psig(r["L6"]["spearman"]["ci95"], f"L6:{m}") for m, r in R.items())
    V["L6"] = "disagreement" if not same else ("agreement" if nsig >= 2 else "neither")
    i_ = all(r["L7"]["i_nC_V_ST"] <= 2 for r in R.values())
    inside = lambda s: -s["margin"] < s["ci90"][0] and s["ci90"][1] < s["margin"]  # noqa: E731
    ii = all(inside(r["L7"]["ii_K"]) and inside(r["L7"]["ii_V"]) for r in R.values())
    iii = all(r["L7"]["iii_rho_ratio"] >= 1 and psig(r["L7"]["iii_theta"]["ci95"], f"L7iii:{m}") and below(r["L7"]["H3"]["ci95"]) for m, r in R.items())
    dii = any((r["L7"]["ii_V"]["ci95"][0] > 0 or r["L7"]["ii_V"]["ci95"][1] < 0) and abs(r["L7"]["ii_V"]["estimate"]) >= r["L7"]["H2"]["estimate"] for r in R.values())
    diii = any(r["L7"]["iii_rho_ratio"] < 1 and below(r["L7"]["iii_theta"]["ci95"]) for r in R.values())
    V["L7"] = {"i": i_, "ii": ii, "iii": iii, "verdict": "agreement" if (i_ and ii and iii) else ("disagreement" if (dii or diii) else "neither")}
    for t in ("L8a", "L8b"):
        V[t] = "agreement" if all(psig(r[t]["ci95"], f"{t}:{m}") for m, r in R.items()) else ("disagreement" if any(below(r[t]["ci95"]) for r in R.values()) else "neither")
    for t in ("L9a", "L9b"):
        dis = any(below(r[t]["ci95"]) for r in R.values()) or sum(r[t]["estimate"] <= 0 for r in R.values()) >= 2
        V[t] = "disagreement" if dis else ("agreement" if sum(psig(r[t]["ci95"], f"{t}:{m}") for m, r in R.items()) >= 2 else "neither")
    return V, H


# ----------------------------------------------------------------------------- comparison with the locked report
def compare(loc, R, V, H):
    rows = []

    def det(what, a, b, tol=1e-9):          # 1e-6 for quantities from the float32-stored read-time arrays (tolerances file)
        rows.append({"what": what, "locked": a, "recomputed": b, "ok": abs(a - b) <= tol * max(1.0, abs(a))})

    def ci(what, a, b):
        tol = max(0.05 * abs(a[1] - a[0]), 1e-9)
        rows.append({"what": what, "locked": a, "recomputed": b, "ok": all(abs(x - y) <= tol for x, y in zip(a, b))})

    def pv(what, a, b):
        rows.append({"what": what, "locked": a, "recomputed": b, "ok": abs(a - b) <= 0.01 + 0.1 * a})

    def mean_(what, a, b, ci90=False, tol=1e-9):
        det(what + " estimate", a["estimate"], b["estimate"], tol)
        det(what + " n", a["n"], b["n"])
        ci(what + " ci95", a["ci95"], b["ci95"])
        if ci90:
            ci(what + " ci90", a["ci90"], b["ci90"])
        pv(what + " p", a["p"], b["p"])

    for m, r in R.items():
        L = loc["per_model"][m]
        det(f"{m} n", L["n"], r["n"])
        for k, k2 in (("theta_V", "theta_V"), ("theta_cm", "theta_cm"), ("theta_dir", "theta_dir")):
            mean_(f"{m} L3 {k}", L["L3"][k], r["L3"][k2])
        det(f"{m} L3 share", L["L3"]["share_cm"]["share"], r["L3"]["share_cm"]["share"])
        ci(f"{m} L3 share ci95", L["L3"]["share_cm"]["ci95"], r["L3"]["share_cm"]["ci95"])
        for k in ("cm", "dir"):
            mean_(f"{m} L4 {k}", L["L4"][k], r["L4"][k], tol=F32)
        det(f"{m} L4 share", L["L4"]["share_cm"]["share"], r["L4"]["share_cm"]["share"], F32)
        ci(f"{m} L4 share ci95", L["L4"]["share_cm"]["ci95"], r["L4"]["share_cm"]["ci95"])
        for k in ("V_mod_vs_meas", "K_lin_vs_meas"):
            det(f"{m} L1 {k}", L["L1"][k]["median_abs_ln"], r["L1"][k]["median_abs_ln"], F32)
            det(f"{m} L1 {k} n", L["L1"][k]["n_slot_steps"], r["L1"][k]["n_slot_steps"])
        for name, a, b in (("L2", L["L2"], r["L2"]), ("L6", L["L6"]["spearman"], r["L6"]["spearman"])):
            det(f"{m} {name} rho", a["rho"], b["rho"], F32)
            ci(f"{m} {name} rho ci95", a["ci95"], b["ci95"])
            pv(f"{m} {name} p", a["p"], b["p"])
        det(f"{m} L6 model ln ratio", L["L6"]["model_ln_ratio_linK"], r["L6"]["model_ln_ratio_linK"], F32)
        mean_(f"{m} L6 H1", L["L6"]["theta_H1"], r["L6"]["theta_H1"])
        det(f"{m} L5 median", L["L5"]["median_abs_ln_ratio"], r["L5"]["median_abs_ln_ratio"])
        det(f"{m} L5 n", L["L5"]["n"], r["L5"]["n"])
        det(f"{m} L7 i", L["L7"]["i_nC_V_ST"], r["L7"]["i_nC_V_ST"])
        mean_(f"{m} L7 ii K", L["L7"]["ii_STK@nat_vs_C1K"], r["L7"]["ii_K"], ci90=True)
        mean_(f"{m} L7 ii V", L["L7"]["ii_STV@nat_vs_C1V"], r["L7"]["ii_V"], ci90=True)
        det(f"{m} L7 iii ratio", L["L7"]["iii_rho_ratio_K_over_V"], r["L7"]["iii_rho_ratio"])
        mean_(f"{m} L7 iii theta", L["L7"]["iii_STK_vs_STV"], r["L7"]["iii_theta"])
        mean_(f"{m} H3", L["L7"]["H3_replication"], r["L7"]["H3"])
        for t in ("L8a", "L8b", "L9a", "L9b"):
            mean_(f"{m} {t}", L[t], r[t], tol=(1e-9 if t == "L9a" else F32))
    for k, a in loc["holm_p"].items():
        pv(f"Holm {k}", a, H[k])
    for t, a in loc["verdicts"].items():
        b = V[t]
        same = (a == b) if not isinstance(a, dict) else all(a[x] == b[x] for x in ("i", "ii", "iii", "verdict"))
        rows.append({"what": f"verdict {t}", "locked": a, "recomputed": b, "ok": same, "decision": True})
    bad = [x for x in rows if not x["ok"]]
    return {"n_compared": len(rows), "n_outside_tolerance": len(bad), "outside_tolerance": bad,
            "verdict_disagreements": [x for x in bad if x.get("decision")], "rows": rows}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(RV / "v6_runs"))
    a = ap.parse_args(argv)
    runs = Path(a.runs)
    rep4 = RV / "v4_runs" / "v4_report.json"
    assert hashlib.sha256(rep4.read_bytes()).hexdigest() == V4_REPORT_SHA256, "v4_report.json SHA-256 differs"
    margins = {m: json.loads(rep4.read_text(encoding="utf-8"))["per_model"][m]["margin"] for m in MODELS}
    R = {m: model(runs, m, margins[m]) for m in MODELS}
    V, H = decide(R)
    out = {"what": "v6 independent recomputation (RECOMPUTATION_TOLERANCES_V6.md)", "per_model": R, "verdicts": V, "holm_p": H}
    if (runs / "v6_report.json").exists():
        loc = json.loads((runs / "v6_report.json").read_text(encoding="utf-8"))
        out["comparison"] = compare(loc, R, V, H)
        c = out["comparison"]
        print(f"compared {c['n_compared']}; outside tolerance {c['n_outside_tolerance']}; verdict disagreements {len(c['verdict_disagreements'])}")
    (runs / "v6_recomputation.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(V, indent=1))


if __name__ == "__main__":
    main()
