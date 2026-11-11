"""X9 analysis (v3/phase7/FOLLOWUP_X9.md section 8). EXPLORATORY. CPU only; reads v3_runs_x9 (X9 records) and v3_runs (locked records, read only).

  venv/Scripts/python.exe research_v2/v3/phase7/analyze_x9.py      -> v3/phase7/x9_results.json, X9_RESULTS.md

Only models with x9_complete.json and no REPRODUCTION_MISMATCH.json are analysed; the others are listed with their status.
(a) and (b) are exploratory tests with the pre-stated rules; (c) and (d) are descriptive.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

P7 = Path(__file__).resolve().parent
RV = P7.parents[1]
sys.path.insert(0, str(RV))
import analyze_v3 as L                          # noqa: E402  (locked primitives, imported unchanged)

OUT = RV / "v3_runs_x9"
LOCKED_RUNS = RV / "v3_runs"
ORDER = ["qwen-base", "qwen-instruct", "llama-base", "llama-instruct"]
SEED = 20261211
ISO = 8 / 64
LABEL = "EXPLORATORY (FOLLOWUP_X9.md): (a), (b) exploratory tests; (c), (d) descriptive"
L0NAT = "NAT-Kstar-L0nat"


def family(m):
    return m.split("-")[0]


# ----------------------------------------------------------------------------- loading
def load_jsonl(p: Path):
    return L.load_jsonl(p)


def load_slots(m, tag, ids):
    out = {}
    for i in ids:
        with np.load(OUT / m / "slots" / f"{tag}_{i}.npz") as z:
            out[i] = {k: z[k] for k in z.files}
    return out


# ----------------------------------------------------------------------------- (a) query-subspace share of the natural key error
def a_fractions(slots: dict, key_e="eK", key_p="pK"):
    """slots: idx -> {eK, pK, ...} arrays [S, L, n_kv]. Returns F_i (all layers), F_i without layer 0, and per-problem per-layer sums."""
    ids = sorted(slots)
    F = np.array([slots[i][key_p].sum() / slots[i][key_e].sum() for i in ids])
    F0 = np.array([slots[i][key_p][:, 1:].sum() / slots[i][key_e][:, 1:].sum() for i in ids])
    num = np.stack([slots[i][key_p].sum(0) for i in ids])          # [n, L, n_kv]
    den = np.stack([slots[i][key_e].sum(0) for i in ids])
    return ids, F, F0, num, den


def ratio_ci(num, den, rng):
    """Pooled ratio sum(num) / sum(den) over problems (axis 0), percentile CI from the problem bootstrap. num, den: [n, ...]."""
    est = num.sum(0) / den.sum(0)
    bi = L.boot_idx(num.shape[0], rng)
    b = np.stack([num[k].sum(0) / den[k].sum(0) for k in bi])
    return est, np.percentile(b, 2.5, axis=0), np.percentile(b, 97.5, axis=0)


def verdict_a(s, holm_p):
    if s["ci95"][1] < 0 and holm_p < 0.05:
        return "supported"
    if s["ci95"][0] > 0:
        return "contradicted"
    return "inconclusive"


# ----------------------------------------------------------------------------- (b) share of the NAT-K* excess removed by layer 0 natural
def share_G(l_kstar, l_l0, l_k, rng):
    """G = mean(l_K* - l_L0) / mean(l_K* - l_K) over problems (log per-problem mean JS); problem-bootstrap percentile CI of the ratio."""
    num, den = np.asarray(l_kstar) - np.asarray(l_l0), np.asarray(l_kstar) - np.asarray(l_k)
    G = float(num.mean() / den.mean())
    bi = L.boot_idx(len(num), rng)
    b = num[bi].mean(1) / den[bi].mean(1)
    return G, [float(x) for x in np.nanpercentile(b, [2.5, 97.5])]


def verdict_b(G, ci):
    if G >= 0.5 and ci[0] > 0.2:
        return "supported"
    if ci[1] < 0.2:
        return "contradicted"
    return "inconclusive"


# ----------------------------------------------------------------------------- (c) H-A without layer 0 (descriptive)
def no_l0(rec, key):
    """Per step: the layer-summed predictor over layers 1..L-1 (sum of PL_x) or, for P_norm, the mean of PL_norm over layers 1..L-1."""
    rows = rec[key]
    if key == "PL_norm":
        return [float(np.mean(r[1:])) for r in rows]
    return [float(np.sum(r[1:])) for r in rows]


def rho_series(recs: dict, series):
    """idx -> Spearman over steps 2..S of (series(rec), JS); H-A exclusions (S < 8, undefined Spearman)."""
    out = {}
    for i, r in recs.items():
        if r["S"] < L.MIN_STEPS_HA:
            continue
        v = L.spearman(series(r)[1:], r["js"][1:])
        if v == v:
            out[i] = v
    return out


def c_block(nat: dict, rng):
    full = {x: rho_series(nat, lambda r, x=x: r[x]) for x in ("P_logit", "P_val", "P_anormK", "P_norm")}
    drop = {x: rho_series(nat, lambda r, k=k: no_l0(r, k)) for x, k in (("P_logit", "PL_logit"), ("P_val", "PL_val"), ("P_anormK", "PL_anormK"),
                                                                       ("P_norm", "PL_norm"))}
    out = {}
    for x in ("P_logit", "P_val", "P_anormK"):
        ids = sorted(set(full[x]) & set(drop[x]) & set(full["P_norm"]) & set(drop["P_norm"]))
        dr = [drop[x][i] - full[x][i] for i in ids]
        D = [full[x][i] - full["P_norm"][i] for i in ids]
        D0 = [drop[x][i] - drop["P_norm"][i] for i in ids]
        out[x] = {"n": len(ids), "rho_full": _ci(np.array([full[x][i] for i in ids]), rng), "rho_no_l0": _ci(np.array([drop[x][i] for i in ids]), rng),
                  "delta_rho": _ci(np.array(dr), rng), "D": _ci(np.array(D), rng), "D_no_l0": _ci(np.array(D0), rng)}
    n_layers = len(next(iter(nat.values()))["PL_logit"][0])
    prof = []
    for l in range(n_layers):
        r = rho_series(nat, lambda rec, l=l: [row[l] for row in rec["PL_logit"]])
        prof.append(_ci(np.array(list(r.values())), rng) if r else None)
    out["per_layer_rho_P_logit"] = prof
    return out


def _ci(v, rng):
    if len(v) == 0:
        return None
    b = v[L.boot_idx(len(v), rng)].mean(1)
    return {"mean": float(v.mean()), "ci95": [float(x) for x in np.percentile(b, [2.5, 97.5])], "n": int(len(v))}


# ----------------------------------------------------------------------------- (d) effective NAT-K* scale per layer (descriptive)
def eff_scale(slots: dict, rho_star):
    """Per layer: median over (problems, steps, KV heads) of rho* / rho_slot, rho_slot = sqrt(eK) / fK; slots with eK = 0 are counted, not used."""
    eK = np.concatenate([slots[i]["eK"] for i in sorted(slots)])     # [N, L, n_kv]
    fK = np.concatenate([slots[i]["fK"] for i in sorted(slots)])
    out = []
    for l in range(eK.shape[1]):
        e, f = eK[:, l].ravel(), fK[:, l].ravel()
        ok = (e > 0) & (f > 0)
        r = rho_star / (np.sqrt(e[ok]) / f[ok])
        out.append({"median": float(np.median(r)) if r.size else None, "p90": float(np.percentile(r, 90)) if r.size else None,
                    "n_zero_error_slots": int((~ok).sum())})
    return out


# ----------------------------------------------------------------------------- per model
def status(m):
    if (OUT / m / "REPRODUCTION_MISMATCH.json").exists():
        return "reproduction mismatch (not analysed)"
    if not (OUT / m / "x9_complete.json").exists():
        return "not complete (not analysed)"
    return "complete"


def analyse_model(m, rng):
    meta = json.loads((OUT / m / "x9_complete.json").read_text())
    cal = json.loads((LOCKED_RUNS / m / "calibration.json").read_text())
    nat = load_jsonl(OUT / m / "NAT.jsonl")
    ids = sorted(nat)
    res = {"n": len(ids), "arms": meta["arms"]}
    s_nat = load_slots(m, "NAT", ids)
    _, F, F0, num, den = a_fractions(s_nat)
    res["a"] = {"test": L.summarize(np.log(F / ISO), rng), "F_mean": float(F.mean()), "F_no_l0_mean": float(F0.mean()),
                "A_no_l0": L.summarize(np.log(F0 / ISO), rng)}
    est, lo, hi = ratio_ci(num.sum(2), den.sum(2), rng)
    res["a"]["F_per_layer"] = [{"F": float(e), "ci95": [float(a), float(b)]} for e, a, b in zip(est, lo, hi)]
    res["a"]["F_per_slot"] = (num.sum(0) / den.sum(0)).tolist()
    _, Fv, F0v, numv, denv = a_fractions(s_nat, "eV", "pV")
    estv, lov, hiv = ratio_ci(numv.sum(2), denv.sum(2), rng)
    res["a"]["V_descriptive"] = {"A": L.summarize(np.log(Fv / ISO), rng), "F_per_layer": [{"F": float(e), "ci95": [float(a), float(b)]}
                                                                                          for e, a, b in zip(estv, lov, hiv)]}
    lk = load_jsonl(LOCKED_RUNS / m / "tierA" / "NAT-K.jsonl")
    lc = load_jsonl(LOCKED_RUNS / m / "tierA" / "C1-K.jsonl")
    th = np.array([L.logm(lk[i]) - L.logm(lc[i]) for i in ids])
    bi = L.boot_idx(len(ids), rng)
    res["a"]["spearman_lnF_vs_locked_theta_NATK_C1K"] = {"rho": L.spearman(np.log(F), th),
                                                         "ci95": [float(x) for x in np.nanpercentile([L.spearman(np.log(F)[k], th[k]) for k in bi], [2.5, 97.5])]}
    ks = load_jsonl(OUT / m / "NAT-Kstar.jsonl")
    lks, lkk = np.array([L.logm(ks[i]) for i in ids]), np.array([L.logm(lk[i]) for i in ids])
    bench = cal["dose_slope_b_hat"] * math.log(cal["scale_factor_K"])
    sk = L.summarize(lks - lkk, rng)
    res["b"] = {"benchmark_b_ln_sK": bench, "E_Kstar": sk["estimate"] - bench, "E_Kstar_ci95": [x - bench for x in sk["ci95"]]}
    if (OUT / m / f"{L0NAT}.jsonl").exists() and L0NAT in meta["arms"]:
        l0 = load_jsonl(OUT / m / f"{L0NAT}.jsonl")
        ll0 = np.array([L.logm(l0[i]) for i in ids])
        G, ci = share_G(lks, ll0, lkk, rng)
        s0 = L.summarize(ll0 - lkk, rng)
        e0 = [x - bench for x in s0["ci95"]]
        res["b"].update({"G": G, "G_ci95": ci, "E_L0": s0["estimate"] - bench, "E_L0_ci95": e0,
                         "E_L0_within_factor_2": bool(-math.log(2) < e0[0] and e0[1] < math.log(2)),
                         "role": "test" if family(m) == "qwen" else "control (descriptive)"})
        res["b"]["verdict"] = verdict_b(G, ci) if family(m) == "qwen" else "descriptive"
    else:
        res["b"]["G"] = None
        res["b"]["role"] = "NAT-K*-L0nat not run (pre-declared reduction)"
    res["c"] = c_block(nat, rng)
    res["d"] = {"s_K": cal["scale_factor_K"], "rho_star": cal["rho_star"], "per_layer": eff_scale(s_nat, cal["rho_star"])}
    return res


def analyse():
    rng = np.random.default_rng(SEED)
    out = {"label": LABEL, "status": {m: status(m) for m in ORDER}, "per_model": {}}
    for m in ORDER:
        if out["status"][m] == "complete":
            out["per_model"][m] = analyse_model(m, rng)
    ms = list(out["per_model"])
    if ms:
        for m, h in zip(ms, L.holm([out["per_model"][m]["a"]["test"]["p"] for m in ms])):
            a = out["per_model"][m]["a"]
            a["holm_p"] = float(h)
            a["verdict"] = verdict_a(a["test"], h)
    return out


def to_md(o):
    Lm = ["# X9 results (EXPLORATORY; generated by analyze_x9.py)", "", f"Status: {o['status']}", ""]
    for m, r in o["per_model"].items():
        a, b = r["a"], r["b"]
        Lm += [f"## {m} (n = {r['n']})", "",
               f"- (a) A = mean ln(F/0.125) {a['test']['estimate']:+.3f} [{a['test']['ci95'][0]:+.3f}, {a['test']['ci95'][1]:+.3f}], "
               f"Holm p {a['holm_p']:.4f}: **{a['verdict']}** (mean F {a['F_mean']:.4f}; without layer 0 {a['F_no_l0_mean']:.4f})",
               f"- (a) F per layer: " + ", ".join(f"L{k} {x['F']:.3f}" for k, x in enumerate(a["F_per_layer"])),
               f"- (b) E(NAT-K*) {b['E_Kstar']:+.3f}" + (f"; G {b['G']:+.3f} [{b['G_ci95'][0]:+.3f}, {b['G_ci95'][1]:+.3f}] ({b['role']}): "
                                                        f"**{b['verdict']}**; E(L0nat) {b['E_L0']:+.3f}, within factor 2: {b['E_L0_within_factor_2']}"
                                                        if b.get("G") is not None else f"; {b['role']}"),
               f"- (c) Δρ(P_logit, no layer 0) {r['c']['P_logit']['delta_rho']['mean']:+.3f} {r['c']['P_logit']['delta_rho']['ci95']}; "
               f"D {r['c']['P_logit']['D']['mean']:+.3f} → D without layer 0 {r['c']['P_logit']['D_no_l0']['mean']:+.3f}",
               f"- (d) median effective NAT-K* scale per layer (s_K {r['d']['s_K']:.2f}): "
               + ", ".join(f"L{k} {x['median']:.2f}" for k, x in enumerate(r["d"]["per_layer"]) if x["median"] is not None), ""]
    return "\n".join(Lm) + "\n"


def main():
    o = analyse()
    (P7 / "x9_results.json").write_text(json.dumps(o, indent=1, default=float))
    md = to_md(o)
    (P7 / "X9_RESULTS.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
