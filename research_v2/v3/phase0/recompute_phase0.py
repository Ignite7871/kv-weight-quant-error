"""v3 Phase 0: independent recomputation of the numbers behind the known manuscript inconsistencies. CPU only.

Deliberately imports nothing from the existing analysis/audit code (analyze_stage*.py, paper/audit/*, paper/exploratory/*).
Reads only the raw per-problem records. Outputs phase0_results.json next to this file.

  1. H5 (oracle - cache-only) 90% percentile-bootstrap CI: exact bootstrap distribution (B -> infinity) plus the spread of
     endpoints over 1,000 independent 10,000-resample streams, for Stage 1 (n=930) and held-out excluding 47 exposed (n=342).
  2. Unparseable-output counts in Stage 1 under the two definitions (no '####' vs no '####'-followed-by-a-number).
  3. Generated-length definitions behind '~108' and '112'.

  venv/Scripts/python.exe research_v2/v3/phase0/recompute_phase0.py
"""

import json
import math
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RV = HERE.parents[1]
ARMS = ["fp16", "nf4", "cache_only", "token_only", "oracle_flip"]
DRIFT = re.compile(r"\n\s*Question\s*:", re.IGNORECASE)
HASHNUM = re.compile(r"####\s*(-?\d[\d,]*(?:\.\d+)?)")


def records(d):
    out = {}
    for a in ARMS:
        rs = [json.loads(x) for x in (d / f"{a}.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        out[a] = {r["index"]: r for r in rs}
        assert len(out[a]) == len(rs)
    return out


def binom_pmf(n, p):
    k = np.arange(n + 1)
    if p <= 0:
        return (k == 0).astype(float)
    if p >= 1:
        return (k == n).astype(float)
    lg = np.array([math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1) for j in k])
    return np.exp(lg + k * math.log(p) + (n - k) * math.log1p(-p))


def exact_boot_diff(npos, nneg, n):
    """Exact distribution of sum(d*) for a bootstrap resample of n values in {-1,0,+1} with counts npos/nneg."""
    pp, pn = npos / n, nneg / n
    pa = binom_pmf(n, pp)
    pmf = np.zeros(2 * n + 1)                         # index s + n, s = A - C
    for a in range(n + 1):
        if pa[a] < 1e-300:
            continue
        rest = n - a
        q = pn / (1 - pp) if pp < 1 else 0.0
        pc = binom_pmf(rest, q)
        pmf[a - np.arange(rest + 1) + n] += pa[a] * pc
    return pmf


def h5_ci(d, label, n_streams=1000, B=10_000):
    n = len(d)
    npos, nneg = int((d == 1).sum()), int((d == -1).sum())
    pmf = exact_boot_diff(npos, nneg, n)
    cdf = np.cumsum(pmf)
    s = np.arange(-n, n + 1)
    lo_exact = s[np.searchsorted(cdf, 0.05)] / n * 100
    hi_exact = s[np.searchsorted(cdf, 0.95)] / n * 100
    # finite-B streams, numpy default 'linear' percentile as in the locked analysis
    rng = np.random.default_rng(np.frombuffer(b"v3-phase0-h5-" + label.encode(), dtype=np.uint8).astype(np.uint32).sum() * 7919)
    probs = np.array([npos, n - npos - nneg, nneg]) / n
    his, los = [], []
    for _ in range(n_streams):
        draw = rng.multinomial(n, probs, size=B)
        m = (draw[:, 0] - draw[:, 2]) / n * 100
        lo, hi = np.percentile(m, [5, 95])
        los.append(round(lo, 2)); his.append(round(hi, 2))
    upper_lattice = {f"{k:+.2f}": int(v) for k, v in zip(*np.unique(his, return_counts=True))}
    lower_lattice = {f"{k:+.2f}": int(v) for k, v in zip(*np.unique(los, return_counts=True))}
    k_hi = int(np.searchsorted(cdf, 0.95))
    return {"n": n, "n_pos": npos, "n_neg": nneg, "estimate_pp": 100 * d.mean(), "lattice_step_pp": 100 / n,
            "exact_bootstrap_ci90_pp": [lo_exact, hi_exact],
            "exact_cdf_around_upper": {f"{s[j] / n * 100:+.2f}": float(cdf[j]) for j in range(k_hi - 2, k_hi + 2)},
            "streams": n_streams, "B_per_stream": B,
            "upper_endpoint_frequency_over_streams": upper_lattice, "lower_endpoint_frequency_over_streams": lower_lattice,
            "delta_star_exact_pp": max(abs(lo_exact), abs(hi_exact))}


def parse_class(text):
    m = DRIFT.search(text)
    t = text[: m.start()] if m else text
    if "####" not in t:
        return "no_hash"
    return "parsed" if HASHNUM.search(t) else "hash_without_number"


def main():
    s1 = records(RV / "stage1_runs" / "eval")
    s2 = records(RV / "stage2_runs" / "eval")
    exposed = set(json.loads((RV / "stage2_sample.json").read_text())["exposed_gold_trajectory_test_0_199"])
    res = {}

    # 1. H5 CIs
    def dvec(R, idx):
        return np.array([(R["oracle_flip"][i]["status"] == "SOLVED") - (R["cache_only"][i]["status"] == "SOLVED") for i in idx], dtype=int)
    idx1 = sorted(s1["fp16"])
    idx342 = sorted(i for i in s2["fp16"] if i not in exposed)
    assert len(idx1) == 930 and len(idx342) == 342
    res["H5_stage1"] = h5_ci(dvec(s1, idx1), "stage1")
    res["H5_heldout342"] = h5_ci(dvec(s2, idx342), "heldout342")
    res["H5_stored"] = {
        "locked_analysis (Table B.5/B.6)": {"stage1": json.load(open(RV / "stage1_analysis" / "stage1_report.json"))["primary_family"]["oracle_flip - cache_only"]["ci90_pp"],
                                            "heldout342": json.load(open(RV / "stage2_analysis" / "stage2_report.json"))["sensitivity_excluding_prior_exposure"]["primary_family"]["oracle_flip - cache_only"]["ci90_pp"]},
        "exploratory_audit (Table D.19)": {k: json.load(open(RV / "paper" / "exploratory" / "oracle_equivalence_results.json"))["samples"][k]["ci90_boot_pp"] for k in ("stage1", "heldout342")}}

    # 2. unparseable, Stage 1
    up = {}
    for a in ARMS:
        c = {"parsed": 0, "no_hash": 0, "hash_without_number": 0}
        for i in idx1:
            c[parse_class(s1[a][i]["text"])] += 1
        c["no_hash_pct"] = 100 * c["no_hash"] / 930
        c["unparseable"] = c["no_hash"] + c["hash_without_number"]
        c["unparseable_pct"] = 100 * c["unparseable"] / 930
        up[a] = c
    res["stage1_unparseable"] = up

    # 3. length definitions
    L = {}
    for name, R, idx in (("stage1_n930", s1, idx1), ("heldout_n389", s2, sorted(s2["fp16"]))):
        for a in ARMS:
            L[f"{name}/{a}"] = {"mean_len_ids": float(np.mean([len(R[a][i]["ids"]) for i in idx])),
                                "mean_steps": float(np.mean([R[a][i]["steps"] for i in idx])),
                                "mean_prompt_tokens": float(np.mean([R[a][i]["prompt_tokens"] for i in idx]))}
        L[f"{name}/all_arms"] = {"mean_len_ids": float(np.mean([len(R[a][i]["ids"]) for a in ARMS for i in idx])),
                                 "mean_steps": float(np.mean([R[a][i]["steps"] for a in ARMS for i in idx]))}
    e1 = {json.loads(x)["index"]: json.loads(x) for x in (RV / "e1_runs" / "tierA" / "NAT.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()}
    e1idx = sorted(e1)
    allfp = {**s1["fp16"], **s2["fp16"]}
    L["e1_tierA_n200/ref_len (source of 112)"] = float(np.mean([e1[i]["ref_len"] for i in e1idx]))
    L["e1_tierA_n200/stored_fp16_len_ids"] = float(np.mean([len(allfp[i]["ids"]) for i in e1idx]))
    L["e1_tierA_n200/stored_fp16_steps"] = float(np.mean([allfp[i]["steps"] for i in e1idx]))
    L["e1_tierA_n200/ref_len_equals_stored_len_ids_all"] = all(e1[i]["ref_len"] == len(allfp[i]["ids"]) for i in e1idx)
    res["lengths"] = L

    # 4. Tier B accuracy: realized minimum detectable paired difference (two-sided 5%, 80% power, normal approx., n=400)
    tb = {a: {json.loads(x)["index"]: json.loads(x)["status"] == "SOLVED" for x in (RV / "e1_runs" / "tierB" / f"{a}.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()}
          for a in ("FP16", "NAT", "C1", "C2")}
    tidx = sorted(tb["FP16"])
    assert all(sorted(v) == tidx for v in tb.values())
    z = 1.959963984540054 + 0.8416212335729143
    mde = {"n": len(tidx), "formula": "(z_.975 + z_.80) * sqrt(discordance / n)"}
    for a, b in (("NAT", "C1"), ("NAT", "C2"), ("C2", "C1"), ("NAT", "FP16"), ("C1", "FP16"), ("C2", "FP16")):
        disc = float(np.mean([tb[a][i] != tb[b][i] for i in tidx]))
        mde[f"{a}-{b}"] = {"discordance": disc, "mde_pp": 100 * z * math.sqrt(disc / len(tidx))}
    mde["protocol_projection"] = "E1 protocol draft: discordance 0.16-0.20 at n=500 -> 5-6 points"
    res["tierB_accuracy_mde"] = mde

    (HERE / "phase0_results.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
