"""Synthetic end-to-end test of analyze_v6 (fake records in the rv6 array layout; writes only to a temporary folder).

Planted: H2 > 0 split into common-mode and direction parts; L8 alignment of NAT-V above C1-V; L9 storage less directional; storage K-first.
  python research_v2/v6/tests/test_analyze_v6.py
"""
import importlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V6 = HERE.parent
sys.path.insert(0, str(V6))
GEOM = {"qwen2.5-0.5b": (24, 2, 64), "llama3.2-1b": (16, 8, 64), "qwen2.5-1.5b": (28, 2, 128)}
ARMS = ["NAT-K", "NAT-V", "C1-K", "C1-V", "NAT-K-cen", "NAT-V-cen", "ST-K", "ST-V", "ST-K@nat", "ST-V@nat", "ST-V@nat-cen", "ST-Kc", "ST-Kc@nat"]
READ = ["m", "N_D", "meas", "lin_V", "mod_V", "lin_K", "modK_c", "modK_w", "jensen", "meas_dm", "lam_K", "contrast", "s2", "mu2_V"]
N = 25


def build(root):
    g = np.random.default_rng(3)
    for m, (L, H, d) in GEOM.items():
        dd = root / "test" / m
        (dd / "arrays").mkdir(parents=True)
        lines = {a: [] for a in ARMS}
        for i in range(N):
            S = int(g.integers(20, 65))
            base = np.exp(g.normal(-7, 0.4, S))
            eff = {"NAT-V": 0.9, "NAT-V-cen": 0.6, "C1-V": 0.0, "ST-V@nat": 0.05, "ST-V@nat-cen": 0.02, "NAT-K": -1.0, "C1-K": 0.0, "NAT-K-cen": -1.0,
                   "ST-K": 0.4, "ST-V": -0.2, "ST-K@nat": 0.0, "ST-Kc": 0.3, "ST-Kc@nat": 0.0}
            for a in ARMS:
                js = base * math.exp(eff[a] + g.normal(0, 0.3))
                js[0] = 0.0
                rec = {"index": i, "S": S, "js": js.tolist(), "stored_rel_K": [0.1 if a.startswith("ST-K") else 0.05] * S,
                       "stored_rel_V": [0.06 if a == "ST-V" else 0.15] * S}
                lines[a].append(json.dumps(rec))
                z = {k: np.abs(g.normal(1, 0.1, (S, L, H))).astype(np.float32) for k in READ}
                z["meas"] = z["meas"] * math.exp(eff[a])
                z["mod_V"] = z["meas"] * 1.05
                z["lin_K"] = z["meas"] * 1.1
                al = {"NAT-V": 0.4, "NAT-V-cen": 0.4, "ST-V@nat": 0.05, "ST-V@nat-cen": 0.05}.get(a, 0.0)
                z["A_o"] = np.exp(g.normal(al, 0.1, (S, L, H))).astype(np.float32)
                z["A_cov"] = np.exp(g.normal(al, 0.1, (S, L, H))).astype(np.float32)
                n = S
                z["normsK"] = np.ones((n, L, H))
                z["normsV"] = np.ones((n, L, H))
                z["S1K"] = g.normal(0, 1, (L, H, d)) * 2
                z["S1V"] = g.normal(0, 1, (L, H, d)) * (0.2 if a == "ST-V" else 3)
                z["S2K_pair"] = np.full((L, H, d // 2), 2.0 * n / (d // 2) * 1.0)
                z["S1K_pre"] = z["S1K"] * 1.0
                z["S2K_pre_pair"] = z["S2K_pair"].copy()
                z["S1x"] = g.normal(0, 1, (L, 32)) * n
                z["S2x"] = np.full(L, 32.0 * n * 2)
                np.savez_compressed(dd / "arrays" / f"{a}_{i}.npz", **z)
        for a, ls in lines.items():
            (dd / f"{a}.jsonl").write_text("\n".join(ls) + "\n", encoding="utf-8")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        build(t)
        os.environ["V6_RUNS"] = str(t)
        import analyze_v6 as A
        A = importlib.reload(A)
        A.B = 3000
        import analyze_r_v6 as RV6
        RV6.B = 3000
        R = {m: A.per_model(m, 0.25) for m in A.MODELS}
        v, hp = A.verdicts(R)
        for m, r in R.items():
            assert abs(r["L3"]["theta_V"]["estimate"] - r["L3"]["theta_cm"]["estimate"] - r["L3"]["theta_dir"]["estimate"]) < 1e-12
        print(json.dumps(v, indent=1))
        assert v["L8a"] == "agreement" and v["L8b"] == "agreement", v
        assert v["L9a"] == "agreement" and v["L9b"] == "agreement", v
        assert v["L1"] == "agreement", v
        print("OK")
