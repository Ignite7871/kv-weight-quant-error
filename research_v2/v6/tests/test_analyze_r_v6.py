"""Synthetic test of analyze_r_v6: planted effects must give 'agreement' on R1, R3, R4; a null world must not.

  python research_v2/v6/tests/test_analyze_r_v6.py      (writes only to a temporary folder)
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
GEOM = {"qwen2.5-0.5b": 64, "qwen2.5-1.5b": 128, "qwen2.5-3b": 128, "qwen2.5-7b": 128, "llama3.2-1b": 64, "llama3.2-3b": 128,
        "llama3.1-8b": 128, "mistral-7b-v0.3": 128, "U1": 128, "U2": 64}
NPROB, S, LAY, KV = 30, 64, 2, 2


def build(root: Path, planted: bool, seed: int):
    g = np.random.default_rng(seed)
    sys.path.insert(0, str(V6))
    import analyze_r_v6 as R
    for u, hd in GEOM.items():
        base = (root / ("v4" if not u.startswith("U") else "v5") / u / "test")
        (base / "slots").mkdir(parents=True, exist_ok=True)
        theta, sc = R.UNITS[u][2], R.UNITS[u][3]
        A = R.dirichlet(S, R.inv_freq(hd, theta, sc))
        recs = {a: [] for a in R.ARMS}
        for i in range(NPROB):
            c = g.uniform(0.05, 0.6)                          # planted common-mode share of the value error
            t = np.arange(1, S + 1, dtype=float)
            base_js = np.exp(g.normal(-7, 0.5, S))
            grow = (1 + c * 3 * np.log(t)) if planted else np.ones(S)
            jsV_nat = base_js * grow * np.exp(g.normal(0, 0.3, S))
            jsV_c1 = base_js * np.exp(g.normal(0, 0.3, S))
            for arm, js in (("NAT-V", jsV_nat), ("C1-V", jsV_c1), ("NAT-K", base_js), ("C1-K", base_js)):
                js = js.copy()
                js[0] = 0.0
                recs[arm].append({"index": i, "S": S, "js": js.tolist()})
            norms = np.ones((S, LAY, KV))
            cv = c if planted else g.uniform(0.05, 0.6)
            dirV = g.standard_normal((LAY, KV, hd))
            dirV /= np.linalg.norm(dirV, axis=-1, keepdims=True)
            S1V = dirV * S * math.sqrt(cv) + g.standard_normal((LAY, KV, hd)) * math.sqrt(S / hd)
            h = hd // 2
            amp = np.sqrt(A) if planted else np.ones(h)
            k = g.standard_normal((LAY, KV, h)) * amp
            k2 = g.standard_normal((LAY, KV, h)) * amp
            S1K = np.concatenate([k, k2], axis=-1) * S * 0.3
            np.savez_compressed(base / "slots" / f"NAT_{i}.npz", normsK=norms, normsV=norms, S1K=S1K, S1V=S1V)
        for arm, rs in recs.items():
            (base / f"{arm}.jsonl").write_text("\n".join(json.dumps(r) for r in rs) + "\n", encoding="utf-8")


def run_world(planted, seed):
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        build(d, planted, seed)
        os.environ["V6_V4_RUNS"], os.environ["V6_V5_RUNS"], os.environ["V6_OUT"] = str(d / "v4"), str(d / "v5"), str(d / "out")
        sys.path.insert(0, str(V6))
        import analyze_r_v6 as R
        R = importlib.reload(R)
        R.B = 500                                            # speed only; the real run uses 10,000
        rep = R.run()
        return {k: v["verdict"] for k, v in rep["verdicts"].items()}, rep["L_block_gate"]


if __name__ == "__main__":
    v, gate = run_world(True, 1)
    print("planted:", v, gate)
    assert v == {"R1": "agreement", "R3": "agreement", "R4": "agreement"}, v
    v0, gate0 = run_world(False, 2)
    print("null:", v0, gate0)
    assert "agreement" not in v0.values(), v0
    print("OK")
