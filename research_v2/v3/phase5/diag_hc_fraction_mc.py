"""Phase 5 diagnostic (not locked): Monte-Carlo spread of the llama-base H-C fraction CI, the one recomputation disagreement.
Re-runs the locked hc_model (analyze_v3, imported unchanged) and the independent hc_one with 30 seeds each; prints the spread of the endpoints."""
import json, sys
from pathlib import Path
import numpy as np
RV = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RV)); sys.path.insert(0, str(Path(__file__).parent))
import analyze_v3 as L
import recompute_v3_independent as I
data, _ = I.load()
A = data["llama-base"]; ids = sorted(A["NAT"])
st = L.hc_stats(A["NAT"], A["C1"], ids); steps = I.hc_steps(A["NAT"], A["C1"], ids)
lo_L, hi_L, lo_I, hi_I, bands = [], [], [], [], set()
for s in range(30):
    r, _ = L.hc_model(st, np.random.default_rng(1000 + s)); lo_L.append(r["fraction_ci95"][0]); hi_L.append(r["fraction_ci95"][1]); bands.add(r["band"])
    q = I.hc_one(steps, I.Boot(2000 + s)); lo_I.append(q["fraction_ci95"][0]); hi_I.append(q["fraction_ci95"][1]); bands.add(q["band"])
out = {"beta0": r["beta0"], "beta1": r["beta1"], "fraction": r["fraction"],
       "locked_method_lo": [float(np.min(lo_L)), float(np.median(lo_L)), float(np.max(lo_L)), float(np.std(lo_L))],
       "independent_lo": [float(np.min(lo_I)), float(np.median(lo_I)), float(np.max(lo_I)), float(np.std(lo_I))],
       "locked_method_hi": [float(np.min(hi_L)), float(np.max(hi_L))], "independent_hi": [float(np.min(hi_I)), float(np.max(hi_I))],
       "bands_seen": sorted(bands)}
(Path(__file__).with_suffix(".json")).write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
