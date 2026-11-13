"""PROTOCOL_V5 planning numbers (v5/plan/, PROTOCOL_V5 section 9). Reads v4 measured ledgers/design read-only; no GPU, no network.
Usage: python -I v5_plan_numbers.py <research_v2 dir> <out.json>"""
import json, math, sys
from pathlib import Path
from statistics import NormalDist

R = Path(sys.argv[1]); RUNS = R / "v4_runs"
design = json.loads((RUNS / "design.json").read_text())
fit = design["fit"]; A, B, HIST = fit["host_s_per_layer"], fit["device_s_per_byte"], fit["hist_ratio"]
NF4 = json.loads((R / "v4" / "plan" / "v4_plan_numbers.json").read_text())["nf4_bytes_per_param"]
D1 = 1.48                                  # v4 deviation D1 factor (measured Step 0 ratio of real to fitted time)
RATE = 0.59                                # A40 Secure on demand, $/h (v4; re-read before creation)
BILL = 19.07 / 15.601                      # v4 billed pod hours / GPU-ledger hours (11.253 $ GPU / 0.59 = 19.07 h)
N_TEST, N_G8, N_CAL = 100, 3, 50

# measured v4 GPU-ledger hours per model (gates + calibration + test), from gpu_ledger.jsonl
agg = {}
for line in (RUNS / "gpu_ledger.jsonl").read_text().splitlines():
    r = json.loads(line); agg[r["block"]] = agg.get(r["block"], 0) + r["seconds"]
def measured(m):
    return sum(agg.get(f"{b}:{m}", 0) for b in ("gates", "calibrate", "test")) / 3600
def gates_h(m):
    return agg.get(f"gates:{m}", 0) / 3600

GEOM = {  # layers, linear params (transformer linears, no embedding/head), total params
    "qwen2.5-0.5b": (24, 357826560, 493961216),
    "qwen2.5-1.5b": (28, 1310195712, 1543569408),
    "qwen2.5-7b": (28, 6525288448, 7615283200),
    "mistral-7b-v0.3": (32, 6979321856, 7247757312),
    # new, from the pinned config.json (HF, read 2026-11-11): per-layer q,k,v,o + 3 MLP matrices
    "qwen3-0.6b-base": (28, 28 * (1024 * 2048 + 2 * 1024 * 1024 + 2048 * 1024 + 3 * 1024 * 3072), None),
    "smollm2-1.7b": (24, 24 * (4 * 2048 * 2048 + 3 * 2048 * 8192), None),
}
EMB = {"qwen3-0.6b-base": 151936 * 1024, "smollm2-1.7b": 49152 * 2048}
for m in EMB:
    L, lin, _ = GEOM[m]; GEOM[m] = (L, lin, lin + EMB[m])

def fitted_problem_s(m):
    L, lin, tot = GEOM[m]
    by = 2.0 * tot + NF4 * lin + 2.0 * (tot - lin)
    t_arm = A * L + B * by
    return t_arm, 6 * t_arm + HIST * t_arm

def fitted_gpu_h(m, gates=0.25):
    t_arm, _ = fitted_problem_s(m)
    t_hist = HIST * t_arm
    return D1 * ((N_TEST + N_G8) * (6 * t_arm + t_hist) + N_CAL * (4 * t_arm + t_hist)) / 3600 + gates

check = {m: {"measured_h": round(measured(m), 2), "fit_x1.48_h": round(fitted_gpu_h(m, gates_h(m)), 2)} for m in ("qwen2.5-0.5b", "qwen2.5-1.5b", "qwen2.5-7b", "mistral-7b-v0.3")}

WIKI = 64 / 61.1      # WikiText histories always reach 64 steps; v4 GSM8K mean S was 60.4-63.9 (Mistral 61.1)
UNIT = {  # GPU-h per (model, condition), NF4 GSM8K = measured v4 where it exists
    "qwen2.5-0.5b": measured("qwen2.5-0.5b"), "qwen2.5-7b": measured("qwen2.5-7b"), "mistral-7b-v0.3": measured("mistral-7b-v0.3"),
    "qwen2.5-1.5b": measured("qwen2.5-1.5b"),
    "qwen3-0.6b-base": fitted_gpu_h("qwen3-0.6b-base", 0.16), "smollm2-1.7b": fitted_gpu_h("smollm2-1.7b", 0.18),
}
DL_GB = {"qwen2.5-0.5b": 1.0, "qwen3-0.6b-base": 1.2, "smollm2-1.7b": 3.4, "qwen2.5-7b": 15.2, "mistral-7b-v0.3": 14.5}

def test_share(m):
    t = agg.get(f"test:{m}", 0) / 3600
    return t / measured(m) if measured(m) else 0.62

def plan(hqq_ratio, b_models, d_models, gm=0.40, a_models=("qwen3-0.6b-base", "smollm2-1.7b"), anchor=True, n_bd=100):
    rows = []
    if anchor: rows.append(("anchor NF4 GSM8K", "qwen2.5-0.5b", UNIT["qwen2.5-0.5b"]))
    for m in a_models: rows.append(("A NF4 GSM8K", m, UNIT[m]))
    for m in b_models: rows.append(("B HQQ GSM8K", m, UNIT[m] * hqq_ratio * (1 - test_share(m) * (1 - (n_bd + 3) / 103))))
    for m in d_models: rows.append(("D NF4 WikiText-2", m, UNIT[m] * WIKI * (1 - test_share(m) * (1 - (n_bd + 3) / 103))))
    gpu = gm + sum(r[2] for r in rows)
    models = {r[1] for r in rows}
    billed = (gpu * BILL + 0.5 + 0.25 * len(models)) * 1.05     # first boot + one download/setup per distinct model + 5 %
    return {"rows": [(a, m, round(h, 2)) for a, m, h in rows], "gpu_h": round(gpu, 2), "billed_h": round(billed, 2),
            "pod_usd": round(billed * RATE, 2), "models": sorted(models), "download_gb": round(sum(DL_GB[m] for m in models), 1)}

def storage(gb, usd_gb_month, hours):
    return round(gb * usd_gb_month / (30 * 24) * hours, 2)

out = {"check_v4_fit_x1.48_vs_measured": check, "unit_gpu_h": {k: round(v, 2) for k, v in UNIT.items()}, "wiki_factor": round(WIKI, 3),
       "billed_over_gpu": round(BILL, 3), "plans": {}}
for r in (1.5, 2.0, 3.0):
    for name, b, d in (("full: B=D={0.5B,7B,Mistral}", ["qwen2.5-0.5b", "qwen2.5-7b", "mistral-7b-v0.3"], ["qwen2.5-0.5b", "qwen2.5-7b", "mistral-7b-v0.3"]),
                       ("core: B=D={0.5B,Mistral}", ["qwen2.5-0.5b", "mistral-7b-v0.3"], ["qwen2.5-0.5b", "mistral-7b-v0.3"]),
                       ("B={0.5B,Mistral}, D={0.5B,7B,Mistral}", ["qwen2.5-0.5b", "mistral-7b-v0.3"], ["qwen2.5-0.5b", "qwen2.5-7b", "mistral-7b-v0.3"]),
                       ("min: B={0.5B,Mistral}, D={0.5B}", ["qwen2.5-0.5b", "mistral-7b-v0.3"], ["qwen2.5-0.5b"]),
                       ("no B: D={0.5B,Mistral}", [], ["qwen2.5-0.5b", "mistral-7b-v0.3"])):
      for nbd in (100, 60):
        p = plan(r, b, d, n_bd=nbd); name2 = f"{name} n_BD={nbd}"
        hours_alive = p["billed_h"] + 48
        vol = 50
        p["storage_usd_hp_50gb"] = storage(vol, 0.14, hours_alive)
        p["container_disk_usd"] = round(30 * 0.10 / (30 * 24) * p["billed_h"], 2)
        p["total_usd"] = round(p["pod_usd"] + p["storage_usd_hp_50gb"] + p["container_disk_usd"], 2)
        out["plans"][f"hqq x{r} | {name2}"] = p

# power: MDE at 80 %, two-sided, per-problem SD (v4 max 0.64 from CI half-widths), Bonferroni worst case of Holm
z = NormalDist().inv_cdf
def mde(sd, n, a): return (z(1 - a / 2) + z(0.8)) * sd / math.sqrt(n)
out["power"] = {f"n{n}_m{k}": round(mde(0.64, n, 0.05 / k), 3) for n in (100, 60) for k in (1, 2, 3, 6, 9)}
out["power_h5"] = {f"n{n}_m{k}": round(mde(0.30, n, 0.05 / k), 3) for n in (100, 60) for k in (4, 6, 12)}
Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
print(json.dumps(out["check_v4_fit_x1.48_vs_measured"]), json.dumps(out["unit_gpu_h"]))
for k, p in out["plans"].items():
    print(f"{k:68s} GPU {p['gpu_h']:5.2f}h billed {p['billed_h']:5.2f}h pods ${p['pod_usd']:5.2f} stor ${p['storage_usd_hp_50gb']:.2f} total ${p['total_usd']:5.2f} dl {p['download_gb']}GB")
print(out["power"], out["power_h5"])
