"""Planning numbers for PROTOCOL_V4 (no GPU, no network).

Inputs:
- v3 records (per-problem SDs and wall times; read-only);
- model geometry: Qwen2.5-1.5B/3B/7B and Mistral-7B-v0.3 from their HF config.json (fetched 2026-11-02);
  Llama-3.2-3B and Llama-3.1-8B from the model cards (gated; re-read from the snapshot at gate G7);
  Qwen2.5-0.5B and Llama-3.2-1B(-Instruct) from the local cache;
- RunPod on-demand prices from runpod.io/pricing (read 2026-11-02). The study GPU's secure price was confirmed by the corresponding author on
  2026-11-07 (r4: A40, $0.59/h, good stock; RTX A6000 N/A for network volumes). The other rows stay for comparison.
r5 (2026-11-07): ceiling $30 including storage, test n = 100, calibration n = 50, one quantizer, 100 GB volume prorated to the expected duration.
All time and memory figures are projections under the stated assumptions; gate G6 replaces them with measurements.
"""
import json, math, sys
from statistics import mean, stdev, NormalDist

R = sys.argv[1]
N = NormalDist()

# name: (hidden, intermediate, layers, heads, kv_heads, head_dim, vocab, tied, licence, source)
SPECS = {
    "qwen2.5-0.5b": (896, 4864, 24, 14, 2, 64, 151936, True, "apache-2.0", "local cache"),
    "qwen2.5-1.5b": (1536, 8960, 28, 12, 2, 128, 151936, True, "apache-2.0", "HF config"),
    "qwen2.5-3b": (2048, 11008, 36, 16, 2, 128, 151936, True, "qwen-research", "HF config"),
    "qwen2.5-7b": (3584, 18944, 28, 28, 4, 128, 152064, False, "apache-2.0", "HF config"),
    "llama3.2-1b": (2048, 8192, 16, 32, 8, 64, 128256, True, "llama3.2", "local cache"),
    "llama3.2-3b": (3072, 8192, 28, 24, 8, 128, 128256, True, "llama3.2", "model card"),
    "llama3.1-8b": (4096, 14336, 32, 32, 8, 128, 128256, False, "llama3.1", "model card"),
    "mistral-7b-v0.3": (4096, 14336, 32, 32, 8, 128, 32768, False, "apache-2.0", "HF config"),
}
NF4_B = 0.5 + 1 / 64 + 4 / (64 * 256)
GPUS = {  # name: (memory GB, bandwidth GB/s, secure $/h, community $/h)
    "RTX A6000": (48, 768, 0.59, 0.33),
    "A40": (48, 696, 0.59, 0.35),
    "L40S": (48, 864, 1.09, 0.79),
    "A100 80GB PCIe": (80, 1935, 1.79, 1.19),
}
STUDY_GPU = "A40"      # D5 (r4)

# --- v3 records: per-problem SD of the core contrasts and wall time per arm-problem ---
V3 = ["qwen-base", "llama-instruct", "qwen-instruct", "llama-base"]
CORE = ["NAT", "C1", "NAT-K", "C1-K", "NAT-V", "C1-V"]
v3 = {}
for m in V3:
    lm, wall = {}, {}
    for a in CORE:
        recs = [json.loads(l) for l in open(f"{R}/v3_runs/{m}/tierA/{a}.jsonl")]
        lm[a] = {r["index"]: math.log(max(mean(r["js"]), 1e-9)) for r in recs}
        wall[a] = mean(r["wall_seconds"] for r in recs)
    ids = sorted(lm["NAT"])
    con = {}
    for a, b in [("NAT", "C1"), ("NAT-K", "C1-K"), ("NAT-V", "C1-V"), ("NAT-K", "NAT-V")]:
        d = [lm[a][i] - lm[b][i] for i in ids]
        con[f"{a}|{b}"] = dict(est=mean(d), sd=stdev(d))
    v3[m] = dict(n=len(ids), contrasts=con, wall_per_arm_problem=wall)
SD_MAX = max(c["sd"] for m in v3 for c in v3[m]["contrasts"].values())

# host-bound constant: v3 laptop seconds per layer per arm-problem (NAT arm; includes v3's predictor overhead)
c_host = max(v3["qwen-base"]["wall_per_arm_problem"]["NAT"] / 24, v3["llama-base"]["wall_per_arm_problem"]["NAT"] / 16)

STEPS = 64
ARMS_TEST = 7          # 6 core arms + FP16 reference/history (counted as a full arm: conservative)
ARMS_CAL = 5           # FP16, NAT, C1 at alpha 0.5/1/2
N_TEST, N_CAL = 100, 50  # r5: the first 100 and 50 of the fixed seeded orders
G8_PROBLEMS = 3        # G8 runs the first 3 test problems twice (counted from r5, because the budget is tight)
GM_H = 0.5             # GM once: the CPU suites took about 20 min on the laptop (r5: counted explicitly)
FIRST_BOOT_H = 0.5     # pod_setup.sh once: venv, torch, packages (billed, not GPU-ledger)
AUTHOR_WAIT_H = 48.0   # volume alive with no pod: Step 0 decision, G6 stop, gaps between models (assumed)
CEILING, ALT_CEILING, RESERVE = 30.0, 40.0, 1.0   # hard ceiling; the other figure in the r5 request; close-out reserve (final download, storage until deletion)
VOLUME_GB, STORAGE_RATE, MONTH_H = 100, 0.07, 720.0
VENV_GB, RECORDS_GB = 12.0, 3.0  # assumed: torch cu126 + packages; records, npz and checkpoints at n = 100
GATES_H = 0.35         # v3 per-model gates
SETUP_H = 0.25         # download, load, checkpoint upload per model (billed, not GPU-ledger)
IDLE = 1.20            # billed pod time / GPU-ledger time
MARGIN = 1.05
EFF_FP16, EFF_NF4 = 0.6, 0.3   # fraction of peak bandwidth reached (assumed)

def model_numbers(name, gpu):
    h, I, L, nq, nkv, hd, V, tied, lic, src = SPECS[name]
    q = h * nq * hd; kv = 2 * h * nkv * hd; o = nq * hd * h; mlp = 3 * h * I
    lin = L * (q + kv + o + mlp)
    emb = V * h; head = 0 if tied else V * h
    gb = 1e9
    mem = (2 * (lin + emb + head) + NF4_B * lin) / gb + 1.5   # dual weights + 1.5 GB context/activations/cache
    mem_gb, bw = GPUS[gpu][0], GPUS[gpu][1]
    dev_fp16 = 2 * (lin + V * h) / (EFF_FP16 * bw * gb)             # lm_head read every pass
    dev_nf4 = (NF4_B * lin + 2 * V * h) / (EFF_NF4 * bw * gb)
    t_arm = c_host * L + STEPS * (dev_fp16 + dev_nf4)                # seconds per arm-problem
    gpu_h = (t_arm * (ARMS_TEST * (N_TEST + G8_PROBLEMS) + ARMS_CAL * N_CAL)) / 3600 + GATES_H
    return dict(licence=lic, geometry_source=src, layers=L, kv_heads=nkv, head_dim=hd, linear_params=lin,
                total_params=lin + emb + head, dual_mem_gb=round(mem, 1), fits=mem < 0.9 * mem_gb,
                s_per_arm_problem=round(t_arm, 2), gpu_h=round(gpu_h, 2))

proj = {}
for gpu in GPUS:
    rows = {m: model_numbers(m, gpu) for m in SPECS}
    step0_extra = model_numbers("llama3.2-1b", gpu)["gpu_h"]          # Llama-3.2-1B-Instruct, same geometry
    core_h = sum(r["gpu_h"] for r in rows.values()) + step0_extra
    billed_h = (core_h + GM_H) * IDLE + SETUP_H * (len(SPECS) + 1) + FIRST_BOOT_H
    proj[gpu] = dict(models=rows, step0_llama_instruct_h=step0_extra, gpu_ledger_h=round((core_h + GM_H) * MARGIN, 1),
                     billed_h=round(billed_h * MARGIN, 1),
                     usd_secure=round(billed_h * MARGIN * GPUS[gpu][2], 0),
                     usd_community=round(billed_h * MARGIN * GPUS[gpu][3], 0))

# r5: the study under the $30 ceiling (GPU-h, billed h, USD), every model and without Mistral, at the projected speed and when everything runs slower
def scenario(gpu, drop_mistral=False, speed=1.0):
    rows = proj[gpu]["models"]
    names = [m for m in rows if not (drop_mistral and m == "mistral-7b-v0.3")]
    gpu_h = (sum(rows[m]["gpu_h"] for m in names) + proj[gpu]["step0_llama_instruct_h"] + GM_H) * speed
    billed = (gpu_h * IDLE + SETUP_H * (len(names) + 1) + FIRST_BOOT_H) * MARGIN
    pod = billed * GPUS[gpu][2]
    storage = VOLUME_GB * STORAGE_RATE * (billed + AUTHOR_WAIT_H) / MONTH_H
    return dict(gpu_h=round(gpu_h * MARGIN, 1), billed_h=round(billed, 1), pod_usd=round(pod, 2), storage_usd=round(storage, 2), total_usd=round(pod + storage, 2),
                margin_under_30=round(CEILING - pod - storage, 2), margin_under_30_after_reserve=round(CEILING - RESERVE - pod - storage, 2),
                margin_under_40=round(ALT_CEILING - pod - storage, 2), wall_days=round((billed + AUTHOR_WAIT_H) / 24, 1))


def breakeven(gpu, drop_mistral, limit):
    a, b = scenario(gpu, drop_mistral, 1.0)["total_usd"], scenario(gpu, drop_mistral, 2.0)["total_usd"]
    return round(1 + (limit - a) / (b - a), 2)


budget = dict(ceiling_usd=CEILING, other_figure_in_request_usd=ALT_CEILING, closeout_reserve_usd=RESERVE, stop_usd=CEILING - RESERVE, volume_gb=VOLUME_GB,
              storage_usd_per_day=round(VOLUME_GB * STORAGE_RATE / 30, 3), author_wait_h=AUTHOR_WAIT_H,
              all_models=scenario(STUDY_GPU), without_mistral=scenario(STUDY_GPU, True),
              speed_sensitivity={f"x{s}": dict(all_models=scenario(STUDY_GPU, False, s)["total_usd"], without_mistral=scenario(STUDY_GPU, True, s)["total_usd"]) for s in (1.0, 1.25, 1.5, 2.0)},
              breakeven_speed_factor_at_stop=dict(all_models=breakeven(STUDY_GPU, False, CEILING - RESERVE), without_mistral=breakeven(STUDY_GPU, True, CEILING - RESERVE)),
              community_all_models_usd=round(proj[STUDY_GPU]["billed_h"] * GPUS[STUDY_GPU][3] + scenario(STUDY_GPU)["storage_usd"], 2))
_a40 = proj[STUDY_GPU]["models"]
_step0_gpu = _a40["qwen2.5-0.5b"]["gpu_h"] + _a40["llama3.2-1b"]["gpu_h"] + proj[STUDY_GPU]["step0_llama_instruct_h"] + GM_H
budget["committed_before_g6_usd"] = round((_step0_gpu * IDLE + SETUP_H * 3 + FIRST_BOOT_H) * MARGIN * GPUS[STUDY_GPU][2], 2)

# storage: the pinned shards from the manifest, plus the venv and the records, against the 100 GB volume
_man = json.load(open(f"{R}/v4/V4_MODEL_MANIFEST.json"))["models"]
shard_gb = sum(s["size"] for e in _man.values() for s in e["shards"].values()) / 1e9
storage_gb = shard_gb + VENV_GB + RECORDS_GB

def mde(sd, n, alpha, power=0.8):
    return (N.inv_cdf(1 - alpha / 2) + N.inv_cdf(power)) * sd / math.sqrt(n)

ALPHAS = [("0.05", 0.05), ("0.05/8", 0.05 / 8), ("0.05/7", 0.05 / 7), ("0.05/16", 0.05 / 16), ("0.05/14", 0.05 / 14)]
power = {f"n{n}": {f"alpha_{lab}": round(mde(SD_MAX, n, a), 3) for lab, a in ALPHAS} for n in [200, 100]}

# r5: H1-H3 and H5 at n = 100, per hypothesis, from the v3 per-problem SDs and effects (H5: SD backed out of the X13 CI at n = 40)
HKEY = {"H1": "NAT-K|C1-K", "H2": "NAT-V|C1-V", "H3": "NAT-K|NAT-V"}
Z95, Z90 = N.inv_cdf(0.975), N.inv_cdf(0.95)
power_r5 = {}
for h, k in HKEY.items():
    sd = max(v3[m]["contrasts"][k]["sd"] for m in V3)
    eff = min(abs(v3[m]["contrasts"][k]["est"]) for m in V3)
    power_r5[h] = dict(sd_max_v3=round(sd, 3), smallest_v3_effect=round(eff, 3), mde_alpha_holm8=round(mde(sd, 100, 0.05 / 8), 3), mde_alpha_holm7=round(mde(sd, 100, 0.05 / 7), 3),
                       mde_alpha_holm8_n200=round(mde(sd, 200, 0.05 / 8), 3), effect_over_mde=round(eff / mde(sd, 100, 0.05 / 8), 1))
_x13 = json.load(open(f"{R}/v3/phase8/x13_results.json"))["per_model"]
_xs = [(p, _x13[m]["a"][p]["test"]) for m in V3 for p in ("K", "V")]
_xsd = max((t["ci95"][1] - t["ci95"][0]) / (2 * Z95) * math.sqrt(t["n"]) for _, t in _xs)
_xmin = min(t["estimate"] for _, t in _xs)
power_r5["H5"] = dict(sd_max_x13_n40=round(_xsd, 3), smallest_v3_L=round(_xmin, 3), ci95_half_width_n100=round(Z95 * _xsd / 10, 3), mde_alpha_holm16=round(mde(_xsd, 100, 0.05 / 16), 3),
                      smallest_L_supported_lower_bound_above_ln4=round(math.log(4) + Z95 * _xsd / 10, 3), threshold_ln4=round(math.log(4), 3))
_nc = {m: v3[m]["contrasts"]["NAT|C1"]["sd"] for m in V3}
power_r5["H4"] = dict(permutation=dict(unit="model", n_labellings=math.comb(7, 3), min_attainable_p=round(1 / math.comb(7, 3), 4), changed_by_n=False),
                      per_model_sign_reading=dict(sd_range_v3=[round(min(_nc.values()), 3), round(max(_nc.values()), 3)],
                                                  ci95_half_width_n100=round(Z95 * max(_nc.values()) / 10, 3), ci95_half_width_n200=round(Z95 * max(_nc.values()) / math.sqrt(200), 3),
                                                  noise_in_theta_hat_n100=round(max(_nc.values()) / 10, 3), noise_in_theta_hat_n200=round(max(_nc.values()) / math.sqrt(200), 3)))
power_r5["equivalence"] = dict(ci90_half_width_n100_sd_max=round(Z90 * SD_MAX / 10, 3), margin_about=0.21, attainable_abs_theta_below=round(0.21 - Z90 * SD_MAX / 10, 3))
power_r5["step0_criterion_1"] = dict(note="largest per-problem SD of delta that still passes, at n = 100, with the point estimate at its 0.10 limit and the smallest locked margin",
                                     sd_delta_max=round((0.2095 - 0.10) / Z90 * 10, 2))

# r5: v3's own numbers on the same subsets (the first 100 test ids and first 50 calibration ids of the fixed orders)
_q = json.load(open(f"{R}/e1_sample.json"))
_t100, _c50 = _q["tierA_ids"][:N_TEST], _q["calibration_train_ids"][:N_CAL]
subset_v3 = {}
for m in V3:
    row = {}
    for a, b in [("NAT", "C1"), ("NAT-K", "C1-K"), ("NAT-V", "C1-V"), ("NAT-K", "NAT-V")]:
        ra, rb = [{r["index"]: r for r in map(json.loads, open(f"{R}/v3_runs/{m}/tierA/{x}.jsonl"))} for x in (a, b)]
        d = [math.log(max(mean(ra[i]["js"]), 1e-9)) - math.log(max(mean(rb[i]["js"]), 1e-9)) for i in _t100]
        hw = Z95 * stdev(d) / 10
        row[f"{a}|{b}"] = dict(est_first_100=round(mean(d), 3), est_all_200=round(v3[m]["contrasts"][f"{a}|{b}"]["est"], 3), normal_ci95_first_100=[round(mean(d) - hw, 3), round(mean(d) + hw, 3)])
    recs = {t: {r["index"]: r for r in map(json.loads, open(f"{R}/v3_runs/{m}/calib/{t}.jsonl"))} for t in ("NAT", "C1_a0.5", "C1_a1", "C1_a2")}
    xs = [math.log(a) for a in (0.5, 1.0, 2.0)]
    xc = [x - mean(xs) for x in xs]

    def cal(ids):
        sl = []
        for i in ids:
            ys = [math.log(max(mean(recs[t][i]["js"]), 1e-9)) for t in ("C1_a0.5", "C1_a1", "C1_a2")]
            sl.append(sum(u * (y - mean(ys)) for u, y in zip(xc, ys)) / sum(u * u for u in xc))
        return round(mean(sl), 3)
    locked = json.load(open(f"{R}/v3_runs/{m}/calibration.json"))
    row["calibration"] = dict(b_hat_first_50=cal(_c50), b_hat_all_100=round(locked["dose_slope_b_hat"], 3), margin_locked=round(locked["margin_log_js"], 4))
    subset_v3[m] = row

out = dict(study_gpu=STUDY_GPU, v3=v3, sd_max=SD_MAX, c_host_s_per_layer=c_host, nf4_bytes_per_param=NF4_B, projection=proj,
           budget_r5=budget, volume_use_gb=dict(pinned_shards=round(shard_gb, 1), venv_assumed=VENV_GB, records_assumed=RECORDS_GB, total=round(storage_gb, 1), volume=VOLUME_GB),
           storage_gb=round(storage_gb, 0), mde_log_units=power, power_r5=power_r5, subset_v3=subset_v3,
           h4_min_perm_p_4v3=1 / math.comb(7, 3), h4_min_perm_p_4v3_onesided_note="exact over C(7,3)=35 labellings")
json.dump(out, open(sys.argv[2], "w"), indent=1)
for gpu in GPUS:
    p = proj[gpu]
    print(gpu, "ledger", p["gpu_ledger_h"], "h billed", p["billed_h"], "h $sec", p["usd_secure"], "$com", p["usd_community"])
for m, r in proj[STUDY_GPU]["models"].items():
    print(m, r)
print("storage", out["volume_use_gb"], "sdmax", SD_MAX, "c_host", c_host)
print(json.dumps(power))
print(json.dumps(budget, indent=1))
print(json.dumps(power_r5, indent=1))
print(json.dumps(subset_v3, indent=1))
