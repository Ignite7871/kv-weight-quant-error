"""v6 driver (PROTOCOL_V6.md), laptop RTX 4060 only.

  python research_v2/v6/run_v6.py sample                         write v6/v6_sample.json (CPU; ids only)
  python research_v2/v6/run_v6.py rehearse --model M             [GPU] needs v6_runs/AUTHORIZED_V6_REHEARSAL.txt: G2, G3 (pilot models), G4 on
                                                                 real entries, 8 rehearsal train problems x 13 arms with logging + NAT-V without
  python research_v2/v6/run_v6.py rehearsal-summary              per-problem SDs and timing only (no contrast means, no alignment values)
  python research_v2/v6/run_v6.py test --model M                 [GPU] needs the lock and v6_runs/AUTHORIZED_V6_TEST.txt; resumable
  python research_v2/v6/run_v6.py ledger                         laptop GPU hours used
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RV = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(RV / "v4"))
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import rv6  # noqa: E402
import run_v4 as R4  # noqa: E402  (prompt_ids, fp16_history, the E1 data; imported unchanged)

import numpy as np  # noqa: E402
import torch  # noqa: E402

V = rv6.V
OUT = Path(os.environ.get("V6_RUNS", RV / "v6_runs"))
SAMPLE = HERE / "v6_sample.json"
LOCK = RV / "V6_PROTOCOL_LOCK.txt"
MODELS = ["qwen2.5-0.5b", "llama3.2-1b", "qwen2.5-1.5b"]
PILOT = {"qwen2.5-0.5b": "qwen-base", "llama3.2-1b": "llama-base"}
N_TEST, N_REH, N_REPRO = 100, 8, 3
TIERS = {"test": "V6A", "rehearsal": "V6reh", "gates": "V6gate"}
G3_TOL = 0.05
CAP_H = 16.0
DEVICE = "cuda"


def lf_sha(b: bytes) -> str:
    return hashlib.sha256(b.replace(b"\r\n", b"\n")).hexdigest()


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ledger(block, seconds, completed):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "gpu_ledger.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"utc": utc(), "block": block, "seconds": seconds, "completed": completed}) + "\n")


def used_hours():
    p = OUT / "gpu_ledger.jsonl"
    return sum(json.loads(l)["seconds"] for l in p.read_text(encoding="utf-8").splitlines() if l.strip()) / 3600 if p.exists() else 0.0


# ----------------------------------------------------------------------------- sample
def make_sample():
    if SAMPLE.exists():
        return json.loads(SAMPLE.read_text(encoding="utf-8"))
    e1 = json.loads((RV / "e1_sample.json").read_text(encoding="utf-8"))
    used_test = set(e1["tierA_ids"]) | set(e1["tierB_ids"]) | set(e1["gate_test_ids_for_G5"]) | set(e1["exposed_test_ids_in_sample"])
    stage12 = set()                                    # r3: the Stage 1/2 FP16 accuracy evaluations covered every test id; reported, not excluded
    for f in ("stage1_runs/eval/fp16.jsonl", "stage2_runs/eval/fp16.jsonl"):
        p = RV / f
        if p.exists():
            stage12 |= {json.loads(l)["index"] for l in p.read_text(encoding="utf-8").splitlines() if l.strip()}
    data = R4.RS1.Data()
    nfs = R4.RS1.N_FEWSHOT
    used_train = set(range(nfs)) | set(e1["calibration_train_ids"]) | set(e1["gate_train_ids"]) | set(e1["excluded_train_ids"])
    key = lambda lab, i: hashlib.sha256(f"{lab}|{i}".encode()).hexdigest()
    test = sorted((i for i in range(len(data.test)) if i not in used_test), key=lambda i: key("rv6-test-2026-11-15", i))[:N_TEST]
    reh = sorted((i for i in range(len(data.train)) if i not in used_train), key=lambda i: key("rv6-rehearsal-2026-11-15", i))[:N_REH]
    s_overlap = len(set(test) & stage12)
    s = {"what": "v6 sample (PROTOCOL_V6 section 3); ids only, no outcome data", "e1_sample_sha256_lf": lf_sha((RV / "e1_sample.json").read_bytes()),
         "n_test_ids_excluded": len(used_test), "n_test_ids_in_stage12_fp16_eval_of_the_sample": s_overlap, "n_train_ids_excluded": len(used_train), "test_ids": test, "rehearsal_train_ids": reh,
         "reproduction_test_ids": e1["tierA_ids"][:N_REPRO], "master_seed_label": rv6.MASTER_SEED, "tiers": TIERS,
         "prompt_sha256": {"test": {str(i): hashlib.sha256(data.problem("test", i)[1].encode()).hexdigest() for i in test},
                           "train": {str(i): hashlib.sha256(data.problem("train", i)[1].encode()).hexdigest() for i in reh}}}
    SAMPLE.write_text(json.dumps(s, indent=1) + "\n", encoding="utf-8", newline="\n")
    return s


# ----------------------------------------------------------------------------- model and one problem
class Slot:
    be = tok = mi = None


def load(m):
    Slot.be = Slot.tok = Slot.mi = None
    gc.collect()
    torch.cuda.empty_cache()
    rv4 = rv6.rv4
    rv4.apply_determinism()
    Slot.be, Slot.tok = rv4.load_model(m, DEVICE, "fp16", "nf4", local_only=True)
    Slot.mi = rv6.ModelInfo(Slot.be)


def seeds(m, arm, idx, tier):
    return (V.seed_for(m, arm, 1.0, idx, tier, master=rv6.MASTER_SEED), V.seed_for(m, f"{arm}-V", 1.0, idx, tier, master=rv6.MASTER_SEED))


def history(data, split, idx):
    ids_t = R4.prompt_ids(Slot.tok, data, split, idx, DEVICE)
    ref = R4.fp16_history(Slot.be, Slot.tok, ids_t)
    S = min(R4.S_STEPS, len(ref) - 1)
    rl = rv6.rv4.reference_logits(Slot.be, ids_t, ref, S)
    return ids_t, ref, S, rl


def run_arm(m, idx, split, tier, arm, ids_t, ref, S, rl, ctx, logging=True):
    t0 = time.perf_counter()
    rec, arrays = rv6.teacher_forced_arm_v6(Slot.be, Slot.mi, ids_t, ref, S, arm, seeds(m, arm, idx, tier), rl, ctx, logging=logging)
    rec.update(index=idx, split=split, model=m, arm=arm, S=S, ref_len=len(ref), wall_seconds=time.perf_counter() - t0)
    return rec, arrays


def run_problem(m, data, idx, split, tier, arms=rv6.ARMS):
    ids_t, ref, S, rl = history(data, split, idx)
    recs, arrs, mu = {}, {}, {}
    for arm in arms:
        ctx = {"mu": mu[rv6.CEN_SRC[arm]].to(DEVICE)} if arm in rv6.CEN_SRC else {}
        rec, arr = run_arm(m, idx, split, tier, arm, ids_t, ref, S, rl, ctx)
        recs[arm], arrs[arm] = rec, arr
        if arm in ("NAT-K", "NAT-V", "ST-V@nat"):
            mu[arm] = rv6.slot_mean(arr, rv6.PATH[arm])
    return recs, arrs, (ids_t, ref, S, rl)


def write(d: Path, idx, recs, arrs):
    d.mkdir(parents=True, exist_ok=True)
    (d / "arrays").mkdir(exist_ok=True)
    for arm, rec in recs.items():
        with open(d / f"{arm}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        if arrs.get(arm) is not None:
            np.savez_compressed(d / "arrays" / f"{arm}_{idx}.npz", **arrs[arm])


def integrity(recs):
    return [f"{a}:{k}" for a, r in recs.items() for k in rv6.FLAGS if not r[k]]


# ----------------------------------------------------------------------------- rehearsal
def g3(m, data, s):
    """NAT-K on the reproduction ids vs the pilot's laptop records."""
    lab = PILOT[m]
    pilot = {}
    for l in (RV / "v3_runs" / lab / "tierA" / "NAT-K.jsonl").read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            pilot[r["index"]] = r
    hist = {}
    for l in (RV / "v3_runs" / lab / "tierA" / "FP16_history.jsonl").read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            hist[r["index"]] = r["ids"]
    out = []
    for idx in s["reproduction_test_ids"]:
        ids_t, ref, S, rl = history(data, "test", idx)
        rec, _ = run_arm(m, idx, "test", TIERS["gates"], "NAT-K", ids_t, ref, S, rl, {}, logging=False)
        p = pilot[idx]
        a, b = np.asarray(rec["js"], float), np.asarray(p["js"], float)
        same_len = len(a) == len(b)
        bit = same_len and bool(np.array_equal(a, b))
        dmax = float(np.max(np.abs(np.log(np.maximum(a[1:], 1e-12)) - np.log(np.maximum(b[1:], 1e-12))))) if same_len else float("inf")
        lm = lambda js: math.log(max(float(np.mean(js)), 1e-9))
        heq = list(ref) == hist[idx]
        dlm = abs(lm(rec["js"]) - lm(p["js"]))
        out.append({"index": idx, "S": S, "pilot_S": p["S"], "history_token_identical": heq, "abs_dlnm": dlm, "pass": heq and dlm <= G3_TOL,
                    "descriptive_bitwise_js": bit, "descriptive_max_abs_dlnjs_per_step": dmax})
    return {"rule": "r4: token-identical history and |d ln m_i| <= 0.05 per problem", "problems": out, "passed": all(x["pass"] for x in out)}


def g4_real(arrs_sample):
    """codec on real kappa^F: torch vs numpy on the stored FP16 keys/values of the loaded model's last prefill."""
    ids = R4.prompt_ids(Slot.tok, arrs_sample[0], arrs_sample[1], arrs_sample[2], DEVICE)
    _, cache = Slot.be.run(ids, None, use_4bit=False)
    x = torch.cat([torch.stack([l.keys[0], l.values[0]]) for l in cache.layers]).double().reshape(-1, Slot.mi.hd)
    return bool(np.array_equal(rv6.codec_per_token(x).cpu().numpy(), rv6.codec_per_token_np(x.cpu().numpy())))


def rehearse(m):
    if not (OUT / "AUTHORIZED_V6_REHEARSAL.txt").exists():
        sys.exit("refused: v6_runs/AUTHORIZED_V6_REHEARSAL.txt missing")
    s = make_sample()
    data = R4.RS1.Data()
    d = OUT / "rehearsal" / m
    t_all = time.perf_counter()
    load(m)
    rep = {"model": m, "utc_start": utc(), "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "problems": []}
    import transformers, bitsandbytes
    rep.update(transformers=transformers.__version__, bitsandbytes=bitsandbytes.__version__, python=platform.python_version(),
               determinism=rv6.rv4.determinism_state())
    rep["G4_real"] = g4_real((data, "train", s["rehearsal_train_ids"][0]))
    if m in PILOT:
        rep["G3"] = g3(m, data, s)
    for idx in s["rehearsal_train_ids"]:
        recs, arrs, h = run_problem(m, data, idx, "train", TIERS["rehearsal"])
        ids_t, ref, S, rl = h
        off, _ = run_arm(m, idx, "train", TIERS["rehearsal"], "NAT-V", ids_t, ref, S, rl, {}, logging=False)
        g2 = off["js"] == recs["NAT-V"]["js"]
        write(d, idx, recs, arrs)
        with open(d / "NAT-V_nolog.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(off) + "\n")
        rep["problems"].append({"index": idx, "S": S, "G2_bitwise": g2, "integrity_failures": integrity(recs),
                                "wall": {a: r["wall_seconds"] for a, r in recs.items()}, "wall_NAT-V_nolog": off["wall_seconds"]})
        print(m, idx, "G2", g2, "fail", integrity(recs), flush=True)
    secs = time.perf_counter() - t_all
    rep["seconds"] = secs
    (d / "rehearsal_report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    ledger(f"rehearsal:{m}", secs, True)


# ----------------------------------------------------------------------------- rehearsal summary: SDs and timing only
CONTRASTS = {"H2 NAT-V vs C1-V": ("NAT-V", "C1-V"), "H1 NAT-K vs C1-K": ("NAT-K", "C1-K"), "H3 NAT-K vs NAT-V": ("NAT-K", "NAT-V"),
             "L3 theta_cm": ("NAT-V", "NAT-V-cen"), "L3 theta_dir": ("NAT-V-cen", "C1-V"), "L7ii ST-K@nat vs C1-K": ("ST-K@nat", "C1-K"),
             "L7ii ST-V@nat vs C1-V": ("ST-V@nat", "C1-V"), "L7iii ST-K vs ST-V": ("ST-K", "ST-V"), "L9a NAT-V-cen vs ST-V@nat-cen": ("NAT-V-cen", "ST-V@nat-cen")}


def logm(r):
    return math.log(max(float(np.mean(r["js"])), 1e-9))


def load_recs(d):
    out = {}
    for f in d.glob("*.jsonl"):
        out[f.stem] = {json.loads(l)["index"]: json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()}
    return out


def rehearsal_summary():
    summ = {"note": "per-problem SDs of outcome contrasts and timing only; no contrast means, no alignment values and no verdicts are computed or printed (PROTOCOL_V6 s10)"}
    for m in MODELS:
        d = OUT / "rehearsal" / m
        if not (d / "rehearsal_report.json").exists():
            continue
        rep = json.loads((d / "rehearsal_report.json").read_text(encoding="utf-8"))
        A = load_recs(d)
        idx = sorted(A["NAT-V"])
        sds = {}
        for name, (a, b) in CONTRASTS.items():
            x = np.array([logm(A[a][i]) - logm(A[b][i]) for i in idx])
            sds[name] = float(x.std(ddof=1))
        # no alignment quantity is read or computed here (author instruction, 2026-11-15): L8a/L8b/L9b power is stated in SD units
        walls = np.array([[p["wall"][a] for a in rv6.ARMS] for p in rep["problems"]])
        nolog = np.array([p["wall_NAT-V_nolog"] for p in rep["problems"]])
        withlog = np.array([p["wall"]["NAT-V"] for p in rep["problems"]])
        summ[m] = {"n_problems": len(idx), "per_problem_sd": sds, "G2_all_bitwise": all(p["G2_bitwise"] for p in rep["problems"]),
                   "integrity_failures": sum(len(p["integrity_failures"]) for p in rep["problems"]), "G3": rep.get("G3"), "G4_real": rep["G4_real"],
                   "seconds_per_problem_all_13_arms": float(walls.sum(1).mean()), "seconds_per_arm_mean": {a: float(walls[:, k].mean()) for k, a in enumerate(rv6.ARMS)},
                   "logging_overhead_factor_NAT-V": float((withlog / nolog).mean()), "block_seconds": rep["seconds"]}
    (OUT / "rehearsal" / "rehearsal_summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1))


# ----------------------------------------------------------------------------- test phase
def test(m):
    if not LOCK.exists():
        sys.exit("refused: no V6_PROTOCOL_LOCK.txt")
    if not (OUT / "AUTHORIZED_V6_TEST.txt").exists():
        sys.exit("refused: v6_runs/AUTHORIZED_V6_TEST.txt missing")
    s = make_sample()
    data = R4.RS1.Data()
    d = OUT / "test" / m
    cut = json.loads((OUT / "design.json").read_text(encoding="utf-8")) if (OUT / "design.json").exists() else {"n": N_TEST, "arms": rv6.ARMS}
    done = set(load_recs(d).get(cut["arms"][-1], {})) if d.exists() else set()
    load(m)
    for idx in s["test_ids"][:cut["n"]]:
        if idx in done:
            continue
        if used_hours() >= CAP_H:
            (OUT / "cap_stop.json").write_text(json.dumps({"utc": utc(), "model": m, "index": idx, "used_h": used_hours()}), encoding="utf-8")
            sys.exit("cap reached: stopped")
        t0 = time.perf_counter()
        recs, arrs, _ = run_problem(m, data, idx, "test", TIERS["test"], cut["arms"])
        bad = integrity(recs)
        write(d, idx, recs, arrs)
        ledger(f"test:{m}:{idx}", time.perf_counter() - t0, True)
        if bad:
            (d / "integrity_failure.json").write_text(json.dumps({"index": idx, "failures": bad}), encoding="utf-8")
            sys.exit(f"integrity failure on {idx}: {bad}")
        print(m, idx, f"{time.perf_counter() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sample", "rehearse", "rehearsal-summary", "test", "ledger"])
    ap.add_argument("--model", choices=MODELS)
    a = ap.parse_args()
    if a.cmd == "sample":
        s = make_sample()
        print({k: (len(v) if isinstance(v, list) else v) for k, v in s.items() if k != "prompt_sha256"})
    elif a.cmd == "rehearse":
        rehearse(a.model)
    elif a.cmd == "rehearsal-summary":
        rehearsal_summary()
    elif a.cmd == "test":
        test(a.model)
    else:
        print(f"{used_hours():.3f} GPU-h used")


if __name__ == "__main__":
    main()
