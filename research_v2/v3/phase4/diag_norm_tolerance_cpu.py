"""CPU-only diagnosis of the qwen-base G2/G3 failure (DEVIATION_LOG_V3 V3-D04). No GPU; locked code is imported unchanged and only wrapped.

Re-runs the gate's numerics loop (run_v3.gpu_gates: gate_numerics_train_ids, alpha 1, synthetic ctx, label qwen-base, tier V3gate, same seeds) on
CPU with the real Qwen2.5-0.5B, and records for every perturbation slot:
  target ||stored - kF|| intended, realized ||stored - kF||, entry norm ||kF||, E1 floor 2^-10 ||kF||,
  rel_current = |realized - target| / (target + floor)          (v3 and E1 code, rv2/controls.py:116)
  rel_maxrule = |realized - target| / max(target, floor)
  construction error = | ||u - kF|| - target | / (target + floor), u = the same entry computed WITHOUT the final FP16 rounding (same generator state)
  rounding norm = ||stored - u||
Caveat: CPU and CUDA kernels differ, so the CPU slot need not be the one that failed on the GPU (the gate saved no per-slot data).
  venv/Scripts/python.exe research_v2/v3/phase4/diag_norm_tolerance_cpu.py --arms CV [--problems 6] [--steps 64]
"""
import argparse
import json
import sys
import time
from pathlib import Path

RV = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RV))

import torch                                     # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

import run_v3 as R                               # noqa: E402
import rv3.core as V                             # noqa: E402

import e1l_support as LS                         # noqa: E402

M = "qwen-base"
CUR = {}
ROWS = []
ORIG = V.stored_entry_v3


def wrapped(arm, alpha, kF, kM, gen, vgen, ctx):
    gs = gen.get_state() if gen is not None else None
    vs = vgen.get_state() if vgen is not None else None
    u, _ = ORIG(arm, alpha, kF.float(), kM, gen, vgen, ctx)          # same draws, no final rounding to FP16
    if gen is not None:
        gen.set_state(gs)
    if vgen is not None:
        vgen.set_state(vs)
    new, info = ORIG(arm, alpha, kF, kM, gen, vgen, ctx)              # exactly what the locked code stores
    step = CUR["step"]
    CUR["step"] += 1
    if info["target"] is None:
        return new, info
    F = kF.float()
    tgt = info["target"].float().squeeze(-1)                          # [2L, n_kv]
    rn = (new.float() - F).norm(dim=-1)
    un = (u.float() - F).norm(dim=-1)
    Fn = F.norm(dim=-1)
    floor = V.FP16_RES * Fn
    ex = info["exact_rows"]
    rel_cur = (rn - tgt).abs() / (tgt + floor)
    rel_max = (rn - tgt).abs() / torch.maximum(tgt, floor).clamp_min(1e-30)
    cons = (un - tgt).abs() / (tgt + floor)
    rnd = (new.float() - u.float()).norm(dim=-1)
    rel_cur[ex] = 0.0
    rel_max[ex] = 0.0
    cons[ex] = 0.0
    k = int(rel_cur.argmax())
    r, h = divmod(k, rel_cur.shape[1])
    ROWS.append({"problem": CUR["idx"], "arm": arm, "step": step, "layer": r // 2, "path": "K" if r % 2 == 0 else "V", "kv_head": h,
                 "target": float(tgt[r, h]), "realized": float(rn[r, h]), "entry_norm": float(Fn[r, h]), "stored_entry_norm": float(new[r, h].float().norm()),
                 "e1_floor": float(floor[r, h]), "rel_current": float(rel_cur[r, h]), "rel_maxrule": float(rel_max[r, h]),
                 "construction_err": float(cons[r, h]), "rounding_norm": float(rnd[r, h]), "target_over_floor": float(tgt[r, h] / floor[r, h]),
                 "step_mean_rel_current": float(rel_cur[~ex].mean()) if bool((~ex).any()) else 0.0,
                 "step_max_construction_err": float(cons.max()), "n_slots_rel_gt_0p05": int((rel_cur > 0.05).sum())})
    return new, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="CV")
    ap.add_argument("--problems", type=int, default=6)
    ap.add_argument("--steps", type=int, default=R.S_STEPS)
    ap.add_argument("--out", default=str(Path(__file__).with_name("diag_norm_tolerance_cpu.json")))
    a = ap.parse_args()
    arms = a.arms.split(",")
    R.S_STEPS = a.steps
    V.stored_entry_v3 = wrapped
    torch.manual_seed(0)
    mid = R.MODELS[M]
    be = LS.build_backend(AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16), "cpu")
    tok = AutoTokenizer.from_pretrained(mid)
    data = R.RS1.Data()
    s = R.make_sample()
    ctx = V.synthetic_ctx(be.model.config, be.device)                 # exactly as gpu_gates
    cap = V.QueryCapture(be.model)
    t0 = time.time()
    out = {"device": "cpu", "model": M, "arms": arms, "steps": a.steps, "problems": s["gate_numerics_train_ids"][:a.problems], "per_step_max": ROWS}
    for idx in out["problems"]:
        for arm in arms:
            CUR.update(idx=idx, step=0)
            recs, _ = R.run_problem(be, tok, data, M, idx, "train", [(arm, 1.0, arm)], "V3gate", ctx, cap, predictors=False)
            errs = R.integrity_errors(recs, predictors=False)
            print(idx, arm, "max rel", max(recs[arm]["norm_dev_max"]), "errs", errs, f"{time.time() - t0:.0f}s", flush=True)
            Path(a.out).write_text(json.dumps(out, indent=1))
    cap.remove()
    summ = {}
    for arm in arms:
        rr = [x for x in ROWS if x["arm"] == arm]
        w = max(rr, key=lambda x: x["rel_current"])
        summ[arm] = {"max_rel_current": w["rel_current"], "worst_slot": w, "max_rel_maxrule": max(x["rel_maxrule"] for x in rr),
                     "max_construction_err": max(x["step_max_construction_err"] for x in rr),
                     "max_step_mean_rel_current": max(x["step_mean_rel_current"] for x in rr), "n_steps": len(rr)}
    out["summary"] = summ
    out["seconds"] = round(time.time() - t0, 1)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
