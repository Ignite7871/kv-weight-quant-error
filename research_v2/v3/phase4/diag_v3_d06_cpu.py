"""CPU-only proxy for DEVIATION_LOG_V3 V3-D06 (before any further GPU use). Not locked; locked code is imported unchanged and only wrapped.

Same procedure as diag_norm_tolerance_cpu.py (run_v3 gate numerics loop: gate_numerics_train_ids, alpha 1, synthetic ctx, label qwen-base, tier
V3gate, same seeds, real Qwen2.5-0.5B on CPU), but on the amended (amendment 2) code, recording for every step:
  - every perturbed slot (target t > 0) whose realized post-rounding relative deviation |r - t| / t exceeds 5 % (layer, K/V, kv head), r = ||stored - kF||;
  - the TOTAL perturbation norm ratio over all slots of the step: sqrt(sum r^2) / sqrt(sum t^2) (exact rows contribute 0 to both).
Caveats: CPU and CUDA kernels differ; the synthetic context has sigma^2 == 1, so CV is isotropic here (in the test it uses the calibration context).
  venv/Scripts/python.exe research_v2/v3/phase4/diag_v3_d06_cpu.py --arms C1,CV,QO,C1-K,QA
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
    new, info = ORIG(arm, alpha, kF, kM, gen, vgen, ctx)              # exactly what the locked code stores (the cache holds it bitwise: rounding_ok)
    step = CUR["step"]
    CUR["step"] += 1
    if info["target"] is None:
        return new, info
    F = kF.double()
    tgt = info["target"].squeeze(-1)                                  # [2L, n_kv] float64
    ex = info["exact_rows"]
    pos = (~ex).unsqueeze(-1).expand_as(tgt) & (tgt > 0)
    rn = (new.double() - F).norm(dim=-1)
    rn = torch.where((~ex).unsqueeze(-1).expand_as(rn), rn, torch.zeros_like(rn))
    rdev = torch.where(pos, (rn - tgt).abs() / tgt.clamp_min(1e-300), torch.zeros_like(tgt))
    flagged = [[int(r) // 2, "K" if int(r) % 2 == 0 else "V", int(h), float(rdev[r, h])] for r, h in (rdev > V.RDEV_FLAG).nonzero().tolist()]
    tt, rt = float(tgt[pos].pow(2).sum().sqrt()), float(rn.pow(2).sum().sqrt())
    ROWS.append({"problem": CUR["idx"], "arm": arm, "step": step, "n_pos": int(pos.sum()), "flagged": flagged,
                 "target_total": tt, "realized_total": rt, "ratio_total": rt / tt if tt > 0 else None})
    return new, info


def q(x, p):
    x = sorted(x)
    return x[min(len(x) - 1, int(p * len(x)))]


def summarize(arms):
    out = {}
    for arm in arms:
        rr = [x for x in ROWS if x["arm"] == arm]
        fl = [f for x in rr for f in x["flagged"]]
        n = len(fl)
        rat = [x["ratio_total"] for x in rr if x["ratio_total"] is not None]
        lr = [abs(__import__("math").log(v)) for v in rat]
        out[arm] = {"n_steps": len(rr), "n_slots": sum(x["n_pos"] for x in rr), "n_flagged_slots": n,
                    "steps_with_flag": sum(1 for x in rr if x["flagged"]),
                    "frac_flagged_layer0": sum(f[0] == 0 for f in fl) / n if n else None,
                    "frac_flagged_K": sum(f[1] == "K" for f in fl) / n if n else None,
                    "frac_flagged_head0": sum(f[2] == 0 for f in fl) / n if n else None,
                    "frac_flagged_L0_K_h0": sum(f[0] == 0 and f[1] == "K" and f[2] == 0 for f in fl) / n if n else None,
                    "flagged_by_slot": {k: v for k, v in sorted(__import__("collections").Counter(f"L{f[0]} {f[1]} h{f[2]}" for f in fl).items(),
                                                                  key=lambda kv: -kv[1])[:8]},
                    "ratio_total": {"n": len(rat), "min": min(rat), "p01": q(rat, 0.01), "p05": q(rat, 0.05), "median": q(rat, 0.5), "p95": q(rat, 0.95),
                                    "p99": q(rat, 0.99), "max": max(rat), "mean": sum(rat) / len(rat)},
                    "abs_log_ratio_total": {"median": q(lr, 0.5), "p99": q(lr, 0.99), "max": max(lr)}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="C1,CV,QO,C1-K,QA")
    ap.add_argument("--problems", type=int, default=6)
    ap.add_argument("--steps", type=int, default=R.S_STEPS)
    ap.add_argument("--out", default=str(Path(__file__).with_name("diag_v3_d06_cpu.json")))
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
    out = {"device": "cpu", "model": M, "arms": arms, "steps": a.steps, "problems": s["gate_numerics_train_ids"][:a.problems], "per_step": ROWS}
    for idx in out["problems"]:
        for arm in arms:
            CUR.update(idx=idx, step=0)
            recs, _ = R.run_problem(be, tok, data, M, idx, "train", [(arm, 1.0, arm)], "V3gate", ctx, cap, predictors=False)
            errs = R.integrity_errors(recs, predictors=False)
            print(idx, arm, "rdev max", max(recs[arm]["rdev_max"]), "errs", errs, f"{time.time() - t0:.0f}s", flush=True)
            if errs:
                raise SystemExit(f"integrity errors: {errs}")
            Path(a.out).write_text(json.dumps(out))
    cap.remove()
    out["summary"] = summarize(arms)
    out["seconds"] = round(time.time() - t0, 1)
    Path(a.out).write_text(json.dumps(out))
    print(json.dumps(out["summary"], indent=1))


if __name__ == "__main__":
    main()
