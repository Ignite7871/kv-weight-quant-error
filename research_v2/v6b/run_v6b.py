"""v6b driver (FOLLOWUP_V6B.md), laptop RTX 4060 only.

  python research_v2/v6b/run_v6b.py rehearse --model M     [GPU] needs v6b_runs/AUTHORIZED_V6B_REHEARSAL.txt: the first 4 v6 rehearsal train
                                                           ids, 6 arms, G1 against the v6 rehearsal records; writes timing and integrity only
  python research_v2/v6b/run_v6b.py rehearsal-summary      timing, integrity and G1 only (no contrast, SD, mean, share or verdict)
  python research_v2/v6b/run_v6b.py test --model M         [GPU] needs the lock and v6b_runs/AUTHORIZED_V6B_TEST.txt; resumable
  python research_v2/v6b/run_v6b.py ledger                 laptop GPU hours used (v6b ledger)

Per problem: history, then NAT-V (mu), C1-V, NAT-V-cen; G1 compares these with the v6 records of the same problem BEFORE any new arm runs;
a G1 failure writes g1_failure.json and stops. Then A, B, C. All six records of a problem are written together after the problem completes.
No arm uses attention logging (D5); mu comes from rv4.SlotSums through rv6b, with v6's arithmetic.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RV = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(RV / "v6"))
import rv6b  # noqa: E402
import run_v6 as R6  # noqa: E402  (load, history, seeds, Slot, make_sample: imported unchanged; R6.OUT is not used here)

import numpy as np  # noqa: E402

rv6 = rv6b.rv6
OUT = Path(os.environ.get("V6B_RUNS", RV / "v6b_runs"))
V6 = Path(os.environ.get("V6_RUNS", RV / "v6_runs"))
LOCK = RV / "V6B_PROTOCOL_LOCK.txt"
MODELS = ["qwen2.5-0.5b", "llama3.2-1b", "qwen2.5-1.5b"]
TIER = {"test": R6.TIERS["test"], "train": R6.TIERS["rehearsal"]}      # v6's tiers, so the reruns reproduce v6 bitwise
V6DIR = {"test": "test", "train": "rehearsal"}
N_TEST = 70
N_REH = 4
G1_TOL = 0.05
CAP_H = 9.0
FLAGS = rv6.FLAGS + ["b_construction_ok"]


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ledger(block, seconds, completed):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "gpu_ledger.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"utc": utc(), "block": block, "seconds": seconds, "completed": completed}) + "\n")


def used_hours():
    p = OUT / "gpu_ledger.jsonl"
    return sum(json.loads(l)["seconds"] for l in p.read_text(encoding="utf-8").splitlines() if l.strip()) / 3600 if p.exists() else 0.0


def logm(js):
    return math.log(max(float(np.mean(js)), 1e-9))


def load_recs(d):
    out = {}
    if d.exists():
        for f in d.glob("*.jsonl"):
            out[f.stem] = {json.loads(l)["index"]: json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()}
    return out


def design():
    p = OUT / "design.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"n": N_TEST, "cuts": []}


# ----------------------------------------------------------------------------- one arm, one problem
def run_arm(m, idx, split, arm, ids_t, ref, S, rl, ctx):
    rv6b.State.reset(arm)
    t0 = time.perf_counter()
    sd = R6.seeds(m, rv6b.SEED_ARM.get(arm, arm), idx, TIER[split])
    rec, _ = rv6.teacher_forced_arm_v6(R6.Slot.be, R6.Slot.mi, ids_t, ref, S, arm, sd, rl, ctx, logging=False)
    rec.update(index=idx, split=split, model=m, arm=arm, S=S, ref_len=len(ref), wall_seconds=time.perf_counter() - t0)
    if arm == "C1-V+mu":
        rec["b_norm_ratio"] = rv6b.State.b_ratio
        rec["b_construction_ok"] = bool(max(rv6b.State.b_dev, default=0.0) <= rv6b.B_TOL)
    else:
        rec["b_construction_ok"] = True
    mu = rv6b.mu_from_state() if arm == "NAT-V" else None
    return rec, mu


def g1(m, split, recs, v6recs):
    """Reruns vs the v6 records of the same problem: same S and history length, |d ln m_i| <= 0.05; bitwise js and top reported."""
    out, ok = {}, True
    for arm, r in recs.items():
        old = v6recs.get(arm, {}).get(r["index"])
        if old is None:
            out[arm] = {"pass": False, "reason": "no v6 record"}
            ok = False
            continue
        same = r["S"] == old["S"] and r["ref_len"] == old["ref_len"]
        d = abs(logm(r["js"]) - logm(old["js"]))
        p = same and d <= G1_TOL
        out[arm] = {"pass": p, "same_S_and_history_length": same, "abs_dlnm": d, "bitwise_js": r["js"] == old["js"], "same_top": r["top"] == old["top"]}
        ok = ok and p
    return ok, out


def run_problem(m, data, idx, split, rerun, v6recs):
    """Returns (records of this problem, G1 result). Stops before the new arms if G1 fails."""
    rv6b.install()
    ids_t, ref, S, rl = R6.history(data, split, idx)
    recs = {}
    rec, mu = run_arm(m, idx, split, "NAT-V", ids_t, ref, S, rl, {})
    recs["NAT-V"] = rec
    ctx = {"mu": mu.to(R6.DEVICE)}
    for arm in [a for a in ("C1-V", "NAT-V-cen") if a in rerun]:
        recs[arm], _ = run_arm(m, idx, split, arm, ids_t, ref, S, rl, ctx if arm in rv6b.NEEDS_MU else {})
    ok, res = g1(m, split, recs, v6recs)
    if not ok:
        return recs, (False, res)
    for arm in rv6b.NEW_ARMS:
        recs[arm], _ = run_arm(m, idx, split, arm, ids_t, ref, S, rl, ctx)
    return recs, (True, res)


def write(d: Path, recs):
    d.mkdir(parents=True, exist_ok=True)
    for arm, rec in recs.items():
        with open(d / f"{arm}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")


def integrity(recs):
    return [f"{a}:{k}" for a, r in recs.items() for k in FLAGS if not r.get(k, False)]


# ----------------------------------------------------------------------------- rehearsal
def rehearse(m):
    if not (OUT / "AUTHORIZED_V6B_REHEARSAL.txt").exists():
        sys.exit("refused: v6b_runs/AUTHORIZED_V6B_REHEARSAL.txt missing")
    s = R6.make_sample()
    data = R6.R4.RS1.Data()
    d = OUT / "rehearsal" / m
    v6recs = load_recs(V6 / "rehearsal" / m)
    t_all = time.perf_counter()
    R6.load(m)
    rep = {"model": m, "utc_start": utc(), "problems": []}
    for idx in s["rehearsal_train_ids"][:N_REH]:
        recs, (ok, res) = run_problem(m, data, idx, "train", rv6b.REPRO_ARMS, v6recs)
        write(d, recs)
        rep["problems"].append({"index": idx, "G1_pass": ok, "G1": res, "integrity_failures": integrity(recs),
                                "wall": {a: r["wall_seconds"] for a, r in recs.items()}})
        print(m, idx, "G1", ok, "fail", integrity(recs), flush=True)
        if not ok:
            break
    secs = time.perf_counter() - t_all
    rep["seconds"] = secs
    (d / "rehearsal_report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    ledger(f"rehearsal:{m}", secs, True)


def rehearsal_summary():
    summ = {"note": "timing, integrity and G1 only; no contrast, SD, mean, share or verdict is computed or printed (FOLLOWUP_V6B s9, r1)"}
    for m in MODELS:
        d = OUT / "rehearsal" / m
        if not (d / "rehearsal_report.json").exists():
            continue
        rep = json.loads((d / "rehearsal_report.json").read_text(encoding="utf-8"))
        walls = {a: float(np.mean([p["wall"][a] for p in rep["problems"] if a in p["wall"]])) for a in rv6b.ARMS}
        summ[m] = {"n_problems": len(rep["problems"]), "G1_all_pass": all(p["G1_pass"] for p in rep["problems"]),
                   "G1_all_bitwise": all(all(x.get("bitwise_js", False) for x in p["G1"].values()) for p in rep["problems"]),
                   "integrity_failures": sum(len(p["integrity_failures"]) for p in rep["problems"]),
                   "seconds_per_arm_mean": walls, "seconds_per_problem_6_arms": float(np.mean([sum(p["wall"].values()) for p in rep["problems"]])),
                   "block_seconds": rep["seconds"], "block_seconds_per_problem": rep["seconds"] / max(1, len(rep["problems"]))}
    (OUT / "rehearsal" / "rehearsal_summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1))


# ----------------------------------------------------------------------------- test phase
CUT1 = "rerun NAT-V only after 10 bitwise G1 problems"
N_BITWISE = 10


def first_bitwise(d, ids):
    """True iff the first N_BITWISE ids of this model all have a G1 entry with bitwise-equal js on all three rerun arms."""
    p = d / "g1.jsonl"
    if not p.exists():
        return False
    g = {json.loads(l)["index"]: json.loads(l)["G1"] for l in p.read_text(encoding="utf-8").splitlines() if l.strip()}
    head = ids[:N_BITWISE]
    return all(i in g and all(a in g[i] and g[i][a].get("bitwise_js") for a in rv6b.REPRO_ARMS) for i in head)


def rerun_for(des, k, ids, d):
    """Cut 1 (FOLLOWUP_V6B s8): after the first 10 problems of a model reproduced v6 bitwise, rerun only NAT-V (needed for mu)."""
    if CUT1 in des.get("cuts", []) and k >= N_BITWISE and first_bitwise(d, ids):
        return ["NAT-V"]
    return list(rv6b.REPRO_ARMS)


def test(m):
    if not LOCK.exists():
        sys.exit("refused: no V6B_PROTOCOL_LOCK.txt")
    if not (OUT / "AUTHORIZED_V6B_TEST.txt").exists():
        sys.exit("refused: v6b_runs/AUTHORIZED_V6B_TEST.txt missing")
    s = R6.make_sample()
    data = R6.R4.RS1.Data()
    d = OUT / "test" / m
    des = design()
    done = set(load_recs(d).get(rv6b.NEW_ARMS[-1], {}))
    v6recs = load_recs(V6 / "test" / m)
    ids = s["test_ids"][:des["n"]]
    R6.load(m)
    for k, idx in enumerate(ids):
        if idx in done:
            continue
        if used_hours() >= CAP_H:
            (OUT / "cap_stop.json").write_text(json.dumps({"utc": utc(), "model": m, "index": idx, "used_h": used_hours()}), encoding="utf-8")
            sys.exit("cap reached: stopped")
        t0 = time.perf_counter()
        recs, (ok, res) = run_problem(m, data, idx, "test", rerun_for(des, k, ids, d), v6recs)
        bad = integrity(recs)
        if not ok:
            ledger(f"test:{m}:{idx}:g1-fail", time.perf_counter() - t0, False)
            d.mkdir(parents=True, exist_ok=True)
            (d / "g1_failure.json").write_text(json.dumps({"index": idx, "G1": res}), encoding="utf-8")
            sys.exit(f"G1 failure on {idx}")
        write(d, recs)
        with open(d / "g1.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"index": idx, "G1": res}) + "\n")
        ledger(f"test:{m}:{idx}", time.perf_counter() - t0, True)
        if bad:
            (d / "integrity_failure.json").write_text(json.dumps({"index": idx, "failures": bad}), encoding="utf-8")
            sys.exit(f"integrity failure on {idx}: {bad}")
        print(m, idx, f"{time.perf_counter() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["rehearse", "rehearsal-summary", "test", "ledger"])
    ap.add_argument("--model", choices=MODELS)
    a = ap.parse_args()
    if a.cmd == "rehearse":
        rehearse(a.model)
    elif a.cmd == "rehearsal-summary":
        rehearsal_summary()
    elif a.cmd == "test":
        test(a.model)
    else:
        print(f"{used_hours():.3f} GPU-h used")


if __name__ == "__main__":
    main()
