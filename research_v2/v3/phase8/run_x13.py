"""X13 driver (v3/phase8/FOLLOWUP_X13.md). EXPLORATORY follow-up outside the locked v3 protocol; the locked and X9 records are read only.

Subcommands (CPU unless marked)
  subset                         write v3/phase8/x13_subset.json (the X9 subset, recomputed from seed 20261209; must equal x9_subset.json)
  manifest                       write v3/phase8/X13_MANIFEST.txt and print the X13 manifest id
  status | ledger                progress / X13 GPU time (own ledger, 2.0 h ceiling)
  run --model M                  [GPU] NAT, NAT-K-cen, NAT-V-cen (+ C1-nc, C1-K-nc in Qwen) on the subset, bitwise NAT gate

GPU use is REFUSED unless ALL of: --authorized-gpu-run; research_v2/v3_runs_x13/AUTHORIZED_X13_BY_AUTHORS.txt exists and contains the text
"FOLLOWUP_X13" and the X13 manifest id; the listed files (and the calibration aux_state.pt files) still hash to the id in X13_MANIFEST.txt;
the v3, E1 and E1L locks and the model's weight hash verify (run_v3.locks_ok); the model is the next in the fixed order, every earlier model
completed, and no reproduction mismatch is on file. Only the corresponding author creates the authorization file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

P8 = Path(__file__).resolve().parent
RV = P8.parents[1]
sys.path.insert(0, str(RV))
sys.path.insert(0, str(P8))

import run_v3 as R                              # noqa: E402  (locked v3 driver, imported unchanged; HF before torch)

import numpy as np                              # noqa: E402
import torch                                    # noqa: E402

import rv3.core as V                            # noqa: E402
import rv3x13 as X                              # noqa: E402

OUT = RV / "v3_runs_x13"
LOCKED_RUNS = RV / "v3_runs"
AUTH = OUT / "AUTHORIZED_X13_BY_AUTHORS.txt"
DESIGN = OUT / "design_x13.json"
SUBSET = P8 / "x13_subset.json"
X9_SUBSET = RV / "v3" / "phase7" / "x9_subset.json"
MANIFEST = P8 / "X13_MANIFEST.txt"
MANIFEST_FILES = ["FOLLOWUP_X13.md", "rv3x13.py", "run_x13.py", "analyze_x13.py", "test_x13.py", "mutation_check_x13.py", "x13_subset.json"]
AUTH_TEXT = "FOLLOWUP_X13"
CAP_HOURS = 2.0
PROJ_LIMIT_H = 1.9
N_SUBSET, N_REDUCED, SUBSET_SEED = 50, 40, 20261209
N_PROJECT = 3
RAW_FIRST = 3                                   # raw Delta saved for the first 3 subset problems of each model (audit only)
TIER = "V3A"                                    # the locked test tier label (seed streams)
ORDER = ["qwen-base", "qwen-instruct", "llama-base", "llama-instruct"]
ARMS_LLAMA = [("NAT", 1.0, "NAT"), ("NAT-K-cen", 1.0, "NAT-K-cen"), ("NAT-V-cen", 1.0, "NAT-V-cen")]
ARMS_QWEN = ARMS_LLAMA + [("C1-nc", 1.0, "C1-nc"), ("C1-K-nc", 1.0, "C1-K-nc")]
GATED = ("NAT",)
IGNORED = {"wall_seconds"}
X13_FIELDS = {"P_logit_c", "PL_logit_c"}
REDUCTIONS = ["n = 40 in Llama", "n = 40 in every model"]


def family(m):
    return m.split("-")[0]


def arms_for(m, design=None):
    return ARMS_QWEN if family(m) == "qwen" else ARMS_LLAMA


# ----------------------------------------------------------------------------- subset and manifest
def lf_bytes(p: Path) -> bytes:
    return p.read_bytes().replace(b"\r\n", b"\n")


def subset_ids(tier_a_ids, seed=None):
    return sorted(int(x) for x in np.random.default_rng(SUBSET_SEED if seed is None else seed).choice(sorted(tier_a_ids), N_SUBSET, replace=False))


def make_subset():
    if SUBSET.exists():
        sys.exit(f"{SUBSET.name} exists already (fixed); refusing to overwrite")
    s = json.loads((RV / "v3_sample.json").read_text())
    ids = subset_ids(s["tierA_ids"])
    if ids != json.loads(X9_SUBSET.read_text())["ids"]:
        sys.exit("the recomputed draw does not reproduce x9_subset.json; refusing to write")
    out = {"ids": ids, "n": len(ids), "seed": SUBSET_SEED,
           "draw": "numpy.random.default_rng(seed).choice(sorted(tierA_ids), 50, replace=False), sorted (identical to v3/phase7/x9_subset.json)",
           "source": "research_v2/v3_sample.json", "source_sha256_lf": hashlib.sha256(lf_bytes(RV / "v3_sample.json")).hexdigest(),
           "exposed_in_subset": sorted(set(ids) & set(s["exposed_test_ids_in_sample"])), "no outcome data used": True}
    SUBSET.write_text(json.dumps(out, indent=1) + "\n")
    return out


def aux_lines(runs: Path = LOCKED_RUNS):
    return [f"aux_state {m} {hashlib.sha256((runs / m / 'calib' / 'aux_state.pt').read_bytes()).hexdigest()}" for m in ORDER]


def manifest_id(root: Path = P8, files=None, runs: Path = LOCKED_RUNS) -> str:
    h = hashlib.sha256()
    for name in files or MANIFEST_FILES:
        b = lf_bytes(root / name)
        h.update(f"{name}\n{len(b)}\n".encode())
        h.update(b)
    for line in aux_lines(runs):
        h.update((line + "\n").encode())
    return h.hexdigest()[:16]


def write_manifest():
    mid = manifest_id()
    lines = [f"X13 manifest id: {mid}", f"utc: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
             "files (in hashing order; LF-normalized; each prefixed by name and byte length):"]
    lines += [f"  {n}  sha256(LF) {hashlib.sha256(lf_bytes(P8 / n)).hexdigest()}" for n in MANIFEST_FILES]
    lines += ["then, in this order (binary sha256 of research_v2/v3_runs/<model>/calib/aux_state.pt):"] + [f"  {x}" for x in aux_lines()]
    lines += ["", f"Authorization (created only by the corresponding author, FOLLOWUP_X13 decision D8): {AUTH.relative_to(RV.parent)} "
                  f"containing the text {AUTH_TEXT} and the id {mid}."]
    MANIFEST.write_text("\n".join(lines) + "\n")
    return mid


def recorded_manifest_id():
    if not MANIFEST.exists():
        return None
    first = MANIFEST.read_text(encoding="utf-8").splitlines()[0]
    return first.split(":", 1)[1].strip() if first.startswith("X13 manifest id:") else None


def auth_ok(text: str, mid: str) -> bool:
    return bool(mid) and AUTH_TEXT in text and mid in text


# ----------------------------------------------------------------------------- reproduction comparator (bitwise)
def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def same(a, b) -> bool:
    """Exact equality after the JSON round trip: floats bitwise (NaN equals NaN, the sign of zero counts), lists elementwise, same types."""
    if isinstance(a, float) and isinstance(b, float):
        if a != a or b != b:
            return a != a and b != b
        return a == b and math.copysign(1.0, a) == math.copysign(1.0, b)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def _diff_detail(a, b):
    d = {"equal": False}
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        bad = [k for k, (x, y) in enumerate(zip(a, b)) if not same(x, y)]
        d.update(n_differing_steps=len(bad), first_differing_step=bad[0] + 1 if bad else None)
        pairs = [(x, y) for x, y in zip(a, b) if _num(x) and _num(y) and x == x and y == y]
        if pairs:
            ab = [abs(x - y) for x, y in pairs]
            d.update(max_abs_diff=max(ab), max_rel_diff=max(abs(x - y) / max(abs(y), 1e-300) for x, y in pairs if x != y) if any(ab) else 0.0)
    else:
        d.update(new=a if not isinstance(a, list) else f"list[{len(a)}]", locked=b if not isinstance(b, list) else f"list[{len(b)}]")
    return d


def compare_records(new: dict, locked: dict) -> dict:
    """Field -> difference detail for every field that differs (empty dict = bitwise identical). `new` is JSON round-tripped first."""
    new = json.loads(json.dumps(new))
    out = {}
    lk, nk = set(locked) - IGNORED, set(new) - IGNORED - X13_FIELDS
    for k in sorted(lk ^ nk):
        out[k] = {"equal": False, "missing_in": "new" if k in lk else "locked"}
    for k in sorted(lk & nk):
        if not same(new[k], locked[k]):
            out[k] = _diff_detail(new[k], locked[k])
    return out


def history_matches(regen, locked_ids) -> bool:
    return [int(x) for x in regen] == [int(x) for x in locked_ids]


# ----------------------------------------------------------------------------- integrity (section 3)
def integrity_x13(recs):
    """The locked integrity rule, except that for the nc arms the float64 construction check (which measures the post-correction residual
    against the natural target) is replaced by nc_ok: the corrected residual is <= the nearest-rounding residual in every slot and step."""
    errs = []
    for e in R.integrity_errors(recs):
        tag = e.split(":", 1)[0]
        if tag in X.NC_ARMS and e.startswith(f"{tag}: construction"):
            continue
        errs.append(e)
    for tag, r in recs.items():
        if tag in X.NC_ARMS and not all(r.get("nc_ok") or [False]):
            errs.append(f"{tag}: norm-corrected residual exceeds the nearest-rounding residual (nc_ok false)")
    return errs


# ----------------------------------------------------------------------------- one problem's arms
def run_arms(be, ids_t, ref, S_i, arms, ctx, cap, m, idx, raw=False):
    """The locked run_problem's arm loop (same record fields), with the nc arms on the locked seed stream and mu handed from NAT to the centred
    arms. Returns (records, npz arrays per arm)."""
    rl = V.reference_logits_v3(be, ids_t, ref, S_i, None, None, be.model.config.num_key_value_heads)
    X.clear()
    ctx.pop("mu", None)
    recs, slots = {}, {}
    try:
        for arm, alpha, tag in arms:
            if arm in X.CEN_ARMS and "mu" not in ctx:
                raise RuntimeError(f"{arm}: NAT must run first in the same problem (mu)")
            X.STEP_LOG.clear()
            n_inst = len(X.INSTANCES)
            t0 = time.perf_counter()
            out = V.teacher_forced_arm_v3(be, ids_t, ref, S_i, arm, alpha, V.arm_seeds(m, X.SEED_LABEL.get(arm, arm), alpha, idx, TIER), rl,
                                          ctx, cap, True)
            rec = {"index": idx, "split": "test", "model": m, "arm": tag, "alpha": alpha, "S": S_i, "ref_len": len(ref), **out,
                   "wall_seconds": time.perf_counter() - t0}
            if len(X.INSTANCES) != n_inst + 1:
                raise RuntimeError(f"{arm}: expected one predictor state")
            inst = X.INSTANCES[-1]
            rec.update(X.record_fields(inst))
            if arm != "NAT":
                rec.update(X.step_fields(list(X.STEP_LOG)))
            slots[tag] = X.slot_arrays(inst, nat=(arm == "NAT"), raw=raw)
            if arm == "NAT":
                ctx["mu"] = X.mu_from(inst)
            recs[tag] = rec
    finally:
        ctx.pop("mu", None)
    return recs, slots


# ----------------------------------------------------------------------------- design (pre-declared reductions, runtime only)
def n_for(m, design):
    return design["n"][m]


def full_design():
    return {"n": {m: N_SUBSET for m in ORDER}, "applied": []}


def apply_reduction(design, k):
    d = json.loads(json.dumps(design))
    if k == 0:
        d["n"].update({m: N_REDUCED for m in ORDER if family(m) == "llama"})
    else:
        d["n"] = {m: N_REDUCED for m in ORDER}
    d["applied"].append(REDUCTIONS[k])
    return d


def projection_h(design, used_s, t_arm, t_over, t_load, done_first):
    """Projected X13 total (h): used so far + the remaining problems of every model (Llama scaled by the locked LLAMA_TIME_RATIO, with its own
    arm count) + a model load for each later model; x 1.05."""
    tot = used_s
    for k, m in enumerate(ORDER):
        ratio = 1.0 if family(m) == "qwen" else R.LLAMA_TIME_RATIO
        remaining = max(0, n_for(m, design) - (done_first if k == 0 else 0))
        tot += remaining * (len(arms_for(m)) * t_arm + t_over) * ratio + (t_load if k > 0 else 0.0)
    return tot * 1.05 / 3600


def choose_design(used_s, t_arm, t_over, t_load, done_first):
    design = full_design()
    steps = [{"design": "full (option A)", "projected_h": projection_h(design, used_s, t_arm, t_over, t_load, done_first)}]
    k = 0
    while steps[-1]["projected_h"] > PROJ_LIMIT_H and k < len(REDUCTIONS):
        design = apply_reduction(design, k)
        steps.append({"design": REDUCTIONS[k], "projected_h": projection_h(design, used_s, t_arm, t_over, t_load, done_first)})
        k += 1
    design.update(projection_steps=steps, pass_=steps[-1]["projected_h"] <= PROJ_LIMIT_H,
                  measured={"t_arm_s": t_arm, "t_overhead_s": t_over, "t_load_s": t_load, "used_s": used_s, "problems": done_first})
    return design


# ----------------------------------------------------------------------------- guards
def configure(m):
    OUT.mkdir(parents=True, exist_ok=True)
    R.RS1.OUT, R.RS1.LEDGER, R.RS1.PROGRESS = OUT, OUT / "gpu_ledger.jsonl", OUT / "current_process.json"
    R.RS1.MODEL = R.MODELS[m] if m else R.RS1.MODEL


def mismatch_files():
    return sorted(OUT.glob("*/REPRODUCTION_MISMATCH.json"))


def complete(m):
    return (OUT / m / "x13_complete.json").exists()


def require_gpu_permission(a, m):
    if not a.authorized_gpu_run:
        sys.exit("Refusing to run: X13 GPU runs need explicit author authorization (--authorized-gpu-run).")
    mid, rec = manifest_id(), recorded_manifest_id()
    if rec is None or mid != rec:
        sys.exit(f"Refusing to run: the X13 files hash to {mid}, but X13_MANIFEST.txt records {rec}.")
    txt = AUTH.read_text(encoding="utf-8", errors="replace") if AUTH.exists() else ""
    if not auth_ok(txt, mid):
        sys.exit(f"Refusing to run: {AUTH.relative_to(RV.parent)} must exist and contain '{AUTH_TEXT}' and the X13 manifest id {mid}. "
                 "Only the corresponding author creates it.")
    if not R.locks_ok(m):
        sys.exit("Refusing to run: the v3 lock (or the E1 / E1L lock, or this model's weight-file hash) does not verify.")
    if mismatch_files():
        sys.exit(f"Refusing to run: a reproduction mismatch is on file ({mismatch_files()[0]}); the authors decide what follows.")
    for prev in ORDER[:ORDER.index(m)]:
        if not complete(prev):
            sys.exit(f"Refusing to run: fixed order; {prev} has not completed X13.")
    if ORDER.index(m) > 0 and not (DESIGN.exists() and json.loads(DESIGN.read_text()).get("pass_")):
        sys.exit("Refusing to run: the X13 design (runtime projection of the first model) is missing or did not fit the ceiling.")


# ----------------------------------------------------------------------------- per-problem work
def load_locked(m, tag):
    out = {}
    for line in (LOCKED_RUNS / m / "tierA" / f"{tag}.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r["index"] in out:
                raise ValueError(f"duplicate locked record {m} {tag} {r['index']}")
            out[r["index"]] = r
    return out


def run_one(be, tok, data, m, idx, arms, ctx, cap, locked, raw):
    ids_t = R.prompt_ids(tok, data, "test", idx, be.device)
    t0 = time.perf_counter()
    regen = R.fp16_history(be, tok, ids_t)
    t_hist = time.perf_counter() - t0
    hist = locked["FP16_history"][idx]["ids"]
    S_i = min(R.S_STEPS, len(hist) - 1)
    recs, slots = run_arms(be, ids_t, hist, S_i, arms, ctx, cap, m, idx, raw)
    rep = {"index": idx, "history_matches_locked": history_matches(regen, hist), "t_history_s": t_hist,
           "integrity_errors": integrity_x13(recs), "diffs": {t: compare_records(recs[t], locked[t][idx]) for t in GATED}}
    rep["ok"] = bool(rep["history_matches_locked"] and not rep["integrity_errors"] and not any(rep["diffs"].values()))
    return recs, slots, rep


def mismatch_report(m, rep, recs, locked):
    idx = rep["index"]
    return {"model": m, "problem": idx, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "history_matches_locked": rep["history_matches_locked"], "integrity_errors": rep["integrity_errors"],
            "differing_fields": rep["diffs"],
            "tokens": {t: {"top_matches": same(json.loads(json.dumps(recs[t]["top"])), locked[t][idx]["top"]),
                           "flip_matches": same(json.loads(json.dumps(recs[t]["flip"])), locked[t][idx]["flip"])} for t in GATED},
            "policy": "FOLLOWUP_X13 section 6 / D4: stop and report; no tolerance; the authors decide"}


def save(m, idx, recs, slots, rep):
    d = OUT / m
    (d / "slots").mkdir(parents=True, exist_ok=True)
    for tag, arr in slots.items():
        np.savez_compressed(d / "slots" / f"{tag}_{idx}.npz", **arr)
    R.E.append_records(d, {"reproduction": {k: v for k, v in rep.items() if k != "diffs"} | {"bitwise_equal": {t: not v for t, v in rep["diffs"].items()}}})
    R.E.append_records(d, recs)                                    # records last: a problem counts as done only once everything is written


def done(m, idx):
    return all(idx in R.done_ids(OUT / m, tag) for _, _, tag in arms_for(m))


def run_model(be, tok, data, m, ids, prior, t0, a):
    cal = json.loads((LOCKED_RUNS / m / "calibration.json").read_text())
    ctx = V.ctx_from_calibration(cal, be.device)
    M = torch.load(LOCKED_RUNS / m / "calib" / "aux_state.pt", map_location="cpu", weights_only=False)["M"].to(be.device)
    X.install(M)
    cap = V.QueryCapture(be.model)
    locked = {t: load_locked(m, t) for t in ("FP16_history",) + GATED}
    first = ORDER.index(m) == 0
    design = json.loads(DESIGN.read_text()) if DESIGN.exists() else None
    state = {"n": 0, "t_first": None, "per_problem": []}

    def step(idx):
        nonlocal design
        if design is not None and ids.index(idx) >= n_for(m, design):
            return True
        tp = time.perf_counter()
        if state["t_first"] is None:
            state["t_first"] = time.time() - t0
        recs, slots, rep = run_one(be, tok, data, m, idx, arms_for(m), ctx, cap, locked, ids.index(idx) < RAW_FIRST)
        if not rep["ok"]:
            (OUT / m).mkdir(parents=True, exist_ok=True)
            (OUT / m / "REPRODUCTION_MISMATCH.json").write_text(json.dumps(mismatch_report(m, rep, recs, locked), indent=1, default=float))
            print(f"[REPRODUCTION MISMATCH / INTEGRITY] {m} problem {idx}: stopping (FOLLOWUP_X13 section 6)", flush=True)
            return False
        save(m, idx, recs, slots, rep)
        state["n"] += 1
        state["per_problem"].append((time.perf_counter() - tp, [recs[t]["wall_seconds"] for _, _, t in arms_for(m)]))
        if first and design is None and state["n"] == N_PROJECT:
            t_arm = float(np.mean([w for _, ws in state["per_problem"] for w in ws]))
            t_over = float(np.mean([tot - sum(ws) for tot, ws in state["per_problem"]]))
            design = choose_design(prior + time.time() - t0, t_arm, t_over, state["t_first"], N_PROJECT)
            design["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            DESIGN.write_text(json.dumps(design, indent=1, default=float))
            print("[X13 design]", json.dumps(design["projection_steps"]), flush=True)
            if not design["pass_"]:
                print(f"[X13 design] projection exceeds {PROJ_LIMIT_H} h after every pre-declared reduction: stop and ask the authors", flush=True)
                return False
        return True

    try:
        ok = R.E.block_loop(f"x13:{m}", be, tok, None, prior, t0, a.cap_hours, ids, step,
                            lambda i: done(m, i) or (design is not None and ids.index(i) >= n_for(m, design)))
    finally:
        cap.remove()
        X.uninstall()
    design = json.loads(DESIGN.read_text()) if DESIGN.exists() else None
    if ok and design is not None and all(done(m, i) for i in ids[:n_for(m, design)]):
        (OUT / m / "x13_complete.json").write_text(json.dumps({"model": m, "n": n_for(m, design), "arms": [t for _, _, t in arms_for(m)],
                                                               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
    return ok


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["subset", "manifest", "status", "ledger", "run"])
    ap.add_argument("--model", choices=ORDER)
    ap.add_argument("--authorized-gpu-run", action="store_true")
    ap.add_argument("--cap-hours", type=float, default=CAP_HOURS)
    a = ap.parse_args()
    if a.cap_hours > CAP_HOURS:
        sys.exit(f"cap above the approved {CAP_HOURS} h is not permitted")
    if a.cmd == "subset":
        print(make_subset())
        return
    if a.cmd == "manifest":
        print("X13 manifest id:", write_manifest())
        return
    configure(a.model)
    if a.cmd == "ledger":
        print(f"X13 GPU wall time: {R.RS1.ledger_seconds() / 3600:.3f} h of {a.cap_hours} h")
        return
    if a.cmd == "status":
        print({"manifest_files_id": manifest_id(), "recorded_id": recorded_manifest_id(), "authorization_file": AUTH.exists(),
               "design": DESIGN.exists(), "mismatches": [str(p) for p in mismatch_files()], "complete": {m: complete(m) for m in ORDER}})
        return
    if not a.model:
        sys.exit("--model is required")
    m = a.model
    require_gpu_permission(a, m)
    ids = json.loads(SUBSET.read_text())["ids"]
    data = R.RS1.Data()
    (OUT / m).mkdir(parents=True, exist_ok=True)
    R.RS1.gpu_process(f"x13:{m}", lambda be, dyn, tok, wb, prior, t0: run_model(be, tok, data, m, ids, prior, t0, a), a.cap_hours)


if __name__ == "__main__":
    main()
