"""X9 driver (v3/phase7/FOLLOWUP_X9.md). EXPLORATORY follow-up outside the locked v3 protocol; the locked records are read only.

Subcommands (CPU unless marked)
  subset                         write v3/phase7/x9_subset.json (50 of the 200 Tier A ids, seed 20261209; refuses to overwrite)
  manifest                       write v3/phase7/X9_MANIFEST.txt and print the X9 manifest id
  status | ledger                progress / X9 GPU time (own ledger, 1.5 h ceiling)
  run --model M                  [GPU] NAT, NAT-K*, NAT-K*-L0nat on the subset, with the bitwise reproduction gate against the locked records

GPU use is REFUSED unless ALL of: --authorized-gpu-run; research_v2/v3_runs_x9/AUTHORIZED_X9_BY_AUTHORS.txt exists and contains the text
"FOLLOWUP_X9" and the X9 manifest id; the listed files still hash to the id in X9_MANIFEST.txt; the v3, E1 and E1L locks and the model's weight
hash verify (run_v3.locks_ok); the model is the next in the fixed order, every earlier model completed, and no reproduction mismatch is on file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

P7 = Path(__file__).resolve().parent
RV = P7.parents[1]
sys.path.insert(0, str(RV))
sys.path.insert(0, str(P7))

import run_v3 as R                              # noqa: E402  (locked v3 driver, imported unchanged; HF before torch)

import numpy as np                              # noqa: E402

import rv3.core as V                            # noqa: E402
import rv3x9 as X                               # noqa: E402

OUT = RV / "v3_runs_x9"
LOCKED_RUNS = RV / "v3_runs"
AUTH = OUT / "AUTHORIZED_X9_BY_AUTHORS.txt"
DESIGN = OUT / "design_x9.json"
SUBSET = P7 / "x9_subset.json"
MANIFEST = P7 / "X9_MANIFEST.txt"
MANIFEST_FILES = ["FOLLOWUP_X9.md", "rv3x9.py", "run_x9.py", "analyze_x9.py", "test_x9.py", "mutation_check_x9.py", "x9_subset.json"]
AUTH_TEXT = "FOLLOWUP_X9"
CAP_HOURS = 1.5
PROJ_LIMIT_H = 1.4
N_SUBSET, N_REDUCED, SUBSET_SEED = 50, 40, 20261209
N_PROJECT = 3
ORDER = ["qwen-base", "qwen-instruct", "llama-base", "llama-instruct"]
ARMS = [("NAT", 1.0, "NAT"), ("NAT-K*", 1.0, "NAT-Kstar"), (X.X9_ARM, 1.0, "NAT-Kstar-L0nat")]
GATED = ("NAT", "NAT-Kstar")
IGNORED = {"wall_seconds"}
X9_FIELDS = set(X.LAYER_KEYS)
REDUCTIONS = ["drop L0nat in Llama", "n = 40 in Llama", "n = 40 in every model"]


def family(m):
    return m.split("-")[0]


# ----------------------------------------------------------------------------- subset and manifest
def lf_bytes(p: Path) -> bytes:
    return p.read_bytes().replace(b"\r\n", b"\n")


def subset_ids(tier_a_ids):
    return sorted(int(x) for x in np.random.default_rng(SUBSET_SEED).choice(sorted(tier_a_ids), N_SUBSET, replace=False))


def make_subset():
    if SUBSET.exists():
        sys.exit(f"{SUBSET.name} exists already (fixed); refusing to overwrite")
    s = json.loads((RV / "v3_sample.json").read_text())
    ids = subset_ids(s["tierA_ids"])
    out = {"ids": ids, "n": len(ids), "seed": SUBSET_SEED,
           "draw": "numpy.random.default_rng(seed).choice(sorted(tierA_ids), 50, replace=False), sorted",
           "source": "research_v2/v3_sample.json", "source_sha256_lf": hashlib.sha256(lf_bytes(RV / "v3_sample.json")).hexdigest(),
           "exposed_in_subset": sorted(set(ids) & set(s["exposed_test_ids_in_sample"])), "no outcome data used": True}
    SUBSET.write_text(json.dumps(out, indent=1) + "\n")
    return out


def manifest_id(root: Path = P7, files=None) -> str:
    h = hashlib.sha256()
    for name in files or MANIFEST_FILES:
        b = lf_bytes(root / name)
        h.update(f"{name}\n{len(b)}\n".encode())
        h.update(b)
    return h.hexdigest()[:16]


def write_manifest():
    mid = manifest_id()
    lines = [f"X9 manifest id: {mid}", f"utc: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
             "files (in hashing order; LF-normalized; each prefixed by name and byte length):"]
    lines += [f"  {n}  sha256(LF) {hashlib.sha256(lf_bytes(P7 / n)).hexdigest()}" for n in MANIFEST_FILES]
    lines += ["", f"Authorization (created only by the authors): {AUTH.relative_to(RV.parent)} containing the text {AUTH_TEXT} and the id {mid}."]
    MANIFEST.write_text("\n".join(lines) + "\n")
    return mid


def recorded_manifest_id():
    if not MANIFEST.exists():
        return None
    first = MANIFEST.read_text(encoding="utf-8").splitlines()[0]
    return first.split(":", 1)[1].strip() if first.startswith("X9 manifest id:") else None


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
    lk, nk = set(locked) - IGNORED, set(new) - IGNORED - X9_FIELDS
    for k in sorted(lk ^ nk):
        out[k] = {"equal": False, "missing_in": "new" if k in lk else "locked"}
    for k in sorted(lk & nk):
        if not same(new[k], locked[k]):
            out[k] = _diff_detail(new[k], locked[k])
    return out


def history_matches(regen, locked_ids) -> bool:
    return [int(x) for x in regen] == [int(x) for x in locked_ids]


# ----------------------------------------------------------------------------- design (pre-declared reductions, runtime only)
def n_for(m, design):
    return design["n"][m]


def arms_for(m, design):
    return [a for a in ARMS if not (a[0] == X.X9_ARM and family(m) == "llama" and not design["llama_l0nat"])]


def full_design():
    return {"llama_l0nat": True, "n": {m: N_SUBSET for m in ORDER}, "applied": []}


def apply_reduction(design, k):
    d = json.loads(json.dumps(design))
    if k == 0:
        d["llama_l0nat"] = False
    elif k == 1:
        d["n"].update({m: N_REDUCED for m in ORDER if family(m) == "llama"})
    else:
        d["n"] = {m: N_REDUCED for m in ORDER}
    d["applied"].append(REDUCTIONS[k])
    return d


def projection_h(design, used_s, t_arm, t_over, t_load, done_first):
    """Projected X9 total (h): used so far + the remaining problems of every model (Llama scaled by the locked LLAMA_TIME_RATIO) + a model load
    for each later model; x 1.05."""
    tot = used_s
    for k, m in enumerate(ORDER):
        ratio = 1.0 if family(m) == "qwen" else R.LLAMA_TIME_RATIO
        remaining = max(0, n_for(m, design) - (done_first if k == 0 else 0))
        tot += remaining * (len(arms_for(m, design)) * t_arm + t_over) * ratio + (t_load if k > 0 else 0.0)
    return tot * 1.05 / 3600


def choose_design(used_s, t_arm, t_over, t_load, done_first):
    design = full_design()
    steps = [{"design": "full", "projected_h": projection_h(design, used_s, t_arm, t_over, t_load, done_first)}]
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
    return (OUT / m / "x9_complete.json").exists()


def require_gpu_permission(a, m):
    if not a.authorized_gpu_run:
        sys.exit("Refusing to run: X9 GPU runs need explicit author authorization (--authorized-gpu-run).")
    mid, rec = manifest_id(), recorded_manifest_id()
    if rec is None or mid != rec:
        sys.exit(f"Refusing to run: the X9 files hash to {mid}, but X9_MANIFEST.txt records {rec}.")
    txt = AUTH.read_text(encoding="utf-8", errors="replace") if AUTH.exists() else ""
    if not auth_ok(txt, mid):
        sys.exit(f"Refusing to run: {AUTH.relative_to(RV.parent)} must exist and contain '{AUTH_TEXT}' and the X9 manifest id {mid}. "
                 "Only the authors create it.")
    if not R.locks_ok(m):
        sys.exit("Refusing to run: the v3 lock (or the E1 / E1L lock, or this model's weight-file hash) does not verify.")
    if mismatch_files():
        sys.exit(f"Refusing to run: a reproduction mismatch is on file ({mismatch_files()[0]}); the authors decide what follows.")
    for prev in ORDER[:ORDER.index(m)]:
        if not complete(prev):
            sys.exit(f"Refusing to run: fixed order; {prev} has not completed X9.")
    if ORDER.index(m) > 0 and not (DESIGN.exists() and json.loads(DESIGN.read_text()).get("pass_")):
        sys.exit("Refusing to run: the X9 design (runtime projection of the first model) is missing or did not fit the ceiling.")


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


def run_one(be, tok, data, m, idx, arms, ctx, cap, locked):
    """One problem: regenerate the FP16 history (compared with the locked one), run the arms on the locked history, gate NAT and NAT-K*
    bitwise. Returns (records, slot arrays, report)."""
    ids_t = R.prompt_ids(tok, data, "test", idx, be.device)
    t0 = time.perf_counter()
    regen = R.fp16_history(be, tok, ids_t)
    t_hist = time.perf_counter() - t0
    hist = locked["FP16_history"][idx]["ids"]
    X.clear()
    recs, _ = R.run_problem(be, tok, data, m, idx, "test", arms, "V3A", ctx, cap, hist=hist)
    if len(X.INSTANCES) != len(arms):
        raise RuntimeError(f"expected {len(arms)} predictor states, got {len(X.INSTANCES)}")
    slots = {}
    for (_, _, tag), inst in zip(arms, X.INSTANCES):
        recs[tag].update(X.layer_fields(inst))
        slots[tag] = X.slot_arrays(inst)
    rep = {"index": idx, "history_matches_locked": history_matches(regen, hist), "t_history_s": t_hist,
           "integrity_errors": R.integrity_errors(recs), "diffs": {}}
    for tag in GATED:
        if tag in recs:
            rep["diffs"][tag] = compare_records(recs[tag], locked[tag][idx])
    rep["ok"] = bool(rep["history_matches_locked"] and not rep["integrity_errors"] and not any(rep["diffs"].values()))
    return recs, slots, rep


def mismatch_report(m, rep, recs, locked):
    idx = rep["index"]
    return {"model": m, "problem": idx, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "history_matches_locked": rep["history_matches_locked"], "integrity_errors": rep["integrity_errors"],
            "differing_fields": rep["diffs"],
            "tokens": {tag: {"top_matches": same(json.loads(json.dumps(recs[tag]["top"])), locked[tag][idx]["top"]),
                             "flip_matches": same(json.loads(json.dumps(recs[tag]["flip"])), locked[tag][idx]["flip"])} for tag in GATED if tag in recs},
            "policy": "FOLLOWUP_X9 section 6 / D4: stop and report; no tolerance; the authors decide"}


def save(m, idx, recs, slots, rep):
    d = OUT / m
    (d / "slots").mkdir(parents=True, exist_ok=True)
    for tag, arr in slots.items():
        np.savez_compressed(d / "slots" / f"{tag}_{idx}.npz", **arr)
    R.E.append_records(d, {"reproduction": {k: v for k, v in rep.items() if k != "diffs"} | {"bitwise_equal": {t: not v for t, v in rep["diffs"].items()}}})
    R.E.append_records(d, recs)                                    # records last: a problem counts as done only once everything is written


def done(m, idx, arms):
    return all(idx in R.done_ids(OUT / m, tag) for _, _, tag in arms)


def run_model(be, tok, data, m, ids, prior, t0, a):
    cal =json.loads((LOCKED_RUNS / m / "calibration.json").read_text())
    ctx = V.ctx_from_calibration(cal, be.device)
    X.install(ctx["U"])
    cap = V.QueryCapture(be.model)
    locked = {t: load_locked(m, t) for t in ("FP16_history",) + GATED}
    first = ORDER.index(m) == 0
    design = json.loads(DESIGN.read_text()) if DESIGN.exists() else None
    state = {"n": 0, "t_first": None, "per_problem": []}

    def step(idx):
        nonlocal design
        if design is not None and ids.index(idx) >= n_for(m, design):
            return True
        arms = arms_for(m, design) if design is not None else ARMS
        tp = time.perf_counter()
        if state["t_first"] is None:
            state["t_first"] = time.time() - t0
        recs, slots, rep = run_one(be, tok, data, m, idx, arms, ctx, cap, locked)
        if not rep["ok"]:
            (OUT / m).mkdir(parents=True, exist_ok=True)
            (OUT / m / "REPRODUCTION_MISMATCH.json").write_text(json.dumps(mismatch_report(m, rep, recs, locked), indent=1, default=float))
            print(f"[REPRODUCTION MISMATCH] {m} problem {idx}: stopping (FOLLOWUP_X9 section 6)", flush=True)
            return False
        save(m, idx, recs, slots, rep)
        state["n"] += 1
        state["per_problem"].append((time.perf_counter() - tp, [recs[t]["wall_seconds"] for _, _, t in arms]))
        if first and design is None and state["n"] == N_PROJECT:
            t_arm = float(np.mean([w for _, ws in state["per_problem"] for w in ws]))
            t_over = float(np.mean([tot - sum(ws) for tot, ws in state["per_problem"]]))
            design = choose_design(prior + time.time() - t0, t_arm, t_over, state["t_first"], N_PROJECT)
            design["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            DESIGN.write_text(json.dumps(design, indent=1, default=float))
            print("[X9 design]", json.dumps(design["projection_steps"]), flush=True)
            if not design["pass_"]:
                print("[X9 design] projection exceeds 1.4 h after every pre-declared reduction: stop and ask the authors", flush=True)
                return False
        return True

    try:
        ok = R.E.block_loop(f"x9:{m}", be, tok, None, prior, t0, a.cap_hours, ids, step,
                            lambda i: done(m, i, arms_for(m, design) if design is not None else ARMS)
                            or (design is not None and ids.index(i) >= n_for(m, design)))
    finally:
        cap.remove()
        X.uninstall()
    design = json.loads(DESIGN.read_text()) if DESIGN.exists() else None
    if ok and design is not None and all(done(m, i, arms_for(m, design)) for i in ids[:n_for(m, design)]):
        (OUT / m / "x9_complete.json").write_text(json.dumps({"model": m, "n": n_for(m, design), "arms": [t for _, _, t in arms_for(m, design)],
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
        print("X9 manifest id:", write_manifest())
        return
    configure(a.model)
    if a.cmd == "ledger":
        print(f"X9 GPU wall time: {R.RS1.ledger_seconds() / 3600:.3f} h of {a.cap_hours} h")
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
    R.RS1.gpu_process(f"x9:{m}", lambda be, dyn, tok, wb, prior, t0: run_model(be, tok, data, m, ids, prior, t0, a), a.cap_hours)


if __name__ == "__main__":
    main()
