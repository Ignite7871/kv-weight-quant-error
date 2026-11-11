"""Second-model feasibility pilot (PILOT2_SPEC.md). 100 fixed GSM8K TRAIN problems; arms FP16, NF4, cache-only (the verified engine); pre-declared gates.
Uses the Stage 1 ledger/watchdog/telemetry machinery unchanged. Cumulative GPU ceiling for the whole pilot (all candidates, loads and checks): 2.4 h, absolute.

  plan           write pilot2_sample.json (seeded; no outcome data)                                     (CPU)
  check-format   tokenizer / prompt / stop-rule validation for a candidate                              (CPU)
  cpu-smoke      2 problems x 8 tokens x 3 arms on CPU to validate the code path                        (CPU)
  lock           write PILOT2_LOCK.txt (SHA-256 of spec, driver, sample)                                (CPU)
  run            the pilot for one candidate                                                            (GPU; needs --authorized-gpu-run and a verified lock)
  ledger         cumulative pilot GPU time
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_stage1 as RS1                       # noqa: E402

import torch                                   # noqa: E402

from rv2.boundary import find_boundary, would_stop_at_boundary   # noqa: E402
from rv2.engine import generate                # noqa: E402
from rv2.policies import StaticFP16, StaticNF4  # noqa: E402
from rv2.stage1 import Cell, generate_cell     # noqa: E402

OUT = HERE / "pilot2_runs"
SAMPLE_FILE = HERE / "pilot2_sample.json"
LOCK_FILE = HERE / "PILOT2_LOCK.txt"
CAP_HOURS = 2.4
N_PROBLEMS = 100
N_EQUIV = 8
EQUIV_TOKENS = 64
SEED = "rv2-pilot2-train-2026-10-27"
CANDIDATES = {"llama32_1b_instruct": "meta-llama/Llama-3.2-1B-Instruct", "smollm2_1_7b": "HuggingFaceTB/SmolLM2-1.7B"}
ARMS = ["fp16", "nf4", "cache_only"]
GATES = {"P3_min_fp16_acc": 0.20, "P3_min_fp16_strict_acc": 0.15, "P4_flip_lo": 0.01, "P4_flip_hi": 0.10, "P4_min_depart_share": 0.50,
         "P6_max_ms_per_step": 132.0, "P7_min_parseable": 0.90, "P1_max_mem_gb": 7.0, "futility_min_first30": 2}
STRICT_RE = re.compile(r"####\s*(-?\d[\d,]*(?:\.\d+)?)")
DRIFT_RE = re.compile(r"\n\s*Question\s*:", re.IGNORECASE)


def sha256_file(p: Path):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def make_sample():
    if SAMPLE_FILE.exists():
        return json.loads(SAMPLE_FILE.read_text())
    s1 = json.loads((HERE / "stage1_sample.json").read_text())
    pilot = json.loads((HERE / "pilot_sample.json").read_text())
    e1 = json.loads((HERE / "e1_sample.json").read_text())
    used = {0, 1, 2} | set(s1.get("equiv_train", [])) | set(s1.get("feasibility_train", [])) | set(s1.get("speed_train", [])) \
        | set(pilot.get("train_equiv", [])) | set(pilot.get("train_calib", [])) | set(e1["calibration_train_ids"]) | set(e1["excluded_train_ids"])
    pool = [i for i in range(7473) if i not in used]
    ids = random.Random(SEED).sample(pool, N_PROBLEMS)
    s = {"seed": SEED, "train_ids": ids, "equiv_ids": ids[:N_EQUIV], "candidates": CANDIDATES, "gates": GATES,
         "note": "GSM8K train split; disjoint from the three few-shot examples, every earlier pilot set and the E1 calibration set. No outcome data was used."}
    SAMPLE_FILE.write_text(json.dumps(s, indent=1))
    return s


def strict_parse(text):
    t = DRIFT_RE.split(text)[0]
    return STRICT_RE.search(t) is not None and "####" in t


def strict_solved(rec):
    t = DRIFT_RE.split(rec["text"])[0]
    m = STRICT_RE.search(t)
    if not m:
        return False
    try:
        return abs(float(m.group(1).replace(",", "")) - rec["ground_truth"]) <= 1e-2
    except ValueError:
        return False


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


def first_div(a, b):
    n = min(len(a), len(b))
    for k in range(n):
        if a[k] != b[k]:
            return k
    return None if len(a) == len(b) else n


# ----------------------------------------------------------------------------- format validation (CPU)
def check_format(model_name: str, data=None, n=5) -> dict:
    from transformers import AutoConfig, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    cfg = AutoConfig.from_pretrained(model_name)
    data = data or RS1.Data()
    s = make_sample()
    rep = {"model": model_name, "eos_token": tok.eos_token, "eos_token_id": tok.eos_token_id, "bos_token": tok.bos_token,
           "config_eos_token_id": cfg.eos_token_id, "pad_token": tok.pad_token, "n_layers": cfg.num_hidden_layers, "n_kv_heads": cfg.num_key_value_heads,
           "problems": []}
    ok = True
    for idx in s["train_ids"][:n]:
        prob, prompt = data.problem("train", idx)
        enc = tok(prompt, return_tensors="pt")
        ids = enc.input_ids
        dec_plain = tok.decode(ids[0], skip_special_tokens=True)
        bos_first = bool(tok.bos_token_id is not None and int(ids[0, 0]) == tok.bos_token_id)
        rt = dec_plain == prompt or dec_plain.lstrip() == prompt.lstrip()
        ends = prompt.rstrip().endswith("Answer:")
        rep["problems"].append({"idx": idx, "prompt_tokens": int(ids.shape[1]), "bos_first": bos_first, "round_trip": rt, "prompt_ends_with_Answer": ends})
        ok &= rt and ends
    # stop rule: a completion that continues with the next few-shot question must be detected as a drift marker
    comp = tok(" 18 dollars.\n#### 18\n\nQuestion: A farmer has 3 cows.\nAnswer:", add_special_tokens=False).input_ids
    b = find_boundary(tok, comp, tok.eos_token_id, 256)
    rep["drift_marker_detected"] = b.boundary_type == "drift_marker" and would_stop_at_boundary(tok, comp)
    rep["offsets_consistent"] = bool(b.offsets_consistent)
    ok &= rep["drift_marker_detected"] and rep["offsets_consistent"]
    rep["ok"] = bool(ok)
    return rep


# ----------------------------------------------------------------------------- GPU pilot
def pilot(be, tok, wb, prior, t0, cand_key, data, s, out):
    import stage1_runtime as R
    rep = {"candidate": cand_key, "model": CANDIDATES[cand_key], "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "peak_mem_after_load_gb": torch.cuda.max_memory_allocated() / 1024 ** 3, "gates": {}}
    out.mkdir(parents=True, exist_ok=True)
    eos = tok.eos_token_id
    stop = lambda ids: would_stop_at_boundary(tok, ids)       # noqa: E731
    # P5: the engine's two-pass cells reproduce the static decoders (tokens and cache bitwise) on 8 problems x 64 tokens
    eq = []
    for idx in s["equiv_ids"]:
        prob, prompt = data.problem("train", idx)
        ids_t = tok(prompt, return_tensors="pt").input_ids.to(be.device)
        cap = {}
        orig = be.run

        def run(i, c, use_4bit, orig=orig, cap=cap):
            o = orig(i, c, use_4bit)
            cap["c"] = o[1]
            return o
        be.run = run
        try:
            res = {}
            for name, pol, cell in (("FF", StaticFP16(), Cell("F", "F", "ff")), ("NN", StaticNF4(), Cell("N", "N", "nn"))):
                a = generate(be, ids_t, pol, EQUIV_TOKENS, eos, 0, instrument=False)
                ca = [(l.keys.clone(), l.values.clone()) for l in cap["c"].layers]
                b = generate_cell(be, ids_t, cell, EQUIV_TOKENS, eos)
                cb = [(l.keys.clone(), l.values.clone()) for l in cap["c"].layers]
                res[name] = a.ids == b.ids and all(torch.equal(x, y) and torch.equal(u, v) for (x, u), (y, v) in zip(ca, cb))
            eq.append(res)
        finally:
            be.run = orig
    rep["gates"]["P5_engine_equivalence"] = {"pass": all(all(r.values()) for r in eq), "detail": eq}
    if not rep["gates"]["P5_engine_equivalence"]["pass"]:
        rep["stopped"] = "P5 correctness check failed"
        (out / "pilot_report.json").write_text(json.dumps(rep, indent=1, default=float))
        return False
    # main runs: problem-major, rotated arm order; 100 train problems; futility rule after 30 problems (FP16 solved fewer than 2)
    recs = {a: {} for a in ARMS}
    for k, idx in enumerate(s["train_ids"]):
        used = prior + (time.time() - t0)
        if used + 120 > CAP_HOURS * 3600:
            rep["stopped"] = f"absolute ceiling before problem {k}"
            break
        why = R.hardware_stop(R.telemetry_now())
        if why:
            rep["stopped"] = f"hardware safety: {why}"
            break
        prob, prompt = data.problem("train", idx)
        ids_t = tok(prompt, return_tensors="pt").input_ids.to(be.device)
        for arm in ARMS[k % 3:] + ARMS[:k % 3]:
            recs[arm][idx] = RS1.run_one(arm, be, tok, ids_t, prob, idx, "train", wb, True)
            with open(out / f"{arm}.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(recs[arm][idx]) + "\n")
        if k + 1 == 30:
            solved = sum(r["status"] == "SOLVED" for r in recs["fp16"].values())
            if solved < GATES["futility_min_first30"]:
                rep["stopped"] = f"futility: FP16 solved {solved} of the first 30 problems (pre-declared stop below {GATES['futility_min_first30']})"
                break
        if (k + 1) % 10 == 0:
            print(f"[pilot {cand_key}] {k + 1}/100 cum GPU {(prior + time.time() - t0) / 3600:.3f} h", flush=True)
    n = len(recs["fp16"])
    rep["n_problems_run"] = n
    if n == 0:
        (out / "pilot_report.json").write_text(json.dumps(rep, indent=1, default=float))
        return False
    idxs = sorted(recs["fp16"])
    acc = {a: sum(recs[a][i]["status"] == "SOLVED" for i in idxs if i in recs[a]) for a in ARMS}
    nn = {a: sum(1 for i in idxs if i in recs[a]) for a in ARMS}
    rep["accuracy_lenient"] = {a: {"solved": acc[a], "of": nn[a], "pct": round(100 * acc[a] / max(nn[a], 1), 1), "wilson95": wilson(acc[a], nn[a]) if nn[a] else None} for a in ARMS}
    strict = {a: sum(strict_solved(recs[a][i]) for i in idxs if i in recs[a]) for a in ARMS}
    rep["accuracy_strict_format"] = {a: {"solved": strict[a], "pct": round(100 * strict[a] / max(nn[a], 1), 1)} for a in ARMS}
    parse = sum(strict_parse(recs["fp16"][i]["text"]) for i in idxs) / n
    g = rep["gates"]
    g["P1_memory"] = {"pass": max((r["peak_mem_allocated"] or 0) for a in ARMS for r in recs[a].values()) / 1024 ** 3 < GATES["P1_max_mem_gb"],
                      "peak_gb": max((r["peak_mem_allocated"] or 0) for a in ARMS for r in recs[a].values()) / 1024 ** 3}
    g["P2_fp16_numerics"] = {"pass": all(len(set(recs["fp16"][i]["ids"])) > 3 and len(recs["fp16"][i]["text"].strip()) > 0 for i in idxs)}
    g["P3_baseline_quality"] = {"pass": acc["fp16"] / n >= GATES["P3_min_fp16_acc"] and strict["fp16"] / n >= GATES["P3_min_fp16_strict_acc"],
                                "lenient": acc["fp16"] / n, "strict": strict["fp16"] / n, "n": n}
    co = [recs["cache_only"][i] for i in idxs if i in recs["cache_only"]]
    flips = sum(sum(a != b for a, b in zip(r["tF"], r["tN"])) for r in co)
    steps = sum(len(r["tF"]) for r in co)
    depart = sum(1 for i in idxs if i in recs["cache_only"] and ((d := first_div(recs["cache_only"][i]["ids"], recs["fp16"][i]["ids"])) is not None and d < 64)) / max(len(co), 1)
    g["P4_non_degenerate"] = {"pass": GATES["P4_flip_lo"] <= flips / max(steps, 1) <= GATES["P4_flip_hi"] and depart >= GATES["P4_min_depart_share"],
                              "branch_argmax_disagreement_rate": flips / max(steps, 1), "share_departing_from_fp16_within_64_tokens": depart}
    ms = 1000 * sum(r["gen_seconds"] for r in co) / max(sum(r["steps"] for r in co), 1)
    g["P6_speed"] = {"pass": ms <= GATES["P6_max_ms_per_step"], "cache_only_ms_per_step": ms,
                     "ms_per_step_by_arm": {a: 1000 * sum(r["gen_seconds"] for r in recs[a].values()) / max(sum(r["steps"] for r in recs[a].values()), 1) for a in ARMS},
                     "mean_s_per_problem_by_arm": {a: sum(r["wall_seconds"] for r in recs[a].values()) / max(len(recs[a]), 1) for a in ARMS}}
    g["P7_format"] = {"pass": parse >= GATES["P7_min_parseable"], "fp16_strict_parseable_share": parse}
    rep["all_gates_passed"] = bool(all(v["pass"] for v in g.values()) and "stopped" not in rep)
    rep["gpu_hours_this_process_so_far"] = (time.time() - t0) / 3600
    (out / "pilot_report.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({k: v for k, v in rep.items() if k != "gates"}, indent=1, default=float))
    print(json.dumps({k: v["pass"] for k, v in g.items()}))
    return bool(rep["all_gates_passed"])


# ----------------------------------------------------------------------------- CPU smoke
def cpu_smoke(model_name):
    from rv2.backends import build_cpu_backend, weight_bytes_per_pass
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    be = build_cpu_backend(AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16))
    data = RS1.Data()
    s = make_sample()
    RS1.MAX_NEW_TOKENS = 8
    wb = weight_bytes_per_pass(be.model)
    for idx in s["train_ids"][:2]:
        prob, prompt = data.problem("train", idx)
        ids_t = tok(prompt, return_tensors="pt").input_ids
        for arm in ARMS:
            r = RS1.run_one(arm, be, tok, ids_t, prob, idx, "train", wb, True)
            print(arm, r["status"], r["steps"], repr(r["text"][:40]))


def lock():
    files = [HERE / "PILOT2_SPEC.md", HERE / "run_pilot2.py", SAMPLE_FILE, HERE / "run_stage1.py", HERE / "stage1_runtime.py"] + sorted((HERE / "rv2").glob("*.py"))
    head = subprocess.run(["git", "-C", str(HERE.parent), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    rels = [f.relative_to(HERE.parent).as_posix() for f in files]
    dirty = subprocess.run(["git", "-C", str(HERE.parent), "status", "--porcelain", "--", *rels], capture_output=True, text=True).stdout.strip()
    if dirty:
        sys.exit(f"Refusing lock: uncommitted changes:\n{dirty}")
    lines = ["pilot lock -- research_v2 second-model feasibility pilot (PILOT2_SPEC.md)", f"utc: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
             f"git commit containing the locked files: {head}", "Private dated hash, NOT a public pre-registration. Created before any pilot GPU use.", ""]
    for r in rels:
        h = hashlib.sha256(subprocess.run(["git", "-C", str(HERE.parent), "show", f"{head}:{r}"], capture_output=True).stdout).hexdigest()
        lines.append(f"{h}  {r}")
    LOCK_FILE.write_text("\n".join(lines) + "\n")
    print(LOCK_FILE.read_text())


def lock_ok():
    if not LOCK_FILE.exists():
        return False
    for l in LOCK_FILE.read_text().splitlines():
        p = l.split("  ", 1)
        if len(p) == 2 and len(p[0]) == 64:
            head = next(x.split(": ")[1] for x in LOCK_FILE.read_text().splitlines() if x.startswith("git commit"))
            h = hashlib.sha256(subprocess.run(["git", "-C", str(HERE.parent), "show", f"{head}:{p[1]}"], capture_output=True).stdout).hexdigest()
            if h != p[0] or subprocess.run(["git", "-C", str(HERE.parent), "status", "--porcelain", "--", p[1]], capture_output=True, text=True).stdout.strip():
                return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "check-format", "cpu-smoke", "lock", "run", "ledger"])
    ap.add_argument("--candidate", choices=list(CANDIDATES), default="llama32_1b_instruct")
    ap.add_argument("--authorized-gpu-run", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    RS1.OUT, RS1.LEDGER, RS1.PROGRESS = OUT, OUT / "gpu_ledger.jsonl", OUT / "current_process.json"
    if a.cmd == "plan":
        s = make_sample()
        print({k: (len(v) if isinstance(v, list) else v) for k, v in s.items() if k not in ("candidates", "gates")})
        return
    if a.cmd == "ledger":
        print(f"pilot cumulative GPU wall time: {RS1.ledger_seconds() / 3600:.3f} h of {CAP_HOURS} h")
        return
    if a.cmd == "lock":
        lock()
        return
    model = CANDIDATES[a.candidate]
    if a.cmd == "check-format":
        rep = check_format(model)
        print(json.dumps(rep, indent=1))
        sys.exit(0 if rep["ok"] else 1)
    if a.cmd == "cpu-smoke":
        cpu_smoke(model)
        return
    if not a.authorized_gpu_run:
        sys.exit("Refusing to run: --authorized-gpu-run is required.")
    if not lock_ok():
        sys.exit("Refusing to run: the pilot lock does not verify.")
    fmt = check_format(model)
    if not fmt["ok"]:
        sys.exit("Refusing to run: prompt/tokenizer/stop-rule validation failed: " + json.dumps(fmt)[:400])
    RS1.MODEL = model
    s = make_sample()
    data = RS1.Data()
    out = OUT / a.candidate
    RS1.gpu_process("pilot_" + a.candidate, lambda be, dyn, tok, wb, prior, t0: pilot(be, tok, wb, prior, t0, a.candidate, data, s, out), CAP_HOURS)


if __name__ == "__main__":
    main()
