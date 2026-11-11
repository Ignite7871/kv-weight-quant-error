"""Controlled performance benchmark (Task A, 2026-10-23). Changes no predictions or routing logic.

Usage (GPU; needs explicit authorization flag):
  venv/Scripts/python.exe research_v2/perf/bench.py --label balanced_A --authorized-gpu-run

Workload (identical for every configuration):
  1. pass-level micro-benchmark: fixed prompt (eval problem 1), FP16 prefill, then repeated single-token
     decode passes at a constant cache length (cache cropped back each time), FP16 and NF4, in
     (a) synchronous mode: cuda.synchronize after every pass  (latency as seen by a step loop)
     (b) asynchronous mode: 40 passes launched back-to-back, one synchronize (launch-overlap throughput)
     20 warm-up passes, then 5 blocks x 40 passes per precision and mode.
  2. generation benchmark: 3 fixed eval problems x {FP16, NF4} x 3 engines x 3 repetitions, 128 new
     tokens max, greedy; engines = rv2 instrumented, rv2 lean (instrument=False), untouched historical
     DynamicQwen.generate. Round-robin order with rotation; one recorded warm-up generation per precision.
     Token identity across engines is checked and reported.
GPU telemetry: nvidia-smi sampled every 500 ms, tagged with the current phase.
GPU-time cap: perf_ledger.jsonl accumulates process wall time; the script refuses to start if the
cumulative total is >= --cap-min, and aborts remaining phases if the projection would exceed it.
"""

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from datasets import load_dataset  # HF before torch
from transformers import AutoTokenizer

import torch

import rv2  # noqa: F401
from rv2.backends import load_gpu_backend
from rv2.engine import generate
from rv2.policies import StaticFP16, StaticNF4
from probe_gsm8k_routing import build_fewshot_prefix

LEDGER = HERE / "perf_ledger.jsonl"
MODEL = "Qwen/Qwen2.5-0.5B"
QUERY = "timestamp,clocks.sm,clocks.mem,utilization.gpu,temperature.gpu,power.draw,pstate,clocks_event_reasons.active"
PHASE = {"name": "load"}


def ledger_minutes():
    return sum(json.loads(l)["seconds"] for l in LEDGER.read_text().splitlines() if l.strip()) / 60 if LEDGER.exists() else 0.0


SAMPLER = {}


def sampler(stop, rows):
    # The process handle is kept in SAMPLER so main() can terminate it explicitly: this thread blocks in
    # readline(), so it cannot be relied on to stop the child (an orphaned nvidia-smi kept the caller's
    # output pipe open in the first two runs).
    p = subprocess.Popen(["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits", "-lms", "500"],
                         stdout=subprocess.PIPE, text=True)
    SAMPLER["p"] = p
    while not stop.is_set():
        line = p.stdout.readline()
        if not line:
            break
        rows.append([time.time(), PHASE["name"]] + [x.strip() for x in line.split(",")])


def power_scheme():
    out = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True).stdout.strip()
    return out


@torch.no_grad()
def microbench(backend, prompt_ids, blocks=5, per_block=40, warm=20):
    model, rs = backend.model, backend.routing_state
    out = {}
    logits = model(prompt_ids, use_cache=True)
    cache = logits.past_key_values
    L0 = cache.get_seq_length()
    tok = logits.logits[:, -1:, :].argmax(-1)
    for prec, use4 in (("fp16", False), ("nf4", True)):
        rs.use_4bit = use4
        PHASE["name"] = f"micro_{prec}_warm"
        for _ in range(warm):
            cache.crop(L0); model(tok, past_key_values=cache, use_cache=True)
        torch.cuda.synchronize()
        # synchronous
        PHASE["name"] = f"micro_{prec}_sync"
        sync_blocks, ev_blocks = [], []
        for _ in range(blocks):
            ws, es = [], []
            for _ in range(per_block):
                cache.crop(L0)
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                t0 = time.perf_counter(); e0.record()
                model(tok, past_key_values=cache, use_cache=True)
                e1.record(); torch.cuda.synchronize()
                ws.append(1e3 * (time.perf_counter() - t0)); es.append(e0.elapsed_time(e1))
            sync_blocks.append(sum(ws) / len(ws)); ev_blocks.append(sum(es) / len(es))
        # asynchronous (launch overlap)
        PHASE["name"] = f"micro_{prec}_async"
        async_blocks = []
        for _ in range(blocks):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            for _ in range(per_block):
                cache.crop(L0); model(tok, past_key_values=cache, use_cache=True)
            torch.cuda.synchronize()
            async_blocks.append(1e3 * (time.perf_counter() - t0) / per_block)
        out[prec] = {"sync_wall_ms_blocks": sync_blocks, "sync_event_ms_blocks": ev_blocks,
                     "async_wall_ms_blocks": async_blocks, "cache_len": L0}
    rs.use_4bit = False
    return out


def run_generation(backend, dyn, tok, prompts, reps, max_new):
    recs = []
    engines = ["instrumented", "lean", "historical"]
    pols = {"fp16": (StaticFP16(), -1.0), "nf4": (StaticNF4(), 100.0)}

    def one(engine, prec, pi, rep, label):
        prompt, ids_t, idx = prompts[pi]
        pol, tau = pols[prec]
        PHASE["name"] = f"gen_{engine}_{prec}"
        torch.cuda.synchronize(); t0 = time.perf_counter()
        if engine == "historical":
            dyn.entropy_threshold = tau
            dyn.generate(prompt, tok, max_new)
            ids = dyn.last_generated_ids
        else:
            ids = generate(backend, ids_t, pol, max_new, tok.eos_token_id, idx, instrument=(engine == "instrumented")).ids
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        recs.append({"engine": engine, "precision": prec, "problem": idx, "rep": rep, "label": label,
                     "steps": len(ids) - 1, "wall_s": wall, "ms_per_step": 1e3 * wall / max(1, len(ids) - 1), "ids": ids})

    for prec in ("fp16", "nf4"):                      # recorded warm-up
        one("instrumented", prec, 0, -1, "warmup")
    k = 0
    for rep in range(reps):
        for pi in range(len(prompts)):
            for prec in ("fp16", "nf4"):
                order = engines[k % 3:] + engines[:k % 3]; k += 1
                for e in order:
                    one(e, prec, pi, rep, "timed")
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--cap-min", type=float, default=45.0)
    ap.add_argument("--authorized-gpu-run", action="store_true")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-new", type=int, default=128)
    a = ap.parse_args()
    if not a.authorized_gpu_run:
        sys.exit("needs --authorized-gpu-run")
    prior = ledger_minutes()
    if prior >= a.cap_min:
        sys.exit(f"perf GPU cap reached ({prior:.1f} min)")
    t_start = time.time()
    stop, rows = threading.Event(), []
    th = threading.Thread(target=sampler, args=(stop, rows), daemon=True); th.start()
    result = {"label": a.label, "power_scheme": power_scheme(), "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    try:
        backend, dyn = load_gpu_backend(MODEL)
        tok = AutoTokenizer.from_pretrained(MODEL)
        train = load_dataset("openai/gsm8k", "main", split="train")
        prefix = build_fewshot_prefix(train.select(range(3)))
        idxs = json.loads((HERE.parent / "pilot_sample.json").read_text())["train_eval"][:3]
        prompts = []
        for i in idxs:
            p = prefix + f"Question: {train[i]['question']}\nAnswer:"
            prompts.append((p, tok(p, return_tensors="pt").input_ids.to("cuda"), i))
        result["load_s"] = time.time() - t_start
        result["micro"] = microbench(backend, prompts[0][1])
        elapsed = (time.time() - t_start) / 60
        if prior + elapsed + 9.0 > a.cap_min:   # generation phase is ~6-8 min; keep the cap hard
            result["generation"] = "SKIPPED: would exceed perf GPU cap"
        else:
            result["generation"] = run_generation(backend, dyn, tok, prompts, a.reps, a.max_new)
    finally:
        PHASE["name"] = "done"
        time.sleep(1.0); stop.set()
        if "p" in SAMPLER:
            SAMPLER["p"].terminate()
            try:
                SAMPLER["p"].wait(timeout=5)
            except subprocess.TimeoutExpired:
                SAMPLER["p"].kill()
        secs = time.time() - t_start
        with open(LEDGER, "a") as f:
            f.write(json.dumps({"label": a.label, "seconds": secs, "start": result["started"]}) + "\n")
        result["telemetry_columns"] = ["t", "phase"] + QUERY.split(",")
        result["telemetry"] = rows
        result["power_scheme_end"] = power_scheme()
        (HERE / f"bench_{a.label}.json").write_text(json.dumps(result))
        print(f"[{a.label}] done in {secs/60:.1f} min; cumulative perf GPU time {ledger_minutes():.1f} min of {a.cap_min}")


if __name__ == "__main__":
    main()
