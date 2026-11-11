"""Real-model CPU checks (PROTOCOL_V4.md section 12), on the laptop, where memory allows (16 GB RAM: models up to 3B in FP16 + NF4).
LOGIC only: CPU NF4 numerics differ from CUDA; no outcome is computed or read. This harness never runs on the pod.

Per model (one subprocess each, so memory is released between models):
  G7  the local snapshot is the pinned revision and every shard's SHA-256 equals the manifest; the loaded config equals the manifest geometry,
      tied flag and vocab;
  G5  prompt SHA-256 against the sample, round trip, BOS, EOS, drift marker, with the real tokenizer and the GSM8K prompts;
  the six arms for 2 steps on test problem #1 (the first of the fixed order) with the real model: integrity flags, step-1 JS 0, X13a slot
  shapes at the model's head_dim, cache geometry and dtype.
Extra: BF16 mode (the D6 fallback) on qwen2.5-0.5b and llama3.2-1b; tokenizer-only G5 for the models whose weights are too large for this laptop (qwen2.5-7b, mistral-7b-v0.3) and
for llama3.2-3b through byte-identical tokenizer files (llama3.1-8b's tokenizer differs and is checked at G5 on the pod).

  venv/Scripts/python.exe research_v2/v4/tests/real_model_cpu_v4.py [--write <json>]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
V4 = HERE.parent
RV = V4.parent
STEPS = 2
FULL = ["qwen2.5-0.5b", "llama3.2-1b-instruct", "llama3.2-1b", "qwen2.5-1.5b", "qwen2.5-3b"]
JOBS = ([("full", m, "fp16", "nf4") for m in FULL] + [("full", m, "bf16", "nf4") for m in ("qwen2.5-0.5b", "llama3.2-1b")]
        + [("tok", m, None, None) for m in ("qwen2.5-7b", "mistral-7b-v0.3", "llama3.2-3b")])
NOT_POSSIBLE = {"qwen2.5-7b": "weights 15.2 GB in FP16 exceed this laptop's 16 GB RAM with the NF4 copy; checked on the pod (G1-G8)",
                "llama3.1-8b": "gated repo not readable with the laptop account, and 16 GB in FP16; checked on the pod",
                "llama3.2-3b": "gated repo not readable with the laptop account; snapshot checks and G5 through identical bytes here, the rest on the pod",
                "mistral-7b-v0.3": "weights 14.5 GB in FP16 exceed this laptop's RAM with the NF4 copy; checked on the pod"}


def one(kind, m, prec, q):
    sys.path.insert(0, str(RV))
    sys.path.insert(0, str(V4))
    import run_v4 as R
    import torch
    import rv4
    from rv2.engine import generate
    from rv2.policies import StaticFP16
    from rv2.boundary import would_stop_at_boundary
    rv4.apply_determinism()
    man = rv4.manifest()
    s = json.loads((V4 / "v4_sample.json").read_text())
    data = R.RS1.Data()
    out = {"kind": kind, "model": m, "precision": prec, "quantizer": q}
    e = man[m]
    if kind == "tok":
        from transformers import AutoTokenizer
        src = m
        if m == "llama3.2-3b":
            src = "llama3.2-1b"
            same = all(man[m]["small_files"][f] == man[src]["small_files"][f] for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"))
            out["tokenizer_bytes_identical_to"] = {"model": src, "identical": same}
        es = man[src]
        tok = AutoTokenizer.from_pretrained(es["hf_id"], revision=es["revision"], local_files_only=True)
        out["G5"] = R.g5(tok, data, s, e)
        out["pass"] = bool(out["G5"]["pass"] and out.get("tokenizer_bytes_identical_to", {"identical": True})["identical"])
        return out
    snap = R.snapshot_dir(e)
    out["G7_snapshot"] = R.check_snapshot(e, snap)
    t0 = time.time()
    be, tok = rv4.load_model(m, "cpu", prec, q, local_only=True)
    out["load_s"] = round(time.time() - t0, 1)
    snapc = rv4.config_snapshot(be.model.config)
    out["G7_config"] = {**snapc, "pass": rv4.geometry_matches(snapc, e["geometry"])}
    out["G5"] = R.g5(tok, data, s, e)
    dtype = rv4.DTYPES[prec]
    idx = s["tierA_ids"][0]
    ids_t = R.prompt_ids(tok, data, "test", idx, "cpu")
    ref = generate(be, ids_t, StaticFP16(), STEPS + 1, tok.eos_token_id, 0, early_stop_fn=lambda ids: would_stop_at_boundary(tok, ids), instrument=False).ids
    t0 = time.time()
    recs, _ = R.run_problem(be, tok, data, m, idx, "test", R.TEST_ARMS, "V4cpu", dtype, slots_for=("NAT",), hist=ref, device="cpu")
    out["arms_s"] = round(time.time() - t0, 1)
    out["integrity_errors"] = R.integrity_errors(recs)
    out["step1_js"] = {a: r["js"][0] for a, r in recs.items()}
    sl = recs["NAT"]["_slots"]
    g = e["geometry"]
    out["slot_shapes"] = {k: list(v.shape) for k, v in sl.items()}
    out["slots_ok"] = list(sl["S1K"].shape) == [g["layers"], g["kv_heads"], g["head_dim"]] and sl["normsV"].shape[1:] == (g["layers"], g["kv_heads"])
    geo = rv4.cache_geometry(be, ids_t, dtype)
    out["G7_cache"] = {**geo, "n_dynamic": be.n_dynamic,
                       "pass": (geo["n_layers"], geo["n_kv_heads"], geo["head_dim"]) == (g["layers"], g["kv_heads"], g["head_dim"]) and geo["all_mode_dtype"]
                       and be.n_dynamic == 7 * g["layers"]}
    out["pass"] = bool(out["G7_snapshot"]["pass"] and out["G7_config"]["pass"] and out["G5"]["pass"] and not out["integrity_errors"]
                       and out["slots_ok"] and out["G7_cache"]["pass"])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", nargs=4)
    ap.add_argument("--write")
    a = ap.parse_args(argv)
    if a.one:
        kind, m, prec, q = [None if x == "-" else x for x in a.one]
        print("RESULT " + json.dumps(one(kind, m, prec, q), default=str))
        return 0
    res = []
    for kind, m, prec, q in JOBS:
        t0 = time.time()
        p = subprocess.run([sys.executable, __file__, "--one", kind, m, prec or "-", q or "-"], capture_output=True, text=True, encoding="utf-8", errors="replace")
        line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
        r = json.loads(line[-1][7:]) if line else {"kind": kind, "model": m, "pass": False, "error": p.stderr[-3000:]}
        r["seconds"] = round(time.time() - t0, 1)
        res.append(r)
        print(f"{'PASS' if r['pass'] else 'FAIL'} {kind:4s} {m:22s} {prec} {q} ({r['seconds']}s)", flush=True)
        if not r["pass"]:
            print(json.dumps(r, default=str)[:3000])
    n = sum(r["pass"] for r in res)
    print(f"{n}/{len(res)} real-model CPU checks passed")
    if a.write:
        Path(a.write).write_text(json.dumps({"passed": n, "n": len(res), "not_possible_on_this_laptop": NOT_POSSIBLE, "results": res}, indent=1, default=str))
    return 0 if n == len(res) else 1


if __name__ == "__main__":
    sys.exit(main())
