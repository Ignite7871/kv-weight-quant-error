"""Real-model CPU checks (PROTOCOL_V5.md section 11), on the laptop only (16 GB RAM; models up to 1.7B in FP16 + 4-bit). LOGIC only: CPU numerics
differ from CUDA; no outcome is computed or read. Never runs on the pod.

Per unit that fits (one subprocess each, so memory is released between them):
  G7  the local snapshot is the pinned revision and every shard's SHA-256 equals the manifest; the loaded config and attention layout (QK-norm,
      KV heads) equal the manifest;
  G5  GSM8K units: prompt SHA-256 against v4's sample, round trip, BOS as the manifest records, EOS, drift marker, with the real tokenizer;
  GW  WikiText units: the pinned parquet hashes; every sample paragraph, prompt and 65-token history rebuilt and equal to the sample's hashes;
  the six arms for 2 steps on the unit's first test problem with the real model (HQQ quantized for U4): integrity flags, step-1 JS 0, slot shapes.
Extra: Mistral-7B tokenizer-only GW (its weights exceed this laptop); the A-X references from the real locked v4 report (SHA-256 checked) equal
the protocol table; the anchor criteria and bitwise share computed on v4's own Qwen2.5-0.5B records against themselves (must pass, share 1).

  venv/Scripts/python.exe research_v2/v5/tests/real_model_cpu_v5.py [--write <json>]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
V5 = HERE.parent
RV = V5.parent
STEPS = 2
JOBS = [("unit", "U0"), ("unit", "U1"), ("unit", "U2"), ("unit", "U3"), ("unit", "U4"), ("tok", "mistral-7b-v0.3"), ("ax", None), ("anchor", None)]
NOT_POSSIBLE = {"U5": "Mistral-7B weights (14.5 GB FP16) exceed this laptop's RAM with the 4-bit copy; tokenizer-level GW here, the rest on the pod",
                "U6": "Mistral-7B HQQ: the same; checked by the CPU tiny-model HQQ tests and on the pod (GH, G1-G8)"}
PROTOCOL_AX = {("U1", "NAT-C1"): (-0.223, 0.204, -0.010), ("U1", "H1"): (-1.290, -0.984, -1.137), ("U2", "NAT-C1"): (0.309, -0.100, 0.104),
               ("U2", "H1"): (-0.757, -1.118, -0.937)}


def one(kind, arg):
    sys.path.insert(0, str(RV))
    sys.path.insert(0, str(V5))
    sys.path.insert(0, str(RV / "v4"))
    import run_v5 as R
    out = {"kind": kind, "arg": arg}
    if kind == "ax":
        import analyze_v5 as A
        r = A.ax_references()
        got = {k: (round(r[k[0]][k[1]]["size_pred"], 3), round(r[k[0]][k[1]]["hd_pred"], 3), round(r[k[0]][k[1]]["boundary"], 3)) for k in PROTOCOL_AX}
        out.update(refs={f"{k[0]} {k[1]}": v for k, v in got.items()}, pass_=all(abs(a - b) <= 0.0015 for k in PROTOCOL_AX for a, b in zip(got[k], PROTOCOL_AX[k])))
        out["pass"] = out.pop("pass_")
        return out
    if kind == "anchor":
        import analyze_v5 as A
        L = A.L
        v4 = A.V4_RUNS / A.ANCHOR_MODEL
        Au = {a: L.load_jsonl(v4 / "test" / f"{a}.jsonl") for a in A.ARMS}
        idx = sorted(set.intersection(*[set(v) for v in Au.values()]))
        cal = json.loads((v4 / "calibration.json").read_text(encoding="utf-8"))
        rep4 = A.v4_report()["per_model"][A.ANCHOR_MODEL]
        saved = A.OUT
        A.OUT = A.V4_RUNS.parent / "_anchor_self_check"
        try:
            import shutil
            if A.OUT.exists():
                shutil.rmtree(A.OUT)
            shutil.copytree(v4, A.OUT / "U0")
            x = A.x13a("U0", idx, A.manifest()[A.ANCHOR_MODEL]["geometry"]["head_dim"])
            r = A.anchor_core(Au, cal, cal, rep4, idx, x)
            bw = A.bitwise_share(idx)
        finally:
            shutil.rmtree(A.OUT, ignore_errors=True)
            A.OUT = saved
        out.update(n=len(idx), anchor_pass=r["pass"], bitwise=bw, h5=r["reported_H5"])
        out["pass"] = bool(r["pass"] and bw["share"] == 1.0 and len(idx) == 100 and all(v["within_0.10"] for v in r["reported_H5"].values()))
        return out
    import rv4
    import rv5
    rv4.apply_determinism()
    if kind == "tok":
        tok = rv5.load_tokenizer(arg)
        task = R.Task("U5")
        rep = R.gw(task, arg)
        out.update(GW=rep, pass_=rep["pass"])
        out["pass"] = out.pop("pass_")
        return out
    u = arg
    m, q, tname, _ = rv5.UNITS[u]
    e = rv5.models()[m]
    task = R.Task(u)
    snap = R.snapshot_dir(e)
    out["G7_snapshot"] = R.check_snapshot(e, snap)
    t0 = time.time()
    be, tok = rv5.load_model(m, "cpu", "fp16", q, local_only=True)
    out["load_s"] = round(time.time() - t0, 1)
    snapc = rv4.config_snapshot(be.model.config)
    att = rv5.attention_snapshot(be.model)
    out["G7_config"] = {**snapc, "attention": att, "pass": bool(rv4.geometry_matches(snapc, e["geometry"]) and rv5.attention_matches(att, e["geometry"]))}
    if task.wiki:
        out["GW"] = R.gw(task, m)
        g5_ok = out["GW"]["pass"]
    else:
        out["G5"] = R.g5(tok, task, e)
        g5_ok = out["G5"]["pass"]
    if q == "hqq":
        out["GH_oracle"] = rv5.hqq_oracle_ok(be.model)
        out["GH_config"] = rv5.hqq_config_ok(be.model)
    idx = task.test_ids(100)[0]
    ids_t = task.prompt_ids(tok, "test", idx, "cpu")
    ref = task.history(be, tok, ids_t, "test", idx)[:STEPS + 1] if task.wiki else \
        R.generate(be, ids_t, R.StaticFP16(), STEPS + 1, tok.eos_token_id, 0, early_stop_fn=lambda ids: R.would_stop_at_boundary(tok, ids), instrument=False).ids
    t0 = time.time()
    recs, _ = R.run_problem(be, tok, task, u, idx, "test", R.TEST_ARMS, "V5cpu", rv4.DTYPES["fp16"], slots_for=("NAT",), hist=ref, device="cpu")
    out["arms_s"] = round(time.time() - t0, 1)
    out["integrity_errors"] = R.integrity_errors(recs)
    out["step1_js"] = {a: r["js"][0] for a, r in recs.items()}
    out["record_fields"] = sorted(recs["NAT"])
    sl = recs["NAT"]["_slots"]
    g = e["geometry"]
    out["slots_ok"] = list(sl["S1K"].shape) == [g["layers"], g["kv_heads"], g["head_dim"]]
    out["no_ids_in_wikitext_records"] = (not task.wiki) or all("top" not in r and "top_sha256" in r for r in recs.values())
    geo = rv4.cache_geometry(be, ids_t, rv4.DTYPES["fp16"])
    out["G7_cache"] = {**geo, "n_dynamic": be.n_dynamic, "pass": (geo["n_layers"], geo["n_kv_heads"], geo["head_dim"]) == (g["layers"], g["kv_heads"], g["head_dim"])
                       and geo["all_mode_dtype"] and be.n_dynamic == 7 * g["layers"]}
    out["pass"] = bool(out["G7_snapshot"]["pass"] and out["G7_config"]["pass"] and g5_ok and not out["integrity_errors"] and out["slots_ok"]
                       and out["G7_cache"]["pass"] and out["no_ids_in_wikitext_records"] and (q != "hqq" or (out["GH_oracle"]["pass"] and out["GH_config"])))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", nargs=2)
    ap.add_argument("--write")
    a = ap.parse_args(argv)
    if a.one:
        kind, arg = [None if x == "-" else x for x in a.one]
        print("RESULT " + json.dumps(one(kind, arg), default=str))
        return 0
    res = []
    for kind, arg in JOBS:
        t0 = time.time()
        p = subprocess.run([sys.executable, __file__, "--one", kind, arg or "-"], capture_output=True, text=True, encoding="utf-8", errors="replace")
        line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
        r = json.loads(line[-1][7:]) if line else {"kind": kind, "arg": arg, "pass": False, "error": p.stderr[-3000:]}
        r["seconds"] = round(time.time() - t0, 1)
        res.append(r)
        print(f"{'PASS' if r['pass'] else 'FAIL'} {kind:6s} {arg} ({r['seconds']}s)", flush=True)
        if not r["pass"]:
            print(json.dumps(r, default=str)[:3000])
    n = sum(r["pass"] for r in res)
    print(f"{n}/{len(res)} real-model CPU checks passed")
    if a.write:
        Path(a.write).write_text(json.dumps({"passed": n, "n": len(res), "not_possible_on_this_laptop": NOT_POSSIBLE, "results": res}, indent=1, default=str))
    return 0 if n == len(res) else 1


if __name__ == "__main__":
    sys.exit(main())
