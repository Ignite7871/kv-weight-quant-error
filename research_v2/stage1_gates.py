"""Stage 1 correctness / feasibility / speed gates and the runtime projection (STAGE1_SPEC.md section 6).

Pass criteria are fixed in the spec BEFORE these run:
  - token ids identical (hard) and final caches bitwise equal (hard) wherever an equivalence is claimed;
  - any failure stops the program; nothing is relaxed after the fact.
"""

from __future__ import annotations

import copy
import json
import math
import statistics
import subprocess
import time
from pathlib import Path

import torch

from rv2.boundary import find_boundary, score, would_stop_at_boundary
from rv2.engine import generate
from rv2.policies import StaticFP16, StaticNF4
from rv2.stage1 import CACHE_ONLY, TOKEN_ONLY, Cell, generate_cell

MAX_NEW = 256
OCT23 = {"fp16_pass_ms_sync": 39.0, "nf4_pass_ms_sync": 66.1, "fp16_step_ms_lean": 39.0, "nf4_step_ms_lean": 63.1,
         "both_branch_step_ms": 102.0}     # perf/PERF_REPORT.md, Balanced run A1 and the cost model


# ----------------------------------------------------------------------------- helpers
def with_cache(be, fn):
    """Run fn() and also return the last cache object produced through Backend.run."""
    cap, orig = {}, be.run

    def run(ids, cache, use_4bit):
        out = orig(ids, cache, use_4bit)
        cap["cache"] = out[1]
        return out
    be.run = run
    try:
        res = fn()
    finally:
        be.run = orig
    return res, cap["cache"]


def cache_t(cache):
    return [(lay.keys.clone(), lay.values.clone()) for lay in cache.layers]


def caches_equal(a, b):
    ta, tb = cache_t(a), cache_t(b)
    return len(ta) == len(tb) and all(torch.equal(k1, k2) and torch.equal(v1, v2) for (k1, v1), (k2, v2) in zip(ta, tb))


@torch.no_grad()
def reference_cell(be, prompt_ids, token_src, kv_src, n):
    """INDEPENDENT reference (no crop/restore, no engine code): the token-source branch runs on a deep copy of the
    cache; the real cache advances with the kept precision only."""
    be.routing_state.use_4bit = False
    out = be.model(prompt_ids, past_key_values=None, use_cache=True)
    cache, ids = out.past_key_values, [int(out.logits[0, -1].argmax())]
    for _ in range(1, n):
        cur = torch.tensor([[ids[-1]]], device=prompt_ids.device)
        tok_cache = copy.deepcopy(cache)
        be.routing_state.use_4bit = token_src == "N"
        lg = be.model(cur, past_key_values=tok_cache, use_cache=True).logits[0, -1]
        be.routing_state.use_4bit = kv_src == "N"
        be.model(cur, past_key_values=cache, use_cache=True)
        ids.append(int(lg.argmax()))
    return ids, cache


@torch.no_grad()
def teacher_forced_cache(be, prompt_ids, ids, kv_src):
    be.routing_state.use_4bit = False
    cache = be.model(prompt_ids, past_key_values=None, use_cache=True).past_key_values
    for s in range(1, len(ids)):
        be.routing_state.use_4bit = kv_src == "N"
        be.model(torch.tensor([[ids[s - 1]]], device=prompt_ids.device), past_key_values=cache, use_cache=True)
    return cache


def stop_fn(tok):
    return lambda ids: would_stop_at_boundary(tok, ids)


def fail(report, out_path, msg):
    report["STOP"] = msg
    report["all_passed"] = False
    out_path.write_text(json.dumps(report, indent=1))
    print(f"[GATE FAILED] {msg}", flush=True)
    return False


# ----------------------------------------------------------------------------- equivalence block (GPU)
def gate_equiv(be, dyn, tok, wb, s, data, out):
    gates = out / "gates"
    gates.mkdir(parents=True, exist_ok=True)
    path = gates / "gate_equiv.json"
    rep = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "criteria": "ids identical AND final caches bitwise equal; any failure stops",
           "problems": s["equiv_train"], "checks": []}
    full = {}
    # ---- A: pure cells == rv2 static arms == the UNTOUCHED historical DynamicQwen.generate (full 256-token generation)
    for idx in s["equiv_train"]:
        prob, prompt = data.problem("train", idx)
        pid = tok(prompt, return_tensors="pt").input_ids.to(be.device)
        rec = {"index": idx}
        for letter, tau in (("F", -1.0), ("N", 100.0)):     # historical forced-FP16 / pure-4-bit thresholds
            dyn.entropy_threshold = tau
            dyn.generate(prompt, tok, MAX_NEW)
            hist_ids = list(dyn.last_generated_ids)
            pol = StaticFP16() if letter == "F" else StaticNF4()
            r_static, c_static = with_cache(be, lambda: generate(be, pid, pol, MAX_NEW, tok.eos_token_id, idx, instrument=False))
            r_cell, c_cell = with_cache(be, lambda: generate_cell(be, pid, Cell(letter, letter), MAX_NEW, tok.eos_token_id))
            ok = (hist_ids == r_static.ids == r_cell.ids) and caches_equal(c_cell, c_static)
            rec[letter] = {"ids_hist_eq_static_eq_cell": hist_ids == r_static.ids == r_cell.ids,
                           "cell_cache_bitwise_eq_static": caches_equal(c_cell, c_static), "n_tokens": len(hist_ids), "ok": ok}
            full[(idx, letter)] = r_static.ids
            print(f"[equiv A] idx={idx} {letter}: {rec[letter]}", flush=True)
            if not ok:
                rep["checks"].append({"A": rec})
                return fail(rep, path, f"pure-cell equivalence failed (problem {idx}, precision {letter})")
        rep["checks"].append({"A": rec})
    # ---- B: hybrids vs the independent deep-copy reference, and cache == single-precision teacher-forced replay
    for idx in s["equiv_train"][:3]:
        prob, prompt = data.problem("train", idx)
        pid = tok(prompt, return_tensors="pt").input_ids.to(be.device)
        n = 96
        for cell in (CACHE_ONLY, TOKEN_ONLY):
            r, c = with_cache(be, lambda: generate_cell(be, pid, cell, n, tok.eos_token_id))
            ids_ref, c_ref = reference_cell(be, pid, cell.token_source, cell.kv_source, n)
            replay = teacher_forced_cache(be, pid, r.ids, cell.kv_source)
            ok_ids = r.ids == ids_ref
            ok_c = caches_equal(c, c_ref)
            ok_rp = caches_equal(c, replay)
            src_ok = all(r.ids[k + 1] == (r.trace["tF"][k] if cell.token_source == "F" else r.trace["tN"][k]) for k in range(len(r.routes)))
            flips = sum(a != b for a, b in zip(r.trace["tF"], r.trace["tN"]))
            rec = {"index": idx, "cell": cell.name, "ids_eq_reference": ok_ids, "cache_bitwise_eq_reference": ok_c,
                   "cache_bitwise_eq_single_precision_replay": ok_rp, "token_equals_declared_source_argmax": src_ok,
                   "local_branch_flips": flips, "n_tokens": len(r.ids)}
            rep["checks"].append({"B": rec})
            print(f"[equiv B] {rec}", flush=True)
            if not (ok_ids and ok_c and ok_rp and src_ok):
                return fail(rep, path, f"hybrid cell {cell.name} deviates from the independent reference (problem {idx})")
    # ---- C: early stop at the drift marker leaves outcomes identical (static arms full length vs early stopped)
    for idx in s["equiv_train"]:
        prob, prompt = data.problem("train", idx)
        pid = tok(prompt, return_tensors="pt").input_ids.to(be.device)
        for letter in ("F", "N"):
            pol = StaticFP16() if letter == "F" else StaticNF4()
            r_e = generate(be, pid, pol, MAX_NEW, tok.eos_token_id, idx, early_stop_fn=stop_fn(tok), instrument=False)
            ids_full = full[(idx, letter)]
            b_f, b_e = find_boundary(tok, ids_full, tok.eos_token_id, MAX_NEW), find_boundary(tok, r_e.ids, tok.eos_token_id, MAX_NEW)
            t_f, t_e = tok.decode(ids_full, skip_special_tokens=True), tok.decode(r_e.ids, skip_special_tokens=True)
            sc_f, sc_e = score(prob, t_f, b_f), score(prob, t_e, b_e)
            ok = (r_e.ids == ids_full[: len(r_e.ids)] and b_f.boundary_char == b_e.boundary_char and
                  t_f[: b_f.boundary_char] == t_e[: b_e.boundary_char] and sc_f["status"] == sc_e["status"])
            rec = {"index": idx, "precision": letter, "steps_full": len(ids_full), "steps_early": len(r_e.ids), "stopped_early": r_e.stopped_early,
                   "prefix_identical": r_e.ids == ids_full[: len(r_e.ids)], "same_boundary": b_f.boundary_char == b_e.boundary_char,
                   "same_status": sc_f["status"] == sc_e["status"], "ok": ok}
            rep["checks"].append({"C": rec})
            print(f"[equiv C] {rec}", flush=True)
            if not ok:
                return fail(rep, path, f"early-stop equivalence failed (problem {idx}, {letter})")
    for idx in s["equiv_train"][:2]:       # hybrid cell, full length vs early stopped
        prob, prompt = data.problem("train", idx)
        pid = tok(prompt, return_tensors="pt").input_ids.to(be.device)
        for cell in (CACHE_ONLY, TOKEN_ONLY):
            r_f = generate_cell(be, pid, cell, MAX_NEW, tok.eos_token_id)
            r_e = generate_cell(be, pid, cell, MAX_NEW, tok.eos_token_id, early_stop_fn=stop_fn(tok))
            b_f, b_e = find_boundary(tok, r_f.ids, tok.eos_token_id, MAX_NEW), find_boundary(tok, r_e.ids, tok.eos_token_id, MAX_NEW)
            t_f, t_e = tok.decode(r_f.ids, skip_special_tokens=True), tok.decode(r_e.ids, skip_special_tokens=True)
            ok = (r_e.ids == r_f.ids[: len(r_e.ids)] and b_f.boundary_char == b_e.boundary_char and
                  score(prob, t_f, b_f)["status"] == score(prob, t_e, b_e)["status"])
            rec = {"index": idx, "cell": cell.name, "steps_full": len(r_f.ids), "steps_early": len(r_e.ids), "ok": ok}
            rep["checks"].append({"C_hybrid": rec})
            print(f"[equiv C hybrid] {rec}", flush=True)
            if not ok:
                return fail(rep, path, f"hybrid early-stop equivalence failed (problem {idx}, {cell.name})")
    rep["all_passed"] = True
    path.write_text(json.dumps(rep, indent=1))
    print("[GATE PASSED] equivalence block: all checks passed", flush=True)
    return True


# ----------------------------------------------------------------------------- feasibility analysis (CPU, on records)
def _load(d, arm):
    p = d / f"{arm}.jsonl"
    return {r["index"]: r for r in (json.loads(l) for l in p.read_text().splitlines() if l.strip())}


def first_divergence(a, b):
    n = min(len(a), len(b))
    for k in range(n):
        if a[k] != b[k]:
            return k
    return None if len(a) == len(b) else n


def check_feasibility(d, indices):
    arms = ["fp16", "nf4", "cache_only", "token_only", "oracle_flip"]
    R = {a: _load(d, a) for a in arms}
    rep = {"n_problems": len(indices), "semantics": {}, "behaviour": {}, "timing": {}}
    ok = True
    for arm in ("cache_only", "token_only"):
        tk = "F" if arm == "cache_only" else "N"
        bad = 0
        for i in indices:
            r = R[arm][i]
            ids, tF, tN = r["ids"], r["tF"], r["tN"]
            assert r["token_source"] == tk and r["kv_source"] == ("N" if arm == "cache_only" else "F")
            for k in range(len(tF)):
                if ids[k + 1] != (tF[k] if tk == "F" else tN[k]):
                    bad += 1
            # step 1: the cache is the shared FP16 prefill, so the branches must equal the static arms' step-1 tokens
            if tF[0] != R["fp16"][i]["ids"][1] or tN[0] != R["nf4"][i]["ids"][1]:
                bad += 1
            # AMENDMENT A1 (STAGE1_SPEC.md section 10): JS is >= 0 in exact arithmetic but is computed in float32 over a 151k
            # vocabulary; the first feasibility run logged -9.7e-9 at a step where both branches were near-identical. The lower
            # bound is therefore -1e-6 (it was 0.0 in the first run, which reported a failure for that reason only).
            if any(not (-1e-6 <= x <= math.log(2) + 1e-6) for x in r["js"]):
                bad += 1
        rep["semantics"][arm] = {"violations": bad}
        ok &= bad == 0
    # flip oracle matches cache-only through its first flip token (same history, same cache, FP16 token)
    viol = 0
    for i in indices:
        fo, co = R["oracle_flip"][i], R["cache_only"][i]
        j = fo["routes"].find("F")
        if j >= 0:
            m = j + 2
            if fo["ids"][:m] != co["ids"][:m]:
                viol += 1
    rep["semantics"]["flip_oracle_matches_cache_only_until_first_flip"] = {"violations": viol}
    ok &= viol == 0
    # non-vacuity and behaviour
    for arm, ref in (("cache_only", "fp16"), ("token_only", "fp16"), ("nf4", "fp16"), ("oracle_flip", "fp16"), ("cache_only", "nf4"), ("token_only", "nf4")):
        div = [first_divergence(R[arm][i]["ids"], R[ref][i]["ids"]) for i in indices]
        diverged = [x for x in div if x is not None]
        rep["behaviour"][f"{arm}_vs_{ref}"] = {"n_diverged": len(diverged), "of": len(indices),
                                                "median_first_divergence_token": statistics.median(diverged) if diverged else None}
    for arm in arms:
        recs = [R[arm][i] for i in indices]
        steps = sum(r["steps"] for r in recs)
        gen = sum(r["gen_seconds"] for r in recs)
        rep["timing"][arm] = {"mean_steps": steps / len(recs), "mean_wall_s": sum(r["wall_seconds"] for r in recs) / len(recs),
                              "ms_per_step": 1000 * gen / steps, "solved": sum(r["status"] == "SOLVED" for r in recs),
                              "peak_mem_gb": max(r["peak_mem_allocated"] for r in recs) / 1e9,
                              "stopped_early": sum(r["stopped_early"] for r in recs)}
    nonvacuous = rep["behaviour"]["cache_only_vs_fp16"]["n_diverged"] > 0 and rep["behaviour"]["token_only_vs_fp16"]["n_diverged"] > 0
    rep["semantics_ok"] = ok
    rep["hybrids_nonvacuous_vs_fp16"] = nonvacuous
    (d / "feasibility_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    print(f"[feasibility] semantics_ok={ok} nonvacuous={nonvacuous}", flush=True)
    return ok


# ----------------------------------------------------------------------------- brief fixed-workload speed re-check (GPU)
def _telemetry():
    q = "clocks.sm,clocks.mem,utilization.gpu,temperature.gpu,power.draw,pstate"
    try:
        return subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"unavailable: {e}"


@torch.no_grad()
def speed_check(be, tok, data, s, out):
    gates = out / "gates"
    gates.mkdir(parents=True, exist_ok=True)
    rep = {"workload": "pass micro-benchmark (fixed prompt, constant cache length, synchronous) + generation of 2 fixed train problems x "
                       "{fp16, nf4, cache_only} x 2 reps, 64 new tokens, lean engine", "telemetry_start": _telemetry()}
    idx = s["speed_train"][0]
    prob, prompt = data.problem("train", idx)
    pid = tok(prompt, return_tensors="pt").input_ids.to(be.device)
    logits, cache = be.run(pid, None, False)
    L = cache.get_seq_length()
    cur = torch.tensor([[int(torch.argmax(logits, -1))]], device=be.device)
    micro = {}
    for name, flag in (("fp16", False), ("nf4", True)):
        for _ in range(20):
            be.run(cur, cache, flag); cache.crop(L)
        blocks = []
        for _ in range(3):
            t0 = time.perf_counter()
            for _ in range(40):
                be.run(cur, cache, flag); torch.cuda.synchronize(); cache.crop(L)
            blocks.append(1000 * (time.perf_counter() - t0) / 40)
        micro[name] = {"ms_per_pass_blocks": blocks, "mean": sum(blocks) / 3}
    rep["micro_ms_per_pass"] = micro
    gens = {a: [] for a in ("fp16", "nf4", "cache_only")}
    toks = {}
    for rep_i in range(3):                                     # first repetition = warm-up (recorded, excluded from the mean)
        for k, i in enumerate(s["speed_train"]):
            p2, pr2 = data.problem("train", i)
            pid2 = tok(pr2, return_tensors="pt").input_ids.to(be.device)
            order = ["fp16", "nf4", "cache_only"]
            order = order[(rep_i + k) % 3:] + order[: (rep_i + k) % 3]
            for arm in order:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                if arm == "cache_only":
                    r = generate_cell(be, pid2, CACHE_ONLY, 64, tok.eos_token_id, log=False)
                else:
                    r = generate(be, pid2, StaticFP16() if arm == "fp16" else StaticNF4(), 64, tok.eos_token_id, 0, instrument=False)
                torch.cuda.synchronize()
                w = time.perf_counter() - t0
                gens[arm].append({"rep": rep_i, "problem": i, "ms_per_step": 1000 * w / max(1, len(r.routes))})
                key = (arm, i)
                toks.setdefault(key, set()).add(tuple(r.ids))
    rep["token_determinism_across_reps"] = all(len(v) == 1 for v in toks.values())
    summary = {}
    for arm, rows in gens.items():
        v = [x["ms_per_step"] for x in rows if x["rep"] > 0]
        summary[arm] = {"mean_ms_per_step": sum(v) / len(v), "sd": statistics.pstdev(v), "n": len(v)}
    rep["generation_ms_per_step"] = summary
    rep["vs_2026_10_23"] = {
        "fp16_pass": micro["fp16"]["mean"] / OCT23["fp16_pass_ms_sync"], "nf4_pass": micro["nf4"]["mean"] / OCT23["nf4_pass_ms_sync"],
        "fp16_step": summary["fp16"]["mean_ms_per_step"] / OCT23["fp16_step_ms_lean"], "nf4_step": summary["nf4"]["mean_ms_per_step"] / OCT23["nf4_step_ms_lean"],
        "both_branch_step_ratio_to_sum_of_single": summary["cache_only"]["mean_ms_per_step"] / (summary["fp16"]["mean_ms_per_step"] + summary["nf4"]["mean_ms_per_step"])}
    rep["telemetry_end"] = _telemetry()
    (gates / "speed.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1), flush=True)
    return True


# ----------------------------------------------------------------------------- runtime projection (CPU)
def pilot_early_stop_steps(tokenizer, pilot_dir):
    """Early-stop length (routed steps) of the pilot's stored full-length FP16 and NF4 trajectories on 30 historical-pool
    TEST problems (gate0). Used only to size the run; no outcomes are read."""
    res = {}
    for arm in ("fp16", "nf4"):
        p = pilot_dir / "gate0" / f"{arm}.jsonl"
        lens = []
        for l in p.read_text().splitlines():
            if not l.strip():
                continue
            ids = json.loads(l)["ids"]
            k = next((k for k in range(1, len(ids) + 1) if would_stop_at_boundary(tokenizer, ids[:k])), len(ids))
            lens.append(k - 1)                     # routed steps = tokens - 1 (token 0 is the prefill's)
        res[arm] = lens
    return res


def project(out, s, arms, cap_hours, used_seconds):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    feas = json.loads((out / "feasibility" / "feasibility_report.json").read_text())
    speed = json.loads((out / "gates" / "speed.json").read_text())
    pilot = pilot_early_stop_steps(tok, Path(__file__).resolve().parent / "pilot_runs")
    mean_steps = {a: sum(v) / len(v) for a, v in pilot.items()}
    se = {a: statistics.pstdev(v) / math.sqrt(len(v)) for a, v in pilot.items()}
    sf, sn = mean_steps["fp16"], mean_steps["nf4"]
    hybrid_steps = (sf + sn) / 2                                       # hybrids follow between the two static trajectories
    steps = {"fp16": sf, "nf4": sn, "cache_only": hybrid_steps, "token_only": hybrid_steps, "oracle_flip": hybrid_steps}
    # ms/step: today's speed re-check for static arms; both-branch arms from the feasibility run (oracle ~ cache-only)
    ms_static = {"fp16": speed["generation_ms_per_step"]["fp16"]["mean_ms_per_step"], "nf4": speed["generation_ms_per_step"]["nf4"]["mean_ms_per_step"]}
    ms_feas = {a: feas["timing"][a]["ms_per_step"] for a in arms}
    ms_both_speed = speed["generation_ms_per_step"]["cache_only"]["mean_ms_per_step"]
    per_arm = {}
    for a in arms:
        ms_c = (ms_static[a] + ms_feas[a]) / 2 if a in ms_static else (ms_both_speed + ms_feas[a]) / 2
        per_arm[a] = {"steps": steps[a], "ms_per_step_central": ms_c, "ms_speed_check": ms_static.get(a, ms_both_speed), "ms_feasibility": ms_feas[a]}
    # per-problem overhead measured in feasibility: wall - gen seconds (prefill, boundary, I/O) per arm run
    overhead = max(0.0, sum(feas["timing"][a]["mean_wall_s"] for a in arms) - sum(feas["timing"][a]["mean_steps"] * feas["timing"][a]["ms_per_step"] / 1000 for a in arms))
    sec = sum(per_arm[a]["steps"] * per_arm[a]["ms_per_step_central"] / 1000 for a in arms) + overhead
    # uncertainty: +-12% run-to-run rate drift (perf study) combined with the standard error of the step counts
    rel_steps = max(se[a] / mean_steps[a] for a in ("fp16", "nf4"))
    low, high = sec * (1 - 0.12 - rel_steps), sec * (1 + 0.12 + rel_steps)
    n = len(s["run_order"])
    remaining_cap = cap_hours * 3600 - used_seconds
    proj = {"n_problems": n, "per_arm": per_arm, "overhead_s_per_problem": overhead, "sec_per_problem_central": sec,
            "sec_per_problem_low": low, "sec_per_problem_high": high,
            "hours_central": sec * n / 3600, "hours_low": low * n / 3600, "hours_high": high * n / 3600,
            "gpu_hours_used_so_far": used_seconds / 3600, "gpu_hours_remaining_under_cap": remaining_cap / 3600,
            "cumulative_hours_central": (used_seconds + sec * n) / 3600, "cumulative_hours_high": (used_seconds + high * n) / 3600,
            "fits_cap_central": used_seconds + sec * n <= cap_hours * 3600,
            "fits_cap_high": used_seconds + high * n <= cap_hours * 3600,
            "step_count_source": "30 historical-pool test problems (pilot gate0), early-stop rule; mean steps fp16/nf4: %.1f / %.1f" % (sf, sn),
            "pilot_mean_steps": mean_steps, "pilot_se_steps": se, "cap_hours": cap_hours}
    (out / "projection.json").write_text(json.dumps(proj, indent=1))
    print(json.dumps(proj, indent=1))
    return proj
