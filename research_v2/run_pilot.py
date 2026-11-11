"""Feasibility pilot driver (PILOT_SPEC.md). GPU runs require --authorized-gpu-run.

Subcommands
  plan                      write pilot_sample.json (if absent) and print arms per block (CPU)
  run --block B [--arms]    generate for a block: gate0 | equiv | calib | eval | eval_random (GPU)
  calibrate                 thresholds from the calib block, exact decision signals + fixed tie rule (CPU)
  derive-random             matched random controls from realized eval budgets -> pilot_random_plan.json (CPU)
  ledger                    print cumulative GPU time
  cpu-smoke                 end-to-end I/O check with real Qwen2.5-0.5B on CPU, few tokens

GPU-time ceiling: every GPU invocation is logged in pilot_runs/gpu_ledger.jsonl (wall time of the
whole process, including model loading). Before each arm the driver projects that arm's time from
measured per-problem rates and refuses to start it if ledger + elapsed + projection > --cap-hours.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from datasets import load_dataset          # HF before torch
from transformers import AutoTokenizer

import torch

import rv2  # noqa: F401
from rv2 import PROJECT_ROOT
from rv2.boundary import find_boundary, score
from rv2.engine import account, generate
from rv2.policies import (CascadeNF4Margin, LagEntropy, LagMargin, OracleFlip, OracleFP16Entropy, OracleJS,
                          RandomBernoulli, StaticFP16, StaticNF4)
from probe_gsm8k_routing import build_fewshot_prefix   # historical prompt builder

MODEL = "Qwen/Qwen2.5-0.5B"
MAX_NEW_TOKENS = 256          # historical
N_FEWSHOT = 3                 # historical: first three train examples
OUT = HERE / "pilot_runs"
LEDGER = OUT / "gpu_ledger.jsonl"
SAMPLE_FILE = HERE / "pilot_sample.json"
THRESH_FILE = HERE / "pilot_thresholds.json"
RANDOM_PLAN = HERE / "pilot_random_plan.json"
TRAIN_SEED, GATE_SEED = "rv2-pilot-train-2026-10-22", "rv2-gate0-2026-10-22"
BUDGETS = {"B10": 0.10, "B30": 0.30}   # target FP16 share of routed steps, full-generation definition
# Historical thresholds exactly as used: p50 = 1.943 (gsm8k_routing_comparison.MEDIAN_THRESHOLD);
# p10/p30 = results/calibration_thresholds.json values used by gsm8k_multibudget_sweep.py.
HIST_TAU = {"p10": 0.06795654445886612, "p30": 0.9259277582168579, "p50": 1.943}
HIST_FP16_TAU, HIST_NF4_TAU = -1.0, 100.0   # historical forced-FP16 / pure-4-bit thresholds
SHARE_TOL = 0.010   # two policies share one matched random control if realized shares differ by <= 1.0 pp
DEFAULT_RATE = {"pre": 10.0, "cascade": 12.0, "oracle": 18.0, "histimpl": 10.0}   # s/problem, pre-measurement


# ----------------------------------------------------------------------------- sampling
def make_sample():
    if SAMPLE_FILE.exists():
        return json.loads(SAMPLE_FILE.read_text())
    train = load_dataset("openai/gsm8k", "main", split="train")
    pool = list(range(N_FEWSHOT, len(train)))          # exclude the few-shot exemplars 0..2
    pick = random.Random(TRAIN_SEED).sample(pool, 100)
    calib, evalb = sorted(pick[:50]), sorted(pick[50:])
    hist930 = sorted(r["index"] for r in json.load(open(PROJECT_ROOT / "results" / "gsm8k_multibudget_p10_entropy_only.json"))["results"])
    assert len(hist930) == 930
    gate0 = sorted(random.Random(GATE_SEED).sample(hist930, 30))
    s = {"train_calib": calib, "train_eval": evalb, "test_gate0": gate0, "train_equiv": evalb[:5],
         "seeds": {"train": TRAIN_SEED, "gate0": GATE_SEED}, "n_train_split": len(train)}
    SAMPLE_FILE.write_text(json.dumps(s, indent=1))
    return s


# ----------------------------------------------------------------------------- arms
class HistImpl:
    """Not a policy: drives the UNTOUCHED historical DynamicQwen.generate at a threshold (gate0 only)."""
    kind = "histimpl"
    deployable = True

    def __init__(self, tau, name):
        self.tau, self.name = tau, name

    def describe(self):
        return {"name": self.name, "kind": "histimpl", "tau": self.tau,
                "implementation": "src/dynamic_model.py DynamicQwen.generate (unmodified)"}


EVAL_PRIORITY = [   # pre-declared run order; later items are dropped first if the GPU cap binds
    "fp16", "nf4", "oracle_flip", "lag_entropy_B10", "lag_margin_B10", "cascade_nf4_margin_B10",
    "@random:B10",                                        # derived controls for the B10 primary arms
    "lag_entropy_B30", "lag_margin_B30", "cascade_nf4_margin_B30",
    "@random:B30",
    "oracle_js_B10", "oracle_fp16_entropy_B10",           # secondary diagnostic references
    "@random:answer",                                     # secondary answer-only matched controls
    "@random:seeds",                                      # extra seeds for the primary B10 control
]


def arms_for(block, th=None, plan=None):
    th = th or {}
    if block == "gate0":
        return ([StaticFP16(), StaticNF4()] + [LagEntropy(t, f"hist_lag_entropy_{k}") for k, t in HIST_TAU.items()]
                + [HistImpl(HIST_TAU["p10"], "histimpl_lag_entropy_p10"), HistImpl(HIST_TAU["p30"], "histimpl_lag_entropy_p30"),
                   HistImpl(HIST_TAU["p50"], "histimpl_lag_entropy_p50"), HistImpl(HIST_FP16_TAU, "histimpl_fp16"),
                   HistImpl(HIST_NF4_TAU, "histimpl_nf4")])
    if block == "calib":
        return [StaticFP16(), StaticNF4(), OracleFlip()]
    if block == "equiv":
        return [StaticFP16(), StaticNF4(), CascadeNF4Margin(2.0, "equiv_cascade_always"),
                CascadeNF4Margin(-1.0, "equiv_cascade_never"), OracleJS(-1.0, "equiv_oracle_always_fp16"),
                OracleJS(10.0, "equiv_oracle_never_fp16")]
    if block == "eval":
        a = [StaticFP16(), StaticNF4(), OracleFlip()]
        for b in BUDGETS:
            if f"lag_entropy_{b}" in th:
                a += [LagEntropy(th[f"lag_entropy_{b}"]["tau"], f"lag_entropy_{b}"),
                      LagMargin(th[f"lag_margin_{b}"]["tau"], f"lag_margin_{b}"),
                      CascadeNF4Margin(th[f"cascade_nf4_margin_{b}"]["tau"], f"cascade_nf4_margin_{b}")]
        if "oracle_js_B10" in th:
            a += [OracleJS(th["oracle_js_B10"]["tau"], "oracle_js_B10"),
                  OracleFP16Entropy(th["oracle_fp16_entropy_B10"]["tau"], "oracle_fp16_entropy_B10")]
        return a
    if block == "eval_random":
        return [RandomBernoulli(r["p_fp16"], r["name"], r["name"]) for r in (plan or []) if not r.get("infeasible")]
    raise ValueError(block)


def block_indices(block, s):
    return {"gate0": ("test", s["test_gate0"]), "calib": ("train", s["train_calib"]),
            "eval": ("train", s["train_eval"]), "eval_random": ("train", s["train_eval"]),
            "equiv": ("train", s["train_equiv"])}[block]


# ----------------------------------------------------------------------------- provenance
def sha256(p: Path):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def provenance():
    code = sorted((HERE / "rv2").glob("*.py")) + [Path(__file__)]
    hist = [PROJECT_ROOT / "src" / f for f in ("dynamic_linear.py", "dynamic_model.py", "routing.py", "fragility.py")]
    hist += [PROJECT_ROOT / "scripts" / f for f in ("gsm8k_gate_check.py", "probe_gsm8k_routing.py")]
    git = subprocess.run(["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(PROJECT_ROOT), "status", "--porcelain", "research_v2/rv2", "research_v2/run_pilot.py",
                            "src", "scripts"], capture_output=True, text=True).stdout.strip()
    import transformers, bitsandbytes
    return {"git_head": git, "git_dirty_paths": dirty.splitlines(),
            "code_sha256": {str(p.relative_to(PROJECT_ROOT)): sha256(p) for p in code + hist},
            "versions": {"python": platform.python_version(), "torch": torch.__version__,
                         "transformers": transformers.__version__, "bitsandbytes": bitsandbytes.__version__,
                         "cuda": torch.version.cuda,
                         "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
            "model": MODEL, "max_new_tokens": MAX_NEW_TOKENS}


# ----------------------------------------------------------------------------- GPU ledger / cap
def ledger_seconds():
    if not LEDGER.exists():
        return 0.0
    return sum(json.loads(l)["seconds"] for l in LEDGER.read_text().splitlines() if l.strip())


def measured_rate(kind):
    """Mean measured wall seconds per problem for completed GPU records of this policy kind."""
    tot, n = 0.0, 0
    for p in OUT.glob("*/*.jsonl"):
        if p.parent.name == "smoke":
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("kind") == kind and r.get("wall_seconds"):
                tot += r["wall_seconds"]; n += 1
    return tot / n if n >= 10 else DEFAULT_RATE[kind]


# ----------------------------------------------------------------------------- run
def run_block(block, backend, tok, arms, indices, split, max_new=MAX_NEW_TOKENS, out_dir=OUT, wb=None,
              dyn=None, cap_hours=None, t_proc0=None, prior_seconds=0.0):
    train = load_dataset("openai/gsm8k", "main", split="train")
    ds = train if split == "train" else load_dataset("openai/gsm8k", "main", split="test")
    prefix = build_fewshot_prefix(train.select(range(N_FEWSHOT)))
    (out_dir / block).mkdir(parents=True, exist_ok=True)
    prov = provenance()
    cuda = backend.device.startswith("cuda")
    for pol in arms:
        path = out_dir / block / f"{pol.name}.jsonl"
        done = set()
        if path.exists():   # resume
            done = {json.loads(l)["index"] for l in path.read_text().splitlines() if l.strip()}
        todo = [i for i in indices if i not in done]
        if not todo:
            continue
        if cap_hours is not None:
            proj = 1.15 * measured_rate(pol.kind) * len(todo)   # 15% safety margin keeps the cap hard
            used = prior_seconds + (time.time() - t_proc0)
            if used + proj > cap_hours * 3600:
                print(f"[CAP] not starting {pol.name}: used {used/3600:.2f} h + projected {proj/3600:.2f} h "
                      f"> cap {cap_hours:.2f} h. Stopping.", flush=True)
                return False
        man = {"block": block, "split": split, "indices": indices, "policy": pol.describe(),
               "weight_bytes_per_pass": wb, **prov, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        (out_dir / block / f"{pol.name}.manifest.json").write_text(json.dumps(man, indent=1))
        with open(path, "a", encoding="utf-8") as f:
            for idx in todo:
                prob = ds[idx]
                prompt = prefix + f"Question: {prob['question']}\nAnswer:"           # historical template
                ids_t = tok(prompt, return_tensors="pt").input_ids.to(backend.device)  # historical tokenization
                if cuda:
                    torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                if pol.kind == "histimpl":
                    dyn.entropy_threshold = pol.tau
                    _, _, hroutes = dyn.generate(prompt, tok, max_new)
                    ids, routes, extra = dyn.last_generated_ids, "".join("F" if r == "fp16" else "N" for r in hroutes), {}
                else:
                    res = generate(backend, ids_t, pol, max_new, tok.eos_token_id, idx)
                    ids, routes = res.ids, "".join(res.routes)
                wall = time.perf_counter() - t0
                b = find_boundary(tok, ids, tok.eos_token_id, max_new)
                text = tok.decode(ids, skip_special_tokens=True)
                rec = {"index": idx, "split": split, "policy": pol.name, "kind": pol.kind,
                       "deployable": pol.deployable, "prompt_tokens": int(ids_t.shape[1]),
                       **score(prob, text, b),
                       "boundary": {k: v for k, v in b.to_dict().items() if not k.endswith("_mask")},
                       "answer_token_mask": b.answer_token_mask, "decision_token_mask": b.decision_token_mask,
                       "ids": ids, "routes": routes, "text": text, "wall_seconds": wall,
                       "peak_mem_allocated": torch.cuda.max_memory_allocated() if cuda else None,
                       "peak_mem_reserved": torch.cuda.max_memory_reserved() if cuda else None}
                if pol.kind != "histimpl":
                    rec.update({"accounting": account(res, b.answer_token_mask, wb, b.decision_token_mask,
                                                      prompt_len=int(ids_t.shape[1])),
                                "passes": res.passes, "trace": res.trace})
                f.write(json.dumps(rec) + "\n")
                f.flush()
                share = routes.count("F") / max(1, len(routes))
                print(f"[{block}] {pol.name} idx={idx} {rec['status']} steps={len(routes)} fp16={share:.3f} "
                      f"t={wall:.1f}s", flush=True)
    return True


# ----------------------------------------------------------------------------- calibration (CPU)
def tie_rule_threshold(values, target, direction):
    """Fixed, auditable rule (PILOT_SPEC.md 4.3). Candidates are the distinct observed values of the
    decision signal (exact decision dtype) plus +-inf. For direction '>' the policy routes FP16 iff
    signal > tau; for '<' iff signal < tau. Choose the candidate whose calibration-set FP16 share is
    closest to target; exact ties go to the LOWER FP16 share, then to the smaller tau."""
    v = sorted(values)
    n = len(v)
    cands = sorted(set(v)) + [math.inf, -math.inf]
    best = None
    for t in cands:
        share = (n - bisect.bisect_right(v, t)) / n if direction == ">" else bisect.bisect_left(v, t) / n
        key = (abs(share - target), share, t)
        if best is None or key < best[0]:
            best = (key, t, share)
    return best[1], best[2]


def calibrate():
    def steps(arm, key):
        vals = []
        for line in (OUT / "calib" / f"{arm}.jsonl").read_text().splitlines():
            vals += [x for x in json.loads(line)["trace"][key] if x is not None]
        return vals
    th = {"_rule": "open-loop; exact decision-dtype signals from calib block; tie rule: closest share, then lower share, then smaller tau",
          "_sources": {"lag_*": "calib/nf4 trajectory", "cascade": "calib/nf4 M_nf4", "oracle_*": "calib/oracle_flip trajectory"}}
    for b, B in BUDGETS.items():
        for name, arm, key, d in ((f"lag_entropy_{b}", "nf4", "Hd_lag1", ">"), (f"lag_margin_{b}", "nf4", "Md_lag1", "<"),
                                  (f"cascade_nf4_margin_{b}", "nf4", "Md_nf4", "<")):
            vals = steps(arm, key)
            tau, share = tie_rule_threshold(vals, B, d)
            th[name] = {"tau": tau, "target": B, "calib_share_on_source": share, "n_values": len(vals),
                        "n_distinct": len(set(vals)), "source": f"calib/{arm}:{key}", "direction": d}
    for name, key, d in (("oracle_js_B10", "js", ">"), ("oracle_fp16_entropy_B10", "Hd_fp16", ">")):
        vals = steps("oracle_flip", key)
        tau, share = tie_rule_threshold(vals, BUDGETS["B10"], d)
        th[name] = {"tau": tau, "target": BUDGETS["B10"], "calib_share_on_source": share, "n_values": len(vals),
                    "n_distinct": len(set(vals)), "source": f"calib/oracle_flip:{key}", "direction": d}
    THRESH_FILE.write_text(json.dumps(th, indent=1))
    print(json.dumps(th, indent=1))


# ----------------------------------------------------------------------------- matched random controls (CPU)
def _load(block, arm):
    p = OUT / block / f"{arm}.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else None


def _share(recs, which):
    num = sum(r["accounting"][f"fp16_steps_{which}"] for r in recs)
    den = sum(r["accounting"][f"routed_steps_{which}"] for r in recs)
    return num / den


def _cost_matched_p(recs, wb):
    """p such that a random policy's expected total cost equals the cascade's realized total cost:
    random cost = sum_s [W_N + p (W_F - W_N) + kv_single_pass(L_s)]  (W_* incl. unrouted weights)."""
    kvb = wb["kv_bytes_per_token"]
    wN, wF = wb["nf4_routed"] + wb["unrouted"], wb["fp16_routed"] + wb["unrouted"]
    casc = base = steps = 0
    for r in recs:
        a = r["accounting"]
        casc += a["cost_total_bytes"]
        n, P = a["routed_steps_full"], r["prompt_tokens"]
        steps += n
        base += sum(wN + (3 * (P + s - 1) + 2) * kvb for s in range(1, n + 1))
    return (casc - base) / (steps * (wF - wN))


def derive_random(group):
    """Pre-declared matching rules (PILOT_SPEC.md 4.4). Writes/extends pilot_random_plan.json."""
    plan = json.loads(RANDOM_PLAN.read_text()) if RANDOM_PLAN.exists() else []
    have = {r["name"] for r in plan}
    wb = json.loads((OUT / "eval" / "fp16.manifest.json").read_text())["weight_bytes_per_pass"]
    new = []

    def add(name, p, matched_to, unit, note=""):
        if name not in have:
            new.append({"name": name, "p_fp16": p, "matched_to": list(matched_to), "unit": unit, "group": group, "note": note})

    def pair(b, which, tag):
        """Control matched to lag_entropy_<b>; lag_margin_<b> shares it only if its realized share is
        within SHARE_TOL, otherwise it gets its own control."""
        se = _share(_load("eval", f"lag_entropy_{b}"), which)
        sm = _share(_load("eval", f"lag_margin_{b}"), which)
        if abs(sm - se) <= SHARE_TOL:
            add(f"rand_{tag}_lagE_{b}_s1", (se + sm) / 2, [f"lag_entropy_{b}", f"lag_margin_{b}"], which,
                f"shared: |share diff| = {abs(sm - se):.4f} <= {SHARE_TOL}; p = mean of the two")
        else:
            add(f"rand_{tag}_lagE_{b}_s1", se, [f"lag_entropy_{b}"], which)
            add(f"rand_{tag}_lagM_{b}_s1", sm, [f"lag_margin_{b}"], which)

    if group in ("B10", "B30"):
        pair(group, "full", "full")
        c = _load("eval", f"cascade_nf4_margin_{group}")
        add(f"rand_cost_casc_{group}_s1", _cost_matched_p(c, wb), [f"cascade_nf4_margin_{group}"], "total_cost_bytes")
        if group == "B10":
            add("rand_full_flip_s1", _share(_load("eval", "oracle_flip"), "full"), ["oracle_flip"], "full")
    elif group == "answer":
        pair("B10", "answer", "ans")
    elif group == "seeds":
        base = next(r for r in plan if r["name"] == "rand_full_lagE_B10_s1")
        for k in (2, 3):
            add(f"rand_full_lagE_B10_s{k}", base["p_fp16"], base["matched_to"], "full", "extra seed")
    for r in new:
        # Pre-declared rule: a cascade whose realized total cost exceeds that of static FP16 (p > 1) has
        # no cost-matched random control; it is recorded as infeasible and static FP16 (cheaper, and
        # FP16 on every step) is the cost-dominating comparator.
        if r["p_fp16"] > 1.0:
            r.update({"infeasible": True, "note": "cascade total cost exceeds static FP16; no cost-matched random "
                                                   "control exists; compare against static fp16"})
        assert r["p_fp16"] >= 0.0, r
    plan += new
    RANDOM_PLAN.write_text(json.dumps(plan, indent=1))
    print(json.dumps(new, indent=1))
    return [r["name"] for r in new]


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "run", "calibrate", "derive-random", "ledger", "cpu-smoke"])
    ap.add_argument("--block", choices=["gate0", "calib", "eval", "eval_random", "equiv"])
    ap.add_argument("--arms", nargs="*", help="subset of arm names, run in the order given")
    ap.add_argument("--indices", nargs="*", type=int, help="restrict to these problem indices (investigation runs)")
    ap.add_argument("--group", choices=["B10", "B30", "answer", "seeds"])
    ap.add_argument("--cap-hours", type=float, default=5.0)
    ap.add_argument("--authorized-gpu-run", action="store_true",
                    help="required for any GPU run; only pass after explicit author authorization")
    a = ap.parse_args()

    if a.cmd == "plan":
        s = make_sample()
        print({k: (len(v) if isinstance(v, list) else v) for k, v in s.items()})
        for blk in ("gate0", "equiv", "calib"):
            print(blk, [p.name for p in arms_for(blk)])
        print("eval priority:", EVAL_PRIORITY)
        return
    if a.cmd == "calibrate":
        return calibrate()
    if a.cmd == "derive-random":
        return derive_random(a.group)
    if a.cmd == "ledger":
        print(f"cumulative GPU wall time: {ledger_seconds()/3600:.3f} h")
        return
    if a.cmd == "cpu-smoke":
        from rv2.backends import build_cpu_backend, real_model_cpu, weight_bytes_per_pass
        tok = AutoTokenizer.from_pretrained(MODEL)
        be = build_cpu_backend(real_model_cpu(MODEL))
        s = make_sample()
        th = {k: {"tau": v} for k, v in {"lag_entropy_B10": 3.0, "lag_margin_B10": 0.2, "cascade_nf4_margin_B10": 0.2,
                                         "lag_entropy_B30": 2.0, "lag_margin_B30": 0.4, "cascade_nf4_margin_B30": 0.4,
                                         "oracle_js_B10": 0.01, "oracle_fp16_entropy_B10": 2.0}.items()}
        arms = [p for p in arms_for("eval", th) if p.name in ("fp16", "lag_entropy_B10", "cascade_nf4_margin_B10",
                                                              "oracle_js_B10", "oracle_flip")]
        arms.append(RandomBernoulli(0.1, "smoke_rand", "random_smoke"))
        run_block("smoke", be, tok, arms, s["train_eval"][:2], "train", max_new=12,
                  out_dir=HERE / "validation_logs" / "cpu_smoke", wb=weight_bytes_per_pass(be.model))
        return

    # ---- run (GPU)
    if not a.authorized_gpu_run:
        sys.exit("Refusing to run: GPU pilot runs need explicit author authorization (--authorized-gpu-run).")
    if not torch.cuda.is_available():
        sys.exit("CUDA not available.")
    prior = ledger_seconds()
    if prior >= a.cap_hours * 3600:
        sys.exit(f"GPU cap reached ({prior/3600:.2f} h).")
    t_proc0 = time.time()
    s = make_sample()
    th = json.loads(THRESH_FILE.read_text()) if a.block == "eval" else {}
    plan = json.loads(RANDOM_PLAN.read_text()) if a.block == "eval_random" else None
    arms = arms_for(a.block, th, plan)
    if a.arms:
        by = {p.name: p for p in arms}
        missing = [n for n in a.arms if n not in by]
        if missing:
            sys.exit(f"unknown arms for block {a.block}: {missing}")
        arms = [by[n] for n in a.arms]
    split, idx = block_indices(a.block, s)
    if a.indices:
        assert set(a.indices) <= set(idx), "investigation indices must belong to the block"
        idx = a.indices
    from rv2.backends import load_gpu_backend, weight_bytes_per_pass
    ok = False
    try:
        backend, dyn = load_gpu_backend(MODEL)
        tok = AutoTokenizer.from_pretrained(MODEL)
        ok = run_block(a.block, backend, tok, arms, idx, split, wb=weight_bytes_per_pass(backend.model), dyn=dyn,
                       cap_hours=a.cap_hours, t_proc0=t_proc0, prior_seconds=prior)
    finally:
        OUT.mkdir(parents=True, exist_ok=True)
        with open(LEDGER, "a") as f:
            f.write(json.dumps({"start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_proc0)),
                                "seconds": time.time() - t_proc0, "block": a.block,
                                "arms": [p.name for p in arms], "indices": a.indices, "completed": ok}) + "\n")
        print(f"[ledger] cumulative GPU wall time: {ledger_seconds()/3600:.3f} h of {a.cap_hours} h")


if __name__ == "__main__":
    main()
