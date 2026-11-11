"""Policy-pluggable dual-precision greedy generation with tracing and budget accounting.

Loop structure mirrors DynamicQwen.generate (src/dynamic_model.py) exactly for decide-before-step
policies: fixed-FP16 prefill, token 0 = argmax of the prefill's last logits (not routed), then for
s = 1 .. max_new_tokens-1 one routed step consuming token s-1, greedy argmax, stop after EOS.
tests/run_cpu_tests.py checks token-for-token identity with the historical functions.

Cache handling for policies that compute more than one branch of a step (PILOT_SPEC.md 1.4):
  cascade  NF4 pass appends NF4 KV at position L; on escalation crop(L) then FP16 pass appends FP16 KV.
  oracle   NF4 pass; snapshot the appended NF4 KV slice; crop(L); FP16 pass; decide; if NF4 is chosen,
           crop(L) and re-append the snapshot. The cache after every step therefore holds exactly the
           entries a single-pass router following the same route sequence would hold (tested bitwise).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from .policies import Policy, RandomBernoulli, StepView, float32_stats
from . import PROJECT_ROOT  # noqa: F401
from fragility import js_divergence
from routing import calculate_entropy, calculate_margin


class Backend:
    """Thin wrapper over a model whose DynamicLinear layers share one RoutingState."""

    def __init__(self, model, routing_state, device: str):
        self.model = model
        self.routing_state = routing_state
        self.device = device
        self.call_log: list[str] = []   # "P" prefill, "N" NF4 decode pass, "F" FP16 decode pass
        self.log_calls = False

    def sync(self):
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()

    @torch.no_grad()
    def run(self, input_ids, cache, use_4bit: bool):
        self.routing_state.use_4bit = bool(use_4bit)
        if self.log_calls:
            self.call_log.append("P" if cache is None else ("N" if use_4bit else "F"))
        out = self.model(input_ids, past_key_values=cache, use_cache=True)
        if cache is not None:
            assert out.past_key_values is cache, "cache object was replaced, crop/restore logic invalid"
        return out.logits[:, -1, :], out.past_key_values


def _snapshot_last(cache, L):
    return [(lay.keys[..., L:L + 1, :].clone(), lay.values[..., L:L + 1, :].clone()) for lay in cache.layers]


def _restore(cache, L, saved):
    cache.crop(L)
    for i, (k, v) in enumerate(saved):
        cache.update(k, v, i)


@dataclass
class GenResult:
    ids: list
    routes: list            # per routed step s>=1: "F" or "N" (precision whose output and KV were kept)
    passes: list            # per routed step: "N", "F", or "NF" (passes actually executed, in order)
    trace: dict             # per routed step logging-only statistics (lists aligned with routes)
    timing: dict
    counts: dict
    stopped_early: bool = False
    policy: dict = field(default_factory=dict)


@torch.no_grad()
def generate(backend: Backend, prompt_ids, policy: Policy, max_new_tokens: int, eos_token_id,
             problem_index: int, early_stop_fn=None, rng_override=None, instrument: bool = True) -> GenResult:
    """instrument=False ("lean" mode) executes exactly the same forward passes, cache operations and
    decisions, but without per-component cuda.synchronize timing and without the per-step logging
    statistics (trace lists stay empty; component timings stay 0; only timing['total'] is measured).
    Token and route identity with instrument=True is tested (t9)."""
    rng = None
    if isinstance(policy, RandomBernoulli):
        rng = rng_override if rng_override is not None else policy.make_rng(problem_index)

    def timed(fn, bucket):
        if not instrument:
            return fn()
        backend.sync()
        t0 = time.perf_counter()
        r = fn()
        backend.sync()
        timing[bucket] += time.perf_counter() - t0
        return r

    timing = {"prefill": 0.0, "nf4_pass": 0.0, "fp16_pass": 0.0, "cache_ops": 0.0, "decide": 0.0, "logging": 0.0}
    counts = {"nf4_passes": 0, "fp16_passes": 0, "recomputes": 0, "crops": 0, "restores": 0}
    # H_/M_ : float32 logging statistics.  Hd_/Md_ : the SAME signals computed exactly as the policies
    # compute them (historical functions on the native-dtype logits, FP16 on GPU) -- used for calibration
    # so that thresholds and ties refer to the values the decisions actually compare.
    keys = ["H_lag1", "M_lag1", "H_nf4", "M_nf4", "top_nf4", "H_fp16", "M_fp16", "top_fp16", "js", "t_step",
            "Hd_lag1", "Md_lag1", "Hd_nf4", "Md_nf4", "Hd_fp16", "Md_fp16"]
    trace = {k: [] for k in keys}
    routes, passes = [], []

    t_wall0 = time.perf_counter()
    logits, cache = timed(lambda: backend.run(prompt_ids, None, use_4bit=False), "prefill")
    ids = [int(torch.argmax(logits, dim=-1).item())]
    prev_logits = logits
    stopped_early = False

    if ids[0] != eos_token_id:
        for s in range(1, max_new_tokens):
            t_step0 = time.perf_counter()
            L = cache.get_seq_length()
            cur = torch.tensor([[ids[-1]]], device=prompt_ids.device)
            base = dict(prev_logits=prev_logits, input_token_id=ids[-1], step=s, rng=rng)
            nf4_logits = fp16_logits = None

            if policy.kind == "pre":
                want_fp16 = timed(lambda: policy.decide(StepView(policy.access, **base)), "decide")
                if want_fp16:
                    fp16_logits, _ = timed(lambda: backend.run(cur, cache, False), "fp16_pass")
                    counts["fp16_passes"] += 1
                    out, p = fp16_logits, "F"
                else:
                    nf4_logits, _ = timed(lambda: backend.run(cur, cache, True), "nf4_pass")
                    counts["nf4_passes"] += 1
                    out, p = nf4_logits, "N"

            elif policy.kind == "cascade":
                nf4_logits, _ = timed(lambda: backend.run(cur, cache, True), "nf4_pass")
                counts["nf4_passes"] += 1
                want_fp16 = timed(lambda: policy.decide(StepView(policy.access, nf4_logits=nf4_logits, **base)), "decide")
                if want_fp16:   # FP16 is computed only AFTER the decision
                    timed(lambda: cache.crop(L), "cache_ops")
                    counts["crops"] += 1
                    fp16_logits, _ = timed(lambda: backend.run(cur, cache, False), "fp16_pass")
                    counts["fp16_passes"] += 1
                    counts["recomputes"] += 1
                    out, p = fp16_logits, "NF"
                else:
                    out, p = nf4_logits, "N"

            else:  # oracle (non-deployable diagnostic reference)
                nf4_logits, _ = timed(lambda: backend.run(cur, cache, True), "nf4_pass")
                counts["nf4_passes"] += 1
                saved = timed(lambda: _snapshot_last(cache, L), "cache_ops")
                timed(lambda: cache.crop(L), "cache_ops")
                counts["crops"] += 1
                fp16_logits, _ = timed(lambda: backend.run(cur, cache, False), "fp16_pass")
                counts["fp16_passes"] += 1
                counts["recomputes"] += 1
                want_fp16 = timed(lambda: policy.decide(StepView(policy.access, nf4_logits=nf4_logits,
                                                                 fp16_logits=fp16_logits, **base)), "decide")
                if want_fp16:
                    out = fp16_logits
                else:
                    timed(lambda: _restore(cache, L, saved), "cache_ops")
                    counts["restores"] += 1
                    out = nf4_logits
                p = "NF"

            assert cache.get_seq_length() == L + 1, "cache length invariant violated"
            routes.append("F" if want_fp16 else "N")
            passes.append(p)
            ids.append(int(torch.argmax(out, dim=-1).item()))

            # ---- logging only; computed after the decision, never fed back into any policy ----
            if instrument:
                backend.sync()
                t_log0 = time.perf_counter()
                h, m, _ = float32_stats(prev_logits)
                trace["H_lag1"].append(h); trace["M_lag1"].append(m)
                trace["Hd_lag1"].append(calculate_entropy(prev_logits).item())
                trace["Md_lag1"].append(calculate_margin(prev_logits).item())
                for tag, lg in (("nf4", nf4_logits), ("fp16", fp16_logits)):
                    if lg is None:
                        for k in ("H", "M", "top", "Hd", "Md"):
                            trace[f"{k}_{tag}"].append(None)
                    else:
                        h, m, t = float32_stats(lg)
                        trace[f"H_{tag}"].append(h); trace[f"M_{tag}"].append(m); trace[f"top_{tag}"].append(t)
                        trace[f"Hd_{tag}"].append(calculate_entropy(lg).item())
                        trace[f"Md_{tag}"].append(calculate_margin(lg).item())
                trace["js"].append(js_divergence(fp16_logits, nf4_logits).item()
                                   if (nf4_logits is not None and fp16_logits is not None) else None)
                backend.sync()
                timing["logging"] += time.perf_counter() - t_log0
                trace["t_step"].append(time.perf_counter() - t_step0)

            prev_logits = out
            if eos_token_id is not None and ids[-1] == eos_token_id:
                break
            if early_stop_fn is not None and early_stop_fn(ids):
                stopped_early = True
                break

    timing["total"] = time.perf_counter() - t_wall0
    return GenResult(ids, routes, passes, trace, timing, counts, stopped_early, policy.describe())


def account(res: GenResult, answer_token_mask: list, bytes_per_pass: dict | None = None,
            decision_token_mask: list | None = None, prompt_len: int | None = None) -> dict:
    """Budget accounting in comparable units (PILOT_SPEC.md section 3).

    Routed step s (s >= 1) produces token s; it is answer-relevant iff answer_token_mask[s] == 1.
    Token 0 comes from the fixed-FP16 prefill and is excluded from all budgets (as historically).
    """
    n = len(res.routes)
    assert len(answer_token_mask) == n + 1
    ans = [answer_token_mask[s + 1] == 1 for s in range(n)]
    fp16_full = sum(r == "F" for r in res.routes)
    fp16_ans = sum(r == "F" and a for r, a in zip(res.routes, ans))
    n_ans = sum(ans)
    nf4_passes = sum(p.count("N") for p in res.passes)
    fp16_passes = sum(p.count("F") for p in res.passes)
    assert nf4_passes == res.counts["nf4_passes"] and fp16_passes == res.counts["fp16_passes"]
    out = {
        "routed_steps_full": n,
        "fp16_steps_full": fp16_full,
        "nf4_steps_full": n - fp16_full,
        "fp16_share_full": fp16_full / n if n else None,
        "routed_steps_answer": n_ans,
        "fp16_steps_answer": fp16_ans,
        "nf4_steps_answer": n_ans - fp16_ans,
        "fp16_share_answer": fp16_ans / n_ans if n_ans else None,
        "nf4_passes": nf4_passes,
        "fp16_passes": fp16_passes,
        "recomputes": res.counts["recomputes"],
        "crops": res.counts["crops"],
        "restores": res.counts["restores"],
        "nf4_passes_answer": sum(p.count("N") for p, a in zip(res.passes, ans) if a),
        "fp16_passes_answer": sum(p.count("F") for p, a in zip(res.passes, ans) if a),
        **{f"time_{k}": v for k, v in res.timing.items()},
    }
    if decision_token_mask is not None:
        assert len(decision_token_mask) == n + 1
        dec = [decision_token_mask[s + 1] == 1 for s in range(n)]
        assert all(a or not d for a, d in zip(ans, dec)), "decision region must lie within answer region"
        n_dec = sum(dec)
        f_dec = sum(r == "F" and d for r, d in zip(res.routes, dec))
        out.update({"routed_steps_decision": n_dec, "fp16_steps_decision": f_dec,
                    "fp16_share_decision": f_dec / n_dec if n_dec else None})
    if bytes_per_pass:
        out.update(cost_breakdown(res, bytes_per_pass, prompt_len))
    return out


def cost_breakdown(res: GenResult, wb: dict, prompt_len: int | None) -> dict:
    """Total computational cost of the decode phase, in bytes moved (PILOT_SPEC.md section 3.1).

    Per executed decode pass at step s (cache length before the step L_s = prompt_len + s - 1):
      weights:  W_routed(precision) + W_unrouted
      KV:       (3 * L_s + 2) * kvb   -- attention reads L_s+1 entries; the cache append (torch.cat)
                                         reads L_s and writes L_s+1
    Per cache operation:
      crop      0 (a view)
      snapshot  2 * kvb                -- clone of one position (read + write)
      restore   (2 * L_s + 1) * kvb    -- re-append after crop (cat: read L_s, write L_s+1)
    The prefill is identical for every policy and is excluded (reported separately as time_prefill).
    Policy-signal computation (entropy/margin over the vocabulary, < 1 MB) is not charged.
    Cost is also expressed in FP16-pass equivalents: bytes / (W16_routed + W_unrouted).
    """
    assert prompt_len is not None, "prompt_len required for KV accounting"
    kvb = wb["kv_bytes_per_token"]
    w = {"N": wb["nf4_routed"] + wb["unrouted"], "F": wb["fp16_routed"] + wb["unrouted"]}
    kind = res.policy.get("kind")
    weight = kv = ops = 0
    weight_ans = 0
    for s, (p, r) in enumerate(zip(res.passes, res.routes), start=1):
        L = prompt_len + s - 1
        for c in p:
            weight += w[c]
            kv += (3 * L + 2) * kvb
        if kind == "oracle":
            ops += 2 * kvb                       # snapshot every step
            if r == "N":
                ops += (2 * L + 1) * kvb         # restore NF4 entries
    total = weight + kv + ops
    fpe = w["F"]
    return {"cost_weight_bytes": weight, "cost_kv_bytes": kv, "cost_cacheop_bytes": ops,
            "cost_total_bytes": total, "cost_fpe": total / fpe,
            "cost_fpe_per_step": total / fpe / max(1, len(res.routes))}
