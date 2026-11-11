"""Mutation check for v3 (PROTOCOL_V3.md section 10): every injected fault must make at least one test fail.
Core mutants run against tests/run_v3_tests.py (tiny Qwen2 + tiny GQA Llama); analysis mutants against tests/test_analyze_v3.py --quick plus the
calibration-rule test v14. The E1 and E1L mutation checks (rv2.controls, used here for C1/C2) are re-run unchanged by the CPU validation tool.
  venv/Scripts/python.exe research_v2/tests/mutation_check_v3.py [--write]
"""

from __future__ import annotations

import json
import time
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RV = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(RV))
if "--quick" not in sys.argv:
    sys.argv.append("--quick")              # analysis tests in quick mode

import run_v3_tests as T                    # noqa: E402
import test_analyze_v3 as TA                # noqa: E402
import test_v3_repair as TRP                # noqa: E402  (amendment 1)
import test_v3_resume as TRS                # noqa: E402  (amendment 1)
import v3_repair_torn_line as RP            # noqa: E402

import numpy as np                          # noqa: E402
import torch                                # noqa: E402

import analyze_v3 as A                      # noqa: E402
import rv2.controls as C                    # noqa: E402
import rv3.core as V                        # noqa: E402

RR = T.R
SNAP = {"V": dict(vars(V)), "C": dict(vars(C)), "A": dict(vars(A)), "PS": dict(vars(V.PredictorState)), "CA": dict(vars(V.CalibAccumulator)),
        "R": dict(vars(RR)), "RP": dict(vars(RP))}


def restore():
    for name, mod in (("V", V), ("C", C), ("A", A), ("R", RR), ("RP", RP)):
        for k, v in SNAP[name].items():
            if not k.startswith("__"):
                setattr(mod, k, v)
    for k in ("evaluate", "record_write"):
        setattr(V.PredictorState, k, SNAP["PS"][k])
    for k in ("on_queries", "on_measure"):
        setattr(V.CalibAccumulator, k, SNAP["CA"][k])


O = SNAP["V"]


def wrap_arm(swap):
    orig = O["stored_entry_v3"]
    return lambda arm, *a: orig(swap.get(arm, arm), *a)


# ----------------------------------------------------------------------------- core mutants
def m_qa_qo_swapped():
    V.stored_entry_v3 = wrap_arm({"QA": "QO", "QO": "QA"})


def m_projector_wrong_head():
    V._project = lambda U, g: O["_project"](U.roll(1, dims=1), g)


def m_projector_wrong_layer():
    V._project = lambda U, g: O["_project"](U.roll(1, dims=0), g)


def m_r_off_by_one():
    V.R_SUBSPACE = 9
    V.subspace_from_M = lambda M, r=9: O["subspace_from_M"](M, 9)


def m_natural_kv_swapped():
    V.NATURAL = {**O["NATURAL"], "NAT-K": "V", "NAT-V": "K"}


def m_star_kv_swapped():
    V.stored_entry_v3 = wrap_arm({"NAT-K*": "NAT-V*", "NAT-V*": "NAT-K*"})


def m_c1_kv_swapped():
    V.stored_entry_v3 = wrap_arm({"C1-K": "C1-V", "C1-V": "C1-K"})


def m_loc_inh_swapped():
    V.routing_map = lambda model, mode: O["routing_map"](model, {"LOC": "INH", "INH": "LOC"}[mode])


def m_override_not_reset():
    V.clear_overrides = lambda model: None


def m_norm_pooled_over_heads():
    V._slot_norm = lambda x: x.float().norm(dim=(-2, -1), keepdim=True).expand(*x.shape[:-1], 1)


def m_rho_star_absolute():
    orig = O["stored_entry_v3"]

    def f(arm, alpha, kF, kM, gen, vgen, ctx):
        out, info = orig(arm, alpha, kF, kM, gen, vgen, ctx)
        if arm in ("NAT-K*", "NAT-V*") and alpha != 0:
            F = kF.float()
            D = kM.float() - F
            rows = slice(0, None, 2) if arm == "NAT-K*" else slice(1, None, 2)
            out = out.clone()
            out[rows] = (F[rows] + alpha * float(ctx["rho_star"]) * V._unit_safe(D[rows])).to(kF.dtype)
        return out, info
    V.stored_entry_v3 = f


def _evaluate_variant(shift_hi=0, shift_den=0, unperturbed=False):
    orig = SNAP["PS"]["evaluate"]

    def ev(self, cache, queries):
        if self.n == 0:
            return orig(self, cache, queries)
        if unperturbed:
            import math
            n = self.n
            lo, hi = self.P0, self.P0 + n
            pl = pv = pa = 0.0
            for l, lay in enumerate(cache.layers):
                q = queries[l].double()
                K = lay.keys[0].double().clone()
                K[:, lo:hi] -= self.dK[l, :, :n]
                p = V._attn_probs(q, K, self.hmap, self.scaling)[:, lo:hi]
                dk = self.dK[l, :, :n][self.hmap]
                dv = self.dV[l, :, :n][self.hmap]
                pl += float((p * torch.einsum("gd,gjd->gj", q, dk).abs() / math.sqrt(self.hd)).sum())
                pa += float((p * dk.norm(dim=-1)).sum())
                pv += float(torch.einsum("gj,gjd->gd", p, dv).norm(dim=-1).sum())
            r = orig(self, cache, queries)
            den = self.n_layers * self.n_q
            r.update(P_logit=pl / den, P_val=pv / den, P_anormK=pa / den)
            return r
        saved = self.n
        self.n = max(1, saved + shift_hi)
        try:
            r = orig(self, cache, queries)
        finally:
            self.n = saved
        if shift_den:
            r["P_norm"] = float((self.cum2 / (saved + shift_den)).sqrt().mean())
        return r
    return ev


def m_predictor_includes_current_entry():
    V.PredictorState.evaluate = _evaluate_variant(shift_hi=+1)


def m_predictor_drops_last_written():
    V.PredictorState.evaluate = _evaluate_variant(shift_hi=-1)


def m_predictor_norm_denominator_off_by_one():
    V.PredictorState.evaluate = _evaluate_variant(shift_den=+1)


def m_pre_rope_queries():
    V.post_rope_query = lambda cap, i: cap.qraw[i].view(cap.n_q, cap.head_dim).double()


def m_attention_over_unperturbed_keys():
    V.PredictorState.evaluate = _evaluate_variant(unperturbed=True)


def m_gqa_group_map_shifted():
    V.group_map = lambda n_q, n_kv, device: (O["group_map"](n_q, n_kv, device) + 1) % n_kv


def m_past_entries_perturbed():
    orig = SNAP["C"]["_write_back"]

    def f(cache, L, stacked):
        orig(cache, L, stacked)
        if L > 0:
            for lay in cache.layers:
                lay.keys[:, :, L - 1:L, :] += 2 ** -6
    C._write_back = f


def m_seed_depends_on_data():
    orig = O["teacher_forced_arm_v3"]

    def f(backend, prompt_ids, ref_ids, S, arm, alpha, seeds, *a, **k):
        return orig(backend, prompt_ids, ref_ids, S, arm, alpha, (seeds[0] + int(prompt_ids.sum()), seeds[1]), *a, **k)
    V.teacher_forced_arm_v3 = f


def m_cv_profile_ignored():
    orig = O["stored_entry_v3"]

    def f(arm, alpha, kF, kM, gen, vgen, ctx):
        if arm == "CV":
            ctx = {**ctx, "sigma2": torch.ones_like(ctx["sigma2"])}
        return orig(arm, alpha, kF, kM, gen, vgen, ctx)
    V.stored_entry_v3 = f


def m_qaqo_value_stream_not_shared():
    V.arm_seeds = lambda model_label, arm, alpha, index, tier: (V.seed_for(model_label, arm, alpha, index, tier), V.seed_for(model_label, f"{arm}-V", alpha, index, tier))


def m_natural_not_bitwise():
    orig = O["stored_entry_v3"]

    def f(arm, alpha, kF, kM, gen, vgen, ctx):
        out, info = orig(arm, alpha, kF, kM, gen, vgen, ctx)
        if arm == "NAT" and alpha == 1.0:
            out = (out.float() * (1 + 2 ** -9)).to(out.dtype)
        return out, info
    V.stored_entry_v3 = f


def m_accumulator_pools_all_heads():
    def on_queries(self, queries, n_kv):
        Q = torch.stack([q.double() for q in queries])
        M = torch.einsum("lgd,lge->lde", Q, Q).unsqueeze(1).expand(Q.shape[0], n_kv, Q.shape[2], Q.shape[2]).clone()
        self.M = M if self.M is None else self.M + M
        self.n_q += 1
    V.CalibAccumulator.on_queries = on_queries


def m_accumulator_rho_kv_swapped():
    orig = SNAP["CA"]["on_measure"]

    def on_measure(self, kF, kM):
        orig(self, kF, kM)
        self.rhoK[-1], self.rhoV[-1] = self.rhoV[-1], self.rhoK[-1]
    V.CalibAccumulator.on_measure = on_measure


def m_nat_measured_in_fp16():
    orig = O["measure_pass"]
    V.measure_pass = lambda backend, cur, cache, mode: backend.run(cur, cache, False) if mode == "NF4" else orig(backend, cur, cache, mode)


def m_one_path_arm_touches_other_path():
    orig = O["stored_entry_v3"]

    def f(arm, alpha, kF, kM, gen, vgen, ctx):
        out, info = orig(arm, alpha, kF, kM, gen, vgen, ctx)
        if arm in ("C1-K", "NAT-K*") and alpha != 0:
            out = out.clone()
            out[1::2] = (out[1::2].float() * (1 + 2 ** -8)).to(out.dtype)
        return out, info
    V.stored_entry_v3 = f


# ----------------------------------------------------------------------------- amendment 2 mutants (split check)
def m_prerounding_norm_scaled_1p001():
    """(i) the construction adds 1.001 x the intended perturbation (target unchanged)."""
    orig = O["stored_entry_v3"]

    def f(arm, alpha, kF, kM, gen, vgen, ctx):
        out, info = orig(arm, alpha, kF, kM, gen, vgen, ctx)
        if info["target"] is not None and alpha != 0:
            F = kF.double()
            info["pre"] = F + 1.001 * (info["pre"] - F)
            out = O["to_fp16"](info["pre"])
        return out, info
    V.stored_entry_v3 = f


def m_slot_norm_scaled_1p001():
    """(i) the per-slot norm used by the construction (and its own target) is 1.001 x the true norm."""
    V._slot_norm = lambda x: 1.001 * O["_slot_norm"](x)


def _mag_bits(u):
    a = u.double().abs()
    return a, O["to_fp16"](a).view(torch.int16)


def _signed(bits, u):
    h = bits.view(torch.float16)
    return torch.where(torch.signbit(u), -h, h)


def m_round_truncate():
    """(ii) round toward zero instead of to nearest."""
    def f(u):
        a, b = _mag_bits(u)
        return _signed(b - (b.view(torch.float16).double() > a).to(torch.int16), u)
    V.to_fp16 = f


def m_round_via_float32():
    """(ii) float64 -> float32 -> float16 (double rounding; torch's own double -> half)."""
    V.to_fp16 = lambda u: u.float().half()


def m_round_half_up():
    """(ii) exact ties rounded away from zero instead of to even."""
    def f(u):
        a, b = _mag_bits(u)
        h = b.view(torch.float16).double()
        nu = (b + 1).view(torch.float16).double()
        tie = (h < a) & ((a - h) == (nu - a))
        return _signed(b + tie.to(torch.int16), u)
    V.to_fp16 = f


def m_rounding_skipped():
    """(iii) the float64 pre-rounding entry is returned unrounded."""
    V.to_fp16 = lambda u: u


CORE = [m_qa_qo_swapped, m_projector_wrong_head, m_projector_wrong_layer, m_r_off_by_one, m_natural_kv_swapped, m_star_kv_swapped, m_c1_kv_swapped,
        m_loc_inh_swapped, m_override_not_reset, m_norm_pooled_over_heads, m_rho_star_absolute, m_predictor_includes_current_entry,
        m_predictor_drops_last_written, m_predictor_norm_denominator_off_by_one, m_pre_rope_queries, m_attention_over_unperturbed_keys,
        m_gqa_group_map_shifted, m_past_entries_perturbed, m_seed_depends_on_data, m_cv_profile_ignored, m_qaqo_value_stream_not_shared,
        m_natural_not_bitwise, m_accumulator_pools_all_heads, m_accumulator_rho_kv_swapped, m_nat_measured_in_fp16, m_one_path_arm_touches_other_path,
        m_prerounding_norm_scaled_1p001, m_slot_norm_scaled_1p001, m_round_truncate, m_round_via_float32, m_round_half_up, m_rounding_skipped]


# ----------------------------------------------------------------------------- analysis mutants
def a_holm_is_bonferroni():
    A.holm = lambda p: np.minimum(1.0, np.asarray(p, float) * len(p))


def a_spearman_ordinal_ranks():
    A.rankdata = lambda x: np.argsort(np.argsort(np.asarray(x, float), kind="mergesort")).astype(float) + 1


def a_hc_slope_instead_of_intercept():
    orig = SNAP["A"]["hc_betas"]

    def f(S):
        b0, b1 = orig(S)
        n, sy, sx = S[..., 0], S[..., 1], S[..., 2]
        return b0, (b0 - b1) / np.where(sx != 0, sx / n, 1.0)
    A.hc_betas = f


def a_band_thresholds_swapped():
    A.band = lambda f, ci: ("substantial" if f >= 0.25 and ci[0] > 0 else "partial" if ci[0] > 0 else "no" if ci[1] < 0.5 else "inconclusive")


def a_equivalence_ignores_dose_flag():
    orig = SNAP["A"]["verdict"]
    A.verdict = lambda s, margin=None, informative=True: orig(s, margin, True)


def a_family_contrast_sign_flipped():
    orig = SNAP["A"]["summarize"]

    def f(d, rng, margin=None, informative=True, B=None):
        import inspect
        caller = inspect.stack()[1].code_context or [""]
        if "(c + d) - (a + b)" in caller[0]:
            d = -np.asarray(d)
        return orig(d, rng, margin, informative, B)
    A.summarize = f


def a_sd_guard_inactive():
    A.SD_GUARD = 1e9


def a_hc_precondition_ignored():
    orig = SNAP["A"]["hc_cross"]

    def f(stA, stB, rng, B=None):
        r = orig(stA, stB, rng, B)
        r["precondition_met"] = True
        r.pop("finding", None)
        return r
    A.hc_cross = f


def a_scale_factors_swapped():
    orig = SNAP["A"]["calibrate_from"]

    def f(recs, aux):
        out = orig(recs, aux)
        out["scale_factor_K"], out["scale_factor_V"] = out["scale_factor_V"], out["scale_factor_K"]
        return out
    A.calibrate_from = f


def a_hd_rule_inverted():
    A.SCALE_RANGE = (0.9, 1.1)


def a_ha_uses_all_steps_incl_step1():
    orig = SNAP["A"]["ha_D"]

    def f(nat, idx, pred="P_logit", base="P_norm"):
        D, excl = [], 0
        for i in idx:
            r = nat[i]
            d = A.spearman(r[base], r["js"]) - A.spearman(r[pred], r["js"])
            if d == d:
                D.append((i, d))
        return D, excl
    A.ha_D = f


def a_sensitivity_cut_off_by_one():
    """amendment 2: keeps the step after the first flagged write (already reached by it)."""
    orig = SNAP["A"]["first_unaffected"]
    A.first_unaffected = lambda rec: min(orig(rec) + 1, len(rec["js"]))


ANALYSIS = [a_holm_is_bonferroni, a_spearman_ordinal_ranks, a_hc_slope_instead_of_intercept, a_band_thresholds_swapped, a_equivalence_ignores_dose_flag,
            a_family_contrast_sign_flipped, a_sd_guard_inactive, a_hc_precondition_ignored, a_scale_factors_swapped, a_hd_rule_inverted,
            a_ha_uses_all_steps_incl_step1, a_sensitivity_cut_off_by_one]


# ----------------------------------------------------------------------------- amendment 1 mutants
def r_tierA_filter_removed():
    orig_append = SNAP["R"]["append_records"]

    def tier_step(be, tok, data, m, idx, arms, ctx, cap, stored_e1=None):
        d = RR.mdir(m) / "tierA"
        recs, ref = RR.run_problem(be, tok, data, m, idx, "test", arms, "V3A", ctx, cap)
        RR.append_records(d, recs)
        RR.append_records(d, {"FP16_history": {"index": idx, "ids": ref}})
        return True
    RR.tier_step = tier_step


def r_torn_check_disabled():
    RR.torn_record_files = lambda m: []


def r_heartbeat_missing():
    from contextlib import contextmanager

    @contextmanager
    def lh(block, cap_hours=RR.CAP_HOURS):                  # same ledger line, but no heartbeat while the work runs
        RS1 = RR.RS1
        RS1.OUT, RS1.LEDGER, RS1.PROGRESS = RR.OUT, RR.OUT / "gpu_ledger.jsonl", RR.OUT / "current_process.json"
        RR.OUT.mkdir(parents=True, exist_ok=True)
        t0, st = time.time(), {"completed": False}
        try:
            yield st
        finally:
            with open(RS1.LEDGER, "a") as f:
                f.write(json.dumps({"seconds": time.time() - t0, "block": block, "completed": bool(st["completed"])}) + chr(10))
    RR.ledger_heartbeat = lh


def a_duplicate_assertion_removed():
    def load_jsonl(path):
        out = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    out[r["index"]] = r
        return out
    A.load_jsonl = load_jsonl


def rp_repair_removes_too_much():
    orig = SNAP["RP"]["plan_repair"]

    def plan(b):
        kept, removed = orig(b)
        cut = kept[:-1].rfind(bytes([10])) if kept else -1
        return (kept[:cut + 1], kept[cut + 1:] + removed) if cut >= 0 else (kept, removed)
    RP.plan_repair = plan


def resume_suite():
    TRS._REF.clear()
    return TRS.run_all()


def repair_suite():
    return TRP.run_all()


AMEND = [(r_tierA_filter_removed, resume_suite), (r_torn_check_disabled, resume_suite), (r_heartbeat_missing, resume_suite),
         (a_duplicate_assertion_removed, None), (rp_repair_removes_too_much, repair_suite)]


def analysis_suite():
    fails = TA.run_all()
    for fam in ("qwen",):
        try:
            T.v14_scale_factors_and_hd_rule(fam)
        except Exception as e:  # noqa: BLE001
            fails.append(("v14", str(e)))
    return fails


def main():
    base_core = T.run_suite()
    base_an = analysis_suite()
    base_rs, base_rp = resume_suite(), repair_suite()
    print(f"baseline failures: core {len(base_core)}, analysis {len(base_an)}, resume {len(base_rs)}, repair {len(base_rp)}")
    if base_core or base_an or base_rs or base_rp:
        print(base_core, base_an, base_rs, base_rp)
        sys.exit(2)
    res = []
    for mut, suite in ([(m, T.run_suite) for m in CORE] + [(m, analysis_suite) for m in ANALYSIS]
                       + [(m, s or analysis_suite) for m, s in AMEND]):
        restore()
        T._BE.clear()
        try:
            mut()
            fails = suite()
        except Exception as e:  # noqa: BLE001
            fails = [("mutant raised", f"{type(e).__name__}: {e}")]
        finally:
            restore()
        caught = bool(fails)
        res.append({"mutant": mut.__name__, "caught": caught, "first_failures": [list(f) for f in fails[:3]]})
        print(("CAUGHT " if caught else "MISSED ") + mut.__name__ + (f"  <- {fails[0][0]}" if fails else ""), flush=True)
    n_c = sum(r["caught"] for r in res)
    print(f"{n_c}/{len(res)} mutants caught")
    if "--write" in sys.argv:
        (RV / "validation_logs").mkdir(exist_ok=True)
        (RV / "validation_logs" / "v3_mutation_check.json").write_text(json.dumps({"caught": n_c, "total": len(res), "results": res}, indent=1))
    sys.exit(0 if n_c == len(res) else 1)


if __name__ == "__main__":
    main()
