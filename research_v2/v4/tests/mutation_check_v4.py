"""Mutation check for v4 (PROTOCOL_V4.md section 12): every injected fault must make at least one of its named tests fail. Each mutant is applied
in-process to the module under test, the named tests of test_rv4 / test_ops_v4 / test_analyze_v4 run, and every module is restored before the
next mutant. Core mutants: the v3 core mutants restricted to the six arms, plus the v4 ones (64-sphere at head_dim 128, BF16, X13a sums).
Operations mutants: ledgers, ceiling, watchdog, pod start, secrets, G7, G8, exactly-once, authorization, remote filter. Analysis mutants: X13a,
H4 permutation unit, Holm family, equivalence, readings, flagged / unresolved handling, Step 0 criteria. r4 mutants: the double load, the SKU substring,
the deterministic settings, the GM environment gate, the image record, the lock amendment path. r5 mutants: the ceiling, the close-out reserve, prorated
storage, the sample, Mistral last and first to drop, no other automatic cut, H4 needing all seven, Step 0 like-with-like, the stream encoding, the close-out gate.
r6 mutants: raw-byte hashing of previous locks and appended files (lock_v4 and run_v4), the unquoted known_hosts path.

  venv/Scripts/python.exe research_v2/v4/tests/mutation_check_v4.py [--device cuda] [--write <json>] [--only <name>]
"""

from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
V4 = HERE.parent
RV = V4.parent
sys.path.insert(0, str(RV))
sys.path.insert(0, str(V4))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(RV / "tools"))
sys.path.insert(0, str(RV / "v3" / "phase8"))

import test_ops_v4 as TO  # noqa: E402  (imports run_v4 -> HF before torch)
import test_rv4 as TR  # noqa: E402
import test_analyze_v4 as TA  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import analyze_v4 as A  # noqa: E402
import analyze_x13 as X13  # noqa: E402
import lock_v4 as LK  # noqa: E402
import run_v4 as R  # noqa: E402
import rv2.controls as C  # noqa: E402
import rv3.core as V  # noqa: E402
import rv4  # noqa: E402
import v4_gpu_mutation_check as GMC  # noqa: E402
import v4_ops as O  # noqa: E402
import v4_remote as REM  # noqa: E402
import v4_secret_scan as SS  # noqa: E402
import v4_v3_records as V3R  # noqa: E402

MODS = {"V": V, "C": C, "rv4": rv4, "O": O, "R": R, "A": A, "X13": X13, "SS": SS, "REM": REM, "LK": LK, "GMC": GMC, "V3R": V3R, "TA": TA}
SNAP = {k: dict(vars(m)) for k, m in MODS.items()}
CLS = {(O.CostLedger, "estimate"), (O.CostLedger, "sessions"), (O.Ledgers, "tick"), (rv4.SlotSums, "record"), (rv4.RhoAccumulator, "on_measure"),
       (rv4.DynamicLinearBF16, "forward"), (R.ModelSlot, "drop"), (O.CostLedger, "storage_usd")}
CLS_SNAP = {(c, n): c.__dict__[n] for c, n in CLS}
MUTANTS = []


def restore():
    for k, m in MODS.items():
        for name, v in SNAP[k].items():
            if not name.startswith("__"):
                setattr(m, name, v)
    for (c, n), f in CLS_SNAP.items():
        setattr(c, n, f)
    TR._BE.clear()


def mutant(*tests):
    def deco(fn):
        MUTANTS.append((fn, tests))
        return fn
    return deco


def S_(name):
    return SNAP[[k for k, m in MODS.items() if name in SNAP[k]][0]][name]


def wrap_entry(fn):
    orig = SNAP["rv4"]["stored_entry_v4"]
    rv4.stored_entry_v4 = lambda arm, alpha, kF, kM, gen, vgen, dtype: fn(orig, arm, alpha, kF, kM, gen, vgen, dtype)


# ----------------------------------------------------------------------------- core mutants
@mutant("t04_natural_arms_store_measured_entry_on_own_path", "t01_fp16_equals_locked_v3_arm")
def m_natural_kv_swapped():
    wrap_entry(lambda o, arm, *a: o({"NAT-K": "NAT-V", "NAT-V": "NAT-K"}.get(arm, arm), *a))


@mutant("t05_constructed_arms_norm_path_and_full_sphere", "t01_fp16_equals_locked_v3_arm")
def m_c1_kv_swapped():
    wrap_entry(lambda o, arm, *a: o({"C1-K": "C1-V", "C1-V": "C1-K"}.get(arm, arm), *a))


@mutant("t05_constructed_arms_norm_path_and_full_sphere")
def m_one_path_arm_touches_other_path():
    wrap_entry(lambda o, arm, *a: o("C1" if arm in ("C1-K", "C1-V") else arm, *a))


@mutant("t04_natural_arms_store_measured_entry_on_own_path", "t18_integrity_flags_finite_and_step1")
def m_natural_not_bitwise():
    def f(o, arm, alpha, kF, kM, *a):
        out, info = o(arm, alpha, kF, kM, *a)
        if arm in rv4.NATURAL and alpha == 1.0:
            out = out.clone()
            out.view(torch.int16)[0, 0, 0] ^= 1                            # one ulp in the last bit
        return out, info
    wrap_entry(f)


@mutant("t05_constructed_arms_norm_path_and_full_sphere", "t06_slot_norm_per_kv_head")
def m_norm_pooled_over_heads():
    V._slot_norm = lambda x: x.double().norm(dim=(-2, -1), keepdim=True).expand(*x.shape[:-1], 1)


@mutant("t05_constructed_arms_norm_path_and_full_sphere")
def m_c1_drawn_on_64_sphere():
    def g(shape, gen, device):
        s64 = tuple(shape[:-1]) + (min(64, shape[-1]),)
        z = torch.randn(s64, generator=gen, device=device, dtype=torch.float32)
        return torch.cat([z, torch.zeros(tuple(shape[:-1]) + (shape[-1] - s64[-1],), device=device)], -1) if shape[-1] > 64 else z
    V._gauss = g


@mutant("t19_past_entries_never_revisited")
def m_past_entries_perturbed():
    orig = SNAP["C"]["_write_back"]

    def wb(cache, L, stacked):
        orig(cache, L, stacked)
        if L > 0:
            for lay in cache.layers:
                lay.keys[:, :, L - 1:L, :] += 1e-3
    C._write_back = wb


@mutant("t10_seeds_labels_only")
def m_seed_master_label_dropped():
    rv4.arm_seeds = lambda m, arm, alpha, idx, tier: (V.seed_for(m, arm, alpha, idx, tier), V.seed_for(m, f"{arm}-V", alpha, idx, tier))


@mutant("t11_measure_pass_all_4bit_and_reset")
def m_nat_measured_in_fp16():
    V.measure_pass = lambda backend, cur, cache, mode: backend.run(cur, cache, False)


@mutant("t11_measure_pass_all_4bit_and_reset", "t18_integrity_flags_finite_and_step1")
def m_override_not_reset():
    def mp(backend, cur, cache, mode):
        for _, m in V.dynamic_modules(backend.model):
            m.layer_force_4bit = True
        backend.run(cur, cache, True)
    V.measure_pass = mp


@mutant("t09_fp16_rounding_check_active", "t18_integrity_flags_finite_and_step1")
def m_prerounding_norm_scaled_1p001():
    def f(o, arm, alpha, kF, kM, gen, vgen, dtype):
        out, info = o(arm, alpha, kF, kM, gen, vgen, dtype)
        if info["target"] is not None and alpha != 0.0:
            u = kF.double() + 1.001 * (info["pre"] - kF.double())
            info = dict(info, pre=u)
            out = rv4.to_store(u, dtype)
        return out, info
    wrap_entry(f)


@mutant("t21_fp16_storage_correctly_rounded_on_double_rounding_cases")
def m_fp16_round_via_float32():
    V.to_fp16 = lambda u: u.float().half()


@mutant("t09_fp16_rounding_check_active", "t18_integrity_flags_finite_and_step1", "t21_fp16_storage_correctly_rounded_on_double_rounding_cases")
def m_fp16_round_truncate():
    def tr(u):
        a = u.double().abs().float().half()
        a = torch.where(a.double() > u.double().abs(), torch.nextafter(a.float(), torch.zeros_like(a.float())).half(), a)
        return torch.where(torch.signbit(u), -a, a)
    V.to_fp16 = tr


@mutant("t07_bf16_rounding_equals_independent_oracle")
def m_bf16_round_via_float32():
    rv4.to_bf16 = lambda u: u.float().to(torch.bfloat16)


@mutant("t07_bf16_rounding_equals_independent_oracle")
def m_bf16_oracle_ties_up():
    """The oracle with ties rounded away from zero instead of to even (exact integer arithmetic otherwise)."""
    def o(u):
        x = np.ascontiguousarray(u.detach().double().cpu().numpy())
        bits = x.view(np.uint64)
        sign = (bits >> np.uint64(63)).astype(np.int64)
        e_ = ((bits >> np.uint64(52)) & np.uint64(0x7FF)).astype(np.int64)
        m = (bits & np.uint64((1 << 52) - 1)).astype(np.int64)
        E = e_ - 1023
        sig = m | (1 << 52)
        normal = E >= -126
        shift = np.where(normal, 45, 45 + (-126 - E))
        sh = np.clip(shift, 1, 53)
        q = sig >> sh
        r = sig & ((np.int64(1) << sh) - 1)
        q = q + (r >= (np.int64(1) << (sh - 1))).astype(np.int64)
        q = np.where(shift >= 54, 0, q)
        carry = normal & (q == 256)
        q = np.where(carry, 128, q)
        En = np.where(carry, E + 1, E)
        pat = np.where(normal, np.where(En + 127 >= 255, 0x7F80, ((En + 127) << 7) | (q - 128)), q)
        pat = np.where(e_ == 0, 0, pat)
        out = (sign << 15) | pat
        return torch.from_numpy(np.where(out >= 0x8000, out - 0x10000, out).astype(np.int16)).view(torch.bfloat16)
    rv4.bf16_rne_oracle = o


@mutant("t08_bf16_mode_end_to_end")
def m_bf16_output_cast_fp16():
    f = CLS_SNAP[(rv4.DynamicLinearBF16, "forward")]
    rv4.DynamicLinearBF16.forward = lambda self, x: f(self, x).to(torch.float16)


@mutant("t12_x13a_sums_independent")
def m_x13_sums_of_stored_not_delta():
    def rec(self, stored, kF):
        d = stored.double()
        self.normsK.append(d[0::2].norm(dim=-1))
        self.normsV.append(d[1::2].norm(dim=-1))
        self.S1K = d[0::2].clone() if self.S1K is None else self.S1K + d[0::2]
        self.S1V = d[1::2].clone() if self.S1V is None else self.S1V + d[1::2]
    rv4.SlotSums.record = rec


@mutant("t13_rho_accumulator_independent")
def m_rho_kv_swapped():
    f = CLS_SNAP[(rv4.RhoAccumulator, "on_measure")]

    def om(self, kF, kM):
        f(self, kF, kM)
        self.rhoK[-1], self.rhoV[-1] = self.rhoV[-1], self.rhoK[-1]
    rv4.RhoAccumulator.on_measure = om


# ----------------------------------------------------------------------------- operations mutants
@mutant("o02_cost_estimate_rate_boot_autostop_storage")
def m_cost_rate_ignored():
    f = CLS_SNAP[(O.CostLedger, "sessions")]

    def s(self, now):
        return {b: dict(v, rate=0.59) for b, v in f(self, now).items()}
    O.CostLedger.sessions = s


@mutant("o02_cost_estimate_rate_boot_autostop_storage", "o03_sessions_counted_once_and_recovered")
def m_autostop_omitted():
    f = CLS_SNAP[(O.CostLedger, "estimate")]

    def e(self, now=None):
        saved = self.cfg["autostop_minutes"]
        self.cfg["autostop_minutes"] = 0
        try:
            return f(self, now)
        finally:
            self.cfg["autostop_minutes"] = saved
    O.CostLedger.estimate = e


@mutant("o04_billing_lines_take_the_larger", "o05_ceiling_check_before_problem_and_watchdog")
def m_billing_ignored():
    f = CLS_SNAP[(O.CostLedger, "estimate")]

    def e(self, now=None):
        r = f(self, now)
        return dict(r, used_usd=r["inpod_usd"])
    O.CostLedger.estimate = e


@mutant("o06_pod_start_from_pid1_not_uptime")
def m_pod_start_from_uptime():
    def ps(proc="/proc", now=time.time):
        from pathlib import Path as P_
        return now() - float(P_(proc, "uptime").read_text().split()[0])
    O.pod_start_epoch = ps


@mutant("o18_block_loop_ceiling_and_drift")
def m_ceiling_checked_after_problem():
    def bl(name, tag, ids, step_fn, done_fn, cost, expected_problem_s=None):
        for idx in ids:
            if done_fn(idx):
                continue
            if not step_fn(idx):
                return False
            ok, est = cost.can_start(90.0)
            if not ok:
                (R.mdir(tag) / "ceiling_stop.json").write_text("{}")
                return False
        return True
    R.block_loop = bl


@mutant("o05_ceiling_check_before_problem_and_watchdog")
def m_watchdog_does_not_close():
    def tick(self):
        if self.cost.at_watchdog_limit():
            self.on_expire()
    O.Ledgers.tick = tick


@mutant("o08_secret_scan_formats")
def m_secret_scan_misses_gzip():
    orig = SNAP["SS"]["scan_bytes"]

    def sb(b, where, literals, depth=0):
        if b[:2] == b"\x1f\x8b":
            return [(where, n, o) for n, o in O.find_secrets(b, literals)]
        return orig(b, where, literals, depth)
    SS.scan_bytes = sb


@mutant("o08_secret_scan_formats")
def m_secret_literal_ignored():
    O.find_secrets = lambda b, literals=(): SNAP["O"]["find_secrets"](b, ())


@mutant("o09_redaction_filter")
def m_redaction_off():
    O.redact = lambda text, literals=(): text


@mutant("o17_g7_snapshot_checks")
def m_g7_accepts_unpinned_revision():
    orig = SNAP["R"]["check_snapshot"]

    def cs(e, snap):
        r = orig(dict(e, revision=snap.name), snap)
        return r
    R.check_snapshot = cs


@mutant("o01_authorization_files")
def m_auth_substring():
    def ca(out, tag, lock_id):
        p = O.auth_path(out, tag)
        if lock_id is None or not p.exists():
            return False, "x"
        txt = p.read_text()
        return ("PROTOCOL_V4" in txt and lock_id in txt and tag in txt), "x"
    O.check_auth = ca


@mutant("o15_exactly_once_and_resume")
def m_exactly_once_off():
    R.append_new = lambda d, idx, recs: R.E.append_records(d, recs)


@mutant("o15_exactly_once_and_resume")
def m_torn_check_off():
    R.torn_record_files = lambda tag: []


@mutant("o16_g8_canonical_comparison")
def m_g8_ignores_values():
    R.canonical = lambda recs: sorted(recs)


@mutant("o14_order_rules")
def m_bf16_flag_ignored():
    R.precision = lambda m: "fp16"


@mutant("o12_remote_command_filter")
def m_remote_allows_everything():
    REM.command_allowed = lambda cmd: (True, "ok")


@mutant("o11_checkpoint_pack_and_verify")
def m_checkpoint_verify_always_true():
    O.verify_sha256_file = lambda p: True


@mutant("o14_order_rules")
def m_step0_not_required():
    R.step0_ok = lambda: True


# ----------------------------------------------------------------------------- r4 mutants
def keep_caller_reference_at_reload(k):
    """The pre-r4 fault, at one reload only: the previous model stays referenced while the k-th reload of a slot runs."""
    orig = CLS_SNAP[(R.ModelSlot, "drop")]

    def drop(self):
        if self.be is not None:
            self.reloads = getattr(self, "reloads", 0) + 1
            if self.reloads == k:
                self.kept = (self.be, self.tok)
        orig(self)
    R.ModelSlot.drop = drop


@mutant("o22_bf16_reload_does_not_double_load")
def m_double_load_bf16_site():
    keep_caller_reference_at_reload(1)


@mutant("t22_reload_releases_the_previous_model")
def m_double_load_real_backend():
    keep_caller_reference_at_reload(1)


@mutant("o22_bf16_reload_does_not_double_load")
def m_reload_without_garbage_collection():
    def drop(self):
        self.be = self.tok = None
    R.ModelSlot.drop = drop


@mutant("o23_sku_check_is_exact")
def m_sku_substring():
    O.sku_ok = lambda name: name is not None and O.SKU in name


@mutant("o24_deterministic_settings_applied_and_required")
def m_determinism_not_applied():
    rv4.apply_determinism = lambda: rv4.determinism_state()


@mutant("o24_deterministic_settings_applied_and_required")
def m_cublas_workspace_not_set():
    def ap():
        torch.use_deterministic_algorithms(True)
        return rv4.determinism_state()
    rv4.apply_determinism = ap


@mutant("o24_deterministic_settings_applied_and_required")
def m_determinism_not_required():
    rv4.determinism_ok = lambda state=None: True


@mutant("o25_gm_environment_logged_and_gated")
def m_gm_ignores_environment():
    orig = SNAP["GMC"]["verdict"]
    GMC.verdict = lambda r1, r2, t, m, env_complete, determinism_ok, image, v3_records_ok: orig(r1, r2, t, m, True, determinism_ok, image, v3_records_ok)


@mutant("o25_gm_environment_logged_and_gated")
def m_gm_ignores_determinism():
    orig = SNAP["GMC"]["verdict"]
    GMC.verdict = lambda r1, r2, t, m, env_complete, determinism_ok, image, v3_records_ok: orig(r1, r2, t, m, env_complete, True, image, v3_records_ok)


@mutant("o25_gm_environment_logged_and_gated")
def m_gm_environment_always_complete():
    O.gpu_environment_complete = lambda env: True


@mutant("o25_gm_environment_logged_and_gated")
def m_gm_ignores_image_record():
    orig = SNAP["GMC"]["verdict"]
    GMC.verdict = lambda r1, r2, t, m, env_complete, determinism_ok, image, v3_records_ok: orig(r1, r2, t, m, env_complete, determinism_ok, {"digest": "x"}, v3_records_ok)


@mutant("o26_image_digest_recorded_once")
def m_image_digest_overwritten():
    def rec(out, fetch=None, now=time.time):
        d = {"image": O.IMAGE_REPO, "tag": O.IMAGE_TAG, "digest": fetch(), "utc": O.utc(now())}
        (out / "image_digest.json").write_text(json.dumps(d))
        return d
    O.record_image_digest = rec


@mutant("o26_image_digest_recorded_once")
def m_image_record_any_tag():
    def rec(out):
        p = out / "image_digest.json"
        return json.loads(p.read_text()) if p.exists() else None
    O.image_record = rec


@mutant("o28_lock_amendment_preserves_and_links", "o30_lock_verify_catches_a_broken_amendment")
def m_amend_does_not_preserve_old_lock():
    LK.preserve = lambda path, old_bytes: None


@mutant("o29_lock_amendment_refused_after_data_or_on_a_dirty_tree")
def m_amend_allowed_after_data():
    LK.data_records = lambda: []


@mutant("o29_lock_amendment_refused_after_data_or_on_a_dirty_tree")
def m_amend_ignores_old_lock_mismatch():
    LK.committed_differences = lambda text, head: []


@mutant("o28_lock_amendment_preserves_and_links", "o30_lock_verify_catches_a_broken_amendment")
def m_verify_ignores_amendments():
    LK.verify_amendments = lambda lines: []


@mutant("o28_lock_amendment_preserves_and_links")
def m_verify_ignores_head():
    """Only the lock's own commit and a clean tree are checked: a later commit that changes a locked file passes."""
    orig = SNAP["LK"]["git"]
    LK.git = lambda *a: orig(*(("show", a[1].replace("HEAD:", f"{LK.LOCK.read_text(encoding='utf-8').splitlines()[2].split(': ')[1]}:")) if a[0] == "show" else a))


# ----------------------------------------------------------------------------- r5 mutants
@mutant("o05_ceiling_check_before_problem_and_watchdog", "o27_cloud_files_name_the_study_gpu_and_image")
def m_ceiling_still_150():
    O.CEILING_USD = 150.0


@mutant("o05_ceiling_check_before_problem_and_watchdog")
def m_no_closeout_reserve():
    O.CLOSEOUT_RESERVE_USD = 0.0


@mutant("o31_storage_prorated_and_stops_at_deletion")
def m_storage_ignores_deletion():
    O.CostLedger.storage_usd = lambda self, t0, t1: max(0.0, t1 - t0) / O.MONTH_S * self.cfg["volume_gb"] * O.STORAGE_USD_PER_GB_MONTH


@mutant("o31_storage_prorated_and_stops_at_deletion")
def m_storage_allowance_full_month():
    O.storage_allowance = lambda volume_gb, billed_h: volume_gb * O.STORAGE_USD_PER_GB_MONTH


@mutant("o32_sample_is_the_first_100_and_50")
def m_sample_keeps_200():
    R.N_TEST = 200


@mutant("o32_sample_is_the_first_100_and_50")
def m_sample_calibration_100():
    R.N_CAL = 100


@mutant("o32_sample_is_the_first_100_and_50")
def m_g6_problems_are_calibration_problems():
    R.sample_ids = lambda q: (q["tierA_ids"][:R.N_TEST], q["calibration_train_ids"][:R.N_CAL], q["calibration_train_ids"][:R.N_CAL][-R.N_G6:])


@mutant("o34_mistral_last_and_first_to_drop")
def m_mistral_not_last():
    R.ORDER = R.STEP0 + [R.DROP_FIRST] + [m for m in R.LADDER_NEW if m != R.DROP_FIRST]


@mutant("o34_mistral_last_and_first_to_drop")
def m_mistral_start_not_checked():
    R.drop_first_check = lambda cmd, est: None


@mutant("o19_g6_fit_and_reduction_order")
def m_design_never_drops_mistral():
    R.design_steps = lambda: [None]


@mutant("o19_g6_fit_and_reduction_order")
def m_design_cuts_another_model():
    """An automatic cut beyond Mistral: H4 would lose a model without anyone deciding it."""
    R.design_steps = lambda: [None, R.DROP_FIRST, "llama3.1-8b"]


@mutant("o19_g6_fit_and_reduction_order")
def m_project_ignores_g8():
    R.N_G8 = 0


@mutant("a20_h4_needs_all_seven")
def m_h4_estimated_without_all_seven():
    orig = SNAP["A"]["h4"]
    A.h4 = lambda models, est, excluded=(): orig(models, est, [m for m in A.QWEN + A.LLAMA if m not in models])


@mutant("a16_step0_on_v3_records_themselves")
def m_step0_compares_with_all_100():
    A.v3_calibration_on = lambda label: json.loads((A.V3_RUNS / label / "calibration.json").read_text(encoding="utf-8"))


@mutant("o35_stream_survives_the_console_encoding")
def m_stream_keeps_strict_encoding():
    O.safe_stream = lambda stream: stream


@mutant("o36_closeout_report_gates_deletion")
def m_closeout_always_ready():
    O.closeout_report = lambda out, tags, drop_first=None: {"ready": True, "missing": []}


@mutant("o36_closeout_report_gates_deletion")
def m_closeout_ignores_final_state():
    O.STOP_FILES = O.STOP_FILES + ("SHA256SUMS", )
    orig = SNAP["O"]["closeout_report"]

    def cr(out, tags, drop_first=None):
        for t in tags:
            (out / t).mkdir(parents=True, exist_ok=True)
            (out / t / "SHA256SUMS").write_text("x")
        return orig(out, tags, drop_first)
    O.closeout_report = cr


# ----------------------------------------------------------------------------- analysis mutants
@mutant("a03_x13a_null_uses_model_head_dim", "a13_h5_isotropic_falsified")
def m_x13a_s1_unnormalized():
    """The OBSERVED share with ||S1||^2 instead of ||S1 / n||^2; the isotropic null keeps the correct share (scaling both would cancel in x)."""
    good, null = SNAP["X13"]["cm_share"], SNAP["X13"]["null_lognc"]
    X13.cm_share = lambda norms, S1: float((S1 ** 2).sum() / ((norms ** 2).sum(0) / norms.shape[0]).sum())

    def nl(norms, rng, draws=200, hd=64):
        X13.cm_share = good
        try:
            return null(norms, rng, draws, hd)
        finally:
            X13.cm_share = lambda n_, s_: float((s_ ** 2).sum() / ((n_ ** 2).sum(0) / n_.shape[0]).sum())
    X13.null_lognc = nl


@mutant("a03_x13a_null_uses_model_head_dim", "a13_h5_isotropic_falsified")
def m_x13a_null_always_64():
    orig = SNAP["X13"]["null_lognc"]
    X13.null_lognc = lambda norms, rng, draws=200, hd=64: orig(norms, rng, draws, 64)


@mutant("a01_h4_permutation_exact", "a10_h4_incomplete_separation_inconclusive")
def m_h4_permutation_over_problems():
    def perm(theta_hat, qwen, llama):
        rng = np.random.default_rng(0)
        rep = 25                                                       # pseudo-problems: each model's estimate repeated as if problems were the unit
        vals = np.array([theta_hat[m] for m in list(qwen) + list(llama)]).repeat(rep)
        lab = np.array([0] * (len(qwen) * rep) + [1] * (len(llama) * rep))
        t_obs = vals[lab == 1].mean() - vals[lab == 0].mean()
        ts = []
        for _ in range(2000):
            p = rng.permutation(lab)
            ts.append(vals[p == 1].mean() - vals[p == 0].mean())
        ts = np.array(ts)
        return {"T": float(t_obs), "n_labellings": len(ts), "p_one_sided": float(((ts >= t_obs).sum() + 1) / (len(ts) + 1)),
                "min_attainable_p": 1 / 2001, "complete_separation": bool(max(theta_hat[m] for m in qwen) < min(theta_hat[m] for m in llama))}
    A.h4_permutation = perm


@mutant("a05_truth_supported_everywhere")
def m_holm_over_wrong_family():
    def ft(models, est, key, sign):
        ps = [est[m]["contrasts"][k]["p"] for k in ("H1", "H2", "H3") for m in models]
        ph = A.L.holm(ps)[:len(models)] if ps else []
        rd = {m: A.reading_directional(est[m]["contrasts"][key], float(h), sign) for m, h in zip(models, ph)}
        return {"models": models, "holm_p": dict(zip(models, map(float, ph))), "readings": rd, "result": A.universal(rd)}
    A.family_tests = ft


@mutant("a08_equivalence_needs_informative_dose")
def m_equivalence_ignores_dose_flag():
    orig = SNAP["A"]["summarize"]
    A.summarize = lambda d, keys, margin=None, informative=True, B=None: orig(d, keys, margin, True, B)


@mutant("a02_readings_rules", "a07_equivalence_falsifies_h2")
def m_equivalence_not_falsifying():
    def rd(s, holm_p, sign):
        lo, hi = s["ci95"]
        if holm_p < 0.05 and ((hi < 0) if sign < 0 else (lo > 0)):
            return "supported"
        return "falsified" if ((lo > 0) if sign < 0 else (hi < 0)) else "inconclusive"
    A.reading_directional = rd


@mutant("a02_readings_rules")
def m_h5_support_at_ln2():
    A.reading_h5 = lambda s, holm_p: "supported" if (s["ci95"][0] > A.LN2 and holm_p < 0.05) else ("falsified" if s["ci95"][1] < A.LN2 else "inconclusive")


@mutant("a18_h4_sign_rules_unit")
def m_h4_llama_rule_dropped():
    orig = SNAP["A"]["h4"]

    def h4(models, est, excluded=()):
        r = orig(models, est, excluded)
        if r.get("result") == "falsified" and not any(est[m]["contrasts"]["NAT-C1"]["ci95"][0] > 0 or est[m]["contrasts"]["NAT-C1"]["verdict"] == "equivalent"
                                                      for m in r.get("qwen", [])):
            r["result"] = "supported" if r["permutation"]["p_one_sided"] <= 0.05 else "inconclusive"
        return r
    A.h4 = h4


@mutant("a11_flagged_with_and_without")
def m_flagged_not_excluded():
    orig = SNAP["A"]["hypotheses"]
    keep = {}

    def hy(models, est, x13s, excluded=()):
        if "all" not in keep:
            keep["all"] = list(models)
        return orig(keep["all"], est, x13s)
    A.hypotheses = hy


@mutant("a12_unresolved_model_and_h4_not_estimable")
def m_unresolved_included():
    orig = SNAP["A"]["model_status"]
    A.model_status = lambda tag: "stopped" if orig(tag) == "unresolved" else orig(tag)


@mutant("a16_step0_on_v3_records_themselves")
def m_step0_no_point_rule():
    A.STEP0_D_MAX = 1e9


@mutant("a16_step0_on_v3_records_themselves")
def m_step0_verdict_rule_for_small_theta():
    A.STEP0_VERDICT_MIN = 0.0


@mutant("a16_step0_on_v3_records_themselves")
def m_step0_calibration_tolerance_off():
    A.CAL_REL_TOL = 1e9


# ----------------------------------------------------------------------------- r6 mutants
@mutant("o38_lock_hashes_ignore_line_endings")
def m_lock_text_files_hashed_as_raw_bytes():
    """The pre-r6 behaviour: previous locks and appended files hashed as raw bytes, so a CRLF checkout and an LF checkout disagree."""
    LK.file_sha = lambda p: LK.sha(Path(p).read_bytes())


@mutant("o38_lock_hashes_ignore_line_endings")
def m_locked_line_raw_bytes():
    import hashlib
    R.locked_line = lambda prefix, path: path.exists() and any(l.startswith(prefix) and hashlib.sha256(path.read_bytes()).hexdigest() in l for l in R.lock_text().splitlines())


@mutant("o39_known_hosts_path_with_spaces_stays_one_file")
def m_known_hosts_unquoted():
    REM.known_hosts_option = lambda: f"UserKnownHostsFile={(REM.LOCAL_RUNS / 'known_hosts').as_posix()}"



@mutant("o25_gm_environment_logged_and_gated")
def m_gm_ignores_v3_records():
    orig = SNAP["GMC"]["verdict"]
    GMC.verdict = lambda r1, r2, t, m, env_complete, determinism_ok, image, v3_records_ok: orig(r1, r2, t, m, env_complete, determinism_ok, image, True)


@mutant("o40_v3_records_verified_against_manifest_and_v3_lock")
def m_v3_records_hash_not_checked():
    orig = SNAP["V3R"]["verify"]

    def verify(*a, **k):
        ok, probs = orig(*a, **k)
        probs = [x for x in probs if not x.startswith("hash differs")]
        return not probs, probs
    V3R.verify = verify


@mutant("o40_v3_records_verified_against_manifest_and_v3_lock")
def m_v3_records_ignore_v3_lock():
    orig = SNAP["V3R"]["verify"]

    def verify(*a, **k):
        ok, probs = orig(*a, **k)
        probs = [x for x in probs if "V3 lock" not in x]
        return not probs, probs
    V3R.verify = verify


@mutant("o40_v3_records_verified_against_manifest_and_v3_lock")
def m_v3_manifest_not_locked():
    orig = SNAP["LK"]["locked_files"]
    LK.locked_files = lambda: [p for p in orig() if p.name not in ("v3_records_manifest.json", "v4_v3_records.py")]


@mutant("o41_tests_fail_loudly_without_their_inputs")
def m_a16_returns_without_records():
    orig = SNAP["TA"]["a16_step0_on_v3_records_themselves"]

    def a16():
        if not (TA.RV / "v3_runs" / "qwen-base" / "tierA" / "NAT.jsonl").exists():
            return
        return orig()
    TA.a16_step0_on_v3_records_themselves = a16


@mutant("o41_tests_fail_loudly_without_their_inputs")
def m_a04_skips_missing_records():
    orig = SNAP["TA"]["a04_calibrate_core_reproduces_v3"]

    def a04():
        if not (TA.RV / "v3_runs" / "qwen-base" / "calib").exists():
            return
        return orig()
    TA.a04_calibrate_core_reproduces_v3 = a04



@mutant("o42_deviation_recorded_appended_and_verified")
def m_deviation_hashes_ignored():
    LK.deviation_hashes = lambda ids: {}


@mutant("o42_deviation_recorded_appended_and_verified")
def m_deviation_accepts_any_content():
    orig = SNAP["LK"]["deviation_hashes"]
    LK.deviation_hashes = lambda ids: {rel: (LK.sha(LK.git("show", f"HEAD:{rel}")), "HEAD", i) for rel, (h, c, i) in orig(ids).items()}


@mutant("o42_deviation_recorded_appended_and_verified")
def m_append_deviation_unchecked():
    LK.append_deviation = lambda d: LK.append(f"DEVIATION {d}: {LK.file_sha(LK.deviation_record(d))}  x  files 1 utc x")


@mutant("o43_drift_reference_and_gm_after_deviation")
def m_drift_reference_factor_dropped():
    R.DRIFT_REF_FACTOR = 1.0


@mutant("o43_drift_reference_and_gm_after_deviation")
def m_gm_ignores_deviation():
    R.active_deviation = lambda: None


@mutant("o44_resume_check_bitwise_against_stored_records")
def m_resume_check_always_passes():
    R.resume_check = lambda *a: True


@mutant("o44_resume_check_bitwise_against_stored_records")
def m_run_test_skips_resume_check():
    orig = SNAP["R"]["run_test"]

    def run_test(*a, **k):
        saved = R.resume_check
        R.resume_check = lambda *x: True
        try:
            return orig(*a, **k)
        finally:
            R.resume_check = saved
    R.run_test = run_test

SUITES = {"t": TR, "o": TO, "a": TA}


def run_tests(names):
    """Returns the names of the tests that FAILED (a caught mutant has at least one)."""
    failed = []
    for n in names:
        mod = SUITES[n[0]]
        fn = next(f for f in mod.TESTS if f.__name__ == n)
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                fn()
        except Exception:  # noqa: BLE001
            failed.append(n)
    return failed


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--write")
    ap.add_argument("--only")
    a = ap.parse_args(argv)
    TR.DEV = a.device
    TA.QUICK = True
    rv4.apply_determinism()
    res = []
    base = run_tests(sorted({n for _, ts in MUTANTS for n in ts}))
    if base:
        print("UNMUTATED TESTS FAIL; the mutation check is not meaningful:", base)
        return 2
    for fn, tests in MUTANTS:
        if a.only and a.only not in fn.__name__:
            continue
        t0 = time.time()
        restore()
        try:
            fn()
            failed = run_tests(tests)
        finally:
            restore()
        res.append({"mutant": fn.__name__, "caught": bool(failed), "failed_tests": failed, "seconds": round(time.time() - t0, 1)})
        print(f"{'CAUGHT ' if failed else 'MISSED '} {fn.__name__} ({time.time() - t0:.1f}s) by {failed}", flush=True)
    n_c = sum(r["caught"] for r in res)
    print(f"{n_c}/{len(res)} mutants caught on {a.device}")
    if a.write:
        Path(a.write).write_text(json.dumps({"device": a.device, "caught": n_c, "n": len(res), "all_caught": n_c == len(res), "mutants": res}, indent=1))
    return 0 if n_c == len(res) else 1


if __name__ == "__main__":
    sys.exit(main())
