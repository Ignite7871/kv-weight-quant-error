"""Mutation check for v5 (PROTOCOL_V5.md section 11): every injected fault must make at least one of its named tests fail. Each mutant is applied
in-process to the module under test, the named tests of test_rv5 / test_ops_v5 / test_analyze_v5 run, and every module is restored before the next.
Section 11's list: C1 on the wrong sphere for each new head_dim; slot norms pooled over the KV heads; HQQ with group 128, axis 0, a tuned setting or
NF4's Delta; HQQ dequantization skipped; a WikiText history that is the model's own generation; a prompt cut mid-word; the paragraph order not seeded;
the anchor with the v5 seed label; an A-X boundary not from the locked v4 report; A-X by point estimate only; Holm over the wrong family; U1 or U2
cut by a reduction; the $14 stop ignored; the start check after the problem; the driver's GPU-process guard off; the watcher accepting an
unverified checkpoint; an HF token accepted on the pod. Plus the v5 operational, record-form and anchor mutants below.

  venv/Scripts/python.exe research_v2/v5/tests/mutation_check_v5.py [--device cuda] [--write <json>] [--only <name>]
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
V5 = HERE.parent
RV = V5.parent
sys.path.insert(0, str(RV))
sys.path.insert(0, str(V5))
sys.path.insert(0, str(RV / "v4"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(RV / "tools"))

import test_ops_v5 as TO  # noqa: E402  (imports run_v5 -> HF before torch)
import test_rv5 as TR  # noqa: E402
import test_analyze_v5 as TA  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import analyze_v4 as A4  # noqa: E402
import analyze_v5 as A  # noqa: E402
import lock_v5 as LK  # noqa: E402
import run_v5 as R  # noqa: E402
import rv3.core as V  # noqa: E402
import rv5  # noqa: E402
import rv5_hqq as H  # noqa: E402
import v5_gpu_mutation_check as GMC  # noqa: E402
import v5_ops as O  # noqa: E402
import v5_remote as REM  # noqa: E402

MODS = {"V": V, "rv5": rv5, "H": H, "O": O, "R": R, "A": A, "A4": A4, "LK": LK, "GMC": GMC, "REM": REM, "TO": TO}
SNAP = {k: dict(vars(m)) for k, m in MODS.items()}
CLS = {(O.CostLedger, "can_start"), (O.CostLedger, "storage_usd"), (O.GpuProcessLock, "acquire"), (H.HQQDynamicLinear, "forward"), (R.Task, "history")}
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


def S_(mod, name):
    return SNAP[mod][name]


# ----------------------------------------------------------------------------- core and HQQ
@mutant("t05_c1_full_sphere_and_slot_norms_per_kv_head_new_geometries")
def m_c1_on_64_sphere_for_hd128():
    def g(shape, gen, device):
        s64 = tuple(shape[:-1]) + (min(64, shape[-1]),)
        z = torch.randn(s64, generator=gen, device=device, dtype=torch.float32)
        return torch.cat([z, torch.zeros(tuple(shape[:-1]) + (shape[-1] - s64[-1],), device=device)], -1) if shape[-1] > 64 else z
    V._gauss = g


@mutant("t05_c1_full_sphere_and_slot_norms_per_kv_head_new_geometries")
def m_c1_on_128_sphere_for_hd64():
    def g(shape, gen, device):
        if shape[-1] == 64:
            return torch.randn(tuple(shape[:-1]) + (128,), generator=gen, device=device, dtype=torch.float32)[..., :64].contiguous()
        return torch.randn(shape, generator=gen, device=device, dtype=torch.float32)
    V._gauss = g


@mutant("t05_c1_full_sphere_and_slot_norms_per_kv_head_new_geometries")
def m_slot_norms_pooled_over_kv_heads():
    V._slot_norm = lambda x: x.double().norm(dim=(-2, -1), keepdim=True).expand(*x.shape[:-1], 1)


@mutant("t09_hqq_config_twice_quantize_and_oracle")
def m_hqq_group_128():
    H.HQQ_CONFIG = {"nbits": 4, "group_size": 128, "axis": 1}


@mutant("t09_hqq_config_twice_quantize_and_oracle")
def m_hqq_axis_0():
    H.HQQ_CONFIG = {"nbits": 4, "group_size": 64, "axis": 0}


@mutant("t09_hqq_config_twice_quantize_and_oracle")
def m_hqq_tuned_3bit():
    H.HQQ_CONFIG = {"nbits": 3, "group_size": 64, "axis": 1}


@mutant("t11_hqq_c1_norm_from_hqq_delta", "t10_hqq_measurement_differs_from_nf4_and_alpha0_bitwise")
def m_hqq_dequantization_skipped():
    import torch.nn.functional as F
    H.HQQDynamicLinear.forward = lambda self, x: F.linear(x, self.fp16_weight, self.fp16_bias).to(self.out_dtype)


@mutant("t09_hqq_config_twice_quantize_and_oracle", "t11_hqq_c1_norm_from_hqq_delta")
def m_hqq_unit_measures_with_nf4():
    orig = S_("rv5", "build_backend")
    rv5.build_backend = lambda model, device, precision="fp16", quantizer="nf4", hqq_frozen=None: orig(model, device, precision, "nf4")


@mutant("t12_hqq_frozen_state_roundtrip")
def m_hqq_frozen_state_ignored():
    orig = S_("H", "replace_linear_with_hqq")
    H.replace_linear_with_hqq = lambda model, st, dt, dev, frozen=None: orig(model, st, dt, dev, None)


@mutant("o20_hqq_state_checked_at_every_load")
def m_hqq_state_not_checked_at_load():
    orig = S_("R", "load_unit")
    R.load_unit = lambda u, prec=None, check_state=True: orig(u, prec, False)


@mutant("t07_attention_snapshot_qk_norm_and_kv_heads")
def m_g7_qk_norm_not_checked():
    rv5.attention_matches = lambda snap, geom: snap["k_proj_out"] == geom["kv_heads"] * geom["head_dim"]


# ----------------------------------------------------------------------------- seeds
@mutant("t01_anchor_seeds_are_v4_and_other_units_are_new")
def m_anchor_with_v5_seed_label():
    rv5.master_label = lambda u: f"{rv5.V5_MASTER}|{rv5.UNITS[u][1]}|{rv5.UNITS[u][2]}"


@mutant("t01_anchor_seeds_are_v4_and_other_units_are_new")
def m_units_share_one_label():
    orig = S_("rv5", "master_label")
    rv5.master_label = lambda u: orig(u) if u == "U0" else rv5.V5_MASTER


# ----------------------------------------------------------------------------- WikiText
@mutant("o19_wikitext_history_is_the_real_text_not_a_generation")
def m_wikitext_history_is_own_generation():
    R.Task.history = lambda self, be, tok, ids_t, split, idx: R.fp16_history(be, tok, ids_t)


@mutant("t13_wikitext_paragraphs_and_prompt_cut")
def m_prompt_cut_mid_word():
    rv5.split_prompt = lambda text: (text[:600], text[600:])


@mutant("t13_wikitext_paragraphs_and_prompt_cut")
def m_headings_kept():
    rv5.is_heading = lambda s: False


@mutant("t14_wikitext_eligibility_history_and_order")
def m_paragraph_order_not_seeded():
    rv5.ordered = lambda paras, split: list(paras)


@mutant("t14_wikitext_eligibility_history_and_order")
def m_history_with_special_tokens():
    rv5.history_ids = lambda tok, rest: [int(x) for x in tok(rest, add_special_tokens=True).input_ids[:rv5.WIKI_HIST]]


@mutant("t14_wikitext_eligibility_history_and_order")
def m_eligibility_600_chars():
    rv5.WIKI_MIN_CHARS = 600


@mutant("t15_wikitext_records_hold_no_text_or_ids", "o11_exactly_once_resume_and_hash_only_history")
def m_wikitext_argmax_ids_released():
    rv5.public_record = lambda rec, task: rec


@mutant("t15_wikitext_records_hold_no_text_or_ids", "o11_exactly_once_resume_and_hash_only_history")
def m_wikitext_history_ids_released():
    rv5.history_record = lambda idx, ids, task: {"index": idx, "ids": [int(x) for x in ids]}


# ----------------------------------------------------------------------------- A-X and families
@mutant("a01_ax_references_and_sources_are_the_protocol_table", "a03_truth_at_head_dim_everywhere")
def m_ax_boundary_not_from_v4_report():
    orig = S_("A", "ax_references")

    def f(rep=None):
        r = orig(rep)
        for u in r:
            for c in r[u]:
                r[u][c]["boundary"] = 0.0
                r[u][c]["hd_side"] = 1 if r[u][c]["hd_pred"] > 0 else -1
        return r
    A.ax_references = f


@mutant("a01_ax_references_and_sources_are_the_protocol_table")
def m_ax_smollm2_sources_nearest_size_only():
    A.AX_SOURCES = {**S_("A", "AX_SOURCES"), "U2": {c: {"size": ["qwen2.5-1.5b"], "hd": ["llama3.2-1b"]} for c in ("NAT-C1", "H1")}}


@mutant("a02_ax_classify_and_one_sided_p", "a04_truth_at_size_and_mixed_and_between")
def m_ax_point_estimate_only():
    def f(summary, holm_p, ref):
        est = 0.5 * (summary["ci95"][0] + summary["ci95"][1])
        return "head-dimension pattern" if (est > ref["boundary"]) == (ref["hd_side"] > 0) else "size pattern"
    A.ax_classify = f


@mutant("a02_ax_classify_and_one_sided_p")
def m_ax_without_holm():
    orig = S_("A", "ax_classify")
    A.ax_classify = lambda s, hp, ref: orig(s, 0.0, ref)


@mutant("a02_ax_classify_and_one_sided_p")
def m_ax_two_sided_direction_ignored():
    def f(d, keys, B=A.B_FLIP):
        d = np.asarray(d, float)
        rng = A.rng_for(*keys)
        flips = rng.choice([-1.0, 1.0], size=(B, d.size))
        return float((1 + int(((flips * d).mean(1) >= d.mean()).sum())) / (B + 1)), 1
    A.one_sided_flip_p = f


@mutant("a01_ax_references_and_sources_are_the_protocol_table", "a03_truth_at_head_dim_everywhere")
def m_ax_sides_swapped():
    orig = S_("A", "ax_references")

    def f(rep=None):
        r = orig(rep)
        for u in r:
            for c in r[u]:
                r[u][c]["hd_side"] *= -1
        return r
    A.ax_references = f


@mutant("a04_truth_at_size_and_mixed_and_between")
def m_ax_joint_reading_any():
    orig = S_("A", "ax")

    def f(est, thetas, refs):
        out = orig(est, thetas, refs)
        if "head-dimension pattern" in out["joint_reading_NAT-C1"]["classifications"]:
            out["joint_reading_NAT-C1"]["reading"] = "in these checkpoints the v4 size trend tracks head dimension"
        return out
    A.ax = f


@mutant("a06_holm_within_each_family")
def m_holm_over_the_wrong_family():
    A.FAMILIES = {**S_("A", "FAMILIES"), "B-H2": (("U3", "U4", "U5", "U6"), "H2", +1)}


@mutant("a05_families_falsified_and_inconclusive")
def m_equivalence_ignored():
    def f(s, holm_p, sign):
        lo, hi = s["ci95"]
        if holm_p < 0.05 and ((hi < 0) if sign < 0 else (lo > 0)):
            return "supported"
        return "falsified" if ((lo > 0) if sign < 0 else (hi < 0)) else "inconclusive"
    A4.reading_directional = f


@mutant("a03_truth_at_head_dim_everywhere")
def m_h5_threshold_ln16():
    A4.LN4 = float(np.log(16.0))


@mutant("a08_r1_n60_and_unresolved_units")
def m_unresolved_unit_counted():
    A.design_n = lambda u: 0
    orig = S_("A", "unit_status")
    A.unit_status = lambda u: "complete" if orig(u) == "unresolved" and (A.OUT / u / "calibration.json").exists() else orig(u)


@mutant("a10_anchor_criteria")
def m_anchor_criterion_1_off():
    A.ANCHOR_D_MAX = 10.0
    orig = S_("A", "anchor_core")

    def f(*a):
        r = orig(*a)
        for h in r["contrasts"]:
            r["contrasts"][h]["criterion_1"] = True
        r["pass"] = r["criterion_3"]["rho_K"] and r["criterion_3"]["rho_V"] and r["criterion_3"]["b_hat"]
        return r
    A.anchor_core = f


@mutant("a10_anchor_criteria")
def m_anchor_calibration_tolerance_10pc():
    A.CAL_REL_TOL = 0.10


@mutant("a10_anchor_criteria")
def m_anchor_bitwise_ignores_slots():
    orig = S_("A", "bitwise_share")
    A.bitwise_share = lambda idx: {**orig(idx), "share": 1.0}


# ----------------------------------------------------------------------------- operations
@mutant("o04_ceiling_stop_and_storage_tier")
def m_ceiling_30():
    O.CEILING_USD = 30.0


@mutant("o04_ceiling_stop_and_storage_tier", "o10_drift_reference_and_start_check_before_problem")
def m_stop_ignored():
    O.CostLedger.can_start = lambda self, s: (True, self.estimate())


@mutant("o10_drift_reference_and_start_check_before_problem")
def m_start_check_after_the_problem():
    orig = CLS_SNAP[(O.CostLedger, "can_start")]
    state = {"first": True}

    def f(self, s):
        if state["first"]:
            state["first"] = False
            return True, self.estimate()
        return orig(self, s)
    O.CostLedger.can_start = f


@mutant("o10_drift_reference_and_start_check_before_problem", "o24_deviation_d2_drift_reference_and_first_stop_kept")
def m_d2_factor_missing():
    R.DRIFT_REF_FACTOR_D2 = 1.0


@mutant("o24_deviation_d2_drift_reference_and_first_stop_kept")
def m_d2_tolerance_widened():
    R.DRIFT = 0.50


@mutant("o23_deviation_d1_raises_the_stop_for_u6_only")
def m_d1_never_triggers():
    R.d1_stop = lambda est, cfg: None


@mutant("o23_deviation_d1_raises_the_stop_for_u6_only")
def m_d1_override_leaks_to_later_units():
    orig = S_("O", "set_stop_override")
    O.set_stop_override = lambda v: orig(v) if v is not None else None


@mutant("o23_deviation_d1_raises_the_stop_for_u6_only")
def m_d1_applies_to_u5():
    O.D1_UNIT = "U5"


@mutant("o04_ceiling_stop_and_storage_tier")
def m_storage_standard_rate_always():
    O.CostLedger.storage_usd = lambda self, t0, t1: max(0.0, t1 - t0) / O.MONTH_S * self.cfg["volume_gb"] * 0.07


@mutant("o05_no_token_anywhere")
def m_hf_token_accepted_on_the_pod():
    O.token_problems = lambda *a, **k: []


@mutant("o05_no_token_anywhere", "o13_remote_filter_and_local_fetch_verification")
def m_remote_allows_token():
    REM.FORBIDDEN = tuple(f for f in S_("REM", "FORBIDDEN") if "token" not in f)


@mutant("o02_rehearsal_files_and_guards")
def m_rehearsal_on_an_a40():
    orig = S_("R", "check_rehearsal")
    R.check_rehearsal = lambda name: (True, "rehearsal") if R.rehearsal() else orig(name)


@mutant("o02_rehearsal_files_and_guards")
def m_rehearsal_file_authorizes_a_real_run():
    R.auth_ok = lambda tag, lid: O.check_auth(R.OUT, tag, lid)


@mutant("o01_authorization_files_and_tokens")
def m_auth_tag_substring():
    def f(out, tag, lid):
        p = O.auth_path(out, tag)
        if lid is None or not p.exists():
            return False, "no"
        txt = p.read_text()
        return ("PROTOCOL_V5" in txt and lid in txt and tag in txt), "x"
    O.check_auth = f


@mutant("o06_one_gpu_process_lock")
def m_gpu_process_guard_off():
    O.GpuProcessLock.acquire = lambda self: (True, "ok")


@mutant("o07_order_rules")
def m_anchor_decision_skipped():
    R.anchor_ok = lambda: True


@mutant("o07_order_rules")
def m_design_not_required():
    R.design_locked = lambda: True


@mutant("o07_order_rules")
def m_hqq_record_not_required():
    R.hqq_locked = lambda u: True


@mutant("o07_order_rules")
def m_wikitext_dataset_not_required():
    R.downloaded = lambda name: name != "nothing"


@mutant("o08_unit_start_check_and_cut_units")
def m_unit_start_check_off():
    R.unit_start_check = lambda u, est: None


@mutant("o09_design_reduction_order")
def m_reduction_cuts_an_a_unit():
    R.REDUCTIONS = S_("R", "REDUCTIONS")[:3] + [("R1+R2+R3", {u: 60 for u in R.B_D_UNITS}, ("U6", "U5", "U2"))]


@mutant("o09_design_reduction_order")
def m_reduction_drops_mistral_before_r1():
    R.REDUCTIONS = [("none", {}, ()), ("R2", {}, ("U6",)), ("R1+R2", {u: 60 for u in R.B_D_UNITS}, ("U6",)),
                    ("R1+R2+R3", {u: 60 for u in R.B_D_UNITS}, ("U6", "U5"))]


@mutant("o09_design_reduction_order")
def m_design_without_the_1_48_factor():
    R.FACTOR = 1.0


@mutant("o09_design_reduction_order")
def m_design_without_the_wikitext_factor():
    R.WIKI_FACTOR = 1.0


@mutant("o09_design_reduction_order")
def m_rehearsal_probe_enters_a_real_design():
    orig = S_("R", "load_probes")

    def f():
        saved = R.rehearsal
        R.rehearsal = lambda: True
        try:
            return orig()
        finally:
            R.rehearsal = saved
    R.load_probes = f


@mutant("o10_drift_reference_and_start_check_before_problem")
def m_drift_reference_times_1_48_again():
    orig = S_("R", "expected_problem_seconds")
    R.expected_problem_seconds = lambda u: (orig(u) or 0) * 1.48 or None


@mutant("o11_exactly_once_resume_and_hash_only_history")
def m_records_appended_twice():
    R.append_new = lambda d, idx, recs: R.E.append_records(d, recs)


@mutant("o12_g5_bos_from_manifest")
def m_g5_bos_by_family():
    orig = S_("R", "g5")
    R.g5 = lambda tok, task, e: orig(tok, task, {**e, "expects_bos": e.get("family") in ("llama", "mistral")})


@mutant("o13_remote_filter_and_local_fetch_verification")
def m_watcher_accepts_an_unverified_checkpoint():
    O.verify_sha256_file = lambda p: True


@mutant("o13_remote_filter_and_local_fetch_verification")
def m_checkpoint_carries_the_hqq_state():
    orig = S_("O", "pack_checkpoint")

    def f(out, tag, extra_files=(), dest=None):
        extra = list(extra_files) + sorted((out / tag).glob("hqq_state.pt"))
        tar = orig(out, tag, extra, dest)
        import tarfile
        names = []
        with tarfile.open(tar) as tf:
            names = [n.replace("_extra/", f"{tag}/") for n in tf.getnames()]
        if f"{tag}/hqq_state.pt" in names:
            import shutil
            import tempfile
            with tempfile.TemporaryDirectory() as td, tarfile.open(tar) as tf:
                tf.extractall(td)
                p = Path(td)
                (p / tag).mkdir(exist_ok=True)
                shutil.move(str(p / "_extra" / "hqq_state.pt"), str(p / tag / "hqq_state.pt"))
                with tarfile.open(tar, "w") as tw:
                    for q in sorted(p.rglob("*")):
                        if q.is_file():
                            tw.add(q, arcname=q.relative_to(p).as_posix())
        return tar
    O.pack_checkpoint = f


@mutant("o14_closeout_gate_with_cut_units")
def m_closeout_ignores_final_state():
    O.STOP_FILES = S_("O", "STOP_FILES") + ("manifest.json", "calibration.json", "download.json", "not_a_file") + ("",)


@mutant("o21_gm_environment_reads_both_nvidia_smi_banners")
def m_gm_reads_only_the_old_cuda_banner():
    import re

    def f(run=__import__("subprocess").run):
        env = S_("O", "gpu_environment")(run=run)
        r = run(["nvidia-smi"], capture_output=True, timeout=20, **O.TEXT)
        m = re.search(r"CUDA Version:\s*([0-9][0-9.]*)", r.stdout or "")
        env["cuda_version"] = m.group(1) if m else None
        return env
    O.gpu_environment = f


@mutant("o22_watcher_exit_codes_on_a_local_pod")
def m_watcher_accepts_any_phase():
    import tempfile
    src = (V5 / "watch_v5.sh").read_text(encoding="utf-8")
    bad = src.replace("if d.get('phase')=='$PHASE' else 'other-phase'", "if True else 'other-phase'")
    assert bad != src
    d = Path(tempfile.mkdtemp()) / "research_v2" / "v5"
    d.mkdir(parents=True)
    import shutil
    shutil.copytree(RV / "tools", d.parent / "tools", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(RV / "v5", d, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "tests", "validation"))
    shutil.copy(RV / "V5_PROTOCOL_LOCK.txt", d.parent / "V5_PROTOCOL_LOCK.txt")
    (d / "watch_v5.sh").write_text(bad, encoding="utf-8", newline="\n")
    TO.WATCH_SCRIPT = d / "watch_v5.sh"


@mutant("o15_gm_verdict")
def m_gm_without_image_record():
    GMC.verdict = lambda r1, r2, t, m, env_complete, determinism_ok, image: bool(r1 == 0 and r2 == 0 and t.get("passed") == t.get("n")
                                                                                 and m.get("all_caught") and env_complete and determinism_ok)


@mutant("o16_lock_create_verify_and_v5_appendix_lines", "o17_locked_file_list_and_v4_previous_lock")
def m_lock_hqq_line_maps_to_calibration():
    orig = S_("LK", "appendix_file")
    LK.appendix_file = lambda line: (LK.OUT / line.split()[1].rstrip(":") / "calibration.json") if line.startswith("HQQ ") else orig(line)


@mutant("o17_locked_file_list_and_v4_previous_lock")
def m_lock_without_the_v4_previous_lock():
    LK.PREV_LOCKS = [p for p in S_("LK", "PREV_LOCKS") if "V4" not in p.name]


@mutant("o17_locked_file_list_and_v4_previous_lock")
def m_lock_omits_the_driver():
    orig = S_("LK", "locked_files")
    LK.locked_files = lambda: [p for p in orig() if p.name not in ("run_v5_pod.sh", "watch_v5.sh")]


@mutant("o16_lock_create_verify_and_v5_appendix_lines")
def m_lock_accepts_a_failed_gh():
    orig = S_("LK", "do_append")

    def f(kind, tag=None):
        if kind == "hqq":
            p = LK.appendix_file(f"HQQ {tag}: x")
            d = json.loads(p.read_text())
            d["pass"] = True
            p.write_text(json.dumps(d))
        return orig(kind, tag)
    LK.do_append = f


def run_tests(names, device):
    fails = []
    for n in names:
        mod = TR if n.startswith("t") else (TO if n.startswith("o") else TA)
        fn = getattr(mod, n)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                fn()
        except BaseException as e:  # noqa: BLE001
            fails.append((n, f"{type(e).__name__}: {str(e)[:120]}"))
    return fails


# ----------------------------------------------------------------------------- V5-D3
@mutant("o25_stale_fetched_flag_is_refetched_and_blocks_closeout")
def m_stale_fetched_flag():
    """The fault behind the lost U1 checkpoint (V5-A10): a tag the laptop marks checkpointed is never fetched again."""
    O.stale_tags = lambda tars, status: [t for t in S_("O", "newest_by_tag")(tars) if not (status.get(t) or {}).get("checkpointed")]


@mutant("o25_stale_fetched_flag_is_refetched_and_blocks_closeout")
def m_closeout_accepts_an_old_manifest():
    O.MANIFEST_MAX_AGE_S = 1e12


@mutant("o25_stale_fetched_flag_is_refetched_and_blocks_closeout")
def m_checkpoint_without_gm_deviation_records():
    orig = S_("O", "pack_checkpoint")

    def f(out, tag, extra_files=(), dest=None):
        hidden = [q for q in out.glob("gm_D*.json")]
        for q in hidden:
            q.rename(q.with_suffix(".hidden"))
        try:
            return orig(out, tag, extra_files, dest)
        finally:
            for q in hidden:
                q.with_suffix(".hidden").rename(q)
    O.pack_checkpoint = f


@mutant("o26_d3_calibration_hash_and_surviving_problem_checks")
def m_d3_reference_check_off():
    R.d3_reference_ok = lambda u, idx: True


@mutant("o26_d3_calibration_hash_and_surviving_problem_checks")
def m_d3_calibration_hash_ignored():
    orig = S_("R", "d3_calibration_check")

    def f(u):
        orig(u)
        p = R.mdir(u) / "d3_calibration_check.json"
        rec = __import__("json").loads(p.read_text(encoding="utf-8"))
        rec["equal"] = True
        p.write_text(__import__("json").dumps(rec), encoding="utf-8")
        (R.mdir(u) / "d3_calibration_mismatch.json").unlink(missing_ok=True)
        return True
    R.d3_calibration_check = f


@mutant("o26_d3_calibration_hash_and_surviving_problem_checks")
def m_d3_test_before_calibration_check():
    R.d3_calibration_reproduced = lambda u: True


@mutant("o27_lost_appendix_file_needs_an_appended_deviation_record")
def m_lost_file_accepted_without_appended_record():
    import json as _j
    LK.lost_files = lambda ids: S_("LK", "lost_files")([_j.loads(q.read_text(encoding="utf-8"))["id"] for q in LK.DEV_DIR.glob("V5-D*.json")])


@mutant("o28_d3_stop_is_the_ceiling_in_u1d3_only_and_storage_starts_at_creation")
def m_d3_stop_not_raised():
    O.D3_STOP_USD = 14.0


@mutant("o28_d3_stop_is_the_ceiling_in_u1d3_only_and_storage_starts_at_creation")
def m_d3_stop_for_every_unit():
    orig = S_("O", "set_stop_override")
    O.set_stop_override = lambda v: orig(15.0 if R.d3_active() else v)


@mutant("o28_d3_stop_is_the_ceiling_in_u1d3_only_and_storage_starts_at_creation")
def m_storage_from_the_billing_line():
    orig = CLS_SNAP[(O.CostLedger, "storage_usd")]
    O.CostLedger.storage_usd = lambda self, t0, t1: orig(self, min(t0, O.parse_utc("2026-11-12T00:00:00Z")), t1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--write")
    ap.add_argument("--only")
    a = ap.parse_args(argv)
    TR.DEV = a.device
    R.rv4.apply_determinism()
    res = []
    # every named test must pass on the unmutated code first (a test that fails anyway catches nothing)
    names = sorted({n for _, ts in MUTANTS for n in ts})
    base = run_tests(names, a.device)
    restore()
    if base:
        print("BASELINE FAILURES:", base)
        if a.write:
            Path(a.write).write_text(json.dumps({"caught": 0, "n": len(MUTANTS), "all_caught": False, "baseline_failures": base}, indent=1))
        return 1
    for fn, tests in MUTANTS:
        if a.only and a.only not in fn.__name__:
            continue
        t0 = time.time()
        try:
            fn()
            fails = run_tests(tests, a.device)
        finally:
            restore()
        caught = bool(fails)
        res.append({"mutant": fn.__name__, "caught": caught, "by": fails[:2], "seconds": round(time.time() - t0, 1)})
        print(f"{'CAUGHT' if caught else 'MISSED'} {fn.__name__} ({time.time() - t0:.1f}s) {fails[:1]}", flush=True)
    n = sum(r["caught"] for r in res)
    print(f"{n}/{len(res)} mutants caught on {a.device}")
    if a.write:
        Path(a.write).write_text(json.dumps({"device": a.device, "caught": n, "n": len(res), "all_caught": n == len(res), "results": res}, indent=1))
    return 0 if n == len(res) else 1


if __name__ == "__main__":
    sys.exit(main())
