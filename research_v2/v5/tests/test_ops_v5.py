"""CPU harness of the v5 cloud code paths (PROTOCOL_V5.md sections 9-11): authorization (PROTOCOL_V5, unit tokens, the REHEARSAL word), the
rehearsal guards, the $15 ceiling and $14 stop, the storage tier from the cost config, the no-token rule, the one-GPU-process lock, the order rules
(anchor, design, HQQ record, calibration), the unit start check, the design point and its reduction order (R1, R2, R3, R4; A and the anchor never cut),
the drift reference, the start check before a problem, exactly-once records and the resume check (hash-only WikiText histories), the remote command
filter and the local-pod fetch with verification, the lock tool on a temporary repository (create, verify, HQQ / ANCHOR / DESIGN lines), checkpoint
contents and the close-out gate. No GPU, no network.

  venv/Scripts/python.exe research_v2/v5/tests/test_ops_v5.py [--write <json>] [--only <name>]
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
V5 = HERE.parent
RV = V5.parent
sys.path.insert(0, str(RV))
sys.path.insert(0, str(V5))
sys.path.insert(0, str(RV / "v4"))
sys.path.insert(0, str(RV / "tools"))

import run_v5 as R  # noqa: E402  (HF before torch)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import lock_v5 as LK  # noqa: E402
import rv5  # noqa: E402
import v5_gpu_mutation_check as GMC  # noqa: E402
import v5_ops as O  # noqa: E402
import v5_remote as REM  # noqa: E402

TESTS, RESULTS = [], []
WATCH_SCRIPT = V5 / "watch_v5.sh"
LID = "abcd1234abcd1234"
FAKE_TOKEN = "".join(["hf", "_", "AbCdEfGhIjKlMnOpQrStUvWxYz012345"])
CFG = {"rate_usd_per_h": 0.59, "autostop_minutes": 30, "volume_gb": 50, "volume_created_utc": "2026-11-12T00:00:00Z", "volume_tier": "high_performance",
       "storage_usd_per_gb_month": 0.14}
T0 = O.parse_utc("2026-11-12T00:00:00Z")


def t(fn):
    TESTS.append(fn)
    return fn


class Clock:
    def __init__(self, t0):
        self.t = t0

    def __call__(self):
        return self.t


@contextlib.contextmanager
def tmpdir():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        yield Path(d)


def refused(fn):
    try:
        fn()
    except SystemExit as e:
        return str(e)
    return None


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def sha_lf(p):
    return hashlib.sha256(Path(p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def set_determinism(on):
    if on:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    else:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    torch.use_deterministic_algorithms(on)


@contextlib.contextmanager
def env(**kv):
    saved = {k: os.environ.get(k) for k in kv}
    for k, v in kv.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@contextlib.contextmanager
def driver(d, gpu="NVIDIA A40", verifies=True, tokens=()):
    saved = (R.OUT, R.lock_id, R.lock_verifies, O.gpu_name, O.token_problems, R.LOCK)
    set_determinism(True)
    R.OUT = d
    R.lock_id = lambda: LID
    R.lock_verifies = lambda: verifies
    O.gpu_name = lambda: gpu
    O.token_problems = lambda *a, **k: list(tokens)
    R.LOCK = d / "LOCK.txt"
    R.LOCK.write_text("x\n")
    try:
        with env(V5_REHEARSAL=None, V5_REHEARSAL_N=None):
            yield
    finally:
        R.OUT, R.lock_id, R.lock_verifies, O.gpu_name, O.token_problems, R.LOCK = saved


class Args:
    authorized_gpu_run = True


def auth(d, tag, extra=""):
    O.auth_path(d, tag).write_text(f"PROTOCOL_V5 {LID} {tag} {extra}\n")


def lock_line(prefix, p):
    R.LOCK.write_text(R.LOCK.read_text() + f"{prefix} {sha_lf(p)}  x\n")


def wj(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj))


# ----------------------------------------------------------------------------- authorization and rehearsal guards
@t
def o01_authorization_files_and_tokens():
    with tmpdir() as d:
        assert not O.check_auth(d, "U1", LID)[0]
        O.auth_path(d, "U1").write_text(f"PROTOCOL_V5 {LID} U10\n")
        assert not O.check_auth(d, "U1", LID)[0], "a file naming U10 must not authorize U1"
        O.auth_path(d, "U1").write_text(f"PROTOCOL_V4 {LID} U1\n")
        assert not O.check_auth(d, "U1", LID)[0], "a v4 file authorizes nothing in v5"
        O.auth_path(d, "U1").write_text(f"PROTOCOL_V5 ffff1234abcd1234 U1\n")
        assert not O.check_auth(d, "U1", LID)[0], "wrong lock id"
        auth(d, "U1")
        assert O.check_auth(d, "U1", LID)[0] and O.auth_path(d, "U1").name == "AUTHORIZED_V5_U1.txt"


@t
def o02_rehearsal_files_and_guards():
    with tmpdir() as d, driver(d):
        auth(d, "U1", "REHEARSAL")
        assert not R.auth_ok("U1", LID)[0], "a REHEARSAL file authorizes nothing outside the rehearsal"
        auth(d, "U1")
        assert R.auth_ok("U1", LID)[0]
        with env(V5_REHEARSAL="1"):
            assert not R.auth_ok("U1", LID)[0], "a rehearsal needs the REHEARSAL word"
            auth(d, "U1", "REHEARSAL")
            assert R.auth_ok("U1", LID)[0]
            assert not R.check_rehearsal("NVIDIA A40")[0], "no rehearsal on an A40"
            assert R.check_rehearsal("NVIDIA GeForce RTX 4060 Laptop GPU")[0]
            saved = R.OUT
            R.OUT = R.REAL_OUT
            try:
                assert not R.check_rehearsal("NVIDIA GeForce RTX 4060 Laptop GPU")[0], "never in research_v2/v5_runs"
            finally:
                R.OUT = saved
        assert not R.check_rehearsal("NVIDIA GeForce RTX 4060 Laptop GPU")[0] and R.check_rehearsal("NVIDIA A40")[0]
        assert not R.check_rehearsal("NVIDIA RTX A4000")[0], "exact SKU, not a substring"


@t
def o03_permission_rules():
    a = Args()
    with tmpdir() as d, driver(d):
        wj(d / "cost_config.json", CFG)
        assert "AUTHORIZED" in refused(lambda: R.require(a, "U0", "gates"))
        auth(d, "U0")
        assert "mutation" in refused(lambda: R.require(a, "U0", "gates"))
        b = Args()
        b.authorized_gpu_run = False
        assert "--authorized-gpu-run" in refused(lambda: R.require(b, "U0", "gates"))
    with tmpdir() as d, driver(d, verifies=False):
        wj(d / "cost_config.json", CFG)
        auth(d, "U0")
        assert "lock" in refused(lambda: R.require(a, "U0", "gates"))
    with tmpdir() as d, driver(d, tokens=["variable HF_TOKEN"]):
        wj(d / "cost_config.json", CFG)
        auth(d, "U0")
        assert "token" in refused(lambda: R.require(a, "U0", "gates"))
    with tmpdir() as d, driver(d):
        auth(d, "U0")
        assert "cost_config" in refused(lambda: R.require(a, "U0", "gates"))
        wj(d / "cost_config.json", CFG)
        O.add_billing_line(d, 14.05, "2026-11-12T00:00:00Z")
        assert "ceiling" in refused(lambda: R.require(a, "U0", "gates"))
    with tmpdir() as d, driver(d):
        set_determinism(False)
        wj(d / "cost_config.json", CFG)
        auth(d, "U0")
        assert "deterministic" in refused(lambda: R.require(a, "U0", "gates"))
        set_determinism(True)


@t
def o23_deviation_d1_raises_the_stop_for_u6_only():
    """V5-D1: at U6's start, spent + U6's projection + storage > $14 raises the stop to $16 for U6 only; the decision is written once."""
    a = Args()
    dz = {"per_unit": {"U5": {"usd": 2.2, "billed_h": 3.8}, "U6": {"usd": 4.46, "billed_h": 7.57}}}
    try:
        with tmpdir() as d, driver(d):
            wj(d / "cost_config.json", CFG)
            wj(d / "design.json", dz)
            auth(d, "U5")
            auth(d, "U6")
            O.add_billing_line(d, 14.05, "2026-11-12T00:00:00Z")
            assert "ceiling" in refused(lambda: R.require(a, "U5", "gates")), "U5 keeps the $14 stop"
            assert not (d / "U5" / "d1_stop.json").exists() and O.stop_usd() == 14.0
            why = quiet(refused, lambda: R.require(a, "U6", "gates"))
            assert "ceiling" not in why and "mutation" in why, why
            rec = json.loads((d / "U6" / "d1_stop.json").read_text(encoding="utf-8"))
            assert rec["triggered"] and rec["stop_usd"] == 16.0 and O.stop_usd() == 16.0
            assert rec["used_usd"] >= 14.05 and rec["projected_unit_usd"] == 4.46 and rec["storage_usd"] > 0
            assert abs(rec["projection_usd"] - (rec["used_usd"] + 4.46 + rec["storage_usd"])) < 1e-9
            O.add_billing_line(d, 16.05, "2026-11-12T01:00:00Z")
            assert "ceiling" in refused(lambda: R.require(a, "U6", "test")), "the raised stop is still a stop"
            assert json.loads((d / "U6" / "d1_stop.json").read_text(encoding="utf-8")) == rec, "decided once, at U6's start"
            assert "ceiling" in refused(lambda: R.require(a, "U5", "test")) and O.stop_usd() == 14.0, "the override never leaks to U5"
        with tmpdir() as d, driver(d):
            wj(d / "cost_config.json", CFG)
            wj(d / "design.json", dz)
            auth(d, "U6")
            O.add_billing_line(d, 5.0, "2026-11-12T00:00:00Z")
            quiet(refused, lambda: R.require(a, "U6", "gates"))
            rec = json.loads((d / "U6" / "d1_stop.json").read_text(encoding="utf-8"))
            assert not rec["triggered"] and rec["stop_usd"] == 14.0 and O.stop_usd() == 14.0
    finally:
        O.set_stop_override(None)


# ----------------------------------------------------------------------------- cost
@t
def o04_ceiling_stop_and_storage_tier():
    assert O.CEILING_USD == 15.0 and O.stop_usd() == 14.0
    with tmpdir() as d:
        wj(d / "cost_config.json", CFG)
        assert O.load_cost_config(d)["storage_usd_per_gb_month"] == 0.14
        for bad in (dict(CFG, volume_tier="standard"), {k: v for k, v in CFG.items() if k != "storage_usd_per_gb_month"}, dict(CFG, volume_tier="x")):
            wj(d / "cost_config.json", bad)
            try:
                O.load_cost_config(d)
                raise AssertionError(f"accepted {bad}")
            except (ValueError, FileNotFoundError):
                pass
        clk = Clock(T0 + 3600)
        for tier, rate in (("high_performance", 0.14), ("standard", 0.07)):
            cfg = dict(CFG, volume_tier=tier, storage_usd_per_gb_month=rate)
            c = O.CostLedger(d / tier, cfg, now=clk, pod_start=T0)
            (d / tier).mkdir()
            e = c.estimate()
            assert abs(e["inpod_usd"] - 3600 / O.MONTH_S * 50 * rate) < 1e-12, (tier, e)
        assert abs(O.storage_allowance(50, 20, 0.14) - 50 * 0.14 * 68 * 3600 / O.MONTH_S) < 1e-12
        c = O.CostLedger(d / "s", dict(CFG), now=Clock(T0), pod_start=T0)
        (d / "s").mkdir()
        O.add_billing_line(d / "s", 13.99, "2026-11-12T00:00:00Z")
        ok, _ = c.can_start(60)
        assert not ok, "the start check refuses a problem that would cross $14"


@t
def o05_no_token_anywhere():
    with tmpdir() as d:
        assert O.token_problems(env={}, proc=str(d / "noproc"), hf_home=str(d), home=str(d)) == []
        p = O.token_problems(env={"HF_TOKEN": FAKE_TOKEN}, proc=str(d / "noproc"), hf_home=str(d), home=str(d))
        assert p == ["variable HF_TOKEN"] and FAKE_TOKEN not in json.dumps(p), "names only, never the value"
        (d / "token").write_text("x")
        assert any("file" in x for x in O.token_problems(env={}, proc=str(d / "noproc"), hf_home=str(d), home=str(d)))
        (d / "token").unlink()
        (d / "1").mkdir()
        (d / "1" / "environ").write_bytes(b"A=1\0HUGGING_FACE_HUB_TOKEN=" + FAKE_TOKEN.encode() + b"\0")
        assert O.token_problems(env=None, proc=str(d), hf_home=str(d), home=str(d)) == ["variable HUGGING_FACE_HUB_TOKEN"] or \
            "variable HUGGING_FACE_HUB_TOKEN" in O.token_problems(env=None, proc=str(d), hf_home=str(d), home=str(d)), "PID 1 is read"
    assert os.environ.get("HF_HUB_DISABLE_IMPLICIT_TOKEN") == "1", "run_v5 never sends an implicit token"
    assert "hf_token=" in REM.FORBIDDEN and not REM.command_allowed("HF_TOKEN=abc python x")[0]


@t
def o06_one_gpu_process_lock():
    with tmpdir() as d:
        a = O.GpuProcessLock(d, "test:U1", alive=lambda pid: True)
        assert a.acquire()[0]
        b = O.GpuProcessLock(d, "test:U2", alive=lambda pid: True)
        ok, why = b.acquire()
        assert not ok and "another" in why
        a.release()
        assert b.acquire()[0]
        c = O.GpuProcessLock(d, "x", alive=lambda pid: False)
        assert c.acquire()[0], "a dead holder's lock is taken over"
        assert O.read_jsonl(d / "gpu_process_lock.jsonl")[0]["stale_lock_taken_over"]["block"] == "test:U2"
        assert O.pid_alive(os.getpid()) and not O.pid_alive(2 ** 22 + 12345)


# ----------------------------------------------------------------------------- order, start check, design
def set_gm(d):
    wj(d / "gm.json", {"all_passed": True})
    lock_line("GM:", d / "gm.json")


def set_anchor(d, decision="pass"):
    wj(d / "anchor.json", {"decision": decision})
    lock_line("ANCHOR:", d / "anchor.json")


def downloaded(d, *names):
    for n in names:
        wj(d / "_downloads" / f"{n}.json", {"pass": True})


@t
def o07_order_rules():
    with tmpdir() as d, driver(d):
        set_gm(d)
        downloaded(d, "qwen2.5-0.5b", "qwen3-0.6b-base", "smollm2-1.7b", "mistral-7b-v0.3", "wikitext2")
        assert "order" in refused(lambda: R.order_rules("U1", "gates"))
        assert "DESIGN" not in (refused(lambda: R.order_rules("U0", "gates")) or "")
        assert refused(lambda: R.order_rules("U0", "gates")) is None
        assert "U0 and the anchor" in refused(lambda: R.order_rules("DESIGN", "download"))
        wj(d / "U0" / "test_complete.json", {})
        assert "anchor" in refused(lambda: R.order_rules("U1", "gates"))
        set_anchor(d, "fail: stop and report")
        assert "anchor" in refused(lambda: R.order_rules("U1", "gates"))
        set_anchor(d, "author: proceed")
        assert refused(lambda: R.order_rules("DESIGN", "download")) is None
        assert "design" in refused(lambda: R.order_rules("U1", "gates"))
        wj(d / "design.json", {"pass": True, "n_test": {}, "cut_units": []})
        lock_line("DESIGN:", d / "design.json")
        assert "never revisited" in refused(lambda: R.order_rules("DESIGN", "download"))
        assert refused(lambda: R.order_rules("U1", "gates")) is None
        assert "gates" in refused(lambda: R.order_rules("U1", "calibrate"))
        for u in ("U1", "U2", "U3"):
            wj(d / u / "test_complete.json", {})
        wj(d / "U4" / "gates" / "gates.json", {"all_passed": True})
        wj(d / "U4" / "hqq.json", {"pass": True, "state_sha256": "a" * 64, "frozen": False})
        assert "HQQ" in refused(lambda: R.order_rules("U4", "calibrate")), "GH record before calibration"
        lock_line("HQQ U4:", d / "U4" / "hqq.json")
        assert refused(lambda: R.order_rules("U4", "calibrate")) is None
        assert "calibration" in refused(lambda: R.order_rules("U4", "test"))
        wj(d / "U4" / "precision.json", {"precision": "bf16"})
        assert "flag" in refused(lambda: R.order_rules("U4", "calibrate"))
        (d / "_downloads" / "wikitext2.json").unlink()
        wj(d / "U4" / "test_complete.json", {})
        wj(d / "U5" / "gates" / "gates.json", {"all_passed": True})
        assert "dataset" in refused(lambda: R.order_rules("U5", "calibrate")), "WikiText units need the verified parquet"
        wj(d / "STUDY_STOPPED.json", {})
        assert "stopped" in refused(lambda: R.order_rules("U5", "gates"))


@t
def o08_unit_start_check_and_cut_units():
    with tmpdir() as d, driver(d):
        wj(d / "design.json", {"cut_units": ["U6"], "reduction": "R1+R2", "per_unit": {"U2": {"usd": 3.0}, "U3": {"usd": 1.0}}})
        assert "cut" in refused(lambda: R.unit_start_check("U6", {"used_usd": 1.0}))
        assert refused(lambda: R.unit_start_check("U2", {"used_usd": 10.9})) is None
        msg = refused(lambda: R.unit_start_check("U2", {"used_usd": 11.5}))
        assert msg and "does not fit" in msg
        assert all((d / u / "not_run.json").exists() for u in ("U2", "U3", "U4", "U5", "U6")) and (d / "STUDY_STOPPED.json").exists()
        assert not (d / "U1" / "not_run.json").exists()


def probes(d, r_q=2.0, r_m=2.0, t_q=1.2, t_m=7.0, synthetic=False):
    man = rv5.models()
    for p, (m, q) in R.PROBES.items():
        base = t_q if m == "qwen2.5-0.5b" else t_m
        ta = base * (r_q if (q == "hqq" and m == "qwen2.5-0.5b") else r_m if q == "hqq" else 1.0)
        wj(d / "probes" / f"{p}.json", {"probe": p, "model": m, "quantizer": q, "t_arm_problem_s": ta, "t_hist_ref_problem_s": 0.8 * base,
                                         "t_load_s": 10.0 + (60.0 if q == "hqq" else 0.0), "finite": True, "layers": man[m]["geometry"]["layers"],
                                         "params": man[m]["params"], **({"rehearsal_synthetic": True} if synthetic else {})})


@t
def o09_design_reduction_order():
    with tmpdir() as d, driver(d):
        wj(d / "cost_config.json", CFG)
        probes(d)
        assert quiet(R.make_design)
        dz = json.loads((d / "design.json").read_text())
        assert dz["reduction"] == "none" and dz["cut_units"] == [] and all(n == 100 for n in dz["n_test"].values())
        assert abs(dz["expected_problem_s"]["U4"] - 1.48 * (6 * 2.4 + 0.96)) < 1e-9, "probe x 1.48"
        assert abs(dz["expected_problem_s"]["U3"] - 1.48 * (64 / 61.1) * (6 * 1.2 + 0.96)) < 1e-9, "WikiText x 64/61.1"
        assert abs(dz["hqq_ratio"]["qwen2.5-0.5b"] - 2.0) < 1e-12
    results = {}
    for used in (2.0, 5.0, 7.0, 9.5, 13.0):
        with tmpdir() as d, driver(d):
            wj(d / "cost_config.json", CFG)
            O.add_billing_line(d, used, "2026-11-12T00:00:00Z")
            probes(d, t_q=1.5, t_m=8.0, r_q=2.5, r_m=2.5)
            ok = quiet(R.make_design)
            dz = json.loads((d / ("design.json" if ok else "design_failed.json")).read_text())
            results[used] = (dz["reduction"], tuple(dz["cut_units"]))
            assert not set(dz["cut_units"]) & set(rv5.NEVER_CUT) or not ok
            if ok and dz["reduction"] != "none":
                assert all(dz["n_test"][u] == 60 for u in dz["n_test"] if u in R.B_D_UNITS) and dz["n_test"].get("U1") == 100
            for u in dz["cut_units"] if ok else []:
                assert (d / u / "not_run.json").exists()
    order = ["none", "R1", "R1+R2", "R1+R2+R3", "R4"]
    seq = [order.index(results[u][0]) for u in sorted(results)]
    assert seq == sorted(seq) and seq[0] < seq[-1] and results[13.0][0] == "R4", results
    assert ("R1+R2", ("U6",)) in results.values() or ("R1+R2+R3", ("U6", "U5")) in results.values(), results
    with tmpdir() as d, driver(d):
        wj(d / "cost_config.json", CFG)
        probes(d, synthetic=True)
        assert "rehearsal stand-in" in refused(lambda: quiet(R.make_design))


@t
def o10_drift_reference_and_start_check_before_problem():
    with tmpdir() as d, driver(d):
        wj(d / "design.json", {"expected_problem_s": {"U1": 40.0}})
        assert abs(R.expected_problem_seconds("U1") - 40.0 * 1.38) < 1e-9, "the design projection (with 1.48) times the V5-D2 factor 1.38"
        assert R.expected_problem_seconds("U0") is None
        wj(d / "cost_config.json", CFG)
        cost = O.CostLedger(d, CFG, now=Clock(T0), pod_start=T0)
        O.add_billing_line(d, 13.999, "2026-11-12T00:00:00Z")
        ran = []
        ok = R.block_loop("t", "U1", [1, 2], lambda i: ran.append(i) or True, lambda i: False, cost)
        assert not ok and ran == [] and (d / "U1" / "ceiling_stop.json").exists(), "checked before the problem, not after"
    with tmpdir() as d, driver(d):
        wj(d / "cost_config.json", CFG)
        cost = O.CostLedger(d, CFG, now=Clock(T0), pod_start=T0)
        clock = [0.0]

        def slow(i):
            clock[0] += 0.03
            time.sleep(0.03)
            return True
        ok = R.block_loop("t", "U1", list(range(12)), slow, lambda i: False, cost, expected_problem_s=0.01)
        assert not ok and (d / "U1" / "drift_stop.json").exists()


@t
def o24_deviation_d2_drift_reference_and_first_stop_kept():
    """V5-D2: reference = design projection x 1.38 for U1-U6 (U0 none); tolerance 25 %; a second drift stop never overwrites the first."""
    assert R.DRIFT == 0.25 and R.DRIFT_REF_FACTOR_D2 == 1.38
    with tmpdir() as d, driver(d):
        wj(d / "design.json", {"expected_problem_s": {u: 10.0 for u in ("U1", "U2", "U3", "U4", "U5", "U6")}})
        for u in ("U1", "U2", "U3", "U4", "U5", "U6"):
            assert abs(R.expected_problem_seconds(u) - 13.8) < 1e-9, u
        assert R.expected_problem_seconds("U0") is None
        wj(d / "cost_config.json", CFG)
        cost = O.CostLedger(d, CFG, now=Clock(T0), pod_start=T0)
        first = {"mean_problem_s": 49.95, "expected_problem_s": 36.16, "n": 10, "utc": "2026-11-12T06:51:18Z"}
        wj(d / "U1" / "drift_stop.json", first)
        before = (d / "U1" / "drift_stop.json").read_bytes()
        ran = []

        def fast(i):
            ran.append(i)
            return True
        assert R.block_loop("t", "U1", list(range(12)), fast, lambda i: False, cost, expected_problem_s=R.expected_problem_seconds("U1"))
        assert ran == list(range(12)) and not (d / "U1" / "drift_stop_D2.json").exists(), "a resumed unit runs on below the new reference"

        def slow(i):
            time.sleep(0.03)
            return True
        assert not R.block_loop("t", "U1", list(range(12)), slow, lambda i: False, cost, expected_problem_s=0.01)
        assert (d / "U1" / "drift_stop.json").read_bytes() == before, "the first drift-stop record is kept byte for byte"
        rec = json.loads((d / "U1" / "drift_stop_D2.json").read_text(encoding="utf-8"))
        assert rec["drift_ref_factor"] == 1.38 and rec["n"] == 10


# ----------------------------------------------------------------------------- records
def fake_recs(idx, task, u="U3"):
    recs = {}
    for a in R.rv4.CORE_ARMS:
        r = {"index": idx, "arm": a, "js": [0.0, 0.1], "top": [5, 6], "wall_seconds": 1.0, "unit": u}
        recs[a] = rv5.public_record(r, task)
    recs["NAT"]["_slots"] = {"normsK": np.ones((2, 1, 1)), "normsV": np.ones((2, 1, 1)), "S1K": np.ones((1, 1, 4)), "S1V": np.ones((1, 1, 4))}
    return recs


@t
def o11_exactly_once_resume_and_hash_only_history():
    with tmpdir() as d, driver(d):
        for task in ("wikitext2", "gsm8k"):
            u = "U3" if task == "wikitext2" else "U1"
            recs = fake_recs(7, task, u)
            canon = R.canonical(recs)
            R.write_slots(u, "test", recs)
            R.append_new(d / u / "test", 7, recs)
            R.append_new(d / u / "test", 7, recs)
            assert len((d / u / "test" / "NAT.jsonl").read_text().splitlines()) == 1, "exactly once"
            R.E.append_records(d / u / "test", {"FP16_history": rv5.history_record(7, [1, 2, 3], task)})
            assert R.test_done(u, 7)
            stored, hist = R.stored_canonical(u, 7)
            assert stored == canon and rv5.history_matches(hist, [1, 2, 3], task) and not rv5.history_matches(hist, [1, 2, 4], task)
            text = "".join(p.read_text() for p in (d / u / "test").glob("*.jsonl"))
            if task == "wikitext2":
                assert '"top"' not in text and '"ids"' not in text, "no WikiText token ids in released records"
        p = d / "U1" / "test" / "C1.jsonl"
        p.write_bytes(p.read_bytes() + b'{"index": 8, "js"')
        assert R.torn_record_files("U1") == [p] and "repair" in refused(lambda: R.require_intact("U1"))


@t
def o12_g5_bos_from_manifest():
    class Tok:
        bos_token_id, eos_token_id = 0, 0

        def __init__(self, bos):
            self.bos = bos

        def __call__(self, text, add_special_tokens=True):
            class X:
                input_ids = ([0] if (self.bos and add_special_tokens) else []) + [ord(c) % 50 + 1 for c in text]
            return X()

        def decode(self, ids, skip_special_tokens=True):
            return "".join(chr(i) for i in ids)

    class Task:
        s = {"gate_G5_test_ids": [1], "prompt_sha256": {"test": {"1": hashlib.sha256(b"q").hexdigest()}}}

        def prompt(self, split, idx):
            return "q"
    rep = R.g5(Tok(False), Task(), {"expects_bos": True})
    assert not rep["pass"] and rep["expected_bos"] is True and rep["bos_first"] == [False]
    assert R.g5(Tok(True), Task(), {"expects_bos": False})["bos_first"] == [True]
    man = rv5.models()
    assert man["qwen3-0.6b-base"]["expects_bos"] is False and man["smollm2-1.7b"]["expects_bos"] is False and man["mistral-7b-v0.3"]["expects_bos"] is True


# ----------------------------------------------------------------------------- remote, checkpoints, close-out
@t
def o13_remote_filter_and_local_fetch_verification():
    for bad in ("runpodctl pod stop x", "curl https://rest.runpod.io/v1/pods", "huggingface-cli login", "echo x > v5_runs/AUTHORIZED_V5_U1.txt",
                "export HF_TOKEN=abc"):
        assert not REM.command_allowed(bad)[0], bad
    assert REM.command_allowed("python research_v2/v5/run_v5.py test --tag U1 --authorized-gpu-run")[0]
    assert REM.command_allowed("cat v5_runs/AUTHORIZED_V5_U1.txt")[0]
    with tmpdir() as d:
        pod, lap = d / "pod", d / "lap"
        out = pod / "v5_runs"
        wj(out / "U1" / "test_complete.json", {})
        (out / "U1" / "hqq_state.pt").write_bytes(b"x" * 10)
        wj(out / "probes" / "qwen-nf4.json", {"probe": "qwen-nf4"})
        auth(out, "U1")
        tar = O.pack_checkpoint(out, "U1")
        with tarfile.open(tar) as tf:
            names = tf.getnames()
        assert "U1/test_complete.json" in names and "probes/qwen-nf4.json" in names and "AUTHORIZED_V5_U1.txt" in names
        assert "U1/hqq_state.pt" not in names, "the frozen HQQ state stays on the volume"
        saved = (REM.LOCAL_RUNS, REM.POD_RUNS, REM.POD_ROOT)
        REM.LOCAL_RUNS, REM.POD_RUNS, REM.POD_ROOT = lap, out.as_posix(), pod.as_posix()
        try:
            c = {"local": True}
            assert quiet(REM.cmd_fetch, c, "U1") == 0
            st = json.loads((lap / "status.json").read_text())
            assert st["U1"]["checkpointed"] and (lap / "U1" / "test_complete.json").exists()
            time.sleep(1.1)                                                   # a new tar name (the name carries the second)
            tar2 = O.pack_checkpoint(out, "U1")
            b = bytearray(tar2.read_bytes())
            b[600] ^= 1
            tar2.write_bytes(bytes(b))
            (lap / "U1" / "test_complete.json").unlink()
            msg = refused(lambda: quiet(REM.cmd_fetch, c, "U1"))
            assert msg and "does not match" in msg and not (lap / "U1" / "test_complete.json").exists(), "an unverified checkpoint is never merged"
        finally:
            REM.LOCAL_RUNS, REM.POD_RUNS, REM.POD_ROOT = saved


@t
def o14_closeout_gate_with_cut_units():
    with tmpdir() as d:
        wj(d / "U2" / "manifest.json", {})
        tar = O.pack_checkpoint(d, "U2")
        wj(d / "status.json", {"U2": {"checkpointed": True, "tar": tar.name, "sha256": O.sha256_file(tar)}})
        r = O.closeout_report(d, ["U2"])
        assert not r["ready"] and "final state" in r["missing"][0], "a checkpoint taken mid-unit does not clear the volume for deletion"
    with tmpdir() as d:
        wj(d / "design.json", {"cut_units": ["U6"]})
        r = O.closeout_report(d, ["U0", "DESIGN", "U1", "U6"])
        assert not r["ready"] and not any("U6" in m for m in r["missing"])
        for tg in ("U0", "DESIGN", "U1"):
            wj(d / tg / ("design_complete.json" if tg == "DESIGN" else "test_complete.json"), {})
            tar = O.pack_checkpoint(d, tg)
            st = json.loads((d / "status.json").read_text()) if (d / "status.json").exists() else {}
            st[tg] = {"checkpointed": True, "tar": tar.name, "sha256": O.sha256_file(tar)}
            wj(d / "status.json", st)
        r = O.closeout_report(d, ["U0", "DESIGN", "U1", "U6"])
        assert not r["ready"] and "manifest" in r["missing"][0], "V5-D3: no close-out without the pod's checkpoint manifest"
        write_manifest(d, d / "checkpoints")
        assert O.closeout_report(d, ["U0", "DESIGN", "U1", "U6"])["ready"]


def write_manifest(lap, pod_checkpoints, utc=None):
    text = "".join(q.read_text() for q in sorted(Path(pod_checkpoints).glob("*.tar.sha256")))
    wj(lap / "pod_checkpoints.json", {"utc": utc or O.utc(time.time()), "tars": O.parse_pod_sums(text)})


@t
def o21_gm_environment_reads_both_nvidia_smi_banners():
    """r3 (found by the rehearsal): drivers 6xx print 'CUDA UMD Version: 13.5' where older ones print 'CUDA Version: 12.8'."""
    class Res:
        def __init__(self, out):
            self.returncode, self.stdout = 0, out
    for banner, want in (("| NVIDIA-SMI 570.86   Driver Version: 570.86   CUDA Version: 12.8     |", "12.8"),
                         ("| NVIDIA-SMI 621.14     KMD Version: 621.14        CUDA UMD Version: 13.5     |", "13.5")):
        def run(argv, **k):
            return Res("NVIDIA A40, 570.86, Enabled, 46068 MiB\n") if any("--query-gpu" in a for a in argv) else Res(banner)
        env_ = O.gpu_environment(run=run)
        assert env_["cuda_version"] == want and O.gpu_environment_complete(env_), (banner, env_)


@t
def o22_watcher_exit_codes_on_a_local_pod():
    """r4 (found by the rehearsal): only the expected phase's status counts, and 'driver gone' is confirmed by a second look, so an earlier
    phase's 'done' or a dead driver's leftover status cannot end the watch of a phase that is starting."""
    import shutil
    bash = shutil.which("bash")
    lid = subprocess.run([sys.executable, str(RV / "tools" / "lock_v5.py"), "--lock-id"], capture_output=True, text=True).stdout.strip()
    assert bash and lid
    with tmpdir() as d:
        pod, lap = d / "pod", d / "lap"
        (pod / "v5_runs" / "driver").mkdir(parents=True)
        (pod / "repo").mkdir()
        lap.mkdir()
        O.auth_path(lap, "U1").write_text(f"PROTOCOL_V5 {lid} U1\n")
        wj(lap / "status.json", {})
        envw = dict(os.environ, V5_LOCAL_POD=pod.as_posix(), V5_LOCAL_RUNS=lap.as_posix(), V5_LAPTOP_PY=sys.executable, V5_WATCH_POLL="0",
                    V5_WATCH_CONFIRM="0", MSYS_NO_PATHCONV="1")

        def watch(status, pid_alive):
            (pod / "v5_runs" / "driver" / "status.json").write_text(json.dumps(status))
            lk = pod / "v5_runs" / "driver" / "driver.lockdir"
            shutil.rmtree(lk, ignore_errors=True)
            r = subprocess.run([bash, str(WATCH_SCRIPT), "U1", "units"], env=envw, capture_output=True, text=True, timeout=300)
            return r.returncode, r.stdout
        assert watch({"state": "done", "phase": "design"}, False)[0] == 6, "an earlier phase's done is not this phase's end"
        assert watch({"state": "running", "phase": "units"}, False)[0] == 2
        code, out = watch({"state": "done", "phase": "units"}, False)
        assert code == 0 and "WATCH PHASE DONE" in out
        assert watch({"state": "stopped", "phase": "units"}, False)[0] == 3 and watch({"state": "failed", "phase": "units"}, False)[0] == 1
    txt = (V5 / "watch_v5.sh").read_text(encoding="utf-8")
    assert "gone_confirm" in txt and "$PHASE" in txt and "status starting" in (V5 / "run_v5_pod.sh").read_text(encoding="utf-8")


@t
def o15_gm_verdict():
    good = dict(r1=0, r2=0, t={"passed": 16, "n": 16}, m={"all_caught": True}, env_complete=True, determinism_ok=True, image={"digest": "x"})
    assert GMC.verdict(**good)
    for k, v in (("r1", 1), ("t", {"passed": 15, "n": 16}), ("t", {}), ("m", {"all_caught": False}), ("env_complete", False), ("determinism_ok", False),
                 ("image", None)):
        assert not GMC.verdict(**dict(good, **{k: v})), k


# ----------------------------------------------------------------------------- lock tool on a temporary repository
@contextlib.contextmanager
def lock_repo():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        root = Path(td)
        rv, v5 = root / "research_v2", root / "research_v2" / "v5"
        v5.mkdir(parents=True)
        (rv / "PREV.txt").write_text("previous lock\n")
        (v5 / "V5_MODEL_MANIFEST.json").write_text(json.dumps({"models": {"m": {"hf_id": "o/m", "revision": "r", "shards": {"s.safetensors": {"sha256": "c" * 64}}}},
                                                               "datasets": {"w": {"repo": "o/w", "revision": "q", "files": {"test": {"path": "t.parquet", "sha256": "d" * 64}}}}}))
        (v5 / "PROTOCOL.md").write_text("r2\n")
        (v5 / "DEVIATION_LOG_V5.md").write_text("# log\n")
        run = lambda *a: subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false", *a],  # noqa: E731
                                        capture_output=True, check=True)
        run("init", "-q")
        run("config", "core.autocrlf", "false")
        saved = {k: getattr(LK, k) for k in ("ROOT", "RV", "V5", "LOCK", "OUT", "HISTORY", "DEVIATION_LOG", "DEV_DIR", "PREV_LOCKS", "locked_files")}
        LK.ROOT, LK.RV, LK.V5, LK.LOCK, LK.OUT = root, rv, v5, rv / "V5_PROTOCOL_LOCK.txt", rv / "v5_runs"
        LK.HISTORY, LK.DEVIATION_LOG, LK.DEV_DIR, LK.PREV_LOCKS = v5 / "lock_history", v5 / "DEVIATION_LOG_V5.md", v5 / "deviations", [rv / "PREV.txt"]
        LK.locked_files = lambda: [v5 / "PROTOCOL.md", v5 / "V5_MODEL_MANIFEST.json"]
        run("add", "-A")
        run("commit", "-qm", "first")
        try:
            quiet(LK.create)
            run("add", "-A")
            run("commit", "-qm", "lock")
            yield root, v5
        finally:
            for k, v in saved.items():
                setattr(LK, k, v)


@t
def o16_lock_create_verify_and_v5_appendix_lines():
    with lock_repo() as (root, v5):
        text = LK.LOCK.read_text(encoding="utf-8")
        assert "DATASET w o/w@q: " + "d" * 64 in text and "MODEL m o/m@r: " + "c" * 64 in text and "PROTOCOL_V5" in text
        lid = LK.lock_id()
        assert quiet(LK.verify) == 0
        out = LK.OUT
        wj(out / "U4" / "hqq.json", {"pass": False, "state_sha256": "a" * 64, "frozen": False})
        assert "GH has not passed" in refused(lambda: quiet(LK.do_append, "hqq", "U4"))
        wj(out / "U4" / "hqq.json", {"pass": True, "state_sha256": "a" * 64, "frozen": False})
        quiet(LK.do_append, "hqq", "U4")
        wj(out / "anchor.json", {"decision": "pass"})
        quiet(LK.do_append, "anchor")
        wj(out / "design.json", {"pass": True, "reduction": "R1", "cut_units": []})
        quiet(LK.do_append, "design")
        assert "never replaced" in refused(lambda: quiet(LK.do_append, "design"))
        assert LK.lock_id() == lid and quiet(LK.verify) == 0, "appended lines do not change the lock id"
        wj(out / "U4" / "hqq.json", {"pass": True, "state_sha256": "b" * 64, "frozen": False})
        assert quiet(LK.verify) == 1, "an appended file that changes no longer verifies"
        (v5 / "PROTOCOL.md").write_text("changed\n")
        assert "--amend" in refused(lambda: quiet(LK.create))
        assert "started" in refused(lambda: quiet(LK.amend, "r3", "text")), "no pre-data amendment after data"


@t
def o17_locked_file_list_and_v4_previous_lock():
    rels = {p.relative_to(RV.parent).as_posix() for p in LK.locked_files()}
    for need in ("research_v2/v5/rv5.py", "research_v2/v5/rv5_hqq.py", "research_v2/v5/run_v5.py", "research_v2/v5/run_v5_pod.sh", "research_v2/v5/watch_v5.sh",
                 "research_v2/v5/v5_wikitext_sample.json", "research_v2/v4/rv4.py", "research_v2/v4/v4_ops.py", "research_v2/v4/v4_sample.json",
                 "research_v2/tools/v5_remote.py", "research_v2/v5/RECOMPUTATION_TOLERANCES_V5.md"):
        assert need in rels, need
    assert RV / "V4_PROTOCOL_LOCK.txt" in LK.PREV_LOCKS
    assert LK.appendix_file("HQQ U4: x") == LK.OUT / "U4" / "hqq.json" and LK.appendix_file("ANCHOR: x") == LK.OUT / "anchor.json"


@t
def o18_cloud_files_name_the_study():
    setup = (V5 / "cloud" / "pod_setup_v5.sh").read_text(encoding="utf-8")
    assert O.IMAGE_TAG in setup and "hqq==0.2.8.post1" in setup and "--no-deps" in setup and "DISABLE_CUDA=1" in setup and "HF_TOKEN" in setup
    drv = (V5 / "run_v5_pod.sh").read_text(encoding="utf-8")
    assert "runpodctl" not in drv and "AUTHORIZED" not in drv and "gpu_process.lock" in drv and "driver.lockdir" in drv
    for p in ("anchor", "design", "units", "u1d3"):
        assert f"  {p})" in drv
    w = (V5 / "watch_v5.sh").read_text(encoding="utf-8")
    assert "sync ||" in w and "exit 4" in w and "exit 5" in w and "checkpointed')" not in w


@t
def o19_wikitext_history_is_the_real_text_not_a_generation():
    class Data:
        def problem(self, split, idx):
            txt = " ".join(f"w{i}" for i in range(400))
            p, r = rv5.split_prompt(txt)
            return txt, p, r

    class Tok:
        def __call__(self, text, add_special_tokens=True, return_tensors=None):
            class X:
                input_ids = ([1] if add_special_tokens else []) + [len(w) for w in text.split()]
            return X()
    with tmpdir() as d, driver(d):
        task = R.Task.__new__(R.Task)
        task.u, task.name, task.wiki, task.s, task.data = "U3", "wikitext2", True, {}, Data()

        class NoModel:
            def __getattr__(self, k):
                raise AssertionError("the WikiText history must not come from the model")
        h = task.history(NoModel(), Tok(), None, "test", 0)
        assert h == rv5.history_ids(Tok(), Data().problem("test", 0)[2]) and len(h) == 65 and h[0] != 1


@t
def o20_hqq_state_checked_at_every_load():
    import rv5_hqq as H

    class Be:
        def __init__(self):
            self.model = torch.nn.Linear(2, 2)
    want = H.state_hash(Be().model)
    saved = rv5.load_model
    rv5.load_model = lambda *a, **k: (Be(), object())
    try:
        with tmpdir() as d, driver(d):
            wj(d / "U4" / "hqq.json", {"state_sha256": want, "frozen": False})
            R.load_unit("U4")
            wj(d / "U4" / "hqq.json", {"state_sha256": "0" * 64, "frozen": False})
            assert "differs" in refused(lambda: R.load_unit("U4")) and (d / "U4" / "hqq_state_failure.json").exists()
            seen = {}
            rv5.load_model = lambda *a, **k: (seen.update(frozen=k.get("hqq_frozen")) or Be(), object())
            wj(d / "U4" / "hqq.json", {"state_sha256": want, "frozen": True})
            R.load_unit("U4")
            assert seen["frozen"] and seen["frozen"].endswith("hqq_state.pt"), "a frozen GH record loads the saved state"
    finally:
        rv5.load_model = saved


# ----------------------------------------------------------------------------- V5-D3 (U1 re-execution after the lost checkpoint, V5-A10)
@t
def o25_stale_fetched_flag_is_refetched_and_blocks_closeout():
    """V5-D3: a tag already marked checkpointed on the laptop is fetched again when the pod holds a newer or different checkpoint of it, by the
    sync, by the watcher before it reports a phase done, and close-out refuses until the laptop holds the pod's newest checkpoint of every tag.
    Also: GM <ID> records travel in every checkpoint (the lost gm_D2.json)."""
    import shutil
    bash = shutil.which("bash")
    lid = subprocess.run([sys.executable, str(RV / "tools" / "lock_v5.py"), "--lock-id"], capture_output=True, text=True).stdout.strip()
    assert bash and lid
    with tmpdir() as d:
        pod, lap = d / "pod", d / "lap"
        out = pod / "v5_runs"
        (pod / "repo").mkdir(parents=True)
        out.mkdir(parents=True)
        lap.mkdir()
        for dd in (out, lap):
            O.auth_path(dd, "U1").write_text(f"PROTOCOL_V5 {lid} U1\n")
        wj(out / "U1" / "drift_stop.json", {"n": 10})
        wj(out / "gm_D2.json", {"all_passed": True})
        tar1 = O.pack_checkpoint(out, "U1")
        with tarfile.open(tar1) as tf:
            assert "gm_D2.json" in tf.getnames(), "GM <ID> records travel in every checkpoint"
        saved = (REM.LOCAL_RUNS, REM.POD_RUNS, REM.POD_ROOT)
        REM.LOCAL_RUNS, REM.POD_RUNS, REM.POD_ROOT = lap, out.as_posix(), pod.as_posix()
        c = {"local": True}
        try:
            quiet(REM.cmd_sync, c)
            st = json.loads((lap / "status.json").read_text())
            assert st["U1"]["tar"] == tar1.name and st["U1"]["checkpointed"]
            quiet(REM.cmd_manifest, c)
            assert O.closeout_report(lap, ["U1"])["ready"], "drift-stopped and fetched: ready while the pod holds nothing newer"
            time.sleep(1.1)                                                   # the U1 test resumes and completes; a second checkpoint
            wj(out / "U1" / "test_complete.json", {"n": 100})
            tar2 = O.pack_checkpoint(out, "U1")
            quiet(REM.cmd_manifest, c)
            st = json.loads((lap / "status.json").read_text())
            man = json.loads((lap / "pod_checkpoints.json").read_text())
            assert O.stale_tags(man["tars"], st) == ["U1"], "the checkpointed flag alone must not hide a newer pod checkpoint"
            r = O.closeout_report(lap, ["U1"])
            assert not r["ready"] and any(tar2.name in m for m in r["missing"]), r
            quiet(REM.cmd_sync, c)
            st = json.loads((lap / "status.json").read_text())
            assert st["U1"]["tar"] == tar2.name and (lap / "U1" / "test_complete.json").exists()
            quiet(REM.cmd_manifest, c)
            man = json.loads((lap / "pod_checkpoints.json").read_text())
            assert O.closeout_report(lap, ["U1"])["ready"]
            assert O.stale_tags({tar2.name: "0" * 64}, st) == ["U1"], "same name, different hash: stale"
            assert not O.closeout_report(lap, ["U1"], now=O.parse_utc(man["utc"]) + 3601)["ready"], "a manifest older than an hour"
            st["U1"]["fetched_utc"] = O.utc(O.parse_utc(man["utc"]) + 5)
            wj(lap / "status.json", st)
            assert not O.closeout_report(lap, ["U1"], now=O.parse_utc(man["utc"]) + 10)["ready"], "a manifest from before the last fetch"
        finally:
            REM.LOCAL_RUNS, REM.POD_RUNS, REM.POD_ROOT = saved
        time.sleep(1.1)                                                       # the watcher fetches a newer checkpoint before it reports done
        wj(out / "U1" / "late.json", {})
        tar3 = O.pack_checkpoint(out, "U1")
        (out / "driver").mkdir()
        (out / "driver" / "status.json").write_text(json.dumps({"state": "done", "phase": "u1d3"}))
        envw = dict(os.environ, V5_LOCAL_POD=pod.as_posix(), V5_LOCAL_RUNS=lap.as_posix(), V5_LAPTOP_PY=sys.executable, V5_WATCH_POLL="0",
                    V5_WATCH_CONFIRM="0", MSYS_NO_PATHCONV="1")
        r = subprocess.run([bash, str(WATCH_SCRIPT), "U1", "u1d3"], env=envw, capture_output=True, text=True, timeout=300)
        st = json.loads((lap / "status.json").read_text())
        assert r.returncode == 0 and st["U1"]["tar"] == tar3.name and (lap / "U1" / "late.json").exists(), (r.returncode, r.stdout[-500:])


@t
def o26_d3_calibration_hash_and_surviving_problem_checks():
    """V5-D3: the re-executed U1 calibration must reproduce the locked CALIBRATION U1 hash, every surviving first-run problem must be re-executed
    bitwise equal (wall_seconds aside), and under D3 U1's test is refused until the calibration check has passed."""
    import shutil
    import analyze_v5 as AV
    with tmpdir() as d, driver(d):
        u = "U1"
        for i in (7, 8, 9):
            recs = fake_recs(i, "gsm8k", u)
            R.write_slots(u, "test", recs)
            R.append_new(d / u / "test", i, recs)
            R.E.append_records(d / u / "test", {"FP16_history": rv5.history_record(i, [1, 2, 3], "gsm8k")})
        shutil.copytree(d / u / "test", d / u / R.D3_REF / "test")
        wj(d / u / R.D3_REF / "d3_reference.json", {"ids": [7, 8]})
        p = d / u / R.D3_REF / "test" / "C1.jsonl"
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        rows[0]["wall_seconds"] = 99.0
        p.write_text("".join(json.dumps(r) + "\n" for r in rows))
        assert R.d3_reference_ok(u, 7), "wall_seconds is not part of the comparison"
        assert R.d3_reference_ok(u, 9) and not (d / u / "d3_reference_check.jsonl").read_text().count('"index": 9'), "9 is not a reference problem"
        assert not R.d3_reference_complete(u), "8 has not been checked yet"
        rows[1]["js"] = [0.0, 0.2]
        p.write_text("".join(json.dumps(r) + "\n" for r in rows))
        assert not R.d3_reference_ok(u, 8) and (d / u / "d3_reference_failure.json").exists() and not R.d3_reference_complete(u)
        good = json.dumps({"b_hat": 4.04, "unit": u}, indent=1)
        R.LOCK.write_text(R.LOCK.read_text() + f"CALIBRATION U1: {hashlib.sha256(good.encode()).hexdigest()}  U1/calibration.json\n")
        (d / u / "calib").mkdir(parents=True)
        saved = AV.calibrate
        try:
            AV.calibrate = lambda unit: (AV.OUT / unit / "calibration.json").write_text(good, encoding="utf-8")
            assert quiet(R.d3_calibration_check, u) and R.d3_calibration_reproduced(u) and not (d / u / "d3_calibration_mismatch.json").exists()
            assert AV.OUT == saved.__globals__["OUT"], "the locked analysis module's runs directory is restored"
            AV.calibrate = lambda unit: (AV.OUT / unit / "calibration.json").write_text(good.replace("4.04", "4.05"), encoding="utf-8")
            assert not quiet(R.d3_calibration_check, u) and (d / u / "d3_calibration_mismatch.json").exists() and not R.d3_calibration_reproduced(u)
        finally:
            AV.calibrate = saved
        set_gm(d)
        set_anchor(d)
        downloaded(d, "qwen3-0.6b-base")
        wj(d / "U0" / "test_complete.json", {})
        wj(d / "design.json", {"pass": True, "n_test": {}, "cut_units": []})
        lock_line("DESIGN:", d / "design.json")
        wj(d / u / "gates" / "gates.json", {"all_passed": True})
        wj(d / u / "calibration.json", json.loads(good))
        (d / u / "calibration.json").write_text(good, encoding="utf-8")
        assert refused(lambda: R.order_rules(u, "test")) is None, "before D3 is appended the rule is the locked one"
        R.LOCK.write_text(R.LOCK.read_text() + "DEVIATION D3: x\n")
        assert R.d3_active() and "V5-D3" in refused(lambda: R.order_rules(u, "test"))
        wj(d / u / "d3_calibration_check.json", {"equal": True})
        assert refused(lambda: R.order_rules(u, "test")) is None


@t
def o28_d3_stop_is_the_ceiling_in_u1d3_only_and_storage_starts_at_creation():
    """V5-D3 (corresponding author, before any cloud action): with D3 appended, GM and U1 (phase u1d3) stop at the $15 ceiling, every other tag
    keeps $14; and the billing-based amount adds storage only from the configured (new) volume's creation, never for the time before it."""
    a = Args()
    try:
        with tmpdir() as d, driver(d):
            wj(d / "cost_config.json", CFG)
            wj(d / "design.json", {"per_unit": {"U1": {"usd": 1.47}}})
            for tg in ("GM", "U1", "U2"):
                auth(d, tg)
            O.add_billing_line(d, 14.30, "2026-11-12T00:00:00Z")
            assert "ceiling" in refused(lambda: R.require(a, "U1", "gates")) and O.stop_usd() == 14.0, "before D3: $14"
            R.LOCK.write_text(R.LOCK.read_text() + "DEVIATION D3: x\n")
            why = quiet(refused, lambda: R.require(a, "U1", "gates"))
            assert "ceiling" not in why and O.stop_usd() == 15.0, why
            why = quiet(refused, lambda: R.require(a, "GM", "gm"))
            assert "ceiling" not in why and O.stop_usd() == 15.0, why
            assert "ceiling" in refused(lambda: R.require(a, "U2", "gates")) and O.stop_usd() == 14.0, "no other unit gets $15"
            O.add_billing_line(d, 15.01, "2026-11-12T00:00:01Z")
            assert "ceiling" in refused(lambda: R.require(a, "U1", "test")), "$15 is still a stop"
        with tmpdir() as d:
            cfg = dict(CFG, volume_created_utc="2026-11-22T00:00:00Z")
            O.add_billing_line(d, 5.0, "2026-11-12T00:00:00Z")
            est = O.CostLedger(d, cfg).estimate(O.parse_utc("2026-11-23T00:00:00Z"))
            day = 86400.0 / O.MONTH_S * cfg["volume_gb"] * cfg["storage_usd_per_gb_month"]
            assert abs(est["billing_based_usd"] - (5.0 + day)) < 1e-9, (est, day)
    finally:
        O.set_stop_override(None)


@t
def o27_lost_appendix_file_needs_an_appended_deviation_record():
    """V5-D3: an appended run file whose only copy was lost (gm_D2.json) verifies only as 'recorded lost', and only once a deviation record that
    names it with its hash is appended; a present file that differs, or a file the record does not name, still fails."""
    with lock_repo() as (root, v5):
        run = lambda *a: subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false", *a],  # noqa: E731
                                        capture_output=True, check=True)
        out = LK.OUT
        wj(out / "gm.json", {"all_passed": True})
        quiet(LK.do_append, "gm")
        h = sha_lf(out / "gm.json")
        wj(out / "anchor.json", {"decision": "pass"})
        quiet(LK.do_append, "anchor")
        assert quiet(LK.verify) == 0
        (out / "gm.json").unlink()
        assert quiet(LK.verify) == 1, "a lost appended file fails until a deviation records it"
        (v5 / "PROTOCOL.md").write_text("r2 + D9\n")
        run("add", "-A")
        run("commit", "-qm", "d9 code")
        assert "no appended line" in refused(lambda: quiet(LK.deviation, "D9", "x", {"gm.json": "0" * 64}))
        assert "exists here" in refused(lambda: quiet(LK.deviation, "D9", "x", {"anchor.json": sha_lf(out / "anchor.json")}))
        quiet(LK.deviation, "D9", "x", {"gm.json": h})
        run("add", "-A")
        run("commit", "-qm", "d9 record")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = LK.verify()
        assert rc == 1 and "gm.json" in buf.getvalue() and "recorded lost" not in buf.getvalue(), "a record that is not appended declares nothing"
        quiet(LK.append_deviation, "D9")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = LK.verify()
        assert rc == 0 and "recorded lost" in buf.getvalue(), buf.getvalue()
        wj(out / "gm.json", {"all_passed": False})
        assert quiet(LK.verify) == 1, "a present file that differs from its hash still fails"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--write")
    ap.add_argument("--only")
    a = ap.parse_args(argv)
    RESULTS.clear()
    for fn in TESTS:
        if a.only and a.only not in fn.__name__:
            continue
        t0 = time.time()
        try:
            fn()
            RESULTS.append({"test": fn.__name__, "ok": True})
            print(f"PASS {fn.__name__} ({time.time() - t0:.1f}s)", flush=True)
        except Exception as e:  # noqa: BLE001
            RESULTS.append({"test": fn.__name__, "ok": False, "error": f"{type(e).__name__}: {e}"[:500]})
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
    n_ok = sum(r["ok"] for r in RESULTS)
    print(f"{n_ok}/{len(RESULTS)} passed")
    if a.write:
        Path(a.write).write_text(json.dumps({"passed": n_ok, "n": len(RESULTS), "results": RESULTS}, indent=1))
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
