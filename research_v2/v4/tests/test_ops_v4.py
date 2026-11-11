"""CPU harness of the v4 cloud code paths (PROTOCOL_V4.md sections 10-12): authorization, the cost ledger (rate, pod start, auto-stop window,
storage, billing lines, recovery, ceiling check before a problem, watchdog), the GPU ledger, the pod-start reader, the secret scan (plain, gzip,
tar, tar.gz, npz, literal value) and the redaction filter, checkpoint packing and verification, the remote command filter, the driver's permission
and order rules, exactly-once records, resume after a crash (torn line, partial problem, G8 after restart), G7 snapshot checks and the G6 design.
r4: no double load at the three reload sites of gates(), the exact SKU check, the deterministic settings, the GM environment record, the image
digest record and the lock amendment path. No GPU, no network.

  venv/Scripts/python.exe research_v2/v4/tests/test_ops_v4.py [--write <json>]
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import io
import json
import os
import sys
import tarfile
import tempfile
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
V4 = HERE.parent
RV = V4.parent
sys.path.insert(0, str(RV))
sys.path.insert(0, str(V4))
sys.path.insert(0, str(RV / "tools"))

import run_v4 as R  # noqa: E402  (HF before torch)

import gc  # noqa: E402
import subprocess  # noqa: E402
import weakref  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import lock_v4 as LK  # noqa: E402
import v4_gpu_mutation_check as GMC  # noqa: E402
import v4_ops as O  # noqa: E402
import v4_remote as REM  # noqa: E402
import v4_secret_scan as SS  # noqa: E402
import v4_v3_records as V3R  # noqa: E402

TESTS, RESULTS = [], []
FAKE_TOKEN = "".join(["hf", "_", "AbCdEfGhIjKlMnOpQrStUvWxYz012345"])
FAKE_HF2 = "".join(["hf", "_", "1234567890abcdefghij"])
FAKE_RP = "".join(["rpa", "_", "ZYXWVUTSRQPONMLKJIHGFEDCBA9876543210"])
CFG = {"rate_usd_per_h": 0.59, "autostop_minutes": 30, "volume_gb": 100, "volume_created_utc": "2026-11-10T00:00:00Z"}
T0 = O.parse_utc("2026-11-10T00:00:00Z")


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
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


# ----------------------------------------------------------------------------- authorization
@t
def o01_authorization_files():
    with tmpdir() as d:
        assert not O.check_auth(d, "qwen2.5-0.5b", "abcd1234abcd1234")[0]
        O.auth_path(d, "llama3.2-1b-instruct").write_text("PROTOCOL_V4 abcd1234abcd1234 llama3.2-1b-instruct\n")
        assert O.check_auth(d, "llama3.2-1b-instruct", "abcd1234abcd1234")[0]
        O.auth_path(d, "llama3.2-1b").write_text("PROTOCOL_V4 abcd1234abcd1234 llama3.2-1b-instruct\n")
        assert not O.check_auth(d, "llama3.2-1b", "abcd1234abcd1234")[0], "a file naming the instruct model must not authorize the base model"
        O.auth_path(d, "llama3.2-1b").write_text("PROTOCOL_V4 ffff1234abcd1234 llama3.2-1b\n")
        assert not O.check_auth(d, "llama3.2-1b", "abcd1234abcd1234")[0], "wrong lock id"
        O.auth_path(d, "llama3.2-1b").write_text("PROTOCOL_V3 abcd1234abcd1234 llama3.2-1b\n")
        assert not O.check_auth(d, "llama3.2-1b", "abcd1234abcd1234")[0], "wrong protocol"
        O.auth_path(d, "qwen2.5-1.5b").write_text("PROTOCOL_V4\nlock abcd1234abcd1234\nmodel qwen2.5-1.5b\n")
        assert O.check_auth(d, "qwen2.5-1.5b", "abcd1234abcd1234")[0]
        assert not O.check_auth(d, "qwen2.5-1.5b", None)[0]


# ----------------------------------------------------------------------------- cost ledger
def ledger(d, clk, boot, cfg=CFG):
    return O.CostLedger(d, dict(cfg), now=clk, pod_start=boot, sku="NVIDIA A40")


@t
def o02_cost_estimate_rate_boot_autostop_storage():
    with tmpdir() as d:
        clk = Clock(T0 + 3600)
        c = ledger(d, clk, boot=T0)
        c.open_session("x")
        clk.t = T0 + 2 * 3600
        e = c.estimate()
        pods = (2 * 3600 + 30 * 60) / 3600 * 0.59                 # from POD START (not driver start) plus the auto-stop window
        storage = 2 * 3600 / O.MONTH_S * 100 * 0.07
        assert abs(e["inpod_usd"] - (pods + storage)) < 1e-9, e
        c2 = ledger(d, clk, boot=T0, cfg=dict(CFG, rate_usd_per_h=1.18))
        c2.open = True
        e2 = c2.estimate()
        assert e2["inpod_usd"] > e["inpod_usd"] + 1.0, "the declared rate must be applied"


@t
def o03_sessions_counted_once_and_recovered():
    with tmpdir() as d:
        clk = Clock(T0)
        c = ledger(d, clk, boot=T0)
        c.open_session("a")
        clk.t = T0 + 600
        c.heartbeat()
        # the driver dies: no close line; a second driver in the SAME pod session starts later
        clk.t = T0 + 1200
        c2 = ledger(d, clk, boot=T0)
        c2.open_session("b")
        lines = O.read_jsonl(d / "cost_ledger.jsonl")
        assert [l["kind"] for l in lines] == ["session_open", "recovered_close", "session_open"]
        e = c2.estimate()
        assert e["n_pod_sessions"] == 1
        assert abs(e["inpod_usd"] - ((1200 + 1800) / 3600 * 0.59 + 1200 / O.MONTH_S * 100 * 0.07)) < 1e-9
        c2.close(True)
        # a new pod session (new boot) adds its own auto-stop window
        clk.t = T0 + 7200
        c3 = ledger(d, clk, boot=T0 + 7000)
        c3.open_session("c")
        assert c3.estimate()["n_pod_sessions"] == 2


@t
def o04_billing_lines_take_the_larger():
    with tmpdir() as d:
        clk = Clock(T0)
        c = ledger(d, clk, boot=T0)
        c.open_session("a")
        clk.t = T0 + 3600
        O.add_billing_line(d, 50.0, O.utc(T0 + 1800))
        e = c.estimate()
        assert e["billing_based_usd"] > e["inpod_usd"] and e["used_usd"] == e["billing_based_usd"]
        acc = (3600 - 1800 + 1800) / 3600 * 0.59 + 1800 / O.MONTH_S * 100 * 0.07
        assert abs(e["billing_based_usd"] - (50.0 + acc)) < 1e-9
        O.add_billing_line(d, 0.10, O.utc(T0 + 3000))                # the latest billing line counts
        e = c.estimate()
        assert e["used_usd"] == e["inpod_usd"]


@t
def o05_ceiling_check_before_problem_and_watchdog():
    assert O.CEILING_USD == 30.0 and O.CLOSEOUT_RESERVE_USD == 1.0 and O.stop_usd() == 29.0, "r5: $30 in total, with $1 kept for the close-out"
    with tmpdir() as d:
        clk = Clock(T0)
        c = ledger(d, clk, boot=T0)
        c.open_session("a")
        O.add_billing_line(d, 29.0, O.utc(T0))
        ok, _ = c.can_start(projected_seconds=60.0)
        assert not ok, "the run stops at the stop level, below the ceiling, so the close-out stays paid for"
        O.add_billing_line(d, 28.0, O.utc(T0))
        ok, _ = c.can_start(projected_seconds=60.0)
        assert ok
        ok, _ = c.can_start(projected_seconds=3 * 3600.0)
        assert not ok, "a problem whose projected cost crosses the ceiling must be refused BEFORE it starts"
        fired = []
        g = O.GpuLedger(d, now=clk)
        led = O.Ledgers(g, c, "blk", on_expire=lambda: fired.append(1), period=3600)
        led.gpu.start("blk")
        led.cost = c
        O.add_billing_line(d, 28.9, O.utc(T0))
        led.tick()
        assert fired == [1]
        gl = O.read_jsonl(d / "gpu_ledger.jsonl")
        cl = O.read_jsonl(d / "cost_ledger.jsonl")
        assert gl[-1].get("watchdog_fired") and cl[-1]["kind"] == "session_close" and cl[-1].get("watchdog_fired")


@t
def o06_pod_start_from_pid1_not_uptime():
    with tmpdir() as d:
        (d / "1").mkdir()
        tck = 100
        fields = ["S"] + ["0"] * 18 + [str(500 * tck)] + ["0"] * 10
        (d / "1" / "stat").write_text("1 (bash) " + " ".join(fields))
        (d / "stat").write_text("cpu 1 2 3\nbtime 1000000\n")
        (d / "uptime").write_text("999999.0 1.0\n")
        assert abs(O.pod_start_epoch(str(d), clk_tck=tck) - (1000000 + 500)) < 1e-6    # container PID 1 start, NOT the host boot
        (d / "1" / "stat").unlink()
        now = 2_000_000.0
        assert abs(O.pod_start_epoch(str(d), now=lambda: now) - (now - 999999.0)) < 1e-6


@t
def o07_gpu_ledger_recovery():
    with tmpdir() as d:
        clk = Clock(T0)
        g = O.GpuLedger(d, now=clk)
        g.start("a")
        clk.t = T0 + 100
        g.heartbeat()
        g2 = O.GpuLedger(d, now=clk)
        assert abs(g2.recover() - 100) < 1e-9 and abs(g2.seconds() - 100) < 1e-9
        g2.start("b")
        clk.t = T0 + 150
        g2.close(True)
        assert abs(O.GpuLedger(d).seconds() - 150) < 1e-9


# ----------------------------------------------------------------------------- secrets
@t
def o08_secret_scan_formats():
    with tmpdir() as d:
        lit = [b"my-literal-secret-value-123"]
        (d / "clean.txt").write_text("nothing here hf_short rpa_x")
        (d / "plain.log").write_text(f"token={FAKE_TOKEN}\n")
        (d / "rp.log").write_text(f"key {FAKE_RP}\n")
        (d / "c.log.gz").write_bytes(gzip.compress(f"x {FAKE_TOKEN} y".encode()))
        np.savez_compressed(d / "a.npz", note=np.frombuffer(f"zz{FAKE_TOKEN}".encode(), dtype=np.uint8))
        (d / "lit.txt").write_bytes(b"... my-literal-secret-value-123 ...")
        tb = io.BytesIO()
        with tarfile.open(fileobj=tb, mode="w") as tf:
            for n in ("c.log.gz", "a.npz"):
                tf.add(d / n, arcname=f"inner/{n}")
        (d / "t.tar").write_bytes(tb.getvalue())
        (d / "t.tar.gz").write_bytes(gzip.compress(tb.getvalue()))
        res = {p.name: SS.scan_path(p, lit) for p in sorted(d.iterdir())}
        assert not res["clean.txt"]
        for n in ("plain.log", "rp.log", "c.log.gz", "a.npz", "lit.txt", "t.tar", "t.tar.gz"):
            assert res[n], n
        assert any("gunzip" in w for w, _, _ in res["t.tar.gz"])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            SS.main([str(d / "plain.log")])
        assert FAKE_TOKEN not in out.getvalue() and "value withheld" in out.getvalue()


@t
def o09_redaction_filter():
    buf = io.StringIO()
    s = O.RedactingStream(buf, literals=[b"literal-secret-xyz"])
    s.write(f"a {FAKE_TOKEN} b {FAKE_RP} c literal-secret-xyz d\n")
    v = buf.getvalue()
    assert FAKE_TOKEN not in v and FAKE_RP not in v and "literal-secret-xyz" not in v and "[REDACTED" in v
    assert O.secret_literals({"HF_TOKEN": FAKE_HF2, "X": "y"}) == [FAKE_HF2.encode()]


@t
def o20_pid1_environment_secrets():
    """Pod variables reach PID 1 only: HF_TOKEN is read from /proc/1/environ, and its literal value is redacted and scanned for."""
    with tmpdir() as d:
        (d / "1").mkdir()
        (d / "1" / "environ").write_bytes(b"PATH=/usr/bin\0HF_TOKEN=pid1-only-secret-value-42\0RUNPOD_POD_ID=abc\0")
        assert O.pid1_environ(str(d))["HF_TOKEN"] == "pid1-only-secret-value-42"
        saved = os.environ.pop("HF_TOKEN", None)
        try:
            assert O.pod_env("HF_TOKEN", str(d)) == "pid1-only-secret-value-42"
            lits = O.secret_literals(proc=str(d))
            assert b"pid1-only-secret-value-42" in lits
            assert "pid1-only-secret-value-42" not in O.redact("x pid1-only-secret-value-42 y", lits)
            assert O.find_secrets(b"".join([b"rps", b"_", b"A" * 30]))
        finally:
            if saved is not None:
                os.environ["HF_TOKEN"] = saved


@t
def o10_token_file_check():
    with tmpdir() as d:
        assert O.token_files(str(d / "hf"), str(d)) == []
        (d / "hf").mkdir()
        (d / "hf" / "token").write_text("x")
        assert O.token_files(str(d / "hf"), str(d)) == [str(d / "hf" / "token")]


# ----------------------------------------------------------------------------- checkpoints and the remote tool
@t
def o11_checkpoint_pack_and_verify():
    with tmpdir() as d:
        (d / "m1" / "test").mkdir(parents=True)
        (d / "m1" / "test" / "NAT.jsonl").write_text('{"index": 1}\n')
        (d / "gpu_ledger.jsonl").write_text('{"seconds": 1}\n')
        (d / "AUTHORIZED_V4_m1.txt").write_text("PROTOCOL_V4 x m1")
        tar = O.pack_checkpoint(d, "m1")
        sumf = Path(str(tar) + ".sha256")
        assert sumf.read_bytes().endswith(b"\n") and b"\r" not in sumf.read_bytes()
        assert O.verify_sha256_file(sumf)
        sumf.write_bytes(sumf.read_bytes().replace(b"\n", b"\r\n"))
        assert O.verify_sha256_file(sumf), "CRLF must be tolerated on read"
        with tarfile.open(tar) as tf:
            names = tf.getnames()
            sums = tf.extractfile("SHA256SUMS").read().decode()
        assert "m1/test/NAT.jsonl" in names and "gpu_ledger.jsonl" in names and "m1/test/NAT.jsonl" in sums
        with open(tar, "r+b") as f:
            f.seek(600)
            f.write(b"X")
        assert not O.verify_sha256_file(sumf)
        assert not SS.scan_path(d / "m1")


@t
def o12_remote_command_filter():
    for bad in ("runpodctl stop pod", "curl https://api.runpod.io/graphql", "echo PROTOCOL_V4 > /workspace/v4_runs/AUTHORIZED_V4_x.txt",
                "cp a /workspace/v4_runs/AUTHORIZED_V4_GM.txt", "huggingface-cli login", "touch AUTHORIZED_V4_m.txt"):
        assert not REM.command_allowed(bad)[0], bad
    for good in ("python research_v2/v4/run_v4.py status", "cat /workspace/v4_runs/AUTHORIZED_V4_GM.txt", "nvidia-smi"):
        assert REM.command_allowed(good)[0], good


# ----------------------------------------------------------------------------- driver rules
def set_determinism(on):
    """Directly, not through the code under test, so a mutated apply_determinism cannot leak into later tests."""
    if on:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    else:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    torch.use_deterministic_algorithms(on)


@contextlib.contextmanager
def driver(d, gpu="NVIDIA A40", lid="abcd1234abcd1234", verifies=True):
    saved = (R.OUT, R.lock_id, R.lock_verifies, O.gpu_name, O.token_files, R.LOCK)
    set_determinism(True)
    R.OUT = d
    R.lock_id = lambda: lid
    R.lock_verifies = lambda: verifies
    O.gpu_name = lambda: gpu
    O.token_files = lambda *a, **k: []
    R.LOCK = d / "LOCK.txt"
    try:
        yield
    finally:
        R.OUT, R.lock_id, R.lock_verifies, O.gpu_name, O.token_files, R.LOCK = saved


class Args:
    authorized_gpu_run = True


def refused(fn):
    try:
        fn()
    except SystemExit as e:
        return str(e)
    return None


@t
def o13_permission_rules():
    with tmpdir() as d, driver(d):
        (d / "cost_config.json").write_text(json.dumps(CFG))
        a = Args()
        assert "AUTHORIZED" in refused(lambda: R.require(a, "qwen2.5-0.5b", "gates"))
        O.auth_path(d, "qwen2.5-0.5b").write_text("PROTOCOL_V4 abcd1234abcd1234 qwen2.5-0.5b")
        assert "mutation" in refused(lambda: R.require(a, "qwen2.5-0.5b", "gates"))
        b = Args()
        b.authorized_gpu_run = False
        assert "--authorized-gpu-run" in refused(lambda: R.require(b, "qwen2.5-0.5b", "gates"))
    with tmpdir() as d, driver(d, gpu="NVIDIA L40S"):
        (d / "cost_config.json").write_text(json.dumps(CFG))
        O.auth_path(d, "qwen2.5-0.5b").write_text("PROTOCOL_V4 abcd1234abcd1234 qwen2.5-0.5b")
        assert "D5" in refused(lambda: R.require(Args(), "qwen2.5-0.5b", "gates"))
    with tmpdir() as d, driver(d, verifies=False):
        (d / "cost_config.json").write_text(json.dumps(CFG))
        O.auth_path(d, "qwen2.5-0.5b").write_text("PROTOCOL_V4 abcd1234abcd1234 qwen2.5-0.5b")
        assert "lock" in refused(lambda: R.require(Args(), "qwen2.5-0.5b", "gates"))
    with tmpdir() as d, driver(d):
        O.auth_path(d, "qwen2.5-0.5b").write_text("PROTOCOL_V4 abcd1234abcd1234 qwen2.5-0.5b")
        assert "cost_config" in refused(lambda: R.require(Args(), "qwen2.5-0.5b", "gates"))
        (d / "cost_config.json").write_text(json.dumps(CFG))
        O.add_billing_line(d, 29.5, "2026-11-10T00:00:00Z")
        assert "ceiling" in refused(lambda: R.require(Args(), "qwen2.5-0.5b", "gates"))
    with tmpdir() as d, driver(d):
        (d / "cost_config.json").write_text(json.dumps(CFG))
        O.auth_path(d, "qwen2.5-0.5b").write_text("PROTOCOL_V4 abcd1234abcd1234 qwen2.5-0.5b")
        saved = O.token_files
        O.token_files = lambda *a, **k: ["/workspace/hf/token"]
        try:
            assert "token" in refused(lambda: R.require(Args(), "qwen2.5-0.5b", "gates"))
        finally:
            O.token_files = saved


def set_gm_locked(d):
    (d / "gm.json").write_text(json.dumps({"all_passed": True}))
    import hashlib
    R.LOCK.write_text("x\nGM: " + hashlib.sha256((d / "gm.json").read_bytes()).hexdigest() + "  gm.json\n")


@t
def o14_order_rules():
    with tmpdir() as d, driver(d):
        set_gm_locked(d)
        for m in R.ORDER:
            (d / m).mkdir()
            (d / m / "download.json").write_text("{}")
        assert "order" in refused(lambda: R.order_rules("llama3.2-1b", "gates"))
        for m in R.STEP0:
            (d / m / "test_complete.json").write_text("{}")
        assert "Step 0" in refused(lambda: R.order_rules("qwen2.5-1.5b", "gates"))
        import hashlib
        (d / "step0.json").write_text(json.dumps({"decision": "fail: stop and report"}))
        R.LOCK.write_text(R.LOCK.read_text() + "STEP0: " + hashlib.sha256((d / "step0.json").read_bytes()).hexdigest() + "\n")
        assert "Step 0" in refused(lambda: R.order_rules("qwen2.5-1.5b", "gates"))
        (d / "step0.json").write_text(json.dumps({"decision": "author: proceed"}))
        R.LOCK.write_text(R.LOCK.read_text() + "STEP0: " + hashlib.sha256((d / "step0.json").read_bytes()).hexdigest() + "\n")
        assert "design" in refused(lambda: R.order_rules("qwen2.5-1.5b", "gates"))
        (d / "design.json").write_text(json.dumps({"drop_mistral": False, "n_test": 100, "n_cal": 50}))
        R.LOCK.write_text(R.LOCK.read_text() + "DESIGN: " + hashlib.sha256((d / "design.json").read_bytes()).hexdigest() + "\n")
        assert refused(lambda: R.order_rules("qwen2.5-1.5b", "gates")) is None
        assert "gates" in refused(lambda: R.order_rules("qwen2.5-1.5b", "calibrate"))
        (d / "qwen2.5-1.5b" / "precision.json").write_text(json.dumps({"precision": "bf16"}))
        (d / "qwen2.5-1.5b" / "gates").mkdir()
        (d / "qwen2.5-1.5b" / "gates" / "gates.json").write_text(json.dumps({"all_passed": True}))
        assert "flag" in refused(lambda: R.order_rules("qwen2.5-1.5b", "calibrate"))
        R.LOCK.write_text(R.LOCK.read_text() + "FLAG qwen2.5-1.5b: " + hashlib.sha256((d / "qwen2.5-1.5b" / "precision.json").read_bytes()).hexdigest() + "\n")
        assert refused(lambda: R.order_rules("qwen2.5-1.5b", "calibrate")) is None
        assert "calibration" in refused(lambda: R.order_rules("qwen2.5-1.5b", "test"))


@t
def o15_exactly_once_and_resume():
    with tmpdir() as d, driver(d):
        tag = "m"
        td = d / tag / "test"
        recs = {a: {"index": 7, "arm": a, "js": [0.0, 0.1]} for _, _, a in R.TEST_ARMS}
        R.E.append_records(td, {"NAT": recs["NAT"], "C1": recs["C1"]})          # a crash after two arm files were written
        assert not R.test_done(tag, 7)
        R.append_new(td, 7, recs)
        R.append_new(td, 7, recs)
        for _, _, a in R.TEST_ARMS:
            lines = [l for l in (td / f"{a}.jsonl").read_text().splitlines() if l.strip()]
            assert len(lines) == 1, a
        with open(td / "NAT.jsonl", "ab") as f:
            f.write(b'{"index": 8, "js": [0.0')
        assert R.torn_record_files(tag) == [td / "NAT.jsonl"]
        assert "repair" in refused(lambda: R.require_intact(tag))
        import v3_repair_torn_line as RP
        kept, removed = RP.plan_repair((td / "NAT.jsonl").read_bytes())
        (td / "NAT.jsonl").write_bytes(kept)
        assert R.torn_record_files(tag) == []


@t
def o16_g8_canonical_comparison():
    a = {"NAT": {"index": 1, "js": [0.0, 0.25], "wall_seconds": 1.0, "_slots": {"S1K": np.ones((2, 2, 4))}}}
    b = {"NAT": {"index": 1, "js": [0.0, 0.25], "wall_seconds": 9.0, "_slots": {"S1K": np.ones((2, 2, 4))}}}
    assert R.canonical(a) == R.canonical(b)
    c = {"NAT": {"index": 1, "js": [0.0, float(np.nextafter(0.25, 1))], "wall_seconds": 1.0, "_slots": {"S1K": np.ones((2, 2, 4))}}}
    assert R.canonical(a) != R.canonical(c)
    s = np.ones((2, 2, 4))
    s[0, 0, 0] = np.nextafter(1.0, 2)
    e = {"NAT": {"index": 1, "js": [0.0, 0.25], "wall_seconds": 1.0, "_slots": {"S1K": s}}}
    assert R.canonical(a) != R.canonical(e)


@t
def o17_g7_snapshot_checks():
    with tmpdir() as d:
        rev = "a" * 40
        snap = d / rev
        snap.mkdir()
        (snap / "model.safetensors").write_bytes(b"weights")
        e = {"revision": rev, "shards": {"model.safetensors": {"sha256": O.sha256_file(snap / "model.safetensors")}}, "excluded_files": ["consolidated.safetensors"]}
        assert R.check_snapshot(e, snap)["pass"]
        other = d / ("b" * 40)
        other.mkdir()
        (other / "model.safetensors").write_bytes(b"weights")
        assert not R.check_snapshot(e, other)["pass"], "an unpinned revision must fail G7"
        (snap / "model.safetensors").write_bytes(b"weightz")
        assert not R.check_snapshot(e, snap)["pass"]
        (snap / "model.safetensors").write_bytes(b"weights")
        (snap / "consolidated.safetensors").write_bytes(b"dup")
        assert not R.check_snapshot(e, snap)["pass"]


@t
def o18_block_loop_ceiling_and_drift():
    with tmpdir() as d, driver(d):
        (d / "m").mkdir()

        class Cost:
            def __init__(self, ok):
                self.ok = ok

            def can_start(self, s):
                return self.ok, {"used_usd": 29.9}
        ran = []
        assert not R.block_loop("x", "m", [1, 2], lambda i: ran.append(i) or True, lambda i: False, Cost(False))
        assert ran == [] and (d / "m" / "ceiling_stop.json").exists()
        import stage1_runtime as SR
        saved = SR.telemetry_now
        SR.telemetry_now = lambda: {}
        try:
            def slow(i):
                time.sleep(0.02)
                return True
            ok = R.block_loop("x", "m", list(range(12)), slow, lambda i: False, Cost(True), expected_problem_s=0.001)
            assert not ok and (d / "m" / "drift_stop.json").exists()
        finally:
            SR.telemetry_now = saved


@t
def o19_g6_fit_and_reduction_order():
    meas = {"model": "qwen2.5-0.5b", "layers": 24, "t_arm_problem_s": 7.0, "t_hist_ref_problem_s": 2.0}
    probe = {"model": "llama3.1-8b", "layers": 32, "t_arm_problem_s": 12.5, "t_hist_ref_problem_s": 3.0}
    a, b, h = R.fit_runtime(meas, probe)
    pm = R.plan_numbers()["projection"][O.SKU]["models"]
    for x in (meas, probe):
        assert abs(a * x["layers"] + b * R.weight_bytes(pm[x["model"]]) - x["t_arm_problem_s"]) < 1e-9
    no_mistral = [m for m in R.LADDER_NEW if m != R.DROP_FIRST]
    tot_all, rows_all = R.project(a, b, h, 0.59, R.LADDER_NEW)
    tot_no, rows_no = R.project(a, b, h, 0.59, no_mistral)
    assert tot_all > tot_no > 0 and R.DROP_FIRST in rows_all and R.DROP_FIRST not in rows_no
    r = rows_no["llama3.2-3b"]
    t_arm = r["t_arm_s"]
    want_h = (103 * r["expected_problem_s"] + 50 * (4 * t_arm + h * t_arm)) / 3600 + 0.35
    assert abs(r["gpu_h"] - want_h) < 1e-9 and abs(r["usd"] - (want_h * 1.2 + 0.25) * 1.05 * 0.59) < 1e-9

    def run(used):
        with tmpdir() as d, driver(d):
            (d / "g6_measure.json").write_text(json.dumps(meas))
            (d / "g6_probe.json").write_text(json.dumps(probe))
            (d / "cost_config.json").write_text(json.dumps(dict(CFG, volume_created_utc=O.utc(time.time()))))
            O.add_billing_line(d, used, O.utc(time.time()))
            with contextlib.redirect_stdout(io.StringIO()):
                ok = R.g6_design({})
            name = "design.json" if ok else "design_failed.json"
            return ok, json.loads((d / name).read_text()), (d / R.DROP_FIRST / "not_run.json").exists()
    store_all = O.storage_allowance(100, sum(x["billed_h"] for x in rows_all.values()))
    store_no = O.storage_allowance(100, sum(x["billed_h"] for x in rows_no.values()))
    ok, dz, nr = run(O.stop_usd() - store_all - tot_all - 0.5)                       # everything fits: nothing changes
    assert ok and not dz["drop_mistral"] and dz["models"] == R.LADDER_NEW and dz["n_test"] == 100 and dz["n_cal"] == 50 and not nr
    ok, dz, nr = run(O.stop_usd() - store_no - (tot_no + tot_all) / 2)                # first reduction: Mistral goes, and nothing else
    assert ok and dz["drop_mistral"] and dz["models"] == no_mistral and dz["n_test"] == 100 and dz["n_cal"] == 50 and nr
    ok, dz, nr = run(O.stop_usd() - store_no - tot_no + 0.5)                          # just over without Mistral: stop and report; no other automatic cut
    assert not ok and dz["n_test"] == 100 and dz["n_cal"] == 50 and "stop and report" in dz["stop"]
    assert [s["dropped"] for s in dz["projection_steps"]] == [None, R.DROP_FIRST]


# ----------------------------------------------------------------------------- r4: no double load at the reload sites of gates()
class FakeModel:
    def __init__(self):
        self.me = self                                               # a cycle, as in a real model: only the garbage collector releases it
        self.config = object()


class FakeBackend:
    def __init__(self):
        self.model = FakeModel()


@contextlib.contextmanager
def gates_fakes(d):
    """The real gates() with the model, tokenizer and gate checks replaced. Each load records whether every earlier model was already released;
    the loader itself never collects garbage, so only the code under test can release them."""
    log = {"loads": 0, "double_loads": 0}
    refs = []

    def loader(*a, **k):
        log["loads"] += 1
        log["double_loads"] += sum(r() is not None for r in refs)
        be = FakeBackend()
        refs.append(weakref.ref(be.model))
        return be, object()
    patches = [(R, "snapshot_dir", lambda e: d), (R, "check_snapshot", lambda e, s: {"pass": True}), (R, "load", loader), (R.rv4, "load_model", loader),
               (R.rv4, "config_snapshot", lambda c: {}), (R.rv4, "geometry_matches", lambda s, g: False), (R, "g5", lambda *a: {"pass": False}),
               (R, "g3", lambda *a: (False, ["x"], []))]
    saved = [(o, n, getattr(o, n)) for o, n, _ in patches]
    for o, n, v in patches:
        setattr(o, n, v)
    gc.collect()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            yield log
    finally:
        for o, n, v in saved:
            setattr(o, n, v)


@t
def o22_bf16_reload_does_not_double_load():
    """The one reload left after r5: the FP16 model -> the BF16 model after a G3 failure."""
    with tmpdir() as d, driver(d), gates_fakes(d) as log:
        (d / "qwen2.5-0.5b").mkdir()
        R.gates("qwen2.5-0.5b", None, None)
        assert log["loads"] == 2, log
        assert log["double_loads"] == 0, log
        assert json.loads((d / "qwen2.5-0.5b" / "precision.json").read_text())["precision"] == "bf16"


# ----------------------------------------------------------------------------- r4: SKU, determinism, GM environment, image digest
@t
def o23_sku_check_is_exact():
    assert O.SKU == "A40" and O.sku_ok("NVIDIA A40") and O.sku_ok("A40") and O.sku_ok(" NVIDIA A40\n")
    for name in ("NVIDIA RTX A6000", "NVIDIA RTX A4000", "NVIDIA RTX A400", "NVIDIA RTX A4000 SFF Ada Generation", "NVIDIA A40-48Q", "GRID A40-8Q", "NVIDIA L40S", "", None):
        assert not O.sku_ok(name), name
    with tmpdir() as d, driver(d, gpu="NVIDIA RTX A4000"):
        (d / "cost_config.json").write_text(json.dumps(CFG))
        O.auth_path(d, "qwen2.5-0.5b").write_text("PROTOCOL_V4 abcd1234abcd1234 qwen2.5-0.5b")
        assert "D5" in refused(lambda: R.require(Args(), "qwen2.5-0.5b", "gates"))


@t
def o24_deterministic_settings_applied_and_required():
    try:
        set_determinism(False)
        assert not R.rv4.determinism_ok()
        state = R.rv4.apply_determinism()
        assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8" and torch.are_deterministic_algorithms_enabled()
        assert state == {"cublas_workspace_config": ":4096:8", "deterministic_algorithms": True} and R.rv4.determinism_ok()
        with tmpdir() as d, driver(d):
            torch.use_deterministic_algorithms(False)
            assert "deterministic" in refused(lambda: R.require(Args(), "qwen2.5-0.5b", "gates"))
            set_determinism(True)
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
            assert "deterministic" in refused(lambda: R.require(Args(), "qwen2.5-0.5b", "gates"))
    finally:
        set_determinism(True)
    for path, first_torch_import in ((V4 / "run_v4.py", "import run_controls"), (V4 / "rv4.py", "from transformers"), (V4 / "tests" / "test_rv4.py", "from transformers")):
        src = path.read_text(encoding="utf-8")
        assert -1 < src.index('os.environ["CUBLAS_WORKSPACE_CONFIG"]') < src.index(first_torch_import), f"{path.name}: the variable must be set before torch loads"
    assert 'rv4.apply_determinism()' in (V4 / "run_v4.py").read_text(encoding="utf-8").split("def main():")[1], "main() must apply the settings"


class Smi:
    def __init__(self, query="NVIDIA A40, 570.124.06, Enabled, 46068 MiB\n", banner="| NVIDIA-SMI 570.124.06  Driver Version: 570.124.06  CUDA Version: 12.8 |\n", fail=False):
        self.query, self.banner, self.fail = query, banner, fail

    def __call__(self, argv, **k):
        if self.fail:
            raise OSError("no nvidia-smi")

        class Done:
            returncode = 0
            stdout = self.query if len(argv) > 1 else self.banner
        return Done


@t
def o25_gm_environment_logged_and_gated():
    env = O.gpu_environment(Smi())
    assert env == {"name": "NVIDIA A40", "driver_version": "570.124.06", "ecc_mode": "Enabled", "memory_total": "46068 MiB", "cuda_version": "12.8"}, env
    assert O.gpu_environment_complete(env)
    assert O.gpu_environment_complete(O.gpu_environment(Smi(query="NVIDIA A40, 570.124.06, [N/A], 46068 MiB\n")))
    assert not O.gpu_environment_complete(O.gpu_environment(Smi(fail=True)))
    assert not O.gpu_environment_complete(O.gpu_environment(Smi(banner="no version here\n")))
    assert not O.gpu_environment_complete(O.gpu_environment(Smi(query="NVIDIA A40\n")))
    good = dict(r1=0, r2=0, t={"passed": 3, "n": 3}, m={"all_caught": True}, env_complete=True, determinism_ok=True, image={"digest": "x"}, v3_records_ok=True)
    assert GMC.verdict(**good) is True
    for bad in (dict(r1=1), dict(r2=1), dict(t={"passed": 2, "n": 3}), dict(m={"all_caught": False}), dict(env_complete=False), dict(determinism_ok=False), dict(image=None), dict(v3_records_ok=False)):
        assert GMC.verdict(**{**good, **bad}) is False, bad


class Url:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self.body


@t
def o26_image_digest_recorded_once():
    d1, d2 = "sha256:" + "a" * 64, "sha256:" + "b" * 64
    assert O.fetch_image_digest(opener=lambda u, timeout: Url(json.dumps({"digest": d1, "name": O.IMAGE_TAG}).encode())) == d1
    for body in (b'{"name": "x"}', b'{"digest": "latest"}'):
        try:
            O.fetch_image_digest(opener=lambda u, timeout, b=body: Url(b))
            raise AssertionError("an invalid digest must be refused")
        except ValueError:
            pass
    with tmpdir() as d:
        assert O.image_record(d) is None
        rec = O.record_image_digest(d, fetch=lambda: d1, now=lambda: T0)
        assert (rec["image"], rec["tag"], rec["digest"]) == (O.IMAGE_REPO, O.IMAGE_TAG, d1) and O.image_record(d) == rec
        with contextlib.redirect_stdout(io.StringIO()):
            again = O.record_image_digest(d, fetch=lambda: d2, now=lambda: T0 + 60)
        assert again["digest"] == d1 and json.loads((d / "image_digest.json").read_text())["digest"] == d1, "the first-boot record is never replaced"
        assert O.read_jsonl(d / "image_digest_changes.jsonl")[0]["read"] == d2
        (d / "image_digest.json").write_text(json.dumps(dict(rec, tag="latest")))
        assert O.image_record(d) is None, "a record of another tag is not the locked image"
    with tmpdir() as d, driver(d):
        (d / "cost_config.json").write_text(json.dumps(CFG))
        O.auth_path(d, "GM").write_text("PROTOCOL_V4 abcd1234abcd1234 GM")
        assert "first-boot" in refused(lambda: R.require(Args(), None, "gm", tag="GM"))
        O.record_image_digest(d, fetch=lambda: d1)
        assert refused(lambda: R.require(Args(), None, "gm", tag="GM")) is None


@t
def o27_cloud_files_name_the_study_gpu_and_image():
    setup = (V4 / "cloud" / "pod_setup.sh").read_text(encoding="utf-8")
    book = (V4 / "cloud" / "RUNBOOK_V4.md").read_text(encoding="utf-8")
    assert '"NVIDIA A40"|"A40") ;;' in setup and "A6000" not in setup and "record_image_digest" in setup
    assert "A6000" not in book and "1 × A40" in book and f"{O.IMAGE_REPO}:{O.IMAGE_TAG}" in book
    plan = R.plan_numbers()
    assert plan["study_gpu"] == O.SKU and O.SKU in plan["projection"]
    assert "cu126" in setup and '"torch==2.13.0"' in setup, "torch stays 2.13.0+cu126 (r4)"
    assert "**$30**" in book and "100 GB" in book and "closeout" in book and "$150" not in book and "150 GB" not in book
    b = plan["budget_r5"]
    assert b["ceiling_usd"] == O.CEILING_USD and b["stop_usd"] == O.stop_usd() and b["volume_gb"] == 100 and b["all_models"]["total_usd"] < O.stop_usd()


# ----------------------------------------------------------------------------- r4: the lock amendment path, in a throwaway git repository
@contextlib.contextmanager
def lock_repo():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        root = Path(td)
        rv, v4 = root / "research_v2", root / "research_v2" / "v4"
        v4.mkdir(parents=True)
        (rv / "PREV.txt").write_text("previous lock\n")
        (v4 / "V4_MODEL_MANIFEST.json").write_text(json.dumps({"models": {"m": {"hf_id": "o/m", "revision": "r", "shards": {"s.safetensors": {"sha256": "c" * 64}}}}}))
        (v4 / "PROTOCOL.md").write_text("r3\n")
        (v4 / "DEVIATION_LOG_V4.md").write_text("# log\n")
        run = lambda *a: subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false", *a],  # noqa: E731
                                        capture_output=True, check=True)
        run("init", "-q")
        run("config", "core.autocrlf", "false")
        saved = {k: getattr(LK, k) for k in ("ROOT", "RV", "V4", "LOCK", "OUT", "HISTORY", "DEVIATION_LOG", "PREV_LOCKS", "locked_files")}
        LK.ROOT, LK.RV, LK.V4, LK.LOCK, LK.OUT = root, rv, v4, rv / "V4_PROTOCOL_LOCK.txt", rv / "v4_runs"
        LK.HISTORY, LK.DEVIATION_LOG, LK.PREV_LOCKS = v4 / "lock_history", v4 / "DEVIATION_LOG_V4.md", [rv / "PREV.txt"]
        LK.locked_files = lambda: [v4 / "PROTOCOL.md", v4 / "V4_MODEL_MANIFEST.json"]
        run("add", "-A")
        run("commit", "-qm", "first")

        def commit(msg):
            run("add", "-A")
            run("commit", "-qm", msg)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                LK.create()
            commit("lock")
            yield root, v4, commit
        finally:
            for k, v in saved.items():
                setattr(LK, k, v)


def quiet(fn, *a):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a)


@t
def o28_lock_amendment_preserves_and_links():
    with lock_repo() as (root, v4, commit):
        old_id, old_bytes = LK.lock_id(), LK.LOCK.read_bytes()
        assert quiet(LK.verify) == 0
        assert "--amend" in refused(LK.create)
        (v4 / "PROTOCOL.md").write_text("r4\n")
        commit("r4")
        assert quiet(LK.verify) == 1, "the old lock no longer matches the working tree"
        quiet(LK.amend, "r4", "D5 GPU is now the A40")
        new_id = LK.lock_id()
        kept = LK.HISTORY / f"V4_PROTOCOL_LOCK_{old_id}.txt"
        assert new_id != old_id and kept.read_bytes() == old_bytes and LK.lock_id(kept) == old_id
        text = LK.LOCK.read_text(encoding="utf-8")
        line = next(l for l in text.splitlines() if l.startswith("AMENDMENT "))
        assert LK.AMENDMENT_RX.fullmatch(line) and f"from lock {old_id}" in line and "D5 GPU is now the A40" in line and line in LK.base_text()
        assert f"from lock {old_id} to lock {new_id}" in LK.DEVIATION_LOG.read_text(encoding="utf-8")
        assert quiet(LK.verify) == 0
        with tmpdir() as d:
            O.auth_path(d, "GM").write_text(f"PROTOCOL_V4 {old_id} GM")
            assert not O.check_auth(d, "GM", new_id)[0], "an authorization file of the old lock must not authorize the new one"
        assert "already" in refused(lambda: quiet(LK.amend, "r4", "again"))
        commit("re-lock")
        (v4 / "PROTOCOL.md").write_text("r5\n")
        commit("r5")
        quiet(LK.amend, "r5", "second amendment")
        assert quiet(LK.verify) == 0
        assert sum(l.startswith("AMENDMENT ") for l in LK.LOCK.read_text(encoding="utf-8").splitlines()) == 2, "the chain is carried forward"
        (LK.HISTORY / f"V4_PROTOCOL_LOCK_{old_id}.txt").write_text("changed\n")
        assert quiet(LK.verify) == 1, "the first preserved lock of the chain is still checked"


@t
def o29_lock_amendment_refused_after_data_or_on_a_dirty_tree():
    with lock_repo() as (root, v4, commit):
        before = LK.LOCK.read_bytes()

        def refused_cleanly(why):
            msg = refused(lambda: quiet(LK.amend, "r4", "text"))
            assert msg and why in msg, msg
            assert LK.LOCK.read_bytes() == before and not LK.HISTORY.exists(), "a refusal must leave the old lock in place and write nothing"
        (v4 / "PROTOCOL.md").write_text("uncommitted\n")
        refused_cleanly("uncommitted")
        (v4 / "PROTOCOL.md").write_text("r3\n")
        out = LK.OUT
        out.mkdir()
        (out / "AUTHORIZED_V4_GM.txt").write_text("x")
        (out / "cost_config.json").write_text("{}")
        (out / "gm.json").write_text("{}")
        refused_cleanly("gm.json")
        (out / "gm.json").unlink()
        (out / "m" / "test").mkdir(parents=True)
        refused_cleanly("m/test")
        (out / "m" / "test").rmdir()
        LK.LOCK.write_bytes(before + b"GM: " + b"0" * 64 + b"  gm.json\n")
        msg = refused(lambda: quiet(LK.amend, "r4", "text"))
        assert "GM" in msg
        LK.LOCK.write_bytes(before)
        tampered = before.replace(LK.sha((v4 / "PROTOCOL.md").read_bytes()).encode(), b"0" * 64)
        LK.LOCK.write_bytes(tampered)
        msg = refused(lambda: quiet(LK.amend, "r4", "text"))
        assert "does not match its own commit" in msg, msg
        LK.LOCK.write_bytes(before)


@t
def o30_lock_verify_catches_a_broken_amendment():
    with lock_repo() as (root, v4, commit):
        old_id = LK.lock_id()
        (v4 / "PROTOCOL.md").write_text("r4\n")
        commit("r4")
        quiet(LK.amend, "r4", "text")
        assert quiet(LK.verify) == 0
        log, kept = LK.DEVIATION_LOG, LK.HISTORY / f"V4_PROTOCOL_LOCK_{old_id}.txt"
        good_log, good_kept = log.read_bytes(), kept.read_bytes()
        log.write_text("# log\n")
        assert quiet(LK.verify) == 1, "no deviation row"
        log.write_bytes(good_log)
        kept.write_bytes(good_kept + b"x\n")
        assert quiet(LK.verify) == 1, "changed preserved lock"
        kept.unlink()
        assert quiet(LK.verify) == 1, "missing preserved lock"
        kept.write_bytes(good_kept.replace(b"\n", b"\r\n"))
        assert quiet(LK.verify) == 0, "a CRLF checkout of the preserved lock is the same text"
        kept.write_bytes(good_kept)
        assert quiet(LK.verify) == 0


# ----------------------------------------------------------------------------- r5
@t
def o31_storage_prorated_and_stops_at_deletion():
    day = 86400.0
    with tmpdir() as d:
        c = ledger(d, Clock(T0), boot=T0)
        assert abs(c.storage_usd(T0, T0 + day) - 100 * 0.07 / 30) < 1e-12
        c2 = ledger(d, Clock(T0), boot=T0, cfg=dict(CFG, volume_deleted_utc=O.utc(T0 + 3600)))
        assert abs(c2.storage_usd(T0, T0 + day) - 100 * 0.07 / 720) < 1e-12, "storage stops at deletion"
    assert abs(O.storage_allowance(100, 24.0) - 7.0 * (24 + 48) / 720) < 1e-12, "the allowance is prorated, not a month"


@t
def o32_sample_is_the_first_100_and_50():
    q = json.loads((RV / "e1_sample.json").read_text(encoding="utf-8"))
    t100, c50, g6 = R.sample_ids(q)
    assert t100 == q["tierA_ids"][:100] and len(t100) == 100 and c50 == q["calibration_train_ids"][:50] and len(c50) == 50
    assert g6 == q["calibration_train_ids"][-3:] and not set(g6) & set(c50), "the G6 timing problems are not calibration problems"
    s = json.loads(R.SAMPLE.read_text(encoding="utf-8"))
    assert s["tierA_ids"] == t100 and s["calibration_train_ids"] == c50 and s["g6_train_ids"] == g6 and s["order"] == R.ORDER
    assert set(s["prompt_sha256"]["test"]) == {str(i) for i in t100} and set(s["exposed_test_ids_in_sample"]) <= set(t100)
    assert s["gate_G5_test_ids"] == t100[:8] and s["gate_alpha0_train_ids"] == c50[:10] and s["gate_numerics_train_ids"] == c50[:6]


@t
def o33_seeds_do_not_depend_on_n():
    """The draws of a problem are keyed by its id, so the subset's problems get the same draws they would get in the 200."""
    a = R.rv4.arm_seeds("qwen2.5-0.5b", "C1", 1.0, 172, "V4A")
    assert a == R.rv4.arm_seeds("qwen2.5-0.5b", "C1", 1.0, 172, "V4A") and a != R.rv4.arm_seeds("qwen2.5-0.5b", "C1", 1.0, 811, "V4A")


@t
def o34_mistral_last_and_first_to_drop():
    assert R.ORDER == ["qwen2.5-0.5b", "llama3.2-1b-instruct", "llama3.2-1b", "qwen2.5-1.5b", "llama3.2-3b", "qwen2.5-3b", "qwen2.5-7b", "llama3.1-8b", "mistral-7b-v0.3"]
    assert R.DROP_FIRST == R.ORDER[-1]
    import hashlib
    with tmpdir() as d, driver(d):
        set_gm_locked(d)
        (d / "cost_config.json").write_text(json.dumps(CFG))
        O.auth_path(d, R.DROP_FIRST).write_text(f"PROTOCOL_V4 abcd1234abcd1234 {R.DROP_FIRST}")
        est = {"used_usd": 20.0}
        assert "design" in refused(lambda: R.drop_first_check("download", est))

        def lock_design(dz):
            (d / "design.json").write_text(json.dumps(dz))
            R.LOCK.write_text(R.LOCK.read_text() + "DESIGN: " + hashlib.sha256((d / "design.json").read_bytes()).hexdigest() + "\n")
        lock_design({"drop_mistral": True})
        assert "dropped" in refused(lambda: R.drop_first_check("download", est)) and not (d / R.DROP_FIRST / "not_run.json").exists()
        lock_design({"drop_mistral": False, "per_tag": {R.DROP_FIRST: {"usd": 5.0}}})
        assert refused(lambda: R.drop_first_check("download", {"used_usd": 20.0})) is None, "it still fits"
        assert not (d / R.DROP_FIRST / "not_run.json").exists()
        assert "dropped" in refused(lambda: R.drop_first_check("download", {"used_usd": 24.5})), "it no longer fits at its own start"
        assert (d / R.DROP_FIRST / "not_run.json").exists() and R.not_run(R.DROP_FIRST)
        (d / R.DROP_FIRST / "not_run.json").unlink()
        O.add_billing_line(d, 24.5, O.utc(T0))
        (d / "cost_config.json").write_text(json.dumps(dict(CFG, volume_created_utc=O.utc(T0))))
        assert "dropped" in refused(lambda: R.require(Args(), R.DROP_FIRST, "download")), "require() runs the check"
        (d / R.DROP_FIRST / "not_run.json").unlink()
        (d / R.DROP_FIRST / "download.json").write_text("{}")
        assert "fixed run order" in refused(lambda: R.order_rules(R.DROP_FIRST, "gates")), "Mistral waits for every other model"


@t
def o35_stream_survives_the_console_encoding():
    """The laptop's piped stdout is cp1252 and strict; the pod's progress bars and warnings are UTF-8. Real child process, real stream."""
    child = "import sys; sys.stdout.buffer.write('start\\nbar 12%|\\u258f| \\u26a0\\ufe0f warn\\nend\\n'.encode('utf-8'))"
    raw = io.BytesIO()
    saved = sys.stdout, sys.stderr
    sys.stdout = io.TextIOWrapper(raw, encoding="cp1252", errors="strict", write_through=True)
    sys.stderr = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict", write_through=True)
    try:
        O.install_redaction()
        rc = REM.run_stream([sys.executable, "-c", child])
        sys.stdout.flush()
        out = raw.getvalue().decode("cp1252")
    finally:
        wrapped = sys.stdout
        sys.stdout, sys.stderr = saved
        wrapped.inner.detach()
    assert rc == 0 and out.startswith("start") and "warn" in out and out.rstrip().endswith("end"), out
    for path in [V4 / "run_v4.py", V4 / "v4_ops.py", V4 / "analyze_v4.py", V4 / "tests" / "real_model_cpu_v4.py", RV / "tools" / "v4_remote.py", RV / "tools" / "v4_gpu_mutation_check.py"]:
        for line in path.read_text(encoding="utf-8").splitlines():
            if "text=True" in line:
                assert "encoding=" in line, f"{path.name}: {line.strip()}"


@t
def o36_closeout_report_gates_deletion():
    with tmpdir() as d:
        tags = ["a", "b", R.DROP_FIRST]
        assert not O.closeout_report(d, tags, R.DROP_FIRST)["ready"]
        (d / "checkpoints").mkdir()
        st = {}
        for t_ in tags[:2]:
            p = d / "checkpoints" / f"{t_}.tar"
            p.write_bytes(t_.encode())
            st[t_] = {"checkpointed": True, "tar": p.name, "sha256": O.sha256_file(p)}
            (d / t_).mkdir()
        (d / "status.json").write_text(json.dumps(st))
        (d / "a" / "test_complete.json").write_text("{}")
        (d / "b" / "ceiling_stop.json").write_text("{}")
        r = O.closeout_report(d, tags, R.DROP_FIRST)
        assert not r["ready"] and any(R.DROP_FIRST in m for m in r["missing"]), "Mistral has neither a checkpoint nor a design that dropped it"
        (d / "design.json").write_text(json.dumps({"drop_mistral": True}))
        assert O.closeout_report(d, tags, R.DROP_FIRST)["ready"]
        (d / "a" / "test_complete.json").unlink()
        r = O.closeout_report(d, tags, R.DROP_FIRST)
        assert not r["ready"] and "final state" in r["missing"][0], "a checkpoint fetched before the model ended is not enough"
        (d / "a" / "test_complete.json").write_text("{}")
        (d / "checkpoints" / "b.tar").write_bytes(b"changed")
        assert not O.closeout_report(d, tags, R.DROP_FIRST)["ready"]


@t
def o37_hqq_is_gone_from_the_study_code():
    assert not (V4 / "rv4_hqq.py").exists()
    files = [V4 / n for n in ("run_v4.py", "rv4.py", "v4_ops.py", "analyze_v4.py", "recompute_v4_independent.py", "V4_PACKAGES.txt", "v4_sample.json")]
    files += [V4 / "cloud" / "pod_setup.sh", V4 / "cloud" / "RUNBOOK_V4.md", RV / "tools" / "lock_v4.py", RV / "tools" / "v4_gpu_mutation_check.py", RV / "tools" / "v4_remote.py",
              V4 / "plan" / "v4_plan_numbers.py", V4 / "tests" / "real_model_cpu_v4.py", V4 / "tests" / "test_rv4.py", V4 / "tests" / "test_analyze_v4.py"]
    for p in files:
        assert "hqq" not in p.read_text(encoding="utf-8").lower(), p.name
    assert "hqq" not in json.dumps(R.plan_numbers()).lower()
    assert all("hqq" not in str(p).lower() for p in LK.locked_files())


# ----------------------------------------------------------------------------- r6: Windows (CRLF) and Linux (LF) copies, ssh path with spaces
@t
def o38_lock_hashes_ignore_line_endings():
    """r6: a lock written from one checkout verifies from a CRLF copy and from an LF copy of the same text, for the previous locks and for the
    appended run files; a change of content still fails. run_v4.locked_line agrees with the appended hash for both copies."""
    with lock_repo() as (root, v4, commit):
        prev = LK.PREV_LOCKS[0]
        for b in (b"previous lock\r\n", b"previous lock\n", b"previous lock\r\n"):
            prev.write_bytes(b)
            assert quiet(LK.verify) == 0, f"previous lock with {b!r} must verify"
        prev.write_bytes(b"previous lock, changed\n")
        assert quiet(LK.verify) == 1, "a changed previous lock must still fail"
        prev.write_bytes(b"previous lock\n")
        LK.OUT.mkdir(parents=True, exist_ok=True)
        gm = LK.OUT / "gm.json"
        gm.write_bytes(b'{\r\n "all_passed": true\r\n}\r\n')
        quiet(LK.do_append, "gm")
        saved = R.LOCK
        R.LOCK = LK.LOCK
        try:
            for b in (b'{\r\n "all_passed": true\r\n}\r\n', b'{\n "all_passed": true\n}\n'):
                gm.write_bytes(b)
                assert quiet(LK.verify) == 0, f"appended gm.json with {b!r} must verify"
                assert R.locked_line("GM:", gm), "run_v4 must accept the appended file in either copy"
            gm.write_bytes(b'{\n "all_passed": false\n}\n')
            assert quiet(LK.verify) == 1 and not R.locked_line("GM:", gm), "a changed appended file must still fail"
        finally:
            R.LOCK = saved


@t
def o39_known_hosts_path_with_spaces_stays_one_file():
    """r6: the known_hosts option is quoted, so OpenSSH keeps a path with spaces as one file. Checked with every ssh found, through `ssh -G`;
    a double space shows the difference, since an unquoted path is split and printed with single spaces."""
    import shutil
    saved = REM.LOCAL_RUNS
    with tmpdir() as d:
        REM.LOCAL_RUNS = d / "a  b"
        try:
            want = (d / "a  b" / "known_hosts").as_posix()
            opt = next(x for x in REM.ssh_base({"port": 1, "identity_file": "k"}) if x.startswith("UserKnownHostsFile="))
            assert opt == f'UserKnownHostsFile="{want}"', opt
            exes = {e for e in (shutil.which("ssh"), r"C:\Windows\System32\OpenSSH\ssh.exe", r"C:\Program Files\Git\usr\bin\ssh.exe") if e and Path(e).exists()}
            assert exes, "r8: no ssh executable found; the test must not pass without checking one"
            for exe in exes:
                out = subprocess.run([exe, "-G", "-o", opt, "-p", "1", "example.invalid"], capture_output=True, **O.TEXT).stdout
                line = next(l for l in out.splitlines() if l.lower().startswith("userknownhostsfile "))
                assert line.split(" ", 1)[1].strip() == want, (exe, line)
        finally:
            REM.LOCAL_RUNS = saved



@t
def o40_v3_records_verified_against_manifest_and_v3_lock():
    """r8: the v3 records copied to the pod verify only with every manifest file present and unchanged, and each calibration.json equal to the
    V3 lock's CALIBRATION hash; the repository's own manifest verifies against the laptop records."""
    import hashlib
    with tmpdir() as d:
        files, cal = {}, []
        for lab in V3R.LABELS:
            for f in ("tierA/NAT.jsonl", "tierA/C1.jsonl", "calib/NAT.jsonl", "calibration.json"):
                p = d / "v3_runs" / lab / f
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(f"{lab} {f}\n".encode())
                files[f"v3_runs/{lab}/{f}"] = hashlib.sha256(p.read_bytes()).hexdigest()
            cal.append(f"CALIBRATION {lab}: (appended x, before any test problem of {lab}): {files[f'v3_runs/{lab}/calibration.json']}  research_v2/v3_runs/{lab}/calibration.json")
        man = d / "m.json"
        man.write_text(json.dumps({"files": files}))
        lock = "\n".join(["PROTOCOL_V3 x"] + cal) + "\n"
        assert V3R.verify(d, man, lock) == (True, [])
        assert not V3R.verify(d, d / "absent.json", lock)[0], "a missing manifest must fail"
        bad = d / "v3_runs" / "llama-base" / "calib" / "NAT.jsonl"
        bad.write_bytes(b"changed\n")
        ok, probs = V3R.verify(d, man, lock)
        assert not ok and any("hash differs" in x for x in probs), probs
        bad.write_bytes(b"llama-base calib/NAT.jsonl\n")
        (d / "v3_runs" / "qwen-base" / "tierA" / "C1.jsonl").unlink()
        ok, probs = V3R.verify(d, man, lock)
        assert not ok and any("missing" in x for x in probs), probs
        (d / "v3_runs" / "qwen-base" / "tierA" / "C1.jsonl").write_bytes(b"qwen-base tierA/C1.jsonl\n")
        assert V3R.verify(d, man, lock)[0]
        other = lock.replace(files["v3_runs/llama-instruct/calibration.json"], "0" * 64)
        ok, probs = V3R.verify(d, man, other)
        assert not ok and any("V3 lock" in x for x in probs), "a calibration.json that differs from the V3 lock must fail"
        assert not V3R.verify(d, man, "PROTOCOL_V3 x\n")[0], "no CALIBRATION line must fail"
    ok, probs = V3R.verify()
    assert ok, probs[:5]
    lf = set(LK.locked_files())
    assert V3R.MANIFEST in lf and Path(V3R.__file__).resolve() in {p.resolve() for p in lf}, "r8b: the manifest and its tool must be under the v4 lock"


@t
def o41_tests_fail_loudly_without_their_inputs():
    """r8: a04 and a16 fail (they used to pass without checking) when research_v2/v3_runs is absent."""
    import test_analyze_v4 as TA
    saved = TA.RV
    with tmpdir() as d:
        TA.RV = d
        try:
            for fn in (TA.a04_calibrate_core_reproduces_v3, TA.a16_step0_on_v3_records_themselves):
                try:
                    fn()
                except AssertionError as e:
                    assert "r8" in str(e), (fn.__name__, e)
                else:
                    raise AssertionError(f"{fn.__name__} passed without the v3 records")
        finally:
            TA.RV = saved


@t
def o42_deviation_recorded_appended_and_verified():
    """V4-D1: a post-data deviation is refused before data, needs committed files, records every changed locked file's hash and a log row,
    keeps the lock id, verifies only after it is appended and only at the recorded hashes, and allows exactly one more GM line."""
    with lock_repo() as (root, v4, commit):
        saved_dev = LK.DEV_DIR
        LK.DEV_DIR = v4 / "deviations"
        try:
            lid = LK.lock_id()
            assert "--amend" in refused(lambda: quiet(LK.deviation, "D1", "reason")), "before any record a change is an amendment"
            LK.OUT.mkdir(parents=True, exist_ok=True)
            (LK.OUT / "gm.json").write_text(json.dumps({"all_passed": True}))
            quiet(LK.do_append, "gm")
            assert quiet(LK.verify) == 0
            assert "nothing to record" in refused(lambda: quiet(LK.deviation, "D1", "reason"))
            (v4 / "PROTOCOL.md").write_text("deviated\n")
            assert "uncommitted" in refused(lambda: quiet(LK.deviation, "D1", "reason"))
            commit("deviated file")
            assert quiet(LK.verify) == 1, "a changed locked file without an appended deviation must fail"
            assert "D<number>" in refused(lambda: quiet(LK.deviation, "X1", "reason"))
            quiet(LK.deviation, "D1", "drift reference changed")
            rec = json.loads((v4 / "deviations" / "V4-D1.json").read_text())
            rel = (v4 / "PROTOCOL.md").relative_to(root).as_posix()
            assert list(rec["files"]) == [rel] and rec["lock_id"] == lid and "drift reference changed" in rec["reason"]
            assert rec["files"][rel] in LK.DEVIATION_LOG.read_text() and "| V4-D1 |" in LK.DEVIATION_LOG.read_text()
            assert "not committed" in refused(lambda: quiet(LK.append_deviation, "D1")), "the record must be committed first"
            commit("deviation record")
            assert quiet(LK.verify) == 1, "the deviation counts only once it is appended to the lock"
            quiet(LK.append_deviation, "D1")
            assert LK.lock_id() == lid and LK.active_deviation() == "D1"
            assert quiet(LK.verify) == 0
            assert "already" in refused(lambda: quiet(LK.append_deviation, "D1"))
            (v4 / "PROTOCOL.md").write_text("other\n")
            commit("another change")
            assert quiet(LK.verify) == 1, "a file is accepted at its deviation hash only"
            (v4 / "PROTOCOL.md").write_text("deviated\n")
            commit("back")
            assert quiet(LK.verify) == 0
            log = LK.DEVIATION_LOG.read_text()
            LK.DEVIATION_LOG.write_text(log.replace("| V4-D1 |", "| V4-Dx |"))
            assert quiet(LK.verify) == 1, "the deviation-log row is required"
            LK.DEVIATION_LOG.write_text(log)
            recp = v4 / "deviations" / "V4-D1.json"
            good = recp.read_text()
            recp.write_text(good.replace(rec["files"][rel], "0" * 64))
            assert quiet(LK.verify) == 1, "a changed record no longer matches its appended hash"
            recp.write_text(good)
            assert quiet(LK.verify) == 0
            assert "does not exist" in refused(lambda: quiet(LK.do_append, "gm")), "after a deviation GM goes to gm_D1.json"
            (LK.OUT / "gm_D1.json").write_text(json.dumps({"all_passed": True}))
            quiet(LK.do_append, "gm")
            assert any(l.startswith("GM D1: ") for l in LK.LOCK.read_text().splitlines()) and quiet(LK.verify) == 0
            assert "already" in refused(lambda: quiet(LK.do_append, "gm")), "one GM per deviation"
        finally:
            LK.DEV_DIR = saved_dev


@t
def o43_drift_reference_and_gm_after_deviation():
    """V4-D1: the drift reference is the design's projection x 1.48 (tolerance 25 % unchanged); after a deviation GM must pass again."""
    assert R.DRIFT == 0.25 and R.DRIFT_REF_FACTOR == 1.48
    saved = R.design
    R.design = lambda: {"expected_problem_s": {"m": 10.0}}
    try:
        assert abs(R.expected_problem_seconds("m") - 14.8) < 1e-9 and R.expected_problem_seconds("other") is None
    finally:
        R.design = saved
    import hashlib
    with tmpdir() as d, driver(d):
        set_gm_locked(d)
        assert R.gm_passed()
        R.LOCK.write_text(R.LOCK.read_text() + "DEVIATION D1: " + "0" * 64 + "  v4/deviations/V4-D1.json\n")
        assert R.active_deviation() == "D1" and R.gm_file() == d / "gm_D1.json"
        assert not R.gm_passed(), "the GM before the deviation does not count after it"
        (d / "gm_D1.json").write_text(json.dumps({"all_passed": True}))
        assert not R.gm_passed(), "gm_D1.json must be appended"
        R.LOCK.write_text(R.LOCK.read_text() + "GM D1: " + hashlib.sha256((d / "gm_D1.json").read_bytes()).hexdigest() + "  gm_D1.json\n")
        assert R.gm_passed()


@t
def o44_resume_check_bitwise_against_stored_records():
    """V4-D1: before a test resumes, the last completed problem is rerun and must equal its stored records, FP16 history and NAT slots
    (wall_seconds ignored); a difference is logged and stops; run_test calls the check before anything else."""
    arms = [t for _, _, t in R.TEST_ARMS]
    with tmpdir() as d, driver(d):
        tdir = d / "m" / "test"
        tdir.mkdir(parents=True)
        for i in (3, 5):
            for t in arms:
                with open(tdir / f"{t}.jsonl", "a") as f:
                    f.write(json.dumps({"index": i, "js": [0.1 * i, 0.25], "wall_seconds": 1.0}) + "\n")
            with open(tdir / "FP16_history.jsonl", "a") as f:
                f.write(json.dumps({"index": i, "ids": [1, 2, i]}) + "\n")
            p = R.slot_path("m", "test", "NAT", i)
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(p, k=np.arange(6, dtype=np.float32) * i, v=np.ones(3, dtype=np.float16))
        state = {"js": [0.5, 0.25], "k": np.arange(6, dtype=np.float32) * 5, "ref": [1, 2, 5]}

        def fake_problem(*a, **k):
            recs = {t: {"index": 5, "js": list(state["js"]), "wall_seconds": 9.9} for t in arms}
            recs["NAT"]["_slots"] = {"k": state["k"], "v": np.ones(3, dtype=np.float16)}
            return recs, list(state["ref"])
        saved = R.run_problem
        R.run_problem = fake_problem
        try:
            ids = [3, 5, 7]
            assert R.resume_check(None, None, None, "m", "m", ids, None) is True
            assert json.loads((d / "m" / "resume_check.jsonl").read_text().splitlines()[-1])["index"] == 5
            for key, bad in (("js", [0.5000000000000001, 0.25]), ("k", np.arange(6, dtype=np.float32) * 5 + 1e-3), ("ref", [1, 2, 6])):
                good = state[key]
                state[key] = bad
                assert R.resume_check(None, None, None, "m", "m", ids, None) is False, key
                assert (d / "m" / "resume_check_failure.json").exists()
                (d / "m" / "resume_check_failure.json").unlink()
                state[key] = good
            assert "resume_check_failure.json" in O.STOP_FILES
        finally:
            R.run_problem = saved
        calls = []
        saved = (R.load, R.precision, R.resume_check)
        R.load = lambda m: (None, None)
        R.precision = lambda m: "fp16"
        R.resume_check = lambda *a: calls.append(a[3]) or False
        try:
            s = {"tierA_ids": [3, 5, 7]}
            assert R.run_test("m", s, None, None) is False and calls == ["m"], "run_test must run the resume check first"
        finally:
            R.load, R.precision, R.resume_check = saved

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
