"""Crash-injection tests of the v3 driver (amendment 1). The driver's own functions run on CPU with a tiny random Qwen2 sized to the real tokenizer,
real GSM8K prompts and 8 steps; each scenario runs in a fresh temporary folder and is compared with an uninterrupted run under the same model label
(records compared without the per-run `wall_seconds`).
  venv/Scripts/python.exe research_v2/tests/test_v3_resume.py [--write]
Scenarios: A interruption while computing; B interruption while appending (Tier A); C torn trailing line -> driver stops -> logged repair -> rerun;
D calibration interrupted mid-append; E cumulative ledger + unclean-exit recovery; F heartbeat of the ledger context after a HARD KILL (real
subprocess, os._exit); G analysis refuses duplicate (model, arm, problem) records."""
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

RV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RV))
sys.path.insert(0, str(RV / "tools"))

import run_v3 as R                        # noqa: E402
from transformers import AutoTokenizer    # noqa: E402

import torch                              # noqa: E402
import analyze_v3 as A                    # noqa: E402
import rv3.core as V                      # noqa: E402
import v3_repair_torn_line as RP          # noqa: E402
from rv2.backends import build_cpu_backend, tiny_qwen2  # noqa: E402

M = "qwen-base"
_ENV = {}


def env():
    if not _ENV:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
        be = build_cpu_backend(tiny_qwen2(0, vocab=151936))
        s = R.make_sample()
        R.S_STEPS = 8
        _ENV.update(tok=tok, be=be, data=R.RS1.Data(), ids=s["tierA_ids"][:3], cal=s["calibration_train_ids"][:3],
                    arms=R.arm_list({"dropped_arms": []}), ctx=V.synthetic_ctx(be.model.config, "cpu"), cap=V.QueryCapture(be.model))
    return _ENV


def fresh():
    R.OUT = Path(tempfile.mkdtemp())
    R.DESIGN = R.OUT / "design.json"


def lines(p):
    return p.read_bytes().splitlines(keepends=True) if p.exists() else []


def strip(b):
    r = json.loads(b)
    r.pop("wall_seconds", None)
    return json.dumps(r, sort_keys=True)


def files(m=M, sub="tierA"):
    return {p.name: lines(p) for p in (R.mdir(m) / sub).glob("*.jsonl")}


def step(i):
    e = env()
    return R.tier_step(e["be"], e["tok"], e["data"], M, i, e["arms"], e["ctx"], e["cap"])


def done(i):
    return all(i in R.done_ids(R.mdir(M) / "tierA", tag) for _, _, tag in env()["arms"])


def loop():
    R.require_intact_records(M)                     # exactly what the driver does before tierA
    for i in env()["ids"]:
        if not done(i):
            step(i)


_REF = {}


def reference():
    if not _REF:
        fresh()
        loop()
        _REF.update({k: [strip(x) for x in v] for k, v in files().items()})
    return _REF


def same_as_reference():
    ref = reference()
    got = {k: [strip(x) for x in v] for k, v in files().items()}
    return got == ref


TESTS = []


def t(fn):
    TESTS.append(fn)
    return fn


@t
def A_interrupted_while_computing():
    reference()
    fresh()
    orig = R.run_problem

    def boom(*a, **k):
        if a[4] == env()["ids"][1]:
            raise KeyboardInterrupt
        return orig(*a, **k)
    R.run_problem = boom
    try:
        loop()
    except KeyboardInterrupt:
        pass
    finally:
        R.run_problem = orig
    before = files()
    loop()
    after = files()
    assert all(after[k][:len(v)] == v for k, v in before.items()), "records written before the interruption changed"
    assert all(len(v) == 3 for v in after.values()), "not exactly one record per problem and arm"
    assert same_as_reference(), "resumed run differs from an uninterrupted run"


@t
def B_interrupted_while_appending_tierA():
    reference()
    fresh()
    orig = R.append_records
    target = env()["ids"][1]

    def boom(d, recs):
        if any(r.get("index") == target for r in recs.values()) and "FP16_history" not in recs:
            for k, (tag, r) in enumerate(recs.items()):
                if k == 5:
                    raise KeyboardInterrupt
                orig(d, {tag: r})
        return orig(d, recs)
    R.append_records = boom
    try:
        loop()
    except KeyboardInterrupt:
        pass
    finally:
        R.append_records = orig
    before = files()
    loop()
    after = files()
    assert all(len(v) == 3 for v in after.values()), {k: len(v) for k, v in after.items()}
    assert all(after[k][:len(v)] == v for k, v in before.items()), "records written before the interruption changed"
    assert same_as_reference(), "resumed run differs from an uninterrupted run"


@t
def C_torn_line_stops_driver_then_logged_repair_then_rerun():
    reference()
    fresh()
    orig = R.append_records
    target = env()["ids"][1]

    def torn(d, recs):
        if any(r.get("index") == target for r in recs.values()) and "FP16_history" not in recs:
            for k, (tag, r) in enumerate(recs.items()):
                if k == 3:
                    with open(Path(d) / f"{tag}.jsonl", "a", encoding="utf-8") as f:
                        f.write(json.dumps(r)[:150])      # power loss mid-write
                    raise KeyboardInterrupt
                orig(d, {tag: r})
        return orig(d, recs)
    R.append_records = torn
    try:
        loop()
    except KeyboardInterrupt:
        pass
    finally:
        R.append_records = orig
    try:
        loop()
        raise AssertionError("the driver must stop on a torn record file")
    except SystemExit as e:
        assert "v3_repair_torn_line.py" in str(e), str(e)
    bad = R.torn_record_files(M)
    assert len(bad) == 1, bad
    dev = R.OUT / "dev.md"
    dev.write_text("", encoding="utf-8")
    assert RP.main([str(bad[0]), "--reason", "test: power loss mid-write", "--root", str(R.OUT), "--deviation-log", str(dev)]) == 0
    assert (R.OUT / "repair_log.jsonl").exists() and "v3_repair_torn_line.py" in dev.read_text(encoding="utf-8")
    loop()
    after = files()
    assert all(len(v) == 3 for v in after.values()) and same_as_reference(), "after repair and rerun the records must equal an uninterrupted run"


@t
def D_calibration_interrupted_mid_append():
    e = env()
    fresh()
    R.calib_step(e["be"], e["tok"], e["data"], M, e["cal"][0], e["cap"])
    orig = R.append_records

    def boom(d, recs):
        for k, (tag, r) in enumerate(recs.items()):
            if k == 2:
                raise KeyboardInterrupt
            orig(d, {tag: r})
    R.append_records = boom
    try:
        R.calib_step(e["be"], e["tok"], e["data"], M, e["cal"][1], e["cap"])
    except KeyboardInterrupt:
        pass
    finally:
        R.append_records = orig
    for i in e["cal"]:
        if not R.calib_done(M, i):
            R.calib_step(e["be"], e["tok"], e["data"], M, i, e["cap"])
    aux = torch.load(R.aux_path(M))
    nat = A.load_jsonl(R.mdir(M) / "calib" / "NAT.jsonl")
    assert {len(v) for v in files(sub="calib").values()} == {3}
    assert aux["included"] == e["cal"] and aux["n_meas"] == sum(len(nat[i]["js"]) for i in e["cal"]), "accumulator must count each problem once"


@t
def E_ledger_cumulative_and_unclean_exit_recovered():
    fresh()
    RS1 = R.RS1
    RS1.OUT, RS1.LEDGER, RS1.PROGRESS = R.OUT, R.OUT / "gpu_ledger.jsonl", R.OUT / "current_process.json"
    with open(RS1.LEDGER, "a") as f:
        f.write(json.dumps({"seconds": 3600.0, "block": "s1", "completed": True}) + "\n")
        f.write(json.dumps({"seconds": 120.0, "block": "gpu_mutation_check", "completed": True}) + "\n")
    RS1.write_progress("tierA:qwen-base", time.time() - 1800)
    rec = RS1.recover_open_process(RS1.PROGRESS, RS1.LEDGER)
    assert abs(rec - 1800) < 5 and abs(RS1.ledger_seconds() - (3600 + 120 + rec)) < 1e-6


@t
def F_heartbeat_charges_a_hard_kill():
    tmp = Path(tempfile.mkdtemp())
    code = ("import sys, time, os; sys.path.insert(0, r'%s'); import run_v3 as R; from pathlib import Path; R.OUT = Path(r'%s');\n"
            "cm = R.ledger_heartbeat('gpu_mutation_check'); cm.__enter__(); time.sleep(3); os._exit(9)") % (RV, tmp)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=600)
    assert r.returncode == 9, r.stderr[-500:]
    assert (tmp / "current_process.json").exists(), "a hard-killed process must leave its heartbeat"
    assert not (tmp / "gpu_ledger.jsonl").exists(), "no ledger line can be written by a hard-killed process"
    RS1 = R.RS1
    RS1.OUT, RS1.LEDGER, RS1.PROGRESS = tmp, tmp / "gpu_ledger.jsonl", tmp / "current_process.json"
    rec = RS1.recover_open_process(RS1.PROGRESS, RS1.LEDGER)
    assert rec >= 0.0 and (tmp / "gpu_ledger.jsonl").exists() and not (tmp / "current_process.json").exists(), "the next process must charge it"
    # a clean exit writes exactly one line and leaves no heartbeat
    R.OUT = tmp
    with R.ledger_heartbeat("gpu_mutation_check") as led:
        time.sleep(0.5)
        assert (tmp / "current_process.json").exists(), "the heartbeat must be written while the work runs"
        led["completed"] = True
    L = [json.loads(x) for x in (tmp / "gpu_ledger.jsonl").read_text().splitlines()]
    assert len(L) == 2 and L[-1]["completed"] and L[-1]["block"] == "gpu_mutation_check" and not (tmp / "current_process.json").exists()


@t
def G_analysis_refuses_duplicate_records():
    tmp = Path(tempfile.mkdtemp())
    p = tmp / "NAT.jsonl"
    p.write_text(json.dumps({"index": 1, "js": [0.0]}) + "\n" + json.dumps({"index": 1, "js": [0.0]}) + "\n", encoding="utf-8")
    try:
        A.load_jsonl(p)
        raise AssertionError("duplicate records must fail loudly")
    except ValueError as e:
        assert "exactly once" in str(e)


def run_all():
    fails = []
    for fn in TESTS:
        try:
            fn()
        except BaseException as e:  # noqa: BLE001
            fails.append((fn.__name__, f"{type(e).__name__}: {e}"))
    return fails


if __name__ == "__main__":
    t0 = time.time()
    f = run_all()
    for fn in TESTS:
        print(("FAIL " if fn.__name__ in dict(f) else "PASS ") + fn.__name__)
    for x in f:
        print(*x)
    print(f"{len(TESTS) - len(f)}/{len(TESTS)} passed")
    if "--write" in sys.argv:
        (RV / "validation_logs" / "v3_resume_tests.json").write_text(json.dumps({"passed": len(TESTS) - len(f), "total": len(TESTS),
                                                                                "failures": f, "seconds": round(time.time() - t0, 1)}, indent=1))
    sys.exit(1 if f else 0)
