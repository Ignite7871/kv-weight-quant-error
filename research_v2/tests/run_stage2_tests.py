"""CPU tests for the Stage 2 driver and tools (STAGE2_SPEC.md). They touch NO held-out problem text or outcome: the only held-out
information used is the list of indices.

Run: venv/Scripts/python.exe research_v2/tests/run_stage2_tests.py
"""

import json
import random
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
RV = HERE.parent
sys.path.insert(0, str(RV))

import run_stage1 as RS1          # noqa: E402
import run_stage2 as RS2          # noqa: E402
import stage1_runtime as T        # noqa: E402

RESULTS = []


def test(fn):
    def wrapper():
        try:
            fn()
            RESULTS.append((fn.__name__, "PASS", ""))
        except Exception as e:  # noqa: BLE001
            RESULTS.append((fn.__name__, "FAIL", f"{type(e).__name__}: {e}"))
            traceback.print_exc()
    wrapper.__name__ = fn.__name__
    return wrapper


PATH_NAMES = ("OUT", "LEDGER", "PROGRESS", "RUNTIME_LOG", "SESSIONS", "INTEGRITY_FAILURE", "CHECKPOINT_SNAPSHOT", "PROJECTION_FILE", "SAMPLE_FILE")


def saved():
    return {k: getattr(RS1, k) for k in PATH_NAMES}


def restore(d):
    for k, v in d.items():
        setattr(RS1, k, v)


@test
def g1_sample_is_the_complement_of_the_pool_with_a_reproducible_seeded_order():
    s = json.loads((RV / "stage2_sample.json").read_text())
    s1 = json.loads((RV / "stage1_sample.json").read_text())
    held, pool = s["heldout389"], s["pool930"]
    assert len(held) == 389 and held == sorted(held) and len(set(held)) == 389
    assert pool == s1["pool930"] and held == sorted(s1["reserved389"])
    assert not set(held) & set(pool) and set(held) | set(pool) == set(range(1319))
    assert sorted(s["run_order"]) == held and len(s["run_order"]) == 389
    assert s["run_order"] == random.Random(RS2.ORDER_SEED).sample(held, len(held)) and s["order_seed"] == "rv2-stage2-order-2026-10-25"
    assert s["run_order"] != held                                                  # actually shuffled
    assert s["exposed_gold_trajectory_test_0_199"] == [i for i in held if i < 200] and len(s["exposed_gold_trajectory_test_0_199"]) == 47
    assert s["exposed_probe_test_0_7"] == [4, 5]
    assert s["equiv_train"] == s1["equiv_train"][:3] and s["speed_train"] == s1["speed_train"]
    # regenerating from scratch reproduces the committed file byte for byte
    with tempfile.TemporaryDirectory() as d:
        orig = RS2.SAMPLE_FILE
        RS2.SAMPLE_FILE = Path(d) / "s.json"
        try:
            again = RS2.make_sample2()
        finally:
            RS2.SAMPLE_FILE = orig
        assert again == s


@test
def g2_only_heldout_problems_may_be_evaluated():
    s = json.loads((RV / "stage2_sample.json").read_text())
    RS2.assert_heldout_indices(s["run_order"], s)
    RS2.assert_heldout_indices(s["run_order"][:5], s)
    for bad in ([s["pool930"][0]], [s["run_order"][0], s["pool930"][10]], [1319], [-1]):
        try:
            RS2.assert_heldout_indices(bad, s)
        except AssertionError:
            continue
        raise AssertionError(f"accepted {bad}")


@test
def g3_configure_redirects_every_stage1_path_into_stage2_and_ledger_is_separate():
    keep = saved()
    try:
        with tempfile.TemporaryDirectory() as d:
            out = RS2.configure(Path(d) / "stage2_runs")
            for k in PATH_NAMES:
                v = Path(getattr(RS1, k))
                if k == "SAMPLE_FILE":
                    assert v == RS2.SAMPLE_FILE and v.name == "stage2_sample.json"
                else:
                    assert out in v.parents or v == out, (k, v)
                assert "stage1_runs" not in str(v), (k, v)
            (out / "gpu_ledger.jsonl").write_text(json.dumps({"seconds": 3600.0}) + "\n")
            assert abs(RS1.ledger_seconds() - 3600.0) < 1e-9                       # reads the Stage 2 ledger only
    finally:
        restore(keep)
    assert RS1.ledger_seconds() > 10 * 3600                                         # Stage 1's own ledger is back (10.885 h)


@test
def g4_cap_cannot_be_raised_and_gpu_runs_need_authorization_and_the_gate():
    argv0 = sys.argv[:]
    keep, out0 = saved(), RS2.OUT
    try:
        for argv, needle in ((["x", "ledger", "--cap-hours", "6"], "not permitted"), (["x", "run"], "authorization")):
            sys.argv = argv
            try:
                RS2.main()
            except SystemExit as e:
                assert needle in str(e.code), (argv, e.code)
            else:
                raise AssertionError(f"{argv} did not exit")
        with tempfile.TemporaryDirectory() as d:                                    # no passed correctness gate -> run refuses
            RS2.OUT = Path(d) / "stage2_runs"
            sys.argv = ["x", "run", "--authorized-gpu-run"]
            try:
                RS2.main()
            except SystemExit as e:
                assert "correctness check has not passed" in str(e.code), e.code
            else:
                raise AssertionError("run proceeded without the gate")
            (RS2.OUT / "gates").mkdir(parents=True, exist_ok=True)
            (RS2.OUT / "gates" / "gate_equiv.json").write_text(json.dumps({"all_passed": False}))
            try:
                RS2.main()
            except SystemExit as e:
                assert "correctness check has not passed" in str(e.code)
            else:
                raise AssertionError("run proceeded with a failed gate")
    finally:
        sys.argv = argv0
        RS2.OUT = out0
        restore(keep)


@test
def g5_projection_uses_stage1_session_rates_and_reports_whether_it_fits_5_5_h():
    keep = saved()
    try:
        with tempfile.TemporaryDirectory() as d:
            r = RS2.project(Path(d), 5.5, 0.0)
        p = r["projection"]
        assert p["fastest_session"]["s_per_problem"] < p["pooled"]["s_per_problem"] < p["slowest_session"]["s_per_problem"]
        assert abs(p["pooled"]["main_run_h"] - 389 * p["pooled"]["s_per_problem"] / 3600) < 1e-9
        assert p["pooled"]["cumulative_h"] > p["pooled"]["main_run_h"]               # checks and load are added
        assert r["fits_central"] is True and r["max_problems_at_slowest_rate"] > 0
        with tempfile.TemporaryDirectory() as d:                                     # a budget that is already half spent must NOT fit
            assert RS2.project(Path(d), 5.5, 3 * 3600.0)["fits_central"] is False
    finally:
        restore(keep)


@test
def g6_cleanliness_check_passes_on_the_repository_and_fails_on_a_planted_record():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "c.json"
        r = subprocess.run([sys.executable, str(RV / "tools" / "check_heldout_clean.py"), "--out", str(out)], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-800:]
        rep = json.loads(out.read_text())
        assert rep["clean"] and rep["disjoint"] and rep["union_is_full_test_split"] and not rep["unexpected_hits"]
        assert rep["exposure_disclosure"]["n_gold_trajectory_labelled"] == 47
    # a stored per-problem outcome for a held-out problem must be detected
    import importlib.util
    spec = importlib.util.spec_from_file_location("chk", RV / "tools" / "check_heldout_clean.py")
    chk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(chk)
    s = json.loads((RV / "stage2_sample.json").read_text())
    held = set(s["heldout389"])
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "x.jsonl"
        f.write_text(json.dumps({"index": s["heldout389"][3], "split": "test", "status": "SOLVED"}) + "\n")
        assert chk.index_hits(f, held)
        f.write_text(json.dumps({"index": s["heldout389"][3], "split": "train", "status": "SOLVED"}) + "\n")
        assert not chk.index_hits(f, held)                                           # train-split index is not a test problem
        f.write_text(json.dumps({"index": s["pool930"][3], "split": "test"}) + "\n")
        assert not chk.index_hits(f, held)
        g = Path(d) / "y.json"
        g.write_text(json.dumps({"results": [{"index": i, "status": "WRONG"} for i in s["heldout389"][:10]]}))
        assert chk.index_hits(g, held)
        h = Path(d) / "z.csv"
        h.write_text("problem_id,x\n" + "\n".join(f"{i},1" for i in s["heldout389"][:6]))
        assert chk.index_hits(h, held)


@test
def g7_checkpoint_order_check_uses_the_stage2_run_order_and_detects_repeats_and_skips():
    s = json.loads((RV / "stage2_sample.json").read_text())
    order = s["run_order"]
    with tempfile.TemporaryDirectory() as d:
        e = Path(d) / "eval"
        e.mkdir()
        for a in T.ARMS:
            (e / f"{a}.jsonl").write_text("".join(json.dumps({"index": i}) + "\n" for i in order[:7]))
        r = T.verify_checkpoint(e, None, order, T.ARMS)
        assert r["ok"] and r["n_complete_all_arms"] == 7
        assert RS1.verify_checkpoint(e, None, order, T.ARMS)["ok"]
        (e / "nf4.jsonl").write_text("".join(json.dumps({"index": i}) + "\n" for i in order[:3] + order[4:8]))      # skipped order[3]
        assert not T.verify_checkpoint(e, None, order, T.ARMS)["ok"]
        (e / "nf4.jsonl").write_text("".join(json.dumps({"index": i}) + "\n" for i in sorted(order[:7])))          # reordered
        assert not T.verify_checkpoint(e, None, order, T.ARMS)["ok"]


@test
def g8_provenance_and_budget_constants():
    assert RS2.CAP_HOURS == 5.5 and RS2.ARMS == ["fp16", "nf4", "cache_only", "token_only", "oracle_flip"]
    assert RS2.OUT.name == "stage2_runs"
    cap_s = RS2.CAP_HOURS * 3600
    worst = T.worst_problem_seconds([40.0, 120.0], 1.03)
    assert T.stop_decision(cap_s - worst - T.HARD_MARGIN_S - 1, worst, cap_s) is None
    assert T.stop_decision(cap_s - worst - T.HARD_MARGIN_S + 1, worst, cap_s) == "cap"


def main():
    tests = [v for k, v in globals().items() if k.startswith("g") and k[1:2].isdigit() and callable(v)]
    for t in tests:
        t()
    w = max(len(n) for n, _, _ in RESULTS)
    print("\n" + "=" * (w + 30))
    for name, status, msg in RESULTS:
        print(f"{name:<{w}}  {status}  {msg}")
    n_fail = sum(s == "FAIL" for _, s, _ in RESULTS)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
