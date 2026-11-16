"""CPU unit tests of v6b/rv6b.py and the CPU logic of run_v6b.py (gate G0 of FOLLOWUP_V6B).  python research_v2/v6b/tests/test_rv6b.py"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import rv6b  # noqa: E402
import run_v6b as RB  # noqa: E402

import torch  # noqa: E402

rv6, V, rv4 = rv6b.rv6, rv6b.V, rv6b.rv4
C = rv6.C
L_, NKV, HD = 3, 2, 64
DEV = "cpu"


def entries(seed=0):
    g = torch.Generator().manual_seed(seed)
    kF = (torch.randn(2 * L_, NKV, HD, generator=g) * 2).half()
    kM = (kF.double() + 0.05 * torch.randn(2 * L_, NKV, HD, generator=g, dtype=torch.float64) + 0.03).half()
    mu = 0.02 * torch.randn(L_, NKV, HD, generator=g, dtype=torch.float64) + 0.03
    return kF, kM, mu


def gen(seed=11):
    return C.make_generator(torch.device(DEV), seed)


def vrows(x):
    return x[1::2]


def test_keys_exact_and_rounding():
    kF, kM, mu = entries()
    for arm in rv6b.NEW_ARMS:
        out, info = rv6b.new_entry(arm, kF, kM, gen(), {"mu": mu})
        assert torch.equal(out[0::2], kF[0::2]), f"{arm}: key rows must stay FP16-exact"
        assert bool(info["exact_rows"][0::2].all()) and not bool(info["exact_rows"][1::2].any()), f"{arm}: exact mask"
        assert torch.equal(out, V.fp16_rne_reference(info["pre"])), f"{arm}: rounding"


def test_A_is_centered_unrenormed():
    kF, kM, mu = entries()
    _, info = rv6b.new_entry("NAT-V-cen0", kF, kM, gen(), {"mu": mu})
    F = kF.double()
    c = vrows(kM.double() - F) - mu
    assert torch.allclose(vrows(info["pre"] - F), c, atol=1e-12), "A must store Dn - mu"
    assert torch.allclose(vrows(info["target"]), c.norm(dim=-1, keepdim=True), atol=0), "A target = ||c||"


def test_A_renormed_equals_v6_cen():
    kF, kM, mu = entries()
    _, a = rv6b.new_entry("NAT-V-cen0", kF, kM, gen(), {"mu": mu})
    _, v6 = rv6b._ORIG("NAT-V-cen", kF, kM, gen(), gen(12), {"mu": mu})
    F = kF.double()
    dA = vrows(a["pre"] - F)
    nD = vrows(kM.double() - F).norm(dim=-1, keepdim=True)
    assert torch.allclose(F[1::2] + nD * dA / dA.norm(dim=-1, keepdim=True), vrows(v6["pre"]), atol=1e-12), "A re-normed != NAT-V-cen"


def test_C_renormed_equals_C1V_same_draw():
    kF, kM, mu = entries()
    _, c_ = rv6b.new_entry("C1-V@cen", kF, kM, gen(5), {"mu": mu})
    _, c1 = rv4.stored_entry_v4("C1-V", 1.0, kF, kM, gen(5), gen(6), torch.float16)
    F = kF.double()
    dC = vrows(c_["pre"] - F)
    nD = vrows(kM.double() - F).norm(dim=-1, keepdim=True)
    assert torch.allclose(F[1::2] + nD * dC / dC.norm(dim=-1, keepdim=True), vrows(c1["pre"]), atol=1e-12), "C re-normed != C1-V (same draw)"
    nc = (vrows(kM.double() - F) - mu).norm(dim=-1)
    assert torch.allclose(dC.norm(dim=-1), nc, atol=1e-12), "C norm = ||c||"


def test_B_is_mu_plus_C():
    kF, kM, mu = entries()
    rv6b.State.sums = None
    rv6b.State.reset("C1-V+mu")
    _, b = rv6b.new_entry("C1-V+mu", kF, kM, gen(5), {"mu": mu})
    _, c_ = rv6b.new_entry("C1-V@cen", kF, kM, gen(5), {"mu": mu})
    F = kF.double()
    assert torch.allclose(vrows(b["pre"] - F) - mu, vrows(c_["pre"] - F), atol=1e-12), "B - mu must equal C (same draw)"
    assert b["target"] is None, "B has no fixed-norm target"
    assert len(rv6b.State.b_dev) == 1 and rv6b.State.b_dev[0] <= 1e-10, "B construction check"
    nD = vrows(kM.double() - F).norm(dim=-1)
    want = float((vrows(b["pre"] - F).norm(dim=-1) / nD).mean())
    assert abs(rv6b.State.b_ratio[0] - want) < 1e-12, "B norm ratio diagnostic"


def test_B_with_zero_mu_equals_C():
    kF, kM, _ = entries()
    z = torch.zeros(L_, NKV, HD, dtype=torch.float64)
    ob, _ = rv6b.new_entry("C1-V+mu", kF, kM, gen(9), {"mu": z})
    oc, _ = rv6b.new_entry("C1-V@cen", kF, kM, gen(9), {"mu": z})
    assert torch.equal(ob, oc)


def test_draw_sequence_matches_C1V():
    """Over several steps, B and C consume C1-V's generator exactly as C1-V does (one float32 draw per step)."""
    kF, kM, mu = entries()
    gC1, gB, gC = gen(21), gen(21), gen(21)
    F = kF.double()
    for _ in range(4):
        _, c1 = rv4.stored_entry_v4("C1-V", 1.0, kF, kM, gC1, gen(1), torch.float16)
        _, b = rv6b.new_entry("C1-V+mu", kF, kM, gB, {"mu": mu})
        _, c_ = rv6b.new_entry("C1-V@cen", kF, kM, gC, {"mu": mu})
        u1 = vrows(c1["pre"] - F)
        u1 = u1 / u1.norm(dim=-1, keepdim=True)
        uc = vrows(c_["pre"] - F)
        uc = uc / uc.norm(dim=-1, keepdim=True)
        ub = vrows(b["pre"] - F) - mu
        ub = ub / ub.norm(dim=-1, keepdim=True)
        assert torch.allclose(u1, uc, atol=1e-12) and torch.allclose(u1, ub, atol=1e-12), "direction differs from C1-V's draw"


def test_seed_arm():
    assert rv6b.SEED_ARM["C1-V+mu"] == "C1-V" and rv6b.SEED_ARM["C1-V@cen"] == "C1-V", "B and C must replay C1-V's seed"
    assert rv6b.SEED_ARM["NAT-V-cen0"] not in ("C1-V",), "A draws nothing"
    assert rv6b.ARMS == ["NAT-V", "C1-V", "NAT-V-cen", "NAT-V-cen0", "C1-V+mu", "C1-V@cen"], "run order"


def test_dispatch_delegates_v6_arms():
    kF, kM, mu = entries()
    for arm in ("NAT-V", "C1-V", "NAT-V-cen"):
        a, ia = rv6b.stored_entry(arm, kF, kM, gen(3), gen(4), {"mu": mu} if arm == "NAT-V-cen" else {})
        b, ib = rv6b._ORIG(arm, kF, kM, gen(3), gen(4), {"mu": mu} if arm == "NAT-V-cen" else {})
        assert torch.equal(a, b) and torch.equal(ia["pre"], ib["pre"]), f"{arm}: dispatcher changed a v6 arm"
    rv6b.install()
    assert rv6.stored_entry_v6 is rv6b.stored_entry
    rv6b.uninstall()
    assert rv6.stored_entry_v6 is rv6b._ORIG


def test_dispatch_new_arms_use_gen():
    """Through the dispatcher (as the step calls it), B and C must draw from `gen` (C1-V's seed), not from `vgen`."""
    kF, kM, mu = entries()
    for arm in ("C1-V+mu", "C1-V@cen"):
        rv6b.State.reset(arm)
        a, _ = rv6b.stored_entry(arm, kF, kM, gen(5), gen(99), {"mu": mu})
        b, _ = rv6b.new_entry(arm, kF, kM, gen(5), {"mu": mu})
        assert torch.equal(a, b), f"{arm}: dispatcher must pass gen"


def test_mu_matches_v6_slot_sums():
    rv6b.State.reset("NAT-V")
    news, kFs = [], []
    for s in range(5):
        kF, kM, _ = entries(100 + s)
        out, _ = rv6b.stored_entry("NAT-V", kF, kM, gen(), gen(2), {})
        news.append(out)
        kFs.append(kF)
    mu = rv6b.mu_from_state()
    ref = rv4.SlotSums()
    for o, f in zip(news, kFs):
        ref.record(o, f)
    assert torch.equal(mu, rv6.slot_mean(ref.arrays(), "V")), "mu must use v6's SlotSums arithmetic"
    want = sum((o.double() - f.double())[1::2] for o, f in zip(news, kFs)) / 5
    assert torch.allclose(mu, want, atol=1e-12)
    rv6b.State.reset("C1-V")
    assert rv6b.State.sums is None, "only NAT-V accumulates mu"


def test_degenerate_count():
    kF, kM, _ = entries()
    mu = vrows(kM.double() - kF.double()).clone()            # c = 0 everywhere
    _, info = rv6b.new_entry("NAT-V-cen0", kF, kM, gen(), {"mu": mu})
    assert info["degenerate"] == L_ * NKV


# ----------------------------------------------------------------------------- run_v6b CPU logic
def rec(i, js, S=4, ref_len=5, top=(1, 2, 3, 4)):
    return {"index": i, "js": list(js), "S": S, "ref_len": ref_len, "top": list(top)}


def test_g1():
    old = {"NAT-V": {7: rec(7, [0.0, 0.01, 0.02, 0.03])}}
    ok, res = RB.g1("m", "test", {"NAT-V": rec(7, [0.0, 0.01, 0.02, 0.03])}, old)
    assert ok and res["NAT-V"]["bitwise_js"]
    ok, _ = RB.g1("m", "test", {"NAT-V": rec(7, [0.0, 0.0105, 0.021, 0.0315])}, old)       # ln ratio 0.049
    assert ok, "within tolerance must pass"
    ok, _ = RB.g1("m", "test", {"NAT-V": rec(7, [0.0, 0.0106, 0.0212, 0.0318])}, old)      # ln 1.06 = 0.058
    assert not ok, "beyond 0.05 must fail"
    ok, _ = RB.g1("m", "test", {"NAT-V": rec(7, [0.0, 0.01, 0.02, 0.03], S=3)}, old)
    assert not ok, "different S must fail"
    ok, _ = RB.g1("m", "test", {"NAT-V": rec(8, [0.0, 0.01, 0.02, 0.03])}, old)
    assert not ok, "missing v6 record must fail"


def test_integrity_includes_b_check():
    r = {k: True for k in rv6.FLAGS}
    assert RB.integrity({"C1-V+mu": {**r, "b_construction_ok": True}}) == []
    assert RB.integrity({"C1-V+mu": {**r, "b_construction_ok": False}}) == ["C1-V+mu:b_construction_ok"]
    assert RB.integrity({"NAT-V": dict(r)}) == ["NAT-V:b_construction_ok"], "a missing flag is a failure"


def test_rerun_for_cut1():
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        ids = list(range(100, 120))
        des0 = {"n": 70, "cuts": []}
        des1 = {"n": 70, "cuts": [RB.CUT1]}
        assert RB.rerun_for(des0, 15, ids, d) == rv6b.REPRO_ARMS, "no cut: rerun all"
        assert RB.rerun_for(des1, 15, ids, d) == rv6b.REPRO_ARMS, "cut but no G1 file: rerun all"
        bit = {a: {"bitwise_js": True, "pass": True} for a in rv6b.REPRO_ARMS}
        with open(d / "g1.jsonl", "w", encoding="utf-8") as f:
            for i in ids[:10]:
                f.write(json.dumps({"index": i, "G1": bit}) + "\n")
        assert RB.rerun_for(des1, 5, ids, d) == rv6b.REPRO_ARMS, "first 10 always rerun all"
        assert RB.rerun_for(des1, 15, ids, d) == ["NAT-V"], "cut 1 after 10 bitwise problems"
        nb = {**bit, "C1-V": {"bitwise_js": False, "pass": True}}
        with open(d / "g1.jsonl", "w", encoding="utf-8") as f:
            for k, i in enumerate(ids[:10]):
                f.write(json.dumps({"index": i, "G1": nb if k == 4 else bit}) + "\n")
        assert RB.rerun_for(des1, 15, ids, d) == rv6b.REPRO_ARMS, "a non-bitwise problem among the first 10 blocks cut 1"


class _PastRefusal(Exception):
    pass


def _tripwire(*a, **k):
    raise _PastRefusal("reached data / model loading in a CPU test")


def test_refusals():
    """Without authorization (and lock) the driver must exit before any data or model is loaded; make_sample and load are tripwires."""
    import io, contextlib
    prev = (RB.OUT, RB.R6.make_sample, RB.R6.load)
    RB.R6.make_sample, RB.R6.load = _tripwire, _tripwire
    try:
        with tempfile.TemporaryDirectory() as t:
            RB.OUT = Path(t)
            for fn in (lambda: RB.rehearse("llama3.2-1b"), lambda: RB.test("llama3.2-1b")):
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        fn()
                    raise AssertionError("must refuse without authorization / lock")
                except SystemExit as e:
                    assert "refused" in str(e)
    finally:
        RB.OUT, RB.R6.make_sample, RB.R6.load = prev


def test_constants():
    assert RB.CAP_H == 9.0 and RB.N_TEST == 70 and RB.G1_TOL == 0.05
    assert RB.TIER == {"test": "V6A", "train": "V6reh"}, "reruns must use v6's tiers"


# ----------------------------------------------------------------------------- launcher (exactly-once scan, cuts)
import launch_test_v6b as LT  # noqa: E402


def _launch_dir(t, rows=None, g1=None):
    LT.OUT = Path(t)
    d = LT.OUT / "test" / "llama3.2-1b"
    d.mkdir(parents=True, exist_ok=True)
    rows = rows if rows is not None else {a: [1, 2, 3] for a in LT.REQ + LT.OPT}
    for a, ids in rows.items():
        (d / f"{a}.jsonl").write_text("".join(json.dumps({"index": i, **{k: True for k in LT.FLAGS}}) + "\n" for i in ids), encoding="utf-8")
    (d / "g1.jsonl").write_text("".join(json.dumps({"index": i, "G1": {}}) + "\n" for i in (g1 if g1 is not None else [1, 2, 3])), encoding="utf-8")
    return d


def _stops(fn):
    try:
        fn()
    except SystemExit as e:
        return str(e)
    return None


def test_launcher_scan():
    prev = LT.OUT
    try:
        _launcher_scan()
    finally:
        LT.OUT = prev


def _launcher_scan():
    with tempfile.TemporaryDirectory() as t:
        _launch_dir(t)
        full, bad = LT.scan("llama3.2-1b")
        assert full == {1, 2, 3} and bad == 0
        _launch_dir(t, rows={**{a: [1, 2, 3] for a in LT.REQ}, "C1-V": [1], "NAT-V-cen": [1]})
        assert LT.scan("llama3.2-1b")[0] == {1, 2, 3}, "cut 1: C1-V / NAT-V-cen may be absent"
        _launch_dir(t, rows={**{a: [1, 2, 3] for a in LT.REQ + LT.OPT}, "C1-V@cen": [1, 2, 3, 3]})
        assert "duplicate" in (_stops(lambda: LT.scan("llama3.2-1b")) or "")
        _launch_dir(t, rows={**{a: [1, 2, 3] for a in LT.REQ + LT.OPT}, "C1-V+mu": [1, 2]})
        assert "partial" in (_stops(lambda: LT.scan("llama3.2-1b")) or "")
        _launch_dir(t, rows={**{a: [1, 2] for a in LT.REQ}, "C1-V": [1, 2, 3], "NAT-V-cen": [1, 2]}, g1=[1, 2])
        assert "incomplete" in (_stops(lambda: LT.scan("llama3.2-1b")) or "")
        _launch_dir(t, g1=[1, 2])
        assert "g1" in (_stops(lambda: LT.scan("llama3.2-1b")) or "")


def test_launcher_cuts():
    prev = LT.OUT
    try:
        _launcher_cuts()
    finally:
        LT.OUT = prev


def _launcher_cuts():
    with tempfile.TemporaryDirectory() as t:
        LT.OUT = Path(t)
        (LT.OUT / "rehearsal").mkdir(parents=True)
        def reh(sec):
            (LT.OUT / "rehearsal" / "rehearsal_summary.json").write_text(json.dumps({m: {"block_seconds_per_problem": sec[m]} for m in LT.MODELS}), encoding="utf-8")
            (LT.OUT / "design.json").unlink(missing_ok=True)
        reh({"qwen2.5-0.5b": 60, "llama3.2-1b": 30, "qwen2.5-1.5b": 60})          # 70 x 150 s x 1.25 = 3.6 h: no cut
        assert LT.project_and_cut(0)["cuts"] == []
        reh({"qwen2.5-0.5b": 140, "llama3.2-1b": 45, "qwen2.5-1.5b": 120})        # 70 x 305 x 1.25 = 7.4 h: no cut
        assert LT.project_and_cut(0)["cuts"] == []
        reh({"qwen2.5-0.5b": 200, "llama3.2-1b": 60, "qwen2.5-1.5b": 160})        # 8.75 h? 70 x 420 x 1.25 = 10.2 h -> cut 1 (6.8 h)
        d = LT.project_and_cut(0)
        assert d["cuts"] == [LT.CUT1] and d["n"] == 70, d
        reh({"qwen2.5-0.5b": 300, "llama3.2-1b": 90, "qwen2.5-1.5b": 240})        # 15.3 h -> cut 1 10.2 h -> n 50: 7.3 h
        d = LT.project_and_cut(0)
        assert d["cuts"] == [LT.CUT1, "n = 50"] and d["n"] == 50, d
        reh({"qwen2.5-0.5b": 500, "llama3.2-1b": 150, "qwen2.5-1.5b": 400})
        assert "exceeds the cap" in (_stops(lambda: LT.project_and_cut(0)) or "")


def test_launcher_constants():
    assert LT.CAP_H == 9.0 and LT.N == 70 and LT.CUT1 == RB.CUT1
    assert LT.MODELS == RB.MODELS


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


def main():
    bad = []
    for t in TESTS:
        try:
            t()
        except Exception as e:  # noqa: BLE001
            bad.append((t.__name__, repr(e)))
    print(f"test_rv6b: {len(TESTS) - len(bad)}/{len(TESTS)} pass")
    for n, e in bad:
        print("  FAIL", n, e)
    return not bad


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
