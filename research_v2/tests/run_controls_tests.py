"""CPU validation suite for the E1 perturbation-matched controls (E1_SPEC.md section 9). LOGIC only (CPU NF4 numerics differ from CUDA).

Run:  venv/Scripts/python.exe research_v2/tests/run_controls_tests.py [--real]
  --real  also runs slow checks on real Qwen2.5-0.5B (FP16 + bitsandbytes CPU NF4), a few tokens.
Each test is independently meaningful; tests/mutation_check_controls.py injects known bugs and requires this suite to fail for each.
"""

import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from transformers import AutoTokenizer  # noqa: F401  (HF before torch)

import torch

import rv2  # noqa: F401
import rv2.controls as C
from rv2.backends import build_cpu_backend, real_model_cpu, tiny_qwen2
from rv2.engine import generate
from rv2.policies import StaticFP16
from rv2.stage1 import CACHE_ONLY, generate_cell

RESULTS = []
PROMPTS = [torch.tensor([[5, 17, 33, 2, 99, 64, 7, 81]]), torch.tensor([[3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]])]
N = 20
EOS = -1


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


def be(seed=0):
    return build_cpu_backend(tiny_qwen2(seed))


def cache_tensors(cache):
    return [(lay.keys.clone(), lay.values.clone()) for lay in cache.layers]


def run_capture(backend, fn):
    """Run fn() and return (result, final cache) by wrapping Backend.run."""
    cap = {}
    orig = backend.run

    def run(ids, cache, use_4bit):
        out = orig(ids, cache, use_4bit)
        cap["cache"] = out[1]
        return out
    backend.run = run
    try:
        res = fn()
    finally:
        backend.run = orig
    return res, cap["cache"]


def gc(backend, prompt, arm, alpha, seed=1, n=N):
    return run_capture(backend, lambda: C.generate_control(backend, prompt, arm, alpha, seed, n, EOS, diag=True))


@test
def c01_nat_equals_engine_cache_only_bitwise():
    for p in PROMPTS:
        b = be()
        eng, ec = run_capture(b, lambda: generate_cell(b, p, CACHE_ONLY, N, EOS))
        res, cc = gc(b, p, "NAT", 1.0)
        assert eng.ids == res.ids, "NAT tokens differ from the engine's cache-only tokens"
        assert all(torch.equal(a, c) and torch.equal(x, y) for (a, x), (c, y) in zip(cache_tensors(ec), cache_tensors(cc))), "NAT cache differs bitwise"


@test
def c02_alpha_zero_reproduces_pure_fp16_bitwise():
    for arm in ("C1", "C2"):
        for p in PROMPTS:
            b = be()
            ref, rc = run_capture(b, lambda: generate(b, p, StaticFP16(), N, EOS, 0))
            res, cc = gc(b, p, arm, 0.0)
            assert ref.ids == res.ids, f"{arm} alpha=0 tokens differ from FP16"
            assert all(torch.equal(a, c) and torch.equal(x, y) for (a, x), (c, y) in zip(cache_tensors(rc), cache_tensors(cc))), f"{arm} alpha=0 cache differs"


def synth(shape=(4, 2, 64), rel=0.02, seed=0):
    g = torch.Generator().manual_seed(seed)
    kF = (torch.randn(shape, generator=g) * 3).half()
    D = torch.randn(shape, generator=g) * rel * 3
    kN = (kF.float() + D).half()
    return kF, kN


@test
def c03_norm_matching_per_slot():
    kF, kN = synth()
    D = kN.float() - kF.float()
    for arm in ("C1", "C2"):
        for a in (0.5, 1.0, 2.0):
            g = C.make_generator("cpu", 7)
            new = C._stored_entry(arm, a, kF, kN, g)
            r = (new.float() - kF.float()).norm(dim=-1) / (a * D.norm(dim=-1))
            assert (r - 1).abs().max() < 0.03, f"{arm} alpha={a}: realized/target norm deviates up to {(r - 1).abs().max():.4f}"


@test
def c04_c2_preserves_coordinate_magnitudes_and_randomizes_signs():
    kF, kN = synth(shape=(6, 2, 64))
    D = kN.float() - kF.float()
    new = C._stored_entry("C2", 1.0, kF, kN, C.make_generator("cpu", 3))
    d2 = new.float() - kF.float()
    tol = new.float().abs() * 2.0 ** -10 + kF.float().abs() * 2.0 ** -10 + 1e-6      # fp16 rounding of the stored value (<= half an ulp each)
    assert ((d2.abs() - D.abs()).abs() <= tol).all(), "C2 changed coordinate magnitudes beyond fp16 rounding"
    agree = (torch.sign(d2) == torch.sign(D)).float().mean().item()
    assert 0.35 < agree < 0.65, f"C2 signs not randomized (agreement {agree:.2f})"


@test
def c05_c1_direction_is_unstructured():
    kF, kN = synth(shape=(24, 2, 64), seed=5)
    D = kN.float() - kF.float()
    cos = []
    for s in range(40):
        new = C._stored_entry("C1", 1.0, kF, kN, C.make_generator("cpu", s))
        d1 = new.float() - kF.float()
        cos.append(((d1 * D).sum(-1) / (d1.norm(dim=-1) * D.norm(dim=-1))).flatten())
    cos = torch.cat(cos)
    assert abs(cos.mean().item()) < 0.02, f"C1 direction correlated with D (mean cos {cos.mean():.3f})"
    assert 0.09 < cos.std().item() < 0.17, f"cosine spread {cos.std():.3f} unlike a uniform direction in R^64 (expected about 0.125)"


@test
def c06_determinism_and_seed_dependence():
    p = PROMPTS[0]
    b = be()
    r1, c1 = gc(b, p, "C1", 2.0, seed=11)
    r2, c2 = gc(b, p, "C1", 2.0, seed=11)
    r3, c3 = gc(b, p, "C1", 2.0, seed=12)
    assert r1.ids == r2.ids and all(torch.equal(a, c) for (a, _), (c, _) in zip(cache_tensors(c1), cache_tensors(c2))), "same seed not bitwise identical"
    assert not all(torch.equal(a, c) for (a, _), (c, _) in zip(cache_tensors(c1), cache_tensors(c3))), "different seeds gave identical caches"
    seeds = {C.seed_for(a, al, i, t, rep) for a in C.ARMS for al in (0.5, 1.0, 2.0) for i in range(5) for t in ("A", "B") for rep in (0, 1)}
    assert len(seeds) == 3 * 3 * 5 * 2 * 2, "seed collision"
    assert C.seed_for("C1", 1.0, 3, "A") == C.seed_for("C1", 1.0, 3, "A")


@test
def c07_teacher_forced_identical_history_and_first_step_zero():
    p = PROMPTS[1]
    b = be()
    ref = generate(b, p, StaticFP16(), N, EOS, 0).ids
    S = 12
    rl = C.reference_logits(b, p, ref, S)
    fed = {}
    orig = b.run

    def spy(ids, cache, use_4bit):
        if cache is not None:
            fed.setdefault(cur_arm[0], []).append(int(ids.flatten()[0]))
        return orig(ids, cache, use_4bit)
    cur_arm = [None]
    b.run = spy
    outs = {}
    try:
        for arm in C.ARMS:
            cur_arm[0] = arm
            outs[arm] = C.teacher_forced_arm(b, p, ref, S, arm, 1.0, C.seed_for(arm, 1.0, 0, "A"), rl)
    finally:
        b.run = orig
    first = {a: [t for i, t in enumerate(v) if i % 2 == 0] for a, v in fed.items()}   # per step: NF4 pass then FP16 pass consume the same token
    for a in C.ARMS:
        assert first[a] == ref[:S], f"{a} was not fed the reference token history"
        assert outs[a]["js"][0] == 0.0 and outs[a]["flip"][0] == 0.0, f"{a}: step 1 must equal the FP16 reference (no perturbed entry exists yet)"
        assert max(outs[a]["js"]) > 0, f"{a}: no effect at any step (test would be vacuous)"


@test
def c08_reference_teacher_forced_matches_free_running_fp16():
    p = PROMPTS[0]
    b = be()
    ref = generate(b, p, StaticFP16(), N, EOS, 0).ids
    rl = C.reference_logits(b, p, ref, N - 1)
    assert [int(t) for t in rl.argmax(-1)] == ref[1:N], "teacher-forced FP16 argmax differs from the free-running FP16 tokens"


@test
def c09_prefill_and_earlier_entries_never_modified():
    p = PROMPTS[0]
    b = be()
    L0 = p.shape[1]
    snaps = []
    orig = C.controlled_step

    def spy(backend, cur, cache, arm, alpha, gen, diag=True):
        out = orig(backend, cur, cache, arm, alpha, gen, diag)
        snaps.append(cache_tensors(cache))
        return out
    C.controlled_step = spy
    try:
        run_capture(b, lambda: C.generate_control(b, p, "C1", 2.0, 5, 10, EOS))
        pure = run_capture(b, lambda: generate(b, p, StaticFP16(), 10, EOS, 0))[1]
    finally:
        C.controlled_step = orig
    pt = cache_tensors(pure)
    for l in range(len(pt)):
        assert torch.equal(snaps[0][l][0][:, :, :L0], pt[l][0][:, :, :L0]), "prefill entries changed"
    for t in range(len(snaps) - 1):                       # entries written at step t are not touched at step t+1
        n = L0 + t + 1
        for l in range(len(pt)):
            assert torch.equal(snaps[t][l][0][:, :, :n], snaps[t + 1][l][0][:, :, :n]) and torch.equal(snaps[t][l][1][:, :, :n], snaps[t + 1][l][1][:, :, :n]), \
                "an accumulated entry was modified after its write step"


@test
def c10_perturbation_changes_only_newest_decode_entries():
    p = PROMPTS[0]
    b = be()
    res, cc = gc(b, p, "C1", 1.0, n=8)
    pure = run_capture(b, lambda: generate(b, p, StaticFP16(), 8, EOS, 0))[1]
    L0 = p.shape[1]
    kc, kp = cache_tensors(cc), cache_tensors(pure)
    assert torch.equal(kc[0][0][:, :, :L0], kp[0][0][:, :, :L0])
    assert not torch.equal(kc[0][0][:, :, L0:], kp[0][0][:, :, L0:]), "no decode entry was perturbed"


@test
def c11_diagnostics_report_matched_norms():
    b = be()
    for arm, a in (("C1", 1.0), ("C2", 1.0), ("C1", 0.5), ("C1", 2.0), ("NAT", 1.0)):
        res, _ = gc(b, PROMPTS[0], arm, a, n=10)
        d = res.trace["diag"]
        assert max(d["norm_dev_max"]) < 0.30, f"{arm} alpha={a}: realized norm deviates (max {max(d['norm_dev_max']):.3f}) in the tiny model"
        if arm == "NAT":
            assert max(d["norm_dev_max"]) == 0.0


@test
def c12_c1_noise_independent_across_slots():
    kF, kN = synth(shape=(24, 2, 64), seed=9)
    new = C._stored_entry("C1", 1.0, kF, kN, C.make_generator("cpu", 4))
    d1 = (new.float() - kF.float()).reshape(-1, 64)
    u = d1 / d1.norm(dim=-1, keepdim=True)
    G = u @ u.T
    off = G[~torch.eye(G.shape[0], dtype=torch.bool)]
    assert off.abs().mean().item() < 0.2 and off.abs().max().item() < 0.6, f"slot noise not independent (max |cos| {off.abs().max():.2f})"


@test
def c13_js64_matches_independent_formula():
    g = torch.Generator().manual_seed(1)
    a, b = torch.randn(1, 500, generator=g) * 3, torch.randn(1, 500, generator=g) * 3
    p, q = torch.softmax(a.double(), -1)[0], torch.softmax(b.double(), -1)[0]
    m = (p + q) / 2
    ref = 0.5 * float((p * (p / m).log()).sum()) + 0.5 * float((q * (q / m).log()).sum())
    assert abs(float(C.js64(a, b)) - ref) < 1e-12, "js64 differs from the textbook formula"
    assert abs(float(C.js64(a, b)) - float(C.js64(b, a))) < 1e-15, "js64 not symmetric"
    assert float(C.js64(a, a)) == 0.0, "js64 of identical logits must be exactly 0"
    assert 0 <= float(C.js64(a, b)) <= 0.6931471805599453


def main():
    real = "--real" in sys.argv
    tests = [v for k, v in sorted(globals().items()) if k.startswith("c") and k[1:3].isdigit() and callable(v)]
    for t in tests:
        t()
    if real:
        run_real()
    w = max(len(n) for n, _, _ in RESULTS)
    for n, s, m in RESULTS:
        print(f"{n:<{w}}  {s}  {m}")
    bad = [r for r in RESULTS if r[1] != "PASS"]
    print(f"\n{len(RESULTS) - len(bad)}/{len(RESULTS)} passed")
    sys.exit(1 if bad else 0)


@test
def r01_real_model_cpu_smoke():
    """Real Qwen2.5-0.5B on CPU: alpha=0 reproduces FP16 bitwise, NAT equals the engine's cache-only, norms matched, teacher forcing step 1 exact."""
    from rv2.backends import build_cpu_backend as bcb
    model = real_model_cpu()
    b = bcb(model)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    p = tok("Question: What is 7 times 8?\nAnswer:", return_tensors="pt").input_ids
    n = 5
    ref, rc = run_capture(b, lambda: generate(b, p, StaticFP16(), n, tok.eos_token_id, 0))
    z, zc = run_capture(b, lambda: C.generate_control(b, p, "C1", 0.0, 1, n, tok.eos_token_id))
    assert z.ids == ref.ids and all(torch.equal(a, c) for (a, _), (c, _) in zip(cache_tensors(rc), cache_tensors(zc))), "real model: alpha=0 != FP16"
    e, ec = run_capture(b, lambda: generate_cell(b, p, CACHE_ONLY, n, tok.eos_token_id))
    r, rcn = run_capture(b, lambda: C.generate_control(b, p, "NAT", 1.0, 1, n, tok.eos_token_id))
    assert e.ids == r.ids and all(torch.equal(a, c) and torch.equal(x, y) for (a, x), (c, y) in zip(cache_tensors(ec), cache_tensors(rcn))), "real model: NAT != cache-only"
    for arm, a in (("C1", 1.0), ("C2", 1.0), ("C1", 2.0)):
        g, _ = run_capture(b, lambda: C.generate_control(b, p, arm, a, 2, n, tok.eos_token_id))
        d = g.trace["diag"]
        print(f"   real {arm} alpha={a}: norm_dev_mean {sum(d['norm_dev_mean']) / len(d['norm_dev_mean']):.4f} max {max(d['norm_dev_max']):.4f}; "
              f"rel_scale_mean {sum(d['rel_scale_mean']) / len(d['rel_scale_mean']):.4f}")
        assert max(d["norm_dev_mean"]) < 0.01 and max(d["norm_dev_max"]) < 0.10, "real model: realized norm outside the G2 tolerance"
    S = 4
    rl = C.reference_logits(b, p, ref.ids, S)
    o = C.teacher_forced_arm(b, p, ref.ids, S, "C1", 1.0, 3, rl)
    assert o["js"][0] == 0.0, "real model: teacher-forced step 1 not exact"


def run_real():
    r01_real_model_cpu_smoke()


if __name__ == "__main__":
    main()
