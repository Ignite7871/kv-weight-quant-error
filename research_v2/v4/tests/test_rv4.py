"""CPU validation of the v4 core (PROTOCOL_V4.md section 12). Tiny random models with the real DynamicLinear / bitsandbytes NF4 path and the real
Qwen2 with head_dim 64 and 128, Llama with head_dim 64 and 128, a GQA Llama with 4 KV heads, and Mistral (head_dim 128, RoPE 1e6, no
sliding window). LOGIC only (CPU NF4 numerics differ from CUDA). Checks are computed independently of the code under test where possible.

  venv/Scripts/python.exe research_v2/v4/tests/test_rv4.py [--device cuda] [--write <json>]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
V4 = HERE.parent
RV = V4.parent
sys.path.insert(0, str(RV))
sys.path.insert(0, str(V4))
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # r4: before anything imports torch

from transformers import LlamaConfig, LlamaForCausalLM, MistralConfig, MistralForCausalLM, Qwen2Config, Qwen2ForCausalLM  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import rv2.controls as C  # noqa: E402
import rv3.core as V  # noqa: E402
import rv4  # noqa: E402
from rv2.engine import _snapshot_last  # noqa: E402

DEV = "cpu"
PROMPT = [5, 17, 33, 2, 99, 64, 7, 81, 3, 1, 4]
REF = [3, 9, 27, 81, 1, 5, 7, 11, 13, 17, 19]
S = 6
TESTS, RESULTS = [], []


def t(fn):
    TESTS.append(fn)
    return fn


def qwen(hd, seed=0):
    torch.manual_seed(seed)
    heads = 256 // hd
    return Qwen2ForCausalLM(Qwen2Config(vocab_size=128, hidden_size=256, intermediate_size=256, num_hidden_layers=2, num_attention_heads=heads,
                                        num_key_value_heads=max(1, heads // 2), max_position_embeddings=2048, tie_word_embeddings=True,
                                        initializer_range=0.25, use_sliding_window=False)).eval()


def llama(hd, kv=2, heads=None, seed=0):
    torch.manual_seed(seed)
    heads = heads or 512 // hd
    return LlamaForCausalLM(LlamaConfig(vocab_size=128, hidden_size=heads * hd, intermediate_size=256, num_hidden_layers=2, num_attention_heads=heads,
                                        num_key_value_heads=kv, max_position_embeddings=2048, tie_word_embeddings=True, initializer_range=0.1,
                                        rope_theta=500000.0)).eval()


def mistral(seed=0):
    torch.manual_seed(seed)
    return MistralForCausalLM(MistralConfig(vocab_size=128, hidden_size=256, intermediate_size=256, num_hidden_layers=2, num_attention_heads=2,
                                            num_key_value_heads=1, head_dim=128, max_position_embeddings=2048, sliding_window=None,
                                            tie_word_embeddings=False, initializer_range=0.1, rope_theta=1e6)).eval()


MODELS = {"qwen-hd64": lambda: qwen(64), "qwen-hd128": lambda: qwen(128), "llama-hd64": lambda: llama(64), "llama-hd128": lambda: llama(128),
          "llama-gqa4kv": lambda: llama(128, kv=4, heads=8), "mistral-hd128": mistral}
HD = {"qwen-hd64": 64, "qwen-hd128": 128, "llama-hd64": 64, "llama-hd128": 128, "llama-gqa4kv": 128, "mistral-hd128": 128}
_BE = {}


def be(name, precision="fp16", quantizer="nf4"):
    k = (name, precision, quantizer)
    if k not in _BE:
        _BE[k] = rv4.build_backend(MODELS[name](), DEV, precision, quantizer)
    return _BE[k]


def P():
    return torch.tensor([PROMPT], device=DEV)


def seeds(arm, alpha=1.0, idx=0, tier="T"):
    return rv4.arm_seeds("tiny", arm, alpha, idx, tier)


def run_arm(b, arm, alpha=1.0, slots=False, tier="T", acc=None, return_cache=False, dtype=None):
    dtype = dtype or rv4.DTYPES[b.precision]
    rl = rv4.reference_logits(b, P(), REF, S)
    return rv4.teacher_forced_arm_v4(b, P(), REF, S, arm, alpha, seeds(arm, alpha, tier=tier), rl, dtype, acc=acc, slots=slots, return_cache=return_cache)


def pure_cache(b):
    return V.fp16_teacher_forced_cache(b, P(), REF, S)


def same_cache(c1, c2):
    return all(torch.equal(a.keys, b.keys) and torch.equal(a.values, b.values) for a, b in zip(c1.layers, c2.layers))


# ----------------------------------------------------------------------------- 1. the FP16 path is the locked v3 path
@t
def t01_fp16_equals_locked_v3_arm():
    """Every core arm, FP16 + NF4: records and final cache equal rv3.core.teacher_forced_arm_v3 (same seeds) bitwise."""
    bad = []
    for name in MODELS:
        b = be(name)
        rl = rv4.reference_logits(b, P(), REF, S)
        for arm in rv4.CORE_ARMS:
            sd = seeds(arm)
            r4, c4 = rv4.teacher_forced_arm_v4(b, P(), REF, S, arm, 1.0, sd, rl, torch.float16, return_cache=True)
            r3, c3 = V.teacher_forced_arm_v3(b, P(), REF, S, arm, 1.0, sd, rl, None, None, False, return_cache=True)
            if not (r4["js"] == r3["js"] and r4["top"] == r3["top"] and same_cache(c4, c3)):
                bad.append((name, arm))
    assert not bad, bad


@t
def t02_generic_construction_equals_locked_fp16():
    """stored_entry_generic with FP16 storage equals the locked stored_entry_v3 bitwise (all arms, alpha 0 / 0.5 / 1 / 2, head_dim 64 and 128),
    so the BF16 path builds the same float64 entry from the same draws."""
    for hd in (64, 128):
        g0 = torch.Generator().manual_seed(1)
        kF = (torch.randn(8, 2, hd, generator=g0) * 3).half()
        kM = (kF.float() + 0.05 * torch.randn(8, 2, hd, generator=g0)).half()
        for arm in rv4.CORE_ARMS:
            for alpha in ((0.0, 1.0) if arm in rv4.NATURAL else (0.0, 0.5, 1.0, 2.0)):
                ga, gb = torch.Generator().manual_seed(7), torch.Generator().manual_seed(7)
                a, ia = rv4.stored_entry_generic(arm, alpha, kF, kM, ga, torch.float16)
                b_, ib = V.stored_entry_v3(arm, alpha, kF, kM, gb, torch.Generator().manual_seed(8), None)
                assert torch.equal(a, b_) and torch.equal(ia["exact_rows"], ib["exact_rows"]), (hd, arm, alpha)
                assert torch.equal(ia["pre"], ib["pre"]), (hd, arm, alpha)


# ----------------------------------------------------------------------------- 2. alpha = 0, natural and constructed arms
@t
def t03_alpha0_bitwise_all_modes():
    """G1 logic: every arm at scale 0 reproduces the pure high-precision cache bitwise with JS exactly 0 (FP16 and BF16)."""
    bad = []
    for name in ("qwen-hd128", "llama-hd64", "mistral-hd128"):
        for prec in ("fp16", "bf16"):
            b = be(name, prec)
            cF = pure_cache(b)
            for arm in rv4.CORE_ARMS:
                r, c0 = run_arm(b, arm, 0.0, return_cache=True)
                if not (same_cache(c0, cF) and all(x == 0.0 for x in r["js"]) and r["routing_reset"]):
                    bad.append((name, prec, arm))
    assert not bad, bad


@torch.no_grad()
def manual_step(b, cur, cache):
    """Independent: 4-bit pass (global flag), snapshot, crop, high-precision pass; returns (kF, kM) stacked."""
    L = cache.get_seq_length()
    b.routing_state.use_4bit = True
    b.model(cur, past_key_values=cache, use_cache=True)
    kM = torch.stack([t_[0, :, 0, :] for lay in cache.layers for t_ in (lay.keys[..., L:L + 1, :], lay.values[..., L:L + 1, :])]).clone()
    cache.crop(L)
    b.routing_state.use_4bit = False
    b.model(cur, past_key_values=cache, use_cache=True)
    kF = torch.stack([t_[0, :, 0, :] for lay in cache.layers for t_ in (lay.keys[..., L:L + 1, :], lay.values[..., L:L + 1, :])]).clone()
    return kF, kM, L


@t
def t04_natural_arms_store_measured_entry_on_own_path():
    for name in ("qwen-hd64", "llama-gqa4kv", "mistral-hd128"):
        b = be(name)
        for arm, path in (("NAT", "KV"), ("NAT-K", "K"), ("NAT-V", "V")):
            _, cache = b.run(P(), None, False)
            cur = torch.tensor([[REF[0]]], device=DEV)
            _, c2 = b.run(P(), None, False)
            kF, kM, L = manual_step(b, cur, c2)
            gen = C.make_generator(DEV, 1)
            V.measure_pass(b, cur, cache, "NF4")
            rv4.controlled_step_v4  # noqa: B018
            cache.crop(L)
            _, cache = b.run(P(), None, False)
            rv4.controlled_step_v4(b, cur, cache, arm, 1.0, gen, gen, torch.float16)
            st = C._stack_slices(_snapshot_last(cache, L))
            for r in range(st.shape[0]):
                want = kM[r] if ("K" if r % 2 == 0 else "V") in path else kF[r]
                assert torch.equal(st[r], want), (name, arm, r)


@t
def t05_constructed_arms_norm_path_and_full_sphere():
    """C1 / C1-K / C1-V: pre-rounding per-slot norm equals the natural slot norm (float64, 1e-12), only the declared path changes, and the
    direction is the same Gaussian draw over the FULL head_dim (independent re-draw), with nonzero mass in coordinates >= 64 when head_dim is 128."""
    for hd in (64, 128):
        g0 = torch.Generator().manual_seed(3)
        kF = (torch.randn(6, 4, hd, generator=g0)).half()
        kM = (kF.float() + 0.1 * torch.randn(6, 4, hd, generator=g0)).half()
        D = kM.double() - kF.double()
        for arm, rows in (("C1", slice(None)), ("C1-K", slice(0, None, 2)), ("C1-V", slice(1, None, 2))):
            out, info = rv4.stored_entry_v4(arm, 1.0, kF, kM, torch.Generator().manual_seed(11), torch.Generator().manual_seed(12), torch.float16)
            pre = info["pre"]
            dpre = (pre - kF.double())[rows]
            assert torch.allclose(dpre.norm(dim=-1), D[rows].norm(dim=-1), rtol=1e-12, atol=0), (hd, arm)
            other = torch.ones(6, dtype=torch.bool)
            other[rows] = False
            assert torch.equal(out[other], kF[other]), (hd, arm, "other path changed")
            g = torch.Generator().manual_seed(11)
            shape = D[rows].shape
            z = torch.randn(shape, generator=g, dtype=torch.float32).double()
            u = z / z.norm(dim=-1, keepdim=True)
            assert torch.allclose(dpre, D[rows].norm(dim=-1, keepdim=True) * u, rtol=0, atol=1e-12), (hd, arm, "direction")
            if hd == 128:
                assert float(dpre[..., 64:].abs().sum()) > 0


@t
def t06_slot_norm_per_kv_head():
    """Each (layer, path, KV head) slot keeps its own natural norm: slots with very different natural norms stay different after C1."""
    kF = torch.zeros(2, 4, 128).half()
    kM = kF.clone()
    for h in range(4):
        kM[:, h, 0] = 0.01 * (10 ** h)
    out, info = rv4.stored_entry_v4("C1", 1.0, kF, kM, torch.Generator().manual_seed(1), torch.Generator().manual_seed(2), torch.float16)
    n = (info["pre"] - kF.double()).norm(dim=-1)
    want = (kM.double() - kF.double()).norm(dim=-1)
    assert torch.allclose(n, want, rtol=1e-12, atol=0)


# ----------------------------------------------------------------------------- 3. rounding and the BF16 rule
@t
def t07_bf16_rounding_equals_independent_oracle():
    g = torch.Generator().manual_seed(5)
    x = torch.randn(200000, generator=g, dtype=torch.float64) * torch.exp(torch.randn(200000, generator=g, dtype=torch.float64) * 8)
    # exact ties: midpoints between adjacent BF16 values (both parities), subnormal range, signed zeros
    b = torch.randint(0x0001, 0x7F7E, (20000,), generator=g, dtype=torch.int64)
    lo = (b << 16).to(torch.int32).view(torch.float32).double()
    hi = ((b + 1) << 16).to(torch.int32).view(torch.float32).double()
    ties = (lo + hi) / 2
    sub = torch.randn(5000, generator=g, dtype=torch.float64) * 2.0 ** -130
    x = torch.cat([x, ties, -ties, sub, torch.tensor([0.0, -0.0, 2.0 ** -134, 2.0 ** -133 * 1.5])])
    a, o = rv4.to_bf16(x), rv4.bf16_rne_oracle(x)
    assert torch.equal(a.view(torch.int16), o.view(torch.int16))
    naive = x.float().to(torch.bfloat16)                               # double rounding through float32
    assert not torch.equal(naive.view(torch.int16), o.view(torch.int16)), "the test values must contain double-rounding cases"
    bmax = 3.3895313892515355e38                                      # (2 - 2^-7) 2^127; overflow at bmax + 2^119
    o = rv4.bf16_rne_oracle(torch.tensor([bmax * 1.0001, bmax + 2.0 ** 119, 3.4e38], dtype=torch.float64))
    assert float(o[0]) == bmax and torch.isinf(o[1:]).all()


@t
def t08_bf16_mode_end_to_end():
    for name in ("qwen-hd128", "llama-hd64", "mistral-hd128"):
        b = be(name, "bf16")
        assert all(isinstance(m, rv4.DynamicLinearBF16) for _, m in V.dynamic_modules(b.model))
        r, c = run_arm(b, "C1", return_cache=True)
        assert all(lay.keys.dtype == torch.bfloat16 and lay.values.dtype == torch.bfloat16 for lay in c.layers)
        assert r["rounding_ok"] and r["construction_ok"] and r["finite_ok"] and r["js"][0] == 0.0
        x = torch.randn(1, 1, b.model.config.hidden_size, dtype=torch.bfloat16, device=DEV)   # r7: on the model's device
        m = V.dynamic_modules(b.model)[0][1]
        b.routing_state.use_4bit = True
        y4 = m(x)
        b.routing_state.use_4bit = False
        y16 = m(x)
        assert y4.dtype == y16.dtype == torch.bfloat16 and not torch.equal(y4, y16)


@t
def t09_fp16_rounding_check_active():
    for name in ("qwen-hd64", "llama-hd128"):
        r = run_arm(be(name), "C1-V")
        assert r["rounding_ok"] and r["construction_ok"] and r["exact_ok"]


# ----------------------------------------------------------------------------- 4. seeds, routing, sums, calibration
@t
def t10_seeds_labels_only():
    a, b_ = rv4.arm_seeds("m", "C1", 1.0, 5, "T"), rv4.arm_seeds("m", "C1", 1.0, 5, "T")
    assert a == b_ and a != rv4.arm_seeds("m", "C1", 1.0, 6, "T") and a != rv4.arm_seeds("m2", "C1", 1.0, 5, "T")
    assert a[0] == V.seed_for("m", "C1", 1.0, 5, "T", master="rv4-scale-2026-11-02") and a[0] != V.seed_for("m", "C1", 1.0, 5, "T")
    b = be("qwen-hd128")
    assert run_arm(b, "C1")["js"] == run_arm(b, "C1")["js"]
    assert run_arm(b, "C1")["js"] != run_arm(b, "C1", tier="other")["js"]


@t
def t11_measure_pass_all_4bit_and_reset():
    for name in ("llama-hd64", "qwen-hd128"):
        b = be(name, "fp16")
        _, cache = b.run(P(), None, False)
        with V.precision_probe(b.model) as log:
            V.measure_pass(b, torch.tensor([[3]], device=DEV), cache, "NF4")
        assert log and all(x for _, x in log) and V.overrides_clear(b.model)
        with V.precision_probe(b.model) as log2:
            b.run(torch.tensor([[3]], device=DEV), cache, False)
        assert log2 and not any(x for _, x in log2)


@t
def t12_x13a_sums_independent():
    for name in ("qwen-hd64", "llama-gqa4kv"):
        b = be(name)
        r = run_arm(b, "NAT", slots=True)
        sl = r["_slots"]
        _, cache = b.run(P(), None, False)
        S1K = S1V = None
        nK, nV = [], []
        for s in range(1, S + 1):
            kF, kM, L = manual_step(b, torch.tensor([[REF[s - 1]]], device=DEV), cache)
            C._write_back(cache, L, kM)
            d = kM.double() - kF.double()
            S1K = d[0::2] if S1K is None else S1K + d[0::2]
            S1V = d[1::2] if S1V is None else S1V + d[1::2]
            nK.append(d[0::2].norm(dim=-1))
            nV.append(d[1::2].norm(dim=-1))
        assert np.allclose(sl["S1K"], S1K.cpu().numpy(), rtol=0, atol=1e-12) and np.allclose(sl["S1V"], S1V.cpu().numpy(), rtol=0, atol=1e-12)   # r7: .cpu()
        assert np.allclose(sl["normsK"], torch.stack(nK).cpu().numpy(), atol=1e-12) and np.allclose(sl["normsV"], torch.stack(nV).cpu().numpy(), atol=1e-12)
        assert sl["S1K"].shape[-1] == HD[name] and sl["normsK"].shape[0] == S


@t
def t13_rho_accumulator_independent():
    b = be("llama-hd128")
    acc = rv4.RhoAccumulator()
    run_arm(b, "NAT", acc=acc)
    st = acc.state()
    _, cache = b.run(P(), None, False)
    rk = []
    for s in range(1, S + 1):
        kF, kM, L = manual_step(b, torch.tensor([[REF[s - 1]]], device=DEV), cache)
        C._write_back(cache, L, kM)
        rk.append(((kM.double() - kF.double()).norm(dim=-1) / kF.double().norm(dim=-1))[0::2].flatten())
    assert torch.allclose(st["rhoK"].cpu(), torch.cat(rk).cpu(), rtol=0, atol=1e-12) and st["n_meas"] == S   # r7: compared on the CPU


@t
def t17_cache_geometry_and_config_snapshot():
    for name in MODELS:
        for prec in ("fp16", "bf16"):
            b = be(name, prec)
            g = rv4.cache_geometry(b, P(), rv4.DTYPES[prec])
            cfg = b.model.config
            assert g["head_dim"] == HD[name] and g["n_kv_heads"] == cfg.num_key_value_heads and g["all_mode_dtype"]
            snap = rv4.config_snapshot(cfg)
            assert snap["head_dim"] == HD[name] and rv4.geometry_matches(snap, dict(snap))
            bad = dict(snap, tied_embeddings=not snap["tied_embeddings"])
            assert not rv4.geometry_matches(snap, bad)
            assert b.n_dynamic == 7 * cfg.num_hidden_layers


@t
def t18_integrity_flags_finite_and_step1():
    for name in ("qwen-hd128", "mistral-hd128"):
        for arm in rv4.CORE_ARMS:
            r = run_arm(be(name), arm)
            assert r["js"][0] == 0.0 and all(r[k] for k in rv4.FLAGS), (name, arm)


@t
def t19_past_entries_never_revisited():
    """A controlled step changes only the new position: every earlier cache entry (prefill and earlier decode writes) is bitwise unchanged."""
    for name in ("qwen-hd128", "llama-hd64"):
        b = be(name)
        gen = C.make_generator(DEV, 3)
        _, cache = b.run(P(), None, False)
        for s in range(1, 4):
            L = cache.get_seq_length()
            before = [(lay.keys[..., :L, :].clone(), lay.values[..., :L, :].clone()) for lay in cache.layers]
            with torch.no_grad():
                rv4.controlled_step_v4(b, torch.tensor([[REF[s - 1]]], device=DEV), cache, "C1", 1.0, gen, gen, torch.float16)
            assert cache.get_seq_length() == L + 1
            for (k0, v0), lay in zip(before, cache.layers):
                assert torch.equal(k0, lay.keys[..., :L, :]) and torch.equal(v0, lay.values[..., :L, :]), (name, s)


@t
def t21_fp16_storage_correctly_rounded_on_double_rounding_cases():
    """FP16 storage (the locked one-step rounding) equals the independent oracle on values where rounding through float32 first differs."""
    g = torch.Generator().manual_seed(9)
    h = torch.randint(0x0400, 0x7BFE, (50000,), generator=g, dtype=torch.int64)
    lo = h.to(torch.int16).view(torch.float16).double()
    hi = (h + 1).to(torch.int16).view(torch.float16).double()
    mid = (lo + hi) / 2
    eps = (hi - lo) * 2.0 ** -40                                         # just above / below the midpoint: float32 rounds it ONTO the midpoint
    x = torch.cat([mid + eps, mid - eps, mid, -(mid + eps)])
    a = rv4.to_store(x, torch.float16)
    assert torch.equal(a.view(torch.int16), V.fp16_rne_reference(x).view(torch.int16))
    assert not torch.equal(x.float().half().view(torch.int16), V.fp16_rne_reference(x).view(torch.int16))


@t
def t22_reload_releases_the_previous_model():
    """Real dual-weight backends, used for a real arm: when the next load starts, the previous model is gone, and on CUDA its memory is back."""
    import gc
    import weakref
    import run_v4 as R
    _BE.clear()
    gc.collect()
    on_cuda = DEV.startswith("cuda")
    if on_cuda:
        torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated() if on_cuda else 0
    refs = []

    def loader(quantizer):
        assert all(r() is None for r in refs), "the previous model is still alive when the next load starts"
        if on_cuda:
            assert torch.cuda.memory_allocated() <= base + 64 * 1024, (torch.cuda.memory_allocated(), base)
        b = rv4.build_backend(MODELS["qwen-hd128"](), DEV, "fp16", quantizer)
        run_arm(b, "NAT")
        refs.append(weakref.ref(b.model))
        return b, object()
    slot = R.ModelSlot()
    for q in ("nf4", "nf4", "nf4"):
        slot.load(loader, q)
    assert len(refs) == 3 and refs[-1]() is slot.be.model
    slot.drop()
    assert all(r() is None for r in refs)


def main(argv=None):
    global DEV
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--write")
    ap.add_argument("--only")
    a = ap.parse_args(argv)
    DEV = a.device
    rv4.apply_determinism()
    _BE.clear()
    RESULTS.clear()
    for fn in TESTS:
        if a.only and a.only not in fn.__name__:
            continue
        t0 = time.time()
        try:
            fn()
            RESULTS.append({"test": fn.__name__, "ok": True, "seconds": round(time.time() - t0, 2)})
            print(f"PASS {fn.__name__} ({time.time() - t0:.1f}s)", flush=True)
        except Exception as e:  # noqa: BLE001
            RESULTS.append({"test": fn.__name__, "ok": False, "error": f"{type(e).__name__}: {e}"[:500]})
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
    n_ok = sum(r["ok"] for r in RESULTS)
    print(f"{n_ok}/{len(RESULTS)} passed on {DEV}")
    if a.write:
        Path(a.write).write_text(json.dumps({"device": DEV, "torch": torch.__version__, "passed": n_ok, "n": len(RESULTS), "results": RESULTS}, indent=1))
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
