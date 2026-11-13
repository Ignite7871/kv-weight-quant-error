"""CPU / CUDA validation of the v5 core (PROTOCOL_V5.md sections 2, 3, 5, 11). Tiny random models with the real DynamicLinear / bitsandbytes NF4
path and the real HQQ 0.2.8.post1 path: a Qwen3 with QK-norm (head_dim 128, GQA), a Llama with full multi-head attention (head_dim 64, as
many KV heads as query heads; SmolLM2's layout), a Qwen2 (head_dim 64) and a Mistral (head_dim 128). LOGIC only on the CPU (CPU NF4 numerics differ
from CUDA). No network, no real tokenizer, no real weights: GM runs this on the pod before any download.

  venv/Scripts/python.exe research_v2/v5/tests/test_rv5.py [--device cuda] [--write <json>] [--only <name>]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
V5 = HERE.parent
RV = V5.parent
sys.path.insert(0, str(RV))
sys.path.insert(0, str(RV / "v4"))
sys.path.insert(0, str(V5))
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from transformers import LlamaConfig, LlamaForCausalLM, MistralConfig, MistralForCausalLM, Qwen2Config, Qwen2ForCausalLM, Qwen3Config, Qwen3ForCausalLM  # noqa: E402,E501

import numpy as np  # noqa: E402
import torch  # noqa: E402

import rv2.controls as C  # noqa: E402
import rv3.core as V  # noqa: E402
import rv4  # noqa: E402
import rv5  # noqa: E402
import rv5_hqq as H  # noqa: E402
from rv2.engine import _snapshot_last  # noqa: E402

DEV = "cpu"
PROMPT = [5, 17, 33, 2, 99, 64, 7, 81, 3, 1, 4]
REF = [3, 9, 27, 81, 1, 5, 7, 11, 13, 17, 19]
S = 6
TESTS, RESULTS = [], []
RV5_HQQ_SHA256 = "2284d5cf543f1adaaee89580ee887539e2cecdfeadf8977f2f9399bc8e43da84"     # f3a54cc:research_v2/v4/rv4_hqq.py
V4_IMPORTED = ("research_v2/v4/rv4.py", "research_v2/v4/v4_ops.py", "research_v2/v4/analyze_v4.py", "research_v2/v4/v4_sample.json",
               "research_v2/v4/V4_MODEL_MANIFEST.json", "research_v2/v4/plan/v4_plan_numbers.json")


def t(fn):
    TESTS.append(fn)
    return fn


def qwen3(seed=0):
    torch.manual_seed(seed)
    return Qwen3ForCausalLM(Qwen3Config(vocab_size=128, hidden_size=256, intermediate_size=256, num_hidden_layers=2, num_attention_heads=4,
                                        num_key_value_heads=2, head_dim=128, max_position_embeddings=2048, tie_word_embeddings=True,
                                        initializer_range=0.25, rope_theta=1e6, use_sliding_window=False)).eval()


def llama_mha(seed=0):
    torch.manual_seed(seed)
    return LlamaForCausalLM(LlamaConfig(vocab_size=128, hidden_size=256, intermediate_size=256, num_hidden_layers=2, num_attention_heads=4,
                                        num_key_value_heads=4, max_position_embeddings=2048, tie_word_embeddings=True, initializer_range=0.1,
                                        rope_theta=130000.0)).eval()


def qwen2(seed=0):
    torch.manual_seed(seed)
    return Qwen2ForCausalLM(Qwen2Config(vocab_size=128, hidden_size=256, intermediate_size=256, num_hidden_layers=2, num_attention_heads=4,
                                        num_key_value_heads=2, max_position_embeddings=2048, tie_word_embeddings=True, initializer_range=0.25,
                                        use_sliding_window=False)).eval()


def mistral(seed=0):
    torch.manual_seed(seed)
    return MistralForCausalLM(MistralConfig(vocab_size=128, hidden_size=256, intermediate_size=256, num_hidden_layers=2, num_attention_heads=2,
                                            num_key_value_heads=1, head_dim=128, max_position_embeddings=2048, sliding_window=None,
                                            tie_word_embeddings=False, initializer_range=0.1, rope_theta=1e6)).eval()


MODELS = {"qwen3-qknorm-hd128": qwen3, "llama-mha-hd64": llama_mha, "qwen2-hd64": qwen2, "mistral-hd128": mistral}
HD = {"qwen3-qknorm-hd128": 128, "llama-mha-hd64": 64, "qwen2-hd64": 64, "mistral-hd128": 128}
KV = {"qwen3-qknorm-hd128": 2, "llama-mha-hd64": 4, "qwen2-hd64": 2, "mistral-hd128": 1}
_BE = {}


def be(name, precision="fp16", quantizer="nf4"):
    k = (name, precision, quantizer)
    if k not in _BE:
        _BE[k] = rv5.build_backend(MODELS[name](), DEV, precision, quantizer)
    return _BE[k]


def P():
    return torch.tensor([PROMPT], device=DEV)


def run_arm(b, arm, alpha=1.0, slots=False, unit="U1", tier="T", return_cache=False):
    rl = rv4.reference_logits(b, P(), REF, S)
    return rv4.teacher_forced_arm_v4(b, P(), REF, S, arm, alpha, rv5.arm_seeds(unit, arm, alpha, 0, tier), rl, rv4.DTYPES[b.precision],
                                     slots=slots, return_cache=return_cache)


def same_cache(c1, c2):
    return all(torch.equal(a.keys, b.keys) and torch.equal(a.values, b.values) for a, b in zip(c1.layers, c2.layers))


@torch.no_grad()
def manual_step(b, cur, cache):
    """Independent: 4-bit pass (global flag), snapshot, crop, high-precision pass; returns (kF, kM, L)."""
    L = cache.get_seq_length()
    b.routing_state.use_4bit = True
    b.model(cur, past_key_values=cache, use_cache=True)
    kM = torch.stack([t_[0, :, 0, :] for lay in cache.layers for t_ in (lay.keys[..., L:L + 1, :], lay.values[..., L:L + 1, :])]).clone()
    cache.crop(L)
    b.routing_state.use_4bit = False
    b.model(cur, past_key_values=cache, use_cache=True)
    kF = torch.stack([t_[0, :, 0, :] for lay in cache.layers for t_ in (lay.keys[..., L:L + 1, :], lay.values[..., L:L + 1, :])]).clone()
    return kF, kM, L


# ----------------------------------------------------------------------------- 1. seeds and the locked imports
@t
def t01_anchor_seeds_are_v4_and_other_units_are_new():
    for arm in rv4.CORE_ARMS:
        for alpha in (0.0, 0.5, 1.0, 2.0):
            assert rv5.arm_seeds("U0", arm, alpha, 1302, "V4A") == rv4.arm_seeds("qwen2.5-0.5b", arm, alpha, 1302, "V4A"), (arm, alpha)
    assert rv5.tiers("U0") == rv5.TIERS_V4 and rv5.TIERS_V4["test"] == "V4A" and rv5.TIERS_V4["calibration"] == "V4calib"
    assert rv5.master_label("U3") == "rv5-2026-11-11|nf4|wikitext2" and rv5.master_label("U4") == "rv5-2026-11-11|hqq|gsm8k"
    s = {u: rv5.arm_seeds(u, "C1", 1.0, 7, "V5A") for u in rv5.ORDER}
    assert len(set(s.values())) == len(s), "every unit has its own draws (U3/U4 share the model with U0 but not the label)"
    assert s["U4"][0] == V.seed_for("qwen2.5-0.5b", "C1", 1.0, 7, "V5A", master="rv5-2026-11-11|hqq|gsm8k")


@t
def t02_v4_imports_unchanged_and_hqq_restored_byte_for_byte():
    lock = (RV / "V4_PROTOCOL_LOCK.txt").read_text(encoding="utf-8")
    want = {l.split("  ", 1)[1]: l.split("  ", 1)[0] for l in lock.splitlines() if len(l.split("  ", 1)) == 2 and len(l.split("  ", 1)[0]) == 64}
    for p in sorted((RV / "v4" / "deviations").glob("V4-D*.json")):                # v4's post-data deviations (D1) hold the effective hashes
        want.update(json.loads(p.read_text(encoding="utf-8"))["files"])
    assert want.get("research_v2/v4/v4_ops.py") == "d81fb225e96d07c6f75ed732f462cff69afd049e2e1e03e99db5d499e1287647", "v4 D1 record missing"
    root = RV.parent
    for rel in V4_IMPORTED:
        assert rel in want, f"{rel} is not in the v4 lock"
        got = hashlib.sha256((root / rel).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        assert got == want[rel], f"{rel} differs from the v4 lock"
    got = hashlib.sha256((V5 / "rv5_hqq.py").read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    assert got == RV5_HQQ_SHA256, "rv5_hqq.py is not the v4 r3 rv4_hqq.py byte for byte"


@t
def t03_units_and_order():
    assert rv5.ORDER == ["U0", "U1", "U2", "U3", "U4", "U5", "U6"] and rv5.NEVER_CUT == ("U0", "U1", "U2")
    assert rv5.UNITS["U0"][:3] == ("qwen2.5-0.5b", "nf4", "gsm8k") and rv5.UNITS["U6"][:3] == ("mistral-7b-v0.3", "hqq", "gsm8k")
    assert {rv5.UNITS[u][0] for u in ("U3", "U4", "U5", "U6")} == set(rv5.D_MODELS)
    man = rv5.models()
    v4m = json.loads((RV / "v4" / "V4_MODEL_MANIFEST.json").read_text(encoding="utf-8"))["models"]
    for m in ("qwen2.5-0.5b", "mistral-7b-v0.3"):
        assert man[m]["revision"] == v4m[m]["revision"] and man[m]["shards"] == v4m[m]["shards"], "the v4 pins, unchanged"
    assert man["qwen3-0.6b-base"]["revision"] == "da87bfb608c14b7cf20ba1ce41287e8de496c0cd" and man["qwen3-0.6b-base"]["geometry"]["qk_norm"] is True
    assert man["qwen3-0.6b-base"]["geometry"]["head_dim"] == 128 and man["smollm2-1.7b"]["geometry"]["head_dim"] == 64
    assert man["smollm2-1.7b"]["revision"] == "effd688a12921b4cc83e3312b6feb579f70f9c71" and man["smollm2-1.7b"]["geometry"]["kv_heads"] == 32
    assert all(not e["gated"] for e in man.values()), "section 0: ungated models only"
    assert rv5.dataset_entry()["revision"] == "b08601e04326c79dfdd32d625aee71d232d685c3"


# ----------------------------------------------------------------------------- 2. the core arms on the new geometries
@t
def t04_core_arms_new_geometries_integrity_and_alpha0():
    bad = []
    for name in ("qwen3-qknorm-hd128", "llama-mha-hd64"):
        b = be(name)
        cF = V.fp16_teacher_forced_cache(b, P(), REF, S)
        for arm in rv4.CORE_ARMS:
            r = run_arm(b, arm)
            if not (r["js"][0] == 0.0 and all(r[k] for k in rv4.FLAGS)):
                bad.append((name, arm, "flags"))
            r0, c0 = run_arm(b, arm, 0.0, return_cache=True)
            if not (same_cache(c0, cF) and all(x == 0.0 for x in r0["js"])):
                bad.append((name, arm, "alpha0"))
    assert not bad, bad


@t
def t05_c1_full_sphere_and_slot_norms_per_kv_head_new_geometries():
    """C1 on the unit sphere of R^head_dim (128 for Qwen3, 64 for the MHA layout) with each (layer, path, KV head) slot's own natural norm."""
    for name in ("qwen3-qknorm-hd128", "llama-mha-hd64"):
        b = be(name)
        _, cache = b.run(P(), None, False)
        kF, kM, L = manual_step(b, torch.tensor([[REF[0]]], device=DEV), cache)
        assert kF.shape[-1] == HD[name] and kF.shape[1] == KV[name]
        out, info = rv4.stored_entry_v4("C1", 1.0, kF, kM, torch.Generator(device=DEV).manual_seed(5), torch.Generator(device=DEV).manual_seed(6),
                                        torch.float16)
        D = kM.double() - kF.double()
        dpre = info["pre"] - kF.double()
        assert torch.allclose(dpre.norm(dim=-1), D.norm(dim=-1), rtol=1e-12, atol=0)
        if HD[name] == 128:
            assert float(dpre[..., 64:].abs().sum()) > 0
        g = torch.Generator(device=DEV).manual_seed(5)
        z = torch.randn(D.shape, generator=g, device=DEV, dtype=torch.float32).double()
        assert torch.allclose(dpre, D.norm(dim=-1, keepdim=True) * z / z.norm(dim=-1, keepdim=True), rtol=0, atol=1e-12)


@t
def t06_x13a_slot_shapes_at_head_dim():
    for name in ("qwen3-qknorm-hd128", "llama-mha-hd64"):
        r = run_arm(be(name), "NAT", slots=True)
        sl = r["_slots"]
        assert sl["S1K"].shape == (2, KV[name], HD[name]) and sl["normsV"].shape == (S, 2, KV[name])


@t
def t07_attention_snapshot_qk_norm_and_kv_heads():
    a = rv5.attention_snapshot(be("qwen3-qknorm-hd128").model)
    assert a["q_norm"] and a["k_norm"] and a["n_q_norm"] == 2 and a["k_proj_out"] == 2 * 128
    g = {"layers": 2, "qk_norm": True, "kv_heads": 2, "head_dim": 128}
    assert rv5.attention_matches(a, g) and not rv5.attention_matches(a, dict(g, qk_norm=False)) and not rv5.attention_matches(a, dict(g, kv_heads=1))
    m = rv5.attention_snapshot(be("llama-mha-hd64").model)
    assert not m["q_norm"] and m["k_proj_out"] == 4 * 64
    assert rv5.attention_matches(m, {"layers": 2, "qk_norm": False, "kv_heads": 4, "head_dim": 64})
    assert not rv5.attention_matches(m, {"layers": 2, "qk_norm": False, "kv_heads": 2, "head_dim": 64}), "GQA where the layout is MHA"


@t
def t08_bf16_rule_on_new_geometries():
    for name in ("qwen3-qknorm-hd128", "llama-mha-hd64"):
        b = be(name, "bf16")
        r, c = run_arm(b, "C1", return_cache=True)
        assert all(lay.keys.dtype == torch.bfloat16 for lay in c.layers) and r["rounding_ok"] and r["construction_ok"] and r["finite_ok"]


# ----------------------------------------------------------------------------- 3. HQQ (section 3, GH)
@t
def t09_hqq_config_twice_quantize_and_oracle():
    for name in ("qwen2-hd64", "qwen3-qknorm-hd128"):
        b = be(name, "fp16", "hqq")
        assert rv5.hqq_config_ok(b.model) and H.HQQ_CONFIG == {"nbits": 4, "group_size": 64, "axis": 1}
        m = H.hqq_modules(b.model)[0][1]
        assert m.hqq.meta["quant_zero"] is False and m.hqq.meta["quant_scale"] is False
        h2 = H.state_hash(rv5.build_backend(MODELS[name](), DEV, "fp16", "hqq").model)
        assert H.state_hash(b.model) == h2, "two quantizations of the same weights"
        o = rv5.hqq_oracle_ok(b.model)
        assert o["pass"] and o["n_layers"] == 7 * 2
        assert 4.4 < H.bits_per_weight(b.model) < 4.6


@t
def t10_hqq_measurement_differs_from_nf4_and_alpha0_bitwise():
    bq, bn = be("qwen2-hd64", "fp16", "hqq"), be("qwen2-hd64", "fp16", "nf4")
    rq, rn = run_arm(bq, "NAT"), run_arm(bn, "NAT")
    assert rq["js"] != rn["js"], "HQQ and NF4 measurement must differ"
    assert all(rq[k] for k in rv4.FLAGS) and rq["js"][0] == 0.0
    cF = V.fp16_teacher_forced_cache(bq, P(), REF, S)
    for arm in rv4.CORE_ARMS:
        r0, c0 = run_arm(bq, arm, 0.0, return_cache=True)
        assert same_cache(c0, cF) and all(x == 0.0 for x in r0["js"]), arm


@t
def t11_hqq_c1_norm_from_hqq_delta():
    """C1 under HQQ takes its per-slot norms from HQQ's own Delta = kappa^HQQ - kappa^F (not NF4's)."""
    b = be("qwen2-hd64", "fp16", "hqq")
    _, cache = b.run(P(), None, False)
    kF, kM, L = manual_step(b, torch.tensor([[REF[0]]], device=DEV), cache)
    m0 = H.hqq_modules(b.model)[0][1]
    x = torch.randn(1, 1, m0.in_features, dtype=torch.float16, device=DEV)
    b.routing_state.use_4bit = True
    y = m0(x)
    b.routing_state.use_4bit = False
    assert torch.equal(y, m0.hqq(x).to(torch.float16)), "the 4-bit route is the HQQ layer"
    _, c2 = b.run(P(), None, False)
    gen = C.make_generator(DEV, 11)
    with torch.no_grad():
        rv4.controlled_step_v4(b, torch.tensor([[REF[0]]], device=DEV), c2, "C1", 1.0, gen, gen, torch.float16)
    st = C._stack_slices(_snapshot_last(c2, L))
    n_got = (st.double() - kF.double()).norm(dim=-1)
    n_want = (kM.double() - kF.double()).norm(dim=-1)
    rel = ((n_got - n_want).abs() / n_want.clamp_min(1e-12)).max()
    assert float(rel) < 2e-2, float(rel)


@t
def t12_hqq_frozen_state_roundtrip():
    b = rv5.build_backend(qwen2(seed=3), DEV, "fp16", "hqq")
    h = H.state_hash(b.model)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "state.pt"
        H.save_state(b.model, p)
        b2 = rv5.build_backend(qwen2(seed=3), DEV, "fp16", "hqq", hqq_frozen=str(p))
        assert H.state_hash(b2.model) == h
        b3 = rv5.build_backend(qwen2(seed=4), DEV, "fp16", "hqq", hqq_frozen=str(p))
        assert H.state_hash(b3.model) == h, "a frozen state replaces the fresh quantization"
    try:
        rv5.build_backend(qwen2(), DEV, "fp16", "nf4", hqq_frozen="x")
        raise AssertionError("a frozen HQQ state with NF4 must be refused")
    except ValueError:
        pass


# ----------------------------------------------------------------------------- 4. WikiText rules (stub tokenizer: no network, no real files)
class StubTok:
    """Whitespace tokenizer with a BOS option; ids are deterministic per word."""

    def __init__(self, bos=None):
        self.bos = bos

    def __call__(self, text, add_special_tokens=True, return_tensors=None):
        ids = [int(hashlib.sha256(w.encode()).hexdigest()[:4], 16) for w in text.split()]
        if add_special_tokens and self.bos is not None:
            ids = [self.bos] + ids

        class R:
            input_ids = ids
        return R()


def text_of(n_words, seed=0):
    rng = np.random.default_rng(seed)
    return " ".join("w" + str(int(x)) + ("," if x % 7 == 0 else "") for x in rng.integers(0, 10000, n_words))


@t
def t13_wikitext_paragraphs_and_prompt_cut():
    lines = [" = Title = \n", "\n", " A body line . \n", " = = Section = = \n", "   \n", " Another @-@ body . \n"]
    assert rv5.paragraphs(lines) == [(2, "A body line ."), (5, "Another @-@ body .")]
    for seed in range(20):
        txt = text_of(400, seed)
        p, r = rv5.split_prompt(txt)
        assert p + r == txt and len(p) <= 600 and not p[-1].isspace() and r[0].isspace(), seed
        assert not any(c.isspace() for c in txt[len(p):600 + 1].lstrip()[:0])
        j = max(i for i, c in enumerate(txt[:601]) if c.isspace())
        assert len(p) <= j and txt[len(p):j + 1].isspace(), "cut at the last whitespace at or before 600"
    s = "x" * 599 + "  y" + "z" * 900
    p, r = rv5.split_prompt(s)
    assert p == "x" * 599 and r.startswith("  y"), "a whitespace run before the cut moves to the continuation"


@t
def t14_wikitext_eligibility_history_and_order():
    toks = {"a": StubTok(bos=1), "b": StubTok()}
    long_ = text_of(300, 1)
    assert len(long_) >= 1200
    ok, det = rv5.eligible(long_, toks)
    assert ok and set(det) == {"a", "b"} and det["a"]["history_len"] == 65
    p, r = rv5.split_prompt(long_)
    h = rv5.history_ids(toks["a"], r)
    assert len(h) == 65 and h[0] != 1, "history: no special tokens"
    assert rv5.prompt_token_ids(toks["a"], p)[0] == 1 and rv5.prompt_token_ids(toks["b"], p)[0] != 1, "prompt: the tokenizer's own BOS rule"
    assert not rv5.eligible(long_[:1199], toks)[0], "fewer than 1,200 characters"
    sparse = "a" * 590 + " " + "b" * 700 + " c d e"
    assert len(sparse) >= 1200 and not rv5.eligible(sparse, toks)[0], "fewer than 65 continuation tokens"
    paras = [(i, f"p{i}") for i in range(50)]
    o1, o2 = rv5.ordered(paras, "test"), rv5.ordered(list(reversed(paras)), "test")
    assert o1 == o2 and o1 != paras, "the order is seeded by the label, not by the file order"
    assert rv5.ordered(paras, "train") != o1


@t
def t15_wikitext_records_hold_no_text_or_ids():
    rec = {"index": 5, "js": [0.0, 0.1], "top": [17, 99], "flip": [0.0, 1.0]}
    w, g = rv5.public_record(rec, "wikitext2"), rv5.public_record(rec, "gsm8k")
    assert "top" not in w and w["top_sha256"] == rv5.ids_sha([17, 99]) and g["top"] == [17, 99]
    h = rv5.history_record(5, [3, 4, 5], "wikitext2")
    assert h == {"index": 5, "n": 3, "ids_sha256": rv5.ids_sha([3, 4, 5])} and rv5.history_matches(h, [3, 4, 5], "wikitext2")
    assert not rv5.history_matches(h, [3, 4, 6], "wikitext2") and rv5.history_record(5, [3], "gsm8k") == {"index": 5, "ids": [3]}


@t
def t16_wikitext_sample_file_hashes_only():
    """D-V5-5: the locked sample holds row indices, lengths and SHA-256 hashes only."""
    s = json.loads((V5 / "v5_wikitext_sample.json").read_text(encoding="utf-8"))
    assert (len(s["test"]), len(s["calibration"]), len(s["timing"])) == (100, 50, 3)
    assert s["revision"] == rv5.dataset_entry()["revision"]
    allowed = {"index", "text_sha256", "chars", "prompt_chars", "prompt_sha256", "continuation_sha256", "tok"}
    tok_allowed = {"prompt_ids_sha256", "prompt_len", "history_ids_sha256", "history_len"}
    idx = set()
    for part in ("test", "calibration", "timing"):
        for it in s[part]:
            assert set(it) == allowed and set(it["tok"]) == set(rv5.D_MODELS)
            assert all(set(v) == tok_allowed and v["history_len"] == 65 for v in it["tok"].values())
            assert it["chars"] >= 1200 and it["prompt_chars"] <= 600
            idx.add((part == "test", it["index"]))
    assert len(idx) == 153, "no repeated paragraph within a split"
    keys = [it["index"] for it in s["test"]]
    assert keys == [i for i, _ in rv5.ordered([(k, "") for k in keys], "test")], "the test sample is in the seeded order"


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
