"""CPU validation for the Llama E1 Tier A replication (E1L_SPEC.md): the whole Qwen E1 control suite (c01-c13) is re-run on a tiny random Llama (grouped-query
attention, 8 query / 2 KV heads, head_dim 64) with the real DynamicLinear/bitsandbytes NF4 path, plus Llama-specific cache-layout / dtype / alignment tests.
  venv/Scripts/python.exe research_v2/tests/run_e1l_tests.py [--real]     (--real: real Llama-3.2-1B-Instruct on CPU, a few tokens, ~8 min)
LOGIC only (CPU NF4 numerics differ from CUDA)."""

import copy
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from transformers import AutoTokenizer  # noqa: F401,E402

import torch  # noqa: E402

import e1l_support as L  # noqa: E402
import rv2.controls as C  # noqa: E402
import run_controls_tests as T  # noqa: E402
from rv2.engine import generate  # noqa: E402
from rv2.policies import StaticFP16  # noqa: E402
from rv2.stage1 import CACHE_ONLY, generate_cell  # noqa: E402


def install(backend_factory=L.tiny_backend_cpu, device="cpu"):
    T.be = backend_factory
    T.PROMPTS = [p.to(device) for p in [torch.tensor([[5, 17, 33, 2, 99, 64, 7, 81]]), torch.tensor([[3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]])]]


def lt(fn):
    def wrapper():
        try:
            fn()
            T.RESULTS.append((fn.__name__, "PASS", ""))
        except Exception as e:  # noqa: BLE001
            T.RESULTS.append((fn.__name__, "FAIL", f"{type(e).__name__}: {e}"))
            traceback.print_exc()
    wrapper.__name__ = fn.__name__
    return wrapper


@lt
def l01_cache_layout_fp16_storage_and_gqa_geometry():
    b = T.be()
    p = T.PROMPTS[0]
    g = L.cache_geometry(b, p)
    cfg = b.model.config
    assert g["n_layers"] == cfg.num_hidden_layers and g["n_kv_heads"] == cfg.num_key_value_heads and g["head_dim"] == cfg.hidden_size // cfg.num_attention_heads, g
    assert g["n_kv_heads"] < cfg.num_attention_heads, "test model must be grouped-query"
    assert g["all_fp16"], f"stored dtype is {g['key_dtype']}/{g['value_dtype']}, not float16"
    for arm, a in (("NAT", 1.0), ("C1", 1.0), ("C2", 2.0)):
        res, cache = T.gc(b, p, arm, a, n=6)
        for lay in cache.layers:
            assert lay.keys.dtype == torch.float16 and lay.values.dtype == torch.float16, f"{arm}: stored dtype changed"
            assert lay.keys.shape[1] == cfg.num_key_value_heads and lay.keys.shape[-1] == g["head_dim"]
            assert torch.isfinite(lay.keys).all() and torch.isfinite(lay.values).all(), f"{arm}: non-finite stored entry"


@lt
def l02_slot_alignment_layer_kind_head():
    b = T.be()
    p = T.PROMPTS[0]
    _, cache = b.run(p, None, use_4bit=False)
    n_l, n_kv, hd = len(cache.layers), cache.layers[0].keys.shape[1], cache.layers[0].keys.shape[-1]
    Lpos = cache.get_seq_length() - 1
    marker = torch.arange(2 * n_l * n_kv, dtype=torch.float32, device=cache.layers[0].keys.device).reshape(2 * n_l, n_kv, 1).expand(2 * n_l, n_kv, hd).clone() + 1000.0
    C._write_back(cache, Lpos, marker.half())
    for l in range(n_l):
        for h in range(n_kv):
            assert float(cache.layers[l].keys[0, h, Lpos, 0]) == float(marker[2 * l, h, 0]), f"K slot misaligned at layer {l} head {h}"
            assert float(cache.layers[l].values[0, h, Lpos, 0]) == float(marker[2 * l + 1, h, 0]), f"V slot misaligned at layer {l} head {h}"
    st = C._stack_slices([(lay.keys[:, :, Lpos:Lpos + 1, :], lay.values[:, :, Lpos:Lpos + 1, :]) for lay in cache.layers])
    assert torch.equal(st, marker.half()), "stack/unstack are not inverse"


@lt
def l03_independent_per_kv_head_norm_matching_from_cache_entries():
    b = T.be()
    p = T.PROMPTS[1]
    _, base = b.run(p, None, use_4bit=False)
    Lc = base.get_seq_length()
    cur = torch.tensor([[7]], device=p.device)
    ent = {}
    for arm, a in (("NAT", 1.0), ("C1", 1.0), ("C2", 1.0), ("C1", 2.0)):
        cache = copy.deepcopy(base)
        C.controlled_step(b, cur, cache, arm, a, C.make_generator(p.device, 5), diag=False)
        ent[(arm, a)] = torch.stack([t[0, :, Lc, :].float() for lay in cache.layers for t in (lay.keys, lay.values)])
    cache = copy.deepcopy(base)
    b.run(cur, cache, False)
    kF = torch.stack([t[0, :, Lc, :].float() for lay in cache.layers for t in (lay.keys, lay.values)])
    D = ent[("NAT", 1.0)] - kF
    tn = D.norm(dim=-1)
    for key in (("C1", 1.0), ("C2", 1.0), ("C1", 2.0)):
        rn = (ent[key] - kF).norm(dim=-1)
        tol = 2.0 ** -10 * kF.norm(dim=-1) + 0.02 * key[1] * tn
        assert ((rn - key[1] * tn).abs() <= tol).all(), f"{key}: realized per-(layer,K/V,KV-head) norm differs from alpha*||D||"
    assert (ent[("C1", 1.0)] != ent[("C2", 1.0)]).any()


@lt
def l04_gqa_shapes_in_stacked_diagnostics():
    b = T.be()
    res, _ = T.gc(b, T.PROMPTS[0], "C1", 1.0, n=5)
    cfg = b.model.config
    d = res.trace["diag"]
    assert len(d["norm_dev_max"]) == 4 and max(d["norm_dev_max"]) < 0.3 and all(0.0 < x < 5.0 for x in d["rel_scale_mean"])
    assert cfg.num_key_value_heads == 2 and cfg.num_attention_heads == 8


def run_real_llama():
    b = L.real_backend_cpu()
    tok = AutoTokenizer.from_pretrained(L.MODEL)
    p = tok("Question: What is 7 times 8?\nAnswer:", return_tensors="pt").input_ids
    g = L.cache_geometry(b, p)
    print("   real Llama cache geometry:", g)
    assert g["all_fp16"] and g["n_layers"] == 16 and g["n_kv_heads"] == 8 and g["head_dim"] == 64, g
    n = 5
    ref, rc = T.run_capture(b, lambda: generate(b, p, StaticFP16(), n, tok.eos_token_id, 0))
    z, zc = T.run_capture(b, lambda: C.generate_control(b, p, "C1", 0.0, 1, n, tok.eos_token_id))
    assert z.ids == ref.ids and all(torch.equal(a, c) and torch.equal(x, y) for (a, x), (c, y) in zip(T.cache_tensors(rc), T.cache_tensors(zc))), "real Llama: alpha=0 != FP16"
    e, ec = T.run_capture(b, lambda: generate_cell(b, p, CACHE_ONLY, n, tok.eos_token_id))
    r, rcn = T.run_capture(b, lambda: C.generate_control(b, p, "NAT", 1.0, 1, n, tok.eos_token_id))
    assert e.ids == r.ids and all(torch.equal(a, c) and torch.equal(x, y) for (a, x), (c, y) in zip(T.cache_tensors(ec), T.cache_tensors(rcn))), "real Llama: NAT != cache-only"
    for arm, a in (("C1", 1.0), ("C2", 1.0), ("C1", 2.0)):
        gen, gc_ = T.run_capture(b, lambda: C.generate_control(b, p, arm, a, 2, n, tok.eos_token_id))
        d = gen.trace["diag"]
        print(f"   real Llama {arm} alpha={a}: norm_dev_mean {sum(d['norm_dev_mean']) / len(d['norm_dev_mean']):.4f} max {max(d['norm_dev_max']):.4f} rel_scale {sum(d['rel_scale_mean']) / len(d['rel_scale_mean']):.4f}")
        assert max(d["norm_dev_mean"]) < 0.01 and max(d["norm_dev_max"]) < 0.10
        assert all(l.keys.dtype == torch.float16 and l.values.dtype == torch.float16 for l in gc_.layers)
    rl = C.reference_logits(b, p, ref.ids, 4)
    o = C.teacher_forced_arm(b, p, ref.ids, 4, "C1", 1.0, 3, rl)
    assert o["js"][0] == 0.0, "real Llama: teacher-forced step 1 not exact"


L_TESTS = [l01_cache_layout_fp16_storage_and_gqa_geometry, l02_slot_alignment_layer_kind_head, l03_independent_per_kv_head_norm_matching_from_cache_entries,
           l04_gqa_shapes_in_stacked_diagnostics]


def register_into_T():
    for f in L_TESTS:
        setattr(T, f.__name__, f)
    return sorted(k for k in dir(T) if k[0] in "cl" and k[1:3].isdigit() and callable(getattr(T, k)))


def main():
    install()
    names = register_into_T()
    T.RESULTS.clear()
    for n in names:
        getattr(T, n)()
    if "--real" in sys.argv:
        try:
            run_real_llama()
            T.RESULTS.append(("r01_real_llama_cpu_smoke", "PASS", ""))
        except Exception as e:  # noqa: BLE001
            T.RESULTS.append(("r01_real_llama_cpu_smoke", "FAIL", f"{type(e).__name__}: {e}"))
            traceback.print_exc()
    w = max(len(n) for n, _, _ in T.RESULTS)
    for n, s, m in T.RESULTS:
        print(f"{n:<{w}}  {s}  {m}")
    bad = [r for r in T.RESULTS if r[1] != "PASS"]
    print(f"\n{len(T.RESULTS) - len(bad)}/{len(T.RESULTS)} passed")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
