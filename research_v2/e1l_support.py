"""Llama-3.2-1B-Instruct support for the E1 Tier A replication (E1L_SPEC.md). New file; imports the locked Qwen E1 modules unchanged.

* tiny random Llama (GQA, head_dim 64) for CPU/GPU logic and mutation tests,
* CPU/GPU dual-weight backends for any HF causal LM (same DynamicLinear replacement as the Qwen pipeline),
* layout / dtype / alignment introspection used by the validation tests and the gates.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM  # noqa: E402  (HF before torch)

import torch  # noqa: E402

import rv2  # noqa: F401,E402
from dynamic_linear import RoutingState  # noqa: E402
from dynamic_model import replace_linear_with_dynamic  # noqa: E402
from rv2.engine import Backend  # noqa: E402

MODEL = "meta-llama/Llama-3.2-1B-Instruct"


def tiny_llama(seed: int = 0, vocab: int = 128) -> LlamaForCausalLM:
    """Random Llama with the real model's attention geometry class: grouped-query attention (8 query heads, 2 KV heads), head_dim 64, tied embeddings."""
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=vocab, hidden_size=512, intermediate_size=256, num_hidden_layers=2, num_attention_heads=8, num_key_value_heads=2,
                      max_position_embeddings=2048, tie_word_embeddings=True, initializer_range=0.1, rope_theta=500000.0)
    return LlamaForCausalLM(cfg).eval()


def build_backend(model, device: str) -> Backend:
    model = model.to(torch.float16).to(device).eval()
    state = RoutingState()
    assert replace_linear_with_dynamic(model, state), "no layers replaced"
    return Backend(model, state, device)


def tiny_backend_cpu(seed: int = 0) -> Backend:
    return build_backend(tiny_llama(seed), "cpu")


def tiny_backend_gpu(seed: int = 0) -> Backend:
    return build_backend(tiny_llama(seed), "cuda")


def real_backend_cpu(model_name: str = MODEL) -> Backend:
    return build_backend(AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16), "cpu")


def cache_geometry(backend: Backend, prompt_ids) -> dict:
    """Layer count, KV heads, head_dim and stored dtype as they appear in the real cache after a prefill (not read from the config)."""
    _, cache = backend.run(prompt_ids, None, use_4bit=False)
    ks, vs = [l.keys for l in cache.layers], [l.values for l in cache.layers]
    return {"n_layers": len(ks), "n_kv_heads": int(ks[0].shape[1]), "head_dim": int(ks[0].shape[-1]), "key_dtype": str(ks[0].dtype),
            "value_dtype": str(vs[0].dtype), "all_fp16": all(t.dtype == torch.float16 for t in ks + vs), "shape_K_layer0": list(ks[0].shape),
            "seq_len": int(cache.get_seq_length())}
