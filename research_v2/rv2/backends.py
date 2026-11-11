"""Backend construction: the historical GPU model, and CPU builds for validation.

GPU: uses the historical DynamicQwen constructor unchanged (src/dynamic_model.py), so the weights,
NF4 quantization settings and layer replacement are exactly those behind every stored result.
CPU: applies the same historical replace_linear_with_dynamic to a model on CPU. bitsandbytes 0.49.2
has a CPU NF4 backend, so the real DynamicLinear runs; CPU NF4 numerics differ from CUDA kernels,
so CPU runs validate LOGIC only, never outcomes.
"""

from __future__ import annotations

# HF imports first (project convention on this machine).
from transformers import AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM

import torch

from . import PROJECT_ROOT  # noqa: F401
from dynamic_linear import DynamicLinear, RoutingState   # historical, unmodified
from dynamic_model import replace_linear_with_dynamic     # historical, unmodified
from .engine import Backend


def load_gpu_backend(model_name: str):
    from dynamic_model import DynamicQwen  # requires CUDA
    dyn = DynamicQwen(model_name, entropy_threshold=1.943)  # threshold unused by rv2 policies
    return Backend(dyn.model, dyn.routing_state, "cuda"), dyn


def build_cpu_backend(model) -> Backend:
    model = model.to(torch.float16).eval()
    state = RoutingState()
    replaced = replace_linear_with_dynamic(model, state)
    assert replaced, "no layers replaced"
    return Backend(model, state, "cpu")


def tiny_qwen2(seed: int = 0, vocab: int = 128) -> Qwen2ForCausalLM:
    """Small random Qwen2 for fast logic tests. Larger init scale gives peaked, varied distributions."""
    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=vocab, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048,
                      tie_word_embeddings=True, initializer_range=0.25, use_sliding_window=False)
    return Qwen2ForCausalLM(cfg).eval()


def real_model_cpu(model_name: str = "Qwen/Qwen2.5-0.5B"):
    return AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16).eval()


def weight_bytes_per_pass(model) -> dict:
    """Bytes of weights read per decode pass: routed projections (FP16 vs NF4 incl. quant state) and
    everything that is always FP16 (embeddings / tied lm_head / norms). An accounting proxy only."""
    fp16_routed = nf4_routed = 0
    routed_ids = set()
    for mod in model.modules():
        if isinstance(mod, DynamicLinear):
            fp16_routed += mod.fp16_weight.numel() * mod.fp16_weight.element_size()
            qs = mod.quant_state
            nf4_routed += mod.bit4_weight.numel() * mod.bit4_weight.element_size()
            nf4_routed += qs.absmax.numel() * qs.absmax.element_size()
            if getattr(qs, "state2", None) is not None:
                nf4_routed += qs.state2.absmax.numel() * qs.state2.absmax.element_size()
            if mod.fp16_bias is not None:   # bias is shared FP16 on both paths
                b = mod.fp16_bias.numel() * mod.fp16_bias.element_size()
                fp16_routed += b
                nf4_routed += b
            routed_ids.update({id(mod.fp16_weight), id(mod.bit4_weight)})
            if mod.fp16_bias is not None:
                routed_ids.add(id(mod.fp16_bias))
    unrouted = 0
    seen = set()
    for t in list(model.parameters()) + list(model.buffers()):
        if id(t) in routed_ids or id(t) in seen or t.data_ptr() in seen:
            continue
        seen.update({id(t), t.data_ptr()})   # tied embedding / lm_head counted once
        unrouted += t.numel() * t.element_size()
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    kvb = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * head_dim * 2   # K+V, FP16 cache
    return {"fp16_routed": fp16_routed, "nf4_routed": nf4_routed, "unrouted": unrouted, "kv_bytes_per_token": kvb}
