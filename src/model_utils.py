"""Model loading utilities for FP16 and 4-bit (NF4) precision.

Phase 1 only loads whole models in a single, static precision — either
full FP16 or full 4-bit NF4 via bitsandbytes. Dynamic per-token precision
switching is out of scope until Phase 2.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Default target model, sized to comfortably fit an 8GB VRAM budget in both
# FP16 (~1GB) and 4-bit (~0.4GB). Fall back to an even smaller model if VRAM
# is constrained (e.g. by other processes holding GPU memory).
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B"
FALLBACK_MODEL_NAME = "HuggingFaceTB/SmolLM-135M"

# Cross-architecture check models: Llama-ARCHITECTURE base checkpoints (LlamaForCausalLM),
# used to test whether entropy-gated routing behavior generalizes beyond the Qwen2
# architecture. TinyLlama_v1.1 (trained independently by the TinyLlama project, NOT a Meta
# Llama checkpoint) was used first while access to the gated meta-llama repo was pending;
# access was subsequently granted, and meta-llama/Llama-3.2-1B is the primary cross-check
# model going forward. Both are kept: TinyLlama's results remain valid data, just superseded
# as the primary comparison point.
TINYLLAMA_MODEL_NAME = "TinyLlama/TinyLlama_v1.1"
LLAMA_MODEL_NAME = "meta-llama/Llama-3.2-1B"

# Instruction-tuned counterpart of DEFAULT_MODEL_NAME, same architecture and
# vocab (so calibrated thresholds are NOT interchangeable with the base
# model's -- instruct tuning shifts the output distribution's entropy
# characteristics, see scripts/calibrate_thresholds_instruct.py), used to
# test whether the digit/operator routing asymmetry and its accuracy cost
# hold under realistic chat-template prompting rather than raw few-shot
# completion.
INSTRUCT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# Large-scale cross-scale check: same architecture family as
# DEFAULT_MODEL_NAME (Qwen2), 28x the parameter count. Requires dual FP16 +
# NF4 residency of a ~28GB (FP16) model -- deliberately NOT run on the
# 8GB-VRAM local RTX 4060 this project otherwise targets; intended for a
# single rented 80GB-class GPU (A100/H100). See scripts/CLOUD_SETUP.md.
QWEN14B_MODEL_NAME = "Qwen/Qwen2.5-14B"


def load_tokenizer(model_name: str) -> AutoTokenizer:
    """Load the tokenizer for a given model, adding a pad token if missing.

    Causal LM tokenizers often lack a dedicated pad token; batched generation
    needs one, so we fall back to the EOS token when absent.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model_fp16(model_name: str) -> AutoModelForCausalLM:
    """Load a causal LM with all weights in FP16, placed on the CUDA device.

    All weight tensors and activations are torch.float16. The model is moved
    fully onto the GPU (no CPU offload) since it is small enough to fit.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
    ).to("cuda")
    model.eval()
    return model


def load_model_4bit(model_name: str) -> AutoModelForCausalLM:
    """Load a causal LM with weights quantized to 4-bit NF4 via bitsandbytes.

    Weight tensors are stored as packed 4-bit NF4 values; bitsandbytes
    dequantizes them on the fly to `bnb_4bit_compute_dtype` (FP16 here) for
    each matmul, so compute happens in FP16 even though storage is 4-bit.
    The KV-cache produced during generation stays FP16 regardless, since
    attention/KV tensors are never touched by the weight quantizer.
    """
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map={"": "cuda:0"},
    )
    model.eval()
    return model
