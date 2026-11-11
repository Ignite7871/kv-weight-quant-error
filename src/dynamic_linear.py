"""Dynamic per-token precision-switching Linear layer (Phase 2 core).

RoutingState + DynamicLinear implement the three locked design decisions:
  1. Routing is global-per-token (one decision reused by every
     DynamicLinear in the model at a given decode step) -- enforced simply
     by every DynamicLinear sharing the same RoutingState instance.
  2. The routing decision reaches every layer via a shared mutable state
     object, not monkeypatched forward() signatures or contextvars.
  3. Both FP16 and 4-bit weights stay resident at all times -- no
     on-demand load/evict.
"""

import bitsandbytes as bnb
import torch
import torch.nn as nn
import torch.nn.functional as F


class RoutingState:
    """Shared mutable state read by every DynamicLinear in a model.

    One instance is created when the model is built and passed by
    reference into every DynamicLinear at construction time.
    DynamicQwen.generate() sets `use_4bit` once before each decode step;
    every DynamicLinear.forward() just reads it. This is what makes
    routing global-per-token: all layers see the same flag at the same
    step because they all hold a reference to this one object.
    """

    def __init__(self):
        self.use_4bit: bool = False


class DynamicLinear(nn.Module):
    """Drop-in nn.Linear replacement holding BOTH FP16 and 4-bit NF4 weights.

    Both representations stay resident in VRAM at all times. forward()
    reads `routing_state.use_4bit` (shared across every DynamicLinear in
    the model, set externally once per decode step) to choose which
    weight representation computes this call. The call signature matches
    nn.Linear exactly (`forward(self, x)`), so this is a transparent
    drop-in replacement for the transformers modules that call
    `self.q_proj(hidden_states)` etc. positionally.

    Weight extraction approach (confirmed by
    scripts/sanity_check_bnb_extraction.py, bit-identical output, no
    tolerance issues): the 4-bit path does NOT own a full
    bnb.nn.Linear4bit submodule. It holds a bnb.nn.Params4bit tensor
    (`bit4_weight`) and its bitsandbytes.functional.QuantState
    (`quant_state`) as plain attributes, and calls bnb.matmul_4bit
    directly in forward() -- this reproduces Linear4bit.forward()
    bit-for-bit with no re-instantiation cost per call.

    The bias is NOT quantized by bitsandbytes (Linear4bit only quantizes
    the weight), so a single FP16 bias is shared by both branches.
    """

    def __init__(
        self,
        fp16_linear: nn.Linear,
        routing_state: RoutingState,
        quant_type: str = "nf4",
    ):
        """Build both weight representations from an existing FP16 nn.Linear.

        Args:
            fp16_linear: a real, already-loaded FP16 nn.Linear (pulled from
                a HF model's named_modules()) whose weight/bias values seed
                BOTH the FP16 and 4-bit representations, so the two paths
                compute from identical source weights (not independently
                re-downloaded/re-initialized).
            routing_state: the single shared RoutingState instance for the
                whole model (same object passed to every DynamicLinear).
            quant_type: bitsandbytes quant type for the 4-bit path; "nf4"
                matches model_utils.load_model_4bit's BitsAndBytesConfig,
                so this stays consistent with the Phase 1 4-bit baseline.
        """
        super().__init__()
        self.in_features = fp16_linear.in_features
        self.out_features = fp16_linear.out_features
        self.routing_state = routing_state

        # Ablation-2 diagnostic hook ONLY (src/scripts/per_layer_sensitivity.py):
        # a per-instance static override, independent of the global per-token
        # routing_state used everywhere else. None (the default, for every
        # DynamicLinear that doesn't have it explicitly set) means "defer to
        # routing_state.use_4bit" -- i.e. forward()'s behavior is UNCHANGED
        # from before this attribute existed. Only scripts that explicitly
        # set this on a specific layer's DynamicLinear instances use it; the
        # main entropy-routing path never touches it.
        self.layer_force_4bit: bool | None = None

        device = fp16_linear.weight.device
        has_bias = fp16_linear.bias is not None

        # --- FP16 path: plain resident buffers (inference-only, no grad). ---
        self.register_buffer("fp16_weight", fp16_linear.weight.detach().clone())
        if has_bias:
            self.register_buffer("fp16_bias", fp16_linear.bias.detach().clone())
        else:
            self.fp16_bias = None

        # --- 4-bit path: quantize the SAME weight values. Params4bit is an
        # nn.Parameter subclass, so assigning it as a plain attribute
        # auto-registers it with this nn.Module (moves with .to(device),
        # appears in .parameters()); requires_grad=False makes it inert
        # for autograd, which is correct since this is inference-only. ---
        self.bit4_weight = bnb.nn.Params4bit(
            fp16_linear.weight.detach().clone(),
            requires_grad=False,
            quant_type=quant_type,
            compress_statistics=True,
        ).to(device)  # .to(cuda) triggers quantization (Params4bit._quantize)
        self.quant_state = self.bit4_weight.quant_state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Route this call through FP16 or 4-bit weights.

        Precision choice: `layer_force_4bit` if set (Ablation-2 static
        per-layer diagnostic override), otherwise `routing_state.use_4bit`
        (the normal global per-token routing path -- unchanged behavior).

        Output is explicitly cast back to FP16 before returning, regardless
        of which branch computed it: hidden states flowing between
        transformer layers -- and eventually the KV-cache -- must stay
        FP16 for downstream-layer/cache consistency, even though the
        compute path (and its native output dtype) differed per token.
        This is the project's hard KV-cache-stays-FP16 invariant, enforced
        at the layer-output level.
        """
        use_4bit = self.layer_force_4bit if self.layer_force_4bit is not None else self.routing_state.use_4bit

        if use_4bit:
            # Mirrors bnb.nn.Linear4bit.forward's own call pattern: transpose
            # then bnb.matmul_4bit, which dequantizes internally per-block.
            out = bnb.matmul_4bit(
                x, self.bit4_weight.t(), quant_state=self.quant_state, bias=self.fp16_bias
            )
        else:
            out = F.linear(x, self.fp16_weight, self.fp16_bias)

        return out.to(torch.float16)
