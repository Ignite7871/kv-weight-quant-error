"""Optional second quantizer (PROTOCOL_V4.md section 9, r3): a dual-weight linear holding the high-precision weight and an HQQ 4-bit copy, with
the locked DynamicLinear routing semantics (global RoutingState flag, per-instance layer_force_4bit override).

HQQ is pinned to hqq==0.2.8.post1 and used with the library defaults of BaseQuantizeConfig (nbits 4, group_size 64, axis 1; quant_zero and
quant_scale False) and its default PYTORCH backend (dequantize, then matmul). Nothing is tuned.

The HQQ state of a model (per layer: packed W_q, scale, zero) is hashed in a fixed order. Section 9 freezing rule: if two quantizations of the
same model are not bitwise identical on the pod, the first is saved, its hash goes into the lock appendix before calibration, and every later
load reads it (`frozen=`); otherwise every load re-quantizes and the driver checks the hash against the lock appendix.
"""

from __future__ import annotations

import hashlib

import torch
import torch.nn as nn
import torch.nn.functional as F

import rv2  # noqa: F401
from dynamic_linear import DynamicLinear  # noqa: E402  (historical, unmodified)

HQQ_VERSION = "0.2.8.post1"
HQQ_CONFIG = {"nbits": 4, "group_size": 64, "axis": 1}            # = the BaseQuantizeConfig defaults; passed explicitly so a mutant is visible


def quant_config():
    from hqq.core.quantize import BaseQuantizeConfig
    return BaseQuantizeConfig(**HQQ_CONFIG)


class HQQDynamicLinear(DynamicLinear):
    """Subclass of the locked DynamicLinear only for its routing contract (isinstance checks, layer_force_4bit); the 4-bit copy is an HQQLinear
    instead of bitsandbytes NF4. The high-precision buffers keep the locked attribute names (fp16_weight / fp16_bias, BF16 under the BF16 rule)."""

    def __init__(self, linear: nn.Linear, routing_state, dtype, device):
        nn.Module.__init__(self)
        from hqq.core.quantize import HQQLinear
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.routing_state = routing_state
        self.layer_force_4bit = None
        self.out_dtype = dtype
        self.register_buffer("fp16_weight", linear.weight.detach().clone())
        if linear.bias is not None:
            self.register_buffer("fp16_bias", linear.bias.detach().clone())
        else:
            self.fp16_bias = None
        src = nn.Linear(self.in_features, self.out_features, bias=linear.bias is not None, device=linear.weight.device, dtype=linear.weight.dtype)
        with torch.no_grad():
            src.weight.copy_(linear.weight)
            if linear.bias is not None:
                src.bias.copy_(linear.bias)
        self.hqq = HQQLinear(src, quant_config(), del_orig=True, compute_dtype=dtype, device=str(device))

    def forward(self, x):
        use_4bit = self.layer_force_4bit if self.layer_force_4bit is not None else self.routing_state.use_4bit
        out = self.hqq(x) if use_4bit else F.linear(x, self.fp16_weight, self.fp16_bias)
        return out.to(self.out_dtype)


def hqq_modules(model):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, HQQDynamicLinear)]


def replace_linear_with_hqq(model, routing_state, dtype, device, frozen=None):
    import rv4
    names = rv4.replace_linear_with_cls(model, routing_state, lambda lin, st: HQQDynamicLinear(lin, st, dtype, device))
    if frozen is not None:
        load_state(model, frozen)
    return names


# ----------------------------------------------------------------------------- state: hash, save, load
STATE_KEYS = ("W_q", "scale", "zero")


def layer_state(m: HQQDynamicLinear) -> dict:
    return {"W_q": m.hqq.W_q.detach(), "scale": m.hqq.meta["scale"].detach(), "zero": m.hqq.meta["zero"].detach()}


def state_hash(model) -> str:
    """SHA-256 over (name, key, dtype, shape, bytes) of every layer's W_q, scale and zero, in module order."""
    h = hashlib.sha256()
    for n, m in hqq_modules(model):
        for k, t in layer_state(m).items():
            t = t.contiguous().cpu()
            h.update(f"{n}|{k}|{t.dtype}|{tuple(t.shape)}|".encode())
            h.update(t.view(torch.uint8).numpy().tobytes() if t.dtype != torch.uint8 else t.numpy().tobytes())
    return h.hexdigest()


def save_state(model, path):
    torch.save({n: {k: v.cpu() for k, v in layer_state(m).items()} for n, m in hqq_modules(model)}, path)


def load_state(model, path_or_dict):
    st = torch.load(path_or_dict) if not isinstance(path_or_dict, dict) else path_or_dict
    mods = dict(hqq_modules(model))
    if set(st) != set(mods):
        raise ValueError("frozen HQQ state does not match the model's layers")
    for n, m in mods.items():
        dev = m.hqq.W_q.device
        m.hqq.W_q.data = st[n]["W_q"].to(dev)
        m.hqq.meta["scale"] = st[n]["scale"].to(dev)
        m.hqq.meta["zero"] = st[n]["zero"].to(dev)


# ----------------------------------------------------------------------------- independent dequantization oracle and storage size
def unpack_4bit(W_q):
    """Independent of hqq.core.bitpack: the first half of the rows is the high nibble, the second half the low nibble."""
    w = W_q.to(torch.int16)
    return torch.cat([(w >> 4) & 0xF, w & 0xF], dim=0)


def dequant_oracle(m: HQQDynamicLinear):
    """W = (W_q - z) * s per group of 64 along axis 1, in the compute dtype, reshaped to [out, in]."""
    st = layer_state(m)
    dt = m.hqq.meta["compute_dtype"]
    q = unpack_4bit(st["W_q"]).to(dt)                                    # [out*in/64, 64]
    return ((q - st["zero"].to(dt)) * st["scale"].to(dt)).reshape(m.out_features, m.in_features)


def bits_per_weight(model) -> float:
    bits = n = 0
    for _, m in hqq_modules(model):
        for t in layer_state(m).values():
            bits += t.numel() * t.element_size() * 8
        n += m.in_features * m.out_features
    return bits / n
