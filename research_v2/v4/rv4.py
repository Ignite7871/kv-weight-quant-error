"""v4 core (PROTOCOL_V4.md sections 3, 5, 6): the six core arms on top of the locked v3 construction, the BF16 rule, X13a sums, the per-model
calibration accumulator and the dual-weight backends.

Step (unchanged from v3 section 4, restricted): at decode step s, from token x_{s-1} and the arm's cache S_{s-1}:
  1. measurement pass with the NF4 weights everywhere, snapshot its new entry kappa^M, crop;
  2. high-precision pass (its logits are the arm's logits, kappa^F);
  3. overwrite the new entry with the arm's stored value.
There are no predictors, no query capture and no QA/QO/CV/C2/LOC/INH arms in v4.

Precision. "fp16" (default) runs exactly the v3 path: the locked DynamicLinear, the locked stored_entry_v3 (float64 construction, one-step
correctly rounded FP16) and the locked FP16 rounding oracle. "bf16" is the pre-declared G3 fallback (section 5, D6): both weight copies, the
activations and the cache are BF16; the entry is built in float64 with the same draws as v3 and stored as the correctly rounded BF16 (round to
nearest, ties to even) of the float64 entry, checked against an independent integer-arithmetic oracle.

X13a (section 6): for the NAT arm the per-entry norms ||delta_j|| [n, L, n_kv] and the per-slot sums S1 = sum_j delta_j [L, n_kv, head_dim] of
delta = stored - kappa^F, per path, float64 (the X13 section 4 schema).

Every module-level helper can be replaced by the mutation check.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RV = HERE.parent
sys.path.insert(0, str(RV))
CUBLAS_WORKSPACE = ":4096:8"                                 # r4: required by torch's deterministic mode for cuBLAS
os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE     # before anything imports torch

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402  (HF before torch)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import rv2  # noqa: F401,E402
import rv2.controls as C  # noqa: E402
import rv3.core as V  # noqa: E402
from rv2.engine import Backend, _snapshot_last  # noqa: E402
from dynamic_linear import DynamicLinear, RoutingState  # noqa: E402  (historical, unmodified)
from dynamic_model import DYNAMIC_LINEAR_TARGET_NAMES, replace_linear_with_dynamic  # noqa: E402  (historical, unmodified)

MASTER_SEED = "rv4-scale-2026-11-02"
CORE_ARMS = ["NAT", "C1", "NAT-K", "C1-K", "NAT-V", "C1-V"]
CAL_ARMS = [("NAT", 1.0, "NAT"), ("C1", 0.5, "C1_a0.5"), ("C1", 1.0, "C1_a1"), ("C1", 2.0, "C1_a2")]
NATURAL = {"NAT": "KV", "NAT-K": "K", "NAT-V": "V"}
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}
MANIFEST = HERE / "V4_MODEL_MANIFEST.json"
BF16_MAX_BITS = 0x7F7F                                       # largest finite BF16 magnitude
BF16_INF_BITS = 0x7F80


def manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))["models"]


# ----------------------------------------------------------------------------- deterministic settings (r4)
def apply_determinism() -> dict:
    """torch deterministic algorithms and the cuBLAS workspace. cuBLAS reads the variable when its context is created, at the first BLAS call, so
    the entry points (run_v4, this module, test_rv4) also set it at the top of the file, before anything imports torch."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE
    torch.use_deterministic_algorithms(True)
    return determinism_state()


def determinism_state() -> dict:
    return {"cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"), "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}


def determinism_ok(state: dict | None = None) -> bool:
    s = determinism_state() if state is None else state
    return s.get("cublas_workspace_config") == CUBLAS_WORKSPACE and s.get("deterministic_algorithms") is True


# ----------------------------------------------------------------------------- seeds (labels only; v3 rule with the v4 master label)
def arm_seeds(model_id: str, arm: str, alpha: float, index: int, tier: str):
    """(main, value-side) seeds. The core arms use only the main stream; the value-side stream is passed for the locked signature."""
    return (V.seed_for(model_id, arm, alpha, index, tier, master=MASTER_SEED),
            V.seed_for(model_id, f"{arm}-V", alpha, index, tier, master=MASTER_SEED))


# ----------------------------------------------------------------------------- BF16 rounding (runtime) and its independent oracle
def to_bf16(u):
    """Correctly rounded float64 -> bfloat16 in one step. torch's double -> bfloat16 goes through float32 and can round twice, so the result is
    chosen among the three BF16 neighbours of the truncated float32 rounding: nearest in float64, ties to the even bit pattern."""
    a = u.double().abs()
    b = (a.float().view(torch.int32) >> 16).to(torch.int64)
    cand = torch.stack([(b - 1).clamp_min(0), b, (b + 1).clamp_max(BF16_INF_BITS)])
    vals = (cand << 16).to(torch.int32).view(torch.float32).double()
    err = (vals - a).abs()
    best = err.min(0).values
    score = torch.where(err == best, cand % 2, torch.full_like(cand, 2))
    h = cand.gather(0, score.argmin(0, keepdim=True)).squeeze(0)
    h = (h << 16).to(torch.int32).view(torch.float32).to(torch.bfloat16)          # exact: the value is a BF16 value
    return torch.where(torch.signbit(u), -h, h)


def bf16_rne_oracle(u):
    """Independent float64 -> bfloat16 (round to nearest, ties to even) by integer arithmetic on the float64 bit pattern: 8 significant bits for
    normals (exponent >= -126), fixed 2^-133 quantum for subnormals, overflow to inf at or above max + half an ulp."""
    x = np.ascontiguousarray(u.detach().double().cpu().numpy(), dtype=np.float64)
    bits = x.view(np.uint64)
    sign = (bits >> np.uint64(63)).astype(np.int64)
    e = ((bits >> np.uint64(52)) & np.uint64(0x7FF)).astype(np.int64)
    m = (bits & np.uint64((1 << 52) - 1)).astype(np.int64)
    if bool((e == 0x7FF).any()):
        raise ValueError("non-finite input")
    E = e - 1023
    sig = m | (1 << 52)                                               # 53-bit significand with the hidden bit
    normal = E >= -126
    shift = np.where(normal, 45, 45 + (-126 - E))                     # bits dropped: 45 for normals, more for the subnormal range
    sh = np.clip(shift, 1, 53)
    q = sig >> sh
    r = sig & ((np.int64(1) << sh) - 1)
    half = np.int64(1) << (sh - 1)
    q = q + ((r > half) | ((r == half) & ((q & 1) == 1))).astype(np.int64)
    q = np.where(shift >= 54, 0, q)                                   # below half the smallest subnormal: rounds to zero
    carry = normal & (q == 256)
    q = np.where(carry, 128, q)
    En = np.where(carry, E + 1, E)
    pat_n = np.where(En + 127 >= 255, BF16_INF_BITS, ((En + 127) << 7) | (q - 128))
    pat = np.where(normal, pat_n, q)                                  # subnormal quantum 2^-133; q == 128 is the smallest normal pattern
    pat = np.where(e == 0, 0, pat)                                    # float64 zero / subnormal: far below the BF16 subnormal range
    out = (sign << 15) | pat
    o16 = np.where(out >= 0x8000, out - 0x10000, out).astype(np.int16)
    return torch.from_numpy(o16).view(torch.bfloat16).to(u.device)


def to_store(u, dtype):
    return V.to_fp16(u) if dtype == torch.float16 else to_bf16(u)


def rne_oracle(u, dtype):
    return V.fp16_rne_reference(u) if dtype == torch.float16 else bf16_rne_oracle(u)


# ----------------------------------------------------------------------------- stored entries
def stored_entry_generic(arm: str, alpha: float, kF, kM, gen, dtype):
    """The v3 construction of the six core arms (rv3.core.stored_entry_v3, the same draws in the same order, built in float64), stored with
    `to_store`. Used for BF16; in FP16 it must equal the locked function bitwise (tested)."""
    dev, two_l = kF.device, kF.shape[0]
    exact = torch.zeros(two_l, dtype=torch.bool, device=dev)
    info = {"target": None, "exact_rows": exact, "zero_slots": 0, "leak": 0.0}
    if arm in NATURAL:
        out = kF.clone()
        if alpha == 0.0:
            info["exact_rows"] = torch.ones(two_l, dtype=torch.bool, device=dev)
            info["pre"] = out.double()
            return out, info
        if alpha != 1.0:
            raise ValueError("natural arms take alpha in {0, 1}")
        paths = NATURAL[arm]
        if "K" in paths:
            out[0::2] = kM[0::2]
        else:
            exact[0::2] = True
        if "V" in paths:
            out[1::2] = kM[1::2]
        else:
            exact[1::2] = True
        info["pre"] = out.double()
        return out, info
    Fd = kF.double()
    D = kM.double() - Fd
    DK, DV = D[0::2], D[1::2]
    delta = torch.zeros_like(Fd)
    target = torch.zeros(Fd.shape[:-1] + (1,), dtype=torch.float64, device=dev)
    if arm == "C1":
        delta = alpha * V._slot_norm(D) * V._unit_safe(V._gauss(D.shape, gen, dev).double())
        target = alpha * V._slot_norm(D)
    elif arm == "C1-K":
        delta[0::2] = alpha * V._slot_norm(DK) * V._unit_safe(V._gauss(DK.shape, gen, dev).double())
        target[0::2] = alpha * V._slot_norm(DK)
        exact[1::2] = True
    elif arm == "C1-V":
        delta[1::2] = alpha * V._slot_norm(DV) * V._unit_safe(V._gauss(DV.shape, gen, dev).double())
        target[1::2] = alpha * V._slot_norm(DV)
        exact[0::2] = True
    else:
        raise ValueError(arm)
    if alpha == 0.0:
        exact = torch.ones(two_l, dtype=torch.bool, device=dev)
    u = Fd + delta
    u[exact] = Fd[exact]
    info["target"], info["exact_rows"], info["pre"] = target, exact, u
    return to_store(u, dtype), info


def stored_entry_v4(arm, alpha, kF, kM, gen, vgen, dtype):
    if arm not in CORE_ARMS:
        raise ValueError(f"{arm} is not a v4 arm")
    if dtype == torch.float16:
        return V.stored_entry_v3(arm, alpha, kF, kM, gen, vgen, None)        # the locked v3 construction, unchanged
    return stored_entry_generic(arm, alpha, kF, kM, gen, dtype)


# ----------------------------------------------------------------------------- X13a sums and the calibration accumulator
class SlotSums:
    """delta = stored - kappa^F of every decode-written entry: per-entry norms per path and the per-slot sums S1 (float64)."""

    def __init__(self):
        self.normsK, self.normsV, self.S1K, self.S1V = [], [], None, None

    def record(self, stored, kF):
        d = stored.double() - kF.double()                         # [2L, n_kv, hd]
        dK, dV = d[0::2], d[1::2]
        self.normsK.append(dK.norm(dim=-1))
        self.normsV.append(dV.norm(dim=-1))
        self.S1K = dK.clone() if self.S1K is None else self.S1K + dK
        self.S1V = dV.clone() if self.S1V is None else self.S1V + dV

    def arrays(self) -> dict:
        return {"normsK": torch.stack(self.normsK).cpu().numpy(), "normsV": torch.stack(self.normsV).cpu().numpy(),
                "S1K": self.S1K.cpu().numpy(), "S1V": self.S1V.cpu().numpy()}


class RhoAccumulator:
    """Calibration (section 3): relative natural error ||Delta|| / ||kappa^F|| per slot and step, by path, from the NAT pass."""

    def __init__(self):
        self.rhoK, self.rhoV, self.n_meas = [], [], 0

    def on_measure(self, kF, kM):
        Fd = kF.double()
        D = kM.double() - Fd
        r = D.norm(dim=-1) / Fd.norm(dim=-1).clamp_min(1e-30)
        self.rhoK.append(r[0::2].flatten().cpu())
        self.rhoV.append(r[1::2].flatten().cpu())
        self.n_meas += 1

    def state(self):
        return {"rhoK": torch.cat(self.rhoK), "rhoV": torch.cat(self.rhoV), "n_meas": self.n_meas}


# ----------------------------------------------------------------------------- one controlled step and the teacher-forced arm
def _finite(*ts):
    return all(bool(torch.isfinite(t).all()) for t in ts)


def controlled_step_v4(backend, cur, cache, arm, alpha, gen, vgen, dtype, sums=None, acc=None):
    L = cache.get_seq_length()
    V.measure_pass(backend, cur, cache, "NF4")                   # global 4-bit flag: NF4 weights everywhere
    kM = C._stack_slices(_snapshot_last(cache, L))
    cache.crop(L)
    lg_F, _ = backend.run(cur, cache, False)
    assert cache.get_seq_length() == L + 1
    kF = C._stack_slices(_snapshot_last(cache, L))
    new, info = stored_entry_v4(arm, alpha, kF, kM, gen, vgen, dtype)
    if acc is not None:
        acc.on_measure(kF, kM)
    C._write_back(cache, L, new)
    if sums is not None:
        sums.record(new, kF)
    Ff = kF.float()
    D = kM.float() - Ff
    F64 = kF.double()
    u = info["pre"]
    back = C._stack_slices(_snapshot_last(cache, L))
    ex = info["exact_rows"]
    d = {"exact_ok": bool(torch.equal(new[ex], kF[ex])) if bool(ex.any()) else True,
         "rel_K": float((V._slot_norm(D[0::2]) / V._slot_norm(Ff[0::2]).clamp_min(1e-12)).mean()),
         "rel_V": float((V._slot_norm(D[1::2]) / V._slot_norm(Ff[1::2]).clamp_min(1e-12)).mean()),
         "dn2_K": float((D[0::2] ** 2).sum(-1).mean()), "dn2_V": float((D[1::2] ** 2).sum(-1).mean()),
         "target_norm_mean": float(V._slot_norm(D).mean()),
         "routing_reset": V.overrides_clear(backend.model),
         "finite_ok": _finite(lg_F, kM, kF, new),
         "rounding_ok": bool(new.dtype == dtype and back.dtype == dtype and torch.equal(new, back) and torch.equal(back, rne_oracle(u, dtype)))}
    nbin = V.RDEV_EDGES.numel() + 1
    if info["target"] is None:
        nat_rows = ~ex
        d["natural_ok"] = bool(torch.equal(new[nat_rows], kM[nat_rows])) if bool(nat_rows.any()) else True
        d.update(construction_ok=True, constr_dev_max=0.0, rdev_max=0.0, rdev_n_gt5=0, rdev_n=0, rdev_sum=0.0, rdev_arg=None, rdev_hist=[0] * nbin)
    else:
        tgt = info["target"].squeeze(-1)
        pert = (~ex).unsqueeze(-1).expand_as(tgt)
        pos = pert & (tgt > 0)
        rpre = (u - F64).norm(dim=-1)
        cdev = torch.where(pos, (rpre - tgt).abs() / tgt.clamp_min(1e-300), torch.zeros_like(tgt))
        zero_bad = bool((pert & (tgt == 0) & (rpre != 0)).any())
        d["constr_dev_max"] = float(cdev.max())
        d["construction_ok"] = bool(d["constr_dev_max"] <= V.CONSTR_TOL and not zero_bad)
        rn = (back.double() - F64).norm(dim=-1)
        rdev = torch.where(pos, (rn - tgt).abs() / tgt.clamp_min(1e-300), torch.zeros_like(tgt))
        vals = rdev[pos]
        k = int(rdev.argmax())
        r_, h_ = divmod(k, rdev.shape[1])
        d.update(rdev_max=float(rdev.max()), rdev_n_gt5=int((vals > V.RDEV_FLAG).sum()), rdev_n=int(vals.numel()), rdev_sum=float(vals.sum()),
                 rdev_arg=[r_ // 2, "K" if r_ % 2 == 0 else "V", h_] if vals.numel() else None,
                 rdev_hist=torch.bincount(torch.bucketize(vals.cpu(), V.RDEV_EDGES), minlength=nbin).tolist())
        d["natural_ok"] = True
    return lg_F, d


STEP_KEYS = ["rel_K", "rel_V", "dn2_K", "dn2_V", "target_norm_mean", "constr_dev_max", "rdev_max", "rdev_n_gt5", "rdev_arg"]
FLAGS = ["exact_ok", "natural_ok", "routing_reset", "construction_ok", "rounding_ok", "finite_ok"]


@torch.no_grad()
def teacher_forced_arm_v4(backend, prompt_ids, ref_ids, S, arm, alpha, seeds, ref_lg, dtype, acc=None, slots=False, return_cache=False):
    """One arm on one problem. Returns the record (v3 record fields restricted to the core arms, plus finite_ok) and, with slots=True, the X13a
    arrays as rec['_slots'] (the driver writes them to an npz and removes the key)."""
    gen = C.make_generator(prompt_ids.device, seeds[0])
    vgen = C.make_generator(prompt_ids.device, seeds[1])
    _, cache = backend.run(prompt_ids, None, use_4bit=False)
    sums = SlotSums() if slots else None
    rec = {k: [] for k in ["js", "flip", "top"] + STEP_KEYS}
    flags = {k: True for k in FLAGS}
    hist, rsum, rcount = [0] * (V.RDEV_EDGES.numel() + 1), 0.0, 0
    for s in range(1, S + 1):
        cur = torch.tensor([[ref_ids[s - 1]]], device=prompt_ids.device)
        lg, d = controlled_step_v4(backend, cur, cache, arm, alpha, gen, vgen, dtype, sums, acc)
        r = ref_lg[s - 1:s]
        rec["js"].append(float(C.js64(lg, r)))
        rec["flip"].append(float((torch.argmax(lg, dim=-1) != torch.argmax(r, dim=-1)).reshape(())))
        rec["top"].append(int(torch.argmax(lg, dim=-1).item()))
        for k in STEP_KEYS:
            rec[k].append(d[k])
        hist = [x + y for x, y in zip(hist, d["rdev_hist"])]
        rsum += d["rdev_sum"]
        rcount += d["rdev_n"]
        for k in FLAGS:
            flags[k] = flags[k] and bool(d[k])
    rec.update(flags)
    rec.update(rdev_hist=hist, rdev_sum=rsum, rdev_count=rcount)
    if slots:
        rec["_slots"] = sums.arrays()
    return (rec, cache) if return_cache else rec


@torch.no_grad()
def reference_logits(backend, prompt_ids, ref_ids, S):
    """Pure high-precision teacher-forced logits [S, V] float32 (rv3.core.reference_logits_v3 without query capture)."""
    return V.reference_logits_v3(backend, prompt_ids, ref_ids, S)


# ----------------------------------------------------------------------------- backends
class DynamicLinearBF16(DynamicLinear):
    """The BF16 rule (section 5): the locked DynamicLinear built from BF16 weights (both copies BF16; NF4 codes quantized from the BF16 weight,
    bitsandbytes dequantizes to the quant state's BF16 dtype), with the layer output cast to BF16 instead of FP16. Routing and the per-instance
    override are those of the locked class."""

    def forward(self, x):
        import bitsandbytes as bnb
        use_4bit = self.layer_force_4bit if self.layer_force_4bit is not None else self.routing_state.use_4bit
        if use_4bit:
            out = bnb.matmul_4bit(x, self.bit4_weight.t(), quant_state=self.quant_state, bias=self.fp16_bias)
        else:
            out = F.linear(x, self.fp16_weight, self.fp16_bias)
        return out.to(torch.bfloat16)


def replace_linear_with_cls(model: nn.Module, routing_state, make) -> list:
    """replace_linear_with_dynamic (src/dynamic_model.py) with a constructor argument: same targets, same walk order."""
    names = []
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if child_name in DYNAMIC_LINEAR_TARGET_NAMES and isinstance(child, nn.Linear):
                setattr(parent, child_name, make(child, routing_state))
                names.append(f"{parent_name}.{child_name}" if parent_name else child_name)
    return names


def build_backend(model, device: str, precision: str = "fp16", quantizer: str = "nf4") -> Backend:
    """FP16 + NF4 is exactly the v3 construction (model cast to FP16, moved, then the locked replace_linear_with_dynamic)."""
    dt = DTYPES[precision]
    model = model.to(dt).to(device).eval()
    state = RoutingState()
    if quantizer == "nf4" and precision == "fp16":
        names = replace_linear_with_dynamic(model, state)
    elif quantizer == "nf4":
        names = replace_linear_with_cls(model, state, lambda lin, st: DynamicLinearBF16(lin, st))
    else:
        raise ValueError(quantizer)
    assert names, "no layers replaced"
    be = Backend(model, state, device)
    be.precision, be.quantizer, be.n_dynamic = precision, quantizer, len(names)
    return be


def load_model(model_id: str, device: str, precision: str = "fp16", quantizer: str = "nf4", local_only: bool = True):
    """The pinned snapshot of a manifest model, as a dual-weight backend, and its tokenizer."""
    e = manifest()[model_id]
    dt = DTYPES[precision]
    model = AutoModelForCausalLM.from_pretrained(e["hf_id"], revision=e["revision"], dtype=dt, local_files_only=local_only)
    tok = AutoTokenizer.from_pretrained(e["hf_id"], revision=e["revision"], local_files_only=local_only)
    return build_backend(model, device, precision, quantizer), tok


def cache_geometry(backend, prompt_ids, dtype) -> dict:
    """Layers, KV heads, head_dim and stored dtype as they appear in the real cache after a prefill (e1l_support.cache_geometry, dtype-aware)."""
    _, cache = backend.run(prompt_ids, None, use_4bit=False)
    ks, vs = [l.keys for l in cache.layers], [l.values for l in cache.layers]
    return {"n_layers": len(ks), "n_kv_heads": int(ks[0].shape[1]), "head_dim": int(ks[0].shape[-1]), "key_dtype": str(ks[0].dtype),
            "value_dtype": str(vs[0].dtype), "all_mode_dtype": all(t.dtype == dtype for t in ks + vs), "seq_len": int(cache.get_seq_length())}


def config_snapshot(cfg) -> dict:
    """The values G7 compares with the manifest, read from the loaded config."""
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    return {"layers": cfg.num_hidden_layers, "q_heads": cfg.num_attention_heads, "kv_heads": cfg.num_key_value_heads, "head_dim": hd,
            "vocab": cfg.vocab_size, "tied_embeddings": bool(getattr(cfg, "tie_word_embeddings", False))}


GEOMETRY_KEYS = ("layers", "q_heads", "kv_heads", "head_dim", "vocab", "tied_embeddings")


def geometry_matches(snap: dict, man_geom: dict) -> bool:
    return all(snap[k] == man_geom[k] for k in GEOMETRY_KEYS)
