"""v3 controlled step, arms, routing, query capture, predictors and calibration accumulation (PROTOCOL_V3.md sections 4-6).

Step (section 4). At decode step s, from token x_{s-1} and the arm's cache S_{s-1}:
  1. measurement pass with the arm's precision map (NF4 everywhere; LOC = NF4 only in k_proj/v_proj; INH = NF4 everywhere except k_proj/v_proj),
  2. snapshot its new entry kappa^M, crop,
  3. FP16 pass (logits, kappa^F; queries captured read-only for the predictors),
  4. predictors from the cache exactly as the FP16 pass read it (new position still kappa^F),
  5. overwrite the new entry with the arm's stored value.
Stacked entries are [2*layers, n_kv, head_dim], layer-major, K then V (rv2.controls._stack_slices); even rows = K, odd rows = V.
Amendment 2 (PROTOCOL_V3 section 16): every stored entry is built in float64 (the pre-rounding entry u) and stored as the correctly rounded FP16 of u
in one step (to_fp16). Each step checks (a) construction: per-slot | ||u - kF|| - target | / target <= 1e-4, and (b) rounding: the entry read back
from the cache equals fp16_rne_reference(u) bitwise (an independent implementation). The post-rounding per-slot norm deviation is reported, not gated.
Every module-level helper can be replaced by the mutation check.
"""

from __future__ import annotations

import hashlib
import math
import sys
from contextlib import contextmanager

import numpy as np
import torch

import rv2  # noqa: F401
import rv2.controls as C
from rv2.engine import _snapshot_last
from dynamic_linear import DynamicLinear  # historical, unmodified

MASTER_SEED = "rv3-cache-origin-2026-10-29"
KV_NAMES = ("k_proj", "v_proj")
TEST_ARMS = ["NAT", "C1", "C2", "QA", "QO", "CV", "NAT-K", "NAT-V", "NAT-K*", "NAT-V*", "C1-K", "C1-V", "LOC", "INH"]
CAL_ARMS = [("NAT", 1.0, "NAT"), ("C1", 0.5, "C1_a0.5"), ("C1", 1.0, "C1_a1"), ("C1", 2.0, "C1_a2")]
NATURAL = {"NAT": "KV", "NAT-K": "K", "NAT-V": "V", "LOC": "KV", "INH": "KV"}     # arms that store the measured entry (bitwise) on these paths
MEASURE = {"LOC": "LOC", "INH": "INH"}                                           # every other arm measures with NF4 everywhere
NEEDS_CTX = {"QA", "QO", "CV", "NAT-K*", "NAT-V*"}


def file_tag(arm: str) -> str:
    """Record file / tag name for an arm ('*' is not a valid Windows filename character)."""
    return arm.replace("*", "star")
R_SUBSPACE = 8
N_OUTLIER = 4
FP16_RES = 2.0 ** -10                       # E1 floor; descriptive diagnostic only since amendment 2
CONSTR_TOL = 1e-4                           # amendment 2: construction check (pre-rounding, float64)
RDEV_FLAG = 0.05                            # amendment 2: realized post-rounding deviation flag (reporting / exploratory sensitivity)
RDEV_EDGES = torch.logspace(-8, 1, 361, dtype=torch.float64)   # reporting histogram: 40 bins per decade, plus under/overflow


# ----------------------------------------------------------------------------- seeds (labels only)
def seed_for(model_label: str, arm: str, alpha: float, index: int, tier: str, repeat: int = 0, master: str = MASTER_SEED) -> int:
    h = hashlib.sha256(f"{master}|{model_label}|{arm}|{alpha!r}|{index}|{tier}|{repeat}".encode()).digest()
    return int.from_bytes(h[:8], "big") >> 1


def arm_seeds(model_label, arm, alpha, index, tier):
    """(main, value-side) seeds. QA and QO share the value-side stream (common random numbers)."""
    v_label = "QAQO-V" if arm in ("QA", "QO") else f"{arm}-V"
    return seed_for(model_label, arm, alpha, index, tier), seed_for(model_label, v_label, alpha, index, tier)


# ----------------------------------------------------------------------------- routing (LOC / INH) through the existing per-instance override
def dynamic_modules(model):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, DynamicLinear)]


def routing_map(model, mode: str) -> dict:
    """name -> True (NF4) / False (FP16) for the measurement pass."""
    out = {}
    for n, _ in dynamic_modules(model):
        is_kv = n.split(".")[-1] in KV_NAMES
        if mode == "LOC":
            out[n] = is_kv
        elif mode == "INH":
            out[n] = not is_kv
        else:
            raise ValueError(mode)
    return out


def set_overrides(model, mp: dict):
    for n, m in dynamic_modules(model):
        m.layer_force_4bit = bool(mp[n])


def clear_overrides(model):
    for _, m in dynamic_modules(model):
        m.layer_force_4bit = None


def overrides_clear(model) -> bool:
    return all(m.layer_force_4bit is None for _, m in dynamic_modules(model))


@contextmanager
def precision_probe(model):
    """Records, per DynamicLinear call, the precision actually used (True = NF4). Read-only forward pre-hooks."""
    log = []
    hs = []
    for n, m in dynamic_modules(model):
        def pre(mod, args, n=n):
            log.append((n, bool(mod.layer_force_4bit if mod.layer_force_4bit is not None else mod.routing_state.use_4bit)))
        hs.append(m.register_forward_pre_hook(pre))
    try:
        yield log
    finally:
        for h in hs:
            h.remove()


def measure_pass(backend, cur, cache, mode: str):
    """Appends kappa^M at position L. NF4: exactly the E1 NF4 pass (global routing flag). LOC/INH: per-module overrides, reset afterwards."""
    if mode == "NF4":
        backend.run(cur, cache, True)
        return
    set_overrides(backend.model, routing_map(backend.model, mode))
    try:
        backend.run(cur, cache, False)
    finally:
        clear_overrides(backend.model)


# ----------------------------------------------------------------------------- read-only query capture (post-RoPE)
class QueryCapture:
    """Forward pre-hook on each self_attn (position_embeddings) and forward hook on each q_proj (pre-RoPE query); the post-RoPE query is
    recomputed with the model's own apply_rotary_pos_emb. Active only while .active is True (the FP16 pass). Hooks return None: no effect."""

    def __init__(self, model):
        self.layers = [lay.self_attn for lay in model.model.layers]
        self.active = False
        self.pos = [None] * len(self.layers)
        self.qraw = [None] * len(self.layers)
        self.hs = []
        a0 = self.layers[0]
        self.head_dim = a0.head_dim
        self.n_q = model.config.num_attention_heads
        self.n_kv = model.config.num_key_value_heads
        self.scaling = float(a0.scaling)
        self.rope = sys.modules[type(a0).__module__].apply_rotary_pos_emb
        for i, attn in enumerate(self.layers):
            def pre(mod, args, kwargs, i=i):
                if self.active:
                    self.pos[i] = kwargs.get("position_embeddings")
            def qh(mod, args, out, i=i):
                if self.active:
                    self.qraw[i] = out.detach()
            self.hs.append(attn.register_forward_pre_hook(pre, with_kwargs=True))
            self.hs.append(attn.q_proj.register_forward_hook(qh))

    def remove(self):
        for h in self.hs:
            h.remove()

    def queries(self):
        """list over layers of [n_q, head_dim] post-RoPE queries of the last captured single-token pass."""
        return [post_rope_query(self, i) for i in range(len(self.layers))]


def post_rope_query(cap: QueryCapture, i: int):
    q = cap.qraw[i]                                                   # [1, 1, n_q*hd]
    q = q.view(1, 1, cap.n_q, cap.head_dim).transpose(1, 2).double()  # [1, n_q, 1, hd]
    cos, sin = (x.double() for x in cap.pos[i])                       # rotation applied in float64 to the FP16 pre-RoPE query
    qr, _ = cap.rope(q, q, cos, sin)
    return qr[0, :, -1, :]


# ----------------------------------------------------------------------------- stored entries
def _slot_norm(x):
    return x.double().norm(dim=-1, keepdim=True)


def to_fp16(u):
    """Correctly rounded (round to nearest, ties to even) float64 -> float16 in ONE step (numpy's direct conversion). torch's double -> half goes
    through float32 and can round twice."""
    return torch.from_numpy(u.detach().cpu().numpy().astype(np.float16)).to(u.device)


def fp16_rne_reference(u):
    """Independent correctly rounded float64 -> float16 used by the runtime check: the three FP16 neighbours of a first guess (bit patterns of
    non-negative halves are ordered), nearest in float64, ties to the even pattern."""
    a = u.double().abs()
    b = a.float().half().view(torch.int16).to(torch.int32)
    cand = torch.stack([(b - 1).clamp_min(0), b, (b + 1).clamp_max(0x7C00)])
    err = (cand.to(torch.int16).view(torch.float16).double() - a).abs()
    best = err.min(0).values
    score = torch.where(err == best, cand % 2, torch.full_like(cand, 2))
    h = cand.gather(0, score.argmin(0, keepdim=True)).squeeze(0).to(torch.int16).view(torch.float16)
    return torch.where(torch.signbit(u), -h, h)


def _unit_safe(x):
    n = x.norm(dim=-1, keepdim=True)
    return torch.where(n > 0, x / n.clamp_min(1e-30), torch.zeros_like(x))


def _gauss(shape, gen, device):
    return torch.randn(shape, generator=gen, device=device, dtype=torch.float32)


def _project(U, g):
    """P g with P = U U^T per (layer, KV head). U [L, n_kv, hd, r], g [L, n_kv, hd]."""
    return torch.einsum("lhdr,lhr->lhd", U, torch.einsum("lhdr,lhd->lhr", U, g))


def _kv(x):
    return x[0::2], x[1::2]


def stored_entry_v3(arm: str, alpha: float, kF, kM, gen, vgen, ctx):
    """Returns (stored [2L, n_kv, hd] fp16, info). info: target per-slot norm of (u - kF) [2L, n_kv, 1] float64 (None for natural arms), the
    float64 pre-rounding entry u ('pre'), exact-path mask [2L] (rows that must equal kF bitwise), zero-direction slot count, QA/QO subspace leak
    (pre-rounding). Amendment 2: built in float64 (random draws are the same float32 draws as before, cast exactly); stored = to_fp16(u)."""
    dev = kF.device
    two_l = kF.shape[0]
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
    F = kF.double()
    D = kM.double() - F
    DK, DV = _kv(D)
    FK, FV = _kv(F)
    delta = torch.zeros_like(F)
    target = torch.zeros(F.shape[:-1] + (1,), dtype=torch.float64, device=dev)
    if arm == "C1":                                              # E1 C1 (rv2.controls): same draw, built in float64
        delta = alpha * _slot_norm(D) * _unit_safe(_gauss(D.shape, gen, dev).double())
        target = alpha * _slot_norm(D)
    elif arm == "C2":                                            # E1 C2: same Rademacher draw
        delta = alpha * D * C._rademacher(D.shape, gen, dev).double()
        target = alpha * _slot_norm(D)
    elif arm in ("QA", "QO"):
        U = ctx["U"].double()
        g = _gauss(DK.shape, gen, dev).double()
        pg = _project(U, g)
        dirn = pg if arm == "QA" else g - pg
        dk = alpha * _slot_norm(DK) * _unit_safe(dirn)
        dv = alpha * _slot_norm(DV) * _unit_safe(_gauss(DV.shape, vgen, dev).double())
        delta[0::2], delta[1::2] = dk, dv
        target[0::2], target[1::2] = alpha * _slot_norm(DK), alpha * _slot_norm(DV)
        nk = dk.norm(dim=-1)
        if bool((nk > 0).any()):
            off = (dk - _project(U, dk)) if arm == "QA" else _project(U, dk)
            info["leak"] = float((off.norm(dim=-1)[nk > 0] / nk[nk > 0]).max())
    elif arm == "CV":
        w = ctx["sigma2"].double().clamp_min(0).sqrt() * _gauss(D.shape, gen, dev).double()
        info["zero_slots"] = int((w.norm(dim=-1) == 0).sum())
        delta = alpha * _slot_norm(D) * _unit_safe(w)
        target = alpha * _slot_norm(D) * (w.norm(dim=-1, keepdim=True) > 0)
    elif arm in ("NAT-K*", "NAT-V*"):
        rho = float(ctx["rho_star"])
        if arm == "NAT-K*":
            delta[0::2] = alpha * rho * _slot_norm(FK) * _unit_safe(DK)
            target[0::2] = alpha * rho * _slot_norm(FK) * (DK.norm(dim=-1, keepdim=True) > 0)
            info["zero_slots"] = int((DK.norm(dim=-1) == 0).sum())
            exact[1::2] = True
        else:
            delta[1::2] = alpha * rho * _slot_norm(FV) * _unit_safe(DV)
            target[1::2] = alpha * rho * _slot_norm(FV) * (DV.norm(dim=-1, keepdim=True) > 0)
            info["zero_slots"] = int((DV.norm(dim=-1) == 0).sum())
            exact[0::2] = True
    elif arm == "C1-K":
        delta[0::2] = alpha * _slot_norm(DK) * _unit_safe(_gauss(DK.shape, gen, dev).double())
        target[0::2] = alpha * _slot_norm(DK)
        exact[1::2] = True
    elif arm == "C1-V":
        delta[1::2] = alpha * _slot_norm(DV) * _unit_safe(_gauss(DV.shape, gen, dev).double())
        target[1::2] = alpha * _slot_norm(DV)
        exact[0::2] = True
    else:
        raise ValueError(arm)
    if alpha == 0.0:
        exact = torch.ones(two_l, dtype=torch.bool, device=dev)
    u = F + delta
    u[exact] = F[exact]
    info["target"], info["exact_rows"], info["pre"] = target, exact, u
    return to_fp16(u), info


# ----------------------------------------------------------------------------- predictors (section 6)
def group_map(n_q, n_kv, device):
    """query head g -> KV head (HF repeat_kv: query heads g*n_rep .. are served by KV head g // n_rep)."""
    return torch.arange(n_q, device=device) // (n_q // n_kv)


def _attn_probs(q, K, hmap, scaling):
    """q [n_q, hd] float64, K [n_kv, T, hd] -> softmax over T of q.k*scaling, per query head, float64."""
    lg = torch.einsum("gd,gjd->gj", q, K[hmap]) * scaling
    return torch.softmax(lg, dim=-1)


class PredictorState:
    """Holds delta = stored - kappa^F of every decode-written position (float64) and evaluates the five predictors at each step."""

    def __init__(self, n_layers, n_kv, n_q, head_dim, P0, scaling, S, outliers=None, device="cpu"):
        self.n_layers, self.n_kv, self.n_q, self.hd, self.P0, self.scaling = n_layers, n_kv, n_q, head_dim, P0, scaling
        self.dK = torch.zeros(n_layers, n_kv, S, head_dim, dtype=torch.float64, device=device)
        self.dV = torch.zeros_like(self.dK)
        self.cum2 = torch.zeros(2 * n_layers, n_kv, dtype=torch.float64, device=device)
        self.share = []
        self.n = 0
        self.hmap = group_map(n_q, n_kv, device)
        self.outliers = outliers                                   # LongTensor [2L, n_kv, N_OUTLIER] or None

    def record_write(self, stored, kF):
        d = stored.double() - kF.double()                          # [2L, n_kv, hd]
        self.dK[:, :, self.n, :] = d[0::2]
        self.dV[:, :, self.n, :] = d[1::2]
        self.cum2 += (d * d).sum(-1)
        if self.outliers is not None:
            sq = d * d
            tot = sq.sum(-1)
            out = torch.gather(sq, -1, self.outliers).sum(-1)
            m = tot > 0
            self.share.append(float((out[m] / tot[m]).mean()) if bool(m.any()) else float("nan"))
        self.n += 1

    def evaluate(self, cache, queries):
        """Predictors at the current step from the cache as read by the FP16 pass (stored keys; new position = kappa^F)."""
        n = self.n
        if n == 0:
            return {"P_norm": 0.0, "P_logit": 0.0, "P_val": 0.0, "P_anormK": 0.0, "P_out": float("nan")}
        lo, hi = self.P0, self.P0 + n
        pl = pv = pa = 0.0
        isd = 1.0 / math.sqrt(self.hd)
        for l, lay in enumerate(cache.layers):
            q = queries[l].double()
            K = lay.keys[0].double()
            p = _attn_probs(q, K, self.hmap, self.scaling)[:, lo:hi]                    # [n_q, n]
            dk = self.dK[l, :, :n][self.hmap]                                           # [n_q, n, hd]
            dv = self.dV[l, :, :n][self.hmap]
            pl += float((p * torch.einsum("gd,gjd->gj", q, dk).abs() * isd).sum())
            pa += float((p * dk.norm(dim=-1)).sum())
            pv += float(torch.einsum("gj,gjd->gd", p, dv).norm(dim=-1).sum())
        denom = self.n_layers * self.n_q
        sh = [x for x in self.share[:n] if x == x]
        return {"P_norm": float((self.cum2 / n).sqrt().mean()), "P_logit": pl / denom, "P_val": pv / denom, "P_anormK": pa / denom,
                "P_out": (sum(sh) / len(sh)) if sh else float("nan")}


# ----------------------------------------------------------------------------- calibration accumulation (section 5)
class CalibAccumulator:
    """Per model, over calibration problems: query second moments per (layer, KV head) from the FP16 reference pass (post-RoPE, pooled
    within the query group); from the NAT arm: sum of Delta_c^2 per (2L, n_kv, hd), relative norms ||Delta||/||kappa^F|| per slot (K and V),
    sum of |kappa^F_c|."""

    def __init__(self):
        self.M = None
        self.n_q = 0
        self.d2 = None
        self.absF = None
        self.n_meas = 0
        self.rhoK, self.rhoV = [], []

    def on_queries(self, queries, n_kv):
        Q = torch.stack([q.double() for q in queries])                                   # [L, n_q, hd]
        hmap = group_map(Q.shape[1], n_kv, Q.device)
        M = torch.zeros(Q.shape[0], n_kv, Q.shape[2], Q.shape[2], dtype=torch.float64, device=Q.device)
        for h in range(n_kv):
            Qh = Q[:, hmap == h]                                                          # [L, group, hd]
            M[:, h] = torch.einsum("lgd,lge->lde", Qh, Qh)
        self.M = M if self.M is None else self.M + M
        self.n_q += 1

    def on_measure(self, kF, kM):
        F = kF.double()
        D = kM.double() - F
        self.d2 = (D * D) if self.d2 is None else self.d2 + D * D
        self.absF = F.abs() if self.absF is None else self.absF + F.abs()
        self.n_meas += 1
        nF = F.norm(dim=-1).clamp_min(1e-30)
        r = D.norm(dim=-1) / nF
        self.rhoK.append(r[0::2].flatten().cpu())
        self.rhoV.append(r[1::2].flatten().cpu())

    def state(self):
        return {"M": self.M.cpu(), "n_q": self.n_q, "d2": self.d2.cpu(), "absF": self.absF.cpu(), "n_meas": self.n_meas,
                "rhoK": torch.cat(self.rhoK), "rhoV": torch.cat(self.rhoV)}


def subspace_from_M(M, r=R_SUBSPACE):
    """M [L, n_kv, hd, hd] -> U_r [L, n_kv, hd, r] (float32), E_r [L, n_kv], r80 [L, n_kv]."""
    evals, evecs = torch.linalg.eigh(M.double())                  # ascending
    evals, evecs = evals.flip(-1), evecs.flip(-1)
    tot = evals.sum(-1, keepdim=True).clamp_min(1e-300)
    cum = evals.cumsum(-1) / tot
    E_r = cum[..., r - 1]
    r80 = (cum < 0.80).sum(-1) + 1
    return evecs[..., :r].float().contiguous(), E_r, r80


# ----------------------------------------------------------------------------- one controlled step and the teacher-forced arm
def controlled_step_v3(backend, cur, cache, arm, alpha, gen, vgen, ctx, cap=None, pred=None, acc=None):
    L = cache.get_seq_length()
    measure_pass(backend, cur, cache, MEASURE.get(arm, "NF4"))
    kM = C._stack_slices(_snapshot_last(cache, L))
    cache.crop(L)
    if cap is not None:
        cap.active = True
    try:
        lg_F, _ = backend.run(cur, cache, False)
    finally:
        if cap is not None:
            cap.active = False
    assert cache.get_seq_length() == L + 1
    kF = C._stack_slices(_snapshot_last(cache, L))
    new, info = stored_entry_v3(arm, alpha, kF, kM, gen, vgen, ctx)
    preds = pred.evaluate(cache, cap.queries()) if (pred is not None and cap is not None) else None
    if acc is not None:
        acc.on_measure(kF, kM)
    C._write_back(cache, L, new)
    if pred is not None:
        pred.record_write(new, kF)
    # diagnostics
    F = kF.float()
    D = kM.float() - F
    F64 = kF.double()
    u = info["pre"]
    back = C._stack_slices(_snapshot_last(cache, L))               # what the cache holds at the new position after the write-back
    ex = info["exact_rows"]
    d = {"exact_ok": bool(torch.equal(new[ex], kF[ex])) if bool(ex.any()) else True,
         "rel_K": float((_slot_norm(D[0::2]) / _slot_norm(F[0::2]).clamp_min(1e-12)).mean()),
         "rel_V": float((_slot_norm(D[1::2]) / _slot_norm(F[1::2]).clamp_min(1e-12)).mean()),
         "dn2_K": float((D[0::2] ** 2).sum(-1).mean()), "dn2_V": float((D[1::2] ** 2).sum(-1).mean()),
         "target_norm_mean": float(_slot_norm(D).mean()),
         "zero_slots": info["zero_slots"], "leak": info["leak"], "routing_reset": overrides_clear(backend.model),
         # amendment 2 (b): the stored entry is FP16, is what the cache holds, and is the correctly rounded FP16 of the float64 pre-rounding entry
         "rounding_ok": bool(new.dtype == torch.float16 and back.dtype == torch.float16 and torch.equal(new, back)
                             and torch.equal(back, fp16_rne_reference(u)))}
    nbin = RDEV_EDGES.numel() + 1
    if info["target"] is None:                                   # natural arm: stored rows equal the measured entry bitwise
        nat_rows = ~ex
        d["natural_ok"] = bool(torch.equal(new[nat_rows], kM[nat_rows])) if bool(nat_rows.any()) else True
        d["norm_dev_max"] = d["norm_dev_mean"] = 0.0
        d.update(construction_ok=True, constr_dev_max=0.0, rdev_max=0.0, rdev_n_gt5=0, rdev_n=0, rdev_sum=0.0, rdev_arg=None, rdev_hist=[0] * nbin)
    else:
        tgt = info["target"].squeeze(-1)
        pert = (~ex).unsqueeze(-1).expand_as(tgt)
        pos = pert & (tgt > 0)
        # amendment 2 (a): construction, pre-rounding, float64
        rpre = (u - F64).norm(dim=-1)
        cdev = torch.where(pos, (rpre - tgt).abs() / tgt.clamp_min(1e-300), torch.zeros_like(tgt))
        zero_bad = bool((pert & (tgt == 0) & (rpre != 0)).any())
        d["constr_dev_max"] = float(cdev.max())
        d["construction_ok"] = bool(d["constr_dev_max"] <= CONSTR_TOL and not zero_bad)
        # reported, not gated: realized post-rounding relative deviation per slot
        rn = (back.double() - F64).norm(dim=-1)
        rdev = torch.where(pos, (rn - tgt).abs() / tgt.clamp_min(1e-300), torch.zeros_like(tgt))
        vals = rdev[pos]
        k = int(rdev.argmax())
        r_, h_ = divmod(k, rdev.shape[1])
        d.update(rdev_max=float(rdev.max()), rdev_n_gt5=int((vals > RDEV_FLAG).sum()), rdev_n=int(vals.numel()), rdev_sum=float(vals.sum()),
                 rdev_arg=[r_ // 2, "K" if r_ % 2 == 0 else "V", h_] if vals.numel() else None,
                 rdev_hist=torch.bincount(torch.bucketize(vals.cpu(), RDEV_EDGES), minlength=nbin).tolist())
        # E1 floor rule, descriptive only (comparability with E1/E1L)
        den = tgt + FP16_RES * F64.norm(dim=-1)
        rel = ((rn - tgt).abs() / den)[pert]
        d["norm_dev_max"] = float(rel.max()) if rel.numel() else 0.0
        d["norm_dev_mean"] = float(rel.mean()) if rel.numel() else 0.0
        d["natural_ok"] = True
    return lg_F, d, preds


@torch.no_grad()
def reference_logits_v3(backend, prompt_ids, ref_ids, S, cap=None, acc=None, n_kv=None):
    """Pure FP16 teacher-forced logits (as rv2.controls.reference_logits); optionally feeds post-RoPE queries to the calibration accumulator."""
    logits, cache = backend.run(prompt_ids, None, use_4bit=False)
    out = []
    for s in range(1, S + 1):
        cur = torch.tensor([[ref_ids[s - 1]]], device=prompt_ids.device)
        if cap is not None:
            cap.active = True
        try:
            lg, _ = backend.run(cur, cache, False)
        finally:
            if cap is not None:
                cap.active = False
        if acc is not None:
            acc.on_queries(cap.queries(), n_kv)
        out.append(lg.float())
    return torch.cat(out, dim=0)


@torch.no_grad()
def teacher_forced_arm_v3(backend, prompt_ids, ref_ids, S, arm, alpha, seeds, ref_lg, ctx=None, cap=None, predictors=True, acc=None,
                          return_cache=False):
    if arm in NEEDS_CTX and ctx is None:
        raise ValueError(f"{arm} needs a context (calibration context for runs; synthetic context for logic gates)")
    gen = C.make_generator(prompt_ids.device, seeds[0])
    vgen = C.make_generator(prompt_ids.device, seeds[1])
    _, cache = backend.run(prompt_ids, None, use_4bit=False)
    cfg = backend.model.config
    pred = None
    if predictors and cap is not None:
        out_idx = ctx["outliers"] if (ctx is not None and "outliers" in ctx) else None
        pred = PredictorState(cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.num_attention_heads, cap.head_dim, prompt_ids.shape[1],
                              cap.scaling, S, out_idx, prompt_ids.device)
    keys = ["js", "flip", "top", "norm_dev_max", "norm_dev_mean", "rel_K", "rel_V", "dn2_K", "dn2_V", "target_norm_mean", "zero_slots", "leak",
            "constr_dev_max", "rdev_max", "rdev_n_gt5", "rdev_arg"]
    rec = {k: [] for k in keys}
    hist, rsum, rcount = [0] * (RDEV_EDGES.numel() + 1), 0.0, 0
    pk = ["P_norm", "P_logit", "P_val", "P_anormK", "P_out"]
    for k in pk:
        rec[k] = []
    flags = {"exact_ok": True, "natural_ok": True, "routing_reset": True, "construction_ok": True, "rounding_ok": True}
    for s in range(1, S + 1):
        cur = torch.tensor([[ref_ids[s - 1]]], device=prompt_ids.device)
        lg, d, pr = controlled_step_v3(backend, cur, cache, arm, alpha, gen, vgen, ctx, cap if predictors else None, pred, acc)
        r = ref_lg[s - 1:s]
        rec["js"].append(float(C.js64(lg, r)))
        rec["flip"].append(float((torch.argmax(lg, dim=-1) != torch.argmax(r, dim=-1)).reshape(())))
        rec["top"].append(int(torch.argmax(lg, dim=-1).item()))
        for k in keys[3:]:
            rec[k].append(d[k])
        hist = [x + y for x, y in zip(hist, d["rdev_hist"])]
        rsum += d["rdev_sum"]
        rcount += d["rdev_n"]
        for k in flags:
            flags[k] = flags[k] and bool(d[k])
        if pr is not None:
            for k in pk:
                rec[k].append(pr[k])
    rec.update(flags)
    rec.update(rdev_hist=hist, rdev_sum=rsum, rdev_count=rcount)
    if not predictors or cap is None:
        for k in pk:
            rec.pop(k)
    return (rec, cache) if return_cache else rec


@torch.no_grad()
def fp16_teacher_forced_cache(backend, prompt_ids, ref_ids, S):
    """Pure FP16 teacher-forced cache after S steps (reference for the alpha = 0 gate)."""
    _, cache = backend.run(prompt_ids, None, use_4bit=False)
    for s in range(1, S + 1):
        backend.run(torch.tensor([[ref_ids[s - 1]]], device=prompt_ids.device), cache, False)
    return cache


# ----------------------------------------------------------------------------- contexts
def synthetic_ctx(cfg, device, seed=0, rho_star=0.05):
    """Logic-gate context (no data): random orthonormal U_8 per (layer, KV head), unit channel variances, first 4 channels as outliers."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    L, h, hd = cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.hidden_size // cfg.num_attention_heads
    hd = getattr(cfg, "head_dim", None) or hd
    A = torch.randn(L, h, hd, hd, generator=g, dtype=torch.float64)
    Q, _ = torch.linalg.qr(A)
    return {"U": Q[..., :R_SUBSPACE].float().to(device), "sigma2": torch.ones(2 * L, h, hd, device=device),
            "rho_star": rho_star, "outliers": torch.arange(N_OUTLIER, device=device).expand(2 * L, h, N_OUTLIER).contiguous()}


def ctx_from_calibration(cal: dict, device):
    return {"U": torch.tensor(cal["U8"], dtype=torch.float32, device=device), "sigma2": torch.tensor(cal["sigma2"], dtype=torch.float32, device=device),
            "rho_star": float(cal["rho_star"]), "outliers": torch.tensor(cal["outliers"], dtype=torch.long, device=device)}
