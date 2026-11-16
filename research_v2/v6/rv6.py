"""v6 core (PROTOCOL_V6.md): the logging study's arms, the read-time and write-time logging, and the controlled step.

The step is the locked v4 step, unchanged (rv4.controlled_step_v4 order): NF4 measurement pass -> crop -> FP16 pass (its logits are the arm's
logits; kappa^F) -> overwrite the new entry with the arm's stored value. v6 adds, with no effect on any computed logit or stored entry:
  - read-time logging after the FP16 pass and before the write-back, from the post-RoPE queries captured read-only by the locked
    rv3.core.QueryCapture (hooks return None) and the cache (stored entries on the decode positions, kappa^F at the new position);
  - write-time logging of the stored error delta = stored - kappa^F (slot sums, per-rotary-pair sums, pre-RoPE key error by exact inverse
    rotation, alignment of value error with o_proj and with the prompt's top value-covariance directions);
  - a read-only forward hook on each k_proj (its input, the normalized layer input) for the input common-mode share a_in.
All logging is float64. The four v4 arms use rv4.stored_entry_v4 (the locked construction); every other constructed entry is built in float64
and stored as V.to_fp16 (the locked one-step correctly rounded FP16).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RV = HERE.parent
sys.path.insert(0, str(RV))
sys.path.insert(0, str(RV / "v4"))
import rv4  # noqa: E402  (sets CUBLAS_WORKSPACE_CONFIG before torch)

import numpy as np  # noqa: E402
import torch  # noqa: E402

V, C = rv4.V, rv4.C
from rv2.engine import _snapshot_last  # noqa: E402

MASTER_SEED = "rv6-logging-2026-11-15"
V4_ARMS = ["NAT-K", "NAT-V", "C1-K", "C1-V"]
ARMS = ["NAT-K", "NAT-V", "C1-K", "C1-V", "NAT-K-cen", "NAT-V-cen", "ST-K", "ST-V", "ST-K@nat", "ST-V@nat", "ST-V@nat-cen", "ST-Kc", "ST-Kc@nat"]
PATH = {a: ("K" if ("-K" in a) else "V") for a in ARMS}
CEN_SRC = {"NAT-K-cen": "NAT-K", "NAT-V-cen": "NAT-V", "ST-V@nat-cen": "ST-V@nat"}     # centred arm -> the arm whose problem mean is removed
CODEC_BITS, CODEC_GROUP = 4, 32
QMAX = 2 ** CODEC_BITS - 1
DEGEN_TOL = 1e-12
READ_KEYS = ["m", "N_D", "meas", "lin_V", "mod_V", "lin_K", "modK_c", "modK_w", "jensen", "meas_dm", "lam_K", "contrast", "s2", "mu2_V"]


# ----------------------------------------------------------------------------- codecs (float64 in, float64 out; FP16 scales)
def _fp16(x):
    return x.half().double()


def codec_per_token(x, g=CODEC_GROUP):
    """Per-token asymmetric RTN INT4 along the last dim in groups of g (KIVI-style): FP16 scale and FP16 group minimum as zero point,
    q = clamp(round((x - mn) / scale), 0, 15), x_hat = q * scale + mn. x [..., d] float64."""
    sh = x.shape
    xg = x.reshape(*sh[:-1], sh[-1] // g, g)
    mn = _fp16(xg.amin(-1, keepdim=True))
    scale = _fp16((xg.amax(-1, keepdim=True) - mn) / QMAX)
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    q = torch.clamp(torch.round((xg - mn) / safe), 0, QMAX)
    out = torch.where(scale > 0, q * safe + mn, xg)
    return out.reshape(sh)


def channel_ranges(prompt_keys):
    """Per (layer, KV head, channel) min and max of the prompt's FP16 keys. prompt_keys [L, n_kv, P, d] -> (mn, mx) [L, n_kv, d] float64."""
    k = prompt_keys.double()
    return k.amin(2), k.amax(2)


def codec_per_channel(x, mn, mx):
    """Static per-channel asymmetric RTN INT4 with the prompt-calibrated range (FP16 minimum and scale; values outside clamped). x [L, n_kv, d]."""
    mn16 = _fp16(mn)
    scale = _fp16((mx - mn16) / QMAX)
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    q = torch.clamp(torch.round((x - mn16) / safe), 0, QMAX)
    return torch.where(scale > 0, q * safe + mn16, x)


def codec_per_token_np(x, g=CODEC_GROUP):
    """Independent numpy implementation of codec_per_token (gate G4)."""
    x = np.asarray(x, np.float64)
    sh = x.shape
    xg = x.reshape(*sh[:-1], sh[-1] // g, g)
    mn = xg.min(-1, keepdims=True).astype(np.float16).astype(np.float64)
    scale = ((xg.max(-1, keepdims=True) - mn) / QMAX).astype(np.float16).astype(np.float64)
    nz = scale > 0
    safe = np.where(nz, scale, 1.0)
    q = np.clip(np.round((xg - mn) / safe), 0, QMAX)
    out = np.where(nz, q * safe + mn, xg)
    return out.reshape(sh)


# ----------------------------------------------------------------------------- stored entries
def _rows(path):
    return (slice(0, None, 2), slice(1, None, 2)) if path == "K" else (slice(1, None, 2), slice(0, None, 2))


def _renorm(F, Dp_dir, nD, rows, other, two_l, dev):
    """u = F + nD * unit(Dp_dir) on the path rows; the other path exact. Returns (u, exact, target, degenerate)."""
    exact = torch.zeros(two_l, dtype=torch.bool, device=dev)
    exact[other] = True
    nd = Dp_dir.norm(dim=-1, keepdim=True)
    degen = (nd == 0) & (nD > 0)
    delta = torch.zeros_like(F)
    delta[rows] = nD * V._unit_safe(Dp_dir)
    target = torch.zeros(F.shape[:-1] + (1,), dtype=torch.float64, device=dev)
    target[rows] = torch.where(degen, torch.zeros_like(nD), nD)
    u = F + delta
    u[exact] = F[exact]
    return u, exact, target, int(degen.sum())


def stored_entry_v6(arm, kF, kM, gen, vgen, ctx):
    """(stored FP16 entry, info) with info = {pre, exact_rows, target (None if not norm-matched), kind, degenerate}."""
    if arm in V4_ARMS:
        out, info = rv4.stored_entry_v4(arm, 1.0, kF, kM, gen, vgen, torch.float16)
        info = dict(info)
        info["kind"] = "natural" if arm.startswith("NAT") else "isotropic"
        info["degenerate"] = 0
        return out, info
    path = PATH[arm]
    rows, other = _rows(path)
    dev, two_l = kF.device, kF.shape[0]
    F = kF.double()
    Dn = (kM.double() - F)[rows]                                            # natural error on the path this step
    nD = Dn.norm(dim=-1, keepdim=True)
    if arm in ("ST-K", "ST-V", "ST-Kc"):
        exact = torch.zeros(two_l, dtype=torch.bool, device=dev)
        exact[other] = True
        u = F.clone()
        u[rows] = codec_per_token(F[rows]) if arm != "ST-Kc" else codec_per_channel(F[rows], *ctx["kc_range"])
        return V.to_fp16(u), {"pre": u, "exact_rows": exact, "target": None, "kind": "codec", "degenerate": 0}
    if arm in ("ST-K@nat", "ST-V@nat", "ST-Kc@nat"):
        q = codec_per_token(F[rows]) if arm != "ST-Kc@nat" else codec_per_channel(F[rows], *ctx["kc_range"])
        u, exact, target, dg = _renorm(F, q - F[rows], nD, rows, other, two_l, dev)
        return V.to_fp16(u), {"pre": u, "exact_rows": exact, "target": target, "kind": "codec@nat", "degenerate": dg}
    if arm in ("NAT-K-cen", "NAT-V-cen"):
        c = Dn - ctx["mu"]                                                  # pilot construction (rv3x13._centred)
        degen = (c.norm(dim=-1, keepdim=True) < DEGEN_TOL * nD) & (nD > 0)
        u, exact, target, _ = _renorm(F, torch.where(degen, Dn, c), nD, rows, other, two_l, dev)
        return V.to_fp16(u), {"pre": u, "exact_rows": exact, "target": target, "kind": "centred", "degenerate": int(degen.sum())}
    if arm == "ST-V@nat-cen":
        q = codec_per_token(F[rows])
        ds = q - F[rows]
        Dst = nD * V._unit_safe(ds)                                         # the ST-V@nat error of this step
        c = Dst - ctx["mu"]
        degen = (c.norm(dim=-1, keepdim=True) < DEGEN_TOL * nD) & (nD > 0)
        u, exact, target, _ = _renorm(F, torch.where(degen, Dst, c), nD, rows, other, two_l, dev)
        return V.to_fp16(u), {"pre": u, "exact_rows": exact, "target": target, "kind": "centred", "degenerate": int(degen.sum())}
    raise ValueError(arm)


# ----------------------------------------------------------------------------- logging
def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def inverse_rope(x, cos, sin):
    """Exact inverse of HF apply_rotary_pos_emb (k*cos + rotate_half(k)*sin) at one position."""
    return x * cos - rotate_half(x) * sin


class Log6:
    """Per (model, problem, arm) logging. Geometry and the fixed model tensors (o_proj Gram matrices) come from `ModelInfo`."""

    def __init__(self, mi, P0, S, prompt_values, device):
        self.mi, self.P0, self.S, self.dev = mi, P0, S, device
        L, nkv, d = mi.n_layers, mi.n_kv, mi.hd
        self.D = {"K": torch.zeros(L, nkv, S, d, dtype=torch.float64, device=device),
                  "V": torch.zeros(L, nkv, S, d, dtype=torch.float64, device=device)}
        self.read = {k: torch.full((S, L, nkv), float("nan"), dtype=torch.float64, device=device) for k in READ_KEYS}
        self.A_o = torch.full((S, L, nkv), float("nan"), dtype=torch.float64, device=device)
        self.A_cov = torch.full_like(self.A_o, float("nan"))
        self.sums = rv4.SlotSums()
        h = d // 2
        self.S2K_pair = torch.zeros(L, nkv, h, dtype=torch.float64, device=device)
        self.S1K_pre = torch.zeros(L, nkv, d, dtype=torch.float64, device=device)
        self.S2K_pre_pair = torch.zeros(L, nkv, h, dtype=torch.float64, device=device)
        self.S1x = torch.zeros(L, mi.hidden, dtype=torch.float64, device=device)
        self.S2x = torch.zeros(L, dtype=torch.float64, device=device)
        k = max(1, d // 8)                                                     # top d/8 value-covariance directions of the prompt
        pv = prompt_values.double()                                            # [L, nkv, P0, d]
        pc = pv - pv.mean(2, keepdim=True)
        cov = torch.einsum("lhpd,lhpe->lhde", pc, pc) / max(1, pv.shape[2] - 1)
        _, U = torch.linalg.eigh(cov)
        Uk = U[..., -k:]
        self.Pk = Uk @ Uk.transpose(-1, -2)
        self.k = k

    def read_time(self, cache, s, queries):
        """After the FP16 pass of step s (1-based), before the write-back: decode positions P0..P0+s-2 hold stored entries, P0+s-1 is kappa^F."""
        n = s - 1
        if n < 1:
            return
        mi, P0 = self.mi, self.P0
        hmap, sc = mi.hmap, mi.scaling
        for l, lay in enumerate(cache.layers):
            q = queries[l].double()                                            # [n_q, d]
            Kst, Vst = lay.keys[0].double(), lay.values[0].double()            # [n_kv, T, d]
            dK, dV = self.D["K"][l][:, :n], self.D["V"][l][:, :n]
            Ksh, Vsh = Kst.clone(), Vst.clone()
            Ksh[:, P0:P0 + n] -= dK
            Vsh[:, P0:P0 + n] -= dV
            a = torch.softmax(torch.einsum("gd,gjd->gj", q, Ksh[hmap]) * sc, dim=-1)
            ast = torch.softmax(torch.einsum("gd,gjd->gj", q, Kst[hmap]) * sc, dim=-1)
            o_sh = torch.einsum("gj,gjd->gd", a, Vsh[hmap])
            o_st = torch.einsum("gj,gjd->gd", ast, Vst[hmap])
            aD = a[:, P0:P0 + n]
            m = aD.sum(-1)
            A2 = (aD * aD).sum(-1)
            mm = m.clamp_min(1e-300)
            # value path
            dVg = dV[hmap]                                                     # [n_q, n, d]
            muV = dVg.mean(1)
            eps = dVg - muV[:, None]
            lin_V = torch.einsum("gj,gjd->gd", aD, dVg).pow(2).sum(-1)
            mod_V = m * m * muV.pow(2).sum(-1) + (aD * aD * eps.pow(2).sum(-1)).sum(-1)
            # key path, first order
            ds = torch.einsum("gd,gjd->gj", q, dK[hmap]) * sc                  # [n_q, n]
            beta = ds.mean(1)
            eta = ds - beta[:, None]
            v = Vsh[hmap]
            vD = v[:, P0:P0 + n]
            vbarD = torch.einsum("gj,gjd->gd", aD, vD) / mm[:, None]
            aP = a.clone()
            aP[:, P0:P0 + n] = 0
            vbarP = torch.einsum("gj,gjd->gd", aP, v) / (1 - m).clamp_min(1e-300)[:, None]
            Cterm = (beta * m * (1 - m))[:, None] * (vbarD - vbarP)
            W = torch.einsum("gj,gjd->gd", aD * eta, vD - o_sh[:, None])
            lin_K = (Cterm + W).pow(2).sum(-1)
            etab = (aD * eta).sum(-1) / mm
            var_a = (aD * (eta - etab[:, None]) ** 2).sum(-1) / mm
            jensen = m * (1 - m) * var_a / 2
            meas_dm = ast[:, P0:P0 + n].sum(-1) - m
            qd = torch.einsum("gd,gjd->gj", q, dK[hmap])
            dk2 = dK[hmap].pow(2).sum(-1).sum(-1)
            lam = torch.where(dk2 > 0, qd.pow(2).sum(-1) / (q.pow(2).sum(-1) * dk2 / mi.hd).clamp_min(1e-300), torch.full_like(dk2, float("nan")))
            contrast = (vbarD - vbarP).pow(2).sum(-1)
            s2 = (aD * (vD - o_sh[:, None]).pow(2).sum(-1)).sum(-1) / mm
            vals = {"m": m, "N_D": m * m / A2.clamp_min(1e-300), "meas": (o_st - o_sh).pow(2).sum(-1), "lin_V": lin_V, "mod_V": mod_V,
                    "lin_K": lin_K, "modK_c": Cterm.pow(2).sum(-1), "modK_w": W.pow(2).sum(-1), "jensen": jensen, "meas_dm": meas_dm,
                    "lam_K": lam, "contrast": contrast, "s2": s2, "mu2_V": muV.pow(2).sum(-1)}
            summed = {"meas", "lin_V", "mod_V", "lin_K", "modK_c", "modK_w"}         # energies: sum over the query heads of a KV head
            for k, x in vals.items():
                g = x.reshape(mi.n_kv, -1)
                self.read[k][s - 1, l] = g.sum(-1) if k in summed else g.mean(-1)

    def write_time(self, s, new, kF, kM, cossin, xin):
        d = new.double() - kF.double()
        dK, dV = d[0::2], d[1::2]
        self.D["K"][:, :, s - 1] = dK
        self.D["V"][:, :, s - 1] = dV
        self.sums.record(new, kF)
        h = self.mi.hd // 2
        self.S2K_pair += dK[..., :h] ** 2 + dK[..., h:] ** 2
        nat = (kM.double() - kF.double())[0::2]                                # natural key error, post-RoPE, this step
        cos, sin = cossin
        pre = inverse_rope(nat, cos, sin)
        self.S1K_pre += pre
        self.S2K_pre_pair += pre[..., :h] ** 2 + pre[..., h:] ** 2
        e2 = dV.pow(2).sum(-1)
        ok = e2 > 0
        G, trG = self.mi.G, self.mi.trG
        quad = torch.einsum("lhd,lhde,lhe->lh", dV, G, dV)
        self.A_o[s - 1] = torch.where(ok, (quad / e2.clamp_min(1e-300)) / (trG / self.mi.hd), torch.full_like(e2, float("nan")))
        pk = torch.einsum("lhde,lhe->lhd", self.Pk, dV).pow(2).sum(-1)
        self.A_cov[s - 1] = torch.where(ok, (pk / e2.clamp_min(1e-300)) / (self.k / self.mi.hd), torch.full_like(e2, float("nan")))
        x = xin.double()                                                       # [L, hidden]
        self.S1x += x
        self.S2x += x.pow(2).sum(-1)

    def arrays(self, S_used):
        out = {k: v[:S_used].float().cpu().numpy() for k, v in self.read.items()}
        out.update({"A_o": self.A_o[:S_used].float().cpu().numpy(), "A_cov": self.A_cov[:S_used].float().cpu().numpy()})
        out.update(self.sums.arrays())
        out.update({"S2K_pair": self.S2K_pair.cpu().numpy(), "S1K_pre": self.S1K_pre.cpu().numpy(), "S2K_pre_pair": self.S2K_pre_pair.cpu().numpy(),
                    "S1x": self.S1x.cpu().numpy(), "S2x": self.S2x.cpu().numpy()})
        return out


class ModelInfo:
    """Geometry, the read-only hooks and the fixed o_proj Gram matrices G[l, h] = sum over the query heads g of KV head h of W_o,g^T W_o,g."""

    def __init__(self, backend):
        model = backend.model
        cfg = model.config
        self.cap = V.QueryCapture(model)
        self.n_layers = cfg.num_hidden_layers
        self.n_q, self.n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
        self.hd = self.cap.head_dim
        self.hidden = cfg.hidden_size
        self.scaling = self.cap.scaling
        dev = next(model.parameters()).device
        self.hmap = V.group_map(self.n_q, self.n_kv, dev)
        rep = self.n_q // self.n_kv
        Gs = []
        for lay in model.model.layers:
            op = lay.self_attn.o_proj
            W = (getattr(op, "fp16_weight", None) if getattr(op, "fp16_weight", None) is not None else op.weight).double()   # [hidden, n_q*hd]
            Wg = W.reshape(W.shape[0], self.n_q, self.hd)
            Gq = torch.einsum("xgd,xge->gde", Wg, Wg)                          # [n_q, hd, hd]
            Gs.append(Gq.reshape(self.n_kv, rep, self.hd, self.hd).sum(1))
        self.G = torch.stack(Gs)                                               # [L, n_kv, hd, hd]
        self.trG = torch.diagonal(self.G, dim1=-2, dim2=-1).sum(-1)
        self.xin = [None] * self.n_layers
        self.hooks = []
        for i, lay in enumerate(model.model.layers):
            def kh(mod, args, out, i=i):
                if self.cap.active:
                    self.xin[i] = args[0].detach()[0, -1]
            self.hooks.append(lay.self_attn.k_proj.register_forward_hook(kh))

    def remove(self):
        self.cap.remove()
        for h in self.hooks:
            h.remove()


# ----------------------------------------------------------------------------- the controlled step and one arm
def controlled_step_v6(backend, mi, cur, cache, arm, gen, vgen, ctx, log, s):
    L = cache.get_seq_length()
    V.measure_pass(backend, cur, cache, "NF4")
    kM = C._stack_slices(_snapshot_last(cache, L))
    cache.crop(L)
    mi.cap.active = True
    try:
        lg_F, _ = backend.run(cur, cache, False)
    finally:
        mi.cap.active = False
    assert cache.get_seq_length() == L + 1
    kF = C._stack_slices(_snapshot_last(cache, L))
    if log is not None:
        log.read_time(cache, s, mi.cap.queries())
    new, info = stored_entry_v6(arm, kF, kM, gen, vgen, ctx)
    C._write_back(cache, L, new)
    if log is not None:
        cos, sin = (x.double()[0, -1] for x in mi.cap.pos[0])
        log.write_time(s, new, kF, kM, (cos, sin), torch.stack(mi.xin))
    back = C._stack_slices(_snapshot_last(cache, L))
    u, ex = info["pre"], info["exact_rows"]
    d = {"exact_ok": bool(torch.equal(new[ex], kF[ex])) if bool(ex.any()) else True,
         "rounding_ok": bool(new.dtype == torch.float16 and torch.equal(new, back) and torch.equal(back, V.fp16_rne_reference(u))),
         "finite_ok": rv4._finite(lg_F, kM, kF, new), "routing_reset": V.overrides_clear(backend.model),
         "degenerate": info.get("degenerate", 0)}
    if info["kind"] == "natural":
        nat = ~ex
        d["natural_ok"] = bool(torch.equal(new[nat], kM[nat]))
    else:
        d["natural_ok"] = True
    if info["target"] is not None:
        tgt = info["target"].squeeze(-1)
        pos = (~ex).unsqueeze(-1).expand_as(tgt) & (tgt > 0)
        rpre = (u - kF.double()).norm(dim=-1)
        cdev = torch.where(pos, (rpre - tgt).abs() / tgt.clamp_min(1e-300), torch.zeros_like(tgt))
        d["construction_ok"] = bool(float(cdev.max()) <= V.CONSTR_TOL)
        rn = (back.double() - kF.double()).norm(dim=-1)
        rdev = torch.where(pos, (rn - tgt).abs() / tgt.clamp_min(1e-300), torch.zeros_like(tgt))
        d["rdev_max"], d["rdev_n_gt5"] = float(rdev.max()), int((rdev[pos] > V.RDEV_FLAG).sum())
    else:
        d["construction_ok"], d["rdev_max"], d["rdev_n_gt5"] = True, 0.0, 0
    Ff = kF.double()
    Dn = kM.double() - Ff
    d["rel_K"] = float((V._slot_norm(Dn[0::2]) / V._slot_norm(Ff[0::2]).clamp_min(1e-12)).mean())
    d["rel_V"] = float((V._slot_norm(Dn[1::2]) / V._slot_norm(Ff[1::2]).clamp_min(1e-12)).mean())
    st = (new.double() - Ff)
    d["stored_rel_K"] = float((V._slot_norm(st[0::2]) / V._slot_norm(Ff[0::2]).clamp_min(1e-12)).mean())
    d["stored_rel_V"] = float((V._slot_norm(st[1::2]) / V._slot_norm(Ff[1::2]).clamp_min(1e-12)).mean())
    return lg_F, d


FLAGS = ["exact_ok", "rounding_ok", "finite_ok", "routing_reset", "natural_ok", "construction_ok"]
STEP_KEYS = ["rel_K", "rel_V", "stored_rel_K", "stored_rel_V", "rdev_max", "rdev_n_gt5", "degenerate"]


@torch.no_grad()
def teacher_forced_arm_v6(backend, mi, prompt_ids, ref_ids, S, arm, seeds, ref_lg, ctx, logging=True):
    gen = C.make_generator(prompt_ids.device, seeds[0])
    vgen = C.make_generator(prompt_ids.device, seeds[1])
    _, cache = backend.run(prompt_ids, None, use_4bit=False)
    P0 = cache.get_seq_length()
    if arm in ("ST-Kc", "ST-Kc@nat"):
        pk = torch.stack([lay.keys[0][:, :P0] for lay in cache.layers])        # [L, n_kv, P0, d]
        ctx = dict(ctx or {})
        ctx["kc_range"] = channel_ranges(pk)
    log = None
    if logging:
        pv = torch.stack([lay.values[0][:, :P0] for lay in cache.layers])
        log = Log6(mi, P0, S, pv, prompt_ids.device)
    rec = {k: [] for k in ["js", "flip", "top"] + STEP_KEYS}
    flags = {k: True for k in FLAGS}
    for s in range(1, S + 1):
        cur = torch.tensor([[ref_ids[s - 1]]], device=prompt_ids.device)
        lg, d = controlled_step_v6(backend, mi, cur, cache, arm, gen, vgen, ctx, log, s)
        r = ref_lg[s - 1:s]
        rec["js"].append(float(C.js64(lg, r)))
        rec["flip"].append(float((torch.argmax(lg, dim=-1) != torch.argmax(r, dim=-1)).reshape(())))
        rec["top"].append(int(torch.argmax(lg, dim=-1).item()))
        for k in STEP_KEYS:
            rec[k].append(d[k])
        for k in FLAGS:
            flags[k] = flags[k] and bool(d[k])
    rec.update(flags)
    arrays = log.arrays(S) if log is not None else None
    return rec, arrays


def slot_mean(arrays, path):
    """Per-slot mean of a run's stored error over the problem's decode entries: [L, n_kv, d] float64 (the centred arms' mu)."""
    n = arrays[f"norms{path}"].shape[0]
    return torch.from_numpy(arrays[f"S1{path}"] / n)
