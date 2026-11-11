"""X13 (v3/phase8/FOLLOWUP_X13.md): centred arms, norm-corrected FP16 rounding, the centred logit predictor, Rayleigh sums and signed
rounding deviations.

Process-local overrides of rv3.core through install() / uninstall(); no locked file is modified. CentredPredictorState.evaluate returns the
locked PredictorState.evaluate values unchanged (it calls the locked method) and adds P_logit^c; stored_entry_x13 delegates every arm except
the four X13 arms to the locked stored_entry_v3.
"""

from __future__ import annotations

import math

import torch

import rv3.core as V

CEN_ARMS = {"NAT-K-cen": "K", "NAT-V-cen": "V"}            # centred arms: path whose common mode is removed
NC_ARMS = {"C1-nc": "C1", "C1-K-nc": "C1-K"}                # norm-corrected arms: the locked arm they reproduce (construction and seed stream)
X13_ARMS = set(CEN_ARMS) | set(NC_ARMS)
SEED_LABEL = dict(NC_ARMS)                                  # nc arms draw from the locked arm's seed stream
DEGEN_TOL = 1e-12                                           # ||Delta - mu|| < DEGEN_TOL ||Delta||: keep the natural direction
NC_SEARCH_MIN = 1e-4                                        # lambda-search only where the nearest-rounding residual exceeds this
NC_REFINE_MIN = 1e-2                                        # greedy refinement only where the best lambda residual exceeds this
NC_MAX_MOVES = 64
NC_GRID_K = 4096
NC_CHUNK = 256
FLAG = 0.05                                                 # signed-deviation flag (as amendment 2's 5 %)
FP16_MAX_MAG = 0x7BFF                                       # largest finite FP16 magnitude bit pattern
_LOCKED = {"PredictorState": V.PredictorState, "stored_entry_v3": V.stored_entry_v3, "NEEDS_CTX": V.NEEDS_CTX}
INSTANCES: list = []                                        # CentredPredictorState objects in creation order (one per arm run)
STEP_LOG: list = []                                         # per-step dicts of the constructed arm being run (cleared by the driver per arm)
STATE = {"M": None, "trM": None}                            # calibration query second moment [L, n_kv, hd, hd] (float64) and its trace


# ----------------------------------------------------------------------------- signed deviation
def signed_stats(rn, t, pos, prefix=""):
    """Signed relative deviation (rn - t) / t over the slots in pos (perturbed, t > 0). rn, t: [2L, n_kv] float64."""
    s = ((rn - t) / t.clamp_min(1e-300))[pos]
    if s.numel() == 0:
        return {f"{prefix}sdev_sum": 0.0, f"{prefix}sdev_n": 0, f"{prefix}sdev_min": 0.0, f"{prefix}sdev_max": 0.0,
                f"{prefix}sdev_pos5": 0, f"{prefix}sdev_neg5": 0}
    return {f"{prefix}sdev_sum": float(s.sum()), f"{prefix}sdev_n": int(s.numel()), f"{prefix}sdev_min": float(s.min()),
            f"{prefix}sdev_max": float(s.max()), f"{prefix}sdev_pos5": int((s > FLAG).sum()), f"{prefix}sdev_neg5": int((s < -FLAG).sum())}


def _perturbed(target, exact):
    t = target.squeeze(-1)
    return (~exact).unsqueeze(-1).expand_as(t) & (t > 0), t


# ----------------------------------------------------------------------------- FP16 neighbours and norm-corrected rounding
def _from_bits(mag, neg):
    bits = mag | (neg.to(torch.int32) << 15)
    bits = torch.where(bits >= 0x8000, bits - 0x10000, bits)
    return bits.to(torch.int16).view(torch.float16)


def fp16_neighbors(v):
    """The adjacent FP16 values of v (float16): (next toward +inf, next toward -inf). -0 and +0 step to the smallest subnormals."""
    b = v.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    neg = b >= 0x8000
    mag = b & 0x7FFF
    one = torch.ones_like(mag)
    up_mag = torch.where(neg, torch.where(mag == 0, one, mag - 1), (mag + 1).clamp_max(FP16_MAX_MAG))
    up_neg = neg & (mag > 0)
    dn_mag = torch.where(neg, (mag + 1).clamp_max(FP16_MAX_MAG), torch.where(mag == 0, one, mag - 1))
    dn_neg = neg | (mag == 0)
    return _from_bits(up_mag, up_neg), _from_bits(dn_mag, dn_neg)


def lambda_grid(device):
    """{1} and (2/3) (9/4)^(k/4096), k = 0..4096, ordered by |ln lambda| ascending (lambda = 1 first; stable order on ties)."""
    k = torch.arange(NC_GRID_K + 1, dtype=torch.float64)
    lam = torch.cat([torch.ones(1, dtype=torch.float64), (2.0 / 3.0) * (9.0 / 4.0) ** (k / NC_GRID_K)])
    return lam[torch.argsort(lam.log().abs(), stable=True)].to(device)


def _resid(r16, F, t):
    return ((r16.double() - F).norm(dim=-1) - t).abs() / t


def _first_min(e, dim):
    """Minimum along dim and the index of its FIRST occurrence."""
    m = e.min(dim).values
    return m, (e == m.unsqueeze(dim)).int().argmax(dim)


def _refine(r, F, t, e):
    """Greedy: one coordinate to an adjacent FP16 value per move, the move that minimizes the residual, applied only if the exactly recomputed
    residual is strictly smaller; at most NC_MAX_MOVES. r [N, hd] float16, F [N, hd], t [N], e [N] (current residuals). Returns r, e, moves."""
    moves = torch.zeros(r.shape[0], dtype=torch.long, device=r.device)
    active = torch.ones(r.shape[0], dtype=torch.bool, device=r.device)
    hd = r.shape[1]
    for _ in range(NC_MAX_MOVES):
        ix = active.nonzero(as_tuple=True)[0]
        if ix.numel() == 0:
            break
        cur, Fn, tn = r[ix], F[ix], t[ix]
        diff = cur.double() - Fn
        n2 = (diff * diff).sum(-1)
        up, dn = fp16_neighbors(cur)
        cands = torch.stack([up, dn])                                              # [2, N, hd]
        dc = cands.double() - Fn.unsqueeze(0)
        new = (n2[None, :, None] - diff.unsqueeze(0) ** 2 + dc * dc).clamp_min(0).sqrt()
        ce = ((new - tn[None, :, None]).abs() / tn[None, :, None]).permute(1, 0, 2).reshape(ix.numel(), 2 * hd)
        _, arg = _first_min(ce, -1)
        which, coord = arg // hd, arg % hd
        rows = torch.arange(ix.numel(), device=r.device)
        trial = cur.clone()
        trial[rows, coord] = cands[which, rows, coord]
        te = _resid(trial, Fn, tn)
        ok = te < e[ix]
        r[ix[ok]] = trial[ok]
        e[ix[ok]] = te[ok]
        moves[ix[ok]] += 1
        active[ix[~ok]] = False
    return r, e, moves


def nc_round(F, delta, exact, target):
    """Norm-corrected FP16 storage of u = F + delta (FOLLOWUP_X13 section 3). F, delta float64 [2L, n_kv, hd]; exact [2L] rows stored as
    kappa^F; target [2L, n_kv, 1] the natural per-slot norm. Returns (stored float16, per-step statistics)."""
    u = F + delta
    u[exact] = F[exact]
    pos, t = _perturbed(target, exact)
    r1 = V.fp16_rne_reference(u)
    rn1 = (r1.double() - F).norm(dim=-1)
    e1 = torch.where(pos, (rn1 - t).abs() / t.clamp_min(1e-300), torch.zeros_like(t))
    r = r1.clone()
    sel = pos & (e1 > NC_SEARCH_MIN)
    idx = sel.nonzero(as_tuple=True)
    lam_sel = torch.ones(0, dtype=torch.float64, device=F.device)
    mv = torch.zeros(0, dtype=torch.long, device=F.device)
    if idx[0].numel():
        Fs, ds, ts = F[idx], u[idx] - F[idx], t[idx]                              # [N, hd], [N]
        best_r, best_e = r1[idx].clone(), e1[idx].clone()
        best_l = torch.ones_like(ts)
        grid = lambda_grid(F.device)
        for c in range(0, grid.numel(), NC_CHUNK):
            g = grid[c:c + NC_CHUNK]
            rc = V.fp16_rne_reference(Fs.unsqueeze(0) + g[:, None, None] * ds.unsqueeze(0))   # [g, N, hd]
            ec = ((rc.double() - Fs.unsqueeze(0)).norm(dim=-1) - ts.unsqueeze(0)).abs() / ts.unsqueeze(0)
            emin, arg = _first_min(ec, 0)
            better = emin < best_e
            cols = torch.arange(ts.numel(), device=F.device)
            best_r[better] = rc[arg, cols][better]
            best_e[better] = emin[better]
            best_l[better] = g[arg][better]
        mv = torch.zeros(ts.numel(), dtype=torch.long, device=F.device)
        need = best_e > NC_REFINE_MIN
        if bool(need.any()):
            rr, ee, mm = _refine(best_r[need].clone(), Fs[need], ts[need], best_e[need].clone())
            best_r[need], best_e[need], mv[need] = rr, ee, mm
        r[idx] = best_r
        lam_sel = best_l
    rn = (r.double() - F).norm(dim=-1)
    e = torch.where(pos, (rn - t).abs() / t.clamp_min(1e-300), torch.zeros_like(t))
    cos_min = 1.0
    if idx[0].numel():
        dr = r[idx].double() - F[idx]
        dd = u[idx] - F[idx]
        den = dr.norm(dim=-1) * dd.norm(dim=-1)
        ok = den > 0
        if bool(ok.any()):
            cos_min = float(((dr * dd).sum(-1)[ok] / den[ok]).min())
    st = {**signed_stats(rn, t, pos), **signed_stats(rn1, t, pos, "nn_"),
          "lam_min": float(lam_sel.min()) if lam_sel.numel() else 1.0, "lam_max": float(lam_sel.max()) if lam_sel.numel() else 1.0,
          "n_searched": int(idx[0].numel()), "moves": int(mv.sum()), "cos_min": cos_min,
          "nc_ok": bool((e[pos] <= e1[pos]).all()) if bool(pos.any()) else True}
    return r, st


# ----------------------------------------------------------------------------- stored entries
def _centred(arm, alpha, kF, kM, ctx):
    path = CEN_ARMS[arm]
    dev, two_l = kF.device, kF.shape[0]
    F = kF.double()
    D = kM.double() - F
    sl, other = (slice(0, None, 2), slice(1, None, 2)) if path == "K" else (slice(1, None, 2), slice(0, None, 2))
    exact = torch.zeros(two_l, dtype=torch.bool, device=dev)
    exact[other] = True
    target = torch.zeros(F.shape[:-1] + (1,), dtype=torch.float64, device=dev)
    delta = torch.zeros_like(F)
    Dp = D[sl]
    c = Dp - ctx["mu"].double()[sl]
    nD = Dp.norm(dim=-1, keepdim=True)
    degen = (c.norm(dim=-1, keepdim=True) < DEGEN_TOL * nD) & (nD > 0)
    delta[sl] = alpha * nD * V._unit_safe(torch.where(degen, Dp, c))
    target[sl] = alpha * nD
    if alpha == 0.0:
        exact = torch.ones(two_l, dtype=torch.bool, device=dev)
    u = F + delta
    u[exact] = F[exact]
    out = V.to_fp16(u)
    info = {"target": target, "exact_rows": exact, "zero_slots": int((nD == 0).sum()), "leak": 0.0, "pre": u}
    pos, t = _perturbed(target, exact)
    STEP_LOG.append({**signed_stats((out.double() - F).norm(dim=-1), t, pos), "degenerate": int(degen.sum())})
    return out, info


def _norm_corrected(arm, alpha, kF, kM, gen, vgen, ctx):
    out, info = _LOCKED["stored_entry_v3"](NC_ARMS[arm], alpha, kF, kM, gen, vgen, ctx)
    if alpha == 0.0:
        return out, info
    F = kF.double()
    r, st = nc_round(F, info["pre"] - F, info["exact_rows"], info["target"])
    info = dict(info)
    info["pre"] = r.double()                                   # the stored entry is FP16 by construction; the locked rounding check applies to it
    STEP_LOG.append(st)
    return r, info


def stored_entry_x13(arm, alpha, kF, kM, gen, vgen, ctx):
    if arm in CEN_ARMS:
        return _centred(arm, alpha, kF, kM, ctx)
    if arm in NC_ARMS:
        return _norm_corrected(arm, alpha, kF, kM, gen, vgen, ctx)
    out, info = _LOCKED["stored_entry_v3"](arm, alpha, kF, kM, gen, vgen, ctx)
    if info["target"] is not None and alpha != 0.0:            # a locked constructed arm run under X13: signed deviation only
        pos, t = _perturbed(info["target"], info["exact_rows"])
        STEP_LOG.append(signed_stats((out.double() - kF.double()).norm(dim=-1), t, pos))
    return out, info


# ----------------------------------------------------------------------------- predictor state: P_logit^c and Rayleigh sums
def centred_logit_term(q, K, hmap, scaling, lo, n, dk, hd):
    """One layer's sum over query heads g and ALL stored positions j of p_gj |dl_gj - sum_i p_gi dl_gi| / sqrt(hd), with dl = q . delta k on the
    decode positions lo..lo+n-1 and 0 elsewhere (prefill, new position). q [n_q, hd], K [n_kv, T, hd] float64, dk [n_q, n, hd]."""
    p = V._attn_probs(q, K, hmap, scaling)                                         # [n_q, T]
    dl = torch.zeros_like(p)
    dl[:, lo:lo + n] = torch.einsum("gd,gjd->gj", q, dk)
    m = (p * dl).sum(-1, keepdim=True)
    return float((p * (dl - m).abs()).sum()) / math.sqrt(hd)


class CentredPredictorState(_LOCKED["PredictorState"]):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.plc, self.plc_layers = [], []
        self.RN = torch.zeros(self.n_layers, self.n_kv, dtype=torch.float64, device=self.dK.device)
        self.RD = torch.zeros_like(self.RN)
        INSTANCES.append(self)

    def evaluate(self, cache, queries):
        out = super().evaluate(cache, queries)                                     # the locked values, unchanged
        n = self.n
        if n == 0:
            self.plc.append(0.0)
            self.plc_layers.append([0.0] * self.n_layers)
            return out
        terms = []
        for l, lay in enumerate(cache.layers):
            terms.append(centred_logit_term(queries[l].double(), lay.keys[0].double(), self.hmap, self.scaling, self.P0, n,
                                            self.dK[l, :, :n][self.hmap], self.hd))
        denom = self.n_layers * self.n_q
        self.plc_layers.append([x / denom for x in terms])
        self.plc.append(sum(terms) / denom)
        return out

    def record_write(self, stored, kF):
        super().record_write(stored, kF)
        if STATE["M"] is not None:
            dk = (stored.double() - kF.double())[0::2]                             # [L, n_kv, hd]
            self.RN += rayleigh_num(dk, STATE["M"])
            self.RD += STATE["trM"] / self.hd * (dk * dk).sum(-1)


def rayleigh_num(dk, M):
    """dk^T M dk per slot. dk [L, n_kv, hd], M [L, n_kv, hd, hd]."""
    return torch.einsum("lhd,lhde,lhe->lh", dk, M, dk)


# ----------------------------------------------------------------------------- install / registry / exports
def install(M=None):
    """Process-local: rv3.core uses CentredPredictorState and stored_entry_x13; the centred arms need a context (with 'mu').
    M: calibration query second moment [L, n_kv, hd, hd] for the Rayleigh sums (None: not computed)."""
    if M is not None:
        STATE["M"] = M.double()
        STATE["trM"] = torch.diagonal(STATE["M"], dim1=-2, dim2=-1).sum(-1)
    V.PredictorState = CentredPredictorState
    V.stored_entry_v3 = stored_entry_x13
    V.NEEDS_CTX = set(_LOCKED["NEEDS_CTX"]) | set(CEN_ARMS)


def uninstall():
    STATE["M"] = STATE["trM"] = None
    V.PredictorState = _LOCKED["PredictorState"]
    V.stored_entry_v3 = _LOCKED["stored_entry_v3"]
    V.NEEDS_CTX = _LOCKED["NEEDS_CTX"]


def clear():
    INSTANCES.clear()
    STEP_LOG.clear()


def mu_from(inst):
    """Per-slot mean of the NAT pass's natural Delta over all decode entries: [2L, n_kv, hd] (even rows K, odd rows V)."""
    n = inst.n
    mu = torch.zeros(2 * inst.n_layers, inst.n_kv, inst.hd, dtype=torch.float64, device=inst.dK.device)
    mu[0::2] = inst.dK[:, :, :n].mean(2)
    mu[1::2] = inst.dV[:, :, :n].mean(2)
    return mu


def record_fields(inst) -> dict:
    return {"P_logit_c": list(inst.plc), "PL_logit_c": [list(x) for x in inst.plc_layers]}


def step_fields(log) -> dict:
    """Per-step dicts -> field name -> list over steps."""
    keys = sorted({k for d in log for k in d})
    return {k: [d.get(k) for d in log] for k in keys}


def slot_arrays(inst, nat=False, raw=False) -> dict:
    """npz arrays (float64 unless raw): per-entry norms [n, L, n_kv], per-slot sums S1 [L, n_kv, hd]; NAT: Rayleigh sums [L, n_kv]."""
    n = inst.n
    dK, dV = inst.dK[:, :, :n], inst.dV[:, :, :n]                                  # [L, n_kv, n, hd]
    out = {"normsK": dK.norm(dim=-1).permute(2, 0, 1), "normsV": dV.norm(dim=-1).permute(2, 0, 1), "S1K": dK.sum(2), "S1V": dV.sum(2)}
    if nat:
        out.update(RN=inst.RN, RD=inst.RD)
    if raw:
        out.update(rawK=dK.float(), rawV=dV.float())
    return {k: v.detach().cpu().numpy() for k, v in out.items()}
