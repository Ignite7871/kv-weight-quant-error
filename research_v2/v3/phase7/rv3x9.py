"""X9 (v3/phase7/FOLLOWUP_X9.md): per-layer predictor terms, per-slot query-subspace energies and the NAT-K*-L0nat arm.

Process-local overrides of rv3.core through install() / uninstall(); no locked file is modified. PerLayerPredictorState.evaluate is the locked
PredictorState.evaluate with the per-layer terms kept: its returned values must equal the locked ones bitwise (test_x9.py checks this, and the
GPU reproduction gate checks it against the locked records). stored_entry_x9 delegates every arm except NAT-K*-L0nat to the locked function.
"""

from __future__ import annotations

import math

import torch

import rv3.core as V

X9_ARM = "NAT-K*-L0nat"
SLOT_KEYS = ("eK", "pK", "eV", "pV", "fK", "fV")
LAYER_KEYS = ("PL_logit", "PL_val", "PL_anormK", "PL_norm")
_LOCKED = {"PredictorState": V.PredictorState, "stored_entry_v3": V.stored_entry_v3, "NEEDS_CTX": V.NEEDS_CTX}
INSTANCES: list = []                    # PerLayerPredictorState objects created since the last clear(), in creation order (one per arm run)
SUBSPACE = {"U": None}                  # [L, n_kv, hd, r] float64 query-subspace basis (frozen calibration U8), set by install()


# ----------------------------------------------------------------------------- the NAT-K*-L0nat arm
def stored_entry_x9(arm, alpha, kF, kM, gen, vgen, ctx):
    """NAT-K*-L0nat: the locked NAT-K* entry, except that layer 0's key row (stacked row 0) keeps the measured entry kappa^M bitwise.
    Every other arm: the locked stored_entry_v3, unchanged."""
    if arm != X9_ARM:
        return _LOCKED["stored_entry_v3"](arm, alpha, kF, kM, gen, vgen, ctx)
    out, info = _LOCKED["stored_entry_v3"]("NAT-K*", alpha, kF, kM, gen, vgen, ctx)
    if alpha == 0.0:
        return out, info
    out = out.clone()
    out[0] = kM[0]
    pre, tgt = info["pre"].clone(), info["target"].clone()
    pre[0] = kM[0].double()
    tgt[0] = (kM[0].double() - kF[0].double()).norm(dim=-1, keepdim=True)
    info["pre"], info["target"] = pre, tgt
    return out, info


# ----------------------------------------------------------------------------- per-slot energies
def project(U, x):
    """P_r x with P_r = U U^T per (layer, KV head). U [L, n_kv, hd, r], x [L, n_kv, hd]; float64."""
    return torch.einsum("lhdr,lhr->lhd", U, torch.einsum("lhdr,lhd->lhr", U, x))


def slot_energies(d, F, U):
    """d = stored - kappa^F, F = kappa^F, both [2L, n_kv, hd] float64 (even rows K, odd rows V). Returns name -> [L, n_kv] float64 (CPU)."""
    dK, dV = d[0::2], d[1::2]
    out = {"eK": (dK * dK).sum(-1), "pK": (project(U, dK) ** 2).sum(-1), "eV": (dV * dV).sum(-1), "pV": (project(U, dV) ** 2).sum(-1),
           "fK": F[0::2].norm(dim=-1), "fV": F[1::2].norm(dim=-1)}
    return {k: v.detach().cpu() for k, v in out.items()}


# ----------------------------------------------------------------------------- per-layer predictor state
class PerLayerPredictorState(_LOCKED["PredictorState"]):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.layer_terms = []                                       # one dict per evaluate() call (one per step)
        self.slots = {k: [] for k in SLOT_KEYS}                     # one [L, n_kv] tensor per record_write() call (one per step)
        INSTANCES.append(self)

    def evaluate(self, cache, queries):
        """The locked evaluate (rv3.core.PredictorState.evaluate), with the per-layer terms kept. Same operations, same order."""
        n = self.n
        if n == 0:
            self.layer_terms.append({k: [0.0] * self.n_layers for k in LAYER_KEYS})
            return {"P_norm": 0.0, "P_logit": 0.0, "P_val": 0.0, "P_anormK": 0.0, "P_out": float("nan")}
        lo, hi = self.P0, self.P0 + n
        pl = pv = pa = 0.0
        tl, tv, ta = [], [], []
        isd = 1.0 / math.sqrt(self.hd)
        for l, lay in enumerate(cache.layers):
            q = queries[l].double()
            K = lay.keys[0].double()
            p = V._attn_probs(q, K, self.hmap, self.scaling)[:, lo:hi]                  # [n_q, n]
            dk = self.dK[l, :, :n][self.hmap]                                           # [n_q, n, hd]
            dv = self.dV[l, :, :n][self.hmap]
            a = float((p * torch.einsum("gd,gjd->gj", q, dk).abs() * isd).sum())
            b = float((p * dk.norm(dim=-1)).sum())
            c = float(torch.einsum("gj,gjd->gd", p, dv).norm(dim=-1).sum())
            pl += a
            pa += b
            pv += c
            tl.append(a)
            ta.append(b)
            tv.append(c)
        denom = self.n_layers * self.n_q
        sh = [x for x in self.share[:n] if x == x]
        rms = (self.cum2 / n).sqrt()                                                    # [2L, n_kv]
        self.layer_terms.append({"PL_logit": [x / denom for x in tl], "PL_val": [x / denom for x in tv], "PL_anormK": [x / denom for x in ta],
                                 "PL_norm": [float(x) for x in rms.reshape(self.n_layers, 2 * self.n_kv).mean(-1)]})
        return {"P_norm": float(rms.mean()), "P_logit": pl / denom, "P_val": pv / denom, "P_anormK": pa / denom,
                "P_out": (sum(sh) / len(sh)) if sh else float("nan")}

    def record_write(self, stored, kF):
        super().record_write(stored, kF)
        if SUBSPACE["U"] is None:
            raise RuntimeError("rv3x9.install(U) must be called before an X9 run")
        F = kF.double()
        e = slot_energies(stored.double() - F, F, SUBSPACE["U"])
        for k in SLOT_KEYS:
            self.slots[k].append(e[k])


# ----------------------------------------------------------------------------- install / uninstall / registry
def install(U):
    """Process-local: rv3.core uses PerLayerPredictorState and stored_entry_x9; NAT-K*-L0nat needs a context. U: [L, n_kv, hd, r] (frozen U8)."""
    SUBSPACE["U"] = U.double()
    V.PredictorState = PerLayerPredictorState
    V.stored_entry_v3 = stored_entry_x9
    V.NEEDS_CTX = set(_LOCKED["NEEDS_CTX"]) | {X9_ARM}


def uninstall():
    SUBSPACE["U"] = None
    V.PredictorState = _LOCKED["PredictorState"]
    V.stored_entry_v3 = _LOCKED["stored_entry_v3"]
    V.NEEDS_CTX = _LOCKED["NEEDS_CTX"]


def clear():
    INSTANCES.clear()


def layer_fields(inst) -> dict:
    """Record fields: name -> [S][L] list of per-layer terms."""
    return {k: [t[k] for t in inst.layer_terms] for k in LAYER_KEYS}


def slot_arrays(inst) -> dict:
    """npz arrays: name -> [S, L, n_kv] float64."""
    return {k: torch.stack(inst.slots[k]).numpy() for k in SLOT_KEYS}
