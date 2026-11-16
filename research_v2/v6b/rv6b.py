"""v6b arms (FOLLOWUP_V6B.md section 3): the three new value-path entries A, B, C, and the per-problem common mode mu.

The locked v6 step (rv6.controlled_step_v6 -> rv6.teacher_forced_arm_v6) is used unchanged. v6b extends only the entry construction:
install() replaces the module attribute rv6.stored_entry_v6 by `stored_entry`, which builds the three new arms and delegates every v6 arm
to the original rv6.stored_entry_v6 (same object, unchanged). No locked file is edited.

Per value slot (layer, KV head) at decode step j of the arm's own run (as in v6 NAT-V-cen):
  Dn = kM - kF            natural error written at this step in this run
  mu                      per-slot mean of the stored NAT-V error over the problem (rv4.SlotSums -> rv6.slot_mean; v6's arithmetic)
  c  = Dn - mu            the centered error, norm ||c||
  g  = unit(_gauss(...))  the SAME float32 draw C1-V makes at this step (B and C run with C1-V's seed and draw once per step, like C1-V)
  A  NAT-V-cen0 : c                    (centered, not re-normed)              norm ||c||   (construction check)
  B  C1-V+mu    : mu + ||c|| g         (isotropic position-varying part + mu) norm checked as ||stored - mu|| = ||c|| (own check)
  C  C1-V@cen   : ||c|| g              (isotropic at the centered norm)       norm ||c||   (construction check)
Keys are FP16-exact in every arm. Entries are built in float64 and stored with the locked V.to_fp16.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RV = HERE.parent
sys.path.insert(0, str(RV / "v6"))
import rv6  # noqa: E402  (imports rv4 -> sets CUBLAS_WORKSPACE_CONFIG before torch)

import torch  # noqa: E402

V, rv4 = rv6.V, rv6.rv4
NEW_ARMS = ["NAT-V-cen0", "C1-V+mu", "C1-V@cen"]
REPRO_ARMS = ["NAT-V", "C1-V", "NAT-V-cen"]
ARMS = REPRO_ARMS + NEW_ARMS                       # run order within a problem (FOLLOWUP_V6B s3)
SEED_ARM = {"NAT-V-cen0": "NAT-V-cen0", "C1-V+mu": "C1-V", "C1-V@cen": "C1-V"}   # B and C replay C1-V's stream
NEEDS_MU = {"NAT-V-cen", "NAT-V-cen0", "C1-V+mu", "C1-V@cen"}
DEGEN_TOL = rv6.DEGEN_TOL
B_TOL = V.CONSTR_TOL                               # 1e-4, the locked construction tolerance

_ORIG = rv6.stored_entry_v6                        # the locked v6 construction (kept for delegation and for uninstall)


class State:
    """Per-arm side channel, reset by the driver before each arm: SlotSums for NAT-V (gives mu), diagnostics for B."""
    sums = None
    b_ratio = []        # per step: mean over value slots of ||stored error|| / ||Dn|| (pre-rounding), B only
    b_dev = []          # per step: max relative deviation of ||(stored error) - mu|| from ||c||, B only

    @classmethod
    def reset(cls, arm):
        cls.sums = rv4.SlotSums() if arm == "NAT-V" else None
        cls.b_ratio, cls.b_dev = [], []


def new_entry(arm, kF, kM, gen, ctx):
    """(stored FP16 entry, info) for A, B or C; info has the keys controlled_step_v6 reads."""
    rows, other = rv6._rows("V")
    dev, two_l = kF.device, kF.shape[0]
    F = kF.double()
    Dn = (kM.double() - F)[rows]
    nD = Dn.norm(dim=-1, keepdim=True)
    mu = ctx["mu"].double()
    c = Dn - mu
    nc = c.norm(dim=-1, keepdim=True)
    degen = int(((nc < DEGEN_TOL * nD) & (nD > 0)).sum())
    exact = torch.zeros(two_l, dtype=torch.bool, device=dev)
    exact[other] = True
    target = torch.zeros(F.shape[:-1] + (1,), dtype=torch.float64, device=dev)
    if arm == "NAT-V-cen0":
        d = c
        target[rows] = nc
    elif arm in ("C1-V+mu", "C1-V@cen"):
        g = V._unit_safe(V._gauss(Dn.shape, gen, dev).double())          # the C1-V draw of this step
        iso = nc * g
        if arm == "C1-V@cen":
            d = iso
            target[rows] = nc
        else:
            d = mu + iso
            target = None
    else:
        raise ValueError(arm)
    delta = torch.zeros_like(F)
    delta[rows] = d
    u = F + delta
    u[exact] = F[exact]
    if arm == "C1-V+mu":
        e = (u - F)[rows]
        pos = nc.squeeze(-1) > 0
        dev_ = ((e - mu).norm(dim=-1) - nc.squeeze(-1)).abs() / nc.squeeze(-1).clamp_min(1e-300)
        State.b_dev.append(float(dev_[pos].max()) if bool(pos.any()) else 0.0)
        posD = nD.squeeze(-1) > 0
        State.b_ratio.append(float((e.norm(dim=-1)[posD] / nD.squeeze(-1)[posD]).mean()) if bool(posD.any()) else 1.0)
    return V.to_fp16(u), {"pre": u, "exact_rows": exact, "target": target, "kind": "v6b", "degenerate": degen}


def stored_entry(arm, kF, kM, gen, vgen, ctx):
    if arm in NEW_ARMS:
        out, info = new_entry(arm, kF, kM, gen, ctx)
    else:
        out, info = _ORIG(arm, kF, kM, gen, vgen, ctx)
    if State.sums is not None:
        State.sums.record(out, kF)                 # v6 Log6.write_time records exactly this (stored FP16 entry, kappa^F)
    return out, info


def install():
    rv6.stored_entry_v6 = stored_entry


def uninstall():
    rv6.stored_entry_v6 = _ORIG


def mu_from_state():
    """mu of the NAT-V run just finished: rv6.slot_mean on the SlotSums arrays (identical to v6's arrays -> slot_mean path)."""
    return rv6.slot_mean(State.sums.arrays(), "V")
