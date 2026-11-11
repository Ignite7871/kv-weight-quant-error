"""E1 perturbation-matched cache controls (E1_SPEC.md). Nothing here changes the verified engine / Stage 1 code; it imports it.

Setting. At decode step s the NF4-weight pass and the FP16-weight pass run from the SAME token x_{s-1} and cache state S_{s-1}
(exactly the two-pass machinery of rv2.stage1: the unkept pass first, crop, the kept pass second). In every control arm the KEPT pass is
FP16 and the token (or, in teacher forcing, the logits compared) comes from that FP16 pass; the NF4 pass is used ONLY to measure
    D = kappa^N - kappa^F        (the NF4-induced difference of the NEW entry, per layer, K/V, KV head)
on the current state. The entry then stored for position L is
    NAT   kappa^N                                   (bitwise the engine's cache-only entry; no arithmetic)
    C1    kappa^F + alpha * ||D|| * u,  u uniform on the unit sphere of R^head_dim        (magnitude kept, direction random)
    C2    kappa^F + alpha * (eps * D),  eps_i i.i.d. +-1                                  (per-coordinate magnitudes kept, signs random)
so only the stored VALUES differ between arms. Perturbations are applied ONLY to the newly written entry at write time; entries already in the
cache are never revisited (they carry the perturbation of their own write step, which is exactly what NAT's entries carry). The prompt prefill is
FP16 and unperturbed in every arm. alpha = 0 must reproduce pure FP16 bitwise.

Slot = (layer, K or V, KV head) at the new position; norms are L2 over head_dim. Randomness: one torch.Generator per (arm, alpha, problem, tier)
seeded from SHA-256 of those labels and the master seed (no data, outcome or model output enters a seed); draws are made once per step for all
slots together in a fixed order (layer-major, K then V, then head, then coordinate).
"""

from __future__ import annotations

import hashlib

import torch

from . import PROJECT_ROOT  # noqa: F401
from .engine import Backend, GenResult, _snapshot_last

ARMS = ("NAT", "C1", "C2")
MASTER_SEED = "rv2-E1-controls-2026-10-27"


# ----------------------------------------------------------------------------- helpers (module level so mutation checks can replace them)
def _slot_norm(x):
    """L2 norm over head_dim, kept as a trailing singleton dimension."""
    return x.float().norm(dim=-1, keepdim=True)


def _unit_gaussian(shape, gen, device):
    u = torch.randn(shape, generator=gen, device=device, dtype=torch.float32)
    return u / u.norm(dim=-1, keepdim=True)


def _rademacher(shape, gen, device):
    return torch.randint(0, 2, shape, generator=gen, device=device).float() * 2.0 - 1.0


def _stored_entry(arm: str, alpha: float, kF, kN, gen):
    """The tensor to store for the new position, given the FP16-pass entry kF and the NF4-pass entry kN (both [S, n_kv, head_dim], S = 2*layers)."""
    if arm == "NAT":
        return kN.clone()
    F = kF.float()
    D = kN.float() - F
    if arm == "C1":
        delta = alpha * _slot_norm(D) * _unit_gaussian(D.shape, gen, D.device)
    elif arm == "C2":
        delta = alpha * D * _rademacher(D.shape, gen, D.device)
    else:
        raise ValueError(arm)
    return (F + delta).to(kF.dtype)


def js64(logits_p, logits_q):
    """Jensen-Shannon divergence (nats) between softmax(logits_p) and softmax(logits_q), computed in float64 so that identical logits give exactly 0
    (the float32 historical function has a +-3e-8 noise floor on 150k-token vocabularies). Inputs [1, V]; returns a 0-d tensor."""
    p = torch.softmax(logits_p.double(), dim=-1)
    q = torch.softmax(logits_q.double(), dim=-1)
    m = 0.5 * (p + q)
    js = 0.5 * (torch.special.xlogy(p, p) - torch.special.xlogy(p, m)).sum(-1) + 0.5 * (torch.special.xlogy(q, q) - torch.special.xlogy(q, m)).sum(-1)
    return js.reshape(())


def seed_for(arm: str, alpha: float, index: int, tier: str, master: str = MASTER_SEED, repeat: int = 0) -> int:
    h = hashlib.sha256(f"{master}|{arm}|{alpha!r}|{index}|{tier}|{repeat}".encode()).digest()
    return int.from_bytes(h[:8], "big") >> 1


def make_generator(device, seed: int):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g


def _stack_slices(slices):
    """[(k, v) per layer], each [1, n_kv, 1, head_dim]  ->  [2*layers, n_kv, head_dim] in the order layer0-K, layer0-V, layer1-K, ..."""
    return torch.stack([t[0, :, 0, :] for kv in slices for t in kv])


def _write_back(cache, L, stacked):
    for l, lay in enumerate(cache.layers):
        lay.keys[:, :, L:L + 1, :] = stacked[2 * l].unsqueeze(0).unsqueeze(2)
        lay.values[:, :, L:L + 1, :] = stacked[2 * l + 1].unsqueeze(0).unsqueeze(2)


# ----------------------------------------------------------------------------- one controlled step
@torch.no_grad()
def controlled_step(backend: Backend, cur, cache, arm: str, alpha: float, gen, diag: bool = True):
    """Run the NF4 pass, crop, run the FP16 pass, then overwrite the newest entry per `arm`. Returns (FP16 logits, diag dict of 0-d/1-d tensors).
    The cache afterwards has length L+1 and differs from the pure-FP16 cache only in the newest entry."""
    L = cache.get_seq_length()
    backend.run(cur, cache, True)                                   # 1. NF4 pass (appends kappa^N at L)
    kN = _stack_slices(_snapshot_last(cache, L))
    cache.crop(L)                                                   # 2. restore the pre-step cache (a view; earlier entries untouched)
    lg_F, _ = backend.run(cur, cache, False)                        # 3. FP16 pass (appends kappa^F at L); its logits are the arm's logits
    assert cache.get_seq_length() == L + 1
    kF = _stack_slices(_snapshot_last(cache, L))
    new = _stored_entry(arm, alpha, kF, kN, gen)                    # 4. what is stored
    _write_back(cache, L, new)
    d = {}
    if diag:
        D = kN.float() - kF.float()
        tn = _slot_norm(D)                                          # target per-slot norm
        rn = _slot_norm(new.float() - kF.float())                   # realized per-slot norm of what was added to kappa^F
        # error of the realized norm relative to max(target, fp16 resolution of the entry): a stored FP16 entry cannot represent the added vector more
        # precisely than about 2**-10 of the entry's own norm, so slots whose target is below that scale are judged against that scale.
        den = alpha * tn + (2.0 ** -10) * _slot_norm(kF)
        rel_err = (rn - alpha * tn).abs() / den
        d = {"norm_dev_max": rel_err.max(), "norm_dev_mean": rel_err.mean(),
             "rel_scale_mean": (tn / _slot_norm(kF).clamp_min(1e-12)).mean(), "target_norm_mean": tn.mean()}
    return lg_F, d


# ----------------------------------------------------------------------------- free-running generation (Tier B)
@torch.no_grad()
def generate_control(backend: Backend, prompt_ids, arm: str, alpha: float, seed: int, max_new_tokens: int, eos_token_id,
                     early_stop_fn=None, diag: bool = True) -> GenResult:
    """Greedy generation: tokens are argmax of the FP16-pass logits, FP16 prefill, stored entries per `arm` (NAT/C1/C2)."""
    gen = make_generator(prompt_ids.device, seed)
    logits, cache = backend.run(prompt_ids, None, use_4bit=False)
    ids = [int(torch.argmax(logits, dim=-1).item())]
    stopped = False
    dg = {"norm_dev_max": [], "norm_dev_mean": [], "rel_scale_mean": [], "target_norm_mean": []}
    if ids[0] != eos_token_id:
        for s in range(1, max_new_tokens):
            cur = torch.tensor([[ids[-1]]], device=prompt_ids.device)
            lg, d = controlled_step(backend, cur, cache, arm, alpha, gen, diag)
            top = int(torch.argmax(lg, dim=-1).item())
            for k in dg:
                if k in d:
                    dg[k].append(d[k])
            ids.append(top)
            if eos_token_id is not None and top == eos_token_id:
                break
            if early_stop_fn is not None and early_stop_fn(ids):
                stopped = True
                break
    if diag and dg["norm_dev_max"]:
        dg = {k: torch.stack(v).tolist() for k, v in dg.items()}
    return GenResult(ids, [arm] * (len(ids) - 1), [], {"diag": dg}, {}, {}, stopped, {"arm": arm, "alpha": alpha})


# ----------------------------------------------------------------------------- teacher forcing (Tier A)
@torch.no_grad()
def reference_logits(backend: Backend, prompt_ids, ref_ids, S: int):
    """Pure FP16 teacher-forced next-token logits for steps s = 1..S: step s feeds ref_ids[s-1] and predicts ref_ids[s]. Returns [S, V] float32."""
    logits, cache = backend.run(prompt_ids, None, use_4bit=False)
    out = []
    for s in range(1, S + 1):
        cur = torch.tensor([[ref_ids[s - 1]]], device=prompt_ids.device)
        lg, _ = backend.run(cur, cache, False)
        out.append(lg.float())
    return torch.cat(out, dim=0)


@torch.no_grad()
def teacher_forced_arm(backend: Backend, prompt_ids, ref_ids, S: int, arm: str, alpha: float, seed: int, ref_lg, diag: bool = True) -> dict:
    """Same token history (ref_ids) for every arm; the arm's cache holds its own stored entries. Per step: JS between the arm's FP16-pass
    next-token distribution and the pure-FP16 reference distribution, and whether their argmaxes differ."""
    gen = make_generator(prompt_ids.device, seed)
    logits, cache = backend.run(prompt_ids, None, use_4bit=False)
    js, flip, nd_max, nd_mean, rel, tn = [], [], [], [], [], []
    for s in range(1, S + 1):
        cur = torch.tensor([[ref_ids[s - 1]]], device=prompt_ids.device)
        lg, d = controlled_step(backend, cur, cache, arm, alpha, gen, diag)
        r = ref_lg[s - 1:s]
        js.append(js64(lg, r))
        flip.append((torch.argmax(lg, dim=-1) != torch.argmax(r, dim=-1)).reshape(()).float())
        if diag:
            nd_max.append(d["norm_dev_max"]); nd_mean.append(d["norm_dev_mean"]); rel.append(d["rel_scale_mean"]); tn.append(d["target_norm_mean"])
    out = {"js": torch.stack(js).tolist(), "flip": torch.stack(flip).tolist()}
    if diag:
        out.update(norm_dev_max=torch.stack(nd_max).tolist(), norm_dev_mean=torch.stack(nd_mean).tolist(), rel_scale_mean=torch.stack(rel).tolist(), target_norm_mean=torch.stack(tn).tolist())
    return out
