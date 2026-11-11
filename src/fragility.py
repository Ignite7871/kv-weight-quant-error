"""Offline empirical quantization fragility (EQF) metric -- Phase 0 spike.

This is deliberately kept separate from routing.py: calculate_entropy /
calculate_margin there are DEPLOYED, zero-extra-cost routing signals (they
only ever look at one precision's logits). js_divergence here requires
BOTH the FP16 and 4-bit logits for the same step, which only exists in an
offline analysis harness (build_eqf_spike.py) -- it is never a candidate
routing signal itself, only the ground-truth quantity used to ask whether
entropy is a good proxy for it.
"""

import torch

EPS = 1e-12


def js_divergence(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    """Jensen-Shannon divergence (in nats) between the softmax distributions
    of two logit tensors of identical shape, reduced over the last (vocab)
    dimension.

    Bounded in [0, ln(2)] and symmetric in (p, q), unlike KL divergence,
    which is unbounded whenever q assigns near-zero mass to a token p
    favors -- a real occurrence at the vocab scale here, not a theoretical
    edge case. That boundedness is what keeps a handful of outlier tokens
    from dominating downstream aggregates (correlations, mean-JS-by-group)
    the way unbounded KL tails would.

    Shape contract: same as routing.calculate_entropy -- [batch, vocab] or
    [batch, seq_len, vocab] in, vocab dimension reduced away on the way out.

    Upcasts to float32 internally: logits here are FP16 (this project's
    compute dtype throughout), and EPS=1e-12 is below FP16's representable
    range (min positive FP16 ~6e-5 normal / ~6e-8 subnormal) -- clamping to
    it in FP16 is a no-op, so log(0) -> -inf and 0 * -inf -> NaN propagates
    through every row. float32 has ample headroom for this EPS.
    """
    logits_p = logits_p.float()
    logits_q = logits_q.float()
    log_p = torch.log_softmax(logits_p, dim=-1)
    log_q = torch.log_softmax(logits_q, dim=-1)
    p = log_p.exp()
    q = log_q.exp()
    m = 0.5 * (p + q)
    log_m = torch.log(m.clamp_min(EPS))
    kl_pm = (p * (log_p - log_m)).sum(dim=-1)
    kl_qm = (q * (log_q - log_m)).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm)
