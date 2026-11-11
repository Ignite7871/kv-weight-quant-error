"""Entropy-based routing signal for dynamic precision switching (Phase 2).

This module is a DESIGN SCAFFOLD ONLY -- signature and docstring describe
the intended behavior; no tensor logic is implemented yet.
"""

import torch


def calculate_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Compute the Shannon entropy of the softmax distribution over the vocabulary.

    This is the routing signal for Phase 2: the entropy of the PREVIOUS
    token's output logits determines whether the NEXT token is generated
    through FP16 or 4-bit weights (see DynamicQwen.generate in
    dynamic_model.py). Low entropy (confident prediction) routes to 4-bit;
    high entropy (uncertain prediction) routes to FP16.

    Shape contract:
      - Input `logits`: either [batch, vocab] (a single decode step's
        logits for each sequence in the batch) or [batch, seq_len, vocab]
        (logits for every position, e.g. during prefill/prompt scoring).
      - Output: same shape as input with the vocab dimension reduced away,
        i.e. [batch] or [batch, seq_len] respectively. Each output value
        is the entropy (in nats, natural log base) of that position's
        softmax distribution.

    Performance requirement: this runs on EVERY decode step in the Phase 2
    generation loop, so the implementation must be a single vectorized
    softmax + reduction over the last (vocab) dimension -- no Python loops
    over the batch or sequence dimensions. It must not force a host/device
    sync (no .item(), no Python-level branching on the result inside this
    function); the caller decides the routing branch afterward.

    Numerical stability: implement via log_softmax rather than
    softmax().log(), i.e. H(x) = -sum_v p(v) * log_p(v) where log_p comes
    directly from F.log_softmax(logits, dim=-1) and p = log_p.exp() --
    this avoids log(0) for near-zero-probability vocab entries.

    """
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


def calculate_margin(logits: torch.Tensor) -> torch.Tensor:
    """Compute the top-1/top-2 softmax probability margin (Ablation 1: an
    alternative routing signal to calculate_entropy).

    margin = P(top-1 token) - P(top-2 token). This directly measures
    argmax fragility: a small margin means the model's top choice barely
    beats the runner-up, so quantization noise could plausibly flip the
    argmax; a large margin means the top choice dominates and is unlikely
    to flip.

    POLARITY IS INVERTED relative to calculate_entropy -- do not reuse
    entropy's comparison direction. Low margin (fragile, top choice barely
    wins) should route to FP16; high margin (robust) should route to
    4-bit. Concretely, the routing condition is `margin >= threshold` for
    4-bit (vs. entropy's `entropy <= threshold` for 4-bit) -- the
    inequality direction flips because high entropy and low margin both
    mean "fragile," but they sit at opposite ends of their respective
    scales.

    Shape contract: identical to calculate_entropy -- input [batch, vocab]
    or [batch, seq_len, vocab], output has the vocab dimension reduced
    away ([batch] or [batch, seq_len]). Values are unitless, in [0, 1).

    Performance / numerical-stability requirements: identical to
    calculate_entropy -- single vectorized log_softmax-based computation,
    no Python loops, no host/device sync (torch.topk stays on-device).
    """
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    top2_probs, _ = probs.topk(2, dim=-1)
    return top2_probs[..., 0] - top2_probs[..., 1]
