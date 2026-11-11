"""Routing policies with explicit information access (PILOT_SPEC.md section 1).

Every policy returns `wants_fp16: bool` for the current routed decode step. A policy only ever sees a
StepView exposing the fields it declared in `access`; reading any other field raises
InformationAccessError. The engine (engine.py) additionally never COMPUTES the current-step FP16
output before a non-oracle policy has decided, so a deployable policy cannot see it even by
mistake. Deployable policies may not declare `fp16_logits`.

Field meanings (all for routed decode step s >= 1, which consumes token s-1 and produces token s):
  prev_logits     logits that produced token s-1 (output of step s-1's CHOSEN branch, or the FP16
                  prefill for s = 1). Available before step s runs, at zero extra cost.
  input_token_id  token s-1 (known causally).
  step            s.
  rng             per-problem random.Random (random policies only).
  nf4_logits      step s's output computed with NF4 weights from the current cache (cascade, oracles).
  fp16_logits     step s's output computed with FP16 weights from the current cache (ORACLES ONLY).
"""

from __future__ import annotations

import random

import torch

from . import PROJECT_ROOT  # noqa: F401
from fragility import js_divergence      # historical, unmodified (float32 internally)
from routing import calculate_entropy, calculate_margin  # historical, unmodified

ALL_FIELDS = frozenset({"prev_logits", "input_token_id", "step", "rng", "nf4_logits", "fp16_logits"})
DEPLOYABLE_FIELDS = ALL_FIELDS - {"fp16_logits"}


class InformationAccessError(RuntimeError):
    pass


class StepView:
    """Read-only view exposing only the fields a policy declared."""

    __slots__ = ("_allowed", "_data")

    def __init__(self, allowed: frozenset, **data):
        object.__setattr__(self, "_allowed", allowed)
        object.__setattr__(self, "_data", {k: v for k, v in data.items() if k in allowed})

    def __getattr__(self, name):
        allowed = object.__getattribute__(self, "_allowed")
        if name not in allowed:
            raise InformationAccessError(f"field {name!r} is not in this policy's declared access {sorted(allowed)}")
        data = object.__getattribute__(self, "_data")
        if name not in data:
            raise InformationAccessError(f"field {name!r} was declared but not provided by the engine")
        return data[name]

    def __setattr__(self, name, value):
        raise AttributeError("StepView is read-only")


class Policy:
    name: str = "policy"
    kind: str = "pre"            # "pre": decide before computing the step; "cascade"; "oracle"
    deployable: bool = True
    access: frozenset = frozenset()

    def __init__(self):
        assert self.kind in ("pre", "cascade", "oracle")
        assert self.access <= ALL_FIELDS
        if self.deployable:
            assert self.access <= DEPLOYABLE_FIELDS, f"{self.name}: deployable policy declares fp16_logits"
            assert self.kind != "oracle", f"{self.name}: oracle kind cannot be deployable"
        if self.kind == "pre":
            assert not ({"nf4_logits", "fp16_logits"} & self.access), "pre-step policies cannot see current outputs"
        if self.kind == "cascade":
            assert "fp16_logits" not in self.access

    def decide(self, view: StepView) -> bool:  # True -> FP16 for this step
        raise NotImplementedError

    def describe(self) -> dict:
        return {"name": self.name, "kind": self.kind, "deployable": self.deployable, "access": sorted(self.access)}


# ---------------------------------------------------------------- deployable, decide-before-step
class StaticFP16(Policy):
    name = "fp16"

    def decide(self, view):
        return True


class StaticNF4(Policy):
    name = "nf4"

    def decide(self, view):
        return False


class LagEntropy(Policy):
    """Historical router: 4-bit iff H(prev_logits) <= tau, i.e. FP16 iff H > tau.
    Entropy is computed by the historical calculate_entropy on the logits tensor exactly as returned
    by the model (FP16 on GPU), so decisions are bit-identical to DynamicQwen.generate."""
    access = frozenset({"prev_logits"})

    def __init__(self, tau: float, name: str | None = None):
        self.tau = float(tau)
        self.name = name or f"lag_entropy@{tau:.6g}"
        super().__init__()

    def decide(self, view):
        return bool((calculate_entropy(view.prev_logits) > self.tau).item())

    def describe(self):
        return {**super().describe(), "tau": self.tau}


class LagMargin(Policy):
    """Historical margin router: 4-bit iff margin(prev) >= tau, i.e. FP16 iff margin < tau."""
    access = frozenset({"prev_logits"})

    def __init__(self, tau: float, name: str | None = None):
        self.tau = float(tau)
        self.name = name or f"lag_margin@{tau:.6g}"
        super().__init__()

    def decide(self, view):
        return bool((calculate_margin(view.prev_logits) < self.tau).item())

    def describe(self):
        return {**super().describe(), "tau": self.tau}


class RandomBernoulli(Policy):
    """FP16 with probability p, one draw per routed step, independent of everything else.
    Same draw convention as the historical control: use_4bit = not (rng.random() < p)."""
    access = frozenset({"rng"})

    def __init__(self, p_fp16: float, seed_label: str, name: str | None = None):
        assert 0.0 <= p_fp16 <= 1.0
        self.p = float(p_fp16)
        self.seed_label = seed_label
        self.name = name or f"random@{p_fp16:.4f}#{seed_label}"
        super().__init__()

    def make_rng(self, problem_index: int) -> random.Random:
        # Order-independent and resumable: the stream depends only on (seed_label, problem index).
        # random.Random(str) seeds deterministically (SHA-512 of the string) across runs/platforms.
        return random.Random(f"rv2|{self.seed_label}|{problem_index}")

    def decide(self, view):
        return view.rng.random() < self.p

    def describe(self):
        return {**super().describe(), "p_fp16": self.p, "seed_label": self.seed_label}


# ---------------------------------------------------------------- deployable cascade (4-bit first)
class CascadeNF4Margin(Policy):
    """Compute step s in NF4; if margin(NF4 output) < tau, discard it (cache cropped) and recompute
    step s in FP16, keeping the FP16 output and FP16 KV entries. Uses only information that exists
    at decision time (the NF4 pass it has already paid for). Costs one extra pass per escalation."""
    kind = "cascade"
    access = frozenset({"nf4_logits"})

    def __init__(self, tau: float, name: str | None = None):
        self.tau = float(tau)
        self.name = name or f"cascade_nf4_margin@{tau:.6g}"
        super().__init__()

    def decide(self, view):
        return bool((calculate_margin(view.nf4_logits) < self.tau).item())

    def describe(self):
        return {**super().describe(), "tau": self.tau}


class CascadeNF4Entropy(Policy):
    kind = "cascade"
    access = frozenset({"nf4_logits"})

    def __init__(self, tau: float, name: str | None = None):
        self.tau = float(tau)
        self.name = name or f"cascade_nf4_entropy@{tau:.6g}"
        super().__init__()

    def decide(self, view):
        return bool((calculate_entropy(view.nf4_logits) > self.tau).item())

    def describe(self):
        return {**super().describe(), "tau": self.tau}


# ---------------------------------------------------------------- NON-DEPLOYABLE diagnostic references
class _Oracle(Policy):
    kind = "oracle"
    deployable = False
    access = frozenset({"nf4_logits", "fp16_logits"})


class OracleJS(_Oracle):
    """DIAGNOSTIC UPPER REFERENCE. FP16 iff JS(FP16 output, NF4 output) of the current step > tau.
    Needs both current-step outputs (i.e. pays for the FP16 pass it may then discard). No future
    information: decisions use step s outputs only, never tokens or outputs of later steps."""

    def __init__(self, tau: float, name: str | None = None):
        self.tau = float(tau)
        self.name = name or f"oracle_js@{tau:.6g}"
        super().__init__()

    def decide(self, view):
        return bool((js_divergence(view.fp16_logits, view.nf4_logits) > self.tau).item())

    def describe(self):
        return {**super().describe(), "tau": self.tau}


class OracleFlip(_Oracle):
    """DIAGNOSTIC UPPER REFERENCE. FP16 iff the greedy token differs between the two current-step
    outputs; the budget is whatever this rule realizes. The chosen token always equals the FP16
    argmax given the (possibly NF4-contaminated) cache."""
    name = "oracle_flip"

    def decide(self, view):
        return bool((view.fp16_logits.argmax(-1) != view.nf4_logits.argmax(-1)).any().item())


class OracleFP16Entropy(_Oracle):
    """DIAGNOSTIC (ReQAT-style signal): FP16 iff entropy of the CURRENT-step FP16 output > tau.
    ReQAT section 3.2 routes on same-step BF16 entropy; this is the analogue."""
    access = frozenset({"fp16_logits"})

    def __init__(self, tau: float, name: str | None = None):
        self.tau = float(tau)
        self.name = name or f"oracle_fp16_entropy@{tau:.6g}"
        super().__init__()

    def decide(self, view):
        return bool((calculate_entropy(view.fp16_logits) > self.tau).item())

    def describe(self):
        return {**super().describe(), "tau": self.tau}


def float32_stats(logits: torch.Tensor) -> tuple[float, float, int]:
    """Logging-only statistics (never used for decisions): entropy, margin, argmax in float32."""
    lf = logits.float()
    return calculate_entropy(lf).item(), calculate_margin(lf).item(), int(lf.argmax(-1).item())
