# V3-D04 diagnosis: qwen-base G2/G3 CV norm tolerance (CPU only, no GPU)

2026-10-29. Script `v3/phase4/diag_norm_tolerance_cpu.py`. It re-runs the gate numerics loop on CPU: the 6 gate problems,
α = 1, the synthetic context, label qwen-base, tier V3gate, the same seeds, real Qwen2.5-0.5B. It wraps `stored_entry_v3`
without changing it. Outputs: `diag_norm_tolerance_cpu_CV.json` and `diag_norm_tolerance_cpu_others.json`. The E1/E1L GPU
figures were read (read-only) from `e1_runs/` and `e1l_runs/`.

## Limits of the evidence
- The GPU gate saved only the aggregate (CV: mean 0.0016, max 0.101), so the GPU slot that failed cannot be identified
  without GPU use.
- CPU and CUDA kernels differ, so the CPU run reproduces the procedure, not the GPU values bit for bit. CV did not exceed
  0.10 on CPU.

## Worst CV slot on CPU (same procedure)
- Location: problem 578, decode step index 26, layer 0, K, KV head 0.
- Norms: target 0.1789; realized 0.2119; entry ‖κ^F‖ 250.87; E1 floor 2⁻¹⁰‖κ^F‖ 0.2450. The target is 0.73 × the floor.
- Current rule |r − t| / (t + floor): 0.0778. "Max" rule |r − t| / max(t, floor): 0.1345.
- Construction error before FP16 rounding: 0 at float32 precision. The largest over all CV slots is 6.4e-6.
- FP16 rounding norm ‖stored − unrounded‖: 0.0774. The rounding accounts for the whole deviation.

## All arms on CPU (6 problems × up to 64 steps; worst slot under the current rule)

| Arm | Current rule | Max rule | Construction error (max) | Worst slot |
|---|---|---|---|---|
| CV | 0.078 | 0.135 | 6.4e-6 | L0 K h0 |
| C1 | 0.079 | 0.138 | 6.6e-6 | L0 K h0 |
| QO | 0.075 | 0.140 | 8.4e-6 | L0 K h0 |
| C1-K | 0.072 | 0.144 | 6.1e-6 | L0 K h0 |
| QA | 0.047 | 0.078 | 6.2e-6 | L0 K h0 |
| NAT-K\* | 0.0040 | 0.0041 | 7e-7 | L0 K h1 |
| C2 | 0.0032 | 0.0040 | 0 | L4 K h1 |
| NAT-V\* | 0.0032 | 0.0033 | 4e-7 | L2 V h1 |
| C1-V | 0.0018 | 0.0019 | 2e-7 | L1 V h1 |

Every per-step mean is ≤ 0.0023, against the 0.01 limit.

E1 qwen-base GPU records under the identical rule, per-problem worst slot (200 problems):
- C1 α = 1: median 0.049, max 0.089;
- C1 α = 0.5: max 0.097;
- C1\*: max 0.099.

E1L (Llama): all ≤ 0.013.

## Findings
1. **No CV construction defect.** Before rounding, every arm hits its target to ≤ 1e-5 relative. CV behaves like C1,
   which is expected: in the gate σ² ≡ 1, so the CV draw is isotropic.
2. **The deviation is FP16 rounding** in Qwen layer-0 K head 0. That head's entries have norm ≈ 250, so its
   isotropic-arm targets (≈ 0.15–0.25) lie at or below 2⁻¹⁰‖κ^F‖. Rounding a 250-norm entry moves ‖stored − κ^F‖ by
   about 0.07, i.e. about 0.3 × the floor. This affects every isotropic arm (CV, C1, QO, C1-K, QA), not CV alone. It sat
   just under 0.10 throughout E1, and 0.101 in the v3 GPU gate is a draw from the same distribution.
3. **v3 did not omit E1's floor rule.** `rv3/core.py:416` uses `den = target + 2⁻¹⁰‖κ^F‖`, identical to the E1 code
   (`rv2/controls.py:116`). The E1 code comment says "max(target, fp16 resolution)", but the E1 code, and so the E1
   results, use the sum. The sum is never smaller than the max, so the "max" form is *stricter*: on CPU it would fail
   CV, C1, QO and C1-K (0.13–0.14).

## Consequence for the pre-declared rule
- (a) requires that the slot "would pass under E1's rule". E1's rule as coded is exactly the rule that failed. The
  "max" form fails by more. Applying "E1's floor uniformly to all arms" is already the case, so an amendment would
  change nothing. (a) therefore cannot be carried out as written.
- (b) does not apply, because there is no construction defect.
- (c) is respected: nothing was changed.

The authors need to decide.
