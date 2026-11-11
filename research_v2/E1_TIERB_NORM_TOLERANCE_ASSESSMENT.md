# Qwen E1 Tier B: the 10.5 % maximum per-slot perturbation-norm deviation (follow-up assessment)

Written 2026-10-28 from the existing raw records only (no rerun; `e1_runs/` records and the Qwen E1 lock untouched). Script: `followup/e1_tierB_norm_tolerance.py`; output `e1_runs/tierB_norm_tolerance_assessment.json`. Exploratory follow-up; it does not replace any locked result.

## What the number means
`norm_dev_max` is, per decode step, the largest over all slots (24 layers × K/V × 2 KV heads) of |realized − target norm| / (α·target + 2⁻¹⁰·‖κ^F‖): the error of the stored perturbation's norm relative to the larger of the target and the FP16 resolution of the entry. The locked Tier A integrity check stopped a run above 0.10; the locked Tier B check did not enforce it (documented deviation), so the exceedance was visible only in the diagnostics.

## Prevalence (all Tier B steps; threshold fixed in advance at 10 %)
| Arm | steps | steps > 10 % | share | problems with ≥ 1 such step | maximum |
|---|---|---|---|---|---|
| C1 (random direction) | 43,758 | **2** | **0.0046 %** | **2 of 400** (idx 173, 1024) | 10.50 % |
| C2 (sign-randomized) | 45,585 | 0 | 0 % | 0 | 3.15 % |
| NAT | – | exactly 0 by construction | – | – | 0 |
Descriptive sweep for C1 (steps above 5 / 7.5 / 10 / 12.5 %): 464 / 31 / 2 / 0 — i.e. the distribution has a thin tail ending just above the gate level; no step exceeds 12.5 %. Tier A (teacher-forced, 12,235 steps per arm) had no step above 10 % in any arm (maximum 9.9 %). Mean deviation 0.05 % (C1) and 0.008 % (C2).

## Sensitivity of the conclusions (one pre-specified rule, no alternative rules tried)
Rule: for a contrast, a problem is "affected" if any step of any arm in that contrast exceeds 10 %; affected problems are removed for that contrast.
| Contrast (restricted first-divergence token, τ = 64) | locked result (n = 400) | excluding affected | n removed |
|---|---|---|---|
| NAT − C1 | +3.54 [+1.06, +5.99], p 0.005 (this script's resampling stream; the locked report gave [+1.14, +5.95], Holm 0.026) | **+3.39 [+0.93, +5.84]**, p 0.006 | 2 |
| NAT − C2 | +0.98 [−1.59, +3.54] | identical (no affected problems) | 0 |
The two affected problems contribute NAT − C1 differences of +8 and +58 tokens; removing them lowers the estimate by 0.15 tokens (4 %) and moves the interval endpoints by ≤ 0.12. **The Tier B conclusions are insensitive to the affected observations.** (The CI endpoints differ slightly from the locked report only because the bootstrap stream here starts fresh; the locked values are unchanged and remain the reported ones.)

## Interpretation
The exceedance is a rare tail event of the FP16-resolution floor, affecting 0.005 % of C1 steps and 0.5 % of problems, with no effect on any conclusion; no rerun is warranted, and none was done. For the Llama replication the same per-step maximum is enforced as a **stop condition in both the gates and every record** (the locked Qwen integrity check is reused for Tier A; there is no Tier B).
