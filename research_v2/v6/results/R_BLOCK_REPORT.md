# v6 R block: results (pre-specified secondary analyses of previously seen data)

**Run.** `v6/analyze_r_v6.py` was run once, on 2026-11-14, from commit df2d226. The analysis plan, script and synthetic test were committed before the run.

**Status of the data.** The authors had already seen these records' aggregate locked results. These analyses are therefore not blind and not confirmatory.

**Criteria.** As in THEORY_PLAN_V6 §7 (commit 720760a), applied as in R_ANALYSIS_PLAN_V6.md.

**Units.** 10 units of n = 100 problems each: 8 from v4 and U1, U2 from v5. No problem was dropped for having fewer than 4 steps. The full output is `R_block_report.json`.

**Checks.**
- **Self-check:** the locked v4 L^V (H5, value path) was reproduced exactly for all 8 v4 units, from 2.2951 (Qwen2.5-1.5B) to 2.5854 (Mistral-7B).
- **RoPE provenance:**
  - The Llama-3.2-3B and Llama-3.1-8B configs used were verified byte for byte against the pinned v4 git blobs (`v6/provenance/LLAMA_CONFIG_VERIFICATION.md`), so their RoPE fields are those of the official configs.
  - Dropping llama3 scaling would change Ā_p by at most 8.4 × 10⁻⁴.

## Verdicts

| Test | Verdict | Basis |
|---|---|---|
| R1: the common-mode part of H2 builds over steps | **neither** | 6 of 10 units positive with CI excluding 0 and Holm p < 0.05; agreement needed 8. No unit had a CI entirely below 0, and 8 of 10 point estimates were positive, so not a disagreement. |
| R2: step-2 V contrast (alignment only; descriptive) | (no verdict) | Positive in all 10 units, CIs excluding 0 |
| R3: more common mode, larger V excess (problem level) | **agreement** (weak) | Pooled Spearman ρ = 0.122, 95% CI [0.057, 0.185], Holm p = 0.0021. 7 of 10 units have ρ > 0, which is exactly the ≥ 2/3 threshold. |
| R4: RoPE attenuates the key common mode | **agreement** | ρ_K from 0.898 to 0.995 in all 10 units (> 0.5); ρ_K > placebo ρ_V in all 10 (placebo −0.330 to +0.178); Holm p = 0.0021 in every unit |

**L-block gate (author rule, operationalized in R_ANALYSIS_PLAN_V6):** no disagreement on R1 or R3, so the theory is *not clearly contradicted*.

## Per unit

| Unit | R1 slope [95% CI], Holm p | R2 r₂ [95% CI] | R3 ρ | R4 ρ_K / ρ_V placebo | K share in high-frequency pairs (Ā < 0.05) / V share in the same dims |
|---|---|---|---|---|---|
| Qwen2.5-0.5B | +0.181 [+0.115, +0.249], 0.002 | +0.474 [+0.277, +0.680] | +0.082 | 0.965 / −0.330 | 0.006 / 0.193 (6 of 32 pairs) |
| Qwen2.5-1.5B | +0.078 [+0.016, +0.142], 0.073 | +0.440 [+0.189, +0.689] | +0.396 | 0.977 / −0.221 | 0.007 / 0.187 (12 of 64) |
| Qwen2.5-3B | −0.036 [−0.094, +0.021], 0.66 | +0.885 [+0.585, +1.183] | +0.194 | 0.990 / +0.025 | 0.006 / 0.186 (12 of 64) |
| Qwen2.5-7B | −0.004 [−0.054, +0.048], 1.00 | +0.749 [+0.502, +0.999] | −0.127 | 0.989 / −0.223 | 0.007 / 0.201 (12 of 64) |
| Llama-3.2-1B | +0.061 [+0.016, +0.106], 0.050 | +0.242 [+0.084, +0.404] | +0.105 | 0.898 / −0.163 | 0.007 / 0.173 (5 of 32) |
| Llama-3.2-3B | +0.069 [+0.018, +0.121], 0.050 | +0.542 [+0.373, +0.710] | +0.257 | 0.935 / −0.092 | 0.010 / 0.174 (11 of 64) |
| Llama-3.1-8B | +0.009 [−0.036, +0.054], 1.00 | +0.716 [+0.541, +0.895] | −0.086 | 0.960 / −0.038 | 0.011 / 0.189 (12 of 64) |
| Mistral-7B-v0.3 | +0.063 [+0.022, +0.102], 0.020 | +0.637 [+0.439, +0.834] | +0.161 | 0.995 / −0.107 | 0.008 / 0.188 (12 of 64) |
| Qwen3-0.6B-Base (U1) | +0.105 [+0.039, +0.170], 0.020 | +0.355 [+0.144, +0.569] | −0.030 | 0.940 / +0.178 | 0.003 / 0.187 (12 of 64) |
| SmolLM2-1.7B (U2) | +0.198 [+0.144, +0.253], 0.002 | +0.377 [+0.163, +0.590] | +0.233 | 0.971 / +0.048 | 0.012 / 0.219 (7 of 32) |

Notes:
- Holm-adjusted p is 0.0498 for Llama-3.2-1B and Llama-3.2-3B; the table rounds it to 0.050. Both are below 0.05, so both count as positive-significant.
- In R1, the steps 2–8 / 9–64 windows of the locked H2 sensitivity analysis had been seen before this run.

## Reading (descriptive; no new claim is made here)

1. **R4: strong and uniform.**
   - In every unit, rotary pairs that RoPE should wash out over the decode positions (Ā_p < 0.05) carry about 1% of the key common-mode energy (0.3–1.2%).
   - The same dimensions carry 17–22% of the value common-mode energy, and values are not rotated.
   - This is what RoPE attenuation of a pre-RoPE shared component predicts. It remains confounded with any frequency dependence of the key error itself (THEORY_PLAN_V6 §7); L5 is the decisive test.
2. **R2 is large in every unit.** With a single perturbed entry (step 2) the natural value error is already harsher than isotropic error of the same norm. The common-mode term cannot contribute there (N_D = 1). So a substantial part of H2 is a **direction/alignment effect** that the first-order common-mode model does not explain.
   - This agrees with the pilot's centering result, where removing the common mode closed only 0.27–0.34 of the gap.
   - It suggests that L3's pre-stated band (the common mode explains 0.2–0.8 of θ_V) could fail at the lower end.
   - r₂ is a single-step log ratio, not θ, so the two are not directly comparable.
3. **R1 is split by size.**
   - The V excess grows over steps in the smaller models, Mistral-7B and both v5 units.
   - It does not grow in Qwen2.5-3B, Qwen2.5-7B or Llama-3.1-8B (point estimates −0.036 to +0.009), where R2 is among the largest.
   - Descriptively, the share of H2 that builds up over steps is smaller in the larger models, and the step-2 (alignment) share is larger.
4. **R3 is weak.** The pooled problem-level association is positive but small (ρ ≈ 0.12) and heterogeneous: Qwen2.5-1.5B at 0.40, Qwen2.5-7B at −0.13.

## Gate for the L block

**The rule:** a disagreement on R1 or R3 would block the L block.

**The result:** R1 is *neither* and R3 is *agreement* (weak), so the rule does not block the L block.

**Before PROTOCOL_V6 is finalised:**
- R2 and the size split in R1 point to a large alignment component. That bears on how L3/L4 are framed.
- This is an author decision, and nothing about the L block has been changed.
