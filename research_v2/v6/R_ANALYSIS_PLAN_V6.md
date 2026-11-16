# R_ANALYSIS_PLAN_V6: the R block (R1–R4) on existing v4/v5 records

Status: **committed before the first run** of `v6/analyze_r_v6.py` on real records.
- **What these are:** pre-specified secondary analyses of previously seen data. The authors have seen the aggregate locked results of these records (v4 and v5 reports, manuscript). R1–R4 compute quantities that are not in those reports, but they are not blind and not confirmatory.
- **Authority:** the predictions and criteria are those of THEORY_PLAN_V6.md §7 (commit 720760a). This file fixes the operational details that §7 left open. Nothing here changes a criterion.

## Exposure statement

- Before this plan was written, the analyst (Claude Code) had opened one record, `v4 qwen2.5-0.5b test/NAT-V.jsonl` problem 172. It read the field names, the first three per-step js values and the array shapes of one `slots/NAT_*.npz`, only to learn the record layout.
- The v4 per-model H1–H3 estimates and intervals of Qwen2.5-0.5B, Llama-3.2-1B and Qwen2.5-1.5B were read from `v4_report.json` for the PROTOCOL_V6 power calculation. They are in the manuscript anyway.
- No per-step profile, problem-level correlation or per-pair common-mode profile has been computed or looked at.

## Units

10 units, NF4 / GSM8K, with NAT, C1, NAT-K, C1-K, NAT-V and C1-V test records and NAT slot sums:
- **v4:** qwen2.5-0.5b, qwen2.5-1.5b, qwen2.5-3b, qwen2.5-7b, llama3.2-1b, llama3.2-3b, llama3.1-8b, mistral-7b-v0.3 (`v4_runs/<tag>/test`).
- **v5:** U1 Qwen3-0.6B-Base and U2 SmolLM2-1.7B (`v5_runs/<U>/test`).

**Excluded, by rule:**
- v5 U0: it reruns v4 qwen2.5-0.5b on the same problems, so it is not independent;
- U3 and U5: WikiText-2;
- U4 and U6: HQQ;
- llama3.2-1b-instruct: cross-hardware check only.

**Problems:** all problems present in all four single-path arm files and in the slot files of the unit; n is reported per unit.

## Definitions (array index k of `js` is step k+1; step 1 is exactly 0 by design)

- **R1.**
  - For each problem with S ≥ 4, regress r_t = ln max(js^{NAT-V}_t, 1e-9) − ln max(js^{C1-V}_t, 1e-9) on ln t over steps t = 2…S by OLS. Problems with S < 4 are dropped and counted.
  - The unit statistic is the mean slope.
  - The 95% CI comes from a problem bootstrap and the two-sided p from a paired sign-flip test, via the locked `analyze_v3.summarize` with B = 10,000 and stream `default_rng([20261114, key(unit), key("R1")])`.
- **R2 (descriptive).** Unit mean of r_2, with its CI from the same function and stream key "R2".
- **R3.**
  - x_i^V is the locked H5 per-problem statistic: `analyze_x13.cm_share` and `null_lognc` with 200 draws in R^{head_dim}, stream `default_rng([20261105, 13, key(unit), key("V")])`. For v4 units this reproduces the locked analysis exactly.
  - θ_i^V = logm(NAT-V) − logm(C1-V), using the locked per-problem log mean-JS with the 1e-9 floor.
  - Per unit: Spearman ρ, average ranks for ties. z = atanh(ρ), var = 1.06/(n−3).
  - Pooled: inverse-variance mean of z, with the 95% CI and the two-sided normal p from the pooled z.
- **R4.**
  - Pair p = dims (p, p + d/2), the HF rotate_half layout used by all ten models.
  - Common-mode energy per pair: E_p = Σ over problems, layers and KV heads of ‖S1K[..., pair p] / n_i‖², with n_i = the number of recorded entries. The share is s_p = E_p / Σ_p E_p.
  - Ā_p = mean over problems of A_p(n_i) = [sin(n_i ω_p / 2) / (n_i sin(ω_p / 2))]², with ω_p the model's inverse frequency for pair p.
  - Statistic: Spearman(s_p, Ā_p) over the d/2 pairs. Placebo: the same with S1V on the same dimension pairs.
  - p-value: two-sided permutation of Ā over pairs, 10,000 permutations, stream key "R4".

## RoPE inverse frequencies

- **General rule:** ω_p = θ^{−2p/d}, d = head_dim.
- **Qwen2.5 (all four), Mistral-7B-v0.3, Qwen3-0.6B-Base:** θ = 1e6, no scaling.
- **SmolLM2-1.7B:** θ = 130000, no scaling.
- **Llama-3.2-1B and 3B:** θ = 500000 with llama3 scaling (factor 32, low 1, high 4, original 8192). **Llama-3.1-8B:** the same with factor 8. Scaling follows the transformers `_compute_llama3_parameters` rule.
- **Provenance:**
  - Llama-3.2-1B, the Qwen models, Mistral, Qwen3 and SmolLM2 are read from the cached config at the pinned revision and asserted.
  - Llama-3.2-3B and Llama-3.1-8B: the official repositories are gated and these configs are not cached. Their values are taken from the ungated unsloth mirrors, whose config files differ from the pinned git blobs, so equality of the RoPE fields is assumed, not verified.
  - **Robustness check (reported):** the maximum absolute change in Ā_p when the scaling is dropped. Scaling acts only on wavelengths above 2048 positions, where A_p ≈ 1 for n ≤ 64.

## Reading rule, Holm, and verdicts (criteria from THEORY_PLAN_V6 §7, unchanged)

- **Holm family:** 21 two-sided p-values: R1 per unit (10), the R3 pooled test (1), and R4-K per unit (10).
- **"CI excludes 0" for R1:** a unit counts as CI-excluding-0 *on the predicted side* only if its 95% CI excludes 0 **and** its Holm-adjusted p < 0.05. A unit counts as "CI entirely < 0" if its CI is entirely below 0, regardless of Holm.
- **Verdicts:**
  - **R1.** Agreement: ≥ 8 of 10 units positive-significant, and no unit with CI entirely < 0. Disagreement: any unit with CI entirely < 0, or < 5 units with a positive point estimate. Otherwise *neither*.
  - **R3.** Agreement: pooled ρ > 0 with CI excluding 0 (and Holm p < 0.05), and ≥ 7 of 10 units with ρ > 0 (the ≥ 2/3 rule). Disagreement: pooled CI entirely ≤ 0. Otherwise *neither*.
  - **R4.** Agreement: ρ_K > 0.5 in ≥ 8 of 10 units, and ρ_K > ρ_V(placebo) in ≥ 8 of 10 units. Disagreement: ρ_K ≤ 0 in ≥ 5 of 10 units. Otherwise *neither*.
- **R2:** descriptive, no verdict.
- **Run once.** The script runs once on the real records after this commit. Any change afterwards is a deviation in `v6/DEVIATION_LOG_V6.md`, and both results are reported.
- **Gate for the L block (author instruction, 2026-11-14):** if the R block "clearly contradicts the theory", do not proceed. Operationally: a *disagreement* verdict on R1 or R3, the two tests of the value-path mechanism, counts as clearly contradicting. A disagreement on R4 alone is reported and referred to the author, because R4 is confounded (THEORY_PLAN_V6 §7).
