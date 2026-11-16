# THEORY_PLAN_V6: a first-order model of weight-induced KV-cache error, by path

> **Note (2026-11-15).** After the R block (R_BLOCK_REPORT.md) the author reframed the L block:
> - L3 and L4 are now estimation targets (the decomposition of H2 into common-mode and direction parts), not predictions.
> - L8 and L9 (direction/alignment) were added.
> - L7 gained a per-channel key storage arm.
>
> PROTOCOL_V6.md r2 §7 governs the L block. The rows below are kept as first written.

Status: **plan draft r1, 2026-11-14, for author review.** Not locked. No analysis has been run. Everything here is a plan.
Prior work this builds on (see NOVELTY_V6.md):
- Softmax shift invariance and value pass-through: NOVA-KV Lemma S1; SageAttention 1/2; Namazifar et al. 2023.
- The second-order (variance-driven) mass shift between perturbed and unperturbed blocks: Tuncer et al., arXiv:2605.26266.
- The systematic part of weight-quantization error in layer outputs: Nagel et al. 2019.

What is claimed as ours (if the tests below agree):
- the decomposition of *weight-induced* error into common-mode and position-varying parts per path;
- the first-order common-mode mass shift against the error-free prompt;
- the quantitative link to H2, H3 and H5;
- the account of why weight-induced error ranks V above K while storage error ranks K above V.

## 1. Setting (exactly the measured design)

These are the same objects as in the manuscript (Section 4.2):
- A stored perturbation δ_j = stored_j − κ^F_j is added at write time to the entry of decode position j, for every slot σ = (layer, K or V, KV head).
- Prompt entries are FP16 and unperturbed in every arm.
- At decode step t, a query head h attends over prompt positions 𝒫 and decode positions 𝒟_t = {1,…,t−1}, with FP16-state weights a_j (Σ_j a_j = 1).
- Let m = Σ_{j∈𝒟_t} a_j be the decode mass, and N_D = m² / Σ_{j∈𝒟_t} a_j² ∈ [1, |𝒟_t|] the effective number of attended decode positions.
- v̄_D and v̄_P are the a-weighted mean values over 𝒟_t and 𝒫, and ō = m v̄_D + (1−m) v̄_P is the head output.

**Per-slot decomposition.** Over a problem's n decode positions, δ_j = μ + ε_j with μ = S1/n and Σ_j ε_j = 0. With S2 = Σ_j‖δ_j‖², the
common-mode share is c = ‖μ‖² / (S2/n). This is exactly the H5 quantity C^P, already recorded per slot (S1K, S1V, normsK, normsV in
`test/slots/NAT_*.npz`). Independent isotropic directions give c ≈ 1/n. C1 has c ≈ 1/n by construction.

## 2. Value path

The output change is exact (linear): Δo_V = Σ_{j∈𝒟} a_j δv_j = m μ_V + Σ_{j∈𝒟} a_j ε_j.

**Assumptions:**
- **V1.** The ε_j are uncorrelated across positions and with a, so the cross term 2m μ·Σa_jε_j vanishes in expectation.
- **V2.** Equal per-position norms within a slot, for the algebra only. The logged version uses the actual norms.

Then

  E‖Δo_V‖² ≈ m²‖μ‖² + (1−c) r̄² Σ a_j²,  and for C1-V at the same norms ≈ r̄² Σ a_j²,

so the natural-to-isotropic energy ratio of one slot at one step is

  **R_V = 1 + c (N_D − 1) ≥ 1.**

Natural value error is harsher than isotropic noise of its size whenever it has a common mode (c > 1/n) and attention spreads over
more than one decode position. Isotropic error is averaged down by the attention weights; the common mode is not averaged at
all. This is the value half of (a) applied to an error, and it gives H2 a sign *and* a magnitude.

Two consequences follow:
- At step t = 2 only one perturbed entry exists, so N_D = 1 and R_V = 1. Any natural-vs-isotropic difference at step 2 is the
  pure **alignment** effect (direction relative to downstream sensitivity), with no common-mode contribution.
- R_V grows with t as N_D(t) grows. The common-mode part of H2 should therefore build up over the decode steps.

## 3. Key path

The score perturbation is δs_j = q·δk_j/√d = β + η_j on 𝒟 (β = q·μ_K/√d, η_j = q·ε_j/√d) and 0 on 𝒫. To first order in δs:

  Δo_K = Σ_j a_j (δs_j − δs̄)(v_j − ō) = **β m(1−m)(v̄_D − v̄_P)** + Σ_{j∈𝒟} a_j(η_j − η̄)(v_j − ō),  where δs̄ = Σ_j a_j δs_j.

1. **Cancellation within the decode block.** β drops out of every *within-decode* comparison; this is the key half of (a).
   It survives only through the decode-vs-prompt split, weighted by m(1−m) and by the value contrast ‖v̄_D − v̄_P‖. It has
   either sign. If all attention sat on the decode block (m = 1), the common mode would cancel exactly.
2. **Energy ratio.** With assumptions K1 (the η_j are uncorrelated across positions, like V1) and K2 (‖v_j − ō‖² ≈ s² on 𝒟):

     R_K ≈ λ_K · [ (1 − c_q) + c_q · N_D (1−m)² ρ_PD ],  with ρ_PD = ‖v̄_D − v̄_P‖² / s²,

   where:
   - c_q is the common-mode share in *score* space;
   - λ_K is the query-energy factor of the natural key error, E(q·δk)² / (‖q‖²‖δk‖²/d). The pilot measured the first-order logit energy at
     0.50–0.63 of isotropic (macro rng.x13R, Section 3.6, exposed).

   R_K < 1 (H1, natural key error milder) is predicted when λ_K is small or the common mode falls on near-cancelling
   configurations, i.e. N_D(1−m)²ρ_PD < 1. Unlike H2, **H1 is not signed by the theory alone**: it depends on m and ρ_PD, which are
   not recorded. That is the main reason for the logging protocol.
3. **Second-order term (Tuncer et al.).** Position-varying score noise of variance σ_η² inflates the decode block's partition
   sum by ≈ e^{σ_η²/2}, shifting mass toward the decode block: δm ≈ m(1−m)σ_η²/2. Isotropic C1-K error has the larger
   score-space variance when λ_K < 1, so this term also favours H1. It is cited as theirs and included in the logged
   prediction.

## 4. From head outputs to the measured divergence

JS between nearby distributions is second order: JS ≈ ⅛ Δzᵀ F Δz, with F the softmax Fisher metric at the reference logits.
Writing Δz = Σ_s J_s Δo_s to first order gives E[JS] ≈ ⅛ Σ_{s,s'} E[Δo_sᵀ J_sᵀ F J_{s'} Δo_{s'}].

**Assumptions:**
- **D1.** Cross-slot terms vanish in expectation.
- **D2.** Within a slot, the downstream metric G_s = J_sᵀFJ_s acts on natural and isotropic output perturbations alike, apart from the common-mode structure (no residual alignment).

Under D1–D2 the arm ratio of mean JS is a gain-weighted mean of the slot ratios R_s, summed over steps.

**D2 is known to be only partly true.** In the pilot, centering the value error closed only 0.27–0.34 of the value-path gap (macro rng.x13GV, exposed), so
there is a residual alignment effect. The theory therefore splits H2 as

  θ_V = θ_V^{cm} (common mode, predicted from c, N_D) + θ_V^{align} (residual, estimated, not predicted)

and the tests ask whether θ_V^{cm} is a substantial, correctly predicted part, not the whole.

## 5. Where the common mode comes from (H5)

The natural entry error is Δ_j = ΔW_P x_j + W^N_P e_j (projection error on the FP16-state input, plus the propagated upstream error of the same pass).
- **Inherited from the input.** Hidden states share a large mean direction (anisotropy; massive activations). If ΔW_P is roughly unstructured (assumption **O1**: ΔWᵀΔW ≈ σ²I on the input subspace), then c ≈ a_in = ‖x̄‖² / E‖x_j‖², the input's own common-mode share.
- **Same for K and V before RoPE.** k_proj and v_proj read the *same* normalized input in each layer, so before RoPE the K and V common-mode shares should be similar.
- **RoPE attenuates the key common mode.** Stored keys are post-RoPE, as verified in `v4/rv4.py`: SlotSums records the cache's keys. A pre-RoPE common mode in rotary pair p (frequency ω_p) is rotated by ω_p·pos_j. For n consecutive positions its post-RoPE mean is attenuated by the Dirichlet factor

    A_p(n) = [ sin(nω_p/2) / (n sin(ω_p/2)) ]²,

  so high-frequency pairs lose their common mode and low-frequency pairs keep it. This is a parameter-free prediction from each model's RoPE frequencies (from its config, including Llama-3 rope scaling) and the recorded n_i.

## 6. Why storage error favours K and weight-induced error favours V (the (b) account)

Two measured properties separate the two error sources. Neither needs a new mechanism.
1. **Relative size by path.** NF4 weight error puts 2.40–3.52× more relative error on V than on K at natural size (ρ_V/ρ_K, macro rng.rhoVK, eight-model study, exposed).
   Per-token round-to-nearest *storage* error scales with each group's absmax. Keys carry large shared and outlier channels
   (SageAttention; KIVI), so at equal bits the key error is at least as large: ρ_K^st/ρ_V^st ≥ 1 (prediction, needs logging).
2. **Structure.**
   - Weight-induced error is common-mode heavy (H5), which by §§2–3 softens K and hardens V.
   - Per-token rounding error is close to independent across positions (c ≈ 1/n), except where a channel carries a large shared value that is rounded the same way each token. So storage error should behave close to isotropic noise of its size on both paths.
   - Among isotropic-like errors at equal relative size, keys were the more disruptive path in the pilot (exposed). That is consistent with the storage literature's K-first ranking (KVTuner, AsymKV, QuaRot ablations).

So the account is: storage error ≈ isotropic, K-heavy in size, K-first; weight error = common-mode heavy, V-heavy in size, V-first. Both factors are
measurable in one design, with the same models, problems and outcome.

## 7. Predictions and pre-stated agreement criteria

Labels: **[R]** testable on existing v4/v5 records only. **[L]** requires new logging (PROTOCOL_V6.md). **Exposure** says which
related numbers the authors have already seen. A prediction whose quantity is already in the locked report is a consistency
check, not a test, and is not counted.

"Units" are the 8 v4 models plus the v5 NF4/GSM8K units that hold NAT/C1 single-path records and slot sums. The exact unit
list is fixed in the analysis-plan commit. Intervals use a 10,000-resample problem bootstrap with a seed fixed in that commit. Holm
correction runs over the counted tests of each block (R block, L block).

| ID | Prediction | Statistic | Agreement | Disagreement | Data | Exposure |
|---|---|---|---|---|---|---|
| R1 | Common-mode part of H2 builds over steps (§2) | Per problem, OLS slope of r_t = ln js^{NAT-V}_t − ln js^{C1-V}_t (floored 1e-9) on ln t, t = 2…T_i; unit estimate = mean over problems | Slope > 0 with 95% CI excluding 0 in ≥ 75% of units, and no unit with CI entirely < 0 | Any unit with CI entirely < 0, or < 50% of units positive | [R] per-step js in `test/*.jsonl` | Partial: the locked steps 2–8 vs 9–64 H2 windows (sensitivity table) are known. The step-resolved slope is not computed. |
| R2 | Step-2 V contrast is alignment only (§2) | Unit mean of r_2 | Reported, with no sign prediction. It defines θ_V^{align}(t=2) for R1 and L3 | (none, descriptive) | [R] | Not computed |
| R3 | Problems with more common mode show a larger V excess (§2) | Spearman(x_i^V, θ_i^V) within unit; pooled by inverse-variance Fisher-z meta-analysis | Pooled ρ > 0 with CI excluding 0, and ≥ 2/3 of units with ρ > 0 | Pooled CI entirely ≤ 0 | [R] slot sums + per-problem outcomes | Not computed (X3 is H1 vs NAT−C1) |
| R4 | RoPE attenuates the key common mode (§5) | Per unit, share s_p of Σ‖S1K_p‖² in rotary pair p (HF rotate_half layout), pooled over layers, heads, problems; Spearman(s_p, Ā_p) with Ā_p = mean_i A_p(n_i). Placebo: the same on S1V with the same pair indices | K: ρ > 0.5 in ≥ 75% of units, and ρ_K > ρ_V(placebo) in ≥ 75% | K: ρ ≤ 0 in ≥ 50% of units | [R] S1K, S1V; RoPE frequencies from configs | Not computed. **Confound:** per-dimension key-error energy may itself vary with frequency (low-frequency massive dimensions). The decisive version is L5. |
| C1 | C^V ≥ C^K per layer; signs of H1, H2, H3 | (consistency only) | | | [R] | **Exposed** (locked report, X2) |
| L1 | The first-order model is accurate locally | Per slot-step, predicted vs measured local head-output perturbation energy (§8 of the protocol), V and K | Median abs ln(pred/meas) ≤ ln 1.25 (V) and ≤ ln 1.5 (K) in every model | Median above ln 2 in any model | [L] | New |
| L2 | Head-output energy mediates the outcome | Spearman across problems of Σ_s ln(E_out^{NAT}/E_out^{C1}) vs θ_i, per path, per model | ρ > 0, CI excluding 0, for V in every model | V CI entirely ≤ 0 in any model | [L] | New |
| L3 | The common mode is a substantial, correctly predicted part of H2 | θ̂_V^{cm} = gain-free aggregate of ln R_V from logged c, N_D; share θ̂_V^{cm}/θ_V | Share in [0.2, 0.8] in every model, and the predicted sign correct | Share < 0.1 or wrong sign in any model | [L] | Pilot centering (≈1/3) is known, which motivates the band; the band is fixed now |
| L4 | Centering closes the predicted share of the V gap | Measured G (NAT-V-cen arm) vs predicted G from L3 | abs(G_meas − G_pred) ≤ 0.15 in every model | > 0.30 in any model | [L] | Pilot G known for two of the three models; the third (Qwen2.5-1.5B) is new |
| L5 | RoPE attenuation, decisive | Per pair p, post-RoPE common-mode share c_p (with per-pair totals now logged) vs pre-RoPE c_p × A_p(n) | Median abs ln(c_post/(c_pre·A_p)) ≤ ln 1.5 over pairs with A_p ≥ 0.05; and pre-RoPE c^K ≈ c^V within a factor of 1.5 per layer | Ratio off by more than 2× in most pairs | [L] | New |
| L6 | Key-path sign from logged quantities | Predicted ln R_K (first order + Jensen term) aggregated as L3, vs measured θ_K (H1) | Same sign as H1 in every model, and problem-level Spearman > 0 | Opposite sign in any model | [L] | H1 signs known; the problem-level relation is new |
| L7 | Storage error ≈ isotropic, K-first (§6) | Arms ST-K and ST-V (per-token RTN INT4, group 32, asymmetric, post-RoPE keys) at codec size, and ST@nat re-normed to NF4 natural per-slot norms | (i) n·C^V_st ≤ 2 in every model; (ii) θ(ST@nat-P, C1-P) equivalent to 0 within the locked margin on both paths; (iii) ρ_K^st/ρ_V^st ≥ 1 and θ(ST-K, ST-V) > 0 (CI excluding 0) in every model, while H3 (NAT-K vs NAT-V < 0) replicates | (ii) fails on V by a difference at least as large as H2; or (iii) reversed | [L] | New |

**How to read failures:**
- R1/R3/L3 failing means the common mode is not what makes value error harsh.
- L1 failing means the first-order model is the wrong tool.
- L7(ii) failing means storage error is structured beyond its common mode, and the account in §6 is incomplete.

Each outcome is reportable. None will be reworded after the data.

## 8. Analysis discipline for the [R] tests

- The [R] analysis script, the unit list and this table are committed and hashed **before** the script first reads any per-step js or slot array.
- The script is tested on synthetic records first, as for analyze_v4.
- R1–R4 run once. Any change after the first run is a logged deviation, and both versions are reported.
- The [R] tests use records whose aggregate results the authors have seen. They are therefore reported as *pre-registered secondary
  analyses of existing data*, not as confirmatory tests. The [L] block, on fresh problems, is the confirmatory part.

## 9. Not claimed

- Not claimed: the cancellation and pass-through identities (prior work), or the second-order mass shift (Tuncer et al.).
- No claim beyond first order. The pilot's centering result already shows a residual alignment effect that this model does not predict.
- No claim about models above 1.5B from the [L] block (laptop only). The [R] block covers 0.5B–8B, but only through quantities already logged.
