# PROTOCOL_V3 — Origin and transfer of weight-induced cache error (r4 + amendments 1 and 2, version for the lock)

Amendment 1 (2026-10-29, before any GPU use): record integrity and ledger accounting only; see §15.
Amendment 2 (2026-10-29, after the qwen-base gate failure V3-D04, before any further GPU use): the per-slot norm check is split into a construction check and a rounding check; see §16. Text superseded by §16 is marked below.

Revision r4, 2026-10-29, branch `v3-novelty`. It incorporates the author decisions on drafts r1 (commit c5cdf53), r2 (commit 0638346)
and r3 (commit 94eb9f2), and the pre-lock implementation notes of r4; see §14.
**This is the text that the private dated hash `research_v2/V3_PROTOCOL_LOCK.txt` covers. Locking is not an authorization to run.**
- GPU work (Phase 4) requires the authors to create `research_v2/v3_runs/AUTHORIZED_BY_AUTHORS_V3.txt`, containing the text
  `PROTOCOL_V3` and the lock id printed by `tools/lock_v3.py --lock-id`. The driver and the GPU mutation tool refuse to run
  without it.
- Like E1/E1L, the lock will be a private dated SHA-256 hash, **not a public pre-registration**. The hypotheses are
  exploratory in origin: they follow E1, E1L and the Phase 1 literature check.
- E1 and E1L stay as published evidence. Nothing here edits their code, locks, records or analyses.

## 1. Questions and what is borrowed

**Setting.** Greedy decoding with FP16 weights. At every step a second, reduced-precision pass from the same token and
cache state measures the difference Δ = κ^M − κ^F of the **newly written** KV entry. The arms differ only in what is
stored in place of κ^F (E1 design).

1. **Transfer test.** Do error predictors established for *KV-storage* quantization explain *weight-induced* cache
   errors, and the disagreement between Qwen (θ_A = −0.253, NAT milder than same-norm noise) and Llama (θ_A = +0.058)?
2. **Origin decomposition.** Where does the error come from?
   - **local**: NF4 k_proj/v_proj acting on the current residual;
   - **inherited**: the residual stream reaching k_proj/v_proj was altered by the other NF4 layers.
3. **2×2 models.** Family (Qwen2.5, Llama-3.2) × tuning (base, instruct), **one model per cell**.

**Borrowed, cited as precedent, not claimed:**
- query-weighted logit error as the key-error predictor: PN-Mix arXiv 2610.19605; NOVA-KV arXiv 2608.04074; AATC
  arXiv 2608.14191;
- attention-weighted value error: AATC; NOVA-KV;
- same-magnitude Gaussian and sign-randomized substitutes for a natural quantization error: arXiv 2609.23125, applied
  there to residual-stream activations;
- energy-matched one-path-at-a-time perturbation: QuantMLA arXiv 2609.36760;
- K vs V asymmetry: KVTuner, ICML 2025 (PMLR 267), storage; arXiv 2609.26173, weight projections;
- projection-level weight quantization: arXiv 2609.26173;
- cross-layer (inherited) propagation of weight-quantization error: arXiv 2506.12044; QEP arXiv 2504.09629;
  arXiv 2608.08188;
- persistence through the cache: KVarN arXiv 2606.03458 (storage); Lorup arXiv 2607.28495 (compute precision).

HeadQ (arXiv 2605.03562) is withdrawn and is **not cited**. PN-Mix and arXiv 2609.26173 are cited as arXiv preprints
without a venue.

**Novelty wording (fixed for all outputs).** "To our knowledge, these established tools have not been applied to KV
entries written by a weight-quantized model. The contribution is their combination for weight-induced cache error:
matched controls on the written entries, a local/inherited decomposition and a transfer test of storage-error
predictors, across a family × tuning design with one model per cell." Each component is prior art. "First" is never
used.

## 2. Models, licences, memory

| Model | HF id (cached snapshot) | Layers | Query / KV heads (group) | head_dim | Licence (HF page, 2026-10-29) |
|---|---|---|---|---|---|
| Qwen2.5-0.5B | `Qwen/Qwen2.5-0.5B` (060db649…) | 24 | 14 / 2 (7) | 64 | apache-2.0 |
| Qwen2.5-0.5B-Instruct | `Qwen/Qwen2.5-0.5B-Instruct` (7ae55760…) | 24 | 14 / 2 (7) | 64 | apache-2.0 |
| Llama-3.2-1B | `meta-llama/Llama-3.2-1B` (4e20de36…) | 16 | 32 / 8 (4) | 64 | Llama 3.2 Community License (gated) |
| Llama-3.2-1B-Instruct | `meta-llama/Llama-3.2-1B-Instruct` (92131767…) | 16 | 32 / 8 (4) | 64 | Llama 3.2 Community License (gated) |

- The geometry above is read from the cached `config.json`; gate G7 re-checks it from the live cache.
- **Licences.** Research use and reporting results are compatible with both licences. Releasing Llama outputs or
  derivatives carries the "Built with Llama" and notice obligations; that is an author decision for the data
  statement.
- **Memory.** Same dual-weight implementation as E1/E1L. Qwen2.5-0.5B and Llama-3.2-1B-Instruct ran on this 8 GB GPU,
  and the siblings have identical shapes. One model is loaded at a time.
- **Prompt.** The raw GSM8K 3-shot completion prompt of E1/E1L for all four models. No chat template (§13,
  limitation L1).

## 3. Conventions reused unchanged from E1/E1L

- **Tier A only.**
  - Teacher-forced; steps s = 1…S_i with S_i = min(64, history length − 1).
  - JS divergence in float64. The per-problem mean m_i is floored at 1e-9 before logs.
- **Samples.** The same 200 seeded test ids (`e1_sample.json` tierA_ids, a fixed order) and the same 100 GSM8K-train
  calibration ids. The gate sets are prefixes of these. Exposed test problems are handled by a sensitivity analysis.
- **Histories, per model.** One in-session FP16 greedy generation per problem, capped at 65 tokens with the
  drift-marker stop (E1L procedure), shared by all arms.
  - Gate G5 reports agreement of the Qwen2.5-0.5B histories with the stored E1 histories.
  - Reference distributions come from a pure-FP16 teacher-forced pass in the same session.
- **Inference.**
  - Problem-clustered percentile bootstrap: 10,000 resamples, locked-analysis seed 20261103.
  - Two-sided paired sign-flip permutation tests: 10,000.
  - Holm within each declared family.
  - No step-level p-values.
- **Seeds.** SHA-256 of (master `rv3-cache-origin-2026-10-29`, model label, arm, scale, problem index, tier, repeat):
  labels only.
- **Writes.** Perturbations go **only to the newly written entry, at write time**. The prefill is FP16 and never
  perturbed.
- **α = 0 gate.** Every perturbation arm at scale 0 equals pure FP16 bitwise.
- ~~**Per-slot norm tolerance enforced in ALL arms**, in the gates **and** in the per-problem integrity check.~~
  - ~~Tolerance: mean per-step relative deviation ≤ 1 %, per-slot maximum ≤ 10 %.~~
  - ~~Judged against max(target, 2⁻¹⁰·‖entry‖).~~ (Also mis-stated: E1/E1L and r4 code judge against target + 2⁻¹⁰·‖entry‖.)
  - **Superseded by §16 (amendment 2):** construction check (float64, before rounding, ≤ 1e-4) and rounding check (bitwise), both in ALL
    arms, in the gates and in the per-problem integrity check; the post-rounding deviation is reported, not gated.

## 4. The controlled step and the arms

**Step.** At decode step s, from token x_{s−1} and the arm's cache S_{s−1}:
1. run the **measurement pass** (precision map per arm);
2. snapshot its new entry κ^M;
3. crop;
4. run the **FP16 pass** (logits and κ^F);
5. overwrite the new entry with the arm's stored value.

A stored perturbation first affects step s + 1, so step-1 JS is exactly 0 (integrity check).

**Slot.** (layer, K or V, KV head). Norms are L2 over head_dim and computed in float32. Stored entries are FP16.

**Notation.** Δ = κ^M − κ^F; u is uniform on the unit sphere of R^64; ε is i.i.d. ±1; g ~ N(0, I₆₄). "κ^F" in a
column means that path is left as computed by FP16.

| Arm | Measurement map | Stored K | Stored V | Role |
|---|---|---|---|---|
| FP16 | none | κ^F | κ^F | reference (separate pass) |
| **NAT** | all NF4 | κ^N (bitwise) | κ^N (bitwise) | natural weight-induced error |
| **C1** | all NF4 | κ^F + ‖Δ‖u | κ^F + ‖Δ‖u | same per-slot norm, isotropic |
| C2 | all NF4 | κ^F + ε⊙Δ | κ^F + ε⊙Δ | same per-coordinate magnitudes, random signs |
| **QA** | all NF4 | κ^F + ‖Δ_K‖·P₈g/‖P₈g‖ | κ^F + ‖Δ_V‖u_V | key error inside the top-8 query subspace |
| **QO** | all NF4 | κ^F + ‖Δ_K‖·(I−P₈)g/‖(I−P₈)g‖ | κ^F + ‖Δ_V‖u_V | key error orthogonal to it |
| CV | all NF4 | κ^F + ‖Δ‖·w/‖w‖, w_c = σ_c g_c | same, own σ | population channel-variance profile |
| NAT-K | all NF4 | κ^N_K | κ^F_V | natural, keys only |
| NAT-V | all NF4 | κ^F_K | κ^N_V | natural, values only |
| **NAT-K\*** | all NF4 | κ^F + ρ\*‖κ^F_K‖·Δ_K/‖Δ_K‖ | κ^F_V | energy-matched keys only |
| **NAT-V\*** | all NF4 | κ^F_K | κ^F + ρ\*‖κ^F_V‖·Δ_V/‖Δ_V‖ | energy-matched values only |
| **C1-K** | all NF4 | κ^F + ‖Δ_K‖u | κ^F_V | isotropic at the natural K norm |
| **C1-V** | all NF4 | κ^F_K | κ^F + ‖Δ_V‖u | isotropic at the natural V norm |
| **LOC** | NF4 only in k_proj, v_proj | κ^LOC (bitwise) | κ^LOC (bitwise) | local origin |
| **INH** | NF4 except k_proj, v_proj | κ^INH (bitwise) | κ^INH (bitwise) | inherited origin |

Details:
- **QA/QO.**
  - P₈ = U₈U₈ᵀ, from the calibration query second moment of that (layer, KV head) (§5).
  - The value side uses the same isotropic draw in both arms (shared generator label `QAQO-V`), so QA − QO differs
    only in key direction.
  - Orthogonality is checked on the float32 δ before FP16 rounding (G2b). Post-rounding leakage is reported.
- **CV.** σ_c² is the calibration mean of Δ_c² per (layer, K/V, KV head) from NAT. CV keeps only the *average*
  anisotropy. C2 keeps the *instantaneous* per-coordinate magnitudes.
- **ρ\* and scale factors.** See §5. A slot with Δ = 0 stores κ^F; its count is recorded.
- **LOC/INH routing.** Uses the existing per-instance attribute `DynamicLinear.layer_force_4bit`
  (src/dynamic_linear.py, documented as a diagnostic override). It is set from a declared module map for the
  measurement pass only, and reset to `None` before the FP16 pass. All locked code stays unchanged.
  - Counts: Qwen 168 modules (LOC 48 NF4 / INH 120 NF4); Llama 112 (LOC 32 / INH 80).
  - G8 verifies the map module by module.
- **What LOC and INH mean exactly.**
  - LOC is the error that starts in the k/v projections, *including* its within-step spread through the new
    position's own attention to layers > l.
  - INH is the error that starts in q_proj, o_proj and the MLP and reaches k/v through the residual stream.
  - NAT has both. The decomposition is not additive by construction.

## 5. Calibration (100 GSM8K-train problems; per model; before any test problem of that model)

The arms are FP16 reference, NAT, C1 at α ∈ {0.5, 1, 2}. Teacher-forced, 64 steps. Frozen in
`v3_runs/<model>/calibration.json`; its SHA-256 is appended to the lock before the model's first test problem.

| Quantity | Definition |
|---|---|
| Query second moment M_{l,h} | Σ over calibration problems, steps 1…64 and the query heads of KV head h's group of q qᵀ (post-RoPE FP16-pass query, float64). GQA: pooled within the group (7 Qwen, 4 Llama). |
| U₈ (primary) | top-8 eigenvectors of M_{l,h}. **Reported per model:** captured query energy E₈ = Σ_{k≤8} λ_k / tr M, median and range over (l, h). |
| r₈₀ (diagnostic only, no arm, no test) | smallest r with Σ_{k≤r} λ_k / tr M ≥ 0.80, per (l, h). **Reported per model** with E₈: median and range over (l, h), and the share of slots with r₈₀ ≤ 8. |
| σ_c² (CV) | mean over NAT calibration steps of Δ_c² per (layer, K/V, KV head, channel) |
| ρ_K, ρ_V | calibration medians of ‖Δ_K‖/‖κ^F_K‖ and ‖Δ_V‖/‖κ^F_V‖, pooled over slots and steps of NAT |
| **ρ\*** | √(ρ_K·ρ_V), the geometric mean |
| **Scale factors** | s_K = ρ\*/ρ_K = √(ρ_V/ρ_K) for NAT-K\* and s_V = ρ\*/ρ_V = 1/s_K for NAT-V\*. Reported per model. The per-step distribution of ρ\*‖κ^F‖/‖Δ‖ on test steps is reported descriptively. |
| Outlier channels O (P_out) | the 4 channels (of 64) with the largest calibration mean \|κ^F_c\| per (l, h, K/V) |
| b̂, c, margin | E1 rules: b̂ = mean per-problem OLS slope of log m on log α (α ∈ {0.5, 1, 2}); c = median step-to-step CV of the NAT perturbation norm, clamped to [0.10, 0.40]; margin_m = \|b̂\|·ln(1 + c); equivalence claimable only if b̂ ≥ 0.25 |

**Pre-declared H-D extrapolation rule.** The measured dose range is α ∈ [0.5, 2] (C1 dose arms). If s_K or s_V lies
outside **[0.5, 2]** for a model (equivalently, ρ_V/ρ_K outside [0.25, 4]), H-D for that model is reported as
**descriptive**:
- the energy-matched arms then extrapolate beyond the measured dose range;
- H-D is removed from that model's Holm family (family size 7 → 6);
- this is decided from the calibration file, before any test problem of that model.

## 6. Predictors (per arm, per step t ≥ 2)

**Inputs at step t.**
- J_t: the decode positions written at steps 1…t−1.
- δk_j, δv_j: (stored − κ^F) of position j, recorded at write time in float32 per layer and KV head.
- q_{t,l,g}: the post-RoPE FP16-pass query of query head g. It is captured by read-only hooks (a forward hook on
  q_proj and a forward pre-hook on self_attn for `position_embeddings`) and re-rotated with the model's
  `apply_rotary_pos_emb`.
- p_{t,l,g,j}: softmax over all cached positions of q·k_j/√d, recomputed in float32 from the **stored** keys
  (plug-in form).
- h(g): the KV head of g. d = 64. Averages run over all layers and query heads, i.e. pooled within each KV group.

**First-order origin.** For z_{tj} = q·k_j/√d, a key error changes the logit by q·δk_j/√d to first order. A value
error changes the attention output by Σ_j p_tj δv_j.

| Predictor | Formula |
|---|---|
| **P_norm** (baseline) | (1/(2·L·H_kv)) Σ_l Σ_h Σ_{X∈{K,V}} sqrt( (1/\|J_t\|) Σ_{j∈J_t} ‖δX_{j,l,h}‖² ) |
| **P_logit** | (1/(L·H_q)) Σ_l Σ_g Σ_{j∈J_t} p_{t,l,g,j} · \|q_{t,l,g}·δk_{j,l,h(g)}\| / √d |
| **P_val** | (1/(L·H_q)) Σ_l Σ_g ‖ Σ_{j∈J_t} p_{t,l,g,j} δv_{j,l,h(g)} ‖ |
| P_anormK (secondary) | (1/(L·H_q)) Σ_l Σ_g Σ_{j∈J_t} p_{t,l,g,j} ‖δk_{j,l,h(g)}‖ |
| P_out (secondary) | mean over (l, h, X) and j ∈ J_t of Σ_{c∈O} δX_c² / ‖δX‖² |

How these differ from the precedents (from the full texts):
- **PN-Mix** (§4.2, Eqs. 3–6): an exact per-RoPE-pair logit difference and a separable proxy E_t‖q_i‖·E_s‖k_i‖, with
  query norms averaged over positions and over the query heads sharing a KV head. Keys only.
- **AATC** (Def. 3; Thm. 1, Eq. 4): E‖o − ô‖² with a fixed query, a second-order softmax expansion, white-noise errors
  (Assumption 1) and W_O-weighted terms.
- **NOVA-KV** (§3.1, Eq. 5): the key distortion uses the calibration M_q = QᵀQ; the value distortion is weighted by
  the attention-score second moment (§3.2).
- **Ours:** the realized per-step query and δk; no white-noise assumption (the assumption NAT vs C1 tests); absolute
  first-order terms. M_q is used only to define the QA/QO subspaces.

## 7. Outcomes

- **Per problem:** m_i^a, the mean JS over steps 1…S_i for arm a. Arm contrasts are θ(a,b) = mean_i [log m_i^a −
  log m_i^b]. Positive means a is more disruptive.
- **Per step:** JS_t, the top-1 flip, the five predictors, and per-path norm diagnostics.

## 8. Hypotheses, estimands, decision rules

**Families (Holm within each).**
- **Primary family P (4 tests).** H-A in each model.
- **Per-model family F_m (7 tests per model; 6 if H-D is descriptive by §5).** NAT − C1, H-B, H-C_m, H-D,
  provenance-K, provenance-V, H-E.
- **Cross-model family X (3 tests).** H-F family, tuning and interaction.
- **H-C cross-model decision (§8.4).** A single pre-declared decision rule, evaluated at the 95 % bootstrap level; not
  in a Holm family.

**Verdict rules (E1).**
- **Difference:** the 95 % CI excludes 0. The Holm-adjusted p is reported, and "Holm-significant" is a separate label.
- **Equivalent:** for log-JS contrasts only, the 90 % CI lies inside ±margin_m and b̂_m ≥ 0.25. H-F uses the mean of
  the four margins.
- **Inconclusive:** otherwise.
- **Never equivalence from non-significance.** H-A and H-C have no equivalence category.
- **p-values.** Sign-flip permutation p for the paired mean estimands (H-A, θ contrasts, H-F). For the H-C shrinkage,
  the two-sided bootstrap p = 2·min(P*(S ≤ 0), P*(S ≥ 0)).

### 8.1 H-A (transfer) — primary, per model
- **Estimand.** D_m = mean_i [Spearman_t(P_logit, JS) − Spearman_t(P_norm, JS)].
  - Computed within the **NAT arm only**, over steps t = 2…S_i of problem i, with average ranks.
  - Problems with S_i < 8 are excluded and counted.
  - NAT only avoids circularity: QA/QO are constructed to vary in P_logit.
- **Prediction:** D_m > 0 in each model.
- **Support:** Holm-adjusted p < 0.05 and CI excluding 0 on the positive side.
- **Falsification:** in model m, the 95 % CI of D_m lies entirely below 0, i.e. P_norm ranks NAT's per-step JS better
  than P_logit. A CI including 0 is *inconclusive*, never "no transfer".
- **Secondary (descriptive):**
  - the pooled estimate over models (mean_m D_m, bootstrap over the shared problems);
  - the pooled norm-matched arms version: observations (a, t) over {NAT, C1, C2, QA, QO, CV};
  - P_val, P_anormK and P_out against P_norm in NAT.

### 8.2 H-B (query alignment), F_m
- **Estimand.** θ_B = θ(QA, QO) with U₈.
- **Prediction:** θ_B > 0.
- **Falsification:** the 95 % CI lies entirely below 0, or the equivalence rule is met. Either way, key error along
  the query subspace is not more disruptive than orthogonal key error at equal norm.
- **Diagnostic (no test):** per model, E₈ and r₈₀ (§5). They state how much of the query energy the fixed r = 8 subspace
  covers, and therefore how strong a contrast QA − QO can be.

### 8.3 H-C_m (regression adjustment, per model), F_m
- **Inputs.** Paired step differences for t = 2…S_i:
  - y_it = log JS_t^NAT − log JS_t^C1;
  - x_it = log P_logit,t^NAT − log P_logit,t^C1;
  - floors 1e-12 (JS) and 1e-30 (P).
- **Contrasts.** Unadjusted β₀ = mean y. Adjusted β₁ = intercept of pooled OLS y = β₁ + γx. Problem-clustered
  bootstrap: resample problems, refit.
- **Shrinkage.** S_m = |β₀| − |β₁|. Shrinkage fraction f_m = 1 − |β₁|/|β₀|.
- **Precondition:** the 95 % CI of β₀ excludes 0. Otherwise there is no NAT−C1 contrast to explain: H-C_m is reported
  as *not applicable* and is removed from F_m before Holm. This is decided from the β₀ interval alone.
- **Decision bands for f_m** (95 % percentile-bootstrap CI):

  | Band | Rule |
  |---|---|
  | **Substantial** explanation | f̂_m ≥ 0.50 and CI lower bound > 0 |
  | **Partial** explanation | CI lower bound > 0 and f̂_m < 0.50 |
  | **No** explanation | CI upper bound < 0.25 (includes f_m ≤ 0: adjustment does not shrink, or enlarges, the contrast) |
  | Inconclusive | otherwise |

- **Falsification:** band "No". P_logit does not account for the NAT − C1 contrast in that model.

### 8.4 H-C cross-model decision (the E1/E1L disagreement)
- **Pair.** Qwen2.5-0.5B (QB) vs Llama-3.2-1B-Instruct (LI), the pair that disagreed.
- **Gaps.** g₀ = β₀^{QB} − β₀^{LI} (unadjusted gap) and g₁ = β₁^{QB} − β₁^{LI} (adjusted gap), with β's as in §8.3.
- **Shrinkage fraction.** **F = 1 − |g₁|/|g₀|.** Bootstrap over the 200 shared problems, resampling the same problems
  for both models.
- **Precondition:** the 95 % CI of g₀ excludes 0, i.e. the in-session rerun reproduces a resolved disagreement.
- **If the precondition fails** (95 % CI of g₀ includes 0), the decision rule is not applied. That outcome is then
  **reported as a finding, at abstract level**: "The Qwen-base vs Llama-instruct difference in the NAT−C1 contrast
  seen across sessions in E1/E1L did not replicate when both models were rerun in the same session." g₀ and its CI
  are given. The item-2 problem-mean contrasts θ(NAT, C1) of both models are reported with it. The manuscript may then
  not present the E1/E1L disagreement as an established model difference.
- **Decision** (95 % percentile CI of F):

  | Band | Rule | Reading allowed |
  |---|---|---|
  | **Substantial** | F̂ ≥ 0.50 and CI lower bound > 0 | "query-weighted logit error accounts for at least half of the Qwen/Llama difference in the NAT−C1 contrast" (statistical adjustment, not a causal account) |
  | **Partial** | CI lower bound > 0 and F̂ < 0.50 | "accounts for part of the difference" |
  | **No** | CI upper bound < 0.25 | "does not account for the difference" |
  | Inconclusive | otherwise | interval only |

- **Falsification of the transfer explanation of the disagreement:** band "No".
- The other five model pairs are reported in the same form, descriptively.

### 8.5 H-D (which path), F_m, two-sided with competing predictions
- **Estimand.** θ_D = θ(NAT-K\*, NAT-V\*), at equal relative per-slot size ρ\*.
- **Readings:**

  | Result | Supports | Falsifies |
  |---|---|---|
  | θ_D > 0, CI excludes 0 (K more disruptive) | storage-level account (KVTuner) transfers to weight-induced error | the weight-projection account (V-dominant) for this model |
  | θ_D < 0, CI excludes 0 (V more disruptive) | weight-projection account (2609.26173) | the storage-level account (K-dominant) for this model |
  | Equivalent | neither path dominates per unit relative error | both dominance accounts at this resolution |
  | Inconclusive | — | nothing |

- **Pre-stated caution.** The precedents concern different quantities: equal bit-width storage error, and the
  perplexity of single-projection weight quantization.
- **Descriptive:** natural NAT-K vs NAT-V, the natural ratio ρ_K/ρ_V, and s_K, s_V.
- **Extrapolation rule:** §5 (descriptive if s_K or s_V ∉ [0.5, 2]).

### 8.6 Provenance within each path, F_m
- **Estimands.** θ(NAT-K, C1-K) and θ(NAT-V, C1-V). No directional prediction.
- **Hypothesis tested:** "the natural error on path X is behaviourally interchangeable with isotropic noise of its
  size".
- **Falsification of that hypothesis:** the 95 % CI excludes 0.
- **Support:** the equivalence rule is met.

### 8.7 H-E (origin), F_m
- **Estimand.** θ_E = θ(INH, LOC).
- **Prediction:** θ_E > 0 (INH more disruptive), expected from 2506.12044, QEP and 2608.08188, so confirmation is
  reported as *consistent with prior work*, not as a discovery.
- **Falsification of the expectation (reported as a surprise):** LOC ≥ INH, i.e. the 95 % CI lies entirely below 0,
  or the equivalence rule is met.
- **Descriptive non-additivity:** m^NAT / (m^LOC + m^INH) per problem (median, bootstrap CI), and ‖Δ_NAT‖² vs
  ‖Δ_LOC‖² + ‖Δ_INH‖². These come from different trajectories.

### 8.8 NAT − C1, F_m
- θ(NAT, C1) in-session: the E1/E1L primary.
- **Reading:** replication categories as in E1L §3, against E1 (Qwen-base) and E1L (Llama-instruct). For the two new
  models it is a first estimate.

### 8.9 H-F (2×2), family X — a test across these four models (one model per cell)
- **Per problem.** θ_{m,i} = log m_i^NAT − log m_i^C1, with models a = Qwen-base, b = Qwen-instruct, c = Llama-base,
  d = Llama-instruct, paired by problem.
- **Contrasts:**
  - family = ½[(c+d) − (a+b)];
  - tuning = ½[(b+d) − (a+c)];
  - **interaction = ½[(a − b) − (c − d)]**, pre-declared.
- **Scope of the inference.** These are contrasts between four specific models. With one model per cell, a "family"
  contrast is Qwen2.5-0.5B/Qwen2.5-0.5B-Instruct vs Llama-3.2-1B/Llama-3.2-1B-Instruct, and is confounded with
  architecture, size, data and tokenizer. No claim generalizes to model families or to instruction tuning in general.
  No single cause is claimable.
- **Pre-declared predictions** (from E1/E1L, where Llama-instruct θ exceeded Qwen-base θ by 0.31):
  - *family account*: family > 0;
  - *tuning account*: tuning > 0;
  - interaction: no directional prediction (two-sided).
- **Falsification:**
  - the family account is falsified if the family contrast's 95 % CI lies entirely below 0 or the equivalence rule
    holds;
  - the tuning account likewise for the tuning contrast;
  - "the factor effects are additive" is falsified if the interaction's CI excludes 0.
- **Power.** From E1/E1L per-problem θ: SD 0.53 / 0.46 and r_b = 0.05. The MDE at n = 200, 80 % power, two-sided 5 %
  (log scale) is:
  - family 0.096–0.132;
  - tuning and interaction 0.096 (r_w = 0.05) down to 0.031 (r_w = 0.9);
  - computed by `v3/phase2/project_budget.py`.
- **25 % guard.** If the realized per-model SD of θ_{m,i} exceeds the assumed common SD (0.497) by more than 25 % in
  any model, family X is relabelled descriptive. This check uses only the SDs and runs before the contrasts are
  computed.

### 8.10 Falsification summary (every hypothesis)

| Hypothesis | Falsified when |
|---|---|
| H-A (per model) | 95 % CI of D_m entirely < 0 |
| H-B | 95 % CI of θ(QA, QO) entirely < 0, or equivalence |
| H-C_m | f_m band "No" (CI upper bound < 0.25) |
| H-C cross-model | F band "No" (CI upper bound < 0.25) |
| H-D storage (K-dominant) account | θ_D < 0 with CI excluding 0, or equivalence |
| H-D weight-projection (V-dominant) account | θ_D > 0 with CI excluding 0, or equivalence |
| Provenance (interchangeable with isotropic noise), per path | 95 % CI excludes 0 |
| H-E expectation (INH > LOC) | 95 % CI of θ(INH, LOC) entirely < 0, or equivalence |
| H-F family account / tuning account | the respective contrast's CI entirely < 0, or equivalence |
| H-F additivity | interaction CI excludes 0 |

**Descriptive, labelled.** NAT − C2; C2 − C1; NAT − CV; NAT-K vs NAT-V; E₈; r₈₀; ρ_K, ρ_V, s_K, s_V; ‖Δ‖/‖κ‖ by path
and model.

**Sensitivity (none replaces a primary).** Steps 2–8 and 9–64; per-problem medians; excluding exposed test problems;
problems with S_i ≥ 32; the pooled norm-matched-arm H-A.

## 9. Budget, gates, stop rules

**Projection** (`v3/phase2/project_budget.py` → `budget_projection.json`).
- Basis: measured E1/E1L time per arm-problem (Qwen 5.39 s, Llama 3.32 s), ledger overheads, +15 % assumed predictor
  overhead, 0.35 h gates per model, 5 % margin. 14 test arms.

| Model | Calibration | Test (14 arms × 200) | Overhead + gates | Total (incl. 5 %) |
|---|---|---|---|---|
| Qwen2.5-0.5B | 0.78 h | 4.82 h | 0.46 h | **6.36 h** |
| Qwen2.5-0.5B-Instruct | 0.78 h | 4.82 h | 0.46 h | **6.36 h** |
| Llama-3.2-1B | 0.47 h | 2.97 h | 0.50 h | **4.14 h** |
| Llama-3.2-1B-Instruct | 0.47 h | 2.97 h | 0.50 h | **4.14 h** |
| **Total** | | | | **21.0 h** |

- One test arm across all four models costs ≈ 1.17 h.
- **Hard ceiling: 25.0 GPU-hours cumulative** (gates, GPU mutants, calibration, test). Ledger
  `v3_runs/gpu_ledger.jsonl`. No flag raises it.

**Runtime gate (G6) and reductions — decided once, before any test problem of any model.**
- G6 runs on the first model's calibration problems. It projects all four models with the measured Qwen times and the
  E1L/E1 Llama/Qwen time ratio (0.616).
- If the projection exceeds the remaining budget, the reductions below apply in order, **uniformly to all four models**:
  1. drop **CV** (−1.17 h);
  2. drop **C2** (−1.17 h);
  3. test n 200 → 160 (the first 160 of the fixed order).
- The chosen design is written to the lock appendix before any test problem. It is not revisited afterwards.
- If the projection still exceeds after step 3: stop and ask the authors.
- A per-model runtime drift of more than 15 % above projection triggers a stop-and-report. It does not trigger a new
  reduction, because reductions are fixed before the first test problem.

**Run order and truncation.** Qwen2.5-0.5B → Llama-3.2-1B-Instruct → Qwen2.5-0.5B-Instruct → Llama-3.2-1B.
- Once, before the first model: the GPU mutation check (tiny models of both families; it is not model-specific).
- Per model: gates → calibration → calibration analysis (including the §5 H-D rule) → append the hash to the lock → G6
  (first model: fixes and locks the design) → test.
- If the ceiling truncates the run: the completed problems of the fixed order are used, and a model with fewer than
  100 is unresolved. Family X requires all four models; otherwise it is reported as not estimable.
- **No analysis of test records before all runs finish.**

**Gates (per model; any failure ⇒ stop and report; none relaxed).**
- **G1 — α = 0 bitwise.** Every perturbation arm (C1, C2, QA, QO, CV, NAT-K\*, NAT-V\*, C1-K, C1-V) at
  scale 0 equals pure FP16 bitwise: tokens + cache, 10 train problems × 64 steps.
  - For NAT-K, NAT-V, LOC and INH, the FP16 pass after the measurement pass gives logits bitwise equal to pure FP16
    with the same cache. This proves the routing overrides are reset.
- **G2 — norms** (amendment 2, §16): construction check and rounding check pass for every arm and path.
- **G2b — subspaces.** Before rounding: ‖(I−P)δ_QA‖/‖δ_QA‖ ≤ 1e-5 and ‖Pδ_QO‖/‖δ_QO‖ ≤ 1e-5; ‖U_rᵀU_r − I‖ ≤ 1e-6;
  for r = 8.
- **G3 — finite.** Finite values; step-1 JS = 0 in every arm.
- **G4 — seeds.** Same seed ⇒ identical; different seed ⇒ different.
- **G5 — engine and histories.** NAT equals the engine's cache-only tokens (8 test problems). Qwen-base histories vs
  the stored E1 histories are reported.
- **G6 — runtime**, as above.
- **G7 — geometry** of the live cache.
- **G8 — routing.** The LOC/INH per-module precision log equals the declared map, and the overrides are reset.
- **GPU mutation check.** All Phase 3 core mutants, on CUDA, with a tiny model of each family; run once before the first
  model (`tools/v3_gpu_mutation_check.py`).
- **Per-problem integrity** (stop on failure): step-1 JS = 0; finite; non-negative; construction and rounding checks
  (§16) in all arms; routing reset; shared first token.
- **Hardware stop:** 88 °C.

## 10. CPU validation planned for Phase 3 (before the lock)

- **New modules only** (`rv3/…`, `run_v3.py`, `analyze_v3.py`). They import the engine and E1 modules unchanged; locked
  files are hash-checked by a test.
- **Tests (tiny Qwen2 and tiny GQA Llama, plus a real-model CPU smoke per family):**
  - α = 0 bitwise for every arm;
  - realized norms within FP16 rounding for every arm and path;
  - QA/QO orthogonality and projector properties; E₈ and r₈₀ computation;
  - the ρ\* and scale-factor computation, and the H-D extrapolation rule;
  - LOC/INH routing module by module, and reset;
  - only the newly written entry changes; one-path arms leave the other path bitwise κ^F;
  - seeds are label-only;
  - each predictor matches a slow float64 loop reference (1e-6 relative);
  - the post-RoPE query capture matches hand-computed q_proj + rotary.
- **Mutants (each must be caught):**
  - QA/QO subspace swapped; P from the wrong (layer, head); r off by one; r₈₀ computed from the wrong quantity;
  - K/V swapped in NAT-K/NAT-V, NAT-K\*/NAT-V\*, C1-K/C1-V;
  - ρ\* as an absolute norm; s_K and s_V swapped;
  - LOC/INH maps swapped; override not reset;
  - norm over KV heads instead of per head;
  - predictor off by one (current entry in J_t); pre-RoPE queries; attention over unperturbed keys; GQA group map
    shifted;
  - perturbation applied to all past entries; seed depending on data; CV profile from test problems;
  - E1's 19 mutants re-run.
- **Analysis on simulated truth:**
  - type-I error of the θ contrasts and of the per-model H-A test;
  - Holm over P, F_m and X;
  - equivalence logic;
  - H-C: β₀/β₁, f_m and F recover a planted confounder (substantial band) and show the "No" band when it is absent;
    the precondition logic;
  - the H-D extrapolation rule;
  - the 2×2 contrasts recover planted effects; the 25 % SD guard.
- **CPU harness** (`tests/cpu_harness_v3_paths.py`): the driver's gate, calibration, G6, Tier A and analysis functions end to end on CPU
  (tiny model, real tokenizer and prompts, temporary folder).
- **Lock manifest:** SHA-256 of this protocol, the code, tests, the sample file and the model snapshot hashes. Same
  procedure as E1/E1L.

## 11. Records

`v3_runs/<model>/{calib,tierA}/<arm>.jsonl` (`*` in an arm name is written as `star` in file names, e.g. `NAT-Kstar.jsonl`). Each record holds the per-step js, flip, P_norm, P_logit, P_val,
P_anormK and P_out, the per-path norm diagnostics, wall seconds and the routing check.

## 12. Deviations and AI use

- **Deviations:** every change after the lock goes to `v3/DEVIATION_LOG_V3.md` with date and reason. Only the
  pre-declared reductions and the pre-declared descriptive relabellings (§5 H-D rule, §8.3/§8.4 preconditions, the
  §8.9 25 % guard) may alter what is tested.
- **AI use:** each AI model, version and date is recorded in `v3/AI_USE_LOG_V3.md`.

## 13. Stated limitations (pre-declared; carried into the manuscript)

- **L1 — no chat template.** The instruct models receive the raw 3-shot completion prompt, as in E1L, so the prompt is
  identical across the 2×2. The instruct models therefore run in a format they were not tuned for. The tuning contrast
  measures base vs instruct weights *under this prompt*, not instruct models as deployed.
- **L2 — one model per cell.** Family and tuning contrasts compare four specific models. They are confounded with
  architecture, size, data and tokenizer.
- **L3 — teacher-forced, label-free outcomes.** Tier A measures next-token distribution change under a shared history.
  It says nothing about task accuracy, and the free-running behaviour of the new models is not measured.
- **L4 — first-order, plug-in predictors.** They use the realized (perturbed) attention, not an unperturbed reference.
- **L5 — LOC/INH are not additive.** LOC includes the within-step spread of the local error.
- **L6 — H-C is an adjustment, not mediation.** Shrinkage of a contrast after adjusting for P_logit does not prove that
  P_logit causes the contrast.
- **L7 — samples.** The 200 test problems have been used in earlier work (outcome-free here). The protocol is
  privately locked, not pre-registered, and exploratory in origin.

## 14. Decision record

r1 (commit c5cdf53) → r2, per the author decisions of 2026-10-29:

| # | Decision | Implemented in |
|---|---|---|
| 1 | H-A primary per model, Holm over the four; the pooled estimate is secondary | §8, §8.1 |
| 2 | H-A primary in NAT only; pooled norm-matched arms secondary; QA vs QO under H-B | §8.1, §8.2 |
| 3 | r = 8 primary; captured energy reported per model; 80 %-energy rule as sensitivity | §5, §8.2. **3a (r3):** the 80 % rule is diagnostic only (E₈ and r₈₀ per model, no arm, no test); the extra arms were removed; 21.0 h; reduction order CV → C2 → n160 unchanged |
| 4 | ρ\* = √(ρ_K ρ_V); per-model scale factors reported; H-D descriptive if a factor ∉ [0.5, 2] | §5, §8.5 |
| 5 | H-F a test across these four models (one per cell); interaction pre-declared; 25 % guard | §8.9 |
| 6 | Run order accepted; reductions uniform and fixed before any test problem | §9 |
| 7 | No chat template; stated limitation | §2, §13 L1 |
| 8 | H-C numeric decision rule with substantial / partial / no bands and preconditions | §8.3, §8.4 |
| 9 | Explicit falsification for every hypothesis | §8.1–8.9, summary table §8.10 |

| 10 (r3) | H-C cross-model precondition failure is reported as an abstract-level finding (the cross-session gap did not replicate in-session) | §8.4 |
| 11 (r4, pre-lock, implementation) | The GPU mutation check runs once, before the first model (it uses tiny models and is not model-specific); record files write `*` as `star` (Windows file names); the CPU harness of the driver's GPU code paths is part of the validation. No hypothesis, estimand, arm, sample or decision rule changed. | §9, §10, §11 |

Also changed in r2: the per-model family now has 7 tests (H-A moved to the primary family); H-C uses a shrinkage fraction
with a bootstrap p; the cross-model H-C is a single decision rule outside Holm.

## 15. Amendment 1 (2026-10-29, before any GPU use; author-requested)

Scope: record integrity and ledger accounting only. **No hypothesis, estimand, arm, sample, seed, predictor, calibration rule or decision rule
changed.** Original lock (id 169e756ffbcfb725) preserved at `validation_logs/V3_PROTOCOL_LOCK_original_169e756f.txt`.

1. **No duplicate records.** Tier A now writes only the arms of a problem that are not yet on disk, the rule calibration already used. An
   interruption while a problem's records are being written therefore cannot create a duplicate. On resume, only the missing arms are written.
2. **Exactly once.** The analysis loader fails loudly if any (model, arm, problem) appears more than once.
3. **Torn last line: pre-declared rule.**
   - Before every calibrate or tierA process, the driver checks every record file of the model. If one has an incomplete trailing line (no final
     newline) or an invalid line, the driver **stops**.
   - The incomplete trailing line is removed **only** by `tools/v3_repair_torn_line.py`, never by hand. The script:
     - refuses intact files and files with an invalid earlier line (that is not a torn write: stop and report);
     - keeps the removed bytes;
     - logs SHA-256 before and after in `v3_runs/repair_log.jsonl`;
     - appends a deviation row to `v3/DEVIATION_LOG_V3.md`.
   - The driver then reruns the problem: its already-written arms are kept, and the missing arm is recomputed with the same label-only seed.
   - Each event is reported as a deviation.
4. **Ledger heartbeat for the GPU mutation check.** It now runs inside `run_v3.ledger_heartbeat`, which:
   - charges an earlier unclosed process;
   - refuses at the 25 h ceiling;
   - writes the Stage 1 progress file every 15 s, so a hard kill is charged by the next GPU process;
   - closes with one ledger line.

Validation after the amendment: the full CPU validation was rerun (tiny suites, real-model checks for all four models, analysis tests, harness,
repair-script tests, crash-injection tests A–G, and the v3, E1 and E1L mutation suites, including the new amendment mutants).

## 16. Amendment 2 (2026-10-29, after the qwen-base gate failure, before any further GPU use; author-requested)

Trigger: qwen-base G2/G3 failed on CV only (per-slot maximum 0.101 against 0.10; mean 0.0016). DEVIATION_LOG_V3 V3-D04 records the
decision rule that was declared before the diagnosis, and the diagnosis (`v3/phase4/DIAG_V3_D04.md`, CPU only):
- there is no construction defect (every arm meets its target to ≤ 1e-5 before rounding);
- the deviation is FP16 rounding of Qwen layer-0 K head-0 entries (‖κ^F‖ ≈ 250, target at or below 2⁻¹⁰‖κ^F‖);
- every isotropic arm is affected, and E1 qwen-base C1 reached 0.099 under the same rule.

The previous lock (id 31617d53579174c2) is preserved at `validation_logs/V3_PROTOCOL_LOCK_amend1_31617d53.txt`.

**Unchanged:** every hypothesis, estimand, arm, sample, seed, predictor, calibration rule, decision rule and family.

**Changed (applied to ALL arms):**
1. **Construction (float64, before rounding).** Each stored entry is built in float64, u = κ^F + δ, using the same float32 random draws as
   before, cast exactly. For C1 and C2 this is the E1 construction (same draws) in float64. Per slot: |‖u − κ^F‖ − target| / target ≤ 1e-4,
   and u − κ^F = 0 exactly where the target is 0. This is a gate check (G2) and a per-problem integrity check.
2. **Rounding (bitwise, no tolerance constant).**
   - The stored entry is the correctly rounded FP16 (nearest, ties to even) of u, computed in one step.
     torch's float64 → float16 conversion goes through float32 and can round twice, so it is not used.
   - At every step, the entry read back from the cache must equal an independent implementation of that rounding, bitwise, and be FP16.
   - This is a gate check (G2) and a per-problem integrity check.
3. **The post-rounding per-slot norm limit is removed** as a gate and integrity criterion. Instead the realized post-rounding relative
   deviation |r − t| / t (perturbed slots with t > 0) is **reported** per arm and model:
   - mean;
   - p99, an upper bin edge with 40 bins per decade, i.e. within 6 %;
   - max;
   - the slots dominating steps that contain a slot above 5 %.
   The E1 floor-rule values (target + 2⁻¹⁰‖κ^F‖) are kept in the records as a descriptive diagnostic only.
4. **Pre-declared sensitivity (EXPLORATORY).** The primary and secondary contrasts are re-estimated after excluding what slots with
   realized deviation > 5 % can reach.
   - A stored entry cannot be removed from the cache after the fact: the entry written at step s enters the logits from step s + 1 on. So
     "excluding slots" is operationalized as **truncation**: for each problem and contrast, keep steps 1…f, where f is the first step whose
     write has a slot above 5 % in any arm of the contrast (all steps if none).
   - Problems with fewer than 8 retained steps are dropped and counted.
   - Re-estimated:
     - H-A in NAT. NAT has no constructed slots, so it is unchanged by construction; reported anyway.
     - All F_m θ contrasts and the descriptive θ contrasts.
     - H-C (NAT and C1).
     - The pooled norm-matched H-A secondary.
     - H-F on the truncated per-problem θ(NAT, C1).
   - Each is reported with the number of problems kept and dropped and the median retained fraction.
   - The sensitivity uses its own random stream, so no primary or secondary estimate changes.
5. **Mutation suite additions** (all must be caught; all existing mutants kept):
   - the pre-rounding perturbation scaled by 1.001 (target unchanged);
   - the per-slot norm scaled by 1.001;
   - rounding by truncation, through float32, and with ties away from zero;
   - rounding skipped;
   - (analysis) the sensitivity cut off by one step.
6. **Manuscript correction.** The E1/E1L tolerance was target + 2⁻¹⁰·‖entry‖, not max(target, …). It is corrected in `main_v3.tex` and
   logged as a correction.

Validation after the amendment: the full CPU validation was rerun:
- the tiny suites, including new tests v15 (rounding against an independent oracle, Python `struct`, on random values, signed zeros,
  subnormals, exact ties and double-rounding traps) and v16 (the runtime checks pass on correct runs and catch a 1.001 construction, a
  one-step rounding error and a non-FP16 store);
- the real-model checks for all four models;
- the analysis tests, including new a13;
- the harness, the repair tests and the crash-injection tests;
- the v3, E1 and E1L mutation suites.
