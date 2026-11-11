# EXPLORATORY_PLAN_V3 — dated list of exploratory analyses (v3, after Phase 5)

Dated **2026-10-31**. Branch `v3-novelty`. Lock 7325f563ce2978d8. The plan is unchanged by this file.

**Renaming (2026-10-31, Phase 7a).** The exploratory analyses were first labelled E1, E2 and E3. They are now X1, X2 and
X3, because "E1" also names the earlier perturbation-matched study. Only the labels changed: the content, formulas and
seeds are the same. The original wording is in commits 4b1ca9d and 4692c30.

## Rules for this file

- **Every analysis here is EXPLORATORY.**
  - None is in the locked protocol. None replaces or re-reads a locked result.
  - Each is reported with that label.
- **No other exploratory analysis is run without a new dated entry in this file**, committed before it is computed.
- **These analyses are not blind.**
  - This plan was written after the Phase 5 results were reported to the authors (PHASE5_REPORT.md, commit 1544dd0).
  - Some inputs are already known: the per-path provenance estimates, NAT-K − NAT-V, NAT − C1, and the P_val-type
    H-A descriptive values.
  - The formulas below were fixed without computing any X-quantity. They were not chosen by trying alternatives.
- **Conventions are those of the locked analysis.**
  - Code: `analyze_v3.py` primitives, imported unchanged. These are the problem-clustered percentile bootstrap (10,000
    resamples), the two-sided paired sign-flip test (10,000), the verdict rules and Holm.
  - Data: the locked records `v3_runs/<model>/tierA/*.jsonl`, n = 200 per model.
  - Floors: per-problem mean JS floored at 1e-9.
  - Each analysis has its own seed: X1 20261201, X2 20261202, X3 20261203.
  - Holm runs across the four models within each analysis (and within each part of X2). It is labelled
    exploratory and is not a confirmatory family.

Notation:
- m_i^a is the mean JS over steps 1…S_i of arm a on problem i.
- θ_i(a, b) = log m_i^a − log m_i^b, natural log.

## X1. K/V decomposition of NAT − C1

**Question.** Per model: do the K-path provenance θ_i(NAT-K, C1-K) and the V-path provenance θ_i(NAT-V, C1-V), each
weighted by that path's natural contribution, predict the sign and size of the combined θ_i(NAT, C1)?

**Weighting formula** (fixed here, before computing):

| Quantity | Definition |
|---|---|
| Natural K share | w_K,i = m_i^{NAT-K} / (m_i^{NAT-K} + m_i^{NAT-V}); natural V share w_V,i = 1 − w_K,i. "Natural contribution" is the mean JS that each path's natural NF4 error produces on its own. |
| Predicted θ | pred_i = w_K,i · θ_i(NAT-K, C1-K) + w_V,i · θ_i(NAT-V, C1-V) |
| Observed θ | obs_i = θ_i(NAT, C1) |

**Reported per model:**
- mean w_K;
- mean pred_i [95 % CI], and mean obs_i [95 % CI];
- the difference d_i = obs_i − pred_i: mean [95 % CI], 90 % CI, sign-flip p, Holm p over the four models, and the
  verdict word under the locked rule (difference / equivalent with the model's margin / inconclusive);
- **sign** — "sign predicted" if mean pred and mean obs have the same sign and both 95 % CIs exclude 0;
- **size** — the d verdict;
- descriptive: the per-problem Spearman of pred_i with obs_i (bootstrap CI), and whether the four models' mean pred
  order matches their mean obs order.

The weighted mean of log ratios is a heuristic, not an identity: there is no additivity assumption under which it equals
θ(NAT, C1) exactly.

## X2. H-A with P_val and with a combined predictor

The H-A procedure is reused unchanged: within NAT, steps 2…S_i, average ranks, problems with S_i < 8 excluded and
counted, D = mean_i [Spearman_t(P_x, JS) − Spearman_t(P_norm, JS)].

| Part | Predictor P_x |
|---|---|
| X2a | P_x = P_val. This reproduces the locked descriptive "HA-type P_val vs P_norm (NAT)", so the estimates are identical and only the CIs and p differ by seed. It is reported here with Holm over models. |
| X2b | P_x = P_comb, with P_comb,t = rank_t(P_logit) + rank_t(P_val): within-problem average ranks over steps 2…S_i, summed. The ranks make the combination scale-free; the raw sum P_logit + P_val is not, because the two predictors have different units (logit vs value-vector norm). The raw sum is not run. |
| X2c | the paired difference D_comb − D_logit per problem: does adding P_val change P_logit's ranking of NAT's per-step JS |

**Reported per model, for each part:** estimate [95 % CI], 90 % CI, p, Holm p over the four models, and the verdict
(difference / inconclusive; no equivalence category, as for H-A). Also reported: the pooled mean over the four models,
bootstrapped over the shared problems.

## X3. H-A excluding layer 0

**Question.** H-A recomputed with P_logit and P_norm summed over layers 1…L−1 only. The reason: Qwen's layer-0 keys
have norm about 250.

**Status: NOT COMPUTABLE from the locked records. Not run.**
- `rv3.core.PredictorState.evaluate` sums P_logit, P_val and P_norm over all layers before they are written. The
  records hold no per-layer value.
- It would need a new GPU pass of the NAT arm that records per-layer predictor terms. That needs new, unlocked code,
  GPU time inside the locked 25 h ceiling (1.254 h remain), and an author decision and authorization.
- A dated amendment to this file must record that decision before anything is computed.

## Confirmations (before running)

1. **Units of H-B.**
   - θ_B = mean over problems of [log m_i^QA − log m_i^QO]: the natural log of the per-problem mean JS over steps 1…S_i,
     floored at 1e-9.
   - So H-B estimates are in log mean-JS units: +3.47 means the geometric mean over problems of m^QA / m^QO is
     e^3.47 ≈ 32.
2. **QA/QO norm matching** (`rv3/core.py`, `stored_entry_v3`).
   - The key error is matched per slot (layer, KV head) to ‖Δ_K‖ of that arm's own measurement pass at that step. QA
     points it into the top-8 query subspace and QO into its orthogonal complement.
   - The value error is ‖Δ_V‖ times the same isotropic draw in both arms (shared seed label `QAQO-V`).
   - Construction (float64, ≤ 1e-4) and rounding were checked in every slot and step.
   - The two arms run separate trajectories, so after step 1 their Δ come from different cache states. In the records:
     - at step 1 the targets are identical;
     - over all steps, QA's mean target norm exceeds QO's by log ratio +0.029 (qwen-base), +0.011 (llama-instruct),
       +0.032 (qwen-instruct) and +0.013 (llama-base);
     - the stored ‖δK‖² log ratios are +0.061, +0.023, +0.067 and +0.028.
   - This is matching per slot to the arm's own natural error, as specified. It is not equality of norms across arms.

## Dated entry 2026-10-31 (after X1/X2 were reported): author decisions

- **X3 is dropped and was not run.** The per-layer predictor terms were not recorded: `PredictorState.evaluate` sums
  over layers before writing. Whether Qwen's H-A result is driven by layer 0 is listed as an **open question** in the
  manuscript. No GPU rerun.
- **X2b stays as the rank sum.** No raw-sum variant (P_logit + P_val) is run.
- No other exploratory analysis is added.

## Dated entry 2026-10-31 (Phase 7a, response to an internal hostile review): X4–X8

Committed before any X4–X8 quantity was computed. The rules at the top of this file apply unchanged. Every quantity
below is EXPLORATORY, is computed from the locked records `v3_runs/<model>/tierA/*.jsonl` only, and replaces no locked
result.

**What is already known (not blind).** The review read the locked descriptive values in Table A.10. In particular,
D(P_anormK) and D(P_logit) are both known, and both use the same problems and the same P_norm baseline. So the X5 point
estimate (the mean of ρ(P_logit) − ρ(P_anormK)) already follows exactly from them: about +0.02 to +0.03 per model.
Only its interval and p are new. Nothing else here has been computed.

**Conventions.**
- `analyze_v3.py` primitives are imported unchanged: `summarize` (10,000 problem-clustered percentile bootstrap
  resamples; two-sided paired sign-flip test, 10,000), `holm`, `rankdata`, `spearman`, `logm`, `theta`, `hc_stats`,
  `hc_betas` and `band`.
- Floors: per-problem mean JS at 1e-9; step-level JS at 1e-12 and P at 1e-30 (the locked H-C floors).
- JS is in nats.
- Seeds: X4 20261204, X5 20261205, X6 20261206, X7 20261207, X8 20261208.
- "Holm over four models" means Holm across the four per-model p-values of that analysis, labelled exploratory.
- Within NAT, the step range and exclusions are those of H-A: steps 2…S_i; problems with S_i < 8 are excluded, and
  so is any problem whose Spearman is undefined (a constant series). All exclusions are counted and reported.
- Predictor definitions are those of `rv3.core.PredictorState.evaluate`, which sums over layers and averages over the
  L·H_q query heads:
  - P_logit = Σ_j p_j |q·δk_j| / √d;
  - P_anormK = Σ_j p_j ‖δk_j‖, the same attention weights without the query direction;
  - P_val = ‖Σ_j p_j δv_j‖;
  - P_norm = the mean over slots of the RMS ‖δX‖ over written positions.

### X4. Spearman levels of each predictor within NAT

- **Quantity.** Per model and per problem i: ρ_i(P_x) = Spearman over steps 2…S_i of (P_x,t, JS_t) in the NAT arm,
  for P_x ∈ {P_norm, P_logit, P_val, P_anormK}.
- **Reported per model and predictor:**
  - the mean over problems [95 % CI];
  - the median;
  - the share of problems with ρ_i > 0;
  - the number of problems excluded.
- **Descriptive only.** No test and no Holm: the question is the level, not a difference from 0.

### X5. Paired P_logit − P_anormK

- **Quantity.** d_i = ρ_i(P_logit) − ρ_i(P_anormK), in NAT, over the problems where both are defined.
- **Question.** Does the query direction add to attention-weighted key-error size when ranking per-step JS?
- **Reported per model:** the estimate [95 % CI], 90 % CI, sign-flip p, Holm p over the four models, and the verdict.
  The verdict is difference / inconclusive, with no equivalence category (as for H-A).
- **Also reported:** the pooled mean over the four models (the per-problem mean over models, bootstrapped over the
  shared problems), as in X2.

### X6. Absolute divergence levels

For every model and every one of the 14 test arms, over the 200 problems:
- the arithmetic mean of m_i^a [95 % CI];
- the geometric mean exp(mean_i log m_i^a) [95 % CI];
- the median of m_i^a;
- the number of problems at the 1e-9 floor.

Here m_i^a is the per-problem mean JS over steps 1…S_i. This is the quantity behind every locked contrast, but it is
not floored for the arithmetic mean. Descriptive; no test.

### X7. H-C diagnostics and a joint, rank-based adjustment

**X7a — diagnostics of the locked H-C adjustment.** These are computed from the unchanged `hc_stats` sums (y and x as
locked; x uses P_logit), with problem-clustered bootstrap weights as in `hc_model`. Reported per model:
- β0 and β1, each [95 % CI];
- the slope γ [95 % CI];
- x̄ = Σx / Σn, the pooled step mean of x [95 % CI];
- descriptive: the share of steps with x < 0;
- descriptive: x̄_val, the same mean for log P_val^NAT − log P_val^C1.

**X7b — joint adjustment for P_logit and P_val, rank-based.** The formula is fixed here.
- For problem i and predictor P ∈ {P_logit, P_val}:
  - pool the 2(S_i − 1) values {P_t^NAT, P_t^C1 : t = 2…S_i};
  - compute average ranks R over this pooled set;
  - define u_t^P = [R(P_t^NAT) − R(P_t^C1)] / (2(S_i − 1)).
- Properties: u is scale-free and lies in (−1, 1). It is 0 if and only if P_t^NAT = P_t^C1, and its sign is the sign
  of the raw difference.
- y_t is the locked step-level log JS^NAT − log JS^C1.
- Fit one pooled OLS over all steps of all problems: y = β1J + γ_L u^{P_logit} + γ_V u^{P_val}.
  - β1J is the NAT − C1 contrast at u^{P_logit} = u^{P_val} = 0, read as "at steps where both predictors are equal in
    the two arms".
  - The fit uses per-problem sufficient statistics (n, Σy, Σu_L, Σu_V, Σu_L², Σu_V², Σu_L u_V, Σu_L y, Σu_V y) and the
    same bootstrap weights.
- Reported per model:
  - β0 [CI] (the locked value), β1J [CI], γ_L [CI] and γ_V [CI];
  - the shrinkage S_J = |β0| − |β1J| [CI] with the locked bootstrap p rule;
  - f_J = 1 − |β1J| / |β0| [CI] and the locked band rule;
  - the locked precondition (95 % CI of β0 excludes 0). Where it is not met, the model gets estimates only, with
    "not applicable".
  - Holm over the models where the precondition is met (the bootstrap p of S_J).
- This is a heuristic adjustment, not a mediation analysis. A linear model in rank differences is not implied by any
  theory.

**X7c — control: the rank transform with P_logit alone.** y = β1R + γ u^{P_logit}, reported like X7b. Comparing it
with the locked β1 isolates the effect of the rank transform. Comparing X7b with X7c isolates the effect of adding
P_val.

**X7d — the cross-model decision pair under X7b.** For Qwen-base vs Llama-instruct, with the same problems resampled
for both models:
- g1J = β1J(qwen-base) − β1J(llama-instruct) [CI];
- F_J = 1 − |g1J| / |g0| [CI], with the locked band rule (the locked precondition on g0 was met).

### X8. INH and LOC shares of NAT's divergence

Per model:
- **R_INH** = exp(mean_i [log m_i^INH − log m_i^NAT]) [95 % CI]: the geometric-mean ratio, which is exp of θ(INH, NAT).
- **R_LOC**, the same for LOC.
- Descriptive:
  - the ratio of arithmetic means Σ_i m_i^INH / Σ_i m_i^NAT [95 % CI, problem bootstrap], and the same for LOC;
  - the median per-problem ratio m_i^INH / m_i^NAT;
  - the INH share of the two origin arms, Σ m^INH / (Σ m^INH + Σ m^LOC) [95 % CI].
- **Scope note, to be carried into any text.** In INH only the stored entries are written with k/v projections in
  FP16 and everything else in NF4. The logits, queries and tokens of Tier A remain FP16. INH is therefore the cache
  component of that configuration, not the deployed configuration itself.

### Not computable

All five analyses use only per-step fields that are present in every Tier A record (`js`, `S`, `P_norm`, `P_logit`,
`P_val`, `P_anormK`). If running the script shows that any part cannot be computed, it will be reported as not
computable and not approximated.

## Dated entry 2026-10-31 (Phase 7a, after X4–X8 were reported): X10

Committed before X10 is computed by a script. The rules at the top of this file apply. **Not blind:** the point
values of the observed ratios and of s^b̂ were already computed by hand from the X6 output and the calibration files,
and reported to the authors in the Phase 7a report. That is why this entry exists. Descriptive only, from the existing
records; no test and no Holm. Seed 20261210.

### X10. Path scaling of NAT-K* and NAT-V* against the calibrated dose slope

For each model and path P ∈ {K, V}:

| Quantity | Definition |
|---|---|
| Observed ratio | R_obs = exp(θ(NAT-P*, NAT-P)) = exp(mean_i [log m_i^{NAT-P*} − log m_i^{NAT-P}]), with the 95 % CI (`summarize`, problem bootstrap) mapped through exp; n = 200. |
| Slope benchmark | R_pred = s_P^{b̂}, where b̂ = `dose_slope_b_hat` and s_P = `scale_factor_K` / `scale_factor_V` from the model's frozen `calibration.json`. Also the range s_P^{b̂ ± 1.96·SE_b}, with SE_b = `dose_slope_problem_sd` / √`n_calibration_problems`. |
| Excess | E = ln R_obs − b̂ ln s_P, with a 95 % CI (the θ interval shifted by the constant b̂ ln s_P), and R_obs / R_pred = e^E. |

**Two caveats, to be carried into any text:**
1. b̂ is the slope of C1 (isotropic, both paths) over doses 0.5–2 of its natural per-slot norm, on the calibration
   problems. It is not a per-path slope.
2. NAT-P* does not scale Δ_P uniformly. `stored_entry_v3` sets each slot's error to ρ*·‖κ^F_P‖ along the natural
   direction, so a slot's effective scale is ρ*/ρ_slot (with ρ_slot = ‖Δ_P‖/‖κ^F_P‖). That scale equals s_P only for a
   slot at the calibration median. R_pred is therefore a uniform-scaling benchmark, not a prediction for NAT-P*. A
   large E says that NAT-P* departs from uniform scaling at the slope's rate, without saying why.

No other analysis is added by this entry.

## Dated entry 2026-10-31 (Phase 7b): X11

Committed before X11 is computed. The rules at the top of this file apply. Descriptive only; no test and no Holm.
Seed 20261212 (for the bootstrap intervals of the dilution indicator only). Not blind: the locked H-B estimates
(e^θ = 12.9–32.0) are known.

### X11. A first-order benchmark for H-B (QA vs QO) as a positive control

**Benchmark.** For a slot s = (layer, KV head), let:
- M_s be the frozen calibration query second moment (`calib/aux_state.pt`, key `M`; the one the frozen U₈ comes from);
- E8_s be the captured energy of its top-r eigenvectors (`calibration.json`, `E8_per_slot`), with r = 8 and d = 64.

A key error of norm a, uniform in direction within the top-8 query subspace (QA) or within its orthogonal complement
(QO), gives these expected squared first-order attention-logit perturbations over the slot's queries:
- QA: a² · E8_s · tr(M_s) / r;
- QO: a² · (1 − E8_s) · tr(M_s) / (d − r).

Summed over slots, the first-order K-path logit-variance ratio is

  B = Σ_s w_s · E8_s / r  ÷  Σ_s w_s · (1 − E8_s) / (d − r),

with energy weights w_s = tr(M_s) · Σ‖Δ_K,s‖². The second factor is the calibration NAT natural key-error energy of
the slot (`aux_state.pt`, key `d2`, K rows, summed over channels), because QA and QO are matched to ‖Δ_K‖ per slot.

**Reported per model:**
- B (primary);
- descriptive variants:
  - B_q, with w_s = tr(M_s) only;
  - B_1, with equal weights;
  - the median of the per-slot ratios (E8_s/r)/((1 − E8_s)/(d − r));
- the locked observed ratio e^θ(QA, QO) [95 % CI] next to B.

**Notes, to be carried into any text:**
1. **Dilution by the shared V draw.** QA and QO store the same isotropic value perturbation (shared `uV`). The
   observed JS ratio therefore mixes a K-path ratio near B with a common V-path term, which pulls it toward 1. As an
   indicator of that dilution, this entry reports g(C1-V) / g(QO): the ratio of geometric-mean JS of C1-V (isotropic
   V at its natural norm, keys FP16) to that of QO, with a problem-bootstrap 95 % CI, from the locked records.
   - This is an indicator, not a correction: C1-V uses a different draw and trajectory, and the paths need not add.
2. **First order only.** B ignores the softmax weights p, higher-order terms, and the propagation from attention
   logits to output JS. It is a benchmark for direction and order of magnitude, not a prediction of e^θ.
3. **Mismatch in sample and directions.** M_s and Δ_K come from the calibration problems and steps, not the test
   problems. The realized QA/QO directions are single Gaussian draws, not expectations.

No other analysis is added by this entry.

## Dated entry 2026-11-01 (Phase 7c): X12

Committed before X12 is computed. The rules at the top of this file apply. Descriptive only; no verdict word. Seed
20261213. **Not blind:** the point estimates follow from the locked per-model NAT − C1 estimates, which are known. Only
the intervals and p are new. The manuscript revision requires X12, because the review asked for the per-family tuning
effects behind the locked H-F interaction.

### X12. Per-family tuning effects (simple effects of the locked H-F decomposition)

From the locked per-problem θ_{m,i} = log m_i^NAT − log m_i^C1 (`v3_report_locked.json`,
`theta_NAT_C1_per_problem`), over the 200 shared problems:

| Effect | Definition |
|---|---|
| Qwen tuning effect | T_Q,i = θ_{Qwen-instruct,i} − θ_{Qwen-base,i} |
| Llama tuning effect | T_L,i = θ_{Llama-instruct,i} − θ_{Llama-base,i} |

**Reported:** for each effect, the mean [95 % CI], the 90 % CI and the sign-flip p (`summarize`), with Holm over the two
effects, labelled descriptive. The locked interaction equals −½(mean T_Q − mean T_L), which is a check.
