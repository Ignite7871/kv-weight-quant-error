# v4 exploratory plan (X1 to X4)

**Status:** exploratory. Nothing here is part of the locked v4 analysis (lock b0b191df9bee180e plus deviation D1). No result below
changes a locked verdict (H1 to H5). None of these entries is a hypothesis test, and none produces a p-value used for a claim.

**Not blind.** Every entry was written on 2026-11-09, *after* the locked report (`v4_runs/v4_report.json`, sha256 df57a314…) and
the independent recomputation (214/214 within tolerance) had been read. The author and I had both seen the per-model estimates of
H1, H2, H3, H5 and NAT-C1, the per-layer H5 descriptives, and the H4 result before choosing these analyses. The entries were chosen
to describe patterns already visible in those numbers, so any pattern they show is not independent evidence for that pattern.

**Inputs.** The test records under `v4_runs/<model>/test/` (the same 100 problems and arms as the locked analysis) and the locked
report itself. Per-problem contrast values are computed with the locked primitive `analyze_v4.theta` (= `analyze_v3.theta`)
and the locked `CONTRASTS` map, so a per-model mean below reproduces the locked estimate exactly. This is checked in the output.

**Code and output.** `v4/exploratory_v4.py` writes `v4_runs/v4_exploratory.json` and `v4/exploratory/X_RESULTS_V4.md`.
Exploratory seed: 20261109 (not the locked seed).

**Notation.**
- θ_{m,c,i}: the per-problem value of contrast c for model m and problem i, from `theta(A, a, b, idx)` (difference of log m between arms a and b).
- y_{m,c} = mean_i θ_{m,c,i}: this equals the locked estimate.
- Families: Qwen = {qwen2.5-0.5b, 1.5b, 3b, 7b}; Llama = {llama3.2-1b, llama3.2-3b, llama3.1-8b}. Mistral is a single model and gets no slope.
- Size: P_m = (total bytes of the pinned safetensors shards in `V4_MODEL_MANIFEST.json`) / 2, the 16-bit parameter count of the
  stored checkpoint. This is the same size quantity the locked H4 Spearman used. A tied embedding is counted once, as stored.
  x_m = log10 P_m.

---

## X1 (2026-11-09): size trend within family, H1 and NAT-C1 (descriptive)

**Question.** Within each family, how does the estimate change with model size?

**Formula.** For family F, with n_F models and contrast c ∈ {H1, NAT-C1}:

β_{F,c} = Σ_m (x_m − x̄)(y_{m,c} − ȳ) / Σ_m (x_m − x̄)²   (OLS slope of the estimate on log10 params, per decade of parameters)

**Bootstrap CI.** B = 10,000 draws. In each draw, resample the 100 problems with replacement *independently within each model*,
recompute every y_{m,c}, then recompute β_{F,c}. Report the 2.5 and 97.5 percentiles.

**What the CI covers.** Only the sampling of problems, given these specific checkpoints. It is not a CI over a population of models.
Llama has 3 points (1 residual df) and Qwen has 4 (2 df). A slope is a summary of 3 or 4 points, not a fitted law.

**Also reported.**
- The per-model y with the locked CI.
- The Pearson correlation of x and y per family, as a point value.
- For H1, the slope of |y|. A positive slope of y means H1 moves toward 0 as size grows, since H1 is negative here.

**No new test claims.** No p-values are computed for β.

## X2 (2026-11-09): per-layer H5 descriptives (summary only)

**Source.** `per_model[m].H5_C_per_layer_descriptive` in the locked report: C^P per layer for P ∈ {K, V}, pooled over problems.
Nothing is recomputed from records.

**Summary per model and path.**
- number of layers L;
- mean, median, min, max;
- the layer of the max (0-based, and as relative depth ℓ/(L−1));
- layer 0 value against the mean of layers 1 to L−1;
- the Spearman correlation of C^P with relative depth;
- the fraction of layers where C^V > C^K.

**Limits.** These per-layer values have no CI, no null and no test. They are on the C^P scale, not the locked H5 statistic
(ln(n C) − ν). The per-layer values have no null and no ν, so they are not head_dim-adjusted. Comparisons across models with
different head_dim are not adjusted for it (see X4).

## X3 (2026-11-09): H1 shrinkage against the NAT-C1 sign change (descriptive)

**Question.** Across models, is the move of H1 toward 0 aligned with NAT-C1 changing from negative to positive?

**Across models (n = 8, and n = 7 without Mistral).**
- Pearson r and Spearman ρ of (y_{m,H1}, y_{m,NAT-C1}), as point values only.
- With 8 models, no CI or p is reported, because there is no population of models to sample from.

**Within each model (problem level).**
- Pearson r of (θ_{m,H1,i}, θ_{m,NAT-C1,i}) over the 100 problems.
- A percentile bootstrap CI for r: B = 10,000 resamples of problems, seed 20261109.

The within-model correlation describes whether problems with a large H1 also have a large NAT-C1 in the same model. It says
nothing about the cross-model pattern. Both contrasts are built from overlapping problem runs, so a positive within-model r is
partly expected by construction and is not evidence of a mechanism.

**Reading.** Purely descriptive. A cross-model alignment would be confounded with size and head_dim (X4) and cannot separate them.

## X4 (2026-11-09): head_dim note

**Facts from the manifest.** The smallest model in each family has head_dim 64 (qwen2.5-0.5b, llama3.2-1b). Every other model
has 128, including Mistral.

**What this note states.** Which comparisons are confounded with the 64 → 128 change:
- (a) every within-family size trend (X1), because the smallest point differs in head_dim as well as size;
- (b) the NAT-C1 sign change: its only negative estimate is at head_dim 64, and the near-zero one is the other head_dim-64 model;
- (c) the H4 family comparison, because each family's head_dim-64 model is also its smallest;
- (d) cross-model magnitude comparisons of H1, H2 and H3 that involve the two small models;
- (e) H5, whose null ν depends on head_dim by construction (the isotropic null in R^head_dim). The locked statistic is adjusted
  for head_dim, but the per-layer C^P values of X2 are not.

**Computed, descriptive only.**
- X1's slopes restricted to the head_dim-128 models. For Qwen that is 3 points. For Llama it is 2 points, so the "slope" is the
  difference between the two divided by the gap in log10 P, reported with the same bootstrap.
- The difference in y between each family's head_dim-64 model and its next-larger head_dim-128 model, for H1, H2, H3 and NAT-C1.
  Size and head_dim change together in that step, so it cannot attribute the difference to either.

**Not possible with these data.** Separating size from head_dim. No checkpoint here varies one with the other held fixed. This
note only says which statements stay unaffected: the per-model signs of H2, H3 and H5, which are the same in every model at both
head_dims.
