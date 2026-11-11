# FOLLOWUP_X13 — cross-position common mode, centred arms and norm-corrected rounding

Drafted 2026-11-01 (after Phase 8a) and finalized the same day with the authors' decisions D1–D8 (section 11). Branch
`v3-novelty`.

**Status: protocol final; not run.** The protocol, code, tests and subset are fixed by the X13 manifest
(`v3/phase8/X13_MANIFEST.txt`). No GPU step runs without the authors' authorization file (section 7), which only the
corresponding author creates (D8). The predictions and decision rules of section 8 were stated in the draft before any X13
data existed; finalizing changed no prediction or threshold. The only numbers taken from earlier runs are the X9 wall times
(section 7) and the locked CI half-widths read by the analysis in (d).

## 1. Status and scope

- **Exploratory.** X13 is not part of the locked v3 protocol (lock 7325f563ce2978d8) and changes no locked result. Every
  X13 result is labelled *exploratory*; its tests are exploratory tests with Holm over the models within each path, which
  is not a confirmatory family.
- **Not blind.** X13 was motivated by results already seen:
  - the locked provenance tests (NAT-K milder, NAT-V harsher than per-path isotropic noise);
  - X9a (the top-8 query-subspace deficit is only 0.7–5.3 %);
  - X4 (P_logit Spearman levels near 0 in Qwen);
  - the Qwen rounding issue of amendment V3-A2 (Table A.12, truncation sensitivity);
  - the second internal review, which proposed the cross-position common-mode account.
- **Records.** The locked records `v3_runs/<model>/tierA/*.jsonl` and the X9 records `v3_runs_x9/` are read only. X13
  writes only under `research_v2/v3_runs_x13/`.
- **Questions.**
  - (a) Does the natural cache error share a common component across decode positions?
  - (b) Does removing that component move each path toward its isotropic arm?
  - (c) Does a centred logit predictor or an energy-weighted query alignment describe the key error better than P_logit or
    the top-8 share?
  - (d) Does norm-corrected FP16 rounding change the Qwen isotropic contrasts?

## 2. Sample (decision D7)

- **Draw.** The X9 subset: `numpy.random.default_rng(20261209).choice(sorted(tierA_ids), 50, replace=False)`, sorted.
- **File.** `v3/phase8/x13_subset.json`, written by `run_x13.py subset`. The command recomputes the draw and refuses to write
  unless it reproduces `v3/phase7/x9_subset.json` id for id. The file records the ids, the seed, the SHA-256 of
  `v3_sample.json` (LF-normalized) and the exposed problems in the subset.
- **Histories.** Each problem's FP16 history is regenerated in session with the locked `fp16_history` and compared token by
  token with the locked `FP16_history.jsonl`. The arms then run on the locked history.

## 3. Arms (decisions D1, D2, D5, D6)

**Design (D6 = option A).**

| Model | Arms, in run order per problem |
|---|---|
| qwen-base, qwen-instruct | NAT, NAT-K-cen, NAT-V-cen, C1-nc, C1-K-nc |
| llama-base, llama-instruct | NAT, NAT-K-cen, NAT-V-cen |

Option A runs no per-path C2 arm and no rerun of a locked constructed arm.

| Arm | Stored K | Stored V | Seed stream |
|---|---|---|---|
| NAT | κ^N (bitwise natural) | κ^N | (no draws) |
| NAT-K-cen | κ^F + ‖Δ_K‖ (Δ_K − μ_K)/‖Δ_K − μ_K‖ | κ^F_V (bitwise) | (no draws) |
| NAT-V-cen | κ^F_K (bitwise) | κ^F + ‖Δ_V‖ (Δ_V − μ_V)/‖Δ_V − μ_V‖ | (no draws) |
| C1-nc | locked C1 construction, norm-corrected rounding | same | **locked label `C1`** |
| C1-K-nc | locked C1-K construction, norm-corrected rounding | κ^F_V (bitwise) | **locked label `C1-K`** |

**Notation.**
- Δ_t = κ^M_t − κ^F_t is the arm's own natural error at step t.
- A slot is (layer, K or V, KV head), and norms are per slot over the 64 coordinates.

**Common mode μ (D1 = default; D2 = per slot).**
- **Definition.** For problem i and every slot σ, μ_σ is the mean of the NAT pass's natural Δ over all decode entries the
  NAT pass writes for that problem (steps 1…T_i). It is taken from the NAT pass's predictor state, which holds δ = stored −
  κ^F = Δ for every write. NAT runs first for each problem, and μ is handed to the centred arms of the same problem through
  their context.
- **Non-causal.** μ uses later positions. This is acceptable for an intervention whose teacher-forced histories are shared;
  it is not a deployable method.
- **Own Δ, natural norm.** The centred arms subtract μ from **their own** Δ_t and re-norm to their own ‖Δ_t‖, so the
  per-slot norm stays natural. All layers are centred.
- **Degenerate slots.** If ‖Δ_t − μ‖ < 10⁻¹² ‖Δ_t‖, the slot stores the natural direction (δ = Δ_t), and the number of such
  slots is logged. A slot with Δ_t = 0 stores κ^F (target 0).
- **Residual common mode.** Because the centred arm's trajectory, and so its Δ, differs from NAT's, some common mode
  remains. This is the manipulation check of (b).
- **Rounding.** The centred arms are constructed arms and are stored with the locked correctly rounded FP16 conversion. In
  the Qwen models their layer-0 key slots can therefore show the same rounding deviation as C1-K. The deviation is logged
  signed (section 4) and handled by a descriptive sensitivity in (b).

**Norm-corrected rounding (C1-nc, C1-K-nc; D5 accepted).**
- **Construction.** The arm calls the locked `stored_entry_v3("C1" or "C1-K", …)` with the locked seed stream. Its float64
  pre-rounding entry u = κ^F + δ is therefore built exactly as in the locked arm, and the locked integrity fields are
  computed for the corrected entry against the natural target t = ‖δ‖.
- **Nearest.** r₁ = RNE(u) per slot (the correctly rounded FP16, `fp16_rne_reference`). Its relative residual is
  e₁ = |‖r₁ − κ^F‖ − t| / t.
- **λ-search.**
  - The search runs only for slots with t > 0 and e₁ > 10⁻⁴. All other slots keep r₁ (λ* = 1).
  - The grid is λ ∈ {1} ∪ {(2/3)·(9/4)^{k/4096}, k = 0…4096}, ordered by |ln λ| ascending. For each λ, r(λ) = RNE(κ^F + λδ).
  - The chosen λ* gives the smallest residual. On ties the earliest in the |ln λ| order wins, so λ = 1 wins every tie.
- **Greedy refinement.**
  - It runs for slots whose best relative residual is still > 10⁻².
  - Each move puts one coordinate on its adjacent FP16 value above or below, chosen to minimize the residual. A move is
    applied only if it strictly reduces the residual. At most 64 moves.
- **Guarantee.** The corrected residual is never larger than e₁ (λ = 1 is on the grid and moves only reduce it).
- **Locked step checks.**
  - The corrected entry is FP16 by construction, so the locked rounding check is applied to it with pre = the corrected entry.
  - The locked float64 construction check compares ‖pre − κ^F‖ with t, so for these two arms it measures the post-correction
    residual. The X13 integrity rule therefore replaces it with `nc_ok`: in every slot and step, the corrected residual is
    ≤ e₁.
  - The locked unsigned deviation fields (`rdev_*`) keep their locked meaning relative to t, so they are directly comparable
    with the locked C1 and C1-K records.

## 4. Logging

The locked record fields are logged unchanged. The additions below are excluded from the reproduction comparison.

**Every arm, per problem and slot (`slots/<arm>_<idx>.npz`, float64):**
- `normsK`, `normsV`: ‖δ_j‖ per decode entry [n, L, n_kv];
- `S1K`, `S1V`: Σ_j δ_j [L, n_kv, 64].

Here δ = stored − κ^F, which is the natural Δ in NAT. These give the common-mode share and its isotropic null.

**NAT, per problem (same npz):**
- `RN`, `RD`: the energy-weighted Rayleigh numerator Σ_t δk_tσᵀ M_σ δk_tσ and denominator Σ_t (tr M_σ/64)‖δk_tσ‖², per
  slot [L, n_kv].
- M is the calibration query second moment per (layer, KV head), from the frozen `v3_runs/<model>/calib/aux_state.pt`
  (`M`, [L, n_kv, 64, 64]). Its SHA-256 is part of the manifest.
- `rawK`, `rawV`: raw Δ (float32) for the first 3 subset problems of each model (audit only).

**Every arm, per step (JSONL; predictor arms):**
- `P_logit_c` (layer-summed) and `PL_logit_c` [S][L] (section 8c).

**Constructed arms, per step (JSONL):**
- the signed relative deviation (‖r − κ^F‖ − t)/t over the perturbed slots with t > 0: `sdev_sum`, `sdev_n`, `sdev_min`,
  `sdev_max`, `sdev_pos5` (slots above +5 %) and `sdev_neg5` (below −5 %);
- centred arms: `degenerate` (count);
- nc arms:
  - the same signed statistics for the nearest rounding r₁ (`nn_sdev_*`);
  - `lam_min`, `lam_max`, `n_searched`, `moves`, `cos_min` (the smallest cosine between r − κ^F and δ over corrected slots);
  - `nc_ok`.

## 5. Implementation

**Files** (all in `v3/phase8/`; no locked file is modified):
- `rv3x13.py`:
  - `stored_entry_x13` for the four new arms; every other arm calls the locked `stored_entry_v3`;
  - `nc_round` and `fp16_neighbors`;
  - `CentredPredictorState(PredictorState)`, whose `evaluate` returns the locked values unchanged (it calls the locked
    method) and adds P_logit^c, and whose `record_write` adds the Rayleigh sums;
  - `install()` / `uninstall()`, a per-arm step log and a registry of predictor states.
- `run_x13.py`: the driver (`subset`, `manifest`, `status`, `ledger`, `run`).
  - It has its own per-problem loop `run_arms`, the locked `run_problem` with two changes: the seed-label map for the nc arms,
    and μ handed from NAT to the centred arms.
  - It applies the X13 integrity rule and the bitwise NAT gate.
- `analyze_x13.py`: the analysis.
- `test_x13.py`: CPU tests on the tiny Qwen2 and GQA Llama of `run_v3_tests.py`, and with `--real` a few steps on the real
  models on CPU.
- `mutation_check_x13.py`.

**CPU tests** (each on both tiny families):
1. NAT and several locked arms give records bitwise equal to the locked records with the X13 classes installed;
   P_logit^c is finite.
2. Centred arms:
   - the pre-rounding per-slot norm equals ‖Δ‖ (construction ≤ 10⁻⁴);
   - the direction is that of Δ − μ;
   - the other path is bitwise κ^F;
   - with μ = 0 the arm equals NAT-P bitwise;
   - at α = 0 the cache equals FP16 bitwise;
   - the degenerate fallback works;
   - no integrity error.
3. μ handed to the centred arms equals the numpy mean of NAT's Δ over all decode entries, per slot.
4. Norm-corrected rounding:
   - the residual is ≤ the nearest residual in every slot;
   - λ* = 1 and no move where the nearest residual is ≤ 10⁻⁴;
   - the result is valid FP16;
   - `fp16_neighbors` is exact (no FP16 value strictly between);
   - λ* matches an independent loop over the grid (first minimum), and refined slots end at a local optimum before the
     move cap;
   - the search reduces the residual where nearest rounding is off;
   - the signed deviation statistics match an explicit computation;
   - C1-nc's step-1 pre-rounding entry equals the locked C1's under the same seed.
5. P_logit^c equals a finite-difference total variation of the attention distribution, with the prefill included, to
   ≤ 10⁻⁵ relative.
6. Rayleigh sums match an explicit numpy computation; isotropic synthetic errors give a ratio within 2 % of 1.
7. Common-mode share: an explicit numpy check; isotropic synthetic errors with fixed norms give a mean n·C within 2 % of 1;
   the analysis null has mean n·C within 2 % of 1.
8. Comparator, history check, authorization, manifest (LF-normalized, every listed file, the `aux_state.pt` hashes), subset
   identity with X9, and the runtime projection and reductions.
9. The X13 integrity rule:
   - the construction error is dropped only for the nc arms;
   - `nc_ok` false is an error;
   - centred-arm construction errors are kept.
10. The analysis verdicts, G and its bootstrap, the bands, the cross-model reading and the (d) rule on synthetic values.
11. An end-to-end `run_arms` on a tiny model: every arm runs, μ is set from NAT, the fields are present, and there is no
    integrity error.

**Mutants** (each must make at least one test fail):
- μ not subtracted;
- μ over the wrong axis;
- re-norm to ‖Δ − μ‖;
- the centred arm on the wrong path;
- the λ-search maximizing the residual;
- greedy refinement accepting moves that increase the residual;
- C1-nc on a new seed stream;
- the signed deviation logged unsigned;
- P_logit^c centred on decode positions only;
- the Rayleigh quotient with the wrong head's M;
- the common-mode share with ‖S1‖² instead of ‖S1/n‖²;
- a comparator with a tolerance;
- an authorization check that ignores the manifest id;
- a history check that always matches;
- a changed subset seed;
- an integrity rule that drops every construction error;
- G inverted;
- the (d) rule using the full CI width instead of the half-width.

## 6. Reproduction gate (bitwise; decision D4)

- **NAT.** On every problem run, X13's NAT record must equal the locked NAT record in every field except `wall_seconds` and
  the X13-only fields: exact float equality after the JSON round trip, the same NaN positions and the same key set. The
  regenerated history must equal the locked history token for token.
- **On any mismatch or integrity error, the model stops.** No further problem and no later model runs.
  `v3_runs_x13/<model>/REPRODUCTION_MISMATCH.json` reports the problem, the differing fields (first differing step, maximum
  absolute and relative difference), whether the tokens and flips still match, and the integrity errors. No tolerance is
  pre-approved.
- **Why the gate matters.** (b) and (d) pair in-session X13 arms with the **locked** records of NAT-K, NAT-V, C1, C1-K and
  C1-V on the same problems, as X9b paired with the locked NAT-K. The bitwise NAT gate is the evidence that the pipeline is
  deterministic across sessions. Without it, those pairings are not used.

## 7. GPU budget, ledger and authorization (decisions D6, D8)

**Inputs (measured in X9 on the same 50 problems).**

| Model | Mean wall time per arm-problem | Model load and 50 × per-problem overhead |
|---|---|---|
| qwen-base | 6.78 s | 211 s |
| qwen-instruct | 7.11 s | 218 s |
| llama-base | 4.40 s | 149 s |
| llama-instruct | 4.88 s | 177 s |

The projection adds 10 % to the arm time for the extra logging and the λ-search. Option A (5 Qwen arms, 3 Llama arms,
n = 50) projects to **1.70 h**.

- **Ceiling: 2.0 h, hard.**
  - The ledger is `v3_runs_x13/gpu_ledger.jsonl`, run with the Stage 1 `gpu_process` watchdog and recovery; the problem loop
    stops before the cap.
  - X13 does not draw on the v3 (25 h) or X9 (1.5 h) budgets. The project total becomes 24.92 h plus X13's measured hours.
- **Run order.** qwen-base, qwen-instruct, llama-base, llama-instruct. Each model starts only after the previous one has
  completed and passed the gate.
- **Reductions, by runtime only and never by outcomes.**
  - After the first 3 problems of qwen-base, the driver projects the total. The inputs are the measured per-arm time,
    per-problem overhead and model-load time; Llama is scaled by the locked `LLAMA_TIME_RATIO` (0.616) with 3 instead of 5
    arms; the total gets a ×1.05 margin.
  - If the projection exceeds 1.9 h, these reductions apply in order until it fits:
    1. n = 40 for the Llama models (the first 40 subset ids in sorted order);
    2. n = 40 for every model.
  - If it still does not fit, the driver stops and the authors decide.
  - The design is written to `v3_runs_x13/design_x13.json`.
- **Authorization.** `run_x13.py run` refuses the GPU unless all of the following hold:
  - `--authorized-gpu-run` is passed;
  - `research_v2/v3_runs_x13/AUTHORIZED_X13_BY_AUTHORS.txt` exists and contains the text `FOLLOWUP_X13` and the X13 manifest
    id;
  - the listed files still hash to that id;
  - the v3, E1 and E1L locks and the model's weight hash verify (`run_v3.locks_ok`);
  - the model is the next in the fixed order, every earlier model has completed, and no mismatch is on file.

  The corresponding author creates the authorization file (D8).
- **Manifest id.** The first 16 hex characters of the SHA-256 over these files, in order:
  - `FOLLOWUP_X13.md`, `rv3x13.py`, `run_x13.py`, `analyze_x13.py`, `test_x13.py`, `mutation_check_x13.py`,
    `x13_subset.json` (each LF-normalized and prefixed by its name and byte length);
  - then, for each model in run order, the line `aux_state <model> <sha256>` of its `calib/aux_state.pt`.

  The id is written to `v3/phase8/X13_MANIFEST.txt` and committed.

## 8. Predictions, estimands and decision rules (stated before any X13 data)

**Shared conventions.**
- **Statistics.** The `analyze_v3.py` primitives, unchanged: problem-clustered percentile bootstrap (10,000) and two-sided
  paired sign-flip test (10,000).
- **Seeds (D7).** Every X13 random stream is seeded from 20261209: the bootstrap and sign-flip stream with
  `default_rng(20261209)`, and the isotropic null of (a) with `default_rng([20261209, 13])`. Option A has no new arm draws:
  the centred arms draw nothing, and the nc arms use the locked C1 and C1-K streams.
- **Problems.** n is the number of problems run. A model that failed the gate is reported as such and enters no estimate.
  There are no equivalence verdicts.
- **Holm.** Over the models run, separately for each path and test. It is exploratory.
- **Outcomes.** Log mean-JS outcomes use the locked definition, ℓ_i^a = log m_i^a with the 10⁻⁹ floor, over steps 1…T_i.

### (a) Cross-position common mode — TEST

**Estimand.** For problem i and path P ∈ {K, V}, the energy-pooled common-mode share over the slots σ of path P, from NAT:

C_i^P = Σ_σ ‖S1_σ / n_i‖² / Σ_σ (S2_σ / n_i), with S2_σ = Σ_j ‖δ_jσ‖².

**Isotropic reference.**
- If every δ_jσ kept its norm and had an independent uniform direction, E‖S1_σ/n‖² = (S2_σ/n)/n. The reference is
  therefore 1/n_i.
- For the finite-sample bias of the log, the analysis computes an empirical null per problem: the logged norms, seeded
  uniform directions, and 200 draws. ν_i is the mean over draws of ln(n_i C_i^null).

**Statistic.** x_i^P = ln(n_i C_i^P) − ν_i^P, and L^P = mean_i x_i^P, with a 95 % CI and a sign-flip p.

**Prediction.** The natural share is much larger than isotropic on both paths: L^K > 0 and L^V > 0, by a large factor.

**Verdicts per model and path (D3 as drafted):**
- *supported* if the 95 % lower bound of L^P is > ln 4 (at least 4× the isotropic share) and the Holm p is < 0.05;
- *contradicted* if the 95 % upper bound is < ln 2;
- *inconclusive* otherwise.

**Descriptive.**
- C^P per layer, pooled over problems.
- The energy share of the mean component, Σ n‖S1/n‖² / Σ S2, per layer.
- The same statistic for the stored δ of every constructed arm.

### (b) Removing the common mode — TEST

**Account under test.**
- **Keys.** A key error shared across decode positions shifts their logits together. That shift cancels in the softmax only
  *among the decode positions*. The prompt prefill is FP16 and error-free, so the shift relative to the prompt does not
  cancel. This partial cancellation is the reason the key prediction can fail even if (a) holds.
- **Values.** A shared value error adds coherently in Σ_j p_j δv_j, whereas independent errors partly average out.

**Prediction.** Removing the common mode makes the key path harsher and the value path milder, moving each natural path
toward its isotropic arm.

**Primary estimand per model and path.** θ_cen^P = mean_i (ℓ_i^{P-cen} − ℓ_i^{NAT-P}), with NAT-P the **locked** record
(section 6).
- Predicted signs: θ_cen^K > 0 and θ_cen^V < 0.
- **Gap closed.** G^P = mean_i (ℓ^{P-cen} − ℓ^{NAT-P}) / mean_i (ℓ^{C1-P} − ℓ^{NAT-P}), with C1-P the locked record. The
  denominator is positive for K and negative for V, so G^P > 0 means movement toward C1-P. The CI comes from a problem
  bootstrap of the ratio.
- **Manipulation check.** The statistic L of (a), computed for the centred arm's stored δ on path P. The arm *removed the
  common mode* if the 95 % upper bound of L_cen^P is < ln 2. Otherwise the (b) result for that model and path is reported
  descriptively only, and it does not count in the cross-model reading.

**Direction verdict per model and path:**
- *supported* if θ_cen^P has the predicted sign, its 95 % CI excludes 0 and its Holm p is < 0.05;
- *contradicted* if the 95 % CI lies entirely on the opposite side of 0;
- *inconclusive* otherwise.

**Size band (only when the direction is supported):**
- *most of the gap* if G^P ≥ 0.5 and the 95 % lower bound of G^P is > 0.2;
- *small part* if the 95 % upper bound of G^P is < 0.25;
- *partial* otherwise;
- *overshoot* is added if the 95 % lower bound of G^P is > 1.

**Reading across models, per path:**
- The common-mode account is *supported on path P* if the direction is supported, with the manipulation check passed, in at
  least 3 of the models run.
- It *accounts for most of the provenance gap on P* if, in addition, the band is "most of the gap" in at least 3 models.
- It is *contradicted on path P* if the direction is contradicted in at least 2 models.
- Otherwise the reading is *mixed*.

**Descriptive.**
- In the Qwen models, G^K with the in-session C1-K-nc in place of the locked C1-K.
- θ_cen^P on truncated steps: per problem, the steps before the first write of the centred arm with a slot above 5 %
  unsigned deviation, problems with fewer than 8 such steps dropped, both arms on the same steps. This uses the locked
  `first_unaffected` rule and is relevant only where the centred arm shows the Qwen rounding deviation.

### (c) Centred logit predictor and energy-weighted query alignment — DESCRIPTIVE

**Centred P_logit.** With dℓ_j = q_{t,l,g}·δk_{j,l,h(g)} for decode positions j ∈ J_t, and dℓ = 0 for every other stored
position (the prefill and the new position):

P_logit^c = (1/(L H_q)) Σ_{l,g} Σ_{j ∈ all stored positions} p_{t,l,g,j} |dℓ_j − Σ_i p_{t,l,g,i} dℓ_i| / √d.

This is the first-order total-variation change Σ_j |dp_j| of each head's attention distribution (times 1/√d). The scaled
logits move by dℓ·scaling, and the 1/√d keeps P_logit's normalization.

**Energy-weighted alignment.** For problem i,

R_i = Σ_{t,σ} δk_tσᵀ M_σ δk_tσ / Σ_{t,σ} (tr M_σ/d) ‖δk_tσ‖²

over NAT's stored key errors. Isotropic directions give R = 1 in expectation.

**Reported per model (no verdicts, no Holm):**
- Sp(P_logit^c) and Sp(P_logit), over steps 2…T_i with the H-A exclusions;
- D_c = mean_i [Sp(P_logit^c) − Sp(P_logit)], with a CI;
- mean ln R_i, with a CI and per layer, next to X9a's A = mean ln(φ_i/0.125) on the same problems (from the X9 records);
- the across-problem Spearman of ln R_i with the locked θ_i(NAT-K, C1-K).

**Expectations, stated for reading only.**
- D_c > 0, most clearly in Qwen.
- ln R is more negative than A.

### (d) Qwen: norm-corrected rounding — TEST

**Estimand.** For each of NAT−C1 and NAT-K−C1-K:

Δθ = θ_sub(NAT, C1-nc) − θ_sub(NAT, C1_locked) = mean_i (ℓ_i^{C1,locked} − ℓ_i^{C1-nc})

on the subset, paired by problem, and likewise for the K path.

**Reference half-width h.** The locked 95 % CI half-width of the same contrast on all 200 problems, read by the analysis from
`v3_report_locked.json` (NAT−C1: Qwen-base 0.066, Qwen-instruct 0.071; NAT-K−C1-K: 0.069, 0.079).

**Prediction.** The change is smaller than h.

**Verdicts per Qwen model and contrast (no adjustment across the two models):**
- *supported* if the 95 % CI of Δθ lies inside (−h, +h);
- *contradicted* if it lies entirely outside, that is, its lower bound is > h or its upper bound is < −h;
- *inconclusive* otherwise.

**Reading of an inconclusive result.** It is reported as **not dismissing the rounding confound**: X13 then neither confirms
nor removes a rounding contribution to the Qwen contrasts that use isotropic arms.

**Descriptive.**
- θ_sub(NAT, C1-nc) and θ_sub(NAT-K, C1-K-nc) next to the locked full-sample and subset values.
- The signed deviation before correction (nearest rounding, `nn_sdev_*`) and after it (`sdev_*`): the mean, and the share of
  slots above +5 % and below −5 %.
- The λ* range, the moves, and the smallest cosine.
- The number of steps with an unsigned deviation > 5 % in C1-nc, against the locked C1 on the subset.

**Power caveat.** C1-nc shares C1's draws at step 1 but not later, because the trajectories diverge. Δθ is therefore not a
pure rounding effect after step 1, and its CI on 50 problems may be as wide as h.

## 9. Power

- The locked n = 200 half-widths are about 0.07 log units for the provenance contrasts, so about 0.14 at n = 50.
- The locked provenance gaps are 0.74–1.23 log units. A centring effect that closes half the gap is resolved per model, and
  G^P is estimated to roughly ±0.15–0.2.
- Predictor differences of X5's size (≈ 0.025) are not resolved per model at n = 50.

## 10. Reporting

- Everything from X13 is *exploratory*: (a), (b) and (d) are exploratory tests, and (c) is descriptive.
- In the manuscript, X13 would appear in Section 5 and the appendix, never as a confirmatory result. Its verdict words follow
  the manuscript: a prediction is supported, contradicted or inconclusive.
- Deviations are logged as `X13-Dnn` in `v3/DEVIATION_LOG_V3.md`, and the AI-use log gets a row per session.

## 11. Author decisions (2026-11-01)

- **D1.** μ is the problem-level per-slot mean of the NAT pass's natural Δ (the default).
- **D2.** Centring is per slot, in all layers.
- **D3.** The thresholds are as drafted:
  - (a): ln 4 to support, ln 2 to contradict;
  - (b): the bands 0.5/0.2/0.25 and the 3-of-4 reading;
  - the manipulation-check bound ln 2.
- **D4.** (b) and (d) pair with the locked records under the bitwise NAT gate.
- **D5.** λ-search plus greedy refinement, with the 1 % refinement trigger.
- **D6.** Option A: no per-path C2, no reruns; 1.70 h projected, 2.0 h cap, and the reduction order of section 7.
- **D7.** Seed 20261209: the same subset as X9, and the seed of every X13 random stream (section 8).
- **D8.** The corresponding author creates the authorization file.
- **(d)** An inconclusive (d) is reported as not dismissing the rounding confound.

**Implementation correction (2026-11-01, before the manifest and before any X13 data).** The committed text (b772c41) listed a
CPU test that the norm-corrected residual is "never below the brute-force optimum over the floor/ceil neighbour sets". That
bound is wrong: the λ-search can reach FP16 values beyond the floor/ceil of u, so it can legitimately beat that optimum. The
test was replaced by an independent loop over the λ grid and a local-optimum check of the refinement (section 5, item 4).
No arm, estimand, threshold or decision rule changed.
