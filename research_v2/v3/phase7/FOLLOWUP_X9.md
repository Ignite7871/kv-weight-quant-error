# FOLLOWUP_X9 — per-layer follow-up on the NAT and NAT-K* arms

Drafted 2026-10-31 (Phase 7b) and finalized the same day with the authors' decisions D1–D5. Branch `v3-novelty`.

**Status: protocol final; not run.** The protocol, code, tests and subset are fixed by the X9 manifest
(`v3/phase7/X9_MANIFEST.txt`). No GPU step runs without the authors' authorization file (section 7). Only the authors
create that file.

## 1. Status and scope

- **Exploratory.** X9 is not part of the locked v3 protocol (lock 7325f563ce2978d8). It changes no locked result, and
  every X9 result is labelled *exploratory*.
- **Not blind.** X9 was motivated by results already seen:
  - X4: the predictor Spearman levels are near 0 in Qwen;
  - X5: the query direction adds about +0.025 over P_anormK;
  - X6 and X10: NAT-K*/NAT-K is 73–95× in Qwen against a uniform-scaling benchmark of about 3.6×;
  - X11: the H-B benchmark;
  - the locked H-A split by family.
- **Records.** The locked records `v3_runs/<model>/tierA/*.jsonl` are read only. X9 writes only under
  `research_v2/v3_runs_x9/`.
- **Question.** Where in the network and in the head space does the natural key error sit? And is layer 0 behind the
  Qwen NAT-K* excess and Qwen's near-zero predictor ranks?

## 2. Sample

- **Draw.** 50 problems: `numpy.random.default_rng(20261209).choice(sorted(tierA_ids), 50, replace=False)`, sorted,
  from the 200 Tier A ids of `v3_sample.json`.
- **File.** `v3/phase7/x9_subset.json` holds the ids, the seed, the SHA-256 of `v3_sample.json` (LF-normalized), and
  the exposed problems it contains. It is written by `run_x9.py subset`.
- **Same subset for every model.** It is fixed before any GPU use and drawn without outcome data.
- **Histories.**
  - Each problem's FP16 history is **regenerated in session** with the locked `fp16_history`.
  - The regenerated history is compared token by token with the locked `FP16_history.jsonl`.
  - The arms then run on the locked history.

## 3. Arms (decision D1)

The three arms run per problem in this order: NAT, NAT-K*, NAT-K*-L0nat.

| Arm | Stored K | Stored V | Code path |
|---|---|---|---|
| NAT | κ^N (bitwise natural) | κ^N | locked `stored_entry_v3`, unchanged |
| NAT-K* | κ^F + ρ*‖κ^F_K‖ Δ_K/‖Δ_K‖ in every slot | κ^F | locked `stored_entry_v3`, unchanged; ρ* from the frozen calibration |
| NAT-K*-L0nat | layer 0: κ^N_K (natural, bitwise); layers 1…L−1: exactly the NAT-K* entry | κ^F | `rv3x9.stored_entry_x9` (section 5) |

- **Effective scale.** NAT-K* sets each key slot's error to ρ* times that slot's FP16 key norm. Its effective scale is
  therefore ρ*/ρ_slot, with ρ_slot = ‖Δ_K‖/‖κ^F_K‖, and not s_K.
- **What the new arm isolates.** NAT-K*-L0nat differs from NAT-K* only in layer 0's keys, so the contrast between the
  two isolates layer 0's share.
- **Not run.** There is no uniform-scaling arm (decision D1).

## 4. Logging

The locked fields are logged unchanged. In addition:

**Per step, per layer l, in every arm (JSONL, lists [S][L]):**
- `PL_logit`, `PL_val`, `PL_anormK`: the per-layer terms of `PredictorState.evaluate`, divided by the same
  denominator L·H_q. Their running sum, in layer order, is the locked summed predictor.
- `PL_norm`: the mean over (K/V, KV head) of √(cum2/n) for layer l.

**Per step, per slot (l, KV head), in every arm (`.npz` per model, arm and problem; float64):**
- `eK` = ‖δK‖², `pK` = ‖P₈δK‖², `eV` = ‖δV‖², `pV` = ‖P₈δV‖²;
- `fK` = ‖κ^F_K‖ and `fV` = ‖κ^F_V‖.
- δ = stored − κ^F, which equals the natural Δ in NAT. P₈ = U₈U₈ᵀ, with U₈ from the frozen `calibration.json` (the
  same U₈ as QA/QO).

## 5. Implementation

**Files** (all in `v3/phase7/`; no locked file is modified):
- `rv3x9.py`:
  - `PerLayerPredictorState(PredictorState)`. Its `evaluate` is the locked code with per-layer terms recorded; the
    returned values must be bitwise equal to the locked ones. Its `record_write` calls the locked method and then
    logs the per-slot energies.
  - `stored_entry_x9`. For NAT-K*-L0nat it calls the locked `stored_entry_v3("NAT-K*", …)` and then sets layer 0's K
    row of the stored entry and of `pre` to κ^M, and its target to ‖Δ_K‖ of that row. Every other arm goes to the
    locked function.
  - `install()` / `uninstall()`: process-local replacement of `rv3.core.PredictorState` and `rv3.core.stored_entry_v3`.
  - A registry of predictor-state instances.
- `run_x9.py`: the driver (`subset`, `manifest`, `status`, `ledger`, `run`).
- `analyze_x9.py`: the analysis.
- `test_x9.py`: CPU tests on the tiny Qwen2 and tiny GQA Llama of `run_v3_tests.py`; with `--real`, also a few steps on
  the real models on CPU.
- `mutation_check_x9.py`.

**CPU tests, before any GPU use:**
1. Every arm's full record with the X9 `PredictorState` is bitwise equal to the record with the locked one. Every
   per-layer sum equals the summed predictor to ≤ 1e-12 relative.
2. NAT-K*-L0nat:
   - layers ≥ 1 are bitwise equal to NAT-K*;
   - layer 0's K row equals κ^M bitwise;
   - V equals κ^F bitwise;
   - at α = 0 the arm equals FP16 bitwise;
   - every integrity flag is true, and `integrity_errors` is empty.
3. The subspace energies match an independent numpy implementation (explicit P = U Uᵀ) to ≤ 1e-12 relative. Σ eK per
   step is consistent with the locked `dn2_K` (≤ 1e-6 relative; float32 vs float64).
4. The reproduction comparator detects a 1-ulp change in any float field, a token change, NaN-position changes, and
   missing keys. It ignores `wall_seconds`.
5. The guards: the manifest uses LF normalization and changes when any listed file changes; authorization is refused
   without the file, the text `FOLLOWUP_X9` or the current manifest id; the subset is deterministic; the reduction
   projection follows section 7.
6. `analyze_x9.py` reproduces hand-computed values on synthetic records.

**Mutants** (13). Each must make at least one test fail:
- L0nat on the wrong layer;
- L0nat on V instead of K;
- the projector from the wrong layer;
- the projector from the wrong head;
- the summed predictor accumulated in a different order (Σ term/denom instead of (Σ term)/denom). This is caught by a test
  on 3-layer tiny variants, where L·H_q (12 or 24) is not a power of two. On the 2-layer tiny models L·H_q is 8 or 16,
  so the two orders give identical bits;
- per-layer P_norm grouped over the wrong axis;
- a comparator with a tolerance;
- a comparator blind to NaN positions;
- an authorization check that ignores the manifest id;
- energies computed from the stored entry instead of δ = stored − κ^F;
- an analysis share inverted;
- a history comparison that always matches;
- a changed subset seed.

## 6. Reproduction gate (bitwise; decision D4)

- **Coverage.** For NAT and NAT-K*, on every problem run, every field of the X9 record must equal the locked record
  exactly. `wall_seconds` is excluded. The comparison is exact float equality after the JSON round trip, with NaN in
  the same positions and the same key set.
- **History.** The regenerated FP16 history must equal the locked one token for token.
- **On any mismatch, the model stops.** No further problem and no later model runs. `v3_runs_x9/<model>/REPRODUCTION_MISMATCH.json`
  reports:
  - the problem;
  - every differing field, with its first differing step, maximum absolute and relative difference, and the number of
    differing steps;
  - whether `top` (the per-step argmax tokens) and `flip` still match;
  - whether the regenerated history matched the locked one.
- **No tolerance.** None is pre-approved. The authors decide what follows. No X9 estimate uses a model that failed the
  gate.

## 7. GPU budget, ledger and authorization (decision D3)

- **Estimate.** This uses the locked G6 per-problem overhead (2.7 s for Qwen, history plus reference) and the mean wall
  time per arm-problem (about 7.0 s for Qwen and 4.2 s for Llama).

  | Model | Arms × n | Estimate |
  |---|---|---|
  | qwen-base | 3 × 50 | 0.33 h |
  | qwen-instruct | 3 × 50 | 0.33 h |
  | llama-base | 3 × 50 | 0.20 h |
  | llama-instruct | 3 × 50 | 0.20 h |
  | Model loads and logging overhead | | about 0.15–0.25 h |
  | **Total** | | **about 1.2–1.3 h** |

- **Run order:** qwen-base, qwen-instruct, llama-base, llama-instruct. Each model starts only after the previous one
  has completed and passed the reproduction gate.
- **Ceiling: 1.5 h, hard.** The ledger is `v3_runs_x9/gpu_ledger.jsonl`, run with the Stage 1 `gpu_process` watchdog
  and recovery, and the problem loop stops before the cap. X9 is outside the locked v3 protocol. It does not draw on
  the v3 25 h ceiling (1.254 h unused) and does not extend it. The paper reports 23.75 h (v3) plus X9's measured hours.
- **Reductions.** After the first 3 problems of qwen-base, the driver projects the total:
  - the inputs are the measured per-arm time, per-problem overhead and model-load time;
  - Llama is scaled by the locked `LLAMA_TIME_RATIO` (0.616);
  - the total gets a ×1.05 margin.

  If the projection exceeds 1.4 h, these reductions apply in order until it fits:
  1. drop NAT-K*-L0nat in the two Llama models;
  2. n = 40 for the Llama models (the first 40 subset ids in sorted order);
  3. n = 40 for every model.

  The design is written to `v3_runs_x9/design_x9.json`. Only runtime decides it, never outcomes.
- **Authorization.** `run_x9.py run` refuses to use the GPU unless all of the following hold:
  - `--authorized-gpu-run` is passed;
  - `research_v2/v3_runs_x9/AUTHORIZED_X9_BY_AUTHORS.txt` exists and contains the text `FOLLOWUP_X9` and the X9
    manifest id;
  - the listed files still hash to that id;
  - the v3 lock and the E1/E1L locks still verify, together with the model's weight hash (`run_v3.locks_ok`).
- **Manifest id.** The first 16 hex characters of the SHA-256 over the following files, in this order: `FOLLOWUP_X9.md`,
  `rv3x9.py`, `run_x9.py`, `analyze_x9.py`, `test_x9.py`, `mutation_check_x9.py`, `x9_subset.json`. Each file is
  LF-normalized and prefixed by its name and byte length. The id is written to `v3/phase7/X9_MANIFEST.txt` and
  committed.

## 8. Expectations, estimands and decision rules

Shared conventions:
- The `analyze_v3.py` primitives are imported unchanged: a problem-clustered percentile bootstrap (10,000) and a
  two-sided paired sign-flip test (10,000).
- The analysis seed is 20261211.
- n is the number of problems run (50 unless reduced). A model that failed the gate or did not run is reported as
  such.
- `analyze_x9.py` is in the manifest, so it is fixed before the GPU run.
- There are no equivalence verdicts.

### (a) Natural key error is depleted in the query subspace — TEST

- **Expectation.** The natural Δ_K has less than 8/64 = 0.125 of its energy inside the pooled r = 8 query subspace.
  This would be consistent with NAT-K being milder than C1-K, whose isotropic direction puts 0.125 there in
  expectation.
- **Estimand (NAT).** F_i = Σ pK / Σ eK over all writes 1…S_i, layers and KV heads of problem i. Then
  A = mean_i ln(F_i / 0.125), with a 95 % CI, the sign-flip p, and Holm over the models run.
- **Verdicts:**
  - *supported* if the 95 % CI upper bound is < 0 and the Holm p is < 0.05;
  - *contradicted* if the 95 % CI lower bound is > 0;
  - *inconclusive* otherwise.
- **What would contradict the account.** A ≥ 0 means the query-subspace account cannot explain NAT-K being milder than
  C1-K in that model.
- **Reported descriptively:**
  - F per layer and per (layer, KV head), pooled over problems, with problem-bootstrap CIs;
  - F excluding layer 0;
  - the same quantities for Δ_V against 0.125. U₈ lies in query/key space and values are not dotted with queries, so
    for V this is a geometric description only;
  - the across-problem Spearman of ln F_i with the locked θ_i(NAT-K, C1-K). This is descriptive only: NAT-K's Δ comes
    from its own trajectory.
- **Limits.** This shows consistency, not explanation. Query energy outside the top 8 directions and the query
  magnitudes are not in F.

### (b) Layer 0 and the Qwen NAT-K* excess — TEST in the Qwen models (decision D1)

- **Primary estimand.** The share of NAT-K*'s excess over NAT-K, in log mean-JS, that keeping layer 0 natural
  removes:

  G = mean_i [ℓ_i^{K*} − ℓ_i^{L0}] / mean_i [ℓ_i^{K*} − ℓ_i^{K}], with ℓ_i^a = log m_i^a (the locked floor).

  - Here a = K* is the X9 NAT-K* run (bitwise equal to the locked record once section 6 passes), L0 is NAT-K*-L0nat,
    and K is the **locked** NAT-K record on the same problems.
  - The 95 % CI comes from a problem bootstrap of the ratio.
- **Verdicts per Qwen model** (CI-based as specified, with no multiplicity adjustment across the two models):
  - *supported* if G ≥ 0.5 and the CI lower bound is > 0.2;
  - *contradicted* if the CI upper bound is < 0.2;
  - *inconclusive* otherwise.
- **Descriptive:**
  - E_L0 = ln R_L0 − b̂ ln s_K, with R_L0 = exp mean_i(ℓ^{L0} − ℓ^{K}), and whether its 95 % CI lies inside
    (−ln 2, +ln 2), that is, within a factor of 2 of the dose-slope benchmark;
  - the same for NAT-K*.
- **Llama, as a control (descriptive only).** The expectation is G ≈ 0, reported with its CI.
- **Reading.** A remaining excess would point to the per-slot ρ* equalization in other layers, or to a key
  nonlinearity. Diagnostic (d) bears on the first.

### (c) H-A without layer 0 — DESCRIPTIVE (decision D5)

- **Quantities (NAT).** P_x^(−0) is the sum of `PL_x` over layers 1…L−1, for x ∈ {logit, val, anormK}. P_norm^(−0) is
  the mean of `PL_norm` over layers 1…L−1. The H-A step range and exclusions apply.
- **Reported per model, with 95 % CIs and no verdict or Holm:**
  - Δρ = mean_i [ρ_i(P_logit^(−0)) − ρ_i(P_logit)];
  - D^(−0) = mean_i [ρ(P_logit^(−0)) − ρ(P_norm^(−0))] next to D on the same problems;
  - the same for P_val and P_anormK;
  - the per-layer profile of ρ_i(P_logit^(l)).
- **Expectation, stated for reading only.** In both Qwen models, P_logit's rank rises without layer 0.

### (d) Effective NAT-K* scale per layer — DESCRIPTIVE

From NAT's logs: per layer, the median over steps, heads and problems of ρ*/ρ_slot, with ρ_slot = √eK / fK, next to
s_K.

## 9. Power

The X4 Spearman half-widths are about 0.03 at n = 200 and about 0.06 at n = 50, so effects of X5's size are not
resolved per model. The (a) and (b) effects, if they are of the size X10 suggests, would be.

## 10. Reporting

- Everything from X9 is *exploratory*: (a) and (b) as exploratory tests, (c) and (d) as descriptive.
- In the manuscript, X9 appears in the exploratory subsection and the appendix, never as a confirmatory result.
- Deviations are logged as `X9-Dnn` in `v3/DEVIATION_LOG_V3.md`, and the AI-use log gets a row.

## 11. Author decisions (2026-10-31)

- **D1.** NAT-K*-L0nat as defined, with no uniform-scaling arm. The primary estimand of (b) is G, with the rule in
  section 8; within a factor of 2 of the benchmark is descriptive only.
- **D2.** All four additions are accepted: per-layer P_norm, per-slot ‖κ^F‖ with diagnostic (d), the Llama L0nat control
  (first to drop), and the descriptive link of F_i to θ(NAT-K, C1-K).
- **D3.** A separate X9 budget: a 1.5 h cap on its own ledger. The paper reports 23.75 h plus X9's measured time.
- **D4.** Stop and report on any reproduction mismatch, with its size and whether tokens and histories still match.
  There is no pre-approved tolerance.
- **D5.** n = 50 and seed 20261209; (c) is descriptive; (a) and (b) are tests.
