# PROTOCOL_V6: laptop-only logging study (r4, 2026-11-15)

**Status.**
- r2 implements the author's reframing of 2026-11-15 (after the R block, `v6/results/R_BLOCK_REPORT.md`, commit 4b40549).
- This text, with the predictions in §7, is committed **before** the rehearsal.
- The rehearsal (§10) is used only for gates, timing and per-problem SDs. Its summary prints no contrast means and no alignment values.
- After the rehearsal, only §8 (power) and §9 (runtime) are updated (r3). Then the protocol is locked.

**r3 changes (accepted by the author, 2026-11-15; all made before any test data).**
1. **Fresh test problems (§3).** The Stage 1/2 FP16 accuracy evaluations of the earlier routing work covered all 1,319 GSM8K test ids, so the r2 rule left none.
   - "Fresh" now means: never used in any cache-error study. That excludes the e1 tier A and tier B ids, the gate ids and the exposed ids (400 ids).
   - All 100 drawn ids were in the Stage 1/2 accuracy evaluations; no cache error was measured on them. `v6_sample.json` records this overlap.
2. **Storage codec (§4).** The r2 integer zero point clamped edge values, with errors up to one step. The codec now uses the FP16 group minimum as the zero point (KIVI-style): x̂ = q·scale + mn, with q = clamp(round((x − mn)/scale), 0, 15). The error stays within half a step. The per-channel variant is the same with the prompt range.
3. **L5 (§7).** The predicted post-RoPE share now includes the finite-sample floor of the position-varying part: c_post = A_p·c_pre + (1 − c_pre)/n. This is derived from the same model, before any L5 quantity was computed.
4. **Disclosure.** A GPU smoke run on one rehearsal problem printed one L1-type number: the median absolute ln(lin_K/meas) on that problem was 0.10. No other contrast, alignment value or verdict was computed or displayed before the lock.

**r4 change (author decision, 2026-11-15, before the lock and before any test data).**
5. **G3 amended (§6).** The pass rule is now: token-identical histories **and** |Δ ln m_i| ≤ 0.05 per problem. Per-step differences are reported descriptively only.
   - **How the wording was chosen:** after seeing the rehearsal gate data, in which r3's per-step rule failed (see the G3 rehearsal result in §6). No test data existed.

**Scope and order.**
- **Scope:** the RTX 4060 Laptop GPU (8 GB) only. No cloud.
- **Order:** THEORY_PLAN_V6.md is the theory. Where its §7 rows L3, L4 and L7 differ from this file, this file governs (author reframing, 2026-11-15).

## 1. Why new logging is needed

v4/v5 records hold per-step JS, error norms and per-slot summed NAT errors. They hold no attention weights, queries, values, per-rotary-pair or pre-RoPE key error, alignment quantities, or storage arms. The R block (R1–R4) used everything the existing records allow.

## 2. Models

| Model | Head dim | Pinned revision (from `v4/V4_MODEL_MANIFEST.json`) |
|---|---|---|
| Qwen2.5-0.5B | 64 | `060db649…` |
| Llama-3.2-1B | 64 | `4e20de36…` |
| Qwen2.5-1.5B | 128 | `8faed761…` |

The precision setup is as in v4: FP16 and NF4 copies of every linear projection; embeddings, norms and head in FP16.

The teacher-forced step is the locked v4 step, unchanged:
1. an NF4 measurement pass;
2. a crop;
3. the FP16 pass, whose logits are the arm's logits and which produces κ^F;
4. an overwrite with the arm's stored entry.

The outcome is per-step next-token JS in float64 against the pure FP16 teacher-forced pass. Per-problem m_i is the mean JS over steps 1…T_i, floored at 1e-9. A contrast is θ(a,b) = mean_i [ln m_i^a − ln m_i^b].

## 3. Sample (`v6/v6_sample.json`, written by `run_v6.py sample`; hashed into the lock)

- **Test:** 100 GSM8K test ids.
  - Drawn by sorting all test ids not used in any cache-error study by SHA-256("rv6-test-2026-11-15|" + id) and taking the first 100.
  - "Used in a cache-error study" means the e1 tier A and tier B ids, the gate ids and the exposed ids: 400 excluded (r3 change 1).
  - **Disjointness (verified 2026-11-15):** test ∩ cache-study ids = ∅; rehearsal ∩ calibration, gate, excluded and few-shot train ids = ∅; reproduction ids ∩ test = ∅. Rehearsal ids are train-split ids; test ids are test-split ids.
  - The same ids serve all three models.
- **Rehearsal:** 8 GSM8K train ids, drawn the same way with the label "rv6-rehearsal-2026-11-15". They exclude the few-shot prefix items and every e1 calibration, gate and excluded train id.
- **Reproduction:** the first 3 e1 tier-A test ids (gate G3).

## 4. Arms (13; constructions in `v6/rv6.py`)

| Arm | Stored K / V | Role |
|---|---|---|
| NAT-K, NAT-V | natural (v4 construction, `rv4.stored_entry_v4`) | baseline, H1–H3 replication |
| C1-K, C1-V | isotropic at the natural per-slot norm (v4 construction; seeds from the master label "rv6-logging-2026-11-15") | isotropic reference |
| NAT-K-cen, NAT-V-cen | natural minus its per-slot mean over the problem (from the same problem's NAT-K / NAT-V run), re-normed to the natural per-slot norm (pilot construction) | removes the common mode |
| ST-K, ST-V | storage codec at its own size: per-token asymmetric RTN INT4, groups of 32 along head dim, FP16 scale and FP16 group minimum as zero point (r3 change 2), applied to κ^F | storage error, natural codec size |
| ST-K@nat, ST-V@nat | codec error direction, re-normed per slot and step to the NF4 natural norm | storage error at the weight-error size |
| ST-V@nat-cen | ST-V@nat error minus its per-slot problem mean (from the same problem's ST-V@nat run), re-normed to the natural norm | storage error without common mode |
| ST-Kc, ST-Kc@nat | per-channel key storage: static per-(layer, KV head, channel) INT4 with the prompt's FP16 key range (clamped), at codec size and at natural size | per-channel key variant (descriptive); **first in the cut order** |

**Run order within a problem:** NAT-K, NAT-V, C1-K, C1-V, NAT-K-cen, NAT-V-cen, ST-K, ST-V, ST-K@nat, ST-V@nat, ST-V@nat-cen, ST-Kc, ST-Kc@nat.

**Construction rules:**
- All non-v4 entries are built in float64 and stored with the locked `V.to_fp16`.
- Per step, the flags `exact_ok`, `rounding_ok` (equal to the locked FP16 oracle), `finite_ok`, `routing_reset` and `natural_ok` are checked.
- For norm-matched arms, `construction_ok` (pre-rounding norm within 1e-4) is also checked.

## 5. Logged quantities (float64; `rv6.Log6`)

**Read time.** Logged after the FP16 pass of each step and before the write-back. Inputs:
- the post-RoPE queries, captured read-only by the locked `rv3.core.QueryCapture`;
- the cache;
- the shadow entries κ^F = stored − δ on the decode positions.

| Key | Definition, per (step, layer, query head); summed (energies) or averaged over the query heads of a KV head |
|---|---|
| m, N_D | decode attention mass under shadow attention; m² / Σ_𝒟 a_j² |
| meas | ‖o(stored) − o(shadow)‖² under the same q: the measured local head-output perturbation |
| lin_V, mod_V | ‖Σ_𝒟 a_j δv_j‖² (equals meas when only V is perturbed); and m²‖μ_V‖² + Σ_𝒟 a_j²‖ε_j‖² (cross term dropped) |
| lin_K, modK_c, modK_w | ‖C + W‖² (first order); ‖C‖² with C = β m(1−m)(v̄_D − v̄_P); ‖W‖² with W = Σ_𝒟 a_j η_j (v_j − ō) |
| jensen, meas_dm | m(1−m) Var_a(η)/2 (the second-order mass shift of Tuncer et al.); the measured change in decode mass |
| lam_K, contrast, s2, mu2_V | query-energy factor of δk; ‖v̄_D − v̄_P‖²; Σ_𝒟 a_j‖v_j − ō‖²/m; ‖μ_V‖² |

**Write time.** Logged per step, layer and KV head:
- **slot sums:** normsK/V and S1K/V, as in v4;
- **per-rotary-pair key sums:** S2K_pair;
- **pre-RoPE natural key error:** obtained by exact inverse rotation at the entry's position (S1K_pre, S2K_pre_pair);
- **input common-mode sums** S1x and S2x of the k_proj input.

**Direction (alignment) quantities.** These are new, per written value entry (layer, KV head). They are ratios to the isotropic expectation, which is 1 for a uniformly random direction:
- **A_o** = (δvᵀ G δv / ‖δv‖²) / (tr G / d), with G = Σ over the KV head's query heads of W_o,gᵀ W_o,g, from the FP16 o_proj weights. A_o measures how strongly o_proj amplifies the error's direction.
- **A_cov** = (‖P_k δv‖² / ‖δv‖²) / (k/d), where P_k projects onto the top k = d/8 eigenvectors of the prompt's FP16 value covariance (per layer, KV head and problem). A_cov measures the share of the error in the dominant value-signal directions.

## 6. Gates (before the test phase; each outcome logged)

1. **G1 — environment.** Torch 2.13.0+cu126, transformers 5.13.0, bitsandbytes 0.49.2, as in the v4/v5 pins. Deterministic mode on.
2. **G2 — logging has no effect.** For every rehearsal and gate problem, NAT-V run with logging off and with logging on give identical per-step JS lists, bitwise.
3. **G3 — reproduction (r4).** NAT-K on the 3 reproduction ids is compared with the pilot's laptop records (`v3_runs/<label>/tierA/`) for Qwen2.5-0.5B and Llama-3.2-1B.
   - **Pass:** the FP16 teacher-forced history is token-identical to the pilot's `FP16_history.jsonl` **and** |ln m_i(now) − ln m_i(pilot)| ≤ 0.05 for every problem (m_i = per-problem mean JS, floored at 1e-9).
   - **Reported descriptively:** whether `js` is bitwise equal, and max |Δ ln js| per step.
   - **History of the rule:** r3 required bitwise `js`, or max per-step |Δ ln js| ≤ 0.05. That rule failed in the rehearsal and was replaced by this one; the wording was chosen after seeing rehearsal gate data (no test data).
4. **G4 — codec.** `codec_per_token` equals the independent numpy implementation bitwise on 10,000 random FP16-valued entries and on real κ^F.
5. **G5 — formulas.** On synthetic attention with known inputs, lin_V equals meas (V-only) to 1e-10 relative error, and lin_K/meas → 1 as the K perturbation shrinks (unit test `v6/tests/test_rv6.py`).
6. **Integrity.** Every per-step flag of §4 holds in every record.

**Rehearsal gate results (2026-11-15; `v6/rehearsal/`).**
- **G1:** passed (torch 2.13.0+cu126, transformers 5.13.0, bitsandbytes 0.49.2, deterministic mode).
- **G2:** passed. Logging on and off gave bitwise-identical per-step JS on all 24 rehearsal problems.
- **G4:** passed, on 10,000 synthetic and on real entries.
- **G5:** passed (unit tests).
- **Integrity:** no failed flag in any record.
- **G3: FAILED as written.** NAT-K on the 3 reproduction ids was not bitwise equal to the pilot's laptop records. Max |Δ ln js| per step was 0.25–0.62, above the 0.05 tolerance, in both pilot models.
- **G3 diagnostic** (`G3_diagnostic.json`), in both models:
  - the teacher-forced histories are token-identical to the pilot's and to v4's;
  - the per-problem outcome ln m_i differs from the pilot by 0.005–0.022 and from v4's A40 records by 0.0006–0.045;
  - the per-step differences from v4, which ran the same software on other hardware, are of the same size (0.22–0.96).
- **Reading:** the per-step criterion is dominated by the numerical noise of near-zero single-step JS under a software change. The step itself is deterministic here (G2).
- **Resolution (r4, author decision 2026-11-15):** G3 amended as above.
- **G3 under r4: PASSED** in both models (`G3_r4_evaluation.json`, computed from the diagnostic run): every history token-identical; |Δ ln m_i| = 0.0058, 0.0096 and 0.0050 (Qwen2.5-0.5B) and 0.0217, 0.0140 and 0.0068 (Llama-3.2-1B).
- **Likely cause of the per-step differences.**
  - The local venv's torch 2.13.0+cu126, transformers 5.13.0 and bitsandbytes 0.49.2 were installed on 2026-07-11, before the pilot (2026-10-31), and are unchanged since. The only later installs (2026-11-03) are hqq, einops and termcolor, which are not used here. So the package versions were very probably the same.
  - The pilot (v3) ran without the deterministic settings introduced by v4 r4 (`torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`; recorded in PROTOCOL_V4). v6 runs with them, so cuBLAS kernel selection can differ.
  - The GPU driver at pilot time was not recorded; it is 621.14 now.
  - These differences are the likely cause. Which of them it was is not determined.

## 7. Pre-declared tests and estimation targets (fixed in this commit, before any rehearsal)

**Counted tests (confirmatory, fresh test problems).**
- **Multiplicity:** Holm runs over the model-level p-values of L2, L6, L7(iii), L8a, L8b, L9a and L9b (21 p-values: 7 tests × 3 models).
- **p-values and intervals:** two-sided paired sign-flip p-values (10,000), and problem-bootstrap 95% CIs (10,000, seed `[20261115, key(model), key(test)]`). Both use the locked `analyze_v3.summarize`.
- **"Positive-significant"** means the CI excludes 0 on the predicted side **and** the Holm-adjusted p < 0.05.

| ID | Statement | Statistic | Agreement | Disagreement |
|---|---|---|---|---|
| L1 | The first-order model is accurate locally | Per slot-step with meas > 0: V-only arms ln(mod_V/meas); K-only arms ln(lin_K/meas). Median of the absolute value per model | Median ≤ ln 1.25 (V) and ≤ ln 1.5 (K) in every model | Median > ln 2 in any model |
| L2 | Head-output energy mediates the outcome | Per problem, ln(Σ meas over steps, layers, heads of NAT-V / of C1-V) vs θ_i(NAT-V, C1-V); Spearman per model; p from 10,000 permutations | ρ > 0, positive-significant, in every model | ρ CI entirely ≤ 0 in any model |
| L5 | RoPE attenuates the key common mode (decisive) | From NAT-K: post-RoPE per-pair common-mode share c_p^post = ‖S1K_p/n‖² / (S2K_pair_p/n) and pre-RoPE c_p^pre; ratio r_p = c_p^post / (A_p(n) c_p^pre + (1 − c_p^pre)/n) per pair with A_p ≥ 0.05, pooled over layers, heads and problems (r3 change 3) | Median abs ln r_p ≤ ln 1.5 in every model | Median > ln 2 in any model |
| L6 | Key-path sign from logged quantities | Per problem, ln(Σ lin_K of NAT-K / of C1-K) vs θ_i(NAT-K, C1-K); sign of the model-level ln ratio vs the sign of θ(NAT-K, C1-K); Spearman | Signs agree in every model, and ρ positive-significant in ≥ 2 of 3 | Signs disagree in any model |
| L7 | Storage error ≈ isotropic, K-first | (i) n·C^V of ST-V; (ii) θ(ST-K@nat, C1-K) and θ(ST-V@nat, C1-V); (iii) ρ_K^st/ρ_V^st from ST-K / ST-V stored_rel, and θ(ST-K, ST-V) | (i) n·C^V ≤ 2 in every model; (ii) each 90% CI inside ± the model's v4 locked margin; (iii) ratio ≥ 1 and θ(ST-K, ST-V) positive-significant in every model, with H3 (θ(NAT-K, NAT-V) < 0) replicating | (ii) θ(ST-V@nat, C1-V) with CI excluding 0 and \|estimate\| ≥ the model's H2 estimate; or (iii) reversed in any model |
| L8a | Natural value error is aligned with the dominant value-signal directions | Per problem, mean over steps, layers and KV heads of ln A_cov for NAT-V minus the same for C1-V | Positive-significant in every model | CI entirely < 0 in any model |
| L8b | Natural value error is aligned with what o_proj amplifies | The same with ln A_o | Positive-significant in every model | CI entirely < 0 in any model |
| L9a | Storage error has a smaller direction effect than weight error (supports (b)) | θ(NAT-V-cen, ST-V@nat-cen): both centred, same per-slot norms | Positive-significant in ≥ 2 of 3 models, and no model with CI entirely < 0 | Any model with CI entirely < 0, or estimate ≤ 0 in ≥ 2 models |
| L9b | Storage error is less aligned than weight error | Per problem, mean ln A_cov of NAT-V minus that of ST-V@nat | As L9a | As L9a |

**Basis of the new predictions (stated before any data).**
- **L8.** NF4 quantizes each 64-weight block with its own absmax scale. Output dimensions of W_v with large weights therefore receive large absolute error, and the same dimensions carry the large value signal that o_proj is trained to read. So natural value error should concentrate in high-variance and o_proj-amplified directions.
- **L9.** Per-token INT4 storage error is near-uniform within each 32-dimension group, with only group-level scale differences. It should therefore be less concentrated in those directions than NF4 weight error.
- **Independence from R.** Neither prediction uses R1 or R2.

**Estimation targets (no predicted value; reported with 95% CIs).**
- **L3 — outcome decomposition of H2, per model.**
  - The identity is θ(NAT-V, C1-V) = θ_cm + θ_dir, with θ_cm = θ(NAT-V, NAT-V-cen) and θ_dir = θ(NAT-V-cen, C1-V). It is exact per problem.
  - Report θ_cm, θ_dir and the share θ_cm/θ(NAT-V, C1-V) (problem bootstrap; percentile CI).
  - The K-path analog uses NAT-K-cen and is descriptive.
- **L4 — head-output decomposition, per model.** The same decomposition with the logged measured energies: ln Σmeas(NAT-V)/Σmeas(NAT-V-cen) and ln Σmeas(NAT-V-cen)/Σmeas(C1-V), per problem, with bootstrap CIs.
- **Descriptive:**
  - per-layer A_o and A_cov for every arm;
  - per-layer m and N_D;
  - pre-RoPE c^K vs c^V per layer;
  - a_in per layer;
  - the per-channel key arms (θ(ST-Kc, ST-V), θ(ST-Kc@nat, C1-K), their ρ);
  - the replication of H1–H3 on the fresh problems under the v4 rules.

**Exploratory (informed by R1/R2, labelled as such in any report).**
- Whether θ_dir or the alignment grows with model size.
- The per-step profile of θ_dir and θ_cm, and their relation to the step-2 contrast.
- Problem-level correlation of the alignment quantities with θ_dir.

## 8. Power and minimum detectable effects (r3, from the rehearsal)

**Source.** Rehearsal on 8 train problems per model (`v6/rehearsal/rehearsal_summary.json`).
- Only per-problem SDs of outcome contrasts were extracted. No contrast mean, alignment value or verdict was computed or displayed.
- With 8 problems, an SD is uncertain by a factor of about 0.66–2.0. Planning therefore uses **SD_plan = 1.25 × the largest of the three models' SDs**, the v3 guard factor.

**MDE.** Two-sided, Holm worst case α = 0.05/21 (z = 3.04), 80% power, n = 100: MDE = 0.388 × SD_plan.

| Contrast (test) | Rehearsal per-problem SD (0.5B / 1B / 1.5B) | SD_plan | MDE (n = 100) or CI half-width |
|---|---|---|---|
| H2 θ(NAT-V, C1-V) (replication; L3 total) | 0.485 / 0.672 / 0.402 | 0.84 | MDE 0.33 |
| H1 θ(NAT-K, C1-K) (replication; L6) | 0.231 / 0.471 / 0.442 | 0.59 | MDE 0.23 |
| H3 θ(NAT-K, NAT-V) (L7iii replication) | 0.447 / 0.347 / 0.606 | 0.76 | MDE 0.29 |
| L3 θ_cm = θ(NAT-V, NAT-V-cen) (estimation) | 0.383 / 0.113 / 0.396 | 0.50 | 95% CI half-width ≈ 0.10 |
| L3 θ_dir = θ(NAT-V-cen, C1-V) (estimation) | 0.538 / 0.612 / 0.397 | 0.77 | 95% CI half-width ≈ 0.15 |
| L7(ii) θ(ST-K@nat, C1-K) (equivalence) | 0.460 / 0.561 / 0.859 | 1.07 | 90% CI half-width ≈ 0.18, against margins of 0.251 / 0.210 / 0.216. Equivalence is attainable only if the true difference is within about ±0.03–0.07; otherwise expect *neither* |
| L7(ii) θ(ST-V@nat, C1-V) (equivalence) | 0.585 / 0.521 / 0.704 | 0.88 | 90% CI half-width ≈ 0.14; equivalence if the true difference is within about ±0.07–0.11 |
| L7(iii) θ(ST-K, ST-V) | 1.362 / 0.638 / 0.581 | 1.70 | MDE 0.66 |
| L9a θ(NAT-V-cen, ST-V@nat-cen) | 0.234 / 0.568 / 0.457 | 0.71 | MDE 0.28 |
| L8a, L8b, L9b (alignment differences) | not measured (author instruction: no alignment values before the lock) | — | MDE = 0.39 per-problem SD (standardized) |
| L2, L6 (problem-level Spearman) | — | — | MDE ρ ≈ 0.375 |

**L7(ii) is underpowered for equivalence.** At n = 100 the pre-stated equivalence verdict is likely to be *neither*, unless storage error at the weight-error size is within a few hundredths of isotropic. This is stated now, before any test data. The criteria are unchanged.

## 9. Runtime and stop rules (r3, measured)

**Measured on the RTX 4060 Laptop GPU, 8 rehearsal problems per model (mean decode length 56–58 steps).**

| Model | Wall time per problem, 13 arms with logging (incl. history and reference) | Logging overhead factor (NAT-V with / without logging) |
|---|---|---|
| Qwen2.5-0.5B | 148.5 s | 1.82 |
| Llama-3.2-1B | 104.8 s | 1.69 |
| Qwen2.5-1.5B | 169.2 s | 1.76 |

**Projection.** 100 problems × (148.5 + 104.8 + 169.2) s = **11.7 h**. The ledger already holds 0.98 GPU-h (smoke run, rehearsal, G3 diagnostic), so the projected total is **12.7 h**, under the 16 h cap. Design: **no cut; n = 100, all 13 arms**.

**Stop rules (fixed in r2, unchanged).**
- **Cap:** 16 laptop GPU-hours in the ledger.
- **Cut order, if a projection or a running total requires it:** (1) drop ST-Kc and ST-Kc@nat; (2) n = 70; (3) drop NAT-K-cen; (4) n = 50.
- **Stop:** stop and report if the cap is reached.
- No cloud.

## 10. Rehearsal (before the lock; train problems only)

- **Scope:** 8 rehearsal train ids per model, all 13 arms with logging, plus NAT-V with logging off (G2), plus G3 and G4.
- **Output:** gate results; wall time per problem-arm with and without logging; per-problem SDs of the §7 contrasts.
- **What the analyst does not see:** contrast means and alignment values are not printed, and the analyst does not inspect them before the lock.
- **Records:** written to `v6_runs/rehearsal/` and never used in any §7 analysis.

## 11. Locks, authorization, deviations

- **Lock** (`tools/lock_v6.py`, from committed content): the SHA-256 of every locked file. The lock id is the first 16 hex characters of the SHA-256 of the lock text.
- **Locked files:**
  - this protocol;
  - THEORY_PLAN_V6.md;
  - rv6.py, run_v6.py and analyze_v6.py;
  - the tests;
  - `v6_sample.json`;
  - the model manifest.
- **Test phase.** It runs only with `v6_runs/AUTHORIZED_V6_TEST.txt`, written by or at the explicit instruction of the corresponding author.
- **Ledger.** Each block has a GPU ledger line (`v6_runs/gpu_ledger.jsonl`).
- **Deviations.** Post-lock changes go to `v6/DEVIATION_LOG_V6.md` before they are acted on.
- **AI use.** Logged in `v6/AI_USE_LOG_V6.md`.
