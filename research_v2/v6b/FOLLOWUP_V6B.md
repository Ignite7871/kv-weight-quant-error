# FOLLOWUP_V6B: does the "direction part" survive the re-norming and the order of the split? (r1, 2026-11-16)

**Status.**
- **r1 (2026-11-16):** the author approved D1–D5 as proposed (arm C included; B at the centered norm; 9 h cap and the cut order;
  threshold 0.5 with single-path rejection; no attention logging) and asked for CPU tests with mutants, a short runtime-only
  rehearsal, and the lock. r1 records the implementation details that the code fixes (marked **r1** below). No test record exists.
- After the rehearsal only §8 (runtime) changes (r2). Then the protocol is locked.
- **r2 (2026-11-16):** §8 runtime from the rehearsal. G1 reproduced the v6 rehearsal records bitwise on all 12 problems, and no
  record has a failed flag. No contrast, SD, mean, share or verdict was computed or viewed.
- Decides the headline that fresh-eyes review 3 (item 1a) put in doubt. Until it runs, the manuscript carries the provisional
  title "... Most of the Value-Path Excess Survives Removing Its Common Mode".
- **Scope:** the RTX 4060 Laptop GPU (8 GB) only. No cloud.
- **Exposure statement.** The analyst and the authors know the v6 L3 results: common-mode share 0.24 / 0.23 / 0.13 and
  θ_dir 0.676 / 0.766 / 0.688 for Qwen2.5-0.5B / Llama-3.2-1B / Qwen2.5-1.5B, from NAT-V, NAT-V-cen and C1-V. Path R1 below
  reruns that split and is therefore **not** new evidence. Paths R2, U1 and U2 use arms that have never been run.

## 1. Why

The v6 split (manuscript Equation 4) is
θ(NAT-V, C1-V) = θ(NAT-V, NAT-V-cen) + θ(NAT-V-cen, C1-V) = θ_cm + θ_dir.
The review found three problems with reading θ_dir as "direction":

1. **Re-norming.** NAT-V-cen restores the natural per-slot norm after centering. That scales the position-varying part ε_j
   up by about 1/√(1−c), so θ_cm understates the effect of removing μ and θ_dir overstates the direction part.
2. **Order.** The split removes the common mode first, then randomizes the direction. With interaction, the other order can
   give different shares.
3. **Content.** θ_dir also contains structure across slots that the per-slot isotropic control removes.

This follow-up addresses 1 and 2. Problem 3 needs an arm that randomizes direction jointly across slots, which is outside
this follow-up and stays a stated limitation.

## 2. Models, sample, protocol

- **Models:** Qwen2.5-0.5B, Llama-3.2-1B and Qwen2.5-1.5B, the v6 manifest, NF4 as in v6.
- **Sample:** n = 70 per model, the first 70 test ids of `v6/v6_sample.json`, in sample order.
  - These are exactly the ids Llama-3.2-1B and Qwen2.5-1.5B ran in v6. Verified 2026-11-16 from the v6 NAT-V records.
  - Qwen2.5-0.5B ran these 70 and 30 more.
- **Protocol:** the v6 controlled step and outcome are unchanged: teacher-forced FP16 greedy history, steps 1…T_i, per-problem
  mean JS floored at 1e-9, θ(a, b) = mean_i[ln m_i^a − ln m_i^b].
- **Constructions:** v6 primitives imported unchanged (`rv6.stored_entry_v6`, `rv6.slot_mean`, `V.to_fp16`, the v4 C1
  construction, the v6 seed label "rv6-logging-2026-11-15").
- **r1, implementation.**
  - The locked v6 step (`rv6.teacher_forced_arm_v6`, `rv6.controlled_step_v6`) is used unchanged. `v6b/rv6b.py` replaces only the
    module attribute `rv6.stored_entry_v6` with a dispatcher. The dispatcher builds A, B and C and delegates every v6 arm to the
    original function. No locked v6 file is edited; the v6 lock 00369b8d99cefd69 still verifies.
  - The reruns use v6's seed tiers (test "V6A", rehearsal "V6reh"), so they can reproduce v6 bitwise.

## 3. Arms (6 per problem; value path only, keys FP16)

**Notation.** Per slot σ (layer, V, KV head), over the problem's decode entries:
- δ_j = κ^N_j − κ^F_j is the natural error;
- μ = mean_j δ_j is computed exactly as in v6, `slot_mean` of the same session's NAT-V run;
- ε_j = δ_j − μ;
- u_j is **the same unit vector C1-V draws** at that slot and step (same seed label, so the same direction as v6 C1-V).

| Arm | Stored value error | Per-slot norm | New? |
|---|---|---|---|
| NAT-V | δ_j = μ + ε_j | ‖δ_j‖ | rerun (gives μ; G1) |
| C1-V | ‖δ_j‖ u_j | ‖δ_j‖ | rerun (G1) |
| NAT-V-cen | ‖δ_j‖ · ε_j/‖ε_j‖ (v6 construction, re-normed) | ‖δ_j‖ | rerun (G1) |
| **A: NAT-V-cen0** | ε_j (centered, **not** re-normed) | ‖ε_j‖ | requested arm (a) |
| **B: C1-V+μ** | μ + ‖ε_j‖ u_j | ‖δ_j‖ in expectation over positions; realized ratio reported | requested arm (b), made precise (D2) |
| **C: C1-V@cen** | ‖ε_j‖ u_j (isotropic at the centered norm) | ‖ε_j‖ | completing arm, recommended (D1) |

**r1, how δ_j and μ are computed.**
- As in v6 NAT-V-cen, δ_j at step j is the natural error written in **the arm's own run** (kM − kF at that step). Each arm's
  cache holds its own stored entries, so δ_j differs slightly between arms. μ is fixed per problem.
- μ is accumulated with `rv4.SlotSums` from the stored NAT-V entries and divided by the number of steps (`rv6.slot_mean`). This
  is the arithmetic v6's logged arrays used.
- No arm, including NAT-V, uses attention logging (D5). v6 gate G2 showed that logging does not change any logit.

**Notes on the new arms.**
- **Run order within a problem:** NAT-V, C1-V, NAT-V-cen, A, B, C.
- **Construction:** float64, rounded once with `V.to_fp16`, with the v6 per-step flags (`exact_ok`, `rounding_ok`, `finite_ok`,
  `routing_reset`, `natural_ok`). A and C also carry `construction_ok`: pre-rounding norm within 1e-4 of ‖ε_j‖.
- **Degenerate steps:** steps with ‖ε_j‖ < DEGEN_TOL·‖δ_j‖ are counted. A and C store the tiny ε_j, with no fallback.
- **Arm B norm.** Σ_j‖δ_j‖² = n‖μ‖² + Σ_j‖ε_j‖², because Σ_j ε_j = 0. So B matches the natural energy per slot over the
  problem in expectation (the cross term 2‖ε_j‖ μ·u_j has mean zero), but not position by position. B's realized norm ratio
  ‖B_j‖/‖δ_j‖ is logged and reported.
- **r1, B's own check.** B has no fixed-norm target, so the locked `construction_ok` is trivially true for it. v6b adds
  `b_construction_ok`: per step, ‖(stored error) − μ‖ must equal ‖c‖ within the locked tolerance 1e-4 (pre-rounding). It is part
  of the integrity flags.

**The 2×2 the arms fill.** Rows: is the common mode present? Columns: what is the position-varying part?

| | natural ε_j | isotropic |
|---|---|---|
| μ present | NAT-V | B (μ + ‖ε_j‖u_j) |
| μ removed, re-normed to ‖δ_j‖ (R) | NAT-V-cen | C1-V |
| μ removed, not re-normed (U) | A (ε_j) | C (‖ε_j‖u_j) |

## 4. Estimands (estimation targets, not tests; per model, paired by problem)

Each path telescopes exactly per problem:

| Path | Normalization | Order | Common-mode part | Direction part | Total |
|---|---|---|---|---|---|
| R1 (= v6 L3) | re-normed | cm first | θ(NAT-V, NAT-V-cen) | θ(NAT-V-cen, C1-V) | θ_V = θ(NAT-V, C1-V) |
| R2 | re-normed* | dir first | θ(B, C1-V) | θ(NAT-V, B) | θ_V |
| U1 | not re-normed | cm first | θ(NAT-V, A) | θ(A, C) | θ_U = θ(NAT-V, C) |
| U2 | not re-normed | dir first | θ(B, C) | θ(NAT-V, B) | θ_U |

\*B is norm-matched in expectation only. R2 is reported with B's realized norm ratio.

**Shares.** For each path P, s_dir^P = mean(direction part) / mean(total), with s_cm^P = 1 − s_dir^P.
- 95% CI from the problem bootstrap of the ratio of means.
- **r1:** the code of `analyze_v6.ratio_ci`, copied into `analyze_v6b.py` with the v6b seed. B = 10,000; streams
  `default_rng([20261116, key(model), key(estimand)])`. θ CIs use the locked `analyze_v3.summarize` with the same streams.
- **r1, the problems analysed:** the problems present in all six arms. If cut 1 applied, C1-V and NAT-V-cen of problems 11–70
  come from the v6 records of the same problem.

**Further estimands.**
- **Norm term:** θ_N = θ(C, C1-V), isotropic error at the centered norm against the natural norm. Then θ_V = θ_U + θ_N exactly.
- **Interaction**, per normalization:
  - I_R = θ(NAT-V, B) − θ(NAT-V-cen, C1-V);
  - I_U = θ(NAT-V, B) − θ(A, C).
  - Each is reported with a 95% CI. A CI excluding 0 means the split depends on its order.
- **Re-norming effect on the centered error:** θ(NAT-V-cen, A). This measures problem 1 of §1 directly.
- **Descriptive:**
  - median per-slot ‖δ_j‖/‖ε_j‖ (the re-norm factor);
  - B's realized norm ratio;
  - degenerate-step counts;
  - per-problem reproduction differences against the v6 records (G1).

**Expectations, stated before any data (not tests).**
- Under the first-order model, removing μ without re-norming also shrinks the error, so s_cm^U1 should exceed s_cm^R1.
- θ_N should be negative (isotropic error at the smaller centered norm is milder).
- No size is predicted.

## 5. Pre-stated reading of the headline

The headline under test is **H\***: "the direction part carries most of the value-path excess". Each model is classified
from the four s_dir CIs.

| Verdict for H\* | Rule (all CIs 95%, unadjusted) | Manuscript consequence |
|---|---|---|
| **Keep, unrestricted** | In all 3 models and all 4 paths, the lower bound of s_dir > 0.5. | Title may state that direction carries most of the value-path excess, "under either order and with or without re-norming". The cross-slot caveat (§1, problem 3) stays. |
| **Keep at fixed norm** | In all 3 models, the lower bound > 0.5 in R1 and R2, but not in every U path. No path has an upper bound < 0.5. | Headline restricted: "at fixed per-slot norm, in either order, most of the excess is in the direction part". The U shares are reported as dependence on normalization. |
| **Reject** | In any model, any path has an s_dir upper bound < 0.5, so the common mode carries most of the excess along that path. | H\* is dropped. The title does not claim direction. The paper reports the split as order- or normalization-dependent and names the path. |
| **Undecided** | Anything else. | H\* is not used. The four shares are reported. The provisional title stays. |

**Notes on the rule.**
- The rule uses unadjusted 95% CIs over 12 shares (3 models × 4 paths). "Reject" therefore fires on any single path, which
  makes it hard to keep H\* by chance.
- R1 is a rerun of a known result. If R1 alone differs from v6 by more than the G1 tolerance, the run stops (G1).
- Separately, regardless of the verdict, the manuscript reports I_R and I_U. If either CI excludes 0 in any model, the text
  says the shares depend on the order of the split.

## 6. Gates

- **G0: CPU unit tests**, before the rehearsal. They check identities of the constructions:
  - A re-normed to ‖δ_j‖ equals NAT-V-cen before rounding;
  - B with μ set to 0 equals C;
  - C re-normed to ‖δ_j‖ equals C1-V before rounding;
  - B − μ has norm ‖ε_j‖.
- **G0 is part of a larger CPU check (r1).** `v6b/tests/test_rv6b.py` (constructions, driver and launcher logic) and
  `v6b/tests/test_analyze_v6b.py` (analysis and recomputation on synthetic records with known shares, one fixture per verdict)
  must all pass. `v6b/tests/mutation_check_v6b.py` must kill every injected fault. The result is in
  `v6b/results/mutation_check_v6b.json` and is committed before the lock.
- **G1: reproduction**, on every problem, run **before any new arm of that problem**. The reruns of NAT-V, C1-V and NAT-V-cen
  are compared with the v6 records of the same problem.
  - **r1, the pass rule:** the same number of steps S and the same history length, and |Δ ln m_i| ≤ 0.05 (the v6 G3 rule).
  - The v6 test records store the history length but not the history tokens. History identity is therefore checked through S
    and the history length, with bitwise equality of the per-step js and of the arm's per-step argmax tokens reported.
  - If any problem fails, the driver writes `g1_failure.json` and stops. Refer to the author.
  - The rehearsal checks G1 against the v6 rehearsal records.
  - All contrasts use same-session runs, except under cut 1 (§8).
- **G2: integrity.** Every per-step flag holds in every record. Any failure is reported and the problem is excluded.

## 7. Power

From the v6 rehearsal SDs (PROTOCOL_V6 §8), the 95% CI half-width of θ_cm and θ_dir at n = 70 is about 0.12 and 0.18.
The v6 share CIs at n = 70–100 were about ±0.06–0.08.

With s_dir^R1 at 0.76–0.87, a lower bound above 0.5 in R1 is expected. The rule is informative about whether R2, U1 and U2
also clear 0.5.

**r1:** the author asked for a runtime-only rehearsal, so no rehearsal SDs of the new contrasts are computed. Power rests on
the v6 SDs above.

## 8. Runtime and stop rules

**Measured per-arm wall time in the v6 test records** (`wall_seconds`, mean per problem-arm), plus the per-problem overhead
(history, reference):

| Model | s / arm (v6 test) | Overhead / problem | 6 arms / problem |
|---|---|---|---|
| Qwen2.5-0.5B | 21.8 (run during the throttled period; rehearsal suggested ~10) | ~9 s | ~140 s |
| Llama-3.2-1B | 6.9 | ~3 s | ~45 s |
| Qwen2.5-1.5B | 14.2 | ~7 s | ~92 s |

**Projection.**
- Test phase: 70 × (140 + 45 + 92) s = **5.4 GPU-h**; with the v3 guard factor 1.25, **6.7 h**.
- Rehearsal and gates: about 0.4 h.
- Projected total: **about 7.1 laptop GPU-hours**, likely less, because Qwen2.5-0.5B's per-arm time is taken from a
  throttled period and the new arms need no attention logging (only NAT-V is logged, for μ).

**r2, measured in the rehearsal (2026-11-16; 4 rehearsal train problems per model, all 6 arms, no attention logging).**

| Model | s / arm (mean) | Block s / problem (incl. model load) | Records | G1 |
|---|---|---|---|---|
| Qwen2.5-0.5B | 7.0–7.9 | 50.1 | 24, no failed flag | 4/4 pass, all bitwise (js and tokens) |
| Llama-3.2-1B | 4.7–5.1 | 34.5 | 24, no failed flag | 4/4 pass, all bitwise |
| Qwen2.5-1.5B | 8.2–8.9 | 58.4 | 24, no failed flag | 4/4 pass, all bitwise |

- **Projection:** 70 × (50.1 + 34.5 + 58.4) s = 2.8 GPU-h. With the guard factor 1.25 it is 3.5 h; the rehearsal ledger adds
  0.16 h, for a total of about 3.6 h. No cut is expected under the 9 h cap.
- The v6 per-arm figures above were slower because v6 logged attention on every arm and part of its run was power-throttled.
- The launcher projects with the block seconds per problem (§8 r1).

**Stop rules.**
- **Cap:** 9 laptop GPU-hours in the v6b ledger (D3).
- **Cut order**, if a projection or the running total requires it:
  1. Stop rerunning C1-V and NAT-V-cen after G1 has passed bitwise on the first 10 problems of a model, and use the v6
     records for those two arms. This saves about a third. If any of the first 10 problems is not bitwise on all three
     reruns, cut 1 does not take effect for that model.
  2. n = 50 (the first 50 ids).
- **r1, the projection.** Before each model, `launch_test_v6b.py` computes: ledger + remaining problems × rehearsal seconds
  per problem × ratio. The ratio is the observed test/rehearsal ratio of the models already complete, or the guard 1.25 before
  any is complete. Cut 1 multiplies the remaining cost by 4/6. The cuts are applied in order until the projection is at most 9 h.
- Stop and report if the cap is reached. No cloud.

## 9. Rehearsal (before the lock; train problems only)

- **r1:** the first 4 of the 8 v6 rehearsal train ids per model, all 6 arms, with v6's rehearsal tier.
- **Output:** G1 against the v6 rehearsal records, the integrity flags, and wall time per problem-arm and per problem.
- **What the analyst does not see:** no contrast, SD, mean, share or verdict is computed, printed or inspected before the lock.
  `run_v6b.py rehearsal-summary` computes none.
- Only §8 (runtime) may change after the rehearsal (r2).

## 10. Files, lock, authorization, deviations (as in v6)

**Files (r1, all written before the lock):**
- `v6b/rv6b.py`: the constructions of A, B and C, and μ; imports rv6 unchanged.
- `v6b/run_v6b.py`: the driver (rehearsal, test, G1, exactly-once writing, ledger, cap).
- `v6b/launch_test_v6b.py`: the detached launcher.
  - It does the exactly-once scan, heartbeat recovery of interrupted GPU time, the projection and cuts, the power-throttling
    opt-out and keep-awake, and per-model status files that report integrity counts and GPU-hours only.
  - It is started as a Windows scheduled task running `pythonw.exe` (no console), so it survives a closed terminal. This is how
    the v6 test phase finally ran; a WMI start and a console task both died in v6.
- `v6b/analyze_v6b.py`: the locked analysis.
- `v6b/recompute_v6b_independent.py` with `v6b/RECOMPUTATION_TOLERANCES_V6B.md`: independent recomputation, numpy and
  standard library only.
- `v6b/tests/test_rv6b.py`, `v6b/tests/test_analyze_v6b.py` and `v6b/tests/mutation_check_v6b.py`.
- `tools/lock_v6b.py`.

**Lock.**
- `tools/lock_v6b.py` writes `V6B_PROTOCOL_LOCK.txt` from **committed** content. It refuses if a locked file is uncommitted
  or modified. A lock is never rewritten.
- Locked files:
  - this protocol;
  - rv6b.py, run_v6b.py and analyze_v6b.py;
  - the tests;
  - `v6/v6_sample.json`;
  - the v6 locked files rv6b imports;
  - the lock tool.
- **r1:** the exact list is `FILES` in `tools/lock_v6b.py`. It includes every v6b file above, plus `v6/rv6.py`, `v6/run_v6.py`,
  `v6/v6_sample.json`, `v4/rv4.py`, `v4/run_v4.py`, `v4/V4_MODEL_MANIFEST.json`, `rv3/core.py`, `analyze_v3.py` and
  `V6_PROTOCOL_LOCK.txt`.
- **Lock id:** the first 16 hex characters of the SHA-256 of the lock text. It is a private dated hash, not a third-party
  registration.

**Authorization.** The rehearsal runs only with `v6b_runs/AUTHORIZED_V6B_REHEARSAL.txt`, and the test only with
`v6b_runs/AUTHORIZED_V6B_TEST.txt`. Each is written by, or at the explicit instruction of, the corresponding author.

**During and after the run.**
- **Run once.** The locked analysis runs once. The independent recomputation then compares every reported quantity within
  the declared tolerances.
- **Ledger:** `v6b_runs/gpu_ledger.jsonl`, one line per block.
- **Deviations:** logged in `v6b/DEVIATION_LOG_V6B.md` before they are acted on.
- **AI use:** logged in `v6b/AI_USE_LOG_V6B.md`.

## 11. Author decisions (r1: all approved as proposed, 2026-11-16)

- **D1, arm C (recommended: include).** Without C, U1 can only give θ(NAT-V, A) and θ(A, C1-V). The latter mixes direction
  with the norm change from ‖ε_j‖ to ‖δ_j‖, so the unnormalized split and θ_N are not identified. Cost: one arm, about 1/6 of
  the run.
- **D2, construction of B (recommended: μ + ‖ε_j‖u_j).** The literal "C1-V plus μ", μ + ‖δ_j‖u_j, carries about ‖μ‖² more
  energy per slot than the natural error. R2's common-mode step would then mix removing μ with that extra norm.
- **D3:** the cap of 9 GPU-h and the cut order of §8.
- **D4:** the 0.5 threshold and the four-way rule of §5, including that a single path can reject.
- **D5, attention logging on the new arms (recommended: no).** Logging would give descriptive A_cov for A, B and C at about
  1.7× their cost. The decision in §5 does not need it.
