# Research V2 — GPU Feasibility Pilot Report

**Date:** 2026-10-22. **Protocol:** `PILOT_SPEC.md`, locked at commit `2993717`; lock file sha256 `47bd6caea51c81083c05082925df98a57749a21d384b9a336eb1c9c632e73935` (`COAUTHOR_LOCK_NOTE.md`).
**Model:** Qwen2.5-0.5B, FP16 + bitsandbytes NF4. **Hardware:** local RTX 4060 Laptop 8 GB. **Decoding:** greedy, max 256 new tokens, FP16 prefill for every arm.
**GPU time:** **4.76 of the authorized 5.0 h** (ledger: `pilot_runs/gpu_ledger.jsonl`). No cloud, paid API or external compute was used.

**Status of all results.**
- The protocol was locked privately and is informed by prior exploratory findings. It is **not** a pre-registration.
- All pilot accuracy numbers are **descriptive only**: 50 GSM8K-*train* problems per arm. No method is selected on them.
- McNemar p-values below are descriptive statistics, not tests of a claim.

**Preserved results:**
- raw per-problem records: read-only archive `D:\Projects\Dynamic Precision Scaling\archive_research_v2_pilot_2026-10-22\` (65 files; `PILOT_RUNS_MANIFEST.csv` sha256 `795076e84d84e344d3d5c1d84a680f7d5776e08f4b7411fee7c5e62f544b3e8f`);
- analysis outputs: `pilot_analysis/` (`pilot_report.json`, `paired_outcomes.csv`, `boundary_audit.md`, `criteria.txt`);
- run logs: `pilot_*.log`.

---

## 1. Historical reproduction and equivalence tests — all PASS

**Gate 0** (30 test problems from the historical 930-problem pool):

| Arm | Matches stored per-problem outcome | Token-identical to untouched `DynamicQwen.generate` | 4-bit share, full / answer-only (historical n=930) |
|---|---|---|---|
| fp16 | **30/30** | (static; not separately run) | 0 / 0 |
| nf4 | **30/30** | (static; not separately run) | 100 / 100 |
| historical τ p10 (WikiText) | **30/30** | **30/30** (routes identical 30/30) | 41.5 / 40.9 % (41.0 %) |
| historical τ p30 | **30/30** | **30/30** | 78.0 / 79.7 % (79.0 %) |
| historical τ p50 | **30/30** | **30/30** | 93.4 / 94.8 % (93.5 %) |

- **Exact reproduction** on all 150 problem-arm pairs. No disagreement needed investigation.
- The realized historical budgets on this subset match the historical 930-problem values to within 1 pp.
- `histimpl_fp16` / `histimpl_nf4` were not needed (they run only after a disagreement).

**Equivalence block** (real CUDA kernels, 5 train problems):

| Check | Result |
|---|---|
| cascade τ=2 (always escalate: NF4 pass, crop, FP16 recompute) ≡ static FP16 | **5/5** token-identical |
| cascade τ=−1 (never escalate) ≡ static NF4 | **5/5** |
| oracle, always FP16 (snapshot, crop, FP16, keep) ≡ static FP16 | **5/5** |
| oracle, never FP16 (snapshot, crop, FP16, restore NF4 KV) ≡ static NF4 | **5/5** |

**CPU verification before GPU work:**
- 23/23 tests, including real-model identity with the historical implementation, bitwise cache equality after crop/restore, leakage tests, cost-charging recomputed from the raw backend call log, and boundary isolation.
- 5/5 injected bugs detected, including a disabled cascade rollback.
- Full pipeline rehearsal on CPU.

**Flip-oracle semantics on the GPU runs:** 0 violations across all problems. Every emitted token equals the FP16-branch argmax, and FP16 is kept exactly on argmax disagreements.

**TMLR archive:** re-verified unchanged before GPU work (all 383 files match the manifest; archive read-only). It was not modified.

## 2. Accuracy and per-problem paired outcomes (descriptive)

Eval block: 50 GSM8K-train problems. Every arm ran on the same 50 problems. Per-problem outcomes for every arm are in `pilot_analysis/paired_outcomes.csv` (1 = solved, 0 = wrong; no extraction failures occurred).

**Deployable policies and controls**

| Arm | Solved /50 |
|---|---|
| fp16 | 28 |
| nf4 | 25 |
| lag_entropy_B10 | 26 |
| lag_margin_B10 | 25 |
| cascade_nf4_margin_B10 | 25 |
| lag_entropy_B30 | 23 |
| lag_margin_B30 | 22 |
| cascade_nf4_margin_B30 | 22 |
| rand_full_lagE_B10_s1 (p = 0.0815) | 25 |
| rand_full_lagM_B10_s1 (p = 0.0955) | 24 |
| rand_cost_casc_B10_s1 (p = 0.2186) | 25 |
| rand_full_flip_s1 (p = 0.0500) | 25 |
| rand_full_lagE_B30_s1 (p = 0.2542; shared with lag_margin_B30) | 24 |
| rand_cost_casc_B30_s1 (p = 0.5771) | 22 |

**Non-deployable diagnostic reference**

| Arm | Solved /50 |
|---|---|
| oracle_flip (primary diagnostic) | 24 |

**Matched pairs**

| Pair | Matching unit | Solved | Discordant (A only / B only) |
|---|---|---|---|
| fp16 vs nf4 | anchors | 28 / 25 | 9 / 6 |
| lag_entropy_B10 vs its random | full share | 26 / 25 | 2 / 1 |
| lag_margin_B10 vs its random | full share | 25 / 24 | 3 / 2 |
| cascade_B10 vs cost-matched random | total cost | 25 / 25 | 6 / 6 |
| oracle_flip vs its random | full share | 24 / 25 | 4 / 5 |
| lag_entropy_B30 vs random | full share | 23 / 24 | 6 / 7 |
| lag_margin_B30 vs random | full share | 22 / 24 | 5 / 7 |
| cascade_B30 vs cost-matched random | total cost | 22 / 22 | 3 / 3 |

**Reading:** these 50 train problems have almost no FP16–NF4 headroom (28 vs 25). FP16 solves 56% here against 36% on the historical test set, which is consistent with GSM8K-train contamination, though untested. **No comparison above is informative about targeting**, and none was expected to be at n = 50.

## 3. Precision budgets: full-generation (primary), answer-only (secondary), decision-only (sensitivity)

| Arm | FP16 share: full | answer-only | decision-only | answer steps / all steps |
|---|---|---|---|---|
| lag_entropy_B10 | 0.082 | 0.063 | 0.063 | 0.40 |
| rand_full_lagE_B10_s1 | 0.079 | 0.078 | 0.078 | 0.44 |
| lag_margin_B10 | 0.096 | 0.069 | 0.069 | 0.42 |
| rand_full_lagM_B10_s1 | 0.096 | 0.094 | 0.094 | 0.42 |
| cascade_nf4_margin_B10 | 0.115 | 0.071 | 0.071 | 0.39 |
| rand_cost_casc_B10_s1 | 0.215 | 0.220 | 0.220 | 0.44 |
| lag_entropy_B30 | 0.253 | 0.244 | 0.244 | 0.41 |
| lag_margin_B30 | 0.255 | 0.233 | 0.233 | 0.42 |
| rand_full_lagE_B30_s1 | 0.255 | 0.254 | 0.254 | 0.44 |
| cascade_nf4_margin_B30 | 0.304 | 0.242 | 0.242 | 0.39 |
| rand_cost_casc_B30_s1 | 0.581 | 0.585 | 0.585 | 0.42 |
| oracle_flip (non-deployable) | 0.050 | 0.034 | 0.034 | 0.39 |
| rand_full_flip_s1 | 0.051 | 0.049 | 0.049 | 0.41 |

**Main feasibility finding (accounting confound).**
- Only **39–45% of all decode steps are answer-relevant**; the rest is invented post-answer text, which is never scored.
- Every targeted policy, the oracle included, spends a disproportionate share of its FP16 budget there.
- So at a **matched full-generation budget**, the targeted policy has fewer FP16 steps inside the scored region than its random control:

| Comparison | Answer-only FP16 share | Relative shortfall |
|---|---|---|
| entropy B10 vs random | 6.3% vs 7.8% | 19% fewer |
| margin B10 vs random | 6.9% vs 9.4% | 27% fewer |
| flip oracle vs random | 3.4% vs 4.9% | 31% fewer |
| entropy B30 vs random | 24.4% vs 25.4% | 4% fewer |
| margin B30 vs random | 23.3% vs 25.4% | 8% fewer |

- This directly confirms the confound flagged in the exploratory assessment. It means historical full-generation matching **favoured the random controls** in the scored region.
- **The decision-only budget equals the answer-only budget in every arm:** with this prompt, `#### N` immediately precedes the drift marker. The sensitivity definition is redundant for this model and prompt.

## 4. Inference time, recomputation, cache overhead and memory

| Arm | Cost (FP16-pass equiv./step) | ms/step | ms per NF4 / FP16 pass | Recomputes | Crops / restores | Cache-op time | Logging time | Peak alloc. |
|---|---|---|---|---|---|---|---|---|
| fp16 | 1.020 | 39.7 | – / 36.9 | 0 | 0 / 0 | 0 | 5.4% | 1.36 GB |
| nf4 | 0.483 | 62.3 | 59.1 / – | 0 | 0 / 0 | 0 | 3.9% | 1.36 GB |
| lag_entropy_B10 | 0.527 | 57.4 | 55.9 / 35.0 | 0 | 0 / 0 | 0 | 4.1% | 1.36 GB |
| lag_margin_B10 | 0.535 | 58.0 | 56.6 / 35.5 | 0 | 0 / 0 | 0 | 4.1% | 1.36 GB |
| cascade_B10 | 0.601 | 64.0 | 56.3 / 35.3 | 1,445 | 1,445 / 0 | 0.04% | 3.9% | 1.36 GB |
| cascade_B30 | 0.794 | 57.4 | 45.7 / 28.3 | 3,841 | 3,841 / 0 | 0.07% | 3.9% | 1.36 GB |
| oracle_flip | 1.518 | 97.8 | 56.1 / 34.7 | 12,590 | 12,590 / 11,961 | 2.53% | 3.7% | 1.36 GB |
| rand_cost_casc_B10 | 0.599 | 58.7 | 60.6 / 37.2 | 0 | 0 / 0 | 0 | 4.1% | 1.36 GB |

**Cost model** (bytes moved, in FP16-pass equivalents):
- Measured weight bytes: 716 MB per FP16 pass of the routed projections, 185 MB per NF4 pass, 272 MB of always-FP16 weights.
- KV traffic and cache operations are 2–4% of the total.
- Every executed pass is charged, including the NF4 pass a cascade discards. Every FP16 recomputation and every oracle snapshot/restore is also charged.
- At equal total cost, the cost-matched random control gets **1.9× as many FP16 steps** as the cascade: 21.5% vs 11.5% at B10.

**Wall-clock caveats** (latency numbers are not representative of the hardware's capability):
1. The GPU stayed in a **low-clock state**: SM clock 495 MHz of 3,105 MHz, P-state P4, throttle reason "idle", Balanced power plan, on AC. I did not change system power settings.
2. **NF4 is slower than FP16** per pass (≈ 46–61 ms vs 28–37 ms), consistent with the historical finding. Bytes saved are not time saved on this stack.
3. Per-pass times drifted 15–25% between arms run at different times. Latency differences between arms of under ~25% should not be interpreted.
4. **Instrumentation overhead:** against the untouched historical loop on identical tokens, wall time was 1.07–1.11× for NF4-heavy arms but 1.60× for p10. Run order and clock drift are confounded with this, so it is reported as a range, not an estimate.

**Peak memory:** 1.36 GB allocated in every arm, far below the 8 GB card. Prefill takes about 0.10 s per problem and is identical across arms.

## 5. Calibration thresholds and realized budgets

Open-loop calibration on the 50 calibration problems, using exact decision-dtype (FP16) signals and the fixed tie rule:

| Policy | τ | Source share | Eval realized (target) | Within tolerance |
|---|---|---|---|---|
| lag_entropy_B10 | 1.6865 | 0.1000 | 0.0815 (0.10) | yes (≤ 3 pp) |
| lag_margin_B10 | 0.16907 | 0.1000 | 0.0955 (0.10) | yes |
| cascade_nf4_margin_B10 | 0.17078 | 0.1000 | 0.1151 (0.10) | yes |
| lag_entropy_B30 | 0.69824 | 0.2999 | 0.2530 (0.30) | yes, marginal (≤ 5 pp) |
| lag_margin_B30 | 0.70605 | 0.3000 | 0.2554 (0.30) | yes, marginal |
| cascade_nf4_margin_B30 | 0.71289 | 0.2999 | 0.3039 (0.30) | yes |
| oracle_js_B10 | 0.019772 | 0.1000 | not run (cap) | – |
| oracle_fp16_entropy_B10 | 1.6006 | 0.1000 | not run (cap) | – |

- **Ties did not bind**: 2,684–12,219 distinct signal values, and source shares hit their targets to within 0.01 pp.
- **Lag routers under-spend at B30** (−4.5 to −4.7 pp). They were calibrated on the static-NF4 trajectory, but their own trajectories contain 25% FP16 steps.
- **Random controls realized within 0.4 pp of p** in all six arms.

## 6. Answer-boundary error analysis and manual audit

**Manual audit of the 20 seeded records** (`pilot_analysis/boundary_audit.md`): **20/20 boundaries correct by the rule.**
- 17 drift-marker cases: each boundary falls exactly at the `⏎⏎Question:` following `#### N`.
- 2 end-of-sequence cases, both correct.
- 1 cap case: problem 11 under NF4, an unfinished solution cut at 256 tokens, correctly typed `cap`.

**All 1,170 GPU records:**
- Truncated-text extraction is identical in 100% of records.
- Decision-text extraction is identical in 100% of records.
- 0 extraction failures.

**Offset inconsistencies: 35 records, all explained.**
- 34 come from "÷" and 1 from "≈", each emitted as two byte-level tokens. A prefix decode then ends in U+FFFD.
- All 35 occur after the answer boundary.
- A stricter longest-common-prefix offset rule changes **0** answer masks.

**Failure cases** (kept, not dropped):
- **`cap` boundaries:** 0–4 per eval arm, concentrated on three long problems (11, 1440, 6066). These are counted as answer-relevant through the full 256 tokens.
- **`no_hash` decision types:** 0–5 per arm; for these the decision boundary falls back to the answer boundary.
- **System-prompt-like text inside the answer region:** 6 of 1,170 records, e.g. "…follows instruction extremely well. Help as much as you can." followed by EOS. Scored by the historical extractor as usual (wrong).

## 7. Deployable vs non-deployable policies

**Deployable:**
- lag_entropy and lag_margin at B10 and B30 (zero-cost, decided before the step);
- cascade_nf4_margin at B10 and B30 (decides after an NF4 pass, recomputes in FP16);
- matched random controls.

None of these can access the current-step FP16 output. This was verified by the access-control tests, the call-order test, the counterfactual corruption test, and the GPU equivalence block.

**Non-deployable (diagnostic only):**
- `oracle_flip` was run.
- `oracle_js_B10` and `oracle_fp16_entropy_B10` were **not run** (GPU cap).
- The flip oracle costs 1.52 FP16-pass equivalents per step, more than static FP16 (1.02).
- **Its trajectories diverge from pure FP16 early, on all 50 of 50 calibration problems**: first divergence between token 2 and token 111, median token 20. This happens although every emitted token is the FP16 argmax, so the NF4-computed KV entries kept on agreeing steps are enough to change later FP16 argmaxes.
- **It is not "FP16 behaviour at a 5% budget"**, and, as specified, it is not an upper bound on accuracy.
- Its accuracy (24/50) is uninformative at this headroom.

## 8. Failed cases, anomalies, limitations and items not run

**Not run because of the 5 GPU-hour cap** (dropped in locked priority order):
- secondary oracles (`oracle_js_B10`, `oracle_fp16_entropy_B10`);
- answer-only matched random controls (only the answer-share *gap* is measured);
- extra random seeds s2 and s3, so there is **no estimate of random-seed variance**.

**Limitations:**
- **Headroom.** The train-split pilot has almost no FP16–NF4 gap (3/50), so it cannot inform effect sizes.
- **Latency.** Timings are taken in a low-clock GPU state, with drift between arms and confounded instrumentation overhead (§4). Bytes-based cost is the reliable unit.
- **Open-loop calibration** under-spends for lag routers at B30.
- **Cascade cost.** Escalation costs both passes (1.46 FP16-pass equivalents), so the cascade is costly per FP16 step. At B30 it reached 0.79 equivalents per step against 0.62 for the lag routers.
- **The 6 system-prompt-like outputs and the cap cases** are model behaviours, not pipeline faults.

**No numerical anomalies were found:** no NaNs, no cache-length assertion failures, and no crashes in 4.76 GPU-h.

## 9. Revised GPU cost estimate for the full study (from measured pilot rates)

Measured mean wall time per problem (instrumented, low-clock GPU): **12.9 s** for single-pass arms, **15.3 s** for the cascade, **24.6 s** for oracles.

At n = 1,319 GSM8K test problems, each single-pass arm takes ≈ 4.7 GPU-h, each cascade arm ≈ 5.6 h, each oracle arm ≈ 9.0 h.

| Design | Arms | Estimate |
|---|---|---|
| As specified (both budgets; 8 random arms incl. K=3 seeds; 3 oracles) | 2 anchors, 4 lag, 8 random, 2 cascade, 3 oracle | **≈ 104 GPU-h** + calibration ≈ 1 h + model loads |
| B10 only, answer-matched controls included | anchors, lag ×2, cascade, flip oracle, random ×7 | **≈ 67 GPU-h** |
| Minimal headroom design (B10) | anchors, lag_entropy, its random ×3, flip oracle + its random | **≈ 42 GPU-h** |

- **Measured rates are about 2× the pre-pilot estimate.** This estimate supersedes the earlier 45–60 h.
- **Unmeasured speedups:**
  1. A performance power plan that lets the GPU boost.
  2. A lighter timing mode with fewer synchronizations, keeping full traces only where needed.
  3. Reusing the historically stored, exactly reproduced anchor outcomes on the 930-problem pool (outcomes only; no traces).

  Each would need its own validation, and their size is unknown.
- At about 24 GPU-h per day of continuous laptop use, the as-specified design is about 4–5 days of wall time.

## 10. GO / NO-GO recommendation

**Correctness: GO.**
- Exact historical reproduction (150/150 problem-arm pairs).
- Token identity with the untouched implementation (90/90).
- All GPU equivalence checks pass (20/20).
- 0 oracle-semantics violations.
- Boundary rule correct on 20/20 audited records.
- Every offset anomaly is explained and harmless.

**Feasibility: CONDITIONAL.**
- Budgets are controllable: random arms within 0.4 pp; B10 lag/cascade within 1.9 pp; B30 lag routers under-spend by ~4.6 pp, so closed-loop calibration would be needed there.
- Memory is no constraint.
- Instrumentation overhead is small in the bytes model, and cache operations are below 0.1% of cascade time.
- But the as-specified full study costs **≈ 104 GPU-h** at measured rates, well above the ~70 h ceiling assumed in the plan.

**Scientific value: real, but narrower than planned.**
- The pilot confirms a concrete, quantifiable flaw in the historical comparison design: full-generation budget matching gives random controls 19–31% more FP16 inside the scored text at B10. That alone justifies answer-only matching in any future comparison.
- The flip-oracle observation (NF4 KV entries rapidly change later FP16 argmaxes) bears directly on whether step-level targeting can recover FP16 behaviour.
- The deployable lag routers' expected effects are small; the historical point estimate was +2.6 pp. Even a 1,319-problem study is powered only for differences of about ≥ 3 pp. The full study's most decisive question is the oracle/headroom one, not lag-vs-random.

**Recommendation: NO-GO for the full study *as currently specified*; GO for a revised, smaller study, subject to fresh authorization.** Suggested shape, for your decision:
1. GSM8K test (1,319 problems), **B10 only**.
2. Answer-only-matched random controls promoted to **co-primary** next to the historical full-generation controls, given the measured confound.
3. The flip oracle and its random control as the primary headroom reference.
4. K = 3 random seeds for the primary control.
5. Closed-loop calibration (fixed algorithm, at most 2 iterations) for any budget above 10%.
6. Optionally, a separately validated lean timing mode.

That design is estimated at 45–67 GPU-h; the range depends on which controls are kept. It would need a new protocol lock and your explicit approval. If that budget is not acceptable, a paper built on the existing evidence (exact reproducibility, the measured accounting confound, the lag analysis) without new accuracy claims is a reasonable fallback. That too is your decision.

## 11. Questions for the authors

1. Approve, modify or reject the revised full-study shape in §10, and the GPU-hour budget it implies (≈ 45–67 h local)?
2. Promote the answer-only matched comparison to co-primary in any future study?
3. May I test whether a performance power plan, or a lean timing mode, reduces cost? This needs a short, separately authorized GPU check, and the timing changes would have to be validated for token identity.
4. Should the full study reuse the exactly reproduced historical anchor outcomes on the 930-problem pool, or regenerate them with traces?
