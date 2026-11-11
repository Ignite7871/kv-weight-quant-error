# Stage 1 Specification — token-source × KV-cache-source interventions on GSM8K (Qwen2.5-0.5B)

Date: 2026-10-23. Branch `research-recovery` (local commits only; nothing is pushed or published). Baseline commit at the start of Stage 1: `06a34fee2dd63d2e857e03cb25822db8f81c83d8`.

## 0. Status and provenance (read first)

- **This protocol is exploratory in origin.** It is privately locked with a dated SHA-256 hash for the coauthors. It is **not** a pre-registration and must never be described as one.
- It is informed by earlier looks at these very data and by two earlier studies:
  - the 2026-10-22 pilot (50 GSM8K-*train* problems): the flip oracle's trajectories diverged from pure FP16 on 50/50 problems (median first divergence at token 20) although every emitted token was the FP16 argmax;
  - the historical per-problem outcomes on the 930-problem test pool (FP16 336, NF4 243 solved) and several routing comparisons run on that pool;
  - 30 problems of that pool (gate0) were re-run under several policies in the pilot;
  - `perf/PERF_REPORT.md` and `NEXT_STUDY_DESIGN.md` (cost models, Q3 design).
- Consequently the 930-problem evaluation is an **exploratory replication with pre-declared estimands on data that have been examined before**. It is not independent confirmation of any hypothesis that those looks suggested. Historical outcomes are kept for continuity and are not independent confirmation.
- The **remaining 389 test problems (indices not in the 930 pool) are reserved** for a separately reported, protocol-locked validation. Stage 1 never reads, runs or tunes on them (the driver asserts this; their index list is in `stage1_sample.json`).
- Stage 2 and the comprehensive study are **not authorized**. No randomized decoding, no other model, dataset or budget, no cloud or paid compute.

## 1. Question and what is, and is not, being estimated

**Objective.** Investigate how emitted-token differences and quantization-induced KV-cache differences contribute to changes in downstream GSM8K behaviour.

**Not assumed:** that the two sources are additive; that the flip oracle bounds task accuracy; that any share of an accuracy difference is "attributable" to one source. Estimands are **contrasts of accuracy and of trajectory divergence under the specific interventions below**. Causal statements are conditional on those interventions, on greedy decoding, on this model/prompt/benchmark, and on the numerical stack used (FP16 + bitsandbytes NF4).

## 2. The interventions (exact semantics)

A decode step runs the model on the previously emitted token `x_{s-1}`. That one pass yields both the next-token logits and the K/V entries of position `L` (the position of `x_{s-1}`). A cell chooses independently:

| Symbol | Meaning |
|---|---|
| token source `t` ∈ {F, N} | which pass's logits select `x_s` (greedy argmax) |
| KV source `k` ∈ {F, N} | which pass's K/V entries for position `L` stay in the cache |
| F / N | FP16 weights / NF4 weights (the historical `DynamicLinear` dual-weight construction) |

**Common to every condition:** the prompt is processed by an FP16 prefill, and token 0 is the argmax of the prefill logits. Therefore **prompt K/V are always FP16**; "NF4 cache" means NF4-written *decode* positions only. Greedy decoding, at most 256 new tokens, stop after EOS, historical prompt and extractor.

**Per step for a hybrid cell** (both passes start from the same cache and consume the same emitted token, so caches cannot be combined across histories):
1. run the precision that is **not** kept (it appends its K/V at `L`); keep its logits;
2. crop the cache back to length `L` (a view; the pre-step state is restored exactly);
3. run the kept precision (it appends the K/V that stays); keep its logits;
4. emit the argmax of the token-source logits.

| Cell | Token from | KV kept from | Implementation | Interpretation (conditional) |
|---|---|---|---|---|
| FF | FP16 | FP16 | static FP16 (`engine.generate`) | reference |
| NN | NF4 | NF4 | static NF4 | reference |
| **cache-only (FN)** | FP16 weights | NF4-written | `stage1.generate_cell` | quantization reaches the output only through the **stored K/V of past decode positions**; the FP16 pass still uses its own fresh K/V for the current position |
| **token-only (NF)** | NF4 weights | FP16-written | `stage1.generate_cell` | quantization reaches the output only through the **token choice**; all stored K/V are FP16 |
| **flip oracle** (diagnostic) | FP16 argmax by construction | NF4 on agreeing steps, FP16 on flip steps | `engine.generate` + `OracleFlip` (unchanged pilot code) | **not equivalent to any cell** (its cache mixes precisions chosen retrospectively); non-deployable; myopic; not an accuracy bound |

- **Interventions on token choice vs on cached activations are different manipulations.** Cache-only changes what is *stored*; token-only changes what is *emitted*. Neither isolates "weights" from "cache" in an absolute sense: in cache-only the FP16 weights read NF4-written entries.
- **Histories stay internally consistent:** the emitted token is the single history shared by both passes; the cache of every cell is, bitwise, the cache of a single-precision teacher-forced replay of that cell's own emitted history (tested).
- Until a cell's first token that differs from the FP16 reference, its history equals the FP16 history. At that first-divergence step, cache-only differs from FP16 *only through the NF4-written cache* and token-only *only through NF4 weights selecting the token on an FP16 cache*; static NF4 differs through both. This is the cleanest separation the design offers and is analysed explicitly (section 7).

## 3. Design

- **Problems:** the 930-problem historical test pool (`results/gsm8k_multibudget_p10_entropy_only.json` indices). Run order: seeded permutation (`rv2-stage1-order-2026-10-23`), **problem-major** (all five arms on a problem before the next), arm order rotated per problem so clock drift is balanced across arms. A prefix of the order is a random subset of the pool.
- **Arms (5):** `fp16`, `nf4`, `cache_only`, `token_only`, `oracle_flip`.
- **Early stop at the causal drift marker:** generation stops once the decoded text matches the historical extractor's `\n\s*Question\s*:` marker (a rule on already emitted text; it never uses future information). The scored text is unchanged because the extractor discards everything after the marker (prefix stability, test `t6b`; verified on GPU in gate C). Generations with no marker run to EOS or the 256-token cap.
- **Scoring:** the historical extractor (`SOLVED` / `WRONG` / `EXTRACTION_FAILED`; accuracy counts only `SOLVED`). Answer boundary recorded as in the pilot.
- **Historical continuity:** regenerated `fp16`/`nf4` outcomes are compared problem by problem with the stored historical outcomes; every disagreement is listed, none is dropped. Paired contrasts use the regenerated arms (same code path, same trajectories).
- **Calibration:** no thresholds are involved (all cells are deterministic), so the existing GSM8K-train calibration is not applicable.
- **Budget matching is not part of Stage 1.**

## 4. Preconditions before the main GPU run (each is a hard gate)

| Gate | Check | Pass criterion |
|---|---|---|
| G1 | CPU suite `run_cpu_tests.py` (pilot, 23 tests) | all pass (baseline 23/23 recorded before any change) |
| G2 | CPU suite `run_stage1_tests.py` (10 tests) + `mutation_check_stage1.py` | all pass; 6/6 injected bugs killed; clean code passes |
| G3 | GPU equivalence block (`stage1_gates.gate_equiv`, train problems only) | **A:** FF and NN cells ≡ rv2 static arms ≡ the untouched historical `DynamicQwen.generate` (token ids identical, 5 problems × 2 precisions × 256 tokens) and FF/NN final caches bitwise equal to the static arms'. **B:** both hybrids ≡ the independent deep-copy reference (ids identical, final caches bitwise equal), caches bitwise equal to the single-precision replay of the emitted history, every token equals the argmax of its declared source. **C:** early-stopped generations are prefixes of full generations with identical boundary and score (static and hybrid) |
| G4 | feasibility subset (10 GSM8K-train problems, all five arms) | token == declared-source argmax at every step; branch labels consistent with the static arms at step 1; both hybrids diverge from FP16 on at least one problem (non-vacuous); flip oracle equals cache-only up to its first flip |
| G5 | fixed-workload speed re-check | recorded and compared with 2026-10-23; used for the projection |
| G6 | protocol lock created after G1-G5 and before the main run | committed code; lock file hash recorded |
| G7 | runtime projection | central cumulative estimate (gates + main run) ≤ 12.0 h; otherwise **stop and request fresh approval** |

**Stop rules:** any failed gate stops the program; criteria are not relaxed afterwards. Any NaN, assertion failure or crash in the main run stops it and is reported. The driver refuses to run if the projection is missing or exceeds the remaining budget, refuses any cap above 12 h, and stops before any problem whose projected time would cross the cap.

## 5. GPU budget

12.0 h cumulative on the local RTX 4060, counting **everything** (equivalence, feasibility, speed re-check, main run, model loads) in `stage1_runs/gpu_ledger.jsonl`. No cloud or paid compute. No automatic expansion to more problems, models, budgets or datasets. If the main run is cut short by the cap, it is reported as **truncated** (the completed problems are a random subset of the pool; analyses use only problems completed for all five arms; no extension).

## 6. What is saved (per problem, per arm; all failures and inconclusive cases kept)

Emitted token ids, decoded text, score and extraction status, boundary fields and answer mask, steps, whether early stopped, route string (the KV-source letter per step), pass order, per-step branch argmax (`tF`, `tN`) and JS divergence for the hybrid cells, token/KV source letters, wall seconds and generation seconds, peak GPU memory, byte-cost accounting, and a manifest with git head, code SHA-256 values and library versions.

## 7. Statistical plan (declared before the main run)

Unit = problem; all arms on the same problems. `Y_a(i) = 1` iff arm `a` is `SOLVED` on problem `i`.

**Primary family (Holm-adjusted over five contrasts, paired, problem-level bootstrap):**
1. cache-only − FP16 = `mean(Y_co − Y_ff)` (damage reaching the output through stored K/V)
2. token-only − FP16 = `mean(Y_to − Y_ff)` (damage through the token choice)
3. **interaction** = `mean(Y_nn − Y_co − Y_to + Y_ff)` (equivalently `(NF4−FP16) − [(cache-only−FP16) + (token-only−FP16)]`; zero would mean additivity, nothing is assumed)
4. cache-only − NF4 (what remains when the token choice is also quantized)
5. flip oracle − cache-only (value of FP16 K/V on flip steps; both use the FP16 token)

Exact identity reported as description, not as attribution: `Y_nn − Y_ff = (cache) + (token) + (interaction)`. **No statement of the form "x % of the loss is due to the cache" will be made**; ratios of these contrasts are not estimands here.

- 95% percentile bootstrap CIs (10,000 resamples of problems, seed 20261023); two-sided sign-flip permutation p-values (10,000 permutations); Holm adjustment across the family; exact McNemar tables are descriptive.
- **Nulls:** equivalence bound ±3 pp (TOST, 90% CI). A difference is called absent only if the CI lies inside the bounds; otherwise the result is "inconclusive".
- Everything else (secondary contrasts such as token-only − NF4, per-class or per-length analyses) is **exploratory**.

**Descriptive analyses (never pooled with the family):**
- accuracy and `EXTRACTION_FAILED` counts per arm; paired 2×2 outcome tables against FP16 and NF4;
- **first divergence** from the FP16 reference trajectory (index of the first emitted token that differs): share of problems diverging, distribution, survival curves (share still identical to FP16), separately for answer-region and any position;
- **per-step hazard under identical history:** among steps whose emitted history so far equals the FP16 reference's, the probability that the cell's token differs from FP16's token, in step bins, with problem-clustered bootstrap CIs; compared across cache-only (cache path), token-only (token path) and NF4 (both). The sum of the two single-path hazards is shown next to the NF4 hazard as a description of non-additivity, not as a test;
- local branch disagreement (`tF ≠ tN`) per step in each hybrid cell;
- **how outcomes change when token histories are held identical:** by construction a problem whose emitted history through the scored answer region equals FP16's has an identical outcome; the share of such problems per cell is reported, and outcome transitions are reported only among problems that diverged, as a function of the divergence position;
- comparison with the pilot's flip-oracle observation: the same first-divergence statistics recomputed from the pilot's stored train-problem trajectories (50 problems, FP16 vs flip oracle) next to the Stage 1 test-pool numbers; contradictory or surprising outcomes (for example a cell that is more accurate than FP16, or a diverged problem that is solved while FP16 is not) are listed individually;
- runtime and memory per arm (mean/median seconds per problem, ms per step from the lean engine, byte-cost model, peak allocated memory, total GPU hours).

## 8. Limitations declared in advance

- One model (Qwen2.5-0.5B), one task, greedy decoding, one numerical stack; effects may not transfer.
- The 930 pool was examined before (section 0); results are exploratory replication.
- The minimum detectable paired difference at n = 930 is about 2.3-4.3 pp depending on the discordant-pair rate (NEXT_STUDY_DESIGN section 5); smaller effects cannot be resolved and nulls are only bounded.
- The KV intervention concerns whole-position K/V entries written by whole-network NF4 passes; it does not separate individual layers, heads or K from V.
- Timings come from a laptop GPU in whatever clock state it is in (host-launch-bound); bytes-moved is the reliable cost unit.
- Greedy trajectories are chaotic: small numerical differences can flip answers, so accuracy differences can arise from divergence without any systematic precision effect on correctness.

## 9. Protocol lock and amendments

`tools/lock_protocol.py --stage1` hashes the **committed** contents of this spec, the driver, gates, analysis, `rv2/*.py`, the tests, `stage1_sample.json` and the historical code the run imports, after the gates have passed and before the main run. The lock file and its SHA-256 are for private coauthor review and carry the provenance statement of section 0. Any later change is an **amendment** listed with its reason; analyses affected by it are labelled exploratory.

## 10. Amendments (made after the protocol text above was written; listed with reasons)

**A1 (2026-10-23, after the first feasibility run, before the protocol lock and before any evaluation run).** The auxiliary sanity check in `stage1_gates.check_feasibility` that every logged JS divergence lies in [0, ln 2] used a lower bound of exactly 0. The first feasibility run (10 GSM8K-train problems) logged one value of **-9.7e-9** (problem 1190, token-only cell, step 73), at a step where both branches chose the same token and neighbouring values were about 1e-6: float32 round-off in a quantity that is non-negative in exact arithmetic. The first run therefore reported `semantics_ok = False` for that reason only. Every criterion listed in section 4 (G4) was met: 0 violations of "token equals the argmax of its declared source" in both cells, step-1 branch labels consistent with the static arms, both hybrids diverge from FP16, and the flip oracle matches cache-only through its first flip. The JS check is not an equivalence or cache-consistency check and is not among the G4 criteria. The lower bound was changed to **-1e-6**; the original failing report is kept as `stage1_runs/feasibility/feasibility_report_first_run_FAILED_js_tolerance.json`. No GPU data were affected or re-run. JS divergence is not used in any primary estimand.

**A2 (2026-10-24, after the protocol lock, during the main run; authorized by the author's instruction "Option 1").** *A2 changes only the runtime projection and stopping procedure (plus monitoring and integrity checks), not experimental policies, sampling, outcomes, hypotheses or analysis.*

*Why.* The locked driver stopped itself after 155 of 930 problems (all five arms complete, ledger 2.099 h) because its live projection used only the mean of the *current process* (46.7 s/problem, 40 problems) and projected 12.15 h cumulative against the 12 h cap. The first process had run at 38.1 s/problem (115 problems). Per-step cost in the second process was 17-22 % higher in every arm, about 4 % of the difference is explained by a different mix of generated steps, and the cause of the rest is unknown (no clock or temperature telemetry was logged during that run; A2 adds it). The pooled estimate over all 155 problems is about 40.3 s/problem, i.e. roughly 8.7 h for the remaining 775 problems and about 10.8 h cumulative. A projection from 40 problems of one session is a noisy basis for abandoning a run, whereas the cap itself is a hard limit that can be enforced exactly.

*What A2 changes (all in `run_stage1.py` and the new `stage1_runtime.py`, `tools/verify_checkpoint.py`).*

1. **Stopping.** The projection-based stop is removed. The only stopping rule is the **absolute cumulative GPU ceiling of 12.0 h counting every earlier session** (ledger 2.099 h at the time of the amendment). The run stops *before* any problem for which `used + 1.25 x (longest problem seen in any session, scaled by the measured per-problem overhead) + 60 s > 12.0 h`. A watchdog thread additionally closes the ledger and terminates the process 20 s before the ceiling, even if a generation were to hang. A progress heartbeat every 15 s bounds the unrecorded time after a crash or power loss to about 15 s, which the next start charges to the ledger. If the ceiling is reached first, the experiment is reported as **truncated**: every complete problem is kept, the budget is not extended and no problems are sampled instead.
2. **Projection (informational only).** Completion is projected from the **pooled** per-problem time over all completed problems of all sessions. The current-session mean and the last-20-problem mean are logged separately (`stage1_runs/runtime_log.jsonl`, every 10 problems) with a flag for a sustained slowdown (last-20 / pooled > 1.15 over >= 20 problems), per-arm ms/step this session versus pooled, and the cumulative-hours projection from each rate. The projection never stops the run, and `run` no longer refuses to start when the central estimate exceeds the remaining budget.
3. **Per-problem integrity checks (stop immediately on any failure).** Before a problem's five records are appended: all five arms present with the same index; status in the allowed set; steps / token count / route string consistent; each hybrid's emitted token equals the argmax of its declared token source at every step; step-1 branch tokens equal the static FP16 / NF4 arms' first generated tokens; every hybrid KV letter equals its declared KV source; JS values finite and within [-1e-6, ln 2 + 1e-6]; the first token identical across arms; the flip oracle equals cache-only until its first flip; a problem whose emitted history is identical to FP16's has the same outcome as FP16 (an identical-history identity check, not an accuracy comparison); peak memory below 6 GB. A failure writes `stage1_runs/integrity_failure.json`, appends nothing for that problem and stops the run. The checks read no accuracy contrast.
4. **Hardware safety.** GPU telemetry (temperature, SM clock, power, P-state, throttle reasons) is sampled before every problem and logged every 10; the run stops if the GPU reaches 88 C.
5. **Checkpoint verification.** Before resuming, the first 155 records of each arm are compared line-by-line (SHA-256) with the snapshot taken at the pause (`stage1_runs/checkpoint_155_integrity.json`), and every arm's index sequence must equal the seeded run order exactly (no repeats, skips or reordering). The same check is repeated after the run.
6. **Per-session accounting.** Every process appends its session statistics (problems completed, seconds, mean s/problem, per-arm mean wall time and ms/step, stop reason, temperature and clock ranges) to its ledger line and to `stage1_runs/sessions.jsonl`. Sessions 1 (problems 1-115, 4394 s) and 2 (problems 116-155, 1873 s) were reconstructed from the record files and ledger (timing only).

*What A2 does not change.* The arms, the intervention semantics (`rv2/`), the seeded problem order and the sample, greedy decoding, the 256-token cap and early-stop rule, the records' contents, the 12 h budget, the statistical plan (section 7) and `analyze_stage1.py`. No outcome was inspected to motivate it: the decision rests on runtime and the pooled timing alone. The 155 records produced before A2 are used unchanged. The generation code path is byte-identical; only the order in which a problem's five records are written changed (all five are appended after the integrity checks pass, instead of each arm immediately).

*Consequences.* Analyses use all problems completed on all five arms. If the run is truncated, the achieved sample is a uniformly random subset of the 930 (a prefix of a seeded permutation), so no selection on outcomes or difficulty occurs, but the reported precision is that of the achieved n. Under the session-2 rate (about 46.8 s/problem) the ceiling would bind at about 915 problems; at the pooled rate the run completes with roughly 1.2 h to spare.
