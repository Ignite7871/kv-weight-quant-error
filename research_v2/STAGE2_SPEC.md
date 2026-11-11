# Stage 2 specification — protocol-locked held-out validation of the Stage 1 cache/token experiment

Dated 2026-10-25, written **before any held-out problem was evaluated** and before any GPU use in Stage 2. Local branch `research-recovery` only; nothing is pushed or published.

## 0. Status and provenance (read first)

- **What this is.** A *protocol-locked held-out validation*: the five Stage 1 arms are run, unchanged, on the 389 GSM8K test problems that Stage 1 did not use. It is the first sample on which the Stage 1 contrasts can be *tested* rather than *estimated after the fact*.
- **What this is not.** Not an independent or public pre-registration (the lock is private, see `STAGE2_LOCK_NOTE.md`), and not a test of a hypothesis that was formed without looking at data.
- **Exploratory origin of the hypotheses.** Every hypothesis below was formed after (i) the 2026-10-22 pilot (flip oracle diverged from FP16 on 50/50 train problems, median first divergence at token 20), (ii) the historical routing experiments on the 930-problem pool (FP16 336 vs NF4 243 solved), and (iii) the **Stage 1 results on those 930 problems** (`STAGE1_REPORT.md`). The Stage 1 point estimates are the source of the directions below. The held-out sample is the only data in this programme on which they have not been examined.
- **Author approval (2026-10-25):** the author approved this protocol, keeping all 389 problems in the primary validation and requiring the sensitivity analysis excluding the 47 (342 problems), the five contrasts, Holm adjustment, replication categories and the secondary combined analysis as written.
- **The 389 are not completely untouched.** See §2: two problems (test indices 4, 5) were printed in an 8-problem console probe, and 47 problems (test indices below 200, including 4 and 5) had their *gold* solutions token-labelled by a teacher-forced FP16/INT4 argmax comparison in `results/eqf_spike_dataset_*.csv`. No generated-answer correctness outcome was retained for any held-out problem, but the reserved set must not be described as completely untouched. All 389 are evaluated as authorized; a pre-declared sensitivity analysis (§6.6) excludes the 47.
- **Hypotheses are fixed now.** No hypothesis, contrast, threshold, exclusion or analysis choice will be changed after held-out results are seen. A change forced by a defect is an *amendment* (§9) and affected results are labelled exploratory.

## 1. Question and hypotheses

**Objective.** Test whether the Stage 1 findings about how emitted-token differences and quantization-induced KV-cache differences contribute to changes in downstream GSM8K behaviour replicate on held-out problems. Estimands are contrasts of accuracy and of trajectory divergence under the specific interventions of `STAGE1_SPEC.md` §2, conditional on greedy decoding, Qwen2.5-0.5B, this prompt/benchmark and the FP16 + bitsandbytes-NF4 stack. **No percentage of an accuracy loss is attributed to the cache or to the token path**; the identity `total = cache + token + interaction` is reported as description only.

Let `Y_a(i) = 1` iff arm `a` is `SOLVED` on problem `i`; arms: `ff` FP16, `nn` NF4, `co` cache-only (FP16 token, NF4 KV), `to` token-only (NF4 token, FP16 KV), `fo` flip oracle. Primary family (the same five contrasts as Stage 1; Stage 1 estimate, pp, in brackets):

| | Contrast | Stage 1 estimate | Hypothesis (direction fixed now) |
|---|---|---|---|
| H1 | `co − ff` | −2.37 | negative (cache path alone loses accuracy) |
| H2 | `to − ff` | −3.55 | negative (token path alone loses accuracy) |
| H3 | `nn − co − to + ff` (interaction) | −4.09 | negative (combined loss exceeds the sum of single-path losses) |
| H4 | `co − nn` | +7.63 | positive (cache-only retains accuracy that NF4 loses) |
| H5 | `fo − co` | −0.75 | equivalent within ±3 pp (FP16 K/V on flip steps adds nothing measurable beyond FP16 tokens) |

Secondary, exploratory contrasts (never pooled with the family): `nn − ff` (total gap), `to − nn`, `fo − ff` (does the flip oracle preserve FP16 accuracy; Stage 1: −3.12), `fo − nn`.

## 2. The held-out sample

- **Frozen ids:** `stage2_sample.json` (`heldout389`, sorted; `run_order`, a permutation of them drawn with `random.Random("rv2-stage2-order-2026-10-25")`, so any prefix is a uniformly random subset). It is the complement of the 930-problem pool in the 1,319-problem GSM8K test split (`reserved389` in `stage1_sample.json`).
- **How the split arose (historical).** The pool = test indices 8–37 (gate set) + 250 indices sampled with seed 123 + 650 indices sampled with seed 456, both without replacement (`scripts/gsm8k_routing_comparison_n250.py`, `..._n930.py`). The 389 are the unsampled remainder: a random remainder of the test split, not selected on difficulty or outcome. (Indices 0–7 went to an 8-problem console probe and were later drawn into the pool or left in the remainder at random.)
- **Disjointness and absence of stored outcomes** are checked by `tools/check_heldout_clean.py` (output `validation_logs/stage2_heldout_clean_check.json`): the 389 and 930 are disjoint and together equal the 1,319-problem test split; no Stage 1, pilot or gate record has a held-out test index; across all 241 data files in the repository, every file containing a held-out index is in a closed, explained list (TriviaQA indices from another dataset; pilot *train*-split indices that coincide numerically; the two gold-trajectory labelling files and their supplement copies). Any unexplained hit fails the check.
- **Exposure and sensitivity.** Exposed held-out problems: probe (console only) {4, 5}; gold-trajectory teacher-forced labelling: the 47 held-out problems with test index < 200 (listed in `stage2_sample.json`). The sensitivity analysis of §6.6 repeats the primary family without those 47 problems.
- **No pool problem is evaluated in Stage 2;** the driver refuses any index outside the 389 or inside the 930.

## 3. Interventions and implementation (frozen, identical to Stage 1)

Arms, token-source and KV-source semantics, the per-step crop trick, early stop at the causal drift marker, greedy decoding, the 256-token cap, the prompt template (3 few-shot train examples), the scoring (`SOLVED` only counts), the per-problem integrity checks, the hardware-safety stop and the absolute-ceiling logic are the code of the locked Stage 1 / amendment-A2 commit, imported unchanged: `rv2/*.py`, `run_stage1.py`, `stage1_gates.py`, `stage1_runtime.py`, `analyze_stage1.py` (`git diff` against the Stage 1 lock shows no change to `rv2/`, `src/`, `scripts/`, `analyze_stage1.py`, `stage1_gates.py`; `run_stage1.py` and `STAGE1_SPEC.md` are as of amendment A2). `run_stage2.py` only (a) defines the sample, (b) re-points the Stage 1 driver at `stage2_runs/` and its own GPU ledger, (c) sets the 5.5 h cap. Problems are run problem-major in the seeded order with the arm order rotated per problem (clock-drift balance); all five records of a problem are appended only after its integrity checks pass.

## 4. Preconditions before the held-out run (each a hard gate)

1. CPU: Stage 1 suite (18/18 with `--real`), Stage 1 and A2 mutation checks (6/6, 15/15), analysis tests, and the Stage 2 tests (sample, paths, ledger, replication logic); all must pass.
2. The data-cleanliness check of §2 passes.
3. Everything above is committed and the **private dated protocol lock** (`STAGE2_PROTOCOL_LOCK.txt`, `STAGE2_LOCK_NOTE.md`) is created *before any GPU use*.
4. **Short GPU correctness check** (`run_stage2.py gate-equiv`): the Stage 1 equivalence block on the first three pilot train problems — pure cells ≡ static arms ≡ the untouched historical `DynamicQwen.generate` (tokens and caches bitwise), hybrid cells ≡ the independent deep-copy reference and a single-precision teacher-forced replay, early stop identical. Any failure stops Stage 2.
5. **Speed re-check** (`run_stage2.py speed`) on fixed train problems, and the **runtime projection** (`run_stage2.py project`) from the measured Stage 1 per-session rates: the run proceeds only if the central projection of cumulative Stage 2 GPU time (checks and model loads included) is within 5.5 h.

## 5. GPU budget and stopping

- Hardware: the local RTX 4060 laptop GPU only; no cloud or paid resources.
- **5.5 h additional cumulative GPU time**, including the correctness and speed checks, counted in `stage2_runs/gpu_ledger.jsonl` (separate from Stage 1's 10.885 h). The ceiling is absolute: the run stops before any problem whose worst-case duration (1.25 × the longest problem seen in Stage 1 or Stage 2, scaled by measured overhead) plus a 60 s margin would cross it, and a watchdog closes the ledger and exits 20 s before it. A run cut short is **truncated**: complete records are kept, the budget is not extended and no replacement problems are drawn. A 15 s heartbeat bounds unrecorded time after a crash.
- Stop immediately on any failure of the per-problem integrity checks (token = argmax of its declared source; step-1 labels consistent with the static arms; KV letters; JS range; flip oracle ≡ cache-only until its first flip; first token identical across arms; identical history ⇒ identical outcome; memory), on a failed equivalence check, or at GPU temperature ≥ 88 °C.
- **Monitoring policy.** During the run only integrity, accounting and hardware telemetry are examined. **No accuracy, no arm contrast and no interim analysis are computed or looked at**, and nothing about the design is adapted to outcomes.

## 6. Statistical plan (declared before the held-out run)

Unit = problem; all arms on the same problems; only problems complete on all five arms are analysed. The held-out analysis uses `analyze_stage1.analyze` unchanged (so it is computed by the same code as Stage 1): 95 % percentile bootstrap CIs (10,000 resamples of problems, seed 20261023), two-sided sign-flip permutation p-values (10,000), 90 % CIs for the ±3 pp equivalence (TOST) judgement.

**6.1 Primary family and multiplicity.** H1–H5, held-out sample, **Holm adjustment over these five** two-sided tests. **Both raw and Holm-adjusted p-values are reported for every contrast**; the confirmatory reading of H1–H4 is the Holm-adjusted one, and H5 is judged by equivalence (90 % CI within ±3 pp) rather than by a p-value. Equivalence is declared only if the CI lies within the bounds; otherwise "inconclusive".

**6.2 Descriptive analyses on the held-out sample (never pooled with the family):** accuracy and `EXTRACTION_FAILED` per arm; paired 2×2 outcome tables vs FP16 and NF4; counts of every joint five-arm outcome pattern and all pairwise outcome-disagreement counts; per-problem table (`paired_outcomes.csv`); first-divergence share/median/quartiles from the FP16 trajectory and survival curves; per-step divergence hazard under identical history (bins 1–20, 21–60, 61–256) with bootstrap CIs; local branch disagreement; outcome transitions among diverged problems by divergence position; contradictory cases; runtime and memory per arm.

**6.3 Replication assessment (pre-declared categories).** For each primary contrast the report tabulates the Stage 1 estimate, the held-out estimate and CIs, the held-out raw and Holm p, and the *held-out minus Stage 1* difference with a 95 % independent-bootstrap CI and a permutation p (labels permuted over the pooled 1,319 problems). Categories for the directional contrasts H1–H4:
 1. **replicated:** same sign as Stage 1 and Holm-adjusted p < 0.05;
 2. **same direction, raw p < 0.05, not Holm-significant;**
 3. **same direction, not significant** (inconclusive: a non-significant result with n = 389 is not evidence of absence, see §6.7);
 4. **opposite (or zero) point estimate, compatible with Stage 1** (difference CI includes 0): statistical uncertainty;
 5. **opposite direction and inconsistent with Stage 1** (difference CI excludes 0): directional disagreement.
 For H5: **equivalence replicated** (90 % CI within ±3 pp) / **not demonstrated** (inconclusive) / **contradicted** (90 % CI excludes 0). "Statistical uncertainty" (categories 3–4) is kept distinct from "disagreement in direction" (category 5).

**6.4 Held-out and Stage 1 results are reported separately** and side by side (accuracy, first-divergence summaries, patterns).

**6.5 Secondary combined analysis (n = 1,319).** The same primary family on Stage 1 ∪ held-out, labelled **secondary and not independent confirmation**: it contains the exploratory Stage 1 sample from which the hypotheses were drawn. It is reported for precision, not for confirmation.

**6.6 Pre-declared sensitivity analysis.** The primary family on the held-out problems excluding the 47 whose gold solutions were previously token-labelled (test index < 200). Secondary.

**6.7 Design power (`validation_logs/stage2_design_power.json`, computed from Stage 1 data only).** If the Stage 1 distribution were the truth, a sample of 389 would give: H1 (co−ff, 80 %-power MDE 5.5 pp): P(raw p<.05) 0.23, P(Holm<.05) 0.10; H2 (to−ff, MDE 5.9): 0.40 / 0.21; H3 (interaction, MDE 7.5): 0.34 / 0.18; H4 (co−nn, MDE 6.0): 0.95 / 0.85; H5 (fo−co): P(90 % CI within ±3 pp) 0.53. Stage 1 estimates are probably optimistic (they drew the attention), so the real power is lower. **Consequence, declared in advance:** a non-significant held-out result for H1–H3 is expected with substantial probability even if the Stage 1 effects are real and must be reported as inconclusive, not as failure to replicate; the replication reading rests on the *direction and size of the held-out estimates with their CIs and on the heterogeneity test*, and on the secondary combined analysis for precision. Conversely, a held-out estimate whose CI excludes the Stage 1 estimate is informative.

## 7. Outputs

`stage2_analysis/`: `stage2_report.json` (all numbers), `heldout/` (`heldout_report.json`, `paired_outcomes.csv`, `divergence_survival.csv`, `survival.png`), `combined_secondary/`, `run_accounting/` (sessions, runtime log with telemetry, GPU ledger, final accounting, SHA-256 of the five record files); `STAGE2_REPORT.md`.

## 8. Limitations declared in advance

- One model, one task, greedy decoding, one numerical stack; the held-out sample has 389 problems (MDE 5.5–7.5 pp for H1–H3), so most non-null Stage 1 effects cannot be confirmed individually with high probability.
- The hypotheses come from Stage 1 data; the held-out sample is the first untouched test, but the 389 are not perfectly virgin (§2) and the protocol lock is private, not public.
- The cache intervention swaps whole-position K/V written by whole-network NF4 passes; it does not isolate layers, heads, or K versus V. Token and cache interventions change trajectories, so contrasts are between trajectories, not matched positions.
- Greedy trajectories are chaotic; accuracy differences can arise from divergence without a systematic precision effect on correctness.
- Timings depend on laptop clock state; byte cost is the reliable unit.

## 9. Protocol lock and amendments

`tools/lock_protocol.py --stage2` hashes the committed contents of this spec, `run_stage2.py`, `analyze_stage2.py`, `stage2_sample.json`, the Stage 2 tests and tools, and every Stage 1 file the run imports (including `rv2/*.py`, `run_stage1.py`, `stage1_runtime.py`, `stage1_gates.py`, `analyze_stage1.py` and the historical code). It is created after the CPU gates pass and **before any GPU use**; `STAGE2_LOCK_NOTE.md` is for the coauthors, privately. The lock is the author's to share; the assistant sends nothing. Any later change to a locked file is an amendment, listed below with its reason; analyses affected by it are labelled exploratory.

### Amendments

*(none at the time of the lock)*
