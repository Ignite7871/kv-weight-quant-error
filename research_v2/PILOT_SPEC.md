# Pilot Specification (FINAL for the feasibility pilot) — Signal Timing and Precision-Budget Accounting

Version: final pilot protocol, incorporating the author decisions of 2026-10-22. Branch `research-recovery` (local commits only).

**Scope authorized:**
- local RTX 4060 only;
- at most **5 cumulative GPU-hours**;
- the 30-problem historical correctness gate, the 5-problem GPU equivalence checks, and the 50/50 GSM8K-train calibration/evaluation pilot;
- greedy decoding only.

The full study (300- or 1,319-problem runs), other models, other datasets and larger samples are **not** authorized.

**Provenance statement (for coauthor review).**
- This protocol is informed by prior **exploratory** findings on historical data: the lag-1 vs same-step entropy analysis, power calculations, and the observation that historical budgets count unscored post-answer steps.
- It is locked by a private dated SHA-256 hash shared with the coauthors. It is **not** an independent or public pre-registration and must never be described as one.
- Pilot accuracies are descriptive only. No method is selected on pilot accuracy.

---

## 0. Status of preparatory work (CPU)

| Item | Result |
|---|---|
| Read-only archive of the rejected package | `D:\Projects\Dynamic Precision Scaling\archive_tmlr13386_2026-10-22\`, 383 files, `MANIFEST_SHA256.csv` sha256 `776b7df04c0a37153376da3c66b607c2b52c56f3f75922eb3991c125b3b87bec`, zip sha256 `083964edf225a0fc0968c377a236383b5f1f2077fa404f4bf9e4cee43e51f051`. Working-tree files re-verified against the manifest before GPU work. |
| Alignment of lag-1 vs same-step entropy | independent FP32 CPU recomputation on 20 problems: **passed**; no label leakage (causal-mask test) |
| CPU test suite | `tests/run_cpu_tests.py --real`: 23/23 (20 original + `t5c` cost charging + `t3e` boundary isolation + `t8` tie rule / cost matching) |
| Mutation check | `tests/mutation_check.py`: 5/5 injected bugs detected (oracle restore skipped, wrong snapshot position, stale lag signal, inverted precision flag, cascade rollback disabled) |
| Pipeline rehearsal | every pilot step run on CPU with the real model (2 problems, 12 tokens) into a scratch directory, including the historical-implementation arms, calibration, matched-random derivation and the cap refusal |

## 1. Policies (exact)

Notation:
- The prompt is processed by an **FP16 prefill for every policy** (decision 6). Token 0 = argmax of the prefill's last logits; it is not routed.
- Routed step *s* (1…255) consumes token *s−1* and produces token *s*.
- Greedy decoding throughout (decision 7).

| Arm | Kind | Rule for step *s* | Information used | Status |
|---|---|---|---|---|
| `fp16` / `nf4` | pre | always FP16 / always NF4 | none | baselines |
| `lag_entropy_B10/B30` | pre | FP16 iff `calculate_entropy(p_{s−1}) > τ` | logits that produced token *s−1* | deployable, zero cost |
| `lag_margin_B10/B30` | pre | FP16 iff `calculate_margin(p_{s−1}) < τ` | same | deployable, zero cost |
| `rand_*` | pre | FP16 iff `rng.random() < p`; one draw per step; `rng = random.Random("rv2\|<arm name>\|<problem index>")` | per-problem RNG only | deployable control |
| `cascade_nf4_margin_B10/B30` | cascade | NF4 pass → if `calculate_margin(p_s^NF4) < τ`: crop the cache to its pre-step length, recompute in FP16, keep the FP16 output and KV | NF4 output of step *s* | **deployable**; extra pass per escalation |
| `oracle_flip` | oracle | both passes; FP16 iff argmax(p_s^FP16) ≠ argmax(p_s^NF4) | both outputs of step *s* | **non-deployable — primary diagnostic reference** (decision 5) |
| `oracle_js_B10` | oracle | both passes; FP16 iff JS(p_s^FP16, p_s^NF4) > τ | both outputs | non-deployable, secondary |
| `oracle_fp16_entropy_B10` | oracle | both passes; FP16 iff `calculate_entropy(p_s^FP16) > τ` | FP16 output of step *s* | non-deployable, secondary (ReQAT-style signal) |
| `hist_lag_entropy_p10/p30/p50` | pre | historical WikiText-calibrated thresholds 0.06795654445886612 / 0.9259277582168579 / 1.943 | as lag_entropy | **continuity comparison** (decision 10), gate0 only |
| `histimpl_*` | — | the **untouched** historical `DynamicQwen.generate` at the same thresholds | — | implementation cross-check, gate0 only |

Oracles are myopic: they use only step *s* outputs, never later tokens. **No oracle is a guaranteed upper bound on final-answer accuracy** (decision 5); they are reported as diagnostic references.

## 2. Generation, answer boundary and scoring

**Generation is unchanged from the historical protocol** for every arm:
- stop after EOS or 256 generated tokens;
- no early stopping;
- the answer boundary is computed **only after** generation finishes.

The engine and policy modules do not import the boundary code, and the driver never passes an early-stop function (test `t3e`). Boundaries therefore cannot leak future information into any decision.

**Answer boundary:**
- `boundary_char` = start of the first match of the historical extractor's own `CONTINUATION_DRIFT_RE` (`\n\s*Question\s*:`, case-insensitive) in `decode(ids, skip_special_tokens=True)`.
- Types: `drift_marker`, `eos` or `cap`.
- Answer-relevant step: routed step *s* whose token's start offset `len(decode(ids[:s]))` is below `boundary_char`.

**Decision boundary (sensitivity):** the end of the number that the extractor's strategy A reads after the first `####` in the drift-truncated text; otherwise equal to the answer boundary.

**Scoring:**
- the historical `extract_candidate_answer` on the full text, with tolerance 1e-2;
- identity of extraction on the truncated text is asserted per record;
- decision-text identity is recorded per record.

**Recorded failure cases:** `cap` boundary, `offsets_consistent=False`, `EXTRACTION_FAILED`, extraction mismatches, `no_hash` decision type.

## 3. Budgets and costs

| Role | Definition |
|---|---|
| **Primary (continuity)** | historical full-generation FP16-step share = FP16 steps / all routed steps (decision 1) |
| Secondary | answer-only FP16-step share, with random controls matched on it |
| Sensitivity | decision-only FP16-step share |
| **Primary engineering budget for the cascade** | total computational cost (decision 4), §3.1 |
| Reported separately | realized wall-clock latency and FP16-step share |

### 3.1 Total computational cost (bytes moved in the decode phase)

Per executed pass at step *s*, with cache length L_s = prompt length + s − 1:

| Term | Bytes |
|---|---|
| weights | W_routed(precision) + W_unrouted |
| KV traffic | (3·L_s + 2)·kvb |
| oracle snapshot (per step) | 2·kvb |
| oracle restore | (2·L_s + 1)·kvb |
| cascade crop | 0 (a view) |

Measured constants: W_FP16,routed = 715,708,416 B; W_NF4,routed = 184,646,976 B (incl. quantization state); W_unrouted = 272,357,248 B (tied embeddings/lm_head, norms); kvb = 12,288 B per token.

- The prefill is identical across policies and excluded (its time is reported).
- Policy-signal computation is not charged (< 1 MB).
- Cost is reported in bytes and in FP16-pass equivalents.
- Every executed NF4 pass is charged, including the NF4 pass a cascade discards. So is every FP16 recomputation and every oracle snapshot/restore. `t5c` verifies the charging against the backend's raw call log.

## 4. Pilot design

### 4.1 Sample and seeds (`pilot_sample.json`, unchanged)

- **Train:** `random.Random("rv2-pilot-train-2026-10-22").sample(range(3, 7473), 100)`.
  - The first 50 drawn, sorted, are the **calibration** set; the last 50 are the **evaluation** set.
  - The equivalence set is the first 5 evaluation indices.
- **Correctness gate:** `random.Random("rv2-gate0-2026-10-22").sample(<historical 930 pool>, 30)` test problems.
  - Only historical deterministic arms run there.
  - Nothing is tuned on test data (decision 2).

### 4.2 Blocks, run order and stop conditions

1. **gate0** (30 test problems): `fp16, nf4, hist_lag_entropy_p10, _p30, _p50`, then `histimpl_lag_entropy_p10, _p30, _p50`.
   - `histimpl_fp16` and `histimpl_nf4` run only if a static arm disagrees with the stored outcome.
   - **Target: exact reproduction of all 30 stored outcomes for every historical arm (decision 8).** Every disagreement is investigated:
     - run the untouched historical implementation on that problem;
     - **rv2 ≠ historical implementation** ⇒ an implementation defect ⇒ **stop**;
     - **rv2 = historical implementation ≠ stored outcome** ⇒ historical-run drift, documented per problem; the pilot continues because internal comparisons stay valid.
   - Token-level identity between `hist_lag_entropy_*` and `histimpl_*` must hold on 30/30, or stop.
2. **equiv** (5 train problems): the cascade with τ=2 (always escalate) and the oracle with JS > −1 (always FP16) must be token-identical to `fp16`; the cascade with τ=−1 and the oracle with τ=10 must be token-identical to `nf4`. **Any failure ⇒ stop.**
3. **calib** (50 train problems): `fp16, nf4, oracle_flip`.
4. **calibrate** (CPU), §4.3.
5. **eval** (50 train problems), in this pre-declared priority order; later items are dropped first if the GPU cap binds:
   1. `fp16`, `nf4`, `oracle_flip`, `lag_entropy_B10`, `lag_margin_B10`, `cascade_nf4_margin_B10`
   2. matched random controls for B10 (derive-random B10)
   3. `lag_entropy_B30`, `lag_margin_B30`, `cascade_nf4_margin_B30`, then matched controls for B30
   4. `oracle_js_B10`, `oracle_fp16_entropy_B10` (secondary references)
   5. answer-only matched controls for B10 (secondary)
   6. extra seeds s2 and s3 for the primary B10 control
6. **analyze** (CPU).

### 4.3 Calibration (open-loop only; GSM8K-train calibration set only)

- **Closed-loop calibration is not used in this pilot** (decision 3 allows it only if fixed beforehand; it is not adopted, to stay within the GPU cap). The out-of-sample budget error is a pilot outcome that informs the full study.
- Signals are the **exact decision-dtype values** the policies compare (FP16 on GPU), logged as `Hd_*` / `Md_*`.

| Threshold | Calibration source |
|---|---|
| lag entropy, lag margin | `calib/nf4` trajectory (`Hd_lag1`, `Md_lag1`) |
| cascade | `calib/nf4` `Md_nf4` (exact for the cascade until its first escalation) |
| oracle_js, oracle_fp16_entropy | `calib/oracle_flip` trajectory (`js`, `Hd_fp16`) |

**Fixed tie rule (decision 9).**
- Candidates are the distinct observed signal values plus ±∞.
- Pick the candidate whose calibration-set FP16 share is closest to the target.
- On ties, take the lower FP16 share, then the smaller τ.

Realized budgets are reported for the calibration source and for the evaluation set.

### 4.4 Matched random controls (pre-declared; derived from realized eval budgets)

| Control | Matched to | Unit | Role |
|---|---|---|---|
| `rand_full_lagE_B10_s1` (and `s2`, `s3` last) | realized full share of `lag_entropy_B10` | full | primary |
| `rand_full_lagM_B10_s1` | `lag_margin_B10`; shared with the entropy control if the realized shares differ by ≤ 1.0 pp (then p = their mean) | full | primary |
| `rand_cost_casc_B10_s1` | p solving: random expected total cost = cascade realized total cost (§3.1) | cost | primary for the cascade |
| `rand_full_flip_s1` | realized full share of `oracle_flip` | full | diagnostic comparison |
| `rand_full_lagE/M_B30_s1`, `rand_cost_casc_B30_s1` | as above at B30 | full / cost | primary at B30 |
| `rand_ans_lagE/M_B10_s1` | realized answer-only share (same sharing rule) | answer | secondary |

If a cascade's realized total cost exceeds that of static FP16, the cost-matching equation gives p > 1. That happens when the escalation fraction exceeds about 1 − W_NF4/W_FP16 ≈ 0.54. In that case no cost-matched random control exists: the control is recorded as **infeasible** and static `fp16` (cheaper, and FP16 on every step) is the cost-dominating comparator. The CPU rehearsal with placeholder thresholds triggered this rule.

Analyses are descriptive: solved counts, per-problem paired outcomes, discordant pairs. Exact McNemar p-values are shown only as descriptive statistics; no claims of superiority.

## 5. Leakage prevention (verified on CPU, repeated on GPU by the equivalence block)

- Policies see only the fields in their declared access set; anything else raises `InformationAccessError`.
- Deployable policies cannot declare the current-step FP16 output.
- The current-step FP16 output is computed only by oracles, or by the cascade **after** its decision.
- Boundaries are retrospective (§2).
- **Cache equivalence.** Bitwise tests show the cascade's crop/recompute and the oracle's snapshot/restore leave exactly the cache a single-pass router with the same routes would leave.
- **Counterfactual test.** Corrupting the FP16 weights after the prefill leaves every cascade decision unchanged up to and including the first escalation.

## 6. Pass/fail criteria for recommending the full study

| ID | Criterion | Type |
|---|---|---|
| G1 | CPU suite 23/23 and mutation check 5/5 on the locked code | hard (met) |
| G2 | gate0: rv2 ≡ untouched historical implementation (token ids) on 30/30 for p10, p30 and p50; every disagreement with stored outcomes investigated and classified | hard |
| G3 | equiv: 4/4 equivalence arms token-identical on 5/5 problems | hard, stop on failure |
| G4 | boundary: truncated-text extraction identical on 100% of records; offset inconsistencies 0 or each explained; manual 20-record audit correct | hard |
| G5 | no crash, NaN or cache-length assertion | hard |
| G6 | measured-rate projection for the full study is reported, together with the arm set that fits a stated budget | feasibility input |
| B1 | out-of-sample budget error: |eval − target| ≤ 3 pp (B10), ≤ 5 pp (B30) | informs the calibration design |
| B2 | answer-only share gap of each targeted policy vs its full-matched random control | finding |
| B3 | random arms realize within 1.5 pp of p | sanity |
| D1 | share of `cap` boundaries | finding |

Accuracy is never a criterion.

## 7. Existing historical tests

They cannot detect off-by-one routing, wrong-branch KV, an inverted precision flag, leakage, or boundary/accounting errors. The historical KV dtype check cannot fail by construction. The new suite and mutation check cover these failure modes (details in the previous version of this section, unchanged).

## 8. GPU budget control

- Every GPU process appends its full wall time, including model loading, to `pilot_runs/gpu_ledger.jsonl`.
- Before each arm, the driver projects its time as 1.15 × the measured per-problem rate for that policy kind × the problems remaining. It refuses to start the arm if ledger + elapsed + projection exceeds **5.0 h**.
- When the cap binds, remaining priority items are dropped and reported as not run.

Pre-measurement estimate for the full priority list: about 4.3–4.6 h. The lowest-priority items may therefore be dropped.

## 9. Outputs

- `pilot_runs/<block>/<arm>.jsonl`: one record per problem, with ids, text, routes, passes, per-step trace (float32 and decision-dtype signals), boundary fields and masks, score, full accounting incl. cost, wall seconds, peak memory.
- `<arm>.manifest.json`: provenance and code hashes.
- `pilot_thresholds.json`, `pilot_random_plan.json`, `gpu_ledger.jsonl`.
- `analyze_pilot.py` output: `pilot_report.json`, a per-problem paired-outcome CSV, and the boundary audit file.

## 10. Decisions recorded (2026-10-22) and remaining open issues for the full study

Decisions 1–11 are implemented as above. Open for the full study:
- whether to adopt closed-loop calibration, depending on B1;
- the number of random seeds;
- the headroom stopping rule (the 300-problem headroom experiment is not run in this pilot, decision 11);
- whether the cost unit should incorporate measured latency in addition to bytes.

## 11. Protocol lock

1. Commit.
2. Run `python research_v2/tools/lock_protocol.py`. It hashes the committed contents, refuses if the tree is dirty, and writes `PROTOCOL_LOCK.txt`.
3. Record the lock file's SHA-256 in `COAUTHOR_LOCK_NOTE.md` for private coauthor review.

Changes after locking are amendments, listed with reasons. Analyses affected by an amendment are labelled exploratory.
