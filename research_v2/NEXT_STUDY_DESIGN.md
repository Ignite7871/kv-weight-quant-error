# Next-Study Design (Tasks B and C) — DRAFT for author decision, 2026-10-23

**Status:** proposal only. No accuracy experiment has been run or authorized.
- Cost figures use the performance measured on 2026-10-23 (`perf/PERF_REPORT.md`): lean engine mode, Balanced plan, ±20%.
- **Prior exploration.** Everything below is informed by earlier exploratory analyses: the lag-1 vs same-step entropy analysis on gold trajectories of test problems 0–199, the historical runs on the 930-problem test pool, and the pilot. The 930 pool and 30 of its problems (gate0) have been looked at under several policies.
- Any new protocol will therefore be **privately locked, informed by prior exploration, and not pre-registered**. Where an estimand repeats an exploratory finding on new data, it is labelled a *replication with pre-declared estimands*, not a confirmatory discovery.

---

## 1. The answer-budget confound: what it is and what it is not

### 1.1 Exposure

- In historical and pilot generations, only 39–46% of decode steps fall before the answer boundary; the rest is invented post-answer text that is never scored.
- Targeted policies do not spread FP16 evenly over the generation. A random policy does: its answer-only share equals its overall rate in expectation.
- So "budget-matched on the full generation" does not mean "matched in the text that is scored".

**FP16 share, full generation vs answer-only (measured):**

| Policy | Data | Full | Answer-only | Answer − full |
|---|---|---|---|---|
| entropy, historical p50 τ | test, gate0, n=30 | 6.6% | 5.2% | −1.4 pp (−21% rel.) |
| entropy, historical p30 τ | test, gate0 | 22.0% | 20.3% | −1.7 pp (−8%) |
| entropy, historical p10 τ | test, gate0 | 58.5% | 59.1% | **+0.6 pp** (+1%) |
| lag entropy B10 | train, pilot, n=50 | 8.2% | 6.3% | −1.9 pp |
| lag margin B10 | train, pilot | 9.6% | 6.9% | −2.7 pp |
| flip oracle (non-deployable) | train, pilot | 5.0% | 3.4% | −1.6 pp |
| random controls | pilot | p | ≈ p (±0.4 pp) | ≈ 0 |

**The sign depends on the budget.** At low FP16 budgets, targeted policies under-spend in the scored text. At the historical p10 threshold (59% FP16), entropy routing slightly over-spends there.

### 1.2 How large an accuracy bias could the exposure gap produce? (model-based, exploratory)

- On the 930-problem pool, the historical random-allocation arms rise from 26.1% (0% FP16) to 26.6% (6.5%), 29.1% (21%), 32.2% (43%), 30.9% (59%) and 36.1% (100%).
- That is about **0.08–0.14 accuracy points per point of FP16 share** at low budgets, noisy and non-monotone.
- Multiplied by the exposure gaps above, the predicted bias of a full-generation-matched comparison is about **0.1–0.4 pp at B10**, and about **+0.05–0.1 pp in the entropy router's favour** at the historical p10 point (+2.58 pp).
- **Implication:** the confound is real and should be removed by design. But it is unlikely to have materially changed the historical conclusions, and its accuracy effect at B10 is predicted to be far below what any affordable experiment can resolve (minimum detectable effect ≈ 2.4–2.9 pp, §5).

### 1.3 Limitations of this reasoning

- (a) The adjustment assumes FP16 steps inside the answer are equally valuable whichever policy chooses them. Targeting is precisely the hypothesis that they are not, so the adjustment is a scale estimate, not a substitute for a direct comparison.
- (b) The slope comes from random arms with large sampling error.
- (c) The pilot's exposure numbers are from 50 train problems, and gate0 has 30 test problems.
- (d) The answer boundary is defined by the historical extractor's drift-marker regex. For generations without a marker (2–8% `cap` cases) the whole generation counts as answer.

### 1.4 What can and cannot be matched causally

- **The realized answer boundary is future information.** It must never enter a deployable routing decision. All routing rules below use only past tokens and logits.
- **A causally enforceable answer-only budget exists: stop generation at the drift marker.**
  - The stopping rule is causal: it uses only already generated text.
  - It leaves every pre-boundary token unchanged (prefix stability, test `t6b`), so outcomes are identical to full generation.
  - Under it, "FP16 share over generated steps" is the answer-only share, plus the 1–2 marker tokens.
  - Thresholds calibrated on this accounting (τ_A) and random rates (p) fixed beforehand give a **prospective comparison matched in expectation**. This is a genuine intervention-level comparison.
- **Exact per-run matching is impossible prospectively**, because realized shares vary. When a random control's rate is set to a policy's *realized* share on the evaluation problems (two-stage, as in the pilot), the comparison is a **retrospective, adjusted comparison** and is labelled as such.
- **Historical full-generation accounting requires full generation**, at least for the targeted arm whose realized full share is reported. Random arms never need it: their full-generation share is p in expectation, and their outcome is identical with early stopping.

## 2. Research questions and what kind of evidence each gets

| Question | Evidence type | Primary estimand |
|---|---|---|
| Q1. Does full-generation budget matching give systematically different conclusions from answer-only matching? | policy comparison (intervention) plus model-based adjustment | D = Δ_A − Δ_F, where Δ_X = accuracy(policy calibrated under accounting X) − accuracy(random at matched p) |
| Q2. Do previous-step entropy or margin beat random allocation once the relevant budget is matched? | policy comparison | Δ_A (co-primary) and Δ_F (co-primary, historical continuity) per policy |
| Q3. Does quantization-induced cache divergence explain the flip oracle's failure? | mechanistic intervention (non-deployable factorial) plus descriptive trajectory analysis | cell contrasts of the token-source × KV-source factorial (§3.3) |
| Q4. How much of the confidence–EQF association depends on causal alignment? | descriptive (no policy comparison) | Spearman ρ and AUC of signal lag k ∈ {0, 1, 2} vs EQF on self-generated trajectories |

Policy comparisons (Q1, Q2, Q3 contrasts) and descriptive analyses (Q4, exposure, trajectory divergence) are reported in separate sections and never pooled.

## 3. Arms

Common settings for every arm:
- Qwen2.5-0.5B, GSM8K **test**, greedy decoding, max 256 new tokens, FP16 prefill, historical extractor and scoring;
- lean engine mode, with a 20-problem instrumented spot-check per arm for timing and traces where Q4 needs them;
- target budget **B10** = 10% FP16 (B30 only in the comprehensive design);
- **calibration on GSM8K-train only**: the existing pilot calibration trajectories already contain everything needed for both τ_F (full accounting) and τ_A (answer accounting, via the masks), at **zero GPU cost**. Open-loop, with the fixed tie rule.

### 3.1 Q1/Q2 policy arms (all deployable)

| Arm | Threshold | Generation | Purpose |
|---|---|---|---|
| `lagE_F` | τ_F: FP16 share 10% over full generation (calibration set) | full (needed to report realized full share) | historical-style arm |
| `lagE_A` | τ_A: FP16 share 10% over answer-only (stop-at-marker) steps | early stop | answer-budgeted arm |
| `lagM_F`, `lagM_A` | same for margin | full / early stop | moderate design and above |
| `rand_p10_s1…sK` | p = 0.10 (prospective) | early stop | shared control: matched in expectation to lagX_F on full accounting and to lagX_A on answer accounting |

- **Adjusted analysis (secondary, labelled retrospective):** the realized-share difference between a policy and the random arms is adjusted using the local random dose–response slope.
- Two extra random rates (p = 0.05 and 0.20, early stop) estimate that slope **on the same problems** (moderate design and above).

### 3.2 Anchors

- Historical FP16 and NF4 outcomes on the 930-problem pool are reused for continuity (exact reproduction 150/150 in the pilot).
- They are regenerated with early stopping and traces only where Q3 needs token sequences.

### 3.3 Q3 mechanistic factorial (non-deployable diagnostics)

Each step computes both branches. The **emitted token** and the **KV entries kept** are chosen independently:

| Cell | Token from | KV kept from | Meaning |
|---|---|---|---|
| FP16 | FP16 | FP16 | reference (static) |
| NF4 | NF4 | NF4 | reference (static) |
| **cache-only** | FP16 | NF4 | quantization reaches the output only through the cache |
| **token-only** | NF4 | FP16 | quantization reaches the output only through token choice |
| flip oracle (pilot) | FP16 | NF4 on agreeing steps, FP16 on flips | the pilot diagnostic |

- Requires a small engine extension (decoupled token/KV source), implemented and CPU-tested **before** any run, with the same replay and bitwise-cache tests as the pilot.
- **Q3 contrasts:**
  - cache-only − FP16 (damage via the cache path);
  - token-only − FP16 (damage via token decisions);
  - cache-only − NF4 (residual);
  - flip oracle − cache-only (value of FP16 KV on flip steps).
- **Descriptive:** first-divergence token from the FP16 trajectory, and the share of problems diverging.

### 3.4 Q4 descriptive analysis (no extra arms)

The both-branch Q3 runs log FP16 and NF4 outputs at every step of *self-generated* trajectories, under two history regimes (FP16 KV and NF4 KV). Estimands, with problem-clustered bootstrap CIs:
- ρ(H_{t−k}, JS_t) and ρ(M_{t−k}, JS_t) for k = 0, 1, 2;
- AUC for argmax flips;
- the same with the NF4-branch same-step margin (the cascade's signal).

This replicates the exploratory gold-trajectory result (ρ ≈ 0.53 lag-1 vs 0.91 same-step) on new data, with pre-declared estimands.

## 4. Statistical plan (applies to all designs)

- **Unit:** problem. All arms run on the same problems.
- **Paired analysis:** per-problem difference Y_policy − mean_k Y_random,k, with problem-level bootstrap 95% CIs (10,000 resamples) and a two-sided sign-flip permutation test. Exact McNemar against each seed is secondary.
- **Seeds:** K = 3 random seeds (K = 5 in the comprehensive design). Random streams are keyed by (arm, problem), as in the pilot.
- **Pre-declared primary family** (Holm across the family):
  - minimal: {D(lagE), Δ_A(lagE), Δ_F(lagE)};
  - moderate: add {D(lagM), Δ_A(lagM), Δ_F(lagM)};
  - Q3 contrasts form a separate family of 4.
- **Interpreting nulls:** equivalence bounds of ±3 pp (TOST, 90% CI). A difference is called absent only if the CI lies inside the bounds; otherwise the result is reported as inconclusive.
- **Exposure:** realized full, answer and decision shares are reported for every arm, together with the per-problem share distribution.
- **Everything else is exploratory:** per-class analyses, other budgets, the `=` token, and others.

## 5. Sample size and power

The minimum detectable paired difference at 80% power and two-sided α = .05 is about 2.8·√(p_d / n), where p_d is the discordant-pair rate.

| n | p_d = 0.06 | 0.10 | 0.15 | 0.22 |
|---|---|---|---|---|
| 500 | 3.1 pp | 4.0 | 4.9 | 5.9 |
| 930 | 2.3 | 2.9 | 3.6 | 4.3 |
| 1,319 | 1.9 | 2.4 | 3.0 | 3.6 |

**Planning values of p_d:**
- historical entropy vs random: 8.8–14.7%;
- pilot B10 (train): 6–10%;
- FP16 vs NF4: 21.8%;
- between two nearby thresholds (Q1's D): expected 6–10%, unmeasured.

Averaging K random seeds reduces the random arm's within-problem noise, but by an amount the pilot could not measure (its extra seeds were cut by the cap).

**Expected effects:**

| Quantity | Expected size | Consequence |
|---|---|---|
| Q1 D | predicted 0.1–0.4 pp (§1.2) | **no design here can detect it**; it can only bound it, to about ±2.4–2.9 pp |
| Q2 Δ at B10 | unknown; the historical point estimate was +2.6 pp at a much larger FP16 budget | detectable only if ≥ about 2.4–2.9 pp |
| Q3 contrasts | 0–10 pp (the full FP16–NF4 gap on test is 10.0 pp; p_d 21.8%) | detectable if the cache path or token path carries ≥ ~40–45% of the gap (≥ ~4.3 pp at n = 930, ≥ ~3.6 pp at n = 1,319) |

## 6. Three designs (estimated GPU-hours at measured rates; ±20%)

### Minimal — the budget-accounting question (Q1, entropy only)

**n = 930** (historical pool; anchors reused):

| Arm | GPU-h |
|---|---|
| `lagE_F` (full generation) | 3.9 |
| `lagE_A` (early stop) | 1.7 |
| `rand_p10` × 3 seeds (early stop) | 5.0 |
| **Total** | **≈ 10.5 GPU-h (8.5–12.6)** |

At n = 1,319 the total is ≈ 16 h.

- **Can support:** a bound on how much full-generation matching changes the entropy-vs-random conclusion at B10 (|D| < ~2.5–2.9 pp, or a finding of a larger effect); the answer-region exposure gap on test data; a prospective answer-budgeted comparison Δ_A with a CI.
- **Cannot establish:** effects below ~2.5 pp; anything about margin, the cache mechanism, other budgets or models; a meaningful positive result for Q1, since the predicted D is about 0.3 pp.

### Moderate — adds margin, Q3 mechanism, Q4, adjustment

All at **n = 930**:

| Component | GPU-h |
|---|---|
| Minimal | 10.5 |
| `lagM_F` + `lagM_A` | 5.6 |
| random p = 0.05 and 0.20 (adjustment slope) | 3.3 |
| FP16 / NF4 regenerated with early stop and traces | 2.7 |
| cache-only and token-only cells | 5.5 |
| flip oracle, early stop | 2.8 |
| **Total** | **≈ 30.4 GPU-h (24–36)** |

Add **+5.0 h** contingency if lagM's realized share differs from lagE's by > 1 pp and needs its own control.

- **Can support:** the Q1 bound for two signals; Q2 comparisons at about ±2.9 pp precision; a decomposition of the 10-pp FP16–NF4 gap into cache-path and token-path damage, detecting shares ≥ ~43% of it; Q4 replication on self-generated trajectories.
- **Cannot establish:** small (< 3 pp) routing advantages; behaviour at other budgets; other models or tasks; wall-clock benefit (NF4 is slower on this stack).

### Comprehensive — a substantially stronger empirical paper

All at **n = 1,319**:

| Component | GPU-h |
|---|---|
| Q1/Q2 arms for entropy and margin | 15.8 |
| K = 5 random seeds | 11.7 |
| dose points | 4.7 |
| FP16 / NF4 regenerated with traces | 3.8 |
| Q3 factorial + flip oracle + JS oracle | 15.7 |
| B30 sensitivity (lagE_F, lagE_A, 3 random) | 13.6 |
| **Total (one model)** | **≈ 65 GPU-h (52–78)** |

- **Second model family** (e.g. Llama-3.2-1B, after a headroom pilot of < 1 h): Q1 entropy core + Q3 cells at n = 930, **≈ 25–45 h**. This range is a guess: per-step cost for that model is unmeasured.
- **Total ≈ 90–110 h**, i.e. 4–5 days of continuous laptop use.
- **Can support:** Q2 at about ±2.4 pp; Q3 at about ±3.6 pp; Q1 bounds at two budgets; cross-family replication of the mechanism.
- **Cannot establish:** effects of about 1–2 pp; generality beyond two small models and one task; wall-clock benefit.
- **Feasibility limits:** the second-model cost is unmeasured; local runtime drift is ±12%; the September-to-October 1.7× slowdown is unexplained.

## 7. Recommendation

- **Strongest practical design: Moderate (≈ 30 GPU-h), run in two stages with a decision point.** It is the cheapest design whose main question (Q3) has effect sizes the sample can actually resolve, and it is informative under every outcome:
  - the cache path carries most of the damage;
  - the token path does;
  - neither does (a distributed effect).
- The Minimal design is cheaper, but its central question has a predicted effect (≈ 0.3 pp) about ten times smaller than its resolution. It would almost certainly return an inconclusive bound. That is publishable as a bound, but weak as the paper's main result.

**Specific experiment to run first (Stage 1, ≈ 11 GPU-h at n = 930):**

| Component | GPU-h |
|---|---|
| Q3 factorial: cache-only and token-only cells, early stop | 5.5 |
| FP16 / NF4 regenerated with early stop and traces | 2.7 |
| flip oracle, early stop | 2.8 |
| Q4 analysis of the logged branches | CPU only |

- Prerequisites: the decoupled token/KV engine extension, CPU tests and mutation checks, a GPU equivalence block, and a new private protocol lock.
- **Stage 2** (Q1/Q2: entropy and margin, F and A schemes, K = 3 random, dose points; ≈ 19 GPU-h) only on your authorization after Stage 1.
- Stage 2 is worth running regardless of the Stage 1 outcome, because Q1/Q2 answer the historical paper's open methodological question.
- If Stage 1 shows the cache path dominates, Stage 2's interpretation sharpens: step-level routing that keeps NF4 KV would be structurally limited.

## 8. Unresolved methodological questions

1. **Q1's estimand.** D compares a policy given an answer-matched budget with the same policy given a full-matched budget, against a shared random control. Is that the comparison you want, or should Q1 instead compare the *conclusion* of the historical analysis (realized full-share matching) with a retrospectively re-matched version on the *same* runs (cheaper, but purely adjusted)?
2. **Calibration for τ_A from existing train trajectories.** It is zero-cost but retrospective. Is reusing the pilot calibration data acceptable, or must calibration be rerun under the stop-at-marker rule? The latter is identical by prefix stability for the lag routers, so this mainly matters for documentation.
3. **n = 930 vs 1,319.** The 930 pool lets historical anchors be reused, but these problems were explored before. The 389 unexplored test problems are cleaner but small. Option: report the 389 problems separately as a "fresh-data" sensitivity analysis.
4. **Seeds.** K = 3 is a pragmatic choice. A 1–2 h seed-variance measurement (one random arm × 5 seeds at n = 300) would let K be justified rather than assumed.
5. **Q3 cells and the KV invariant.** In the token-only cell, the next step's input is the NF4-chosen token while the cache holds FP16 entries. Confirm this is the intended "token-path" definition, versus also feeding the FP16 token.
6. **Unexplained 1.7× slowdown** relative to September. Cost estimates assume today's rates. Should a short re-check be allowed before any long run, to catch a changed machine state?
