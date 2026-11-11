# Research Recovery Plan — Entropy-Gated Precision Routing (post-TMLR #13386)

Prepared 2026-10-22. Scope: assessment and plan only. No GPU experiments were run, nothing was rented, and nothing was modified, committed or published. The rejected manuscript (`paper_tmlr/`) and its supplement were only read.

Two CPU-only feasibility checks were run on stored result files, using scripts in the session scratchpad. They are labelled **EXPLORATORY / post hoc** wherever they appear below:
`lag_check.py` (signal timing) and `power_check.py` (power and noise-floor numbers).

---

## 0. Housekeeping found during the audit (needs a decision, no action taken)

- **`paper_tmlr/` is untracked in git.** The rejected manuscript, supplement and internal documents have no version history. They are one accidental deletion away from loss.
  - Recommended: make a read-only archive (zip plus SHA-256) of `paper_tmlr/` and `results/` *as they are now*, stored outside the working tree.
  - This needs your approval because you asked for no commits. Creating an archive is not a commit.
- `results/` is also partly unversioned relative to what the paper cites. Archive it together with the manuscript.

---

## 1. Independent scientific assessment

### 1.1 What the manuscript claims, and whether the stored data support it

I re-derived every headline paired GSM8K statistic from the stored per-problem outcome files. All of them reproduce exactly:

| Result | Paper | Re-derived |
|---|---|---|
| FP16 vs NF4 | 336 vs 243 solved | 148/55 discordant pairs, +10.0 pp |
| Override vs matched random | 82/79, p=0.87 | 82/79, diff +0.32 pp, 95% CI [−2.35, +3.00] |
| Entropy p10 vs random | 78/54, p=0.045 | identical |
| Entropy p30, p50, p70, p90 vs random | as in Table | identical |

The arithmetic is sound. The issues are in design and interpretation.

### 1.2 Point-by-point assessment

**(1) The paired FP16/NF4 EQF definition — sound as a measurement, but the headline pairs it with a lagged signal.**

EQF is the JS divergence between the FP16 and NF4 next-token distributions, computed under an identical FP16 teacher-forced history.
- I verified that the implementation (`scripts/build_eqf_spike.py`) does exactly this. It deep-copies the cache for the NF4 branch and advances the cache with the FP16 branch.
- The definition is clean, bounded, and correctly separated from routing.

The router decides step *t* using the entropy of step *t−1*'s distribution. The paper's headline correlation (ρ = 0.51 / 0.32) pairs exactly that lagged entropy with EQF at step *t*. That is the correct quantity for evaluating *this* router. But the abstract and conclusion phrase it as "entropy correlates moderately with EQF", which reads as a statement about confidence in general.

**New exploratory check (stored data, no GPU).** Row *i+1*'s stored entropy is the FP16 entropy of the same step whose JS is in row *i*. That makes the same-step relationship computable from stored data (problem-level bootstrap CIs, 300 replicates):

| Model | Signal | Spearman ρ with EQF | AUC (EQF>0.01) | AUC (argmax flip) |
|---|---|---|---|---|
| Qwen2.5-0.5B | entropy, lag-1 (router) | 0.528 [0.515, 0.539] | 0.72 | 0.71 |
| | entropy, same step | **0.906** [0.901, 0.909] | 0.91 | 0.91 |
| | margin, lag-1 | 0.501 | 0.71 | 0.69 |
| | margin, same step | **0.898** | 0.90 | **0.96** |
| TinyLlama_v1.1 | entropy, lag-1 (router) | 0.315 [0.298, 0.332] | 0.58 | 0.61 |
| | entropy, same step | **0.814** [0.805, 0.821] | 0.74 | 0.85 |
| | margin, same step | **0.805** | 0.78 | **0.96** |

(Lag-1 ρ differs slightly from the paper's 0.512/0.317 because the last token of each problem is dropped, so that a same-step value exists for every row.)

Interpretation, with caveats:
- Most of the gap between "confidence" and "fragility" in this paper comes from the **one-step lag**, not from confidence itself.
- Part of the same-step association is mechanical. For small logit perturbations, JS scales with the variance of the perturbation under *p*, which shrinks as *p* sharpens. That mechanical link is exactly why same-step confidence is useful.
- Same-step FP16 confidence is **not available for free** at decision time. A deployable approximation exists: compute the step in NF4, read the NF4 margin, and recompute in FP16 when the margin is small (a "cascade"). This costs an extra pass on the escalated steps.
- ReQAT's §3.2 diagnostic routes on **same-step BF16 entropy**. I verified this from its text: "At each decoding step, we compute the token entropy from BF16 model". That signal is non-deployable. This timing difference is a plausible, testable explanation for why ReQAT's routing diagnostic looks strong while this paper's causal router looks weak.

**(2) The entropy-matched digit/operator analysis — careful and honestly labelled, but it matches on the lagged entropy.**
- The composition-vs-fragility logic is correct, and the counterfactual composition test is a good idea.
- The stratification variable is the lag-1 entropy. Whether digit/operator tokens are more fragile *at matched same-step confidence* is untested. It can be tested from stored data (Z3 below).
- The analysis is post hoc with no documented specification date. The paper says so. That status cannot be repaired retroactively; it can only be superseded by a pre-registered re-test on new data.

**(3) Class composition vs class fragility — the strongest conceptual contribution in the paper.**
- The point that a quadrant odds ratio conflates base rates with conditional fragility is correct and general.
- On its own it is a methodological caution rather than a finding. It becomes a contribution when tied to a claim in the literature (ReQAT's "dominated by digits and symbolic operators") and tested against it.

**(4) GSM8K paired accuracy and the matched-random controls — correct statistics; three design problems limit what they can show.**

*a. The budget is uncontrolled across tasks.*
- Thresholds were calibrated on wikitext-2, but the GSM8K realized 4-bit shares were 41.0 / 79.0 / 93.5 / 98.1 / 99.0%.
- Three of the five sweep points (≥93.5% 4-bit) are almost static NF4. At p90, entropy routing is **outcome-identical to pure NF4**: 0 discordant problems out of 930.
- Only two budgets are informative (41% and 79%). The headline override comparison (56.9%) falls between them, at a budget where entropy-only routing was never measured.

*b. Budgets are counted over unscored text.*
- Generation always runs to the 256-token cap: 228–233k steps per 930 problems, about 249 steps per problem.
- The extractor discards everything after the model starts inventing a new `Question:` block (`truncate_at_continuation_drift`).
- The "4-bit share" therefore includes many post-answer steps that cannot affect the score. Gold solutions average about 120 tokens, so plausibly about half of all steps fall in this region (an estimate, not a measurement).
- Entropy routing and random routing may spread their FP16 steps differently across the scored and unscored regions. The comparisons are budget-matched on *total* steps, not on *answer-relevant* steps.
  - Suggestive evidence: at p90, the ~1% of FP16 steps chosen by entropy changed zero outcomes, while a random 1% changed 14.
  - This is consistent with high-entropy steps sitting in the post-answer continuation, but it is not proof.
- The direction of the resulting bias is unknown. Per-step traces were not saved, so this cannot be resolved without re-running.

*c. Power.*
- Discordance between policies is 14–17% even when they differ only in *which* steps are FP16. Greedy GSM8K outcomes are highly sensitive to small perturbations.
  - Random 1% FP16 vs pure NF4: 14 problems change.
  - Entropy vs random at about 2% FP16: 36 problems change.
- At n=930 the minimum detectable effect (80% power, α=0.05) is about **3.5–3.9 pp** at the informative budgets.
- The one nominal effect (+2.58 pp) is below that MDE. Detecting +2.6 pp at 80% power needs about 1,670 problems, more than the 1,319-problem GSM8K test set.
- The override-vs-random null is moderately informative. Its CI [−2.35, +3.00] pp excludes differences larger than about 3 pp. The paper should report that CI, not just p=0.87.
- Internal-consistency check on the p10 advantage: +1.07 pp in the 280-problem pilot, +3.23 pp in the 650-problem extension. Both are positive; neither is reliable on its own.

**(5) Five-budget sweep.** This is effectively a two-budget sweep (see 4a). The range-statistic permutation test for heterogeneity has little power with three near-null points. The paper's narrow reading is appropriate.

**(6) Cross-model perplexity.**
- These are perplexity-vs-routing-fraction curves on 10 windows. They show the trade-off exists qualitatively.
- They say nothing about fragility or accuracy. They are model count, not evidence for the thesis.

**(7) Exploratory 14B probe.**
- Eight problems, one condition, no interval. It cannot carry weight.
- It should be shrunk to a sentence or dropped in any successor paper.

**(8) Limitations and related work — honest, with gaps.** The manuscript cites none of the following:
- **Li et al. 2025, arXiv 2505.11574**, "Quantization Meets Reasoning". Under PTQ (AWQ/GPTQ/SmoothQuant; Qwen and LLaMA, 0.5B–7B; GSM8K/MATH), failures emerge early: "the first vulnerable step flipping and cascading to the final answer". Their fix restores token-level margins at that point. This overlaps heavily with candidate B.
- **QSpec** (Zhao et al., EMNLP 2025) and **QuantSpec** (Tiwari et al., ICML 2025). Both use a low-precision copy of the same model and high-precision verification to get lossless output. Any routing-*method* claim must answer "why not self-speculative decoding?", because when both precisions are resident, verification recovers exact FP16 output.
- Optional: **arXiv 2606.25519** (token inflation under quantization).

The paper's limitations section is otherwise unusually complete. Its main weakness is not candour; it is that the evidence base is narrow and partly confounded (4a, 4b).

### 1.3 Methodological integrity notes

- Earlier ARR-era claims were removed from the TMLR version because no supporting records existed. These include "pre-registered", "77 changes, zero answers" for the learned detector, and the claim that the pattern "held on all three architectures".
  - That removal was correct.
  - None of these claims should reappear in a successor paper without new evidence.
- p50's realized budget (93.5%) is hard-coded from printed output (`P50_REALIZED_PCT_4BIT`) because the per-step counts were never saved. This is minor, but it is another reason to persist traces from now on.
- My own analysis code had a bug, now fixed: a key-matching error that briefly produced 80/84 instead of 82/79. I caught it by checking marginal totals against the paper. Every new analysis script should assert marginal totals before computing paired statistics.

### 1.4 Overall verdict

- The paper is honest and statistically careful in what it reports. Its central downstream thesis — that confidence-based targeting adds little over a random allocation of the same budget — rests on one model, one task, about two informative budgets, an underpowered design, and budget accounting that includes unscored tokens.
- Its fragility thesis ("confidence is a moderate, model-dependent guide") is mainly a statement about **lagged** confidence.
- The most valuable next step is a single, well-powered, pre-registered experiment that separates three things:
  - the headroom step-level targeting could offer at all (an oracle);
  - how much of it a causal zero-cost signal captures;
  - how much a deployable same-step signal captures.

---

## 2. Novelty comparison with existing literature (primary sources checked 2026-10-22)

| Idea | Already shown elsewhere | Status in our study |
|---|---|---|
| Per-token or per-step precision chosen from a confidence/entropy signal | FlexQuant (8/4-bit, entropy plus KL, layer-wise; Findings EMNLP 2025); QuickSilver (entropy → per-token bit width); ReQAT §3.2 (BF16 vs MXFP4, same-step BF16 entropy, threshold 0.6 ≈ top 25%); DMR (trained step-level router between precision tiers, ALFWorld/WebShop) | **Not new.** The paper already concedes this. |
| Low-entropy digit/operator tokens as the locus of 4-bit failures | ReQAT §3.1/B.2 (FP4, R1-distilled 8–14B) | Not new as an observation. **New:** the entropy-matched test of composition vs excess fragility (exploratory). |
| Early token flips cascading to wrong answers under PTQ | Li et al. 2505.11574 (0.5B–7B, GSM8K/MATH; margin restoration) | Candidate B is **largely anticipated**. Any B-style work must position against it. |
| Lossless use of a low-precision copy plus high-precision verification | QSpec (EMNLP 2025), QuantSpec (ICML 2025) | Not cited. Lossy routing must justify itself against these. |
| Real paired FP16-vs-4-bit single-step divergence as ground-truth fragility | FlexQuant uses KL-divergence analysis between precisions to choose layers; ReQAT uses logit-noise injection as a proxy | **Modestly new:** used here as ground truth for evaluating routing signals token by token. |
| Budget-matched random controls for routing policies | Not reported in ReQAT's routing diagnostic (verified); not seen in FlexQuant/QuickSilver/DMR abstracts | **New as an evaluation norm** in this sub-literature. |
| Lag effect: causal routers use stale confidence; same-step confidence predicts fragility much better | Not found | **Potentially new and central** (exploratory so far). It reconciles ReQAT's positive diagnostic with our weak causal result. |
| Oracle headroom decomposition (token-flip damage vs KV-path damage) | Not found | **New, if run** (Experiment 1). |

Bottom line: the defensible novelty is **evaluation methodology**, plus one potentially substantive finding (the lag and headroom), not a routing method.

---

## 3. Candidate experiments, evaluated

Measured cost basis (from `results/gsm8k_multibudget_sweep_log.txt`, RTX 4060, Qwen2.5-0.5B, 256 new tokens, n=930):
- **8,284–9,006 s (2.3–2.5 h) per condition.**
- Condition 5 took 23,445 s (6.5 h) in an earlier run.

Everything else below is an **estimate**, marked "est.".
- Generation with the `\nQuestion:` stop: est. 45–60% of current steps → est. 1.5–2.1 h per condition at n=1,319.
- Both-branch (oracle) runs: est. 2.2–3× a standard condition.
- 1–1.7B models: est. 1.3–2× the 0.5B time.

### A. Cross-model downstream replication

- **Is TinyLlama_v1.1 sound? Probably not.**
  - Its static-NF4 perplexity cost is small (Δppl 0.256), and its GSM8K accuracy as a 1.1B base model is expected to be near the floor (unverified).
  - A near-zero FP16 accuracy or FP16−NF4 gap leaves nothing for routing to recover. TriviaQA already showed that failure mode.
  - Settle it with a 100-problem FP16 + NF4 pilot (est. 20–40 min). Reject the model if FP16 < 20% or the gap < 4 pp.
- **Better candidates that fit 8 GB with both weight copies** (≈2.5 bytes/param):
  - Llama-3.2-1B: already calibrated; different family; Llama-3 groups digits into multi-digit tokens, unlike Qwen's single-digit split, which matters for the token-class thesis.
  - SmolLM2-1.7B.
  - Qwen2.5-1.5B: same family, so weaker as a generality test.
  - gemma-2-2b (2.6B): risky at 8 GB.
  - Choose by the headroom pilot, not by family alone.
- Value: moderate. It adds model count but does not test the lag hypothesis. Novelty: low.
- Cost: est. 25–60 GPU-h for FP16/NF4/entropy/override/random at 2–3 budgets, n=1,319.
- Risk: high. A small NF4 gap means an underpowered null.
- Stored data are insufficient (no runs on other models).

### B. Generated-trajectory fragility

- Value: moderate. Novelty: **low to moderate**, because Li et al. 2505.11574 already shows early flips cascading under PTQ.
- What would still be new: whether *lagged vs same-step* confidence predicts flips on the model's own trajectory, and whether the gold-trajectory composition result holds on generated text.
- Cost: almost zero if bundled into Experiment 1, whose oracle runs compute both branches at every step. Standalone: est. 8–12 GPU-h.
- Must not treat a single-step shift as answer-level causality. The flip-oracle arm of Experiment 1 is an actual intervention (it prevents flips), so it gives a causal estimate at the policy level.
- Stored data: gold-trajectory only; no generations were saved.

### C. Stronger routing baselines plus a non-deployable oracle — **highest value**

- This arm directly tests the central thesis: does targeting *which* steps get FP16 matter beyond *how many*? It also tests the lag explanation.
- The oracle bounds what any step-level signal could achieve. If even the oracle barely beats random, the negative result becomes strong and general. If the oracle wins big but causal signals do not, the bottleneck is located precisely.
- Novelty: moderate to high. Cost: est. 35–60 GPU-h for the full design. Details in §5.

### D. Generality and statistical reliability

- Fold into C rather than run separately:
  - K random seeds for the random arm;
  - the full 1,319-problem test set;
  - Holm correction across a pre-declared primary family;
  - problem-clustered inference;
  - budgets calibrated on GSM8K-train;
  - answer-region budget accounting.
- Extending to a second quantizer (e.g. HQQ, which needs no calibration data) or to sampling-based decoding: defer until after C.

---

## 4. Ranked roadmap

1. **Highest-value experiment: Experiment 1, "Headroom decomposition of step-level precision routing"** (candidate C plus D, with B bundled in). Qwen2.5-0.5B, GSM8K test, pre-registered. Design in §5.
2. **Best second experiment, conditional on Experiment 1:**
   - If the oracle beats random (H1 holds): replicate H1–H3 on one other family (Llama-3.2-1B or SmolLM2-1.7B, chosen by the headroom pilot). Use the B=10% budget only, K=2 random seeds. Est. 20–35 GPU-h.
   - If the oracle does *not* beat random: run a confirmatory H1-only replication on the second family to make the negative result general. Est. 10–15 GPU-h.
   - If the cascade captures most of the oracle's headroom: add a compute-proxy comparison against self-speculative decoding (QSpec-style verification with the resident FP16 copy). Do not pitch a method unless the cascade wins on some axis.
3. **Zero-GPU analyses on existing data (do first, all exploratory, about 1–2 days):**
   - Z1. Formalize the lag check: full bootstrap, per class, per position; same-step margin vs entropy; include in a lab notebook.
   - Z2. Report CIs, minimum detectable effects, and equivalence bounds (TOST) for every paired GSM8K comparison. Holm-adjust the paper's family of tests.
   - Z3. Re-run the entropy-matched digit/operator analysis stratified on same-step entropy and margin. Does "composition, not fragility" survive?
   - Z4. Noise-floor analysis: outcome discordance vs number of perturbed steps (1% random → 1.5% of problems; ~6.5% → 8.8%). This characterizes the "chaotic sensitivity" that limits power, and is useful framing for any reviewer.
   - Z5. Random-routing dose-response frontier from the stored random arms (41, 56.9, 79, 93.5, 98.1, 99% 4-bit) plus FP16 and NF4 anchors. Place the targeted points relative to it, descriptively only.
   - Z6. Cross-tabulate per-problem outcomes: which policies recover the 148 "FP16-solved / NF4-failed" problems, and is recovery concentrated?
   - Z7. Margin in the entropy-matched analysis (the paper notes this can be done without new runs).
4. **Not worth doing now:**
   - Anything at 14B, or anything needing cloud hardware.
   - Wall-clock or kernel work: bitsandbytes NF4 is 1.8× slower, and the systems contribution would compete with speculative decoding.
   - More perplexity windows or checkpoints.
   - Extending TriviaQA (no NF4 loss to recover).
   - Wikitext flip-label classifiers (superseded by the cascade and oracle).
   - Further post hoc token-string mining (the `=` hypothesis), except as a pre-registered secondary in new data.
   - The instruct variant.
   - Running the old five-condition protocol on TinyLlama.
5. **Stopping rules:**
   - S0, reproducibility gate. Rerun FP16, NF4 and entropy-p50 on 100 of the 930 problems with the stop-string protocol. Require ≥98% per-problem agreement with the stored outcomes. If it fails, stop and diagnose software drift before any new science.
   - S1, pilot gate. Pilot on 200 GSM8K-*train* problems at B=10%: oracle, flip-oracle, cascade, lag-1 entropy, random.
     - If the oracle's advantage over random is < 2 pp AND the flip-oracle recovers < 30% of the FP16−NF4 gap: drop the method arms.
     - In that case run only confirmatory H1 at full n to nail down the negative result, and plan a negative-results paper.
   - S2, after Experiment 1.
     - H1 null with upper CI < 3 pp → write up the negative/diagnostic result; no further method work.
     - H1 holds but H2 and H3 fail → diagnostic paper on headroom vs signals; Experiment 2 replicates.
     - H2 holds with capture ≥ 50% → evaluate against speculative decoding before considering a method framing.
   - S3, second-model gate. Proceed only if the pilot shows FP16 ≥ 20% and an FP16−NF4 gap ≥ 4 pp.
   - S4, global time box. Stop new experiments at **150 local GPU-hours or 6 weeks**, whichever comes first, and write up with what exists.

### 4.1 Three budget-tiered plans (all on the local RTX 4060; no rented compute)

| Plan | Contents | GPU | Wall-clock (est.) | Expected outcome |
|---|---|---|---|---|
| **Minimal-cost** | Z1–Z7; archive the manuscript; S0 gate (≈1 GPU-h); S1 pilot on 200 train problems (est. 6–10 GPU-h) | ≈10 GPU-h | ≈1 week | Lag finding documented; effect sizes for the power calculation; go/no-go on Experiment 1. Enough for a revised workshop-length paper. Not a publication-grade upgrade on its own. |
| **Moderate** | Minimal + full Experiment 1 at both budgets, n=1,319, K=3 random seeds | est. 45–70 GPU-h | ≈2–3 weeks of overnight runs | A pre-registered, powered answer to "does targeting matter, and does the lag explain it"; generated-trajectory EQF bundled in. Supports a full journal paper (TMLR resubmission) either way. |
| **Ambitious** | Moderate + Experiment 2 on a second family + one second task with a real NF4 loss (e.g. HumanEval or MBPP for a code model, after a headroom pilot) + an optional second quantizer (HQQ) | est. 110–150 GPU-h | ≈5–6 weeks | Cross-family, cross-task diagnostic study; bounded by S4. |

---

## 5. Detailed design: Experiment 1

**Question.** On GSM8K, at a fixed FP16 budget, how much downstream accuracy can step-level targeting buy over random allocation (oracle headroom)? How much of that headroom do (i) the existing causal lag-1 entropy router, (ii) a lag-1 margin router, and (iii) a deployable same-step cascade capture?

**Model and data.**
- Qwen2.5-0.5B. Same prompt as the paper: 3-shot from the first three train examples, `Question: …\nAnswer:`, greedy decoding, max 256 new tokens.
- **Change 1 (documented): stop at the `\nQuestion:` drift marker.**
  - Generation is causal and the policies are deterministic or seeded per step, so text before the stop is identical with or without stopping.
  - Stored outcomes therefore stay comparable; the S0 gate checks this.
- **Evaluation set: all 1,319 test problems.** Primary analysis on 1,319; the 930-problem subset is reported for continuity.
- **Calibration set: 200 GSM8K-train problems** (indices ≥ 3, disjoint from the few-shot exemplars).

**Budget definition (Change 2).**
- Budget = fraction of *answer-region* decode steps whose output came from FP16 weights. Answer region = steps before the stop marker.
- Separately report a compute proxy: FP16 passes + NF4 passes, so the cascade's extra NF4 pass is visible.
- Two pre-declared budgets: **B1 = 10% FP16 (primary)** and **B2 = 30% FP16 (secondary)**.
  - B1 sits just above the flip rate (~3.6% of gold-trajectory steps flip argmax), where targeting has the most room to matter.

**Arms** (thresholds calibrated on the calibration set to hit each budget within ±1.5 pp; each random arm matched to its policy's *realized* test budget, as in the paper):

| Arm | Rule at step t | Deployable? |
|---|---|---|
| FP16, NF4 | static anchors | — |
| R | FP16 with probability B, independent Bernoulli per step; **K=3 seeds** | yes (control) |
| E1 | FP16 if H(p_{t−1}) > τ (existing router) | yes, zero cost |
| M1 | FP16 if margin(p_{t−1}) < τ | yes, zero cost |
| C (cascade) | run step in NF4; if margin(p_t^NF4) < τ, recompute in FP16 and keep the FP16 output and KV | yes; extra NF4 pass on escalated steps |
| O (EQF oracle) | run both; FP16 if JS(p_t^16, p_t^4) > τ | **no** — diagnostic reference only |
| F (flip oracle) | run both; FP16 iff argmax differs; budget set by the rule itself | **no**; plus a random control matched to F's realized budget |

KV rule: whichever branch is chosen supplies that step's KV entries. This matches a real router. F therefore keeps NF4 KV on agreeing steps, which is what makes the decomposition meaningful.

**Pre-registration.**
- Before any test-set run, freeze the analysis plan and code, record their SHA-256, and timestamp them. The mechanism needs your approval: an OSF registration or a dated hash shared with coauthors.
- **Primary family (Holm, α = 0.05), at B1:**
  - H1: O > R.
  - H2: C > R.
  - H3: E1 > R.
- **Primary statistic:** per-problem difference Y_arm − mean_k Y_R,k. Two-sided paired sign-flip permutation test, 10,000 permutations, unit = problem.
- **Secondary:** exact McNemar against each random seed; odds ratios with CIs.
- **Estimation:**
  - Headroom capture (E1−R)/(O−R) and (C−R)/(O−R), with problem-bootstrap CIs. Interpreted only if the O−R CI excludes 0.
  - Decomposition: FP16 − NF4 = (FP16 − F) [KV-path damage] + (F − NF4) [token-flip damage].
  - Everything above repeated at B2.
- **Interpreting nulls:** pre-declare an equivalence margin of ±3 pp (TOST). "No difference" is claimed only if the 90% CI lies inside it.
- **Exploratory, labelled as such:**
  - EQF along generated trajectories (logged by O and F at no extra cost), compared with gold-trajectory EQF;
  - the digit/operator analysis at matched same-step confidence;
  - first-flip position vs outcome, positioned against Li et al.

**Power.**
- With measured discordance of 10–15% and n = 1,319, the MDE is about 2.4–2.9 pp (80% power). Averaging K=3 random seeds lowers the random arm's noise further (that gain is not quantified yet; S1 provides the estimate).
- H1 is expected to be well powered if the oracle recovers ≥ 5 pp; n ≈ 312–469 suffices for 5 pp.
- H3 may be underpowered if its true effect is ≤ 2.5 pp, as the stored p10 data hint. This is stated in advance and handled by the equivalence analysis.

**Possible outcomes and what each establishes:**

| Outcome | Meaning | Paper |
|---|---|---|
| O ≈ R (null, tight CI) | Step-level targeting has little downstream value on this task. The FP16−NF4 gap is spread out or driven by the KV path. | Strong negative/diagnostic result |
| O ≫ R, C ≈ O, E1 ≈ R | Headroom exists; same-step confidence reaches it; the causal lag is the bottleneck. Reconciles our result with ReQAT. | Diagnostic paper with a clear mechanism; method framing only if it beats speculative decoding on a cost proxy |
| O ≫ R, C and E1 ≈ R | Headroom exists, but confidence signals cannot locate it | Diagnostic paper ("fragility is not visible in confidence") |
| F recovers most of the gap at a small budget | Quantization damage is mostly a few argmax flips | Strong framing result; links to Li et al. |

**Instrumentation (fixes the paper's retention gaps).** Persist, for every arm:
- full generations;
- per-step route, H and margin at lag-1;
- for O, F and C: both-branch JS, both argmaxes, NF4 margin and entropy;
- the answer-region boundary;
- realized budgets;
- KV dtype asserts;
- software versions;
- seeds.

Assert marginal totals in every analysis script.

**Cost (estimate):**
- S0: ≈1 h.
- Calibration: est. 6–10 h (iterative, because routing changes trajectories).
- Test runs at two budgets: R×6, E1×2, M1×2, C×2 (≈1.2× each), O×2 (≈2.5×), F×1 (≈2.5×), F-matched R×1, plus NF4/FP16 for the 389 new problems. About 22 standard-condition equivalents × est. 1.5–2.1 h ≈ **35–50 GPU-h**.
- Total ≈ **45–60 GPU-h**, about 2–3 weeks of overnight runs on the laptop.
- If only the primary budget B1 is run: about **25–35 GPU-h**.

**Engineering needed (in a dev branch, after approval):**
- A `generate_policy()` loop with a pluggable rule.
- A both-branch step that uses DynamicCache crop/append, rather than deepcopy per step, for speed.
- The stop-string handling.
- Trace persistence.
- Unit tests: (a) F with all-agree steps reproduces NF4-KV behaviour; (b) a policy forced to FP16 everywhere reproduces the FP16 outcomes; (c) the cascade never uses FP16-derived information before its decision.

---

## 6. Publication strategies by outcome (policies checked 2026-10-22)

| Paper type | Fit | Notes |
|---|---|---|
| **Diagnostic/evaluation paper** (most likely) | **TMLR resubmission**: criteria are claims supported by convincing evidence plus audience interest, "even if the contribution… is modest"; no fees; resubmission allowed as a new submission with a link to #13386 and a description of changes. **ACL 2027 via ARR**, January 2027 cycle (ARR deadline listed as 2027-01-04 AoE): 8-page long paper; the call explicitly welcomes negative findings. | ARR requires linking prior versions, including desk rejections. The ARR #283 record (meta-review 2, "resubmit next cycle") must be linked. ARR's new reviewer-capacity policy (from October 2026) means an author must serve as a reviewer. Moderate plan: TMLR fits best (no length pressure). |
| **Negative-results paper** | TMLR; ACL/ARR (negative findings explicitly welcome); NeurIPS-style negative-results workshops (ICBINB — no 2026/27 call found; check closer to time) | Negative results are publishable when the null is tight (equivalence) and the oracle shows the null is not a power artefact. Experiment 1 is designed for exactly that. |
| **Routing-method paper** | Only if the cascade beats random clearly **and** is competitive with self-speculative decoding (QSpec/QuantSpec) on a cost proxy. Then ICML 2027 (third-party estimate ~2027-01-22, **not official**) or an efficiency workshop. | Unlikely given bitsandbytes slowness and speculative decoding's losslessness. Not a planning target. |

Disclosure requirements for any venue:
- Do **not** dual-submit.
- Disclose the TMLR desk rejection where the venue asks (TMLR requires the link). Disclose the ARR history to ARR.
- Keep the LLM-use disclosure.
- Never describe the TMLR or ARR versions as reviewed in their current form.

No acceptance probabilities are given; none can be honestly estimated.

---

## 7. Decisions needed from you

1. **Archive:** may I create a read-only zip plus SHA-256 of `paper_tmlr/` and `results/` outside the working tree? Nothing in place would be modified.
2. **Dev workspace:** approve a separate branch or worktree (e.g. `research-v2`) for new code, with no commits to `main`? You asked for no commits at all, so state whether *local* commits on the dev branch are allowed (they make the pre-registration hash meaningful).
3. **Pre-registration mechanism:** OSF (a public timestamp, which counts as publishing) or a dated hash shared privately with coauthors?
4. **Budget tier:** minimal, moderate or ambitious — and the overall cap (proposed: 150 local GPU-h / 6 weeks).
5. **Evaluation set:** full 1,319-problem GSM8K test (recommended), or keep the 930-problem pool only?
6. **Primary budget:** B1 = 10% FP16 as primary, B2 = 30% secondary — or other values?
7. **Protocol changes:** accept the stop-at-`\nQuestion:` rule and answer-region budget accounting as declared changes from the TMLR protocol?
8. **Second model:** approve a headroom pilot (100 problems, FP16 + NF4) on Llama-3.2-1B and SmolLM2-1.7B? TinyLlama is deprioritized unless you want it tested anyway.
9. **Target framing:** are you willing to retitle and refocus around "headroom and signal timing" rather than token-class corrections, if the data support it?
10. **Coauthors:** the internal docs show coauthor approval steps. Should this plan go to them before any GPU work?

---

### Sources (primary, checked 2026-10-22)
- ReQAT, arXiv 2606.15682 (ICML 2026): https://arxiv.org/html/2606.15682
- Li et al., Quantization Meets Reasoning, arXiv 2505.11574: https://arxiv.org/abs/2505.11574
- QSpec, arXiv 2410.11305 / EMNLP 2025: https://arxiv.org/html/2410.11305v3
- QuantSpec, ICML 2025: https://proceedings.mlr.press/v267/tiwari25b.html
- FlexQuant, Findings EMNLP 2025: https://aclanthology.org/2025.findings-emnlp.221.pdf
- QuickSilver, arXiv 2506.22396
- DMR, arXiv 2602.02711: https://arxiv.org/abs/2602.02711
- Token inflation under quantization, arXiv 2606.25519: https://arxiv.org/html/2606.25519v1
- TMLR editorial policies: https://www.jmlr.org/tmlr/editorial-policies.html
- ARR dates: https://github.com/acl-org/aclrollingreview/blob/main/dates.md ; ARR CFP: https://aclrollingreview.org/cfp
- ACL 2027 call: https://2027.aclweb.org/calls/main/
- ICML 2027 deadline estimate (unofficial): https://www.opencurious.com/ai-conference-deadlines/icml-2027
