# E1 report — perturbation-matched cache controls (Qwen2.5-0.5B, RTX 4060)

Run 2026-10-27 under the locked `E1_SPEC.md` (lock verified before the run and again before the analysis; calibration hash `2d066f31…8580` appended to `E1_PROTOCOL_LOCK.txt` before any test problem was evaluated). Authorization: the authors' file `e1_runs/AUTHORIZED_BY_AUTHORS.txt`. Draft 3, the archives and Stage 1/2 are unchanged; nothing was pushed or submitted; no second-model run, no cloud. Machine-readable results: `e1_runs/e1_report.json` / `.md`; raw per-problem records in `e1_runs/{calib,tierA,tierB}/`. **Labels:** *confirmatory-design* = locked primary/secondary hypotheses of the spec; *exploratory* = sensitivity analyses and descriptive tables below. All inference is over problems, paired; intervals are 95 % percentile bootstrap (10,000) unless stated.

## 1. GPU gates and deviations
| Gate | Result |
|---|---|
| G1 α = 0 reproduces FP16 bitwise (tokens + cache) | **pass**: 40/40 (20 train problems × C1, C2, 64 tokens) |
| G2/G3 per-slot norms within tolerance; finite entries/logits; step-1 JS exactly 0 for every arm (token-history/cache-state consistency) | **pass**: no integrity error on 8 problems × 5 arms × 64 steps |
| G4 determinism | **pass**: same seed ⇒ identical (8/8); different seed ⇒ different (8/8) |
| G5 NAT through the new path vs the stored engine cache-only generations (Stage 1/2) | **24/24 token-identical** on the GPU |
| G6 runtime projection ≤ 8 h | passed (3.97 h projected from one short problem — too optimistic; the Stage 1/2-rate estimate was 7.3 h; realized 6.57 h) |
| GPU mutation + correctness checks (authorization item 5) | **pass**: the 13-test suite and all **15/15 injected bugs caught** on a tiny Qwen2 with the real CUDA DynamicLinear/bitsandbytes kernels (`tools/e1_gpu_mutation_check.py`, 48 s) |
Deviations (none changes a hypothesis, outcome, sample, seed stream, arm or analysis): (i) the **GPU mutation gate was added** as a new file because the locked driver had no such step; it is outside the lock and its time is in the ledger. (ii) The locked Tier B integrity check enforces only the shared first token, so Tier B did not apply the per-slot norm tolerance: its diagnostic shows mean deviation 0.05 % and a **maximum per-slot deviation of 10.5 % (C1)**, marginally above the 10 % gate level used in Tier A (where the maximum was 9.9 %). (iii) G6's single-problem projection under-estimated the runtime; no reduction was needed. **No truncation, no reduction rule invoked, no integrity or hardware stop.** Calibration constants (train problems only): α\* = 0.910, dose slope b̂ = 2.486, c = 0.100 (raw 0.084, clamped), Tier A equivalence margin 0.237 log-JS (≈ 27 %).

## 2. Runtime and hardware (measured)
| Block | GPU-hours |
|---|---|
| Gates (G1–G6) | 0.226 |
| GPU mutation check | 0.013 |
| Calibration (100 train problems × 5 arms) | 0.841 |
| Tier A (200 × 6 arms, 64 steps) | 1.907 |
| Tier B (400 × FP16, NAT, C1, C2) | 3.586 |
| **Total** | **6.573 h of the 8.0 h ceiling** |
Per-arm cost: Tier A 5.4 s per arm-problem (64 teacher-forced steps ≈ 84 ms/step); Tier B FP16 3.5 s, NAT 9.8 s, C1 9.2 s, C2 9.7 s per problem. Maximum GPU temperature 70 °C (stop limit 88 °C); no throttle stop. Realized perturbation sizes: the NF4 difference is **14.3–14.6 % of the entry norm** (‖Δ‖/‖κ^F‖) in every arm; realized-vs-target norm deviation, mean 0.05 % (C1) / 0.008 % (C2).

## 3. Tier A (confirmatory-design): teacher-forced next-token JS, identical token history, 200 problems, first ≤ 64 steps
Per-problem mean JS to the pure-FP16 reference (descriptive means): NAT 0.00445, **C1 0.00610**, C2 0.00473, C1 α = 0.5 0.00129, α = 2 0.0381, C1\* (α = 0.91) 0.00477. Top-1 flip rate: NAT 3.43 %, C1 3.86 %, C2 3.44 %, C1(0.5) 1.77 %, C1(2) 10.6 %, C1\* 3.26 %.

**Primary (H_struct):** θ_A = mean[log m^NAT − log m^C1] = **−0.253** (ratio **0.777**), 95 % CI [−0.325, −0.179], sign-flip p = 1×10⁻⁴; 90 % CI [−0.314, −0.191], which is not inside ±0.237, so **not equivalent**. Pre-declared verdict: **"difference" — natural NF4 entries are *less* disruptive than same-norm random-direction perturbations** (by about 22 % in geometric-mean JS).

**Secondary family (Holm over 7; confirmatory-design):**
| Contrast | estimate (log JS unless stated) | 95 % CI | raw p | Holm p |
|---|---|---|---|---|
| NAT − C2 (sign-randomized Δ) | +0.003 | [−0.083, +0.086] | 0.95 | 0.95 |
| C2 − C1 | **−0.256** | [−0.334, −0.176] | 1×10⁻⁴ | 7×10⁻⁴ |
| Dose slope of C1 (log JS per log α, α = 0.5, 1, 2) | **+2.507** | [+2.449, +2.565] | 1×10⁻⁴ | 7×10⁻⁴ |
| Top-1 flip rate NAT − C1 | −0.0043 | [−0.0083, −0.0003] | 0.036 | 0.14 |
| NAT − C1\* (effect-matched, α\* = 0.91 from train problems) | −0.041 | [−0.112, +0.030] | 0.26 | 0.77 |
| Tier B NAT − C2 (tokens) | +0.98 | [−1.54, +3.54] | 0.45 | 0.90 |
| Tier B NAT − C1 (tokens) | **+3.54** | [+1.14, +5.95] | 0.0052 | **0.026** |
Reading: (a) JS scales roughly as dose^2.5, so the matching of size matters a great deal and the dose arms bracket the NAT effect (0.5× is 3.5× smaller, 2× is 8.6× larger). (b) NAT is statistically indistinguishable from C2 (90 % CI inside ±0.237) while C2 differs from C1 by the same amount as NAT does: **the per-coordinate magnitude profile of Δ (its channel/outlier structure) accounts for the difference from isotropic noise; the sign pattern adds nothing detectable here.** (c) The effect-matched C1\* (scale 0.91, i.e. 9 % smaller norm than NAT) reproduces NAT's effect on the test problems, confirming the train-set calibration transferred; equivalently, isotropic noise needs ≈ 9 % less norm than NAT to have the same effect (consistent with the ratio 0.78 given slope 2.5). (d) The top-1 flip-rate contrast is borderline and not Holm-significant.

## 4. Tier B (confirmatory-design): free-running divergence from the in-session FP16 run, 400 problems
Restricted mean first-divergence token (τ = 64): NAT **30.3**, C2 29.3, **C1 26.7** (the NAT value reproduces Stage 1/2: 30.2–30.7). Identical to FP16 through the whole generation: NAT 36/400, C1 30/400, C2 39/400. **θ_B = NAT − C1 = +3.54 tokens** [+1.14, +5.95], raw p 0.0052, Holm 0.026: with the same stored-state size, NAT departs from the FP16 trajectory **later** than random-direction noise — same direction as Tier A, ≈ 12 % of the NAT mean. NAT − C2 = +0.98 [−1.54, +3.54]: not distinguishable. Descriptive accuracy (paired, underpowered): FP16 37.2 %, NAT 33.5 %, C1 32.8 %, C2 32.0 %; NAT − C1 = +0.75 points [−4.0, +5.5]; NAT − FP16 −3.75 [−7.5, 0.0]; C1 − FP16 −4.5 [−8.75, 0.0] — random-direction noise of NF4's size costs about as much accuracy as NF4's own difference, within wide intervals.
Tier B equivalence statements are not claimed (relative margin supportive only; the interval for NAT − C1 excludes 0).

## 5. Exploratory sensitivity analyses (none replaces the primary)
| Analysis | NAT − C1, log JS (Tier A) |
|---|---|
| steps 1–32 | −0.246 [−0.326, −0.166] |
| steps 9–64 | −0.269 [−0.345, −0.193] |
| per-problem median JS instead of mean | −0.105 [−0.176, −0.033] |
| problems with ≥ 32 steps | −0.253 [−0.328, −0.180] |
| relative difference (NAT − C1)/C1 | −0.114 [−0.177, −0.049] |
| excluding the 13 exposed test problems | −0.252 [−0.328, −0.176] |
| pooled steps, problem-clustered | −0.137 [−0.190, −0.083] |
| **steps 2–8 only (few perturbed entries)** | **−0.104 [−0.216, +0.009]** |
| Tier B, τ = 32 / 128 | +1.72 [+0.52, +2.87] / +4.94 [+0.85, +9.06] |
All point estimates share the primary's sign. The effect is **smaller and not resolved in the first few steps** and grows as perturbed entries accumulate; the median-based and pooled estimates are about half the primary's size, i.e. the primary is partly driven by heavy-tailed steps.

## 6. Scientific conclusions justified
- **Positive (confirmatory-design):** under the same token history and the same per-(layer, K/V, head, position) norm, NF4-written entries are **not** equivalent to random-direction noise: they are less disruptive per unit norm (≈ 22 % lower geometric-mean JS), the free-running trajectory departs ≈ 3.5 tokens later, and the effect is in the same direction in both tiers. The difference is carried by Δ's **per-coordinate magnitude profile**: sign-randomized Δ (C2) is indistinguishable from NAT.
- **What this means for the paper's claim:** "NF4-written state" is not merely "a perturbation of that size"; it is a structured perturbation (concentrated in particular channels) that is *milder* than isotropic noise of equal norm. This **refines** the earlier wording rather than strengthening a "cache is dangerous" message: random noise of NF4's size would do at least as much damage.
- **Negative/inconclusive:** no difference between NAT and sign-randomized Δ (Tier A CI ±0.085, Tier B ±2.5 tokens); accuracy contrasts inconclusive; the early-step (2–8) contrast is unresolved; flip-rate contrast not Holm-significant.
- **Limits:** one 0.5B base model, GSM8K, first 64 teacher-forced steps; isotropic Gaussian direction as the random baseline (not the only natural one; RoPE-pair structure not modelled); controls act on stored values, not on the computation that produced them; JS (a distribution-shift measure) is related to, not identical with, accuracy; the effect size is moderate.

## 7. Does E1 materially strengthen the paper beyond the existing literature?
Scientifically, **yes, in a specific and modest way:** none of the closest works (Du et al.; Chodavarapu & Xu; QSpec; QuantSpec; Li et al.) compares a real low-precision cache difference with a norm-matched artificial perturbation, and this is a locked, validated, confirmatory-design result with an explicit dose-response, an effect-matched control, an equivalence rule, and a Tier B behavioural counterpart. It answers the manuscript's largest self-declared limitation and gives a mechanistic statement (channel-magnitude structure, not signs) that single-step grafts cannot. It does **not** change the paper's genre (diagnostic study, one model/benchmark) and does not rescue the accuracy-pathway evidence; for Neurocomputing's method-oriented screening it improves credibility more than fit. No acceptance probability is claimed.

## 8. Is a short Llama E1 replication justified?
**Yes, a short one.** The result is a single-model finding and its most useful next test is whether the same ordering holds on a second family. Justified scope: **Tier A only on Llama-3.2-1B-Instruct** (the pilot-approved model), arms NAT, C1 (α = 0.5, 1, 2), C2, C1\*, 200 test problems plus a 100-problem train calibration, i.e. ≈ 1.5–2.0 GPU-h (teacher-forced 64 steps at ≈ 53 ms/step ⇒ ≈ 3.4 s per arm-problem; calibration ≈ 0.6 h, Tier A ≈ 1.2 h, gates ≈ 0.2 h), with the primary contrast and the C2 contrast fixed in a new lock before any Llama data. It needs your separate approval and a spec/code generalization (≈ 1 day). Tier B on Llama is not recommended (NF4's accuracy effect there is ≈ 1 point; trajectory outcomes add cost without a clear question).

## 9. Recommendation
**Replicate once more, briefly (Tier A on Llama), while starting the manuscript revision; then revise and submit.** If the Llama replication shows the same sign and C2 ≈ NAT, the paper can state the mechanism claim for two model families; if it differs, the paper states the Qwen result with its scope and the difference. If the authors prefer speed over generality evidence, **revise and submit now** with E1 integrated and single-model scope stated — E1 alone already answers the matched-control objection. **Stopping with no manuscript use of E1 is not recommended.** In every case the three-author verification and the AI-use disclosure remain prerequisites; Draft 3 stays unmodified until you decide.
