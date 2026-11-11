# E1 specification — perturbation-matched cache controls (locked version)

Prepared 2026-10-27 (local calendar) for author review. This is the document the private dated hash (`E1_PROTOCOL_LOCK.txt`) covers, together with the code, tests and sample. It is **a private dated hash, not a public pre-registration**; its hypotheses are exploratory in origin (they come from the limitations of Stages 1–2). **Locking is not authorization to run: no E1 GPU evaluation may start without the authors' further approval** (enforced in `run_controls.py`). Nothing in this file changes Stage 1/2 code, results or locks.

## 1. Hypotheses
Under NF4 weight quantization, decoding with FP16-weight tokens over stored K/V entries that carry the actual NF4-induced difference Δ ("NAT") versus entries that carry an artificial perturbation:
- **H_struct (primary, Tier A):** with *identical token history* and *identical per-(layer, K/V, KV-head, position) perturbation norm*, the next-token distribution under NAT differs from that under a random-direction perturbation (C1). H0: equal.
- **H_chan:** NAT vs sign-randomized Δ (C2), and C2 vs C1 (does keeping the per-coordinate magnitude profile of Δ matter?).
- **H_dose:** the effect depends on the size of the perturbation (dose–response of C1 at α = 0.5, 1, 2); needed to interpret any "equivalence".
- **H_eff:** NAT vs an *effect-matched* random control C1\* whose scale α\* is calibrated so that its immediate effect equals NAT's (we do **not** assume norm matching equates effect).
- **H_beh (Tier B):** the NAT−C1 (and NAT−C2) contrast changes *when a free-running generation first departs from the FP16 trajectory*.

## 2. Definitions
**Setting.** Qwen2.5-0.5B, FP16 vs bitsandbytes NF4 (double quantization, FP16 compute), 168 dual-weight projection modules, GSM8K 3-shot prompt as in Stages 1–2, greedy decoding, ≤ 256 new tokens, drift-marker early stop (Tier B). Prefill is FP16 and **never perturbed**.
**Step.** At decode step s both weight precisions run from the same token and state S_{s−1} (unkept pass first, crop, kept pass second, exactly the verified engine). The NF4 pass is used only to measure **Δ = κ^N − κ^F** of the *new* entry; the FP16 pass supplies the logits and κ^F.
**Newly written vs accumulated entries.** A perturbation is applied **only to the newly written entry, at write time**; entries already in the cache are never revisited (each carries the perturbation of its own write step, exactly as in NAT). No other trajectory's entries are ever inserted.
**Slot.** (layer, K or V, KV head) at the new position; norms are L2 over head_dim.
**Arms** (all use FP16-pass logits): NAT stores κ^N bitwise (no arithmetic; equals the engine's cache-only entry, tested). C1 stores κ^F + α‖Δ_slot‖·u, u uniform on the unit sphere of R^head_dim, independent per slot and step. C2 stores κ^F + α·(ε⊙Δ), ε i.i.d. ±1 per coordinate. C1\* is C1 at α = α\* (§4). α = 0 reproduces pure FP16 bitwise (tested; GPU gate G1). FP16 is the pure FP16 run.
**Direction versus magnitude.** C1 keeps magnitude (per slot) and randomizes direction entirely; C2 keeps the per-coordinate magnitude profile and randomizes sign only; the dose arms vary magnitude with direction fixed random; C1\* varies magnitude to equate the *effect*. Together they separate direction from magnitude.
**Activation scale.** Δ is measured against the same κ^F, so ‖δ‖/‖κ^F‖ is matched slot by slot; the distribution of this ratio is recorded every step (`rel_scale_mean`), not assumed.
**Not controlled (scope limit).** How NF4 errors propagate *between layers within a step* is not reproduced: controls act on the stored result, not on the computation that produced it.
**Why norm matching is not trusted alone.** Effect-in-logit-space depends on direction relative to the queries; hence the dose arms, the effect-matched C1\*, the early-step analysis (steps 2–8, where only a few perturbed entries exist and accumulation cannot explain a difference), and the interpretation rules of §9.

## 3. Samples, seeds, leakage control
- **Test sample:** a seeded permutation of the 1,319 GSM8K test problems (`TEST_ORDER_SEED`); Tier B = first 400, Tier A = first 200 (subset of Tier B). The ids are fixed in `e1_sample.json` (hashed in the lock) before any run. No outcome, output or statistic of any model enters their choice. Exposed problems (Stage 2's 47 gold-token-labelled problems) that fall in the sample are listed and analysed in a sensitivity analysis. All 1,319 problems have been used in earlier work; the **controls have no prior outcomes**, and the Tier A/B primary outcomes use no gold label.
- **Calibration sample:** 100 GSM8K **train** problems (seeded; excluding the three few-shot examples and every train problem used in the pilots); first 20 for gates. Calibration constants (α\*, dose slope, margin) are computed **only from calibration records** and frozen in `e1_runs/e1_calibration.json`, whose hash is appended to the lock **before any test problem is evaluated**.
- **Directions without leakage:** every random draw comes from a `torch.Generator` seeded by SHA-256 of (master seed, arm, α, problem index, tier, repeat) — a function of labels only. The C1/C2 draws cannot depend on test outcomes. No calibration-derived *direction* is used in the confirmatory arms (the bank-based C3 of the earlier draft is dropped); calibration-derived *scalars* (α\*, slope, c) come from train problems only.
- **Shared token history (Tier A):** the stored FP16 free-running token sequence of each test problem (Stage 1/2 records; their SHA-256 is in the lock) is fed to **every** arm; the reference distributions r_s are recomputed in-session by a pure-FP16 teacher-forced pass. Step s feeds token s−1 and predicts token s, for s = 1…S_i, S_i = min(64, length − 1). Tier B re-runs FP16 in-session.

## 4. Calibration (train problems; GPU, part of the E1 budget)
Teacher-forced, 64 steps, arms NAT and C1 at α ∈ {0.5, 1, 2, 4}. Let m(·) be the mean over problems of the per-problem mean JS over steps 2–16.
- **α\***: the α at which log m(C1(α)) interpolates (piecewise-linear in log–log) to log m(NAT); clamped to [0.5, 4] with a flag if outside.
- **Dose slope b̂**: mean over problems of the OLS slope of log(per-problem mean JS) on log α for α ∈ {0.5, 1, 2}.
- **c**: median over problems of the step-to-step coefficient of variation of the mean NF4 perturbation norm (NAT arm), clamped to [0.10, 0.40].
- **Equivalence margin (Tier A)** = |b̂|·ln(1 + c) on the log-JS scale, claimable only if **b̂ ≥ 0.25** (otherwise the dose–response is too flat for a matched size to be informative and no equivalence is claimed).

## 5. Outcomes and estimands
**Tier A (primary, label-free).** Per problem and arm, m_i = mean over s = 1…S_i of JS(a_s, r_s) (JS in nats, float64; exactly 0 for identical logits), floored at 1e-9 before logs; also the top-1 flip rate. **Primary estimand θ_A = mean_i [log m_i^NAT − log m_i^C1]** (paired by problem; positive = NAT more disruptive; exp(θ_A) = ratio).
**Secondary (Holm family):** NAT−C2; C2−C1; dose slope of C1 (mean per-problem slope, test vs 0); top-1 flip rate NAT−C1; NAT−C1\*; and from Tier B: NAT−C1 and NAT−C2 restricted-mean first-divergence.
**Tier B (behavioural).** Per problem and arm, Y = min(D, 64), D = index of the first emitted token differing from the in-session FP16 run; problems identical to FP16 through the shared generation get 64. θ_B = mean_i (Y^NAT − Y^C1) (tokens; negative = NAT departs earlier). Accuracy contrasts (paired, strict outcome of the historical scorer) are **descriptive** (power ≈ 5–6 points).

## 6. Inference, clustering, multiplicity, sensitivity
- **Unit and clustering:** the problem. Steps are averaged within problem before inference; every interval resamples problems (10,000 percentile-bootstrap resamples, seed 20261101); p-values are two-sided paired sign-flip permutation tests (10,000). No step-level p-values.
- **Primary test:** θ_A at α = 0.05 (two-sided); verdict "difference" if its 95% CI excludes 0; "equivalent" if its 90% CI lies inside ±margin **and** b̂ ≥ 0.25; otherwise "inconclusive". The primary is tested first at full α; secondary tests are reported regardless of its result and form one **Holm family** (raw and adjusted p both reported). Descriptive items are labelled.
- **Sensitivity analyses (none replaces the primary):** steps 1–32; steps 9–64; steps 2–8 (few perturbed entries); per-problem median instead of mean; problems with ≥ 32 steps; relative-difference scale; step-pooled problem-clustered bootstrap; excluding exposed test problems; Tier B at τ = 32 and 128.
- **No interim analyses; no optional stopping; no sample extension.** If the ceiling truncates a run, inference uses the completed problems of the pre-specified order (a random prefix) and says so.

## 7. Justification of the equivalence margin (substantive, not convenience)
A matched control can only be as exact as the notion of "size" used to match it, and NF4's own perturbation fluctuates from step to step. We therefore call NAT and C1 *practically equivalent* if their effects differ by less than what a relative error of c in perturbation size would cause, where c is the **natural step-to-step relative variability of the NF4 perturbation's own size** (measured, 10–40 %), converted to the effect scale through the **measured dose–response** (b̂). If the effect is insensitive to size (b̂ < 0.25) the matched size cannot be informative and equivalence is not claimable. The ±5-token figure of the earlier draft is **withdrawn**. For Tier B no dose arm exists, so its relative margin c·mean(Y^NAT) is *supportive only*; Tier B conclusions rely on the estimate and its interval.

## 8. Budget, gates and stop rules
Approved: E1 design in principle; **absolute ceiling 8.0 GPU-hours cumulative** for everything E1 does on the GPU (gates, calibration, Tier A, Tier B); no flag raises it; truncation is never extended. Estimated (from Stage 1/2 timings; to be replaced by gate G6): gates 0.3 h, calibration 0.9 h, Tier A 2.0 h (6 arms), Tier B 4.1 h (FP16, NAT, C1, C2) ⇒ ≈ 7.3 h. **Pre-declared reduction order if G6 projects > 8 h:** (1) Tier B drops C2; (2) Tier B n 400 → 300; (3) drop C1\*. Gates: **G1** α = 0 equals FP16 bitwise (tokens and cache; 20 train problems × 64 tokens × C1, C2); **G2** realized perturbation norm within the tolerance (mean per-step relative deviation ≤ 1 %, per-slot maximum ≤ 10 % — fp16 rounding of the stored entry sets the floor, measured 0.05 %/2.3 % on CPU with real weights, deviations judged against max(target, 2^-10 of the entry norm)); **G3** all entries and logits finite; **G4** same seed ⇒ bitwise identical, different seed ⇒ different; **G5** NAT via the new path vs the stored engine cache-only generations on 24 test problems (reported, not a stop); **G6** runtime projection ≤ ceiling. Per-problem integrity checks (step-1 JS exactly 0, finite, non-negative, norm tolerance, shared first token) stop the run on failure. GPU temperature/clock stops as in Stage 1. Hardware-safety or correctness failure ⇒ stop and report.

## 9. CPU validation required before the lock (done; see the validation report)
`tests/run_controls_tests.py` (13 tests + a real-model CPU smoke), `tests/mutation_check_controls.py` (15 injected bugs, all caught), `tests/test_analyze_controls.py` (10 tests on simulated truth, incl. type-I error and equivalence logic), `run_controls.py cpu-smoke`.

## 10. Interpretation fixed in advance
| Outcome | Reading allowed |
|---|---|
| θ_A > 0, CI excludes 0 (NAT more disruptive than same-size random) | NF4-written entries are not just "a perturbation of that size". If NAT ≈ C2 ≠ C1 the per-coordinate magnitude profile carries it; if NAT ≠ C2 sign/direction structure matters; if C1\* ≈ NAT at α\* > 1 the extra effect is equivalent to a larger random dose |
| θ_A < 0, CI excludes 0 | Same-size random noise is more disruptive than NF4's structured difference |
| Equivalent (90% CI ⊂ ±margin, b̂ ≥ 0.25) | At this resolution the immediate effect is explained by perturbation size; "NF4-written" is not needed beyond size |
| Inconclusive | State the interval; no mechanism claim; no extension |
Tier B may disagree with Tier A (history compounding); then both are reported and the paper makes the narrower statement. A flat dose–response, a failed gate or truncation is reported as such.

## 11. Amendments and deviations
Any change after the lock is logged with date and reason in `validation_logs/e1_deviation_log.md`; changes cannot alter hypotheses, estimands, margin rule or sample.
