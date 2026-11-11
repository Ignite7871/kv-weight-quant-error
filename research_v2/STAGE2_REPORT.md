# Stage 2 report — protocol-locked held-out validation (Qwen2.5-0.5B, GSM8K, greedy)

Status: Stage 2 complete, 2026-10-25. **No further experiment has been started.** Nothing was pushed, published or submitted; the archived TMLR #13386 submission is untouched. All commits are local on `research-recovery`.

**Framing.** The five hypotheses were drawn from the Stage 1 results (exploratory origin). The 389 held-out problems are the complement of the Stage 1 pool; no generated-answer correctness outcome was retained for any of them, but **they are not completely untouched**: two (test indices 4, 5) appeared in an earlier eight-problem console probe, and 47 (test index < 200, including those two) had their gold solutions token-labelled by a teacher-forced FP16/INT4 argmax comparison (`results/eqf_spike_dataset_*.csv`). All 389 are in the primary validation; the pre-declared sensitivity analysis excludes the 47 (342 problems). The lock is private and dated (`STAGE2_PROTOCOL_LOCK.txt`, sha256 `8241c4dc…0523`, commit `4dd8ebb`, created 2026-10-25 04:54 UTC before any GPU use), **not a public pre-registration**. Hypotheses, contrasts, Holm family, replication categories and analysis code were not changed after any held-out result existed; there were no protocol amendments.

## 0. Summary

| | Held-out (n = 389) | Reading |
|---|---|---|
| H1 cache-only − FP16 | −4.63 pp [−8.48, −0.77]; raw p 0.027, **Holm 0.137** | same direction as Stage 1 (−2.37); not Holm-significant |
| H2 token-only − FP16 | −4.37 pp [−8.23, −0.51]; raw p 0.035, **Holm 0.141** | same direction as Stage 1 (−3.55); not Holm-significant |
| H3 interaction | **+0.77 pp [−4.11, +5.91]**; raw p 0.84, Holm 1.0 | **sign differs from Stage 1 (−4.09), but the CI includes both 0 and the Stage 1 value: uncertainty, not demonstrated disagreement** |
| H4 cache-only − NF4 | +3.60 pp [−0.77, +7.97]; raw p 0.129, Holm 0.388 | same direction as Stage 1 (+7.63); smaller; inconclusive |
| H5 flip oracle − cache-only | +0.77 pp, 90 % CI [−1.29, +2.83] | **equivalence within ±3 pp replicated** |

The honest one-line summary: *the two single-path effects and the oracle ≈ cache-only result replicate in direction (and the oracle finding in equivalence); the Stage 1 non-additivity (interaction) does not, and nothing in the held-out sample is Holm-significant, as the pre-declared design power (10–21 % for H1–H3) anticipated.* In the secondary pooled sample (n = 1,319, not independent confirmation) H1, H2 and H4 are Holm-significant and the interaction is not.

## 1. Run, checks and accounting

| Item | Value |
|---|---|
| Sample / completion | **389 / 389 problems complete on all five arms, not truncated** |
| Stage 2 GPU time | **4.178 h of the 5.5 h ceiling** (correctness check 0.137 h, speed check 0.021 h, held-out run 4.020 h = one process, 389 problems); 1.32 h unused. The pre-run projection was 4.71 h central (4.43–5.37 h) |
| Correctness check (GPU, before the run) | passed: pure cells ≡ static arms ≡ untouched `DynamicQwen.generate` (tokens and caches bitwise, 6/6 A-checks); hybrids ≡ independent deep-copy reference and single-precision replay (6/6 B-checks); early stop identical (C 6/6, C-hybrid 4/4) |
| Speed check (before the run) | FP16 33.8, NF4 49.9, cache-only 79.3 ms/step (equal to or faster than Stage 1); tokens deterministic across repetitions |
| Per-problem integrity checks | 0 failures in 389 problems at run time; re-check of all records afterwards: 0 failures. Non-diverged problems always had the FP16 outcome (0 exceptions, every arm) |
| Checkpoint | the five arm files hold exactly `run_order[0:389]` in order; no repeats, no skips; all indices in the held-out set; none from the 930-problem pool |
| Hardware | GPU 55–77 °C (sampled at problem boundaries; stop limit 88 °C), no hardware stop |
| Runtime | 37.1 s/problem (arms' wall sum) vs 40.7 in Stage 1's last session; per-step cost FP16 31.2 / NF4 50.2 / cache-only 80.2 / token-only 80.2 / oracle 81.7 ms vs Stage 1 34.1 / 55.5 / 88.3 / 87.9 / 89.5. 6 of 41 pace logs raised the last-20 > 1.15 × pooled flag (max 1.29×), consistent with windows of longer generations (not separately verified); length-controlled per-step cost was *lower* than in Stage 1, so no sustained machine slowdown. Peak allocated memory 1.37 GB (every arm) |
| Protocol deviations | (1) The first launch of `run` aborted before any GPU use (`verify_checkpoint` expected the five record files to exist); no model was loaded, the ledger was unchanged, and the run was relaunched after creating the empty files — no code change (`validation_logs/stage2_gates/deviation_log.md`). (2) None other. During the run only integrity, accounting and telemetry were examined; the run log's per-problem outcome letters were not read, and no accuracy or arm contrast was computed until the run was complete and verified |
| Record files | SHA-256 of the five record files in `stage2_analysis/run_accounting/final_accounting.json` |

## 2. Held-out results (n = 389)

**Accuracy** (`EXTRACTION_FAILED` = 0 in every arm), Stage 1 for comparison:

| Arm | Held-out solved | % | Stage 1 % (n = 930) |
|---|---|---|---|
| FP16 | 145 | 37.28 | 36.13 |
| NF4 | 113 | 29.05 | 26.13 |
| cache-only (F token, N KV) | 127 | 32.65 | 33.76 |
| token-only (N token, F KV) | 128 | 32.90 | 32.58 |
| flip oracle | 130 | 33.42 | 33.01 |

**Primary family** (paired; 95 % percentile bootstrap, 10,000 resamples; sign-flip permutation p; Holm over the five; raw **and** adjusted shown):

| Contrast | Estimate (pp) | 95 % CI | +/− problems | raw p | Holm p |
|---|---|---|---|---|---|
| H1 cache-only − FP16 | −4.63 | [−8.48, −0.77] | 21 / 39 | 0.0274 | 0.137 |
| H2 token-only − FP16 | −4.37 | [−8.23, −0.51] | 21 / 38 | 0.0353 | 0.141 |
| H3 interaction (NF4 − co − to + FP16) | +0.77 | [−4.11, +5.91] | 39 / 39 | 0.839 | 1.000 |
| H4 cache-only − NF4 | +3.60 | [−0.77, +7.97] | 44 / 30 | 0.129 | 0.388 |
| H5 oracle − cache-only | +0.77 | 90 % CI [−1.29, +2.83] | 13 / 10 | 0.685 | 1.000 → **equivalent (±3 pp)** |

Exploratory secondary: NF4 − FP16 −8.23 pp [−12.85, −3.60] (29 / 61; p 0.0016); token-only − NF4 +3.86 [0.00, +7.71]; oracle − FP16 −3.86 [−7.71, 0.00]; oracle − NF4 +4.37 [+0.26, +8.74].

Descriptive identity (**not an attribution**): −8.23 = −4.63 (cache path) − 4.37 (token path) + 0.77 (interaction). No percentage of the accuracy loss is attributed to the cache or to the token path; the estimand supports no such share (the interaction is uncertain and the contrasts are between trajectories).

## 3. Problem-level outcome patterns and disagreement counts

- Joint five-arm outcome patterns (S solved / W not, order FP16, NF4, cache-only, token-only, oracle): 26 distinct; **all five agree on 263 / 389 (67.6 %)** (191 WWWWW, 72 SSSSS), vs 603 / 930 (64.8 %) in Stage 1. Next most common: SWWWW 17, SWSSS 16, SWSWS 11, WSWWW 11, SWWSW 11, WSWSW 10 (`paired_outcomes.csv` lists every problem).
- Pairwise outcome disagreements (held-out; Stage 1 in brackets): FP16–NF4 90 (203), FP16–cache-only 60 (138), FP16–token-only 59 (161), FP16–oracle 61 (147), NF4–cache-only 74 (173), NF4–token-only 59 (174), cache-only–token-only 63 (185), **cache-only–oracle 23 (51)**, token-only–oracle 60 (200).
- Paired 2×2 vs FP16 (both / arm-only / FP16-only / neither): NF4 84/29/61/215; cache-only 106/21/39/223; token-only 107/21/38/223; oracle 107/23/38/221.
- Notable cases: cache-only solved 21 problems FP16 got wrong, token-only 21, oracle 23; 8 problems solved by both FP16 and NF4 are wrong under each hybrid; of the 61 FP16-solved / NF4-wrong problems, 17 are solved by both single-path hybrids and 17 by neither.

## 4. First-divergence summaries (from the FP16 trajectory)

| Arm | Held-out: share diverged, median first-divergence token (IQR) | Stage 1: share, median |
|---|---|---|
| NF4 | 97.2 %, 8 (4–18) | 97.5 %, 9 |
| token-only | 91.5 %, 9 (4–23) | 95.9 %, 10 |
| cache-only | 89.5 %, 20 (12–39) | 89.6 %, 21 |
| flip oracle | 88.7 %, 21 (12–41) | 90.0 %, 21 |

Per-step divergence hazard under identical history (steps 1–20 / 21–60 / 61+; Stage 1 in brackets): NF4 7.4 / 4.2 / 1.3 % (7.2 / 4.4 / 2.2); token-only 5.7 / 2.8 / 1.5 % (6.2 / 3.4 / 1.9); cache-only 2.9 / 2.5 / 2.1 % (2.7 / 2.8 / 2.0); oracle 2.7 / 2.4 / 1.8 % (2.6 / 2.7 / 2.3). Local branch disagreement (`tF ≠ tN`): 3.5 % of steps in both hybrids (Stage 1 3.6 / 3.5 %). Outcome changes occur only after divergence; among problems diverging at tokens 1–20, losses outnumber gains in every arm (NF4 48 lost vs 27 gained; cache-only 26 vs 14; token-only 26 vs 17; oracle 26 vs 16). The trajectory-level picture reproduces Stage 1 almost exactly.

## 5. Do Stage 1's effect directions replicate? (pre-declared categories, §6.3 of the spec)

| Contrast | Stage 1 | Held-out | Held-out − Stage 1 (95 % CI; permutation p) | Stage 1 value inside held-out CI | Category |
|---|---|---|---|---|---|
| H1 cache-only − FP16 | −2.37 | −4.63 | −2.26 [−6.88, +2.29]; 0.35 | yes | same direction; raw p < 0.05, not Holm-significant |
| H2 token-only − FP16 | −3.55 | −4.37 | −0.82 [−5.50, +3.91]; 0.76 | yes | same direction; raw p < 0.05, not Holm-significant |
| H3 interaction | −4.09 | +0.77 | +4.86 [−1.23, +10.95]; 0.13 | yes | **opposite point estimate, compatible with Stage 1 (difference CI includes 0): statistical uncertainty** |
| H4 cache-only − NF4 | +7.63 | +3.60 | −4.04 [−9.14, +1.13]; 0.12 | yes | same direction; not significant (inconclusive) |
| H5 oracle − cache-only | −0.75 | +0.77 | +1.52 [−1.33, +4.35]; 0.31 | yes | **equivalence replicated** |

Statistical uncertainty versus directional disagreement: no contrast meets the pre-declared definition of *directional disagreement* (opposite sign **and** a difference CI excluding 0). The interaction is the one contrast whose sign flipped; the difference CI is wide (±6 pp) and includes zero, so the data neither confirm nor refute Stage 1's non-additivity, though they provide no support for it either. The H4 effect shrank from +7.6 to +3.6 pp (compatible with sampling error and with Stage 1's estimate having been the largest contrast that drew attention). With low design power, the three non-significant results are *inconclusive*, not failures to replicate.

## 6. Secondary analyses (not the primary validation)

**Pre-declared sensitivity, n = 342 (47 previously labelled problems excluded).** Accuracy FP16 38.30 %, NF4 30.70, cache-only 33.92, token-only 33.33, oracle 34.21. H1 −4.39 pp [−8.48, −0.29], raw p 0.060, Holm 0.241; H2 −4.97 [−9.06, −0.88], raw p 0.032, Holm 0.160; H3 +1.75 [−3.51, +7.02], p 0.60, Holm 1.0; H4 +3.22 [−1.46, +7.89], p 0.21, Holm 0.62; H5 +0.29, 90 % CI inside ±3 pp (equivalent). The conclusions are unchanged by excluding the 47 (the cache-only bootstrap CI excludes 0 while the permutation p is 0.06: borderline).

**Combined, n = 1,319 — SECONDARY, not independent confirmation (it contains the Stage 1 data that generated the hypotheses).** Accuracy FP16 36.47 %, NF4 26.99, cache-only 33.43, token-only 32.68, oracle 33.13. H1 −3.03 pp [−5.16, −0.91], Holm 0.020; H2 −3.79 [−6.07, −1.67], Holm 0.004; H3 **−2.65 [−5.46, +0.15], raw p 0.072, Holm 0.144 (inconclusive)**; H4 +6.44 [+4.17, +8.79], Holm 0.0005; H5 −0.30, equivalent within ±3 pp. Total gap −9.48 = −3.03 − 3.79 − 2.65 (description only). Pooled first-divergence medians: NF4 9, token-only 10, cache-only 21, oracle 21.

## 7. Interpretation of the cache/token mechanism (conditional on these interventions)

Supported by Stage 1, held-out and pooled data:
1. **Each path costs accuracy.** Quantizing only the stored K/V (FP16 tokens) or only the token choice (FP16 K/V) each lowers GSM8K accuracy by about 3–5 pp relative to FP16 (pooled −3.0 and −3.8, Holm-significant; held-out −4.6 and −4.4, not Holm-significant at n = 389). The held-out sample cannot distinguish the two magnitudes (difference 0.26 pp).
2. **The flip oracle is not an accuracy bound and behaves like cache-only.** It emits FP16 argmax tokens at every step, yet diverges from FP16 on ~90 % of problems (median token 21) and loses about 3–4 pp (Stage 1 −3.12 [−5.70, −0.54]; held-out −3.86 [−7.71, 0.00]; exploratory), equivalent within ±3 pp to cache-only in both samples. Stored NF4 K/V alone reproduces the oracle's divergence; swapping in FP16 K/V only at flip steps changes little.
3. **Trajectory mechanics are stable.** NF4 and the token path diverge early and with a front-loaded hazard; the cache path diverges later with a roughly constant per-step hazard (~2.5 %). These reproduce across samples to within sampling noise.

**Not supported:** a stated percentage of the loss attributable to the cache; Stage 1's super-additive interaction (held-out +0.77 pp [−4.1, +5.9]; pooled −2.65 pp, CI includes 0). The data are compatible with approximately additive damage, with true non-additivity anywhere from about −4 to +6 pp at this n (held-out CI). Divergence from FP16 is not accuracy loss: most diverged trajectories keep their answer, and gains over FP16 occur alongside losses.

## 8. Limitations

One model (0.5B), one benchmark, greedy decoding, one numerical stack. n = 389: minimum detectable paired effect 5.5–7.5 pp for H1–H3, so non-significance is expected and uninformative. The hypotheses come from Stage 1 data, the held-out set is not completely untouched (§ framing), and the lock is private. The cache intervention swaps whole-position K/V from whole-network NF4 passes (no layer/head or K-vs-V resolution). Trajectories diverge under both interventions, so contrasts compare trajectories, not matched positions; greedy trajectories are chaotic, so accuracy differences can arise without a systematic precision effect. Timings are laptop-clock dependent.

## 9. Publication-readiness assessment

- **Defensible, replicated claims:** (i) quantizing either the stored K/V alone or the token choice alone lowers GSM8K accuracy by a few points, in two independent samples with the same sign and similar size; (ii) a locally FP16-agreeing policy (the myopic flip oracle) neither preserves FP16 trajectories nor FP16 accuracy, and is equivalent to cache-only; (iii) the trajectory-divergence statistics are highly reproducible; (iv) the factorial instrument itself, validated bitwise against independent references, and exactly reproducing the historical FP16/NF4 outcomes (930/930 in Stage 1).
- **Claims the paper should drop or hedge:** "accuracy loss is super-additive in the two paths" (does not replicate; pooled CI includes 0); any attributable-percentage statement; any claim that the cache path is the larger or smaller contributor.
- **Overall:** the evidence now supports a focused, carefully scoped contribution — "per-step agreement with FP16 is not a bound; stored low-precision K/V by itself degrades trajectories and accuracy; the two quantization paths each cost accuracy" — on a single small model and benchmark. That is suitable for a workshop paper or as a mechanism section of a larger paper, and is a sounder basis for a revision than the rejected submission; it is **not by itself a strong main-venue result** without a generality demonstration and an honest statement that the held-out effects are individually underpowered (only the pooled analysis, which is not independent, reaches Holm significance).

## 10. Recommendation: another model, or stop and write?

**Stop experimenting on this question and write the revised paper**, with the claims scoped as in §9. Reasons: (a) the held-out validation did its job — it kept the findings that are real and removed the one that was an artifact of the Stage 1 sample (the interaction); (b) the remaining uncertainty (is damage additive? how large is each path?) is a *sample-size* question, not a *model* question, and a second model would not resolve it unless run at n ≳ 930 per model, ≈ 11 GPU-h or more per model (Stage 1 cost 10.9 h for 0.5B; larger or different models would cost more and are unmeasured); (c) the paper's supportable claims do not require a second model.

A second model is scientifically justified **only if** the revised paper wants to claim generality ("the cache path alone degrades trajectories across model families") — that is a different, clearly separable claim, would need a headroom pilot first, and should be pre-registered as such. I recommend deciding that on what the paper must claim, not on this validation. Stage 3, further models and any additional experiment remain unstarted and unauthorized.

## Files

`research_v2/stage2_analysis/` — `stage2_report.json` (all numbers), `heldout/` (`heldout_report.json`, `paired_outcomes.csv`, `divergence_survival.csv`, `survival.png`), `combined_secondary/`, `run_accounting/` (sessions, runtime log with telemetry, GPU ledger, final accounting with record SHA-256, run log, gates). Gates and pre-run evidence: `validation_logs/stage2_*`. Specification and lock: `STAGE2_SPEC.md`, `STAGE2_PROTOCOL_LOCK.txt`, `STAGE2_LOCK_NOTE.md`. Raw records: `stage2_runs/eval/*.jsonl` (gitignored; hashes recorded).
