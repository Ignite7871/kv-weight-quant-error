# Stage 1 report — token source × KV-cache source (Qwen2.5-0.5B, GSM8K, greedy)

Status: Stage 1 complete, 2026-10-24. **Stage 2 and the comprehensive study are not started and remain unauthorized.** Nothing was pushed, published or submitted; the archived TMLR submission is untouched.

**Framing.** This is an *exploratory replication with pre-declared estimands on previously examined data* (the 930-problem test pool), under a *privately locked protocol informed by prior exploration — not a pre-registration* (`STAGE1_PROTOCOL_LOCK.txt`, sha256 `2344f57b…eb11`, locked commit `4b0ff0b`; amendments A1 and A2 in `STAGE1_SPEC.md` §10). The 389 reserved test problems were never touched. Historical outcomes were reused for continuity only and are not independent confirmation. Every causal statement below is conditional on the specific interventions, greedy decoding, this model/prompt/benchmark and the FP16 + bitsandbytes-NF4 stack.

## 0. Run summary

| Item | Value |
|---|---|
| Achieved sample | **930 / 930 problems, complete on all five arms; not truncated** |
| Cumulative GPU wall time | **10.885 h of the 12.0 h ceiling** (gates 0.358 h, main run 10.528 h) |
| Sessions (main run) | S1: problems 1–115, 1.221 h, 38.1 s/problem · S2: 116–155, 0.520 h, 46.7 s/problem (stopped itself under the pre-A2 projection rule) · S3 (after A2): 156–930, 775 problems, 8.787 h, 40.7 s/problem |
| Per-arm ms/step, S1 / S2 / S3 | FP16 33.6 / 40.2 / 34.1 · NF4 53.9 / 62.7 / 55.5 · cache-only 85.0 / 102.9 / 88.3 · token-only 86.0 / 97.5 / 87.9 · flip oracle 87.2 / 104.5 / 89.5 |
| Slowdown | S2 ran 17–22 % slower per step in every arm; it did **not** persist in S3 (per-step cost 1.5–4 % above S1 in every arm). Cause of the S2 slowdown unknown (no telemetry existed then). S3: GPU 53–78 °C at problem boundaries (stop limit 88 °C). 10 of 80 pace logs (problems 385–635) raised the last-20 > 1.15 × pooled flag (max 1.81×) in windows of long generations; the length-controlled per-step cost did not rise, and the pooled projection stayed 10.9–11.2 h throughout |
| Checkpoint integrity | First 155 records of each arm byte-identical to the snapshot (SHA-256 per line) before **and** after the run; all five arms' index sequences equal the seeded run order exactly (930 each; no repeats, no skips); no reserved problem touched |
| Per-problem integrity checks (A2) | 0 failures in 775 problems at run time; re-check of all 930 after the run: 0 failures |

## 1. Correctness and equivalence reports

All gates were passed before the main run (details in `validation_logs/`; lock note table):

- **CPU:** 24/24 pilot suite and 5/5 pilot mutants killed; Stage 1 suite and mutation check (6/6 killed); analysis tests 8/8. After A2: suite 18/18 (including real-Qwen CPU hybrids vs an independent deep-copy reference and the A2 run path), A2 mutation check 15/15, Stage 1 mutation check 6/6 (`validation_logs/stage1_a2_*`).
- **GPU equivalence (G3):** pure cells ≡ static FP16/NF4 arms ≡ the untouched `DynamicQwen.generate` (10/10, tokens and caches bitwise); hybrids ≡ an independent reference that deep-copies the cache (6/6, caches bitwise, and equal to a single-precision teacher-forced replay of the emitted history); early stop identical (14/14).
- **Feasibility (G4):** 10 train problems; token = argmax of its declared source at every step in both hybrids; both hybrids diverge from FP16; flip oracle ≡ cache-only until its first flip. Passed after amendment A1 (a float32 round-off of −9.7e-9 in an auxiliary JS-divergence sanity bound; first failing report preserved).
- **Main run:** the A2 per-problem checks (token = argmax of its declared source at every step, step-1 branch labels consistent with the static arms, KV letters, JS range, flip oracle ≡ cache-only until its first flip, first token identical across arms, identical-history ⇒ identical outcome, memory) held for all 930 problems. Non-diverged problems always had the same outcome as FP16 (0 exceptions in every arm).
- **Continuity:** the regenerated static FP16 and NF4 arms reproduce the historical per-problem outcomes on **930/930** problems (336 and 243 solved, 0 disagreements). This shows the harness reproduces the historical pipeline; it is not independent confirmation of a hypothesis.

## 2. Stage 1 results

**Accuracy (n = 930, `EXTRACTION_FAILED` = 0 in every arm)**

| Arm | token source | KV source | Solved | % |
|---|---|---|---|---|
| FP16 | F | F | 336 | 36.13 |
| NF4 | N | N | 243 | 26.13 |
| cache-only | F | N | 314 | 33.76 |
| token-only | N | F | 303 | 32.58 |
| flip oracle (diagnostic) | F | F on flip steps, N otherwise | 307 | 33.01 |

**Primary family (paired, problem-level bootstrap, 10,000 resamples; Holm over five):**

| Contrast | Estimate (pp) | 95 % CI | Discordant (+/−) | p (sign-flip) | Holm p | ±3 pp TOST |
|---|---|---|---|---|---|---|
| cache-only − FP16 | −2.37 | [−4.84, +0.11] | 58 / 80 | 0.074 | 0.149 | inconclusive |
| token-only − FP16 | −3.55 | [−6.34, −0.86] | 64 / 97 | 0.010 | **0.041** | differs |
| **interaction** (NF4 − cache-only − token-only + FP16) | **−4.09** | [−7.42, −0.65] | 88 / 131 | 0.021 | 0.063 | differs (unadjusted) |
| cache-only − NF4 | +7.63 | [+4.94, +10.43] | 122 / 51 | <0.001 | <0.001 | differs |
| flip oracle − cache-only | −0.75 | [−2.26, +0.75] | 22 / 29 | 0.394 | 0.394 | **equivalent within ±3 pp** |

Exploratory secondary contrasts: NF4 − FP16 = −10.0 pp [−12.9, −7.1] (55 / 148); token-only − NF4 = +6.45 [+3.66, +9.25]; flip oracle − FP16 = **−3.12 [−5.70, −0.54]**; flip oracle − NF4 = +6.88 [+4.09, +9.68].

Descriptive identity (not an attribution): −10.0 = (−2.37 cache) + (−3.55 token) + (−4.09 interaction). **No share of the loss is attributed to either source**; the singly-quantized arms recover only part of the NF4 gap, and the leftover is the interaction term.

Paired 2×2 tables vs FP16 (both / arm-only / FP16-only / neither): NF4 188/55/148/539; cache-only 256/58/80/536; token-only 239/64/97/530; flip oracle 248/59/88/535 (`paired_outcomes.csv` has every problem).

## 3. Mechanisms the results support (conditional on these interventions)

1. **Both paths matter, and they differ in kind.** The cache path alone (FP16 tokens, NF4 K/V) leaves trajectories close to FP16 for longer: 89.6 % of problems diverge, median first divergence at token 21, and the per-step hazard of leaving the FP16 trajectory is roughly constant (2.7 % / 2.8 % / 2.0 % in steps 1–20 / 21–60 / 61+). The token path alone (NF4 argmax, FP16 K/V) behaves like NF4 early: 95.9 % diverge, median 10, hazard 6.2 % in steps 1–20 falling to 3.4 % and 1.9 %. NF4 itself: 97.5 % diverge, median 9, hazard 7.2 % → 4.4 % → 2.2 %.
2. **Trajectory divergence is sub-additive but accuracy loss is super-additive.** Per-step divergence hazard of NF4 (7.2 % in steps 1–20) is below the sum of the single-path hazards (8.9 %); yet the accuracy loss of NF4 (−10.0 pp) is *larger* than the sum of the single-path losses (−5.9 pp; interaction −4.1 pp). Divergence is not accuracy loss: most diverged trajectories still reach the right answer, and many change the outcome in either direction (§5). The sub-additive hazard is a description, not a test; the super-additive accuracy interaction is the more interesting and less certain result (see below).
3. **Local branch disagreement is common:** the two branches' argmaxes differ at 3.6 % (cache-only) and 3.5 % (token-only) of steps; each hybrid keeps a coherent single-source history, so these disagreements are the points at which the intervention can matter.
4. **Outcome changes occur almost only after divergence** (by construction, plus the check that non-diverged problems never differ). Among problems that diverge at tokens 1–20, losses outnumber gains in every arm (NF4 115 lost vs 47 gained; cache-only 45 vs 32; token-only 78 vs 53); for divergence at tokens 21–60, NF4 30 vs 8, cache-only 29 vs 25, token-only 19 vs 10. Late-diverging cache-only trajectories are nearly outcome-neutral (29 lost vs 25 gained), which is consistent with — but does not prove — a smaller systematic harm from the cache path.

**What the data do not support:** a claim that the cache is responsible for a stated percentage of the loss; that the flip oracle bounds or preserves accuracy; that the cache effect is distinguishable from zero (cache-only − FP16 has a CI that includes 0, and is not equivalent within ±3 pp); or that the interaction survives the family-wise correction (Holm p = 0.063; the unadjusted CI excludes 0).

## 4. Comparison with the pilot's flip-oracle observation

- **Pilot (50 train problems, full 256 tokens):** the flip oracle diverged from FP16 on 50/50 problems, median first divergence token 20 (min 2, max 111), although every emitted token was FP16's argmax; NF4 diverged earlier (median 6).
- **Stage 1 (930 test problems, early-stopped at the drift marker):** flip oracle diverges on 837/930 (90.0 %), median 21 (min 2, max 223); NF4 907/930, median 9. The medians reproduce. The lower divergence share (90 % vs 100 %) is not explained by the data: 93 Stage 1 problems are token-for-token identical to FP16, and the pilot used different (train) problems and full 256-token generations.
- **New information:** the oracle's trajectories are almost indistinguishable from cache-only's (837 vs 833 diverged, same median, 22/29 discordant, equivalent within ±3 pp). Both policies emit FP16 argmax tokens throughout, so the oracle's divergence is reproduced by the NF4 K/V it still stores on the steps where the branches agree; swapping in FP16 K/V only on flip steps changes little. And the oracle is **not accuracy-lossless**: it solves 307 vs FP16's 336 (−3.1 pp, CI −5.7 to −0.5, exploratory). Emitting FP16's argmax at every step does not preserve FP16 accuracy when the stored K/V is NF4.

## 5. Contradictory or surprising cases (all listed in `paired_outcomes.csv` / report JSON)

- **Gains over FP16:** cache-only solved 58 problems FP16 got wrong, token-only 64, flip oracle 59. Hybrids are not "FP16 plus noise that only hurts": the net loss is the difference of two large flows (e.g. cache-only 58 gained vs 80 lost). Greedy trajectories are chaotic, and some changes in the answer are in the right direction.
- **Non-monotone recovery:** of the problems FP16 solved and NF4 got wrong (148), 50 are solved by **both** single-path interventions, 33 by neither, and the rest by exactly one — so rescue by quantizing only one source is common, but quantizing both loses it (the interaction).
- **Both references right, hybrid wrong:** 20 problems solved by FP16 and NF4 are wrong under cache-only, 26 under token-only: hybrids can fail where both pure arms succeed.
- No arm produced an `EXTRACTION_FAILED` outcome; the 256-token cap is an arm-independent bound.

## 6. Runtime and memory cost

Hybrid cells run two network passes per step: 1.50 FP16-pass-equivalents per step (byte model) and 88 ms/step vs FP16 34 and NF4 56 (laptop GPU, host-launch-bound — the NF4 weights are not faster in wall time). Peak allocated memory ≤ 1.38 GB for every arm. The hybrids are **diagnostic instruments, not efficient inference policies**; their cost shows nothing about deployable savings.

## 7. Limitations

- One model (0.5B), one benchmark, greedy decoding, one numerical stack. The 930-problem pool was examined before; this is an exploratory replication, not confirmation, and the protocol lock is private, not a pre-registration.
- The minimum detectable paired effect at n = 930 is about 2.3–4.3 pp. Cache-only − FP16 (−2.4 pp) is below that: it is neither established nor excluded. Of the primary family only token-only − FP16, cache-only − NF4 survive Holm at 0.05; the interaction does not (Holm p = 0.063) and should be read as suggestive.
- The "cache" intervention swaps whole-position K/V written by whole-network NF4 passes; it does not isolate layers, heads, or K versus V. The "token" intervention changes which logits select the token; histories differ afterwards, so contrasts are between *trajectories*, not between matched positions.
- Amendment A2 (post-lock; runtime/stopping only) and A1 (auxiliary tolerance) are documented; the generation code, order, sample and analysis are byte-identical to the lock. The first 155 problems were produced before A2 under the old driver; their records are unchanged. S2 ran ~20 % slower per step for an unknown reason (timing only).
- Timings are laptop-clock dependent; byte cost is the reliable unit.

## 8. Updated publication-contribution assessment

- **What is new and defensible:** a clean 2×2 factorial separating *which logits choose the token* from *which precision wrote the stored K/V*, validated bitwise against independent references; evidence that (i) the flip-oracle's divergence is explained by cache precision, (ii) a locally FP16-agreeing policy still loses ≈3 pp, (iii) accuracy damage is not additive over the two paths. These bear directly on why per-step routing/oracle arguments in the rejected TMLR paper do not behave as a bound.
- **What limits it:** a single small model/task; modest effects with CIs that touch zero for the cache path; the key interaction is only unadjusted-significant; exploratory provenance. As it stands this supports a workshop paper or a focused analysis section, not a stand-alone strong-venue claim. It would become a sound main-venue contribution only with a locked validation (389 reserved problems) and at least one more model or task, and with the mechanism claim restricted to what the factorial identifies.

## 9. Recommendation on Stage 2

**Scientifically worthwhile only in a narrow form, and not before you decide.** In priority order:
1. **Protocol-locked validation on the 389 reserved problems** with the same five arms and only the primary family (≈ 389 × 40.7 s ≈ 4.4 h on this machine; it alone would test whether the interaction and the token-only effect replicate, which Stage 1 cannot do).
2. If (1) replicates, **one additional model** (same arms, same family) to test generality of the non-additivity.
3. Only then a mechanism-resolving study (layer/head or K-vs-V KV swaps; teacher-forced identical-history comparison) — otherwise the "what part of the cache" question stays open.

I do not recommend the comprehensive study before (1). Any of these needs fresh authorization and a new GPU budget; none has been started.

## Files

`stage1_analysis/` — `stage1_report.json` (all numbers above), `paired_outcomes.csv`, `divergence_survival.csv`, `survival.png`, `run_accounting/` (sessions, runtime log with telemetry, GPU ledger, checkpoint snapshot, final accounting with SHA-256 of the five record files). Raw records: `stage1_runs/eval/*.jsonl` (gitignored; hashes recorded).
