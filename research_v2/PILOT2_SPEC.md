# Second-model feasibility pilot — specification (fixed before any pilot GPU use)

Prepared 2026-10-27. Authorized by the authors: **a separate GPU pilot with a 2.4-hour cumulative ceiling**; first candidate Llama-3.2-1B-Instruct, backup SmolLM2-1.7B; the pre-existing baseline-quality and correctness gates; **no full second-model replication**; no cloud or paid services. The pilot is a feasibility check, not a hypothesis test: no significance test is computed, no model or threshold is chosen from any contrast, and pilot results are not used as evidence about the paper's hypotheses. Private dated hash (`PILOT2_LOCK.txt`), not a pre-registration.

## 1. Design
- **Problems:** 100 fixed GSM8K **train** problems (`pilot2_sample.json`, seeded by `rv2-pilot2-train-2026-10-27`; disjoint from the three few-shot examples, every earlier pilot set and the E1 calibration set). Deterministic greedy decoding, ≤ 256 new tokens, FP16 prefill, drift-marker early stop, exactly the Stage 1 generation code (`run_stage1.run_one`) and historical scorer.
- **Prompt:** the project's raw 3-shot completion prompt (`Question: …\nAnswer:`) with the model's own tokenizer and default special-token handling (BOS added where the tokenizer adds it). **No chat template** in the pilot (it would change the stop rule and the intervention's meaning); if the raw format fails P3 the candidate fails — the format is not tuned to rescue it.
- **Format validation (CPU, before any GPU time):** prompt round-trips through the tokenizer; ends in `Answer:`; BOS handling recorded; EOS ids recorded; a completion that continues with `\nQuestion:` is detected as a drift marker by `find_boundary`/`would_stop_at_boundary` with consistent offsets. Failure ⇒ the candidate is not run.
- **Arms:** FP16, NF4, cache-only (FP16 token choice over NF4-written entries) with the verified engine; the same intervention definitions as the manuscript.
- **Order:** candidate 1; candidate 2 only if candidate 1 fails a gate (or its process cannot complete within the ceiling). The first candidate passing all gates is selected; the other is not run.

## 2. Gates (pre-declared; any failure ⇒ the candidate is infeasible; no gate is relaxed to rescue a candidate)
| Gate | Criterion |
|---|---|
| P1 loading/memory | dual FP16+NF4 load succeeds; peak allocated GPU memory < 7.0 GB |
| P2 numerics | FP16 outputs on every problem non-empty and not degenerate (> 3 distinct tokens); no NaN/Inf collapse |
| **P3 baseline quality** | **FP16 accuracy ≥ 20 % (historical lenient scorer) and ≥ 15 % under the strict `####` format** on the 100 problems |
| P4 intervention non-degenerate | branch-argmax disagreement (FP16 vs NF4 on the same state, cache-only run) between 1 % and 10 % of steps; cache-only departs from the FP16 trajectory within 64 tokens on ≥ 50 % of problems |
| P5 correctness (stop condition) | the engine's two-pass cells FF and NN reproduce the static FP16 and static NF4 decoders token-for-token **and cache-bitwise** on 8 problems × 64 tokens |
| P6 speed | cache-only ≤ 132 ms/step (1.5 × the Qwen2.5-0.5B Stage 1 cost of 88 ms) |
| P7 format | ≥ 90 % of FP16 outputs contain a parseable `#### number` before the drift marker |
Futility stop: if FP16 solves fewer than 2 of the first 30 problems, the candidate is stopped (P3 cannot realistically be met; saves budget).

## 3. Reported (descriptive, no tests)
FP16, NF4 and cache-only accuracy with Wilson 95 % intervals (lenient and strict), ms/step and s/problem by arm, peak memory, gate results, GPU hours used, tokenizer facts.

## 4. Safety and budget
Cumulative pilot GPU time (loads and checks included) ≤ 2.4 h, absolute, enforced by the Stage 1 ledger/watchdog (no override). Hardware stops (temperature/clock) as in Stage 1. Any correctness (P5) or hardware-safety failure stops the pilot and is reported. Expected: ≈ 0.7–0.9 h per candidate.
