# Second-model feasibility pilot — report (Llama-3.2-1B-Instruct)

Run 2026-10-26/27 on the local RTX 4060 Laptop GPU under `PILOT2_SPEC.md` (lock re-created after a pre-GPU bug fix, see `validation_logs/e1_deviation_log.md`). No cloud or paid service. **Descriptive feasibility results only: no significance test was run, no threshold or model was chosen from any contrast, and none of this is evidence about the paper's hypotheses.** Raw records and the report: `pilot2_runs/llama32_1b_instruct/` (`fp16.jsonl`, `nf4.jsonl`, `cache_only.jsonl`, `pilot_report.json`).

## 1. Outcome
**Candidate 1 passed every pre-declared gate (P1–P7), so the backup (SmolLM2-1.7B) was not run.** GPU accounting: 4.4 s lost to the first, crashed launch plus 858.6 s for the pilot = **0.242 h of the 2.4 h ceiling** (ledger `pilot2_runs/gpu_ledger.jsonl`). No hardware-safety or correctness stop occurred.

| Gate | Criterion | Result |
|---|---|---|
| P1 memory | < 7.0 GB peak | 2.97 GB peak during generation (4.6 GB after dual loading incl. caches) — pass |
| P2 numerics | non-degenerate FP16 outputs | pass |
| P3 baseline quality | FP16 ≥ 20 % lenient and ≥ 15 % strict | **44 % / 44 %** — pass |
| P4 non-degenerate intervention | branch disagreement 1–10 %; ≥ 50 % of cache-only runs depart within 64 tokens | **3.3 %**; **72 %** — pass |
| P5 engine equivalence (stop condition) | FF = static FP16 and NN = static NF4, tokens and cache bitwise, 8 problems × 64 tokens | all identical — pass |
| P6 speed | cache-only ≤ 132 ms/step | **53.4 ms/step** — pass |
| P7 format | ≥ 90 % strict-parseable FP16 outputs | **99 %** — pass |
Format validation (CPU, before GPU): prompt round-trips through the Llama-3 tokenizer, BOS added, prompt ends in `Answer:`, a `\nQuestion:` continuation is detected as a drift marker with consistent offsets; EOS = `<|eot_id|>`; no chat template (raw 3-shot completion prompt).

## 2. Reported numbers (100 fixed GSM8K train problems, greedy, ≤ 256 tokens)
| | FP16 | NF4 | cache-only |
|---|---|---|---|
| Accuracy, lenient scorer (Wilson 95 %) | 44 % (34.7–53.8) | 43 % (33.7–52.8) | 42 % (32.8–51.8) |
| Accuracy, strict `####` format | 44 % | 42 % | 42 % |
| ms per decode step | 21.9 | 34.4 | 53.4 |
| seconds per problem | 1.46 | 2.60 | 3.60 |
Descriptive (no tests): per-step FP16-vs-NF4 argmax disagreement 3.3 %; mean branch JS 0.0036; NF4 leaves the FP16 trajectory on 93/100 problems (median first divergence 11 tokens), cache-only on 76/100 (median 20.5; restricted mean over 64 tokens: NF4 18.1, cache-only 33.5). Per-problem outcome flips are common in both directions (FP16 vs NF4 differ on 21 problems; FP16 vs cache-only on 8) although net accuracy differs by only 1–2 problems.

## 3. What this does and does not show
- **Feasible:** the dual-weight code, the two-pass engine and the scoring pipeline run unchanged on a Llama-family model (16 layers, 8 KV heads, GQA, 128k vocabulary); the pipeline is **2–2.8× faster per step** than on Qwen2.5-0.5B (host-bound: 16 layers instead of 24), so a replication is cheap.
- **Qualitatively similar trajectory behaviour:** NF4 departs from FP16 early, cache-only later (cf. Qwen Stage 1/2: KM medians 9–11 vs 23–24 tokens). This is a pilot-size descriptive observation, not a replication.
- **Not shown / a caution for generalizability:** on this instruct model NF4 costs about **1 point** of GSM8K accuracy (n = 100, CI ± 10), not the 8–10 points seen on Qwen2.5-0.5B. The accuracy-pathway contrasts (H1–H5) would therefore be **uninformative** in a Llama replication (MDE ≫ effect); only trajectory-level outcomes (divergence timing, per-step distribution shifts, the E1 controls) could replicate. The model also differs from Qwen in being instruction-tuned and in its architecture, so any difference is not attributable to one factor.
- Prompt format: the raw few-shot completion prompt works for this instruct model (44 % FP16, 99 % parseable). The format was fixed in advance and not tuned.
- Models and thresholds were fixed before the run; the backup was not consulted because candidate 1 passed.

## 4. Deviation
The first launch crashed after 4.4 GPU-seconds from a keyword-argument bug in my P5 wrapper (and the same bug in the E1 gates). Fixed, a CPU harness for GPU-only paths added, both locks re-created before the successful run; no gate or threshold changed.
