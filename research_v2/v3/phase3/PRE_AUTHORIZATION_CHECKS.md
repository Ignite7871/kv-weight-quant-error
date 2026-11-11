# Pre-authorization checks (CPU only, 2026-10-29)

These checks used the locked code unchanged, called from scratch scripts outside the repository. Lock id
169e756ffbcfb725 still verifies; no locked file was edited.

## 1. Remaining models load with the expected geometry

The checks are the same as the locked real-model test (`run_v3_tests.py --real`): geometry from the live cache,
routed-module count, the LOC/INH maps, all 14 arms at α = 0 bitwise equal to FP16 (logits and cache), and the
integrity check at α = 1.

| Model | Layers | Query / KV heads | head_dim | FP16 cache | Routed modules | LOC NF4 / INH NF4 | α = 0 bitwise, 14 arms | Integrity errors |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B (locked validation) | 24 | 14 / 2 | 64 | yes | 168 | 48 / 120 | yes | none |
| **Qwen2.5-0.5B-Instruct** | 24 | 14 / 2 | 64 | yes | 168 | 48 / 120 | yes | none |
| **Llama-3.2-1B (base)** | 16 | 32 / 8 | 64 | yes | 112 | 32 / 80 | yes | none |
| Llama-3.2-1B-Instruct | 16 | 32 / 8 | 64 | yes | 112 | 32 / 80 | yes | none |

Llama-3.2-1B base was also covered by the locked validation run.

## 2. Interruption / resume behaviour of the locked driver

Test setup: a tiny Qwen2 sized to the real tokenizer, real GSM8K prompts and 8 steps. Each scenario was run in a fresh
folder and compared with an uninterrupted run under the same model label.

| Scenario | Result |
|---|---|
| A. Interrupted while computing a problem | **Supported.** Nothing is written for the interrupted problem; on resume it is rerun. Records completed before the interruption are **byte-identical** after resume. Exactly one record per problem and arm. The rerun equals an uninterrupted run except for `wall_seconds`. |
| B. Interrupted while appending a problem's records (between arm files; a window of milliseconds after ~1–2 min of computation per problem) | **Not clean.** Arm files already written for that problem get a **second, identical record** on resume (5 of 15 files in the test). The records are identical to each other and to an uninterrupted run (except timing), and the analysis loader keys records by problem index, so no estimate changes. But the files are not byte-identical to an uninterrupted run. Calibration does not have this problem (it filters already-written arms). |
| C. Torn (partially written) last line, e.g. a power loss mid-write | The driver **stops** with a JSON decode error on resume. It never continues silently. Recovery needs the incomplete line removed, which would be a logged deviation. |
| D. Calibration interrupted after the accumulator update and during the record append | **Supported.** One record per problem and arm; the accumulator counts each problem exactly once. |
| E. GPU-hour accounting | **Cumulative across sessions** against 25.0 h: every process adds its line to one ledger (`v3_runs/gpu_ledger.jsonl`), and the GPU mutation check writes to the same ledger. A process that dies uncleanly is charged on the next start from its last heartbeat (written every ~15 s): 1,800 s charged in the test, cumulative total exactly 1.0 + 0.033 + 0.5 h. The watchdog fires at the ceiling minus all prior use. The per-problem loop never starts a problem whose worst-case time could cross the ceiling. |

**Residual gap.** The GPU mutation check writes its ledger line in a `finally` block but has no heartbeat. A hard kill
or power loss during that run (minutes) would not be charged.

**Fixing B cleanly requires a lock amendment** (the one-line already-written filter that calibration uses). Locked code
was not changed.
