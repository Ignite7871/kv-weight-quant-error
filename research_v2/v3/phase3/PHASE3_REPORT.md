# v3 Phase 3 — CPU implementation, validation and lock

2026-10-29, branch `v3-novelty`. **No GPU was used.** Every check below ran on CPU. The numbers come from
`validation_logs/v3_validation_summary.json`, `v3_cpu_validation.json`, `v3_mutation_check.json` and
`v3_cpu_harness.json`.

## What was built (new files only; locked engine and E1/E1L code imported unchanged)

| File | Role |
|---|---|
| `rv3/core.py` | Controlled step, the 14 arms, LOC/INH routing via the existing `DynamicLinear.layer_force_4bit`, read-only post-RoPE query capture, the predictors (§6), the calibration accumulator, synthetic and calibration contexts |
| `run_v3.py` | Driver: plan / ledger / status / gates / calibrate / g6 / tierA. Authorization, lock, order and ledger guards. 25.0 h absolute ceiling (Stage 1 ledger/watchdog machinery) |
| `analyze_v3.py` | Calibration constants (b̂, c, margin, U₈, E₈, r₈₀, σ², outlier channels, ρ\*, s_K, s_V, the H-D rule) and every estimand / family / decision rule of §8 |
| `tests/run_v3_tests.py` | 14 tests × 2 tiny families (Qwen2 4/2 heads, head_dim 16; GQA Llama 8/2 heads, head_dim 64), `--real` real-model CPU checks, `--static` E1/E1L lock checks. Device-agnostic (reused on CUDA) |
| `tests/test_analyze_v3.py` | 11 tests on simulated truth |
| `tests/mutation_check_v3.py` | 26 core + 11 analysis mutants |
| `tests/cpu_harness_v3_paths.py` | The driver's gate, calibration, G6, Tier A and analysis code end to end on CPU (tiny model, real tokenizer and GSM8K prompts) |
| `tools/lock_v3.py`, `tools/v3_gpu_mutation_check.py`, `tools/v3_cpu_validation.py` | Lock manifest; CUDA mutation check (refuses without authorization); validation runner |
| `v3_sample.json` | The 200 test ids and 100 calibration ids copied from the locked E1 sample (sha256 of e1_sample.json recorded); gate subsets are prefixes |

## Validation results (all on CPU)

| Step | Result |
|---|---|
| Tiny-model suite (both families) | **28/28** passed |
| Real Qwen2.5-0.5B on CPU | geometry 24 layers / 2 KV heads / head_dim 64, FP16 cache; 168 routed modules; α = 0 bitwise for all 14 arms; no integrity errors at α = 1 |
| Real Llama-3.2-1B on CPU | 16 / 8 / 64, FP16 cache; 112 routed modules; α = 0 bitwise for all 14 arms; no integrity errors |
| E1 and E1L locks verify (static) | yes / yes |
| Analysis on simulated truth | **11/11** |
| CPU harness of the GPU code paths | pass |
| v3 mutation check | **37/37 caught** |
| E1 mutation check (unchanged) | 15/15 caught |
| E1L mutation check (unchanged) | 19/19 caught |

The checks are independent of the code under test:
- predictors are compared with a slow float64 loop that uses its own hooks and its own rotary implementation (1e-6
  relative);
- stored entries are checked against norms computed in the test;
- routing is checked against a module map built in the test;
- earlier cache entries are snapshotted around every step.

The analysis tests confirm:
- type-I error within bounds;
- H-C bands recovered from planted truth (substantial, partial, no, not applicable);
- the non-replication finding text;
- the H-D extrapolation rule (s ∉ [0.5, 2] ⇒ descriptive);
- 2×2 contrasts and the 25 % SD guard;
- Holm;
- Spearman with ties.

## Defects found and fixed before the lock

The harness found **two defects that would have stopped the GPU run**. Neither affects any estimand.
1. The calibration accumulator file was written before its folder existed, so the first calibration problem would
   crash. Fixed: the folder is created first.
2. `NAT-K*.jsonl` is not a valid Windows file name, so the first Tier A problem would crash. Fixed: record files use
   `star` (`NAT-Kstar.jsonl`). Arm names are unchanged.

Also during implementation:
- the post-RoPE query is rotated in float64 (from the FP16 pre-RoPE query) so that the predictor definition is
  numerically exact;
- the bootstrap counts are read at call time (locked values unchanged: 10,000).

## Pre-lock protocol notes (r4; recorded in PROTOCOL_V3.md §14, item 11)

- The GPU mutation check runs once, before the first model. It uses tiny models and is not model-specific.
- Record file names write `*` as `star`.
- The CPU harness is part of the validation.

No hypothesis, estimand, arm, sample or decision rule changed.

## GPU sequence after authorization (Phase 4)

1. `venv/Scripts/python.exe research_v2/tools/v3_gpu_mutation_check.py --authorized-gpu-run`
2. For each model in the fixed order (qwen-base, llama-instruct, qwen-instruct, llama-base):
   1. `run_v3.py gates --model M --authorized-gpu-run`
   2. `run_v3.py calibrate --model M --authorized-gpu-run`
   3. `analyze_v3.py calibrate --model M`
   4. `tools/lock_v3.py --append-calibration M`
   5. `run_v3.py g6 --model M --authorized-gpu-run`
   6. first model only: `tools/lock_v3.py --append-design`
   7. `run_v3.py tierA --model M --authorized-gpu-run`
3. After all four models: `analyze_v3.py analyze`, then the independent recomputation (Phase 5).

Every GPU step refuses unless its prerequisites hold.

## Amendment 1 (2026-10-29, before any GPU use)

See PROTOCOL_V3.md §15 and DEVIATION_LOG_V3.md V3-A1. Effects on the Phase 4 procedure:
- If `run_v3.py` stops with "incomplete or invalid record file(s)", run
  `tools/v3_repair_torn_line.py <file> --reason "<event>"` (logged; adds a deviation row), then rerun the same `run_v3.py` command.
- The GPU mutation check now has a ledger heartbeat.
- Analysis fails loudly on any duplicate record.

## Amendment 2 (2026-10-29, after the qwen-base gate failure V3-D04, before any further GPU use)

See PROTOCOL_V3.md §16, DEVIATION_LOG_V3.md V3-D04, V3-A2 and V3-D05, and `v3/phase4/DIAG_V3_D04.md`.

What changes for Phase 4:
- G2 and the per-problem integrity check are now the construction check (float64, ≤ 1e-4) and the rounding check (bitwise), in every arm.
- The post-rounding per-slot norm deviation is no longer a stop criterion. It is reported in the analysis.
- qwen-base gates are rerun under the amended G2. The failed attempt's report is kept as `v3_runs/qwen-base/gates/gates_attempt1_failed_V3-D04.json`.
- The authorization file names the lock id, so the authors must update it to the new lock id before any GPU step.

