# Performance Investigation (Task A) — local RTX 4060 Laptop, 2026-10-23

**GPU time used:** 20.9 of the authorized 45 min (`perf_ledger.jsonl`: A1 6.9, B1 7.1, A2 6.9 min). No cloud.
**Settings:** Windows power scheme switched once, Balanced → vendor "Performance" for run B1, then restored to Balanced.
- Restore was verified twice: by `powercfg /getactivescheme` after B1, and again before and after A2.
- Not touched: clocks, voltages, power limits, thermal behaviour, NVIDIA driver settings, and the vendor "Turbo" scheme (raises power limits; deliberately avoided).
- The original Balanced configuration is active now.

## Workload (identical in every run; `bench.py`)

- Qwen2.5-0.5B with the historical `DynamicQwen` construction (FP16 + bitsandbytes NF4 resident).
- **Pass-level micro-benchmark** (fixed prompt, constant cache length of 467 tokens, FP16 and NF4):
  - 20 warm-up passes, then 5 blocks × 40 passes;
  - synchronous mode: synchronize after every pass, with CUDA events;
  - asynchronous mode: 40 back-to-back passes, then one synchronize.
- **Generation benchmark:**
  - 3 fixed GSM8K-train problems × {FP16, NF4} × 3 engines × 3 repetitions, 128 new tokens, greedy;
  - engines: rv2 *instrumented* (as used in the pilot), rv2 *lean* (`instrument=False`, identical computation without per-component synchronization or logging), and the untouched historical `DynamicQwen.generate`;
  - round-robin order with rotation, and one recorded warm-up generation per precision.
- **Telemetry:** `nvidia-smi` every 500 ms (SM/memory clocks, utilization, temperature, power, P-state, clock-event reasons).
- **Design:** A1 (Balanced) → B1 (Performance) → A2 (Balanced), so that drift over time can be separated from the configuration.

**Prediction integrity:** tokens were identical across all engines, repetitions and configurations, for every (problem, precision). The lean mode is token-, route- and pass-identical for every policy kind in the CPU suite (test `t9`).

## Results

Generation, ms per decode step, mean ± 95% CI over 9 timed generations:

| Run | FP16 historical | FP16 instrumented | FP16 lean | NF4 historical | NF4 instrumented | NF4 lean |
|---|---|---|---|---|---|---|
| A1 Balanced | 38.5 ± 1.0 | 40.0 ± 1.3 | 39.0 ± 1.6 | 61.9 ± 1.4 | 65.2 ± 2.4 | 63.1 ± 2.6 |
| B1 Performance | 41.7 ± 4.5 | 41.1 ± 4.1 | 38.2 ± 2.7 | 65.5 ± 9.0 | 68.9 ± 8.7 | 62.2 ± 4.9 |
| A2 Balanced | 39.8 | 39.7 | 36.0 | 60.6 | 66.8 | 63.0 |

Micro-benchmark, ms per pass, synchronous / asynchronous:

| Run | FP16 | NF4 |
|---|---|---|
| A1 | 39.0 / 36.2 | 66.1 / 62.3 |
| B1 | 35.5 / 34.5 | 65.1 / 60.4 |
| A2 | 35.9 / 33.2 | 57.7 / 54.9 |

CUDA-event time equals wall time to within 0.1 ms in every block.

## Diagnosis

1. **The workload is host/launch-bound, not GPU-bound.**
   - GPU utilization stays at 21–36%, the dominant clock-event reason is "GPU idle" (0x1), and clocks wander between 730 and 2,250 MHz across P0–P5.
   - Back-to-back asynchronous launches are only 3–8% faster than synchronized ones.
   - Per-step time is set by Python-side dispatch through 168 wrapped projection layers per step, plus the extra dequantization launches of the bitsandbytes NF4 path. This is why NF4 is ~1.6× **slower** than FP16 on this stack.
2. **The power plan has no detectable effect.** B1 against the mean of A1 and A2 on the lean engine: FP16 +1.7%, NF4 −1.4%. Both are inside run-to-run variation:
   - generation coefficient of variation is 6–12%;
   - micro-benchmark blocks drift by up to ±12% within a run;
   - the A1 vs A2 NF4 micro-benchmark differs by 13% under the same configuration.
3. **The instrumentation overhead is small:** instrumented / lean = 1.068 for both precisions (pooled over 3 runs), and lean ≈ untouched historical (FP16 0.95×, NF4 1.00×).
   - The 1.07–1.60× ratios seen in the pilot came mostly from drift and run order, as suspected at the time.
4. **Warm-up:** the first generation is 1–8% slower than the timed mean. Model loading takes about 7 s (warm file cache).
5. **Thermals:** maximum 65 °C, with no thermal or power-cap event reasons recorded.
6. **Open question:** the historical September runs implied about 36 ms per step for NF4-heavy generation; today the untouched historical code takes about 61–66 ms.
   - The difference is not the power plan and not the instrumentation.
   - Candidate causes, all untested: CPU frequency or turbo state under the vendor's own controls, driver 616.56, background load (many desktop processes hold GPU contexts), Windows updates.
   - Today's rates are used for all cost estimates, with ±20% uncertainty.

## Measured or derived speed changes available without changing predictions

| Lever | Effect | Status |
|---|---|---|
| Lean engine mode (no per-component synchronization or logging) | −6.4% time per step | measured; token-identical |
| Performance power plan | none detectable (±2%) | measured; reverted |
| **Early stop at the causal drift-marker rule** | runs 0.41–0.46 of the steps on test problems (≈105 vs ≈250 steps); outcome-identical by prefix stability | derived from pilot records; valid only where full-generation accounting is not needed |
| Reuse of the 930-problem historical FP16/NF4/entropy outcomes | arm cost → 0 where traces are not needed | exact reproduction shown in the pilot (150/150) |

Not attempted, because each could change numerics or routing behaviour and would need separate validation: CUDA graphs, `torch.compile`, fused NF4 kernels, batching.

## Per-problem cost model used for planning (lean mode, Balanced, today; ±20%)

| Arm type | Full generation (~250 steps) | Early stop (~105 steps) |
|---|---|---|
| FP16 static | ≈ 8.3 s | ≈ 3.9 s |
| NF4 static | ≈ 15.6 s | ≈ 6.5 s |
| B10 routed (≈90% NF4) | ≈ 15.2 s | ≈ 6.4 s |
| B30 routed | ≈ 13.8 s | ≈ 5.8 s |
| Both-branch (oracles, factorial cells; ≈102 ms/step incl. cache operations) | ≈ 25.7 s | ≈ 10.7 s |

Hours per arm = n × seconds / 3600. For example, a B10 early-stopped arm costs 1.65 h at n = 930 and 2.34 h at n = 1,319.
