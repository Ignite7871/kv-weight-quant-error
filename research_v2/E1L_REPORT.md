# Llama E1 Tier A replication — report (Llama-3.2-1B-Instruct vs the locked Qwen2.5-0.5B E1)

Run 2026-10-28 on the local RTX 4060 under `E1L_SPEC.md`. Authorization: the corresponding author's file `e1l_runs/AUTHORIZED_BY_AUTHORS_LLAMA.txt` (kept unchanged) and the message approving the test-only amendment. Draft 3, the Qwen E1 lock and all Qwen raw records are unchanged (both locks verified before and after). No Tier B, no other model or dataset, no cloud, no push or submission. Raw records: `e1l_runs/{calib,tierA}/`; machine-readable: `e1l_runs/e1l_report.json`, `qwen_llama_comparison.md`. **Labels:** *confirmatory-design* = locked primary/secondary hypotheses; *exploratory* = sensitivity and descriptive tables. All intervals are 95 % problem-clustered percentile bootstrap unless stated.

## 1. Locks, gates, incident, amendment
| | |
|---|---|
| Original Llama lock | sha256 `97bb7f1171b1e11f…` (commit 2eadb86), preserved at `validation_logs/E1L_PROTOCOL_LOCK_original_97bb7f11.txt` |
| Amendment 1 (test-only, approved) | after all gates G1–G7 passed, the GPU mutation check failed because test `l02` created its marker tensor on the CPU while the cache is on CUDA (5.6 GPU-s, run stopped as instructed). Fix: one line (`device=cache.layers[0].keys.device`). `git diff`: 1 insertion, 1 deletion, the only tracked-file change. The lock was re-created; the new lock lists the same 33 files with identical hashes except `tests/run_e1l_tests.py` (checked programmatically); an AMENDMENT line records the reason and both hashes. Qwen E1 lock verifies unchanged |
| Calibration hash | `04906d31f8b9f02a…` appended to the lock before any test problem; the final lock file sha256 is `b2c99ec5…b554` |
| Re-validation after the fix | CPU: 17/17 tests (18/18 with the real-Llama smoke), 19/19 mutants, 10/10 analyzer tests (`validation_logs/e1l_cpu_revalidation_after_l02_fix.json`) |
**GPU gates after the amendment (all passed):** G1 α = 0 bitwise 20/20; G2/G3 no integrity errors; G4 same seed identical, different seeds differ 6/6; G5 NAT equals the engine's cache-only generation 8/8; G6 projection 1.89 h ≤ 2.5 h; G7 live cache = 16 layers, 8 KV heads, head_dim 64, all stored tensors FP16; GPU mutation/correctness check **19/19 mutants caught** on the CUDA kernels. The first (pre-amendment) gate pass is kept as `e1l_runs/gates/gates_first_run_pre_amendment.json`.
**Calibration (100 train problems only):** α\* = 0.958, dose slope b̂ = 2.104 (informative, ≥ 0.25), c = 0.10 (raw 0.064, clamped), Tier A margin 0.2005 log-JS (≈ 22 %).

## 2. Results — Tier A (confirmatory-design), 200 test problems, teacher-forced, identical token history, mean 56.7 steps
Per-problem mean JS to the pure-FP16 reference (descriptive): **NAT 0.00263**, C1 0.00254, **C2 0.00210**, C1(0.5) 0.00064, C1(2) 0.01162, C1\* (α = 0.958) 0.00228. Top-1 flip rates: NAT 2.56 %, C1 2.51 %, C2 2.38 %, C1(0.5) 1.29 %, C1(2) 5.79 %.
**Primary (H_struct): θ_A = log JS(NAT) − log JS(C1) = +0.058 (ratio 1.060), 95 % CI [−0.007, +0.121], p = 0.078; 90 % CI [+0.003, +0.110] lies inside ±0.2005 and b̂ ≥ 0.25 ⇒ the locked rule's verdict is "equivalent".** This is **not** a replication of Qwen's difference (θ_A = −0.253). Reading it correctly: the interval is compatible with NAT being between 0.7 % milder and 13 % *more* disruptive than same-norm isotropic noise; "equivalence" holds only at the margin the calibration gives (≈ ±22 %), which is as wide as Qwen's whole effect, so it is a coarse equivalence, and it is **not** inferred from non-significance.
| Secondary (Holm over 5) | estimate (log JS) | 95 % CI | raw p | Holm p |
|---|---|---|---|---|
| NAT − C2 (sign-randomized Δ) | **+0.266** | [+0.195, +0.336] | 1×10⁻⁴ | 5×10⁻⁴ |
| C2 − C1 | **−0.209** | [−0.274, −0.142] | 1×10⁻⁴ | 5×10⁻⁴ |
| Dose slope of C1 (α = 0.5, 1, 2) | **+2.148** | [+2.098, +2.198] | 1×10⁻⁴ | 5×10⁻⁴ |
| Top-1 flip rate NAT − C1 | +0.0005 | [−0.0027, +0.0038] | 0.76 | 0.76 |
| NAT − C1\* (effect-matched) | **+0.168** | [+0.101, +0.235] | 1×10⁻⁴ | 5×10⁻⁴ |
Reading: JS scales as ≈ dose^2.1; **NAT is clearly more disruptive than sign-randomized Δ (≈ +30 %)** while isotropic noise of the same norm is ≈ equal to NAT, so on Llama the *sign/direction structure of Δ matters* and the per-coordinate magnitude profile alone (C2) understates NAT; **the train-calibrated effect-matched control did not transfer** (NAT − C1\* = +0.168, interval excludes 0), i.e. the calibration (steps 2–16 on train problems) under-predicts NAT's effect over steps 1–64 on test problems — pre-declared criterion "calibration performance" is **not met**.
Diagnostics: realized norm deviations (C1: mean 0.025 %, max 0.66 %; C2: 0.009 %, 0.77 %; dose arms ≤ 1.2 %) — well inside the gate; ‖Δ‖/‖κ^F‖ = 13.0 % in every arm (Qwen 14.3 %).

## 3. Exploratory sensitivity (NAT − C1, log JS; none replaces the primary)
steps 1–32 +0.117 [+0.050, +0.183]; steps 9–64 +0.052 [−0.016, +0.118]; per-problem median +0.055 [+0.006, +0.104]; problems with ≥ 32 steps +0.044 [−0.023, +0.109]; relative-difference scale +0.169 [+0.099, +0.242]; excluding the 4 exposed test problems +0.058 [−0.008, +0.122]; pooled steps, problem-clustered +0.040 [+0.011, +0.068]; steps 2–8 +0.073 [−0.033, +0.181]. All point estimates are positive, small, and mostly unresolved — NAT is neither milder than isotropic noise (as on Qwen) nor clearly harsher.

## 4. Qwen / Llama side by side (full table: `e1l_runs/qwen_llama_comparison.md`)
| Quantity | Qwen2.5-0.5B (E1) | Llama-3.2-1B-Instruct (E1L) | Replication category (pre-declared) |
|---|---|---|---|
| Primary θ_A (NAT − C1) | **−0.253** [−0.325, −0.179] | **+0.058** [−0.007, +0.121] | **Not replicated** (locked rule: "equivalent to C1"); between-model difference +0.311 [+0.214, +0.407] (descriptive) |
| NAT − C2 | +0.003 [−0.083, +0.086] (≈ equal) | **+0.266** [+0.195, +0.336] | **Different** |
| C2 − C1 | −0.256 | −0.209 | **Replicated** (both negative, resolved) |
| Dose slope | +2.507 | +2.148 | **Replicated** (steep, same sign) |
| NAT − C1\* (effect-matched) | −0.041 [−0.112, +0.030] | +0.168 [+0.101, +0.235] | **Calibration not transferred on Llama** |
| Top-1 flip NAT − C1 | −0.004 (Holm 0.14) | +0.0005 (n.s.) | n/a |
| Calibrated margin / b̂ / α\* | 0.237 / 2.49 / 0.910 | 0.200 / 2.10 / 0.958 | – |
| ‖Δ‖/‖κ^F‖ | 14.3 % | 13.0 % | similar |
**Architecture/model-specific deviations (not attributable to a single cause):** Qwen2.5-0.5B (base, 24 layers, 2 KV heads, 896 hidden) vs Llama-3.2-1B-Instruct (instruction-tuned, 16 layers, 8 KV heads, 2048 hidden, 128k vocabulary) differ in family, tuning, depth, KV-head count and scale at once. The Qwen finding "NF4-written entries are milder than isotropic noise, explained by the per-coordinate magnitude profile" is therefore **Qwen-specific in this evidence**; what generalizes across both models is (i) a steep dose–response (JS ∝ dose^2.1–2.5), (ii) that preserving Δ's per-coordinate magnitudes with random signs is less disruptive than isotropic noise of the same norm (C2 < C1), and (iii) that the NF4 cache difference is a sizeable perturbation (13–14 % of the entry norm).

## 5. Runtime and hardware (measured)
| Block | GPU-hours |
|---|---|
| Gates (first pass) + failed GPU mutation check | 0.118 + 0.0015 |
| Gates (after amendment) | 0.120 |
| GPU mutation check (19/19) | 0.022 |
| Calibration (100 problems × 5 arms) | 0.508 |
| Tier A (200 × 6 arms + in-session FP16 reference) | 1.254 |
| **Total** | **2.023 h of the 2.5 h ceiling** (no truncation, no integrity or hardware stop) |
Per arm-problem 3.3 s (64 teacher-forced steps ≈ 58 ms/step). Maximum GPU temperature 72 °C (stop 88 °C).

## 6. Conclusions justified
- **Positive:** a strong, reproducible dose–response; and C2 < C1 on both families.
- **Negative / not replicated:** NAT is *not* milder than isotropic noise on Llama (ratio 1.06, CI includes 1); the Qwen "structure beyond size" statement (NAT ≈ C2 < C1) does not hold on Llama, where NAT ≈ C1 > C2.
- **Inconclusive:** the early-step (2–8) contrast on both models; top-1 flip contrast.
- **Limits:** two models, one benchmark, first ≤ 64 teacher-forced steps, isotropic Gaussian as the only random-direction baseline; the effect-matched calibration is itself model-dependent; the models differ in several respects at once.
