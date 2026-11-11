# E1 — CPU validation and protocol-lock report

Written 2026-10-26/27. Branch `research-recovery`. Draft 3 and all archives untouched. **No E1 GPU work was done; the E1 GPU evaluation is not authorized to start and the driver refuses to run it** (verified: `gates`, `tierA` with `--authorized-gpu-run` stop with "e1_runs/AUTHORIZED_BY_AUTHORS.txt does not exist").

## 1. What was built (new files only; no locked Stage 1/2 file changed)
| File | Role |
|---|---|
| `E1_SPEC.md` | the locked protocol (hypotheses, definitions, samples, calibration, estimands, inference, equivalence-margin rule, gates, interpretation) |
| `rv2/controls.py` | the controlled step (NF4 pass → crop → FP16 pass → overwrite newest entry per arm), free-running and teacher-forced generators, float64 JS, seeded generators |
| `run_controls.py` | driver: `plan`, `cpu-smoke`, GPU `gates` / `calibrate` / `tierA` / `tierB` with the absolute 8.0 h ledger, watchdog, hardware stops, integrity checks and authorization guards |
| `analyze_controls.py` | calibration constants, Tier A/B estimands, bootstrap/permutation, TOST rule, Holm, sensitivity analyses, report |
| `tests/run_controls_tests.py`, `tests/mutation_check_controls.py`, `tests/test_analyze_controls.py` | validation |
| `e1_sample.json` | seeded sample: Tier B 400 test ids, Tier A 200 (subset), calibration 100 train ids, gate sets, exposed ids, SHA-256 of the FP16 reference records |
| `tools/lock_e1.py`, `E1_PROTOCOL_LOCK.txt` | lock/verify; lock commit 795756a, lock file sha256 `d56e9aa7…4d96`, committed as 9937ac3 |

## 2. How each methodological requirement is met
| Requirement | Where/how |
|---|---|
| Identical reference token histories in Tier A | every arm is teacher-forced on the same stored FP16 token sequence; test c07 checks the fed tokens per arm and that step-1 JS is exactly 0 for all arms; the reference distributions come from an in-session pure-FP16 pass; test c08 checks teacher-forced FP16 equals free-running FP16 |
| Newly written vs accumulated entries | spec §2: perturb only the newly written entry at write time; accumulated entries never revisited; prefill never perturbed. Tests c09 (earlier entries bitwise unchanged after each step; prefill unchanged) and c10; mutants "revisit previous entry" and "perturb prefill" are caught |
| Match per-layer, per-head, K/V, position magnitudes | slot = (layer, K/V, KV head) at the new position, matched separately at every step; tests c03 (norms within 3 % on synthetic), c11 and real-weight CPU run (mean deviation 0.05 %, max 2.3 % judged against max(target, 2^-10 of entry norm)); gate G2 |
| Separate direction from magnitude | C1 (magnitude kept, direction random), C2 (per-coordinate magnitude kept, sign random), dose arms (magnitude varied), C1\* (effect-matched magnitude); tests c04, c05, c12 |
| No test-outcome leakage in random/sign/calibration quantities | seeds are SHA-256 of labels only (c06); no calibration-derived direction is used (C3 bank dropped); calibration scalars come from train problems only and are hashed into the lock before any test evaluation (`lock_e1.py --append-calibration`) |
| FP16, NAT, C1, C2 included; 0.5×/2× dose arms included | Tier A has all, plus C1\*; Tier B has FP16, NAT, C1, C2 (cost 7.3 h within the 8 h ceiling; pre-declared reduction order in spec §8) |
| Zero perturbation reproduces FP16 bitwise | test c02 (tiny model, tokens and cache) and real Qwen2.5-0.5B on CPU; GPU gate G1 repeats it on the real kernels |
| Do not assume equivalence from norm matching | dose–response arms, effect-matched C1\*, early-step (2–8) analysis, equivalence claimable only if the measured dose slope ≥ 0.25 |
| Avoid incompatible cache histories | each arm's cache is built only from its own writes on the shared token history; no entry from another trajectory or problem is ever inserted (c09/c10; spec §2) |
| Predeclare estimands, paired tests, clustering, multiplicity, sensitivity | spec §5–6, implemented in `analyze_controls.py`, tested on simulated truth (t01–t10: Holm against a hand calculation, recovery of a known effect, 150-simulation type-I check, equivalence logic, calibration recovery, bootstrap vs independent normal approximation) |
| Justify the equivalence margin substantively | spec §7: margin = |b̂|·ln(1 + c), c = measured natural step-to-step relative variability of the NF4 perturbation's own size (clamped 10–40 %), converted through the measured dose–response; withdrawn: ±5 tokens; claimable only if the dose-response is informative |
| Report positive, null, inconclusive outcomes | spec §10 and the report generator |

## 3. Validation results (`validation_logs/e1_cpu_validation.json`, committed and locked)
- Unit suite: **13/13 pass on the tiny CPU model plus a real-model CPU smoke (14/14 with `--real`)**. 
- Mutation check: **15/15 injected bugs caught** (norm ×1.1, shared noise across slots, no sign randomization, direction = Δ, α ignored, NAT stores FP16 entry, wrong-layer write, keys-only write, constant seed, revisiting an earlier entry, perturbing the prefill, the "NF4" pass run in FP16, token taken from the NF4 logits, off-by-one reference history, JS replaced by KL). The suite found no failure on the unmutated code.
- Analysis tests: **10/10**.
- End-to-end I/O smoke on real Qwen2.5-0.5B (CPU, few tokens): Tier A records pass the integrity checks, Tier B records written for FP16/NAT/C1/C2.
- Two defects found and fixed *by* these checks before the lock: (i) the historical float32 JS has a ±3×10⁻⁸ noise floor on a 150k vocabulary, so identical logits did not give 0 — replaced by a float64 JS (`js64`, tested against the textbook formula); (ii) the per-slot norm check was dominated by slots whose perturbation is below fp16 resolution — the deviation is now judged against max(target, 2⁻¹⁰·entry norm) (spec G2).
- Measured on real weights (CPU): the NF4 difference is ≈ 15 % of the entry norm (‖Δ‖/‖κ^F‖ ≈ 0.148), so the controls operate in a physically large regime.

## 4. What the CPU checks do not show
CPU NF4 numerics differ from CUDA, so the GPU gates G1–G6 must still run; the GPU-only paths of `run_controls.py` (`gates`, `block_loop`, the ledger interplay) have been exercised only through their shared functions and the CPU smoke, not on a GPU; per-step overhead of the perturbation code on CUDA is estimated, not measured (gate G6 measures it).

## 5. Cost estimate (replaces "≈ 6 h")
| Block | Estimate |
|---|---|
| Gates (G1–G6) | 0.3 h |
| Calibration (100 train problems × NAT + 4 C1 doses, 64 steps) | 0.9 h |
| Tier A (200 problems × 6 arms) | 2.0 h |
| Tier B (400 problems × FP16, NAT, C1, C2) | 4.1 h |
| **Total** | **≈ 7.3 h of the 8.0 h ceiling** |
The increase over the earlier ≈ 6 h comes from calibration (required for the margin and C1\*), C2 in Tier B and C1\*. If gate G6 projects > 8 h the spec's pre-declared reduction order applies (drop Tier B C2; Tier B n 400 → 300; drop C1\*), which brings it to ≈ 5.0–6.0 h.

## 6. Authorization status
Locked, not launched. To start E1 the authors create `research_v2/e1_runs/AUTHORIZED_BY_AUTHORS.txt`; the sequence is then `gates` → `calibrate` → `analyze_controls.py calibrate` → `lock_e1.py --append-calibration` → `tierA` → `tierB` → `analyze_controls.py analyze`, each refusing to start unless the previous step passed.
