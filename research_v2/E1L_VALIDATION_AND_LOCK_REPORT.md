# Llama E1 Tier A — validation report, protocol lock and coauthor review packet

Written 2026-10-28. Branch `research-recovery`. **No Llama-E1 GPU work has been done.** The driver refuses to run until the corresponding author creates the separate authorization file (below). Qwen E1 lock, all Qwen raw records, Draft 3 and the archives are unchanged (both locks verify).

## 1. Lock
| | |
|---|---|
| Spec | `E1L_SPEC.md` (locked) |
| Lock file | `E1L_PROTOCOL_LOCK.txt`, committed 2eadb86; **lock-file sha256 `97bb7f1171b1e11f4d6e7e2520f609ec0e448d69f70f210d9816c24b3fa2cd38`**; locked files committed at 7277cf2 |
| Pins the Qwen protocol | the lock records the sha256 of `E1_PROTOCOL_LOCK.txt` (`de876ab9…50c1`) and hashes every locked Qwen E1 module the replication imports unchanged (`run_controls.py`, `analyze_controls.py`, `rv2/*`, `E1_SPEC.md`, tests) |
| Verify | `venv/Scripts/python.exe research_v2/tools/lock_e1l.py --verify` and `tools/lock_e1.py --verify` (both pass now) |
| Frozen | problem ids (200 test + 100 train calibration, copied from the Qwen E1 sample; sha256 of that source recorded), arm definitions, dose levels 0.5/1/2 and α\* grid 0.5–4, seeds (labels `Lcalib`/`LA`/`Lgate`), estimands, tests, Holm family (5), margin rule, gates, decision categories |
| **Authorization file** | `research_v2/e1l_runs/AUTHORIZED_BY_AUTHORS_LLAMA.txt` — must exist, be non-empty and contain the word **Llama** (a copy of the Qwen file will not work). I have not created it and will not. Suggested content: one sentence confirming that you reviewed `E1L_SPEC.md` and the lock and authorize the Llama E1 Tier A run under the 2.5 h ceiling |
| GPU budget | absolute 2.5 h cumulative (gates, GPU mutation check, calibration, Tier A), own ledger and watchdog; truncation, never extension |

## 2. Validation (CPU; `validation_logs/e1l_cpu_validation.json`, part of the lock)
- **Cache layout / FP16 storage / alignment / GQA (item 2):** tests l01–l04 on a tiny random GQA Llama (8 query / 2 KV heads, head_dim 64): the live cache has the expected layers/KV-heads/head_dim, every stored K and V tensor is `float16` and finite after every arm; layer–K/V–head slot stacking and write-back verified with unique markers (l02); per-KV-head norm matching recomputed independently from cache entries (l03). On the **real Llama-3.2-1B-Instruct (CPU)**: 16 layers, 8 KV heads, head_dim 64, all stored tensors FP16.
- **Zero perturbation (item 3):** α = 0 reproduces pure FP16 bitwise (tokens and cache) on the tiny Llama and on the real Llama; NAT equals the engine's cache-only bitwise on both.
- **Norms, seeds, causal histories, precision (item 4):** the full Qwen control suite c01–c13 re-run on Llama geometry (determinism, seed independence, identical teacher-forced history with step-1 JS exactly 0, earlier entries/prefill never modified, only the newest entry perturbed, float64 JS); real-Llama norm deviations (CPU): C1 mean 0.02 %/max 0.4 %, C2 0.01 %/0.08 %; the NF4 difference is **14.8 % of the entry norm** (Qwen: 14.3–14.6 %).
- **Mutation tests (item 5):** **19/19** injected bugs caught (15 Qwen mutants + KV swapped in write-back, head index rolled, stored values scaled by 2⁻⁹, norm pooled across KV heads); the unmutated suite passes 17/17 (18/18 with the real-Llama smoke).
- **Analysis code:** the locked analyzer's 10 simulated-truth tests pass unchanged.
- **GPU-only code paths:** a CPU harness with the real Llama (`tests/cpu_harness_e1l_paths.py`) exercised the gates function (G1, G2/G3, G4, G5, G6, G7), the calibration step, the Tier A step with the in-session FP16 reference, and the record writers without error. This caught no bug (the Qwen-run wrapper bug had been fixed in the code it reuses), but CPU timing is meaningless for G6.
- **Not yet done (needs the GPU, hence the authorization):** GPU gates G1–G7 on the real kernels; the **GPU mutation/correctness check** on the CUDA kernels (`tools/e1l_gpu_mutation_check.py`, ≈ 1–2 min); calibration; Tier A.

## 3. Projected cost (item 8)
From the feasibility pilot rates (Llama hybrid 53 ms/step, FP16 22 ms/step) and the Qwen E1 finding that a controlled teacher-forced arm costs the same as a hybrid step: per test problem ≈ FP16 reference generation 1.4 s + reference logits 1.3 s + 6 arms × 3.3 s ≈ **23 s**; per calibration problem ≈ 19 s. Calibration ≈ 0.53 h, Tier A ≈ 1.28 h, gates + mutation check ≈ 0.17 h, process loads ≈ 0.1 h ⇒ **≈ 2.1 h of the 2.5 h ceiling** (≈ 16 % margin; Qwen's realized/estimated ratio was 0.9). Gate G6 re-projects from measured times and **stops the run before calibration if the total would exceed 2.5 h**, with no scope reduction.

## 4. What to review (coauthor packet; share privately)
1. `E1L_SPEC.md` — §2 (the only differences from Qwen) and §3 (pre-declared replication categories; "nonsignificant ≠ equivalent").
2. `E1L_PROTOCOL_LOCK.txt` and the hash above; `validation_logs/e1l_cpu_validation.json`.
3. `e1l_sample.json` — the 200 + 100 ids (identical to the Qwen sample).
4. The sequence in spec §6, each step refusing unless the previous one passed.
After the authorization file exists the run sequence is: `gates` → `tools/e1l_gpu_mutation_check.py --authorized-gpu-run` → `calibrate` → `analyze_e1l.py calibrate` → `lock_e1l.py --append-calibration` → `tierA` → `analyze_e1l.py analyze` → `followup/compare_qwen_llama.py`.

## 5. Companion deliverable already done: Qwen Tier B norm-tolerance assessment
See `E1_TIERB_NORM_TOLERANCE_ASSESSMENT.md`: 2 of 43,758 C1 steps (0.0046 %) and 2 of 400 problems exceed 10 % (max 10.5 %); C2 never; excluding the 2 problems moves NAT − C1 from +3.54 to +3.39 tokens (interval [+0.93, +5.84]); conclusions insensitive; no Qwen rerun.
