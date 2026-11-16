# v6 independent recomputation: tolerances (declared before any v6 test-phase analysis)

Date: 2026-11-15. Not part of the lock (00369b8d99cefd69): added post-lock as deviation V6-D2, at the corresponding author's
instruction, **before `analyze_v6.py` has been run and before any test-phase estimate, contrast, alignment value or verdict has been
computed or viewed**. It follows v4/v5 practice (`v4/RECOMPUTATION_TOLERANCES_V4.md`, `v5/RECOMPUTATION_TOLERANCES_V5.md`).

## Independence

- `recompute_v6_independent.py` imports numpy, json, math, hashlib, argparse and pathlib only. It imports no project module
  (`analyze_v6`, `analyze_r_v6`, `analyze_v3`, `analyze_x13`, `rv6`, `run_v6`, ...).
- It is written from PROTOCOL_V6 §2, §4, §5 and §7, reads the raw records and arrays, and re-checks every record's integrity flags and
  that no (model, arm, problem) appears twice.
- RoPE parameters (head dim, θ, llama3 scaling) are read from the pinned `config.json` of each model in the local Hugging Face cache, after
  checking its git-blob SHA-1 against `v4/V4_MODEL_MANIFEST.json`; the llama3 frequency rule and the Dirichlet factor
  A_p(n) = [sin(nω_p/2) / (n sin(ω_p/2))]² are re-implemented.
- The v4 equivalence margins are read from `v4_runs/v4_report.json`, whose SHA-256 is checked.
- **Different methods from the locked analysis:**
  - mean-contrast CIs: multinomial-count bootstrap (10,000), seed **6262061** (locked: index resampling, seed 20261115 streams);
  - sign-flip p: two-sided, **100,000** Monte-Carlo flips, seed 6262062 (locked: 10,000);
  - Spearman: own tie-averaged ranks; permutation p with **100,000** permutations, seed 6262063 (locked: 10,000); bootstrap CI of ρ by
    index resampling, 10,000, seed 6262064;
  - ratio shares (L3, L4): multinomial-count bootstrap, seed 6262065;
  - own Holm step-down over the same 21 p-values, own verdict rules written from the §7 table.

## Tolerances (a quantity outside its tolerance is a disagreement, reported first)

| Quantity | Tolerance |
|---|---|
| Deterministic quantities from the records and float64 arrays: n; θ estimates (incl. L3, L9a); L3 share; L5 median; L7(i) n·C^V; L7(iii) ρ ratio | \|Δ\| ≤ 1e-9 · max(1, \|x\|) |
| Deterministic quantities from the read-time arrays, which are **stored as float32** (meas, mod_V, lin_K, A_o, A_cov; found when checking dtypes on 2026-11-15, before any analysis): L1 medians, L2/L6 Spearman ρ, L4 estimates and share, L6 model ln ratio, L8a/L8b/L9b estimates. The recomputation works in float64 | \|Δ\| ≤ 1e-6 · max(1, \|x\|) |
| Bootstrap CI endpoints (95 % and 90 %; θ, shares, Spearman ρ) | \|Δ\| ≤ 5 % of the locked CI width, or 1e-9 |
| Sign-flip and permutation p | \|Δ\| ≤ 0.01 + 10 % of the locked value |
| Holm-adjusted p | \|Δ\| ≤ 0.01 + 10 % of the locked value |
| Decisions: every §7 verdict word (L1, L2, L5, L6, L7 with its parts i/ii/iii, L8a, L8b, L9a, L9b) | identical |

A decision that differs is always reported as a disagreement, with whether the deciding interval endpoint or p lies within the tolerance of
its threshold (a Monte-Carlo borderline case). Any verdict disagreement is reported as **unresolved**; neither result is preferred.

## Reading of the §7 rules used by the recomputation (written from the table; stated before any data)

- Positive-significant: 95 % CI excludes 0 on the predicted (positive) side and Holm-adjusted p < 0.05.
- L7(iii) "reversed": ρ_K/ρ_V < 1 **and** θ(ST-K, ST-V) with 95 % CI entirely < 0.
- L7(ii) disagreement: θ(ST-V@nat, C1-V) 95 % CI excludes 0 and |estimate| ≥ the model's H2 estimate in this run.
- Problems per model: those present in every arm file of that model (qwen2.5-0.5b: 100 problems x 13 arms; the other two: 70 x 10
  after V6-D1).
