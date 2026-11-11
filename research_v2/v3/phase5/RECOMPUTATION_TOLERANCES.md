# Phase 5: independent recomputation — tolerances (declared before the recomputation is run)

Date: 2026-10-31. Branch `v3-novelty`, lock 7325f563ce2978d8.

**Status when this was written.**
- The locked analysis (`analyze_v3.py analyze`, exactly as locked) has been run: `v3_runs/v3_report.json`,
  sha256 f9ce364a8392bd42a279041a122ca1951c3f4645523b8af7f1dd6123d8f6078b.
- Its results have not been read. The only exception is the last two lines of its console output, which were shown when
  the run's exit status was checked: the llama-base exploratory truncation-sensitivity lines.
- The independent recomputation (`recompute_v3_independent.py`) has not been run.

PROTOCOL_V3 states no recomputation tolerance (PHASE3_REPORT only says "then the independent recomputation"). The
tolerances below follow the precedent of the earlier audit (`paper/audit/VERIFIED_STATS.md`).

## Independence

- **Imports.** No import of `analyze_v3`, `run_v3`, `rv3` or any project module; numpy, json and pathlib only.
- **Written from the protocol.** It is written from the PROTOCOL_V3 text (§5–§8, §16), not adapted from the locked code. It reads the raw records
  directly and re-checks their integrity flags.
- **Different methods:**
  - multinomial-count bootstrap (10,000 resamples) with seed **4242031** (locked: index resampling, seed 20261103);
  - sign-flip p with **100,000** Monte-Carlo flips (locked: 10,000);
  - its own tie-averaged ranking, Holm step-down and OLS;
  - the H-C point estimates are refit by least squares on the step-level data (locked: sufficient statistics);
  - the exploratory truncation sensitivity uses seed 4242032.

## Tolerances (a quantity outside its tolerance is a disagreement and is reported first)

| Quantity | Tolerance |
|---|---|
| Deterministic quantities: point estimates (D_m, θ, β₀, β₁, S_m, f_m, g₀, g₁, F, H-F contrasts, pooled H-A), n, excluded/kept/dropped counts, per-model SDs, realized-deviation mean/max/p99 bin edge/step counts, median retained fraction, sensitivity estimates | \|Δ\| ≤ 1e-9 · max(1, \|x\|) |
| Bootstrap CI endpoints (95 % and 90 %; different seed and method) | \|Δ\| ≤ 5 % of the locked CI width (≈ 5 Monte-Carlo SE for B = 10,000), or 1e-9 |
| p-values (sign-flip permutation; H-C bootstrap p) and Holm-adjusted p | \|Δ\| ≤ 0.01 + 10 % of the locked value |
| Decisions: verdict word, H-C bands, preconditions/applicability, "supported"/"falsified", Holm-significant at 0.05, H-D/H-E readings, the H-D extrapolation rule, the H-F 25 % guard | identical |

A decision that differs is always reported as a disagreement. The report adds whether the deciding interval endpoint
or p lies within the tolerance of its threshold, i.e. whether the difference is a Monte-Carlo borderline case.

Quantities that only the recomputation computes are labelled "recomputation only" and are not compared. Example: the
E1L §3 replication category of NAT − C1, which the locked code does not implement.
