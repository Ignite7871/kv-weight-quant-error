# v6b independent recomputation: tolerances (declared before the lock and before any v6b data)

Date: 2026-11-16. Part of the v6b lock. Follows v6 practice (`v6/RECOMPUTATION_TOLERANCES_V6.md`).

## Independence

- `recompute_v6b_independent.py` imports numpy, json, math, os and pathlib only. It imports no project module.
- It is written from FOLLOWUP_V6B §4–§5. It reads the raw v6b records, filling C1-V and NAT-V-cen from the v6 records where cut 1 skipped
  the rerun, re-checks every record's flags, and checks that no (model, arm, problem) appears twice.
- **Different methods from the locked analysis:**
  - per-problem ln m_i by an explicit float64 loop;
  - mean CIs: multinomial-count bootstrap (10,000), seeds 6262161 + offsets (locked: index resampling, seed 20261116 streams);
  - share CIs: multinomial-count bootstrap, seeds 6262162 + offsets;
  - its own verdict function, written from the §5 table.

## Tolerances (a quantity outside its tolerance is a disagreement, reported first)

| Quantity | Tolerance |
|---|---|
| n and failed-flag counts per model | identical |
| θ estimates (θ_V, θ_N, path parts and totals, interactions, re-norm effect) and share point estimates | \|Δ\| ≤ 1e-9 · max(1, \|x\|) |
| Bootstrap CI endpoints (θ and shares) | \|Δ\| ≤ 5 % of the locked CI width, or 1e-9 |
| The H\* verdict word | identical |

A verdict that differs is always reported as a disagreement. The report says whether a deciding CI endpoint lies within the tolerance of 0.5
(a Monte-Carlo borderline case). Such a disagreement is reported as **unresolved**; neither result is preferred.
