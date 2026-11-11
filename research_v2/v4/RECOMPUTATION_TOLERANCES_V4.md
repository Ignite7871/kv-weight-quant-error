# v4 independent recomputation: tolerances (declared before any v4 record exists)

Date: 2026-11-03, Phase 3, before the lock. PROTOCOL_V4 §8 requires an independent recomputation with tolerances declared before it runs.
These follow v3's `v3/phase5/RECOMPUTATION_TOLERANCES.md`.

## Independence

- `recompute_v4_independent.py` imports numpy, json, math, itertools and pathlib only. It imports no `analyze_v4`, `analyze_v3`, `analyze_x13`,
  `rv4` or any other project module.
- It is written from the protocol text (§3, §6, §7, §9), reads the raw records, and re-checks every record's integrity flags.
- **Different methods:**
  - multinomial-count bootstrap (10,000) with seed **4242041** (locked: index resampling, seed 20261105 streams);
  - sign-flip p with **100,000** Monte-Carlo flips (locked: 10,000);
  - its own Holm step-down, verdict rule, readings, H4 enumeration and Spearman;
  - its own X13a common-mode share and isotropic null (200 draws, seed 4242042; locked: the X13 functions with their own streams).

## Tolerances (a quantity outside its tolerance is a disagreement, reported first)

| Quantity | Tolerance |
|---|---|
| Deterministic quantities: θ point estimates, n, the H4 statistic T and its exact permutation p, the per-model θ̂ entering H4 | \|Δ\| ≤ 1e-9 · max(1, \|x\|) |
| Bootstrap CI endpoints (95 % and 90 %) of θ | \|Δ\| ≤ 5 % of the locked CI width, or 1e-9 |
| Sign-flip p and Holm-adjusted p | \|Δ\| ≤ 0.01 + 10 % of the locked value |
| X13a L^P (the point estimate contains the Monte-Carlo null ν, 200 draws) | \|Δ\| ≤ 0.02 + 5 % of the locked CI width |
| X13a CI endpoints | \|Δ\| ≤ 0.02 + 10 % of the locked CI width |
| Decisions: verdict word, the H1–H3 and H5 per-model readings, the universal result of each family, the H4 result, with and without the BF16-flagged models | identical |

A decision that differs is always reported as a disagreement, with whether the deciding interval endpoint or p lies within the tolerance of its
threshold (a Monte-Carlo borderline case). Any verdict disagreement is reported as **unresolved** (§8).
