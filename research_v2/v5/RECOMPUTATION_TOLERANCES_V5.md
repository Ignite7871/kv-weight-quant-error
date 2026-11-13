# v5 independent recomputation: tolerances (declared before any v5 record exists)

Date: 2026-11-11, Phase 3, before the lock. PROTOCOL_V5 §8 requires an independent recomputation with tolerances declared and locked before
any data. These follow v4's `v4/RECOMPUTATION_TOLERANCES_V4.md`, extended to the A-X classification.

## Independence

- `recompute_v5_independent.py` imports numpy, json, math, hashlib and pathlib only. It imports no `analyze_v5`, `analyze_v4`, `analyze_v3`,
  `analyze_x13`, `rv5` or any other project module.
- It is written from the protocol text (§3, §6, §7), reads the raw records, and re-checks every record's integrity flags.
- **Different methods:**
  - multinomial-count bootstrap (10,000) with seed **5252051** (locked: index resampling, seed 20261111 streams);
  - two-sided sign-flip p with **100,000** Monte-Carlo flips (locked: 10,000);
  - its own Holm step-down, verdict rule, readings and family results;
  - its own X13a common-mode share and isotropic null (200 draws, seed 5252052);
  - its own A-X references (the named source models' estimates in the locked v4 report, whose SHA-256 it checks), boundary, one-sided
    sign-flip p (100,000 flips, seed 5252053), Holm over the two A models per contrast, classification and joint reading.

## Tolerances (a quantity outside its tolerance is a disagreement, reported first)

| Quantity | Tolerance |
|---|---|
| Deterministic quantities: θ point estimates, n | \|Δ\| ≤ 1e-9 · max(1, \|x\|) |
| A-X boundary b (a mean of locked v4 estimates) | \|Δ\| ≤ 1e-12 |
| Bootstrap CI endpoints (95 % and 90 %) of θ | \|Δ\| ≤ 5 % of the locked CI width, or 1e-9 |
| Sign-flip p (two-sided, and the A-X one-sided p) | \|Δ\| ≤ 0.01 + 10 % of the locked value |
| X13a L^P (contains the Monte-Carlo null ν, 200 draws) | \|Δ\| ≤ 0.02 + 5 % of the locked CI width |
| X13a CI endpoints | \|Δ\| ≤ 0.02 + 10 % of the locked CI width |
| Decisions: verdict word; every per-unit reading of A-H1/H2/H3/H5, B-H2/H3/H5, D-H2/H5; each family result; each A-X classification and the joint NAT−C1 reading; all with and without BF16-flagged units | identical |

A decision that differs is always reported as a disagreement, with whether the deciding interval endpoint or p lies within the tolerance of
its threshold (a Monte-Carlo borderline case). Any verdict, reading or classification disagreement is reported as **unresolved** (§8).
