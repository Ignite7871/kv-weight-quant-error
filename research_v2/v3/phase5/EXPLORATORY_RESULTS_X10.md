# EXPLORATORY results X10 — 2026-10-31

**Exploratory, descriptive and not blind** (EXPLORATORY_PLAN_V3.md, dated entry committed 7697aaf before this script ran; the point values had already been computed by hand). Source: `exploratory_v3_x10.json`. n = 200 per model.

R_pred = s^b̂ is a uniform-scaling benchmark: b̂ is C1's dose slope (both paths, isotropic), and NAT-P* sets each slot's error to ρ*·‖κ^F_P‖, so its per-slot effective scale is ρ*/ρ_slot, not s.

| Model | Path | s | b̂ (SE) | R_obs [95 % CI] | R_pred = s^b̂ [b̂ ± 1.96 SE] | E = ln R_obs − b̂ ln s [95 % CI] | R_obs / R_pred |
|---|---|---|---|---|---|---|---|
| qwen-base | K | 1.701 | 2.463 (0.039) | 95.421 [82.460, 110.330] | 3.697 [3.549, 3.852] | +3.251 [+3.105, +3.396] | 25.81 |
| qwen-base | V | 0.588 | 2.463 (0.039) | 0.315 [0.307, 0.322] | 0.270 [0.260, 0.282] | +0.151 [+0.127, +0.175] | 1.16 |
| llama-instruct | K | 1.812 | 2.199 (0.036) | 3.152 [3.078, 3.230] | 3.694 [3.545, 3.851] | -0.159 [-0.183, -0.134] | 0.85 |
| llama-instruct | V | 0.552 | 2.199 (0.036) | 0.344 [0.336, 0.352] | 0.271 [0.260, 0.282] | +0.240 [+0.217, +0.264] | 1.27 |
| qwen-instruct | K | 1.690 | 2.422 (0.048) | 73.179 [63.735, 84.079] | 3.563 [3.391, 3.742] | +3.022 [+2.884, +3.161] | 20.54 |
| qwen-instruct | V | 0.592 | 2.422 (0.048) | 0.340 [0.331, 0.351] | 0.281 [0.267, 0.295] | +0.193 [+0.164, +0.223] | 1.21 |
| llama-base | K | 1.877 | 2.198 (0.034) | 3.238 [3.157, 3.323] | 3.992 [3.830, 4.160] | -0.209 [-0.235, -0.183] | 0.81 |
| llama-base | V | 0.533 | 2.198 (0.034) | 0.292 [0.287, 0.297] | 0.251 [0.240, 0.261] | +0.152 [+0.135, +0.169] | 1.16 |
