# EXPLORATORY results X11 — 2026-10-31

**Exploratory and descriptive** (EXPLORATORY_PLAN_V3.md, dated entry committed 9706f11 before this script ran). Source: `exploratory_v3_x11.json`. B = Σ w E8/r ÷ Σ w (1−E8)/(d−r), with r = 8, d = 64 and w = tr(M)·Σ‖Δ_K‖² per (layer, KV head) from the frozen calibration state.

First order only, and the observed ratio is diluted by the V draw that QA and QO share. g(C1-V)/g(QO) is a dilution indicator, not a correction.

| Model | B (energy-weighted) | B_q (tr M) | B_1 (equal) | per-slot median [min, max] | observed e^θ(QA, QO) [95 % CI] | g(C1-V)/g(QO) [95 % CI] |
|---|---|---|---|---|---|---|
| qwen-base | 56.0 | 65.8 | 46.8 | 57.9 [12.5, 255.0] | 32.04 [29.46, 34.86] | 0.714 [0.669, 0.763] |
| llama-instruct | 43.3 | 44.3 | 43.9 | 46.6 [19.6, 99.0] | 12.90 [12.15, 13.73] | 0.749 [0.708, 0.794] |
| qwen-instruct | 53.7 | 64.5 | 45.7 | 54.6 [12.8, 261.5] | 23.62 [21.40, 26.16] | 0.690 [0.649, 0.732] |
| llama-base | 48.8 | 49.4 | 49.2 | 52.5 [20.3, 120.2] | 15.07 [14.21, 15.98] | 0.668 [0.628, 0.708] |

Checks (should be ~0): qwen-base 0.0e+00; llama-instruct 0.0e+00; qwen-instruct 0.0e+00; llama-base 0.0e+00
