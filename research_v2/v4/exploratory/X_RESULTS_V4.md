# v4 exploratory results X1-X4 (EXPLORATORY, descriptive only, not blind)

Plan: `v4/EXPLORATORY_PLAN_V4.md` (commit 1409b94, written 2026-11-09 after the locked results were read). Seed 20261109, B = 10,000 problem resamples per model. Per-model means reproduce the locked estimates (max |dev| = 0.0e+00). No p-values; nothing here changes a locked verdict.

## Sizes

| Model | params (16-bit, stored) | log10 P | head_dim |
|---|---|---|---|
| qwen2.5-0.5b | 0.494 B | 8.694 | 64 |
| qwen2.5-1.5b | 1.544 B | 9.189 | 128 |
| qwen2.5-3b | 3.086 B | 9.489 | 128 |
| qwen2.5-7b | 7.616 B | 9.882 | 128 |
| llama3.2-1b | 1.236 B | 9.092 | 64 |
| llama3.2-3b | 3.213 B | 9.507 | 128 |
| llama3.1-8b | 8.030 B | 9.905 | 128 |
| mistral-7b-v0.3 | 7.248 B | 9.860 | 128 |

## X1: slope of estimate per decade of parameters (within family)

| Family | contrast | slope / decade | 95 % bootstrap CI (problems only) | Pearson(x, y) |
|---|---|---|---|---|
| qwen (4 models) | H1 | +0.504 | [+0.381, +0.631] | +0.994 |
| qwen (4 models) | magnitude of H1 | -0.504 | [-0.631, -0.381] | -0.994 |
| qwen (4 models) | NAT-C1 | +0.519 | [+0.408, +0.630] | +0.952 |
| llama (3 models) | H1 | +0.638 | [+0.477, +0.795] | +0.875 |
| llama (3 models) | magnitude of H1 | -0.638 | [-0.795, -0.477] | -0.875 |
| llama (3 models) | NAT-C1 | +0.629 | [+0.471, +0.790] | +0.954 |

The CI covers problem sampling given these checkpoints only, not a population of models (Qwen 2 df, Llama 1 df).

## X2: per-layer C^P (from the locked report; summary only, no CI, not head_dim-adjusted)

| Model | path | L | mean | median | min | max | argmax layer (rel. depth) | layer 0 | mean layers 1+ | Spearman vs depth |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b | K | 24 | 0.219 | 0.217 | 0.149 | 0.319 | 0 (0.00) | 0.319 | 0.214 | +0.06 |
| qwen2.5-0.5b | V | 24 | 0.210 | 0.208 | 0.147 | 0.405 | 0 (0.00) | 0.405 | 0.202 | -0.45 |
| qwen2.5-1.5b | K | 28 | 0.214 | 0.209 | 0.152 | 0.322 | 0 (0.00) | 0.322 | 0.210 | -0.45 |
| qwen2.5-1.5b | V | 28 | 0.194 | 0.194 | 0.130 | 0.346 | 0 (0.00) | 0.346 | 0.188 | -0.64 |
| qwen2.5-3b | K | 36 | 0.230 | 0.228 | 0.173 | 0.308 | 13 (0.37) | 0.238 | 0.230 | -0.54 |
| qwen2.5-3b | V | 36 | 0.214 | 0.205 | 0.135 | 0.303 | 13 (0.37) | 0.297 | 0.211 | -0.70 |
| qwen2.5-7b | K | 28 | 0.206 | 0.207 | 0.137 | 0.272 | 1 (0.04) | 0.141 | 0.209 | -0.54 |
| qwen2.5-7b | V | 28 | 0.189 | 0.178 | 0.103 | 0.310 | 1 (0.04) | 0.163 | 0.190 | -0.65 |
| llama3.2-1b | K | 16 | 0.202 | 0.204 | 0.158 | 0.237 | 4 (0.27) | 0.204 | 0.202 | +0.58 |
| llama3.2-1b | V | 16 | 0.205 | 0.192 | 0.156 | 0.258 | 14 (0.93) | 0.192 | 0.205 | +0.81 |
| llama3.2-3b | K | 28 | 0.179 | 0.178 | 0.132 | 0.213 | 26 (0.96) | 0.213 | 0.178 | +0.28 |
| llama3.2-3b | V | 28 | 0.197 | 0.199 | 0.149 | 0.262 | 1 (0.04) | 0.233 | 0.196 | +0.46 |
| llama3.1-8b | K | 32 | 0.182 | 0.181 | 0.146 | 0.270 | 1 (0.03) | 0.246 | 0.180 | -0.12 |
| llama3.1-8b | V | 32 | 0.196 | 0.188 | 0.152 | 0.381 | 1 (0.03) | 0.305 | 0.192 | +0.23 |
| mistral-7b-v0.3 | K | 32 | 0.267 | 0.232 | 0.202 | 0.585 | 0 (0.00) | 0.585 | 0.257 | -0.64 |
| mistral-7b-v0.3 | V | 32 | 0.274 | 0.224 | 0.187 | 0.769 | 0 (0.00) | 0.769 | 0.259 | -0.71 |

| Model | fraction of layers with C^V > C^K |
|---|---|
| qwen2.5-0.5b | 0.50 |
| qwen2.5-1.5b | 0.29 |
| qwen2.5-3b | 0.19 |
| qwen2.5-7b | 0.29 |
| llama3.2-1b | 0.56 |
| llama3.2-3b | 0.82 |
| llama3.1-8b | 0.69 |
| mistral-7b-v0.3 | 0.31 |

## X3: H1 against NAT-C1

Across models (point values, no CI): all 8 Pearson +0.959, Spearman +0.976; 7 without Mistral Pearson +0.931, Spearman +0.964.

| Model | within-model problem-level Pearson(H1_i, NAT-C1_i) | 95 % bootstrap CI |
|---|---|---|
| qwen2.5-0.5b | -0.054 | [-0.228, +0.156] |
| qwen2.5-1.5b | +0.040 | [-0.136, +0.221] |
| qwen2.5-3b | +0.016 | [-0.178, +0.211] |
| qwen2.5-7b | +0.354 | [+0.088, +0.563] |
| llama3.2-1b | +0.021 | [-0.141, +0.181] |
| llama3.2-3b | +0.433 | [+0.176, +0.605] |
| llama3.1-8b | +0.262 | [+0.059, +0.429] |
| mistral-7b-v0.3 | +0.294 | [+0.075, +0.477] |

Both contrasts share problem runs, so a positive within-model r is partly expected by construction.

## X4: head_dim

head_dim: qwen2.5-0.5b 64, qwen2.5-1.5b 128, qwen2.5-3b 128, qwen2.5-7b 128, llama3.2-1b 64, llama3.2-3b 128, llama3.1-8b 128, mistral-7b-v0.3 128.

| Family | contrast | slope / decade, head_dim-128 models only | 95 % bootstrap CI |
|---|---|---|---|
| qwen (3 models) | H1 | +0.423 | [+0.200, +0.655] |
| qwen (3 models) | NAT-C1 | +0.276 | [+0.057, +0.489] |
| llama (2 models) | H1 | +0.009 | [-0.336, +0.349] |
| llama (2 models) | NAT-C1 | +0.275 | [-0.103, +0.673] |

| Family | step | Δlog10 P | contrast | Δ estimate | 95 % bootstrap CI |
|---|---|---|---|---|---|
| qwen | qwen2.5-0.5b (64) → qwen2.5-1.5b (128) | 0.495 | H1 | +0.306 | [+0.143, +0.467] |
| qwen | qwen2.5-0.5b (64) → qwen2.5-1.5b (128) | 0.495 | H2 | +0.127 | [-0.018, +0.274] |
| qwen | qwen2.5-0.5b (64) → qwen2.5-1.5b (128) | 0.495 | H3 | -0.424 | [-0.580, -0.264] |
| qwen | qwen2.5-0.5b (64) → qwen2.5-1.5b (128) | 0.495 | NAT-C1 | +0.427 | [+0.284, +0.571] |
| llama | llama3.2-1b (64) → llama3.2-3b (128) | 0.415 | H1 | +0.511 | [+0.390, +0.632] |
| llama | llama3.2-1b (64) → llama3.2-3b (128) | 0.415 | H2 | +0.213 | [+0.103, +0.321] |
| llama | llama3.2-1b (64) → llama3.2-3b (128) | 0.415 | H3 | -0.044 | [-0.162, +0.073] |
| llama | llama3.2-1b (64) → llama3.2-3b (128) | 0.415 | NAT-C1 | +0.400 | [+0.255, +0.545] |

Size and head_dim change together in this step; the difference cannot be attributed to either.
