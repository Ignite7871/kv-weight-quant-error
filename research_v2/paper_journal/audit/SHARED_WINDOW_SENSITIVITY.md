# Exploratory shared-window sensitivity analysis (CPU only; stored records; replaces no original result)

Seed 20261201 (new stream), 10000 paired problem-clustered resamples (the same resampled problems for both models, so between-model differences are paired). Log mean-JS contrasts, as in the locked analysis (floor 1e-09).

## Reference histories

* Qwen: mean S 61.2, median 64, S=64 in 83.5% of problems, mean stored-generation length 112.1 tokens (capped at 65 in Tier A).
* Llama: mean S 56.7, median 64, S=64 in 56.5%, mean capped length 57.7.
* Per problem: Llama window shorter in 78, equal in 105, longer in 17 problems; Spearman correlation of S across models +0.42; both S=64 in 105 problems; both S>=32 in 188.
* A token-level shared prefix between the two histories is **not defined** (different tokenizers, different generating models).
* Text level (decoded, common character prefix of the first 65 tokens of each history): mean 10.6 characters, median 5; identical for >=20 characters in 16% and >=60 characters in 1% of problems.

## Contrasts on shared windows (NAT-C1 is the locked primary; the others are secondary)

| Window / problem set | n | Qwen NAT-C1 | Llama NAT-C1 | Llama-Qwen NAT-C1 | Qwen NAT-C2 | Llama NAT-C2 | Qwen C2-C1 | Llama C2-C1 |
|---|---|---|---|---|---|---|---|---|
| original windows (each model's own S_i): reproduces the locked estimates | 200 | -0.253 [-0.326, -0.180] | +0.058 [-0.009, +0.120] | +0.311 [+0.216, +0.406] | +0.003 [-0.081, +0.084] | +0.266 [+0.196, +0.334] | -0.256 [-0.334, -0.174] | -0.209 [-0.274, -0.141] |
| shared per-problem window: steps 1..min(S_Qwen, S_Llama) | 200 | -0.264 [-0.339, -0.191] | +0.057 [-0.010, +0.121] | +0.321 [+0.225, +0.417] | +0.013 [-0.070, +0.094] | +0.267 [+0.196, +0.335] | -0.278 [-0.357, -0.196] | -0.210 [-0.276, -0.143] |
| steps 1-32, problems with S>=32 in both models | 188 | -0.248 [-0.329, -0.167] | +0.109 [+0.039, +0.175] | +0.356 [+0.248, +0.461] | +0.048 [-0.047, +0.143] | +0.286 [+0.218, +0.354] | -0.295 [-0.394, -0.196] | -0.178 [-0.253, -0.104] |
| steps 1-64, problems with S=64 in both models | 105 | -0.317 [-0.421, -0.215] | +0.050 [-0.039, +0.133] | +0.367 [+0.237, +0.494] | -0.103 [-0.212, +0.005] | +0.275 [+0.179, +0.362] | -0.214 [-0.324, -0.103] | -0.224 [-0.316, -0.136] |
| steps 1-16, problems with S>=16 in both models | 200 | -0.187 [-0.282, -0.095] | +0.121 [+0.029, +0.213] | +0.308 [+0.182, +0.435] | +0.127 [+0.027, +0.222] | +0.349 [+0.259, +0.438] | -0.314 [-0.416, -0.209] | -0.228 [-0.324, -0.134] |

Reading: see the manuscript's exploratory appendix. These analyses are exploratory, were defined after the original results were known, and none replaces a locked result.