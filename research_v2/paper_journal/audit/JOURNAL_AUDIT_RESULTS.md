# Journal-manuscript numerical audit: 52 checks, 0 failed

Independent re-derivation of every E1/E1L number from the raw records (no analysis module imported), against the locked report JSONs; locks and frozen manifest verified.

| Check | Result |
|---|---|
| lock_e1.py --verify | ok |
| lock_e1l.py --verify | ok |
| frozen manifest: 62 files unchanged | ok |
| Qwen Tier A: 200 test problems on all 6 arms | ok |
| Qwen Tier A: primary estimate | ok |
| Qwen Tier A: primary CI low | ok |
| Qwen Tier A: primary CI high | ok |
| Qwen Tier A: primary 90% low | ok |
| Qwen Tier A: primary p (sign-flip) | ok |
| Qwen Tier A: A: NAT - C2 (log JS) estimate | ok |
| Qwen Tier A: A: NAT - C2 (log JS) CI low | ok |
| Qwen Tier A: A: C2 - C1 (log JS) estimate | ok |
| Qwen Tier A: A: C2 - C1 (log JS) CI low | ok |
| Qwen Tier A: A: NAT - C1* (effect-matched, log JS) estimate | ok |
| Qwen Tier A: A: NAT - C1* (effect-matched, log JS) CI low | ok |
| Qwen Tier A: dose slope | ok |
| Qwen Tier A: flip-rate contrast | ok |
| Qwen Tier A: Holm re-applied to the reported raw p (7 tests) | ok |
| Qwen Tier A: 100 calibration problems on 5 arms | ok |
| Qwen Tier A: alpha* re-derived | ok |
| Qwen Tier A: dose slope b re-derived | ok |
| Qwen Tier A: margin re-derived | ok |
| Llama Tier A: 200 test problems on all 6 arms | ok |
| Llama Tier A: primary estimate | ok |
| Llama Tier A: primary CI low | ok |
| Llama Tier A: primary CI high | ok |
| Llama Tier A: primary 90% low | ok |
| Llama Tier A: primary p (sign-flip) | ok |
| Llama Tier A: A: NAT - C2 (log JS) estimate | ok |
| Llama Tier A: A: NAT - C2 (log JS) CI low | ok |
| Llama Tier A: A: C2 - C1 (log JS) estimate | ok |
| Llama Tier A: A: C2 - C1 (log JS) CI low | ok |
| Llama Tier A: A: NAT - C1* (effect-matched, log JS) estimate | ok |
| Llama Tier A: A: NAT - C1* (effect-matched, log JS) CI low | ok |
| Llama Tier A: dose slope | ok |
| Llama Tier A: flip-rate contrast | ok |
| Llama Tier A: Holm re-applied to the reported raw p (5 tests) | ok |
| Llama Tier A: 100 calibration problems on 5 arms | ok |
| Llama Tier A: alpha* re-derived | ok |
| Llama Tier A: dose slope b re-derived | ok |
| Llama Tier A: margin re-derived | ok |
| Qwen Tier B: 400 problems on 4 arms | ok |
| Qwen Tier B: primary estimate | ok |
| Qwen Tier B: CI low | ok |
| Qwen Tier B: CI high | ok |
| Qwen Tier B: NAT - C2 estimate | ok |
| Qwen Tier B: mean restricted first divergence NAT | ok |
| Qwen Tier B: mean restricted first divergence C1 | ok |
| Qwen Tier B: mean restricted first divergence C2 | ok |
| Qwen Tier A reference (stored Stage 1/2 FP16 generation, first 65 tokens) == in-session FP16 generation of Tier B, all 200 problems | ok |
| Llama reference histories are in-session FP16 greedy generations (<= 65 tokens), one per test problem | ok |
| neither experiment used a gold-answer reference history (both are FP16-generated) | ok |