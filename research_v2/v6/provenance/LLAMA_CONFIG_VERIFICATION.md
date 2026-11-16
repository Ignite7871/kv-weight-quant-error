# RoPE parameters of Llama-3.2-3B and Llama-3.1-8B: verified against the v4 pins (2026-11-14)

**Question.** R4 uses each model's rotary frequencies. For Llama-3.2-3B and Llama-3.1-8B:
- the official configs are gated, could not be downloaded with the local token, and were never cached locally;
- the unsloth mirror configs had first been used, unverified (R_ANALYSIS_PLAN_V6.md, "RoPE inverse frequencies").

**Pins.** `v4/V4_MODEL_MANIFEST.json` records, for `config.json` at each pinned revision, the size and the git-blob SHA-1:

| Model | Revision | Size | Git blob |
|---|---|---|---|
| llama3.2-3b | 13afe512… | 844 | 47d4a5aa69cdef91a53b77f5c5583647a578ca0e |
| llama3.1-8b | d04e592b… | 826 | cccf055d6f8f210387a248c91dc40e0c7a4bafab |

**Method.**
1. `reconstruct_and_verify_configs.py` starts from the mirror configs (https://huggingface.co/unsloth/Llama-3.2-3B and unsloth/Meta-Llama-3.1-8B, `main`, fetched 2026-11-14).
2. It removes the fields the mirror adds, varies `transformers_version` and the `rope_scaling` key order, and serializes in the Hugging Face format.
3. It keeps a candidate only if its size and git-blob SHA-1 both equal the pin. A SHA-1 match of the full file is a byte-for-byte identification; the chance of an accidental match is negligible.

**Result.** Both models matched.

| Model | Reconstructed file | Fields that differ from the mirror | RoPE fields |
|---|---|---|---|
| Llama-3.2-3B | `llama3.2-3b_config_13afe512.json` (844 bytes, blob 47d4a5aa…) | mirror adds `pad_token_id` and `unsloth_fixed`; `transformers_version` 4.52.4 (mirror) vs 4.45.0.dev0 (official) | **identical:** rope_theta 500000, llama3 scaling factor 32, low 1, high 4, original 8192; head_dim 128 |
| Llama-3.1-8B | `llama3.1-8b_config_d04e592b.json` (826 bytes, blob cccf055d…) | mirror adds `_name_or_path`, `head_dim`, `pad_token_id` and `unsloth_fixed`; `transformers_version` 4.49.0.dev0 vs 4.43.0.dev0 | **identical:** rope_theta 500000, llama3 scaling factor 8, low 1, high 4, original 8192. Head dim 128 = hidden 4096 / 32 heads, which is what the official config implies without the field |

**Consequence.**
- The RoPE inputs that `analyze_r_v6.py` uses for these two models are those of the pinned official configs.
- The provenance caveat in R_ANALYSIS_PLAN_V6.md is resolved without any change to the script or its inputs.
- No mirror source needs to be stated in the paper. The paper can say that every model's RoPE parameters come from its pinned configuration.

Files here are stored byte-exact (`.gitattributes`: `-text`), so their hashes can be rechecked.
