# PHASE2_LITERATURE (v4): scale literature, HQQ and Mistral-7B-v0.3 facts from primary sources

**Status: approved by the corresponding author on 2026-11-03. Every §5 edit is applied in PROTOCOL_V4 r3. The novelty entries of
§5 item 5 were added only after each full PDF was read (2026-11-03); r3 §1 states what each PDF shows.**

2026-11-02. Phase 2 of PROTOCOL_V4 (r2, commit adbed03). **No protocol text is changed here.** The edits this record implies are
listed in §5 as proposed r3 changes, for author review. No code, no runs, no cloud actions. Sources were read on 2026-11-02 with
Claude Code's web tools. Where a page could only be summarized, not quoted, the record says so.

## 1. HQQ (the optional second quantizer)

| Fact | Value | Primary source |
|---|---|---|
| Method | Half-Quadratic Quantization: calibration-free. It minimizes a sparsity-promoting ℓ_p loss (p < 1) of the weight reconstruction error with a half-quadratic splitting (auxiliary variable W_e, generalized soft-threshold step, closed-form zero-point update). The article: "we fix the scaling parameter s and only optimize for the zero-point z" | Badri & Shaji, "Half-Quadratic Quantization of large machine learning models", Dropbox Tech, 22 Oct 2025 (a republication of the Mobius team's 2023 post) |
| Original post | mobiusml.github.io/hqq_blog/: **returned 404 on 2026-11-02**. Search results date it November 2023 | (unavailable) |
| Citation | No arXiv paper exists. Cite the blog post: Badri, H., Shaji, A. (2023), "Half-Quadratic Quantization of Large Machine Learning Models", Mobius Labs blog; republished by Dropbox Tech, 2025 | as above |
| Code | `github.com/mobiusml/hqq` now resolves to **`dropbox/hqq`** | GitHub |
| Latest release | **0.2.8.post1, 20 Oct 2025** (previous: 0.2.8, 18 Aug 2025). This corrects r1/r2, which quoted 0.2.2 from a third-party mirror. Licence Apache-2.0 | pypi.org/project/hqq |
| Config | `BaseQuantizeConfig(nbits=4, group_size=64, quant_zero=False, quant_scale=False, offload_meta=False, view_as_float=False, axis=1)`, the defaults in `hqq/core/quantize.py` | dropbox/hqq master |
| Axis guidance | README: "You should start with `nbits=4, group_size=64, axis=1`"; "In general `axis=0` gives better results than `axis=1`, especially at lower bits" | README |
| Layer | `HQQLinear(linear_layer, quant_config, del_orig=True, compute_dtype=torch.float16, device="cuda", initialize=True)`; methods `dequantize()` and `unpack()` | `quantize.py`, README |
| Default backend | `HQQBackend.PYTORCH` ("forward_pytorch_backprop": dequantize, then matmul). Others: PYTORCH_COMPILE, ATEN (axis = 0 only), torchao tiny_gemm (bf16, axis 1), gemlite (fp16, axis 1) | `quantize.py`, README |
| Optimizer | `optimize_weights_proximal` = `optimize_weights_proximal_legacy`, `opt_params = {"lp_norm": 0.7, "beta": 1e1, "kappa": 1.01, "iters": 20}`. It stops at the first iteration that does not improve the error | `hqq/core/optimize.py` |
| Randomness | None in the proximal path. `torch.rand` appears only in the autograd fake-data optimizer, which is not used | `optimize.py` |
| Build | `DISABLE_CUDA=1 pip install …` skips the ATEN CUDA kernels. The PYTORCH backend does not need them | README |
| Not found | No determinism note; no stated compatibility with torch 2.13 / transformers 5.13; no statement that `HQQLinear` runs on CPU (device defaults to `"cuda"`) | — |

**Consequences for v4 (proposed in §5):**
- **Pin** `hqq==0.2.8.post1`, installed with `DISABLE_CUDA=1`, PYTORCH backend only. Compatibility with torch 2.13.0 is checked
  at implementation (Phase 3): an import test, a CPU or GPU round-trip test, and a dequantization match against an independent
  re-implementation of the affine dequantization (W = (W_q − z)·s per group).
- **Settings:** nbits 4, group 64, **axis 1** (the README's 4-bit starting point and the config default), and the library's default
  optimizer parameters. No tuning, because tuning would make HQQ a choice informed by results.
- **Effective size:** quant_zero and quant_scale are False, so each group of 64 stores an FP16 scale and zero:
  4 + 32/64 = **4.5 bits/weight**, against NF4's ≈ 4.13 (with double quantization). This is reported as a difference between the
  quantizers, not equalized.
- **Different error model.** HQQ's error is a minimized-ℓ_0.7 affine-grid error, against NF4's absmax code book. This is the
  intended contrast for FQ.
- **CPU tests:** if `HQQLinear` cannot run on CPU, the tiny-model tests quantize with HQQ's `Quantizer.quantize` and dequantize in
  PyTorch on CPU. The full HQQLinear path is then exercised only on the pod, in GM and G1/G2/G8.

## 2. Mistral-7B-v0.3

| Fact | Value | Source |
|---|---|---|
| Licence | apache-2.0 | HF API `cardData.license`; model card |
| Gated | **`"gated": false`** (resolves the r1/r2 "to be confirmed"). No HF token is needed | HF API, models/mistralai/Mistral-7B-v0.3 |
| Revision to pin | `caa1feb0e54d415e2df31207e5f4e273e33509b1` (lastModified 2025-07-24) | HF API |
| Description | "a Mistral-7B-v0.2 with extended vocabulary" (32768). Base model; the Instruct model is a separate fine-tune | model card |
| Geometry | config.json: hidden 4096, intermediate 14336, 32 layers, 32 heads, 8 KV heads, vocab 32768, untied, `sliding_window: null`, torch_dtype bfloat16. params.json (summarized, not quoted): head_dim 128, rope_theta 1,000,000, norm eps 1e-5 | config.json, params.json |
| Tokenizer | `add_bos_token: true`, `add_eos_token: false`. The rest of `tokenizer_config.json` was truncated by the tool (read at G5) | tokenizer_config.json |
| Files | both the HF shards (`model-0000{1..3}-of-00003.safetensors`) and `consolidated.safetensors` (mistral_inference format) | HF API siblings |

**Consequences:**
- The download uses `allow_patterns` for the HF shards, config and tokenizer files only. This **excludes `consolidated.safetensors`**,
  which would otherwise duplicate about 14.5 GB on the volume. G7 hashes the downloaded shards.
- `sliding_window: null`, so attention is full, as in the other families, and there is no special case in the cache.
- Mistral's RoPE base (1e6) differs from Llama 3.x (5e5, with scaled RoPE). This is one more within-"family" architecture difference,
  noted in L2.

## 3. Ladder geometry and licences from primary sources

- **Llama 3.2-1B / 3.2-3B / 3.1-8B (gated on HF).** Meta's `llama-models/models/sku_list.py` gives:

  | Model | dim | n_layers | n_heads | n_kv_heads | ffn_dim_multiplier | multiple_of | rope_theta | HF repo |
  |---|---|---|---|---|---|---|---|---|
  | Llama-3.2-1B | 2048 | 16 | 32 | 8 | 1.5 | 256 | 5e5, scaled | meta-llama/Llama-3.2-1B |
  | Llama-3.2-3B | 3072 | 28 | 24 | 8 | 1.0 | 256 | 5e5, scaled | meta-llama/Llama-3.2-3B |
  | Llama-3.1-8B | 4096 | 32 | 32 | 8 | 1.3 | 1024 | 5e5, scaled | meta-llama/Llama-3.1-8B |

  - head_dim = dim / n_heads = 64, 128, 128.
  - Meta's FFN rule gives intermediate sizes 8192, 8192 and 14336. All of this agrees with the r2 table, which used the model cards.
  - Not established from this source: the tied embeddings of 3.2-3B (the r2 projection assumed tied, as the HF config is understood to
    have it) and the vocab constant. G7 reads both from the pinned snapshot. An untied 3B head would add about 0.8 GB to the dual-weight
    memory (still far within 48 GB) and a small device-time term.
- **Qwen2.5-7B:** gated false, apache-2.0, revision `d149729398750b98c0af14eb82c78cfe92750796`. Qwen2.5-3B's research licence was
  verified in r1. The revisions of the other Qwen sizes are pinned at implementation, from the HF API.

## 4. Literature update: scale and the v4 hypotheses

**Search scope.** Web searches on 2026-11-02 for:
- weight-quantization error in the KV cache across model scale;
- K/V asymmetry across sizes;
- the common-mode (token-shared) component of quantization error;
- quantization error against equal-norm Gaussian noise.

Abstracts were read on arXiv for the items marked "verified". The search was not exhaustive, so the novelty statement stays "to our
knowledge".

| Work | What it shows | Bearing on v4 | Status |
|---|---|---|---|
| arXiv 2609.26173 (Dewage, Pensky, De Silva; IEEE ICMLA 2026 per page header) | Per-projection weight-only RTN/GPTQ at 3–4 bits; 9 models, including Qwen2.5-1.5B/7B and LLaMA-3-8B; V dominates ΔPPL in 7 of 9 | The **closest scale precedent** for H3: V > K at the weight level, at several sizes, on perplexity, with no cache or decoding measurement and no random controls | verified in v3 Phase 1 |
| arXiv 2608.08188 (signal–noise view of quantization degradation) | Scale interacts with bit width, method and task, rather than acting through one channel; the method shapes how error aligns with task activations | Motivates expecting the H1/H2 effects to change with scale; supports "alignment, not size" readings | cited in v3; the scale point from the search summary is to be checked in the PDF before any manuscript use |
| arXiv 2605.23901, "LLMs as Noisy Channels" (Ouyang et al.; ICML 2026 per comments) | A Shannon scaling law, validated with Gaussian weight noise, quantization and SFT on Pythia (to 12B held out) and OLMo2 | Gaussian weight noise and quantization studied together across scale. Its abstract does not say whether they are compared at equal norm; that needs the full text before any manuscript use | abstract verified |
| arXiv 2601.17187 (high-rate quantized matrix multiplication, theory) | Isotropic codebooks give distortion governed by trace(Σ), and covariance-aware schemes by det(Σ) | Theory for why natural (structured) error differs from isotropic error, the H1/H2 premise. Background only | search summary only; verify before citing |
| arXiv 2608.02691, OptR (Yun et al.; v4, 23 Sep 2026; no venue) | INT2 **KV-storage** quantization; an "attention-equivalent key reparameterization" removes large channel-wise key offsets without changing the softmax | Related to H5 (a shared offset in keys is softmax-invariant among positions). Storage, not weights | abstract verified |
| KVTuner (ICML 2025), KIVI (arXiv 2402.02750) | Storage-level K > V sensitivity; key outlier channels | Competing account for H3, as in v3 | v3-verified / known |
| arXiv 2606.09864 (Xu, Kumarappan, Zhou; v2, 8 Aug 2026; preprint) | Refusals erode under KV-cache quantization; 11 instruct models, 3.8B–72B | The search snippet claimed a K/V asymmetry ratio; **the abstract has no K/V comparison**. Not cited for K/V unless the PDF shows it. Storage, not weights | abstract verified; claim not confirmed |
| HeadQ, arXiv 2605.03562 | Key distortion modulo token-shared shifts | **Withdrawn (v3 Phase 1); not cited**, although it is the closest idea to H5's softmax argument | excluded |

**Gap statement (unchanged in substance from v3, extended to scale).** No work was found that measures weight-induced KV-cache error
against matched isotropic controls, by path, across a size ladder in several families. The scale precedents are perplexity-level
(2609.26173) or theoretical/scaling-law work (2605.23901, 2608.08188). Wording: "to our knowledge"; never "first".

## 5. Proposed r3 edits to PROTOCOL_V4 (not applied; for author review)

1. **§9 HQQ:**
   - replace the version caveat with: pin `hqq==0.2.8.post1` (PyPI, 2025-10-20), installed with `DISABLE_CUDA=1`, PYTORCH backend;
   - settings nbits 4, group 64, **axis 1**, the default proximal optimizer (lp 0.7, β 10, κ 1.01, 20 iterations), quant_zero and
     quant_scale False;
   - state 4.5 vs ≈ 4.13 bits/weight;
   - add the dequantization-oracle test and the CPU fallback test (§1);
   - code source `dropbox/hqq`;
   - citation: the Badri & Shaji blog (2023; Dropbox republication 2025).
2. **§2 Mistral:**
   - "gated: false" (verified);
   - revision `caa1feb0e54d415e2df31207e5f4e273e33509b1`;
   - download `allow_patterns` that exclude `consolidated.safetensors`;
   - RoPE 1e6 against Llama's 5e5 (scaled) added to L2.
3. **§2 table source column:** Llama geometry from Meta `sku_list.py`. The tied 3B head and the vocab are checked at G7. Revision pins:
   Qwen2.5-7B `d1497293…`; the others at implementation.
4. **§10 storage:** with the Mistral exclusion, about 65 GB of snapshots, so 150 GB still leaves room for the venv and records. No
   change to the ceiling.
5. **§1 novelty:** add 2609.26173 as the scale precedent for H3 (already cited in v3), and 2605.23901 / 2608.08188 as scale context,
   each only after its PDF is read.
6. **Bibliography candidates** (to `refs_v4_candidates.bib` in Phase 3): HQQ blog, 2605.23901, 2608.02691, 2601.17187 (once verified),
   Meta llama-models (geometry source).

## 6. AI use

Claude Opus 5.5 (claude-opus-5-5) in Claude Code, 2026-11-02. It did the web searches and page reads behind this record. Every
statement above comes from the named page. Pages that could only be summarized (params.json, the tokenizer config past 100k
characters, the Dropbox article's formulation) are marked.
