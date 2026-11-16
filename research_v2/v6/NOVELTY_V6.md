# v6 step 0: novelty check (plan phase, no GPU, no manuscript edits)

## Update 2026-11-14 (second pass): full-text reading of the five unverified papers

The author chose option 1, adjusted. The five papers were downloaded from arXiv and read from their PDF text. For the two
SageAttention papers, the method sections and every passage on smoothing, means or invariance were read; their kernel and
benchmark appendices were scanned, not read line by line. **Result: none of them covers (b). (b) remains open.**

| Paper | What it shows | Overlap with (a) | Overlap with (b) | Overlap with the decode-vs-prompt mass shift |
|---|---|---|---|---|
| Tuncer, Becker, Pfeil, "Quantized Keys Steal Attention: Bias Correction for KV-Cache Compression in Video Diffusion", arXiv:2605.26266 (25 May 2026) | KV-*storage* INT2/INT4. Zero-mean key noise of variance σ_i² inflates each cached term of the partition sum by E[e^{ε}] ≈ e^{σ²/2} (Jensen). Quantized cached keys therefore "steal" attention mass from the unquantized current chunk, ΔP_S > 0. Correction b_i ≈ σ_i²/2. Appendix O: the same structure in LLM partial prefill. | None beyond the setting. The paper states the noise is zero-mean and that "quantizers with nonuniform or biased error may likewise require a modified derivation". | No. Storage only, keys only, no weight quantization, no common mode, values not analysed. | **Partial, second order only.** Its structure (perturbed cached block against an unperturbed block in one softmax) is ours (perturbed decode entries against the FP16 prompt). It derives the *variance-driven* mass shift toward the perturbed block. It does not treat the *first-order* shift produced by a common-mode key error (q·μ added to every perturbed score), which has either sign and cancels only within the perturbed block. |
| Zhang et al., SageAttention, ICLR 2025, arXiv:2410.02367 | Smooth K before INT8 quantization of QKᵀ: "Each token's key is actually a large bias shared by all tokens, plus a small token-wise signal", and σ(q(K−mean(K))ᵀ) = σ(qKᵀ). | Key side of (a), applied to the *signal* for kernel-precision reasons. | No. | No. |
| Zhang et al., SageAttention2, arXiv:2411.10958 | Smooths Q per block and K. "Adding a common bias to an entire row of S does not affect the result after softmax." Optional V smoothing: subtract the channel mean of V and add it back to the output; needed only where V has channel-wise bias, "absent in some models, such as Llama3". | Both halves of (a), operationally, for the signal. | No. | No. |
| Fernández-Menduiña et al., "Spend Bits Where Queries Look" (NOVA-KV), arXiv:2608.04074. **Already cited as fernandezmenduina2026novakv** | KV-storage transform coding. Lemma S1: replacing every key k_j by k_j − c leaves attention unchanged. Also: "since S1 = 1, a shared value offset satisfies S(V − 1c_V) = SV − 1c_V", so it is subtracted and added back exactly. | **Both halves of (a), stated formally.** | No. Storage only. A shared *error* component is not analysed. | No. |
| D'Alberto, "Statistical Inference and Quality Measures of KV Cache Quantisations Inspired by TurboQuant", arXiv:2605.08114 (notes; acknowledges preparation with Claude) | Storage only. "If b_i = b were constant across all keys, it would cancel exactly"; "if σ_i² is uniform across keys the factor e^{σ²/2} cancels". K enters through inner products and V linearly, so "the per-instance bias in V quantisation propagates directly and additively to the output". | (a), informally. | No. Its K-V asymmetry is the storage-side one (K more damaging). | No. |

**Consequences for v6:**
1. (a) is prior work. NOVA-KV's Lemma S1 states both identities, and NOVA-KV is already in the bibliography. The theory plan cites it and SageAttention rather than deriving the identities as new.
2. The *second-order* decode-vs-prompt mass shift is Tuncer et al.'s Jensen bias and must be cited as theirs.
3. Still open, as far as this reading goes, and therefore the target of THEORY_PLAN_V6.md:
   - the *first-order* common-mode mass shift;
   - its combination with value-side accumulation;
   - the account of why weight-induced error ranks V above K when storage error ranks K above V.

Date: 2026-11-14. Status: **stopped for author review.** The stop condition in the v6 brief was met: (a) is largely
known and (c) is largely known in direction, so THEORY_PLAN_V6.md, PROTOCOL_V6.md and the JMLR fit notes were not drafted.

Search scope: targeted and English-only, mostly arXiv, run on 2026-11-14. The primary sources marked "read" were opened
directly (abstract page, PDF text or source code). The ones marked "abstract only" were checked only at the abstract level,
and their full text still needs reading before anything is cited. This is a search result, not a proof of absence.

## Verdict in one paragraph

(a) The core identity is known. A key perturbation shared by every attended position adds the same scalar q·δ to every
logit, and softmax is shift-invariant, so it cancels. A value perturbation shared by every position passes through
unchanged, because the attention weights sum to 1. Both appear in the bias-term literature and in attention and
KV-quantization methods that rely on them. (b) Partly open. The KV-storage literature uses the key-side cancellation, and
weight-quantization literature knows weight error produces a systematic mean shift in layer outputs. We found no work that
combines the two to explain why *weight-induced* cache error is harsher on V than K, the opposite of the usual
KV-storage finding that K is more sensitive. (c) The direction is known: V-projection weights are the most sensitive
attention projection, and llama.cpp's default 4-bit mixes protect attn_v but never attn_k. We found no locked comparison
of FP16 protection of v_proj vs k_proj with an equal-parameter control. That is the only new part, and it is incremental.

## (a) Softmax cancels shared key perturbations; shared value perturbations accumulate

| Source | Status | What it shows |
|---|---|---|
| Namazifar, Hazarika, Hakkani-Tür, "Role of Bias Terms in Dot-Product Attention", ICASSP 2023, arXiv:2302.08626 | read (abstract and summary) | Proves the key-projection bias is redundant (no effect on attention); the value bias has a "more prominent role". The paper notes the key-bias result was also reported independently elsewhere. This is exactly the position-shared key perturbation, in bias form. |
| Zhang et al., SageAttention, arXiv:2410.02367; SageAttention2, arXiv:2411.10958 | secondary summaries only; primary text to be read | Subtract the token-mean of K before quantizing because softmax is invariant to it (no add-back). SageAttention2 also smooths V but must add the V mean back to the output, which is the value pass-through stated operationally. |
| Ruiz Williams, HeadQ, arXiv:2605.03562 (v3, 19 May 2026; **marked withdrawn**) | abstract only | "For keys, the visible object is score error modulo constant shifts." KV-storage only, no weight quantization. |
| Laus et al., "KV Cache Compression Through the Lens of Transform Coding" (AATC), arXiv:2608.14191 | abstract only | Under a white-noise model, the expected attention-aware distortion splits into additive key and value terms. White noise by construction, so no common-mode analysis. Already cited in the manuscript as laus2026aatc. |
| arXiv:2608.04074, arXiv:2605.08114 | abstract only | A search summary attributed a "constant added to all keys" lemma and a "common key bias cancels" remark to these. **Not confirmed** at abstract level. Do not cite them for this until the full text is read. |
| Our own manuscript, Section 3.6 (pilot) | ours | States the same cancellation before the data as a candidate reason. The key error shared across decode positions cancels among decode positions but not against the error-free prompt. Removing the common mode (centering) closed about a third of the value-path gap and did nothing on the key path. |

**Is it already known?** Yes, at the level of the identity. A v6 theory section cannot claim the cancellation or the pass-through
as a result. What our setting adds is narrower, and each item would still need its own check:
1. **Partial cancellation.** Only the decode-written entries carry error; the prompt entries are error-free. So a shared key error does not
   fully cancel. It shifts the decode-versus-prompt attention mass by about q·δ × (mass on decode positions) × (mass on
   prompt positions). That gives a testable prediction: harm grows as the decode share of attention mass approaches one half.
2. **Variance reduction.** The position-varying value error is averaged by the attention weights, so its energy scales with Σ a_j².
   The common-mode value error is not averaged at all. That predicts the V-path excess over isotropic should scale with
   the common-mode share C^V divided by Σ a_j².
3. **Quantitative test.** Both predictions can be checked against the v4/v5 per-problem records (C^P per path and the
   natural-versus-isotropic contrasts) with no new data, if the records hold what is needed (see "Open checks").

## (b) Structured or common-mode compression error linked to K-vs-V sensitivity

| Source | Status | What it shows |
|---|---|---|
| Nagel et al., "Data-Free Quantization Through Weight Equalization and Bias Correction", ICCV 2019, arXiv:1906.04721 | abstract plus summary | Weight quantization introduces a *biased* (systematic) error in layer outputs, E[ΔW x] = ΔW E[x], which they correct. This is the origin of a common-mode component: any input mean shared across positions turns weight error into a shared output error. |
| KIVI (arXiv:2402.02750), KVQuant, KVTuner, AsymKV (COLING 2025), QuaRot (arXiv:2404.00456) | KIVI, KVTuner, KVQuant already cited; others abstract only | In KV-*storage* quantization keys are the more sensitive path (per-channel outliers; error passes through the softmax exponential, while value error enters linearly). |
| arXiv:2605.26266, "Quantized Keys Steal Attention" (video diffusion) | title and summary only | Quantized keys bias attention mass toward cached keys, the non-cancelling case. Possibly close to point 1 under (a). **Read before claiming anything about partial cancellation.** |

**Is it already known?** The pieces are known: weight error has a systematic part, and shared key error cancels. We did not find them
combined into an account of why weight-induced error is harsher on V while storage error is harsher on K. That
reconciliation is the most defensible new contribution available to v6. It is a modest one: a first-order argument plus a test on
existing records, not a new phenomenon.

## (c) FP16 protection of v_proj vs k_proj, with an equal-parameter control

| Source | Status | What it shows |
|---|---|---|
| Dewage et al. 2026 (manuscript ref dewage2026componenttype) | already cited | Quantizing one attention projection at a time in nine 1.3B–8B models: the value projection carries the largest share of the perplexity increase in most. |
| Hu et al., "Identifying Sensitive Weights via Post-quantization Integral", arXiv:2503.01901 | read (PDF) | Llama-3.2-1B, WikiText-2: "v proj has the largest average F_PQI, denoting greatest impact on final results, followed by k proj and o proj." This is per-element sensitivity, not a protection experiment. |
| llama.cpp `src/llama-quant.cpp` (master, read 2026-11-14) | read (source) | Q4_K_M raises attn_v to Q6_K on the layers selected by `use_more_bits` (first and last eighth, every third in between); Q4_K_S raises the first four attn_v to Q5_K. **No Q4_K_M/Q4_K_S rule raises attn_k.** This is current practice: protect V, not K. |
| Hugging Face blog, "Sensitivity Aware Mixed Precision Quantization V1" | secondary | v_proj markedly sensitive in GPT-Neo and TinyLlama; k_proj and q_proj less so. |
| QUIK (arXiv:2310.09259) | abstract only | Keeping down_proj at 8 bits is critical on Llama-2, the same kind of per-layer-type protection result. |

**Is it already known?** The direction is, and it is used in deployed software. The proposed arms (i)–(v) would largely replicate it.
What is new is narrow:
- an equal-parameter control, and
- locked, pre-declared inference with power.

Two design facts weaken even that:
1. **K and V are already the same size.** In every proposed model (Qwen2.5-0.5B, Qwen2.5-1.5B and Llama-3.2-1B, all GQA), k_proj and v_proj have identical shapes. A K-vs-V comparison is equal-parameter by construction. The control has to be a different same-size block, such as an equal-count slice of q_proj, o_proj or MLP rows, or random rows.
2. **The likely result is not surprising.** "V-protection recovers more perplexity than K-protection" is the expected answer given the three sources above. A JMLR reviewer would likely call it a confirmation unless it is tied to the mechanism in (b).

## What this means for the v6 plan (author decision)

The brief asked me to stop here, so I have not chosen. The options as I see them:

1. **Mechanism-first, no new GPU.** Build THEORY_PLAN_V6 around (b): the reconciliation of weight-induced V-harsh vs
   storage-induced K-harsh, with partial cancellation against the prompt and Σa² averaging as the quantitative
   predictions, tested on the v4/v5 records. Cite (a) as known. This is the part with a real claim to novelty.
2. **Mechanism plus a small mitigation check.** Do option 1, then run PROTOCOL_V6 with fewer arms:
   - V-protect, K-protect and a same-size control;
   - one or two models on the laptop;
   - framed as a test of a prediction the theory makes, not as a new finding. llama.cpp practice and Dewage et al. stated up front.
3. **Drop the mitigation arm.** The prior art makes it a replication. Spend the effort on generality instead (more families or a second task) in the existing measurement design.

My recommendation is option 1, plus a very small version of option 2 if the theory predicts something the protection arms can
separate, for example a size or head-dimension dependence that the known "protect V" heuristic does not.

## Open checks before any of this is used

- Read the full text of SageAttention/SageAttention2, arXiv:2605.26266, arXiv:2608.04074 and arXiv:2605.08114. All were
  seen at summary or abstract level only. If 2605.26266 already derives the prompt-versus-cache mass shift, point 1 under (a) is not new.
- Check whether the v4/v5 per-problem records hold per-position attention weights or the decode-versus-prompt attention mass.
  The predictions under (a) need them. If not, they "require new logging" and cannot be tested on existing records.
- A systematic search in Semantic Scholar or Google Scholar for "common-mode" / "mean shift" + "value cache" / "value projection".
  This one was targeted.

## Sources

- https://arxiv.org/abs/2302.08626
- https://arxiv.org/abs/2410.02367 , https://arxiv.org/abs/2411.10958
- https://arxiv.org/abs/2605.03562
- https://arxiv.org/abs/2608.14191
- https://arxiv.org/abs/2608.04074 , https://arxiv.org/abs/2605.08114
- https://arxiv.org/abs/1906.04721
- https://arxiv.org/abs/2402.02750 , https://aclanthology.org/2025.coling-main.158.pdf , https://arxiv.org/abs/2404.00456
- https://arxiv.org/abs/2605.26266
- https://arxiv.org/abs/2503.01901
- https://github.com/ggml-org/llama.cpp/blob/master/src/llama-quant.cpp
- https://huggingface.co/blog/badaoui/sensitivity-aware-mixed-precision-quantizer-v1
- https://arxiv.org/abs/2310.09259
