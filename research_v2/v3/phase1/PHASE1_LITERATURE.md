# v3 Phase 1 — literature verification (no GPU)

2026-10-29, branch `v3-novelty`. Every item below was checked against its arXiv abstract page. Where a detail
needed the paper body, I used the arXiv HTML full text and cite section numbers. Page text was read through Claude
Code's WebFetch, whose page summaries come from a model I cannot identify (see `AI_USE_LOG_V3.md`). Anything not
confirmed on a primary page is marked **[UNVERIFIED]**.

Terms used below:
- **Storage codec** = the KV entries are computed exactly and then compressed or quantized for storage.
- **Weight-induced** = our setting: the entries are *computed* by a model with NF4 weights, so the error enters before
  any storage.
- **Single-step** = error measured at one forward pass or teacher-forced position.
- **Persistent** = an error written to the cache that is read again at every later decoding step.

## A. The eight requested sources

**1. arXiv 2607.28495.** Lorup (single author; affiliation printed as "Openhagen"). *Stage-Replay Divergence Follows
the KV Cache: Fixed-Prefix Precision Controls and Bidirectional Cache Transplantation.* v1, 30 Jul 2026. No venue.
- What it does: on a 14B Qwen2.5-derived checkpoint, rebuilding a reasoning-stage boundary state by live-cache vs
  one-shot prefill of the same tokens changes 166/200 suffixes. Whole-cache K/V transplants (all 48 layers, K and V
  jointly) make the continuation follow the cache donor (24/24; 43/43 in a replication). FP32 removes the divergence.
- How it differs: "precision" here means BF16 vs FP32 *compute* (TF32 flags). There is no weight quantization, no
  random or norm-matched control and no K/V separation.
- Relevance: prior art for "the cache carries a persistent trajectory change". It is not about the *origin* of the
  cache error, and not about weight-induced error.

**2. arXiv 2609.23125.** Sathyanarayanan (single author). *Perplexity Cost Understates What Activation Quantisation
Breaks.* v1, 19 Sep 2026. No venue.
- What it does: perturbs the **residual-stream activation** at one block (forward hook; §3). Controls are matched
  Gaussian noise with the same per-coordinate second moment (§2/§3), sign-randomized error with |ε| preserved per
  coordinate per token (§7.1), and rotated-basis quantization (§8). §7.1 is titled verbatim "A ladder of substitutes"
  ("substitutes that hold progressively more of the error fixed"). Finding: magnitude alone does not explain the damage.
- How it differs: activation quantization, not weights. The KV cache is explicitly *not* perturbed (§3: "not the
  linear-layer inputs or KV-cache entries"). Scoring is teacher-forced perplexity plus probes; free-running decoding
  is not described.
- Relevance: the **closest methodological precedent for our C1/C2 controls**. Same-magnitude Gaussian and
  sign-randomized substitutes for a natural quantization error are therefore *not new as a control idea*. What is new
  in our use is the object they are applied to: weight-induced, newly written cache entries.

**3. arXiv 2610.19605.** Song, Yoo. *PN-Mix: Pair-Norm Guided Mixed-Precision Key Cache Quantization for Efficient
Long-Output LLM Inference.* v1, 14 Oct 2026. Comments: "EMNLP 2026".
- What it does: an attention-logit perturbation analysis at RoPE channel-pair level shows that key-quantization error
  in pairs with large query–key pair-norm products drives logit variation. Salient pairs get more bits; they are
  chosen at prefill and reused during decoding.
- How it differs: storage codec, keys only.
- Relevance: prior art for **query-weighted logit error as the predictor of key error** (our P_logit, P_out).

**4. arXiv 2608.04074.** Fernández-Menduiña, Ziashahabi, Pavez, Ortega, Avestimehr. *Spend Bits Where Queries Look:
KV Cache Vector Quantization with Attention-Preserving Transforms (NOVA-KV).* v1, 4 Aug 2026. No venue.
- What it does: treats KV quantization as transform coding with attention-product distortion. Gives closed-form key and
  value transforms from calibration statistics. The key transform is non-orthogonal and turns attention distortion
  into MSE.
- How it differs: storage codec, evaluated on long-context retrieval at 2 bits.
- Relevance: prior art for **query-statistics-weighted distortion** of keys and values (our P_logit / P_val, and the
  QA/QO idea of a calibration-estimated query subspace).

**5. arXiv 2608.14191.** Laus, Mayrink Verdun, Wang, Calmon, Krahmer. *KV Cache Compression Through the Lens of
Transform Coding (AATC).* v1, 14 Aug 2026. No venue.
- What it does: under a white-noise model, proves that expected attention-aware distortion splits into additive key
  and value terms that factor across tokens and channels. Allocates bits by reverse water-filling. Evaluated on
  Llama-3.1-8B-Instruct and Qwen-2.5-7B-Instruct (incl. GSM8K).
- How it differs: storage codec. The white-noise assumption is what our NAT-vs-C1 contrast tests empirically for
  *weight-induced* error.
- Relevance: theory that our P_logit/P_val predictors would borrow. It also suggests a **testable assumption**: is
  weight-induced Δ "white" with respect to the query distribution?

**6. arXiv 2609.36760.** Su, Sun, Tan, Zhang, Hu, Xie, Cai, Wong. *QuantMLA: Function-Aligned Dual-Path Quantization
for Low-Bit MLA KV Caching.* v1 29 Sep, v2 30 Sep 2026. No venue.
- What it does: §2.2 and App. B.1 "perturb one path's cache at a time, keeping the other path and model parameters
  fixed" and "rescale each perturbation to a common cache NMSE of 0.01", keeping the direction. Finding: RoPE-path
  errors cause more attention-output distortion than content-path errors at equal energy.
- How it differs: storage codec on MLA caches. The analysis is per-layer on fixed sequences, not tracked across a
  decoding trajectory. The phrase "matched-error path perturbation" is **not** the paper's own; its terms are
  "Matched-Error Protocol" (App. B.1) and "equal-energy".
- Relevance: **direct precedent for H-D**. Comparing two cache paths requires energy-matching each path's
  perturbation. Our NAT-K vs NAT-V as specified (Δ on keys only vs values only, each at its *natural* size) is
  confounded by magnitude unless norm-matched variants are added. → Phase 2 design note.

**7. arXiv 2502.04420.** Li, Xing, Li, Qu, Zhen, Liu, Yao, Pan, Yuan. *KVTuner: Sensitivity-Aware Layer-Wise
Mixed-Precision KV Cache Quantization for Efficient and Nearly Lossless LLM Inference.* v1 6 Feb 2025 … v5 20 Nov
2025. Comments: "Accepted by ICML25".
- What it does: theoretically links layer-wise attention patterns to KV quantization error. Explains why **keys
  generally matter more than values**, then searches layer-wise K/V precision pairs offline (e.g., Qwen2.5-7B-Instruct
  needs 4.0 bits, Llama-3.1-8B-Instruct 3.25).
- How it differs: storage codec, accuracy-oriented.
- Relevance: prior art for the K>V prior. If our H-D finds K>V for weight-induced error, that **confirms a transfer**;
  it is not a discovery. Its Qwen-more-sensitive-than-Llama observation is suggestive context for our model contrast
  (different scale and tuning; not evidence).

**8. arXiv 2608.08188.** Zhou, Cao, Ye, Yu, Yu, Li, Zhao, Liu. *Quantization Degradation in Large Language Models: A
Signal-Noise Perspective.* v1, 8 Aug 2026. No venue.
- What it does: **weight-only** PTQ. Source SNR per module is decomposed as M + S − N (§4.3, Eq. 8). N is the
  activation energy along the weight-error direction: 10·log10(δwᵀΣ̄δw/‖δw‖²), with Σ̄ from clean activations. The paper
  also measures cross-layer attenuation or amplification and finds weaker amplification in larger models.
- How it differs: covers q/k/v projections only as weight modules. No KV cache, attention logits, decoding steps or
  random controls.
- Relevance: the **closest weight-quantization analogue of our "alignment" predictors**, at the weight level. Our P_logit
  moves the same idea one step downstream: the alignment of the *cache-entry* error with the *query* distribution. Its
  cross-layer analysis is a depth analogue of our LOC/INH question.

## B. Search for weight-quantization-induced cache error, matched controls, or local-vs-inherited decomposition

I ran six searches: four extended and two standard. Additional relevant work verified on arXiv abstract pages:

| Work | Verified facts | Relation to v3 |
|---|---|---|
| **Dewage, Pensky, De Silva, arXiv 2609.26173** *Component Type, Not Reconstruction Error, Predicts Attention Quantization Sensitivity* (v1, Aug 2026; page header: accepted at IEEE ICMLA 2026) | Weight-only RTN/GPTQ 3–4 bit on **one projection (Q, K, V or O) at a time**, everything else full precision; 9 models incl. Qwen2.5-1.5B/7B and LLaMA-3-8B. V projections dominate ΔPPL in 7 of 9 models; reconstruction error is a weak within-component predictor (median R² 0.044). Perplexity only; no KV-cache error, no decoding, no random controls, no k/v-alone vs all-other-layers comparison (§III-C, §IV). | **Closest precedent for the LOC arm.** Projection-level weight quantization is *not new*. New in v3: measuring the resulting *cache-entry* error and its persistent effect, plus the complementary INH arm. Its V>K finding at the *weight* level contrasts with KVTuner's K>V at the *storage* level, which is a sharp, citable motivation for H-D. |
| **Chang, Zhang, Thomason, Jia, arXiv 2506.12044** *Why Do Some Inputs Break Low-Bit LLM Quantization?* (v2, Sep 2025; EMNLP 2025) | Weight-only 3–4 bit; per-example errors strongly correlated across methods; full-precision residual-stream magnitude predicts them; amplification across layers; activation patching localizes to late-layer residuals and MLP gates. No KV-cache analysis. | Prior art that weight-quantization error is largely **inherited through the residual stream**. This makes a strong INH > LOC result *expected*, not surprising. Phase 2 should state this prior. |
| **Arai, Ichikawa, arXiv 2504.09629** *Quantization Error Propagation: Revisiting Layer-Wise PTQ* (v3, Jan 2026; "Accepted at NeurIPS 2025") | Errors accumulating across layers degrade layer-wise PTQ; QEP propagates and compensates them. No KV/decoding analysis. | Same: cross-layer (inherited) error is established for weights; not for cache entries. |
| **Muller et al., arXiv 2606.03458** *KVarN* (v1, Jun 2026) | KV-storage quantization errors **accumulate across decoding timesteps**, mainly through wrong token scales. | Prior art for *persistent* accumulation of storage-codec error. Our persistence claim must cite it. |
| **Ruiz Williams, arXiv 2605.03562** *HeadQ* | Score-space (query-basis) error predicts attention KL better than key MSE; query-PCA and null-space controls. **v3 is WITHDRAWN** ("ethical concerns identified after posting"). | **Do not cite as evidence.** If mentioned at all, mark it as withdrawn. The QA/QO idea has to rest on PN-Mix, NOVA-KV and AATC instead. |
| **Xu, Kumarappan, Zhou, arXiv 2606.09864** *Alignment Collapse Under KV Cache Quantization* (v2, Aug 2026; preprint) | KV-storage quantization reduces refusals at small perplexity cost. | Peripheral; storage codec. |

Already in the manuscript and still the closest *weight-quantization × cache* prior art: Du et al. 2609.26621 (TMLR
2026; single-step cross-precision KV grafts) and Chodavarapu & Xu 2604.15409 (FP16 cache-on/off divergence). Also
QSpec/QuantSpec, which reuse KV across precisions.

### What I did **not** find

I found no work that:
- (a) measures the error in KV entries *written by a weight-quantized model* (weight-induced cache error) as an object of
  study,
- (b) compares it with norm-matched isotropic or sign-randomized substitutes *on the cache entries*,
- (c) splits it into a local part (quantized k/v projections) and an inherited part (residual stream altered by other
  quantized layers), or
- (d) tests whether KV-storage error predictors transfer to it.

**This is a search result, not a proof of absence.** The searches were web-indexed (arXiv-heavy), English-only, and run
on 2026-10-29. The safe wording is: "to our knowledge, … has not been examined". Each component *idea* is prior art:
- the matched controls (2609.23125, QuantMLA);
- query-weighted predictors (PN-Mix, NOVA-KV, AATC);
- the K vs V asymmetry (KVTuner, 2609.26173);
- projection-level weight quantization (2609.26173);
- cross-layer inherited error (2506.12044, QEP, 2608.08188);
- persistent accumulation (KVarN, Lorup).

The defensible novelty is the **combination applied to weight-induced cache error**, plus the cross-model
explanation test.

## C. Consequences for the Phase 2 protocol (for author decision; nothing locked)

1. **H-D (NAT-K vs NAT-V)** needs norm-matched versions, as in QuantMLA's equal-energy protocol. Otherwise a K/V
   difference may just reflect ‖Δ_K‖ ≠ ‖Δ_V‖. Proposal: keep the natural arms and add K-only and V-only arms rescaled
   to a common per-slot norm.
2. **Pre-declared priors.**
   - KVTuner predicts K > V (storage).
   - 2609.26173 predicts V-projection weight error dominates (weight level).
   - 2506.12044, QEP and 2608.08188 predict a large inherited component (INH > LOC).
   The protocol should state these as the *expected* outcomes, so that confirming them is reported as transfer, not
   discovery.
3. **P_logit** should be defined compatibly with the PN-Mix/AATC key term (query-weighted logit error, invariant to
   constant shifts across positions). **P_val** should be compatible with AATC's value term, so that "transfer" is
   tested against the published form, not a variant of our own.
4. **QA/QO:** cite PN-Mix, NOVA-KV and AATC for the query-subspace rationale, not HeadQ (withdrawn).
5. **Abstract and Related Work wording:** C1/C2-type substitutes are borrowed (2609.23125). Our contribution is
   applying them to weight-induced cache entries under persistent decoding.

## D. Not verified / open

- Full texts of PN-Mix, NOVA-KV and AATC were not read; their summaries rest on the abstracts. Exact formulas must be
  read before Phase 2 fixes P_logit/P_val. **[UNVERIFIED: exact key/value distortion formulas in 2610.19605,
  2608.04074, 2608.14191]**
- KVTuner's K>V argument was read from the abstract only.
- The remaining ~27k characters of 2609.23125 and ~42k of 2609.36760 were not read (truncated HTML). The quoted
  sections are from the part that was read.
- Venues: PN-Mix "EMNLP 2026" and KVTuner "ICML25" are from the arXiv Comments field only. 2609.26173 "IEEE ICMLA 2026"
  is from the HTML header note. These are author-provided statements, **[UNVERIFIED]** against the proceedings.
