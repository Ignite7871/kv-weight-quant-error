# PROTOCOL_V5: size vs head dimension, a second quantizer and a second task (r2)

Draft r1 and r2, 2026-11-11, branch `v3-novelty`. **r2 records the corresponding author's decisions D-V5-1 to D-V5-8 (§15) and
the history-source statement (§1, §3, §12, §13).** No code, no lock, no cloud action and no authorization file exist at r2. Every
number below comes from `v5/plan/v5_plan_numbers.py`, which reads the locked v4 report, the v4 GPU ledger and the v4 G6 design
read-only, with no GPU and no network. Model facts were read from the Hugging Face API and the pinned `config.json` files on
2026-11-11.

- New, separately locked study, with the v4 discipline:
  - a dated private SHA-256 lock (not a public pre-registration);
  - gates and a CUDA mutation check (GM);
  - a Step-0-style anchor against the v4 records;
  - an independent recomputation, with tolerances declared and locked before any data;
  - a deviation log;
  - a GPU ledger and a cost ledger.
- **Origin.** The v5 questions come from the v4 results: the size/head-dimension confound, one quantizer and one task. The
  predictions are therefore exploratory in origin. They are locked before any v5 data, and v5 is reported as a separate study,
  not as part of the v4 lock.
- Nothing from v4, v3, X9, X13, E1/E1L or Stage 1/2 is edited. Locked v4 code is imported unchanged and hash-checked.

## 0. Read first

1. **Three components, one lock.** A (size vs head dimension), B (second quantizer, HQQ) and D (second task, WikiText-2) run
   under one lock and one budget. A and the anchor are never cut. B and D carry the pre-declared reductions (§9).
2. **Hard ceiling $15 in total, storage included, with a ledger stop at $14.** The prepaid balance stays at about $17.87 (the
   author's figure, D-V5-7). It is above the ceiling, so unlike v4 **the balance is not a hard stop outside the code: the $14 ledger
   stop is the limit**, enforced by the per-problem start check, the per-unit start check and the watchdog.
3. **Projection (the v4 factor of 1.48 built in).** The core design is $11.59 if HQQ costs 1.5× NF4, $12.92 at 2×, and $15.58
   at 3×. At 3× the first reduction (n = 60 for B and D) brings it to $12.88. The HQQ speed has never been measured; the
   rehearsal and an on-pod probe measure it before the design is fixed (§9).
4. **Only ungated models.** No HF token, no RunPod secret. Every model and the dataset are public and not gated (§2).
5. **The A design cannot fully separate size from head dimension.** Qwen3-0.6B-Base changes head dimension *and* adds
   QK-norm *and* is a new model generation. SmolLM2-1.7B changes family, tokenizer, data and attention layout (no GQA). The
   crossed design gives a pattern reading, not a causal one (§7.2, L-A).

## 1. Questions

- **A. Size or head dimension?** In v4, size and head dimension changed together in both families: the smallest model of each
  family had head dimension 64, every other model 128. The key-path advantage (H1) shrank and NAT−C1 rose across exactly that
  step. v5 adds a small model with head dimension 128 and a mid-size model with head dimension 64, and asks which v4 pattern each
  follows.
- **B. A second quantizer.** Do the value-path harshness (H2), the natural-size ranking (H3) and the common mode (H5) hold when
  the 4-bit weights come from HQQ instead of bitsandbytes NF4?
- **D. A second task.** Do H2 and H5 hold when the history is natural encyclopedic text (WikiText-2), teacher-forced, instead of
  the model's own greedy GSM8K continuation? **The WikiText runs teacher-force the real text, whereas v4 teacher-forced the model's own
  greedy history, so D tests the task and the history source together**: a difference between D and the GSM8K units cannot be
  attributed to either one alone.

**Scope.** Core arms only: FP16 reference, NAT, C1, NAT-K, C1-K, NAT-V, C1-V, plus calibration doses of C1 and the common-mode
statistic (X13a sums). No H4 (one or two models per family; H4 is closed by v4). Novelty wording as before: "to our knowledge",
never "first".

## 2. Models, conditions and data

**Units.** A unit is one (model, quantizer, task) run: gates, calibration, and the test sample.

| Unit | Component | Model (HF id) | Quantizer | Task | n test / cal |
|---|---|---|---|---|---|
| U0 | anchor | `Qwen/Qwen2.5-0.5B` (v4 rev `060db649…`) | NF4 | GSM8K | 100 / 50 |
| U1 | A | `Qwen/Qwen3-0.6B-Base` | NF4 | GSM8K | 100 / 50 |
| U2 | A | `HuggingFaceTB/SmolLM2-1.7B` | NF4 | GSM8K | 100 / 50 |
| U3 | D | `Qwen/Qwen2.5-0.5B` | NF4 | WikiText-2 | 100 / 50 |
| U4 | B | `Qwen/Qwen2.5-0.5B` | HQQ | GSM8K | 100 / 50 |
| U5 | D | `mistralai/Mistral-7B-v0.3` (v4 rev `caa1feb0…`) | NF4 | WikiText-2 | 100 / 50 |
| U6 | B | `mistralai/Mistral-7B-v0.3` | HQQ | GSM8K | 100 / 50 |

**New checkpoints (read 2026-11-11 from the HF API and the pinned `config.json`; G7 re-reads them on the pod).**

| | Qwen3-0.6B-Base | SmolLM2-1.7B |
|---|---|---|
| Revision | `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` | `effd688a12921b4cc83e3312b6feb579f70f9c71` |
| Shard (SHA-256, bytes) | `model.safetensors` `cd2a5120…9223eba`, 1,192,135,096 | `model.safetensors` `11935289…c300c0`, 3,422,777,952 |
| Licence, gating | Apache-2.0, not gated | Apache-2.0, not gated |
| Architecture | `Qwen3ForCausalLM` | `LlamaForCausalLM` |
| Layers, hidden | 28, 1024 | 24, 2048 |
| Q / KV heads | 16 / 8 (GQA) | 32 / 32 (**no GQA**) |
| **Head dimension** | **128** | **64** |
| Params (computed) | 0.60 B (0.44 B in transformer linears) | 1.71 B (1.61 B in transformer linears) |
| Vocab, tied | 151,936, tied | 49,152, tied |
| RoPE base | 1e6 | 1.3e5 |
| Stored dtype | bfloat16 | bfloat16 |
| BOS / EOS ids | 151643 / 151643 | 0 / 0 |

**Stated confounds (carried into the manuscript as L-A).**
- **Qwen3-0.6B-Base.** Qwen3 attention applies an RMSNorm to each head's query and key vectors (QK-norm) before RoPE, so the
  stored key entries are normalized per head and the key-path error passes through that normalization. QK-norm acts directly on
  the key path that H1 and NAT−C1 measure. It is also a newer model generation with different training data. Same vocabulary
  size as Qwen2.5 (151,936). Phase 1 verifies the `q_norm`/`k_norm` tensors in the pinned safetensors header, and G7 checks them.
- **SmolLM2-1.7B.** A different family (Hugging Face), a different tokenizer (49,152 tokens), different training data, full
  multi-head attention (32 KV heads per layer, no GQA, against 2–8 in the v4 models), a different RoPE base,
  and an 8,192-token context. Its architecture class is Llama's.
- Neither model is a size–head-dimension swap *within* one family. A model that differs from Qwen2.5-0.5B only in head dimension
  does not exist among public checkpoints we know of.

**Anchor and the v4 models.** Qwen2.5-0.5B and Mistral-7B-v0.3 use the v4 pins, manifests and shard hashes unchanged.

**Licences (D-V5-5).** Every model above is Apache-2.0. WikiText-2 (`Salesforce/wikitext`, config `wikitext-2-raw-v1`, revision
`b08601e04326c79dfdd32d625aee71d232d685c3`, not gated) is CC BY-SA 3.0 and GFDL. **No released v5 file stores WikiText text or
token ids.** The sample file, the records and the logs of U3 and U5 identify text only by the pinned dataset revision, the split,
the segment (paragraph) index and SHA-256 hashes of each segment, its prompt and its history ids; the pod rebuilds the text from the
pinned parquet files and checks every hash (GW, G5). Two record fields change for the WikiText units only: the per-step argmax
token ids (`top`) are replaced by their SHA-256 (`top_sha256`), and the history record holds `ids_sha256` and the length instead of
the ids. The WikiText-derived numbers (divergences, contrasts, statistics) are released under CC BY 4.0 with the GSM8K records.

**Memory.** Dual-weight memory (v4 construction) is at most about 20 GB (Mistral). HQQ adds a per-group FP16 scale and zero
(4.5 bits/weight) and, with the PYTORCH backend, a transient dequantized FP16 copy of one linear at a time. Every unit fits the A40's
48 GB.

## 3. Protocol, arms and tasks

**Unchanged from v4 §3** (and so from v3 with amendments 1 and 2): teacher forcing; steps s = 1…S_i with S_i = min(64, history
length − 1); FP16 prefill never perturbed; perturbation of the newly written entry only, at write time; JS in float64 with m_i
floored at 1e-9; θ(a, b) = mean_i [log m_i^a − log m_i^b]; amendment 2's float64 construction check and bitwise FP16 rounding
check; exactly-once records and the torn-line rule; the six arms and the C1 construction on the unit sphere of R^head_dim.

**Seeds.** U0 uses the v4 master label `rv4-scale-2026-11-02`, so its C1 draws equal v4's and its records can be compared with
v4 bitwise. Every other unit uses the new master label `rv5-2026-11-11`, with the quantizer and the task in the label.

**GSM8K (U0, U1, U2, U4, U6).** Exactly v4: the first 100 of the seeded 200 test ids, the first 50 of the 100 calibration ids,
the raw 3-shot prompt, each model's in-session FP16 greedy history (≤ 65 tokens, drift-marker stop).

**WikiText-2 (U3, U5): teacher-forced natural-text continuation.**
- *Source:* `wikitext-2-raw-v1` parquet files at the pinned revision (test split for the sample, train split for calibration and
  timing). The raw text is used as is, including its ` @-@ `-style tokenization artefacts.
- *Paragraph:* a line that is non-empty after stripping and is not a heading (` = … = `).
- *Eligible:* at least 1,200 characters, and at least 65 continuation tokens (below) under the tokenizers of every D model
  (Qwen2.5 and Mistral). Eligibility is computed on the laptop before the lock.
- *Order:* eligible test paragraphs are ordered by SHA-256 of `rv5-wikitext2-test-order-2026-11-11|<paragraph index>`; the first
  100 are the test sample. Train paragraphs, ordered the same way with their own label: the first 50 are calibration, the next 3
  are timing problems. The sample (indices and text hashes) is hashed into the lock.
- *Prompt:* the paragraph's first 600 characters, cut back to the last whitespace, with no template; BOS handling per tokenizer as
  in v4 (G5).
- *History:* the first 65 tokens of the remaining text (with its leading space), tokenized separately with no special tokens.
  The history is human text, not the model's own generation. It is identical in every arm of a model. **This is the second change
  besides the task:** v4 (and the GSM8K units here) teacher-force the model's own FP16 greedy history, so D tests task and history
  source together (§1).
- *Outcome:* identical in form to v4: per step, JS between the arm's next-token distribution and the pure-FP16 teacher-forced
  pass; m_i is the mean over steps 1…64 (S_i is always 64 here, against a v4 mean of 60–64); contrasts are θ(a, b).

**HQQ (U4, U6).** The v4 r3 implementation (`v4/rv4_hqq.py` at commit `f3a54cc`, removed by v4 r5), restored unchanged:
- `hqq==0.2.8.post1`, installed with `DISABLE_CUDA=1` and `--no-deps` (plus `einops`, `termcolor`), so the pinned torch stack is
  untouched;
- `BaseQuantizeConfig` defaults passed explicitly: nbits 4, group_size 64, axis 1; quant_zero and quant_scale False; the default
  proximal optimizer; `HQQBackend.PYTORCH` (dequantize, then matmul). **Nothing is tuned;**
- the measurement pass uses HQQ weights everywhere, exactly where v4 used NF4. Δ = κ^HQQ − κ^FP16, and C1, C1-K and C1-V take
  their per-slot norms from HQQ's own Δ. H2 and H3 under HQQ are therefore norm-matched within the quantizer;
- determinism and freezing (v4 r3-7): the model is quantized twice on the pod. If the two are bitwise identical, every load
  re-quantizes and checks the hash. If not, the first state is saved, and its hash goes into the lock appendix before that unit's
  calibration, and every later load reads it;
- 4.5 bits/weight against NF4's ≈ 4.13: any cross-quantizer comparison of raw divergence is descriptive and not like-for-like.
  No test compares quantizers.

**Calibration** per unit, by the v4 rules (C1 at α ∈ {0.5, 1, 2}, 50 problems, 64 steps): b̂, c, margin = |b̂|·ln(1 + c),
ρ_K, ρ_V. The HQQ units calibrate on HQQ's Δ, and the WikiText units on WikiText train paragraphs. Each calibration is frozen
and appended to the lock before the unit's first test problem.

## 4. Anchor (Step-0 style; U0 first)

U0 reruns Qwen2.5-0.5B (NF4, GSM8K) with the v4 pins, seed label and sample, on an A40 again.
- **Reported, not gated:** the share of U0 records bitwise equal to the v4 records. Equality is expected if the host's driver and
  cuBLAS behave as in v4, but a different host or driver can change kernels.
- **Gated (the v4 Step 0 criteria, with v4 as the reference instead of v3):**
  1. for θ(NAT, C1), θ(NAT-K, C1-K), θ(NAT-V, C1-V) and θ(NAT-K, NAT-V): δ = θ_v5 − θ_v4, paired by problem; the 90 % CI of δ
     lies within ±(the v4 margin, 0.2508) and |δ̂| ≤ 0.10;
  2. every contrast with |θ_v4| ≥ 0.2 keeps its v4 verdict category;
  3. ρ_K and ρ_V within ±5 % relative, and b̂ within ±0.25, of v4;
  4. H5: L^K and L^V within ±0.10 of v4 (reported, not gated).
- **Decision:** pass ⇒ continue to the design point (§9). Fail ⇒ stop and report; the authors decide whether to proceed with
  every comparison to v4 labelled "not agreed", or stop.

The anchor also makes U3 and U4 interpretable: they share model, pins and hardware with U0 and differ from it only in task or
quantizer.

## 5. Gates (per unit; any failure ⇒ stop and report; nothing relaxed)

v4's GM and G1–G8 unchanged, per unit, with these additions:

| Gate | Addition |
|---|---|
| GM | Restricted to the v5 arms, plus the v5 mutants (§11). Runs once, before U0. Logs driver, CUDA, ECC, deterministic settings and the image record, as v4 |
| G1 | HQQ units: every constructed arm at scale 0 equals FP16 bitwise |
| G2 | HQQ units: the float64 construction and bitwise rounding checks on HQQ's Δ |
| G3 | Finite values in FP16; the v4 BF16 rule (D6) applies unchanged, with reporting with and without the flagged unit |
| G5 | SmolLM2's tokenizer (BOS = EOS = 0) and Qwen3's: prompt round-trip, BOS/EOS handling, drift-marker detection (GSM8K). WikiText: every history has 65 tokens and equals the pre-computed hash |
| G7 | Geometry from the pinned snapshot equals §2; Qwen3: the `q_norm`/`k_norm` tensors exist; SmolLM2: 32 KV heads; snapshot and shard SHA-256 equal the lock |
| GH (new) | HQQ: two quantizations bitwise equal, or the frozen state's hash in the lock appendix; an independent dequantization oracle (unpack 4-bit, W = (W_q − z)·s per group of 64 along axis 1) equals `HQQLinear.dequantize()` bitwise; the installed version is 0.2.8.post1 |
| GW (new) | WikiText: the sample file's hash equals the lock; the parquet SHA-256 equals the pinned revision's |

**Deterministic settings** (v4 r4), the **hardware stop** at 88 °C, and the **per-problem integrity** rules are unchanged.

## 6. Outcomes

As v4 §6: per-problem m_i^a and contrasts θ(a, b); the H5 statistic x_i^P = ln(n_i C_i^P) − ν_i^P (empirical isotropic null of
200 seeded draws in R^d); ρ_K, ρ_V, ρ_V/ρ_K.

## 7. Hypotheses, decision rules, falsification

**Common rules (v4 §7, unchanged):** difference if the 95 % CI excludes 0; equivalent if the 90 % CI lies within ±margin and
b̂ ≥ 0.25; inconclusive otherwise; equivalence never from non-significance; two-sided paired sign-flip permutation p (10,000);
problem-clustered percentile bootstrap CIs (10,000, a new locked analysis seed). A per-model test is **supported** if its CI lies
entirely on the predicted side and its Holm p < 0.05; **falsified** if the CI lies entirely on the other side or the contrast is
equivalent to 0. "In every tested model" as in v4: supported only if every model's test is supported; falsified if any model
falsifies it.

### 7.1 A: the v4 claims in the two new models (U1, U2)

| Test | Prediction | Holm family |
|---|---|---|
| A-H1 | θ(NAT-K, C1-K) < 0 | over U1, U2 (2) |
| A-H2 | θ(NAT-V, C1-V) > 0 | over U1, U2 (2) |
| A-H3 | θ(NAT-K, NAT-V) < 0 | over U1, U2 (2) |
| A-H5 | L^P above isotropic: 95 % lower bound > ln 4 (falsified if the upper bound < ln 2) | over U1, U2 × K, V (4) |

### 7.2 A-X: the crossed reading (stated before data)

v4 contains two patterns that size and head dimension produce together. v5 asks which one each new model follows. The reference
values are the **locked v4 estimates** (`v4_runs/v4_report.json`, sha256 `df57a314…dccb`). Each reference value,
with its source model (D-V5-4, midpoint rule accepted):

| Source model (v4) | head_dim | NAT−C1 | H1 |
|---|---|---|---|
| Qwen2.5-0.5B | 64 | −0.2233 | −1.2901 |
| Qwen2.5-1.5B | 128 | +0.2036 | −0.9840 |
| Qwen2.5-3B | 128 | +0.3005 | −0.8513 |
| Llama-3.2-1B | 64 | +0.0230 | −0.9461 |
| Llama-3.2-3B | 128 | +0.4228 | −0.4350 |

The predictions and boundaries:

| New model | Contrast | "Size drives" predicts (v4 reference) | "Head dimension drives" predicts (v4 reference) | Boundary b |
|---|---|---|---|---|
| Qwen3-0.6B (0.6 B, hd 128) | NAT−C1 | −0.223 (Qwen2.5-0.5B: same size class) | +0.204 (Qwen2.5-1.5B: nearest hd-128 Qwen) | **−0.010** |
| Qwen3-0.6B | H1 | −1.290 (Qwen2.5-0.5B) | −0.984 (Qwen2.5-1.5B) | **−1.137** |
| SmolLM2-1.7B (1.7 B, hd 64) | NAT−C1 | +0.309 (mean of the hd-128 models of 1.5–3.2 B: Qwen2.5-1.5B, Qwen2.5-3B, Llama-3.2-3B) | −0.100 (mean of the two hd-64 models) | **+0.104** |
| SmolLM2-1.7B | H1 | −0.757 (same three models) | −1.118 (the two hd-64 models) | **−0.937** |

The boundary is the midpoint of the two predictions. The references are fixed by these rules, written before any v5 data.
- **Per model and contrast:** "**head-dimension pattern**" if the 95 % CI lies entirely on the head-dimension side of b;
  "**size pattern**" if it lies entirely on the size side; "**unresolved**" otherwise. A one-sided sign-flip test of θ − b in the
  direction of the observed side gives a p-value, with Holm over the two models for each contrast (families AX-N for NAT−C1, the
  primary, and AX-1 for H1, secondary). A classification needs both the CI condition and Holm p < 0.05.
- **Joint reading (NAT−C1, the primary contrast):**
  - both models follow the head-dimension pattern ⇒ "in these checkpoints the v4 size trend tracks head dimension";
  - both follow the size pattern ⇒ "it tracks size";
  - one each, or any unresolved ⇒ "neither factor alone accounts for it in these checkpoints".
- **SmolLM2's references come from other families.** Every reference value for SmolLM2 is a Qwen2.5 or Llama-3.2 estimate; no
  SmolLM2 model is in v4. A SmolLM2 reading on the "size" side is therefore also consistent with a **family offset** (SmolLM2's own
  level being higher than the Qwen/Llama hd-64 models for reasons other than size), and a reading on the "head-dimension" side is
  likewise consistent with an offset in the other direction. The manuscript states this next to every SmolLM2 reading.
- **What a reading can mean.** None of these readings is causal. Qwen3's QK-norm and generation, and SmolLM2's family, tokenizer
  and full multi-head attention, are alternative explanations of any departure from either pattern (L-A). The manuscript wording
  is fixed now: "consistent with … in these checkpoints".
- **Directional predictions are not made.** The authors do not predict which pattern wins. Both patterns are stated, so neither
  reading can be claimed as confirmed.

### 7.3 B: HQQ (U4, U6)

| Test | Prediction | Holm family |
|---|---|---|
| B-H2 | θ(NAT-V, C1-V) > 0 under HQQ | over U4, U6 (2) |
| B-H3 | θ(NAT-K, NAT-V) < 0 under HQQ | over U4, U6 (2) |
| B-H5 | L^K and L^V above isotropic under HQQ | over U4, U6 × K, V (4) |

H1 and NAT−C1 under HQQ are reported descriptively, with no prediction. ρ_V/ρ_K under HQQ is reported next to B-H3.

### 7.4 D: WikiText-2 (U3, U5)

| Test | Prediction | Holm family |
|---|---|---|
| D-H2 | θ(NAT-V, C1-V) > 0 on WikiText-2 | over U3, U5 (2) |
| D-H5 | L^K and L^V above isotropic on WikiText-2 | over U3, U5 × K, V (4) |

H1, H3 and NAT−C1 on WikiText-2 are reported descriptively, with no prediction.

### 7.5 Power (n = 100; n = 60 after reduction R1)

- **Assumption:** the largest per-problem SD of any v4 contrast in any model, backed out of the v4 95 % CIs: 0.64 (range
  0.39–0.64). For H5: 0.30 (v4 half-widths about 0.05).
- **MDE** at 80 % power, two-sided, Holm worst case (Bonferroni):

  | | m = 1 | m = 2 | m = 4 (H5 families) |
  |---|---|---|---|
  | n = 100, SD 0.64 | 0.179 | 0.197 | |
  | n = 60, SD 0.64 | 0.231 | 0.255 | |
  | n = 100, H5 (SD 0.30) | | | 0.100 |
  | n = 60, H5 (SD 0.30) | | | 0.129 |

- **Against the v4 effects in the same or comparable models:** H2 +0.876 (Qwen2.5-0.5B) to +1.174 (Mistral); H3 −0.953 to
  −1.112; v4's L^P 2.32–2.67 against the support level of about ln 4 + 0.06 = 1.45. Each is 3.4× or more the n = 60 MDE. A
  failure under HQQ or WikiText would therefore show up as a real change of sign or size, not as power loss. Effects that shrink
  to about 0.2–0.26 are no longer reliably detected.
- **A-H1 in the new models:** v4's hd-64 models had H1 of −0.95 to −1.29; the hd-128 models of comparable size −0.43 to −0.98.
  Both are above 2× the MDE. Mistral-like attenuation (−0.124 in v4) would not be resolved, as in v4.
- **A-X classification:** if the true value equals one pattern's reference, the probability that the 95 % CI lies entirely on its
  side of the boundary (SD at its v4 maximum) is **0.96 (Qwen3) and 0.94 (SmolLM2) for NAT−C1**, and 0.67 and 0.81 for H1. A true
  value between the two references is likely to read "unresolved", which is itself informative.

## 8. Descriptive and sensitivity analyses; independent recomputation

- **Descriptive:** all four θ contrasts, L^K and L^V, ρ_K, ρ_V and C^P per layer, per unit; the U0–U3–U4 and U5–U6 comparisons
  (task and quantizer on the same model), labelled descriptive and, for quantizers, not like-for-like.
- **Sensitivity** (v4 §8 set): exposed GSM8K test ids excluded (GSM8K units only); steps 2–8 and 9–64; per-problem medians;
  amendment 2's truncation sensitivity.
- **Independent recomputation** (v4 §8): a script with no project imports recomputes every estimate, CI, Holm decision, the A-X
  classifications and the H5 statistic from the raw records. **`RECOMPUTATION_TOLERANCES_V5.md` is written and locked before any
  data** (point estimates exact to 1e-12; CI bounds within the v4 Monte-Carlo tolerance; every verdict and classification must
  agree; any disagreement is reported as unresolved).
- **No analysis of test records until every unit has finished** (the v4 rule). During the run only gates, calibration constants,
  the anchor decision, the design point, GPU-hours, USD and integrity events are reported.

## 9. GPU, time, cost, ceiling and reductions

**GPU (D-V5-8).** One A40 48 GB, Secure Cloud, on demand ($0.59/h in v4; re-read before creation). **Datacenter rule:** prefer a
datacenter that has A40 stock and Standard network volumes ($0.07/GB-month); if none has, CA-MTL-1 with a HIGH_PERFORMANCE volume
($0.14/GB-month), as in v4. A40 stock is re-read with the RunPod MCP at the lock (recorded in the runbook) and again immediately
before anything is created, and the price is stated to the author first. The volume tier's rate goes into `cost_config.json`
(`storage_usd_per_gb_month`), so the ledger uses the tier actually created; the projections below use the HIGH_PERFORMANCE rate (the
larger). **No SKU change mid-study** (v4 D5): no A40, then wait or stop and report.

**Projection model, with the v4 factor built in.**
- v4 models (Qwen2.5-0.5B, Mistral-7B-v0.3): the **measured** v4 GPU-ledger hours per model (gates + calibration + test): 1.20 h
  and 2.25 h. These are the real times that D1's factor of 1.48 corrected the v4 G6 fit towards.
- New models: the v4 G6 fit (`design.json`: 0.1147 s per layer + 4.349e-11 s per byte, history ratio 0.762) **× 1.48**, plus
  gates. Check on v4's own models: fit × 1.48 gives 1.22 / 1.48 / 1.81 / 1.99 h against measured 1.20 / 1.42 / 1.82 / 2.25 h
  (Qwen2.5-0.5B / 1.5B / 7B / Mistral). Result: Qwen3-0.6B-Base 1.42 h, SmolLM2-1.7B 1.31 h.
- WikiText units: × 64/61.1 (every history reaches 64 steps; v4's GSM8K mean was 60.4–63.9).
- HQQ units: × r, the HQQ/NF4 time ratio, **unknown**. HQQ's PYTORCH backend dequantizes each linear to FP16 before every matmul,
  so the measurement pass reads and writes far more bytes than NF4's 4-bit kernel. Scenarios r = 1.5, 2 and 3. The local
  rehearsal measures r on Qwen2.5-0.5B on the RTX 4060 (reported, used as a prior), and the design point measures it on the pod.
- GM 0.40 h; billed pod hours = GPU-ledger hours × 1.222 (v4's measured ratio, 19.07 billed / 15.60 ledger) + 0.5 h first boot
  + 0.25 h per distinct model, + 5 %; container disk 30 GB; storage on a 50 GB HIGH_PERFORMANCE volume at $0.14/GB-month,
  prorated over the billed hours plus 48 h with no pod.

**Projection (USD, total incl. storage and disk; the design that fits is chosen at the design point):**

| Design | GPU-h | Billed h | r = 1.5 | r = 2 | r = 3 |
|---|---|---|---|---|---|
| **Core, n = 100** (U0–U6) | 13.1 / 14.8 / 18.3 | 18.4 / 20.6 / 25.0 | **$11.59** | **$12.92** | $15.58 (over) |
| R1: B and D at n = 60 | 10.9 / 12.2 / 14.8 | 15.6 / 17.2 / 20.5 | $9.87 | $10.86 | **$12.88** |
| R1 + R2 (no U6) | 8.4 / 8.8 / 9.7 | 12.3 / 12.9 / 14.0 | $7.91 | $8.25 | $8.94 |
| R1 + R2 + R3 (no U5, U6) | 6.6 / 7.1 / 7.9 | 9.8 / 10.4 / 11.5 | $6.37 | $6.72 | $7.42 |
| For comparison: B and D on three models (+ Qwen2.5-7B), n = 100 | 17.8 / 20.4 / 25.6 | | $15.33 (over) | $17.38 (over) | $21.44 (over) |

Per unit at r = 2, n = 100 (GPU-h): U0 1.20, U1 1.42, U2 1.31, U3 1.25, U4 2.40, U5 2.35, U6 4.49.

- **Why two models for B and D, not three.** A third model (Qwen2.5-7B, adding within-family size) is over $15 at every r with
  n = 100, and $14.29 at r = 2 with n = 60, too close to the $14 stop. Qwen2.5-0.5B and Mistral-7B-v0.3 span the size range
  (0.5 B to 7.2 B) and two families, reuse the anchor and v4 pins, and need no new download beyond Mistral.
- **Hard ceiling $15 in total, storage and disk included. Ledger stop at $14**, leaving $1 for close-out (last download,
  storage until deletion). Enforced as in v4 by the cost ledger (in-pod estimate and billing lines; the larger is used), a
  per-problem start check and a watchdog. The balance is not a hard stop (§0).
- **Design point** (after U0 passes): measure on the pod Qwen2.5-0.5B NF4 (from U0), a 3-problem HQQ probe on Qwen2.5-0.5B, and a
  3-problem NF4 and HQQ probe on Mistral-7B-v0.3; refit; **multiply by 1.48** as above; project the remaining units in USD with
  storage. If the projection exceeds the money left below the $14 stop, the **reduction order** applies, in this order, until
  it fits:
  1. **R1:** B and D units at n = 60 (the first 60 of the same samples; calibration stays 50);
  2. **R2:** drop U6 (Mistral HQQ). B is then one model, reported as such;
  3. **R3:** drop U5 (Mistral WikiText). D is then one model;
  4. **R4:** stop and report.

  **A (U1, U2) and the anchor (U0) are never cut.** The design is written to the lock appendix before U1 and is not revisited.
- **Drift stop.** The drift reference of every unit is its design-point projection (which already includes the 1.48 factor);
  a per-unit drift of more than 25 % above it stops and reports. A unit also starts only if its projected cost fits under the
  $14 stop; otherwise it is recorded as not run, and the study closes out.
- **Run order** (grouped by model, so each model is downloaded once; the cut units come last):
  GM → U0 (anchor) → anchor decision → design point → U1 → U2 → U3 → U4 → U5 → U6.
- **Volume: 50 GB.** Downloads 20.1 GB (Qwen2.5-0.5B 1.0, Qwen3-0.6B-Base 1.2, SmolLM2-1.7B 3.4, Mistral HF shards 14.5); venv
  about 12 GB (v4 assumption, plus hqq); frozen HQQ states up to about 4.3 GB (Mistral at 4.5 bits/weight); records about 0.5 GB;
  WikiText under 0.1 GB. About 37 GB, so 50 GB leaves about 35 % headroom. The volume is deleted by the author right after the
  last verified download.
- **Truncation:** a unit with fewer than its design n is unresolved.

## 10. Cloud operations

- **Pod lifecycle** through the official RunPod MCP, every create, start, stop or delete approved by the corresponding author in
  the permission prompt (v4 A3, D8). Network-volume deletion stays denied in the tool settings; the author deletes the volume.
  The price is stated before anything billable is created. Only resources created for v5 are touched.
- **No secrets.** Every model and the dataset are ungated. The pod gets no HF token (`HF_HUB_DISABLE_IMPLICIT_TOKEN=1`); a
  start-up check refuses to run if `HF_TOKEN` is set or a token file exists. The secret scan still runs before every download and
  commit.
- **Authorization files**, one per unit plus GM: `v5_runs/AUTHORIZED_V5_<unit>.txt` naming `PROTOCOL_V5`, the lock id and the unit.
  They are written only at the corresponding author's explicit instruction, and each writing is logged in `DEVIATION_LOG_V5.md`.
- **On-pod driver from the start** (v4's prepared fallback made primary): `run_v5_pod.sh`, launched once through the remote-exec
  tool with `setsid -f nohup bash … > log 2>&1 < /dev/null` (the form v4 verified to detach; a plain `setsid nohup … &` does not).
  It holds a `flock`, refuses to start if any study GPU process is running, runs the units in order with every v4 stop rule (gate,
  drift, $14 ledger stop, resume check), checkpoints each unit (tar + `SHA256SUMS` + secret scan) and writes `status.json`. It
  never calls RunPod and never writes an authorization file. An SSH drop or laptop sleep no longer stops the run.
- **Laptop watcher:** polls every 5 minutes, fetches and verifies each checkpointed unit (`sha256sum -c`), and exits with one line
  on done, a protocol stop, the driver gone, or the pod unreachable for 3 polls. On a protocol stop or the end, the pod is stopped
  through the MCP (approved by the author) and the cost is reported.
- **Resume** (v4 rules): lock check, open progress file charged, resume check (rerun the last completed problem, bitwise), then
  continue.
- **Standing local-rehearsal rule (RTX 4060).** Before any pod run, in a clean clone of the locked commit on the laptop:
  - the full test suite and the CUDA mutation check;
  - a 3-problem end-to-end run of every unit type that fits 8 GB: U0, U1, U2, U3 and U4 (Qwen2.5-0.5B HQQ). Mistral units are
    covered by the CPU 2-step test only;
  - the on-pod driver and the watcher, run locally against a local "pod" directory, including a forced stop and a resume;
  - the HQQ/NF4 time ratio r measured on Qwen2.5-0.5B and reported.

## 11. Implementation (Phase 3, planned; nothing exists yet)

- **New modules only:** `v5/rv5.py` (task layer for WikiText, the HQQ path, the new geometries), `v5/run_v5.py`,
  `v5/run_v5_pod.sh`, `v5/watch_v5.sh`, `v5/analyze_v5.py`, `v5/recompute_v5_independent.py`, `tools/lock_v5.py`; `v5/rv5_hqq.py`
  restored byte-for-byte from `f3a54cc:research_v2/v4/rv4_hqq.py` with its tests. Locked v4 code (`rv4`, `v4_ops`, the X13a sums)
  is imported unchanged and hash-checked.
- **Tests:** tiny Qwen3 (with QK-norm) and tiny Llama-MHA (32 KV heads, head dim 64) configs; the real-model CPU checks for
  Qwen3-0.6B-Base and SmolLM2-1.7B; the HQQ dequantization oracle and determinism; the WikiText sampler (eligibility, order,
  prompt cut, 65-token history, hash); the A-X classification on simulated truth (each pattern, between, and both sides); the
  reduction order; the ceiling and stop at $15/$14; the driver/watcher harness.
- **Mutants (all must be caught),** in addition to v4's restricted to the core arms: C1 on the 64-sphere for Qwen3 (hd 128) or on
  the 128-sphere for SmolLM2 (hd 64); slot norms over the KV heads (SmolLM2's 32); HQQ with group size 128, axis 0, or a tuned
  setting; C1-HQQ norms taken from NF4's Δ; HQQ dequantization skipped (FP16 used as "HQQ"); a WikiText history that is the
  model's own generation; a prompt cut mid-word; the paragraph order not seeded; the anchor with the v5 seed label; an A-X boundary
  not from the locked v4 report; A-X classified by point estimate only; Holm over the wrong family; U1 or U2 cut by a
  reduction; the $14 stop ignored; the start check after the problem; the driver starting while another runs; the watcher
  accepting an unverified checkpoint; an HF token accepted on the pod.
- **Lock manifest:** this protocol, the code, the tests, the GSM8K sample (v4's), the WikiText sample, the planning JSON,
  `RECOMPUTATION_TOLERANCES_V5.md`, the pinned revisions and shard hashes, and the package list. Lock file
  `research_v2/V5_PROTOCOL_LOCK.txt`.

## 12. Records, deviations, AI use, limitations

- **Records:** `v5_runs/<unit>/{calib,test}/<arm>.jsonl` (v4 schema plus `task` and `quantizer` fields), calibration, gate logs,
  manifest; ledgers; HQQ state hashes.
- **Deviations:** `v5/DEVIATION_LOG_V5.md`. Only the pre-declared reductions, the G3 rule and the anchor decision may change what is
  tested.
- **AI use:** `v5/AI_USE_LOG_V5.md`. This draft: Claude Opus 5.5 (`claude-opus-5-5`) in Claude Code, 2026-11-11.
- **Limitations, carried into the manuscript:**
  - **L-A:** the crossed design uses checkpoints that differ in more than head dimension (§2): QK-norm and generation (Qwen3),
    family, tokenizer, data and attention layout (SmolLM2). Readings are patterns, not causes.
  - **L-B:** one alternative quantizer (HQQ, defaults, PYTORCH backend) at 4.5 bits/weight; at most two models; no
    like-for-like quantizer comparison.
  - **L-D:** one alternative task (WikiText-2), at most two models; the history is human text, teacher-forced, whereas v4 and the
    GSM8K units teacher-force the model's own greedy history. D therefore tests the task and the history source together, and a D
    result cannot be attributed to either alone.
  - **L-R:** if R1–R3 apply, the reduced n and model counts.
  - v4's L1, L4–L7 carry over (teacher-forced label-free outcome, cross-hardware, no chat template, reused GSM8K problems,
    private lock and exploratory origin).

## 13. How v5 enters the paper

- **Proposal: one manuscript, with v5 as a second locked study.** v4 stays the main result, unchanged. A new section after the v4
  results, "Disentangling size and head dimension, a second quantizer and a second task (v5)", reports v5 with its own lock id,
  deviation record and recomputation, generated macros (`make_v5.py`, `tables_v5/numbers_v5.tex`) and the same number audit.
- **What changes in the existing text, by outcome:**
  - **Abstract and conclusions:** the sentence that size and head dimension are confounded is replaced by the A-X reading
    ("consistent with head dimension / size / neither in these checkpoints"), with the L-A qualifier. B and D add one sentence:
    "H2 and H5 held / did not hold under HQQ and on WikiText-2 in the two models tested". The WikiText sentence says that the
    WikiText runs teacher-force the real text, whereas v4 teacher-forced the model's own greedy history, so that result covers the
    task and the history source together.
  - **Limitations:** v4's L3 (size and head dimension) and L8 (one quantizer) are rewritten around the v5 results and L-A, L-B,
    L-D; a task limitation is added.
  - **Discussion:** the key-path interpretation (H1 and NAT−C1) is revisited with A-X.
  - The title stays unless v5 contradicts its claim; a B or D failure of H2 would make "consistently harsher" need a qualifier
    (D-V5-6).
- **Reporting rule:** v5 predictions are exploratory in origin (from v4) and locked before v5 data; the manuscript says so, and
  says that v5 was designed after the v4 results were known.
- **Timeline:** about 1 day of implementation and rehearsal, about 21 billed pod hours (about a day of running), then analysis,
  recomputation and manuscript integration, with a STOP after each phase.

## 14. Phases (a STOP after each)

1. Draft r1 → author decisions → r2 (this text). Done 2026-11-11.
2. Literature and facts check from primary sources: Qwen3 QK-norm in the pinned weights, SmolLM2 data and tokenizer facts,
   WikiText-2 licence and format, HQQ 0.2.8.post1 still installable → STOP.
3. CPU implementation, validation, tolerances file, lock; local rehearsal on the RTX 4060 → STOP. ← *current* (the author asked on
   2026-11-11 for Phases 2 and 3 together, ending with the lock id and the RunPod setup steps; no cloud action).
4. GM and U0 → anchor decision → design point → STOP.
5. U1–U6 under the reporting rule → STOP.
6. Locked analysis, independent recomputation, report → STOP.
7. Manuscript integration → STOP.

## 15. Decisions of the corresponding author (2026-11-11, recorded in r2)

| Decision | Outcome |
|---|---|
| D-V5-1, B models | **Accepted:** Qwen2.5-0.5B and Mistral-7B-v0.3. |
| D-V5-2, D models | **Accepted:** the same pair. |
| D-V5-3, reduction order | **Accepted:** R1 (n = 60 for B and D) before dropping a model; then R2, R3, R4 (§9). |
| D-V5-4, A-X references | **Accepted:** the midpoint rule; each reference value listed with its source model; for SmolLM2 the references come from other families, so a "size" reading is also consistent with a family offset (§7.2). |
| D-V5-5, WikiText records | **No WikiText text or token ids in released files;** segment indices, SHA-256 per segment and the pinned dataset revision; WikiText-derived numbers under CC BY 4.0 (§2). |
| D-V5-6, manuscript route | **Accepted:** v5 as a second locked study in the same manuscript (§13). |
| D-V5-7, balance | **Balance stays at $17.87; the $14 ledger stop is the limit** (§0, §9). |
| D-V5-8, datacenter | **Re-check A40 stock; prefer A40 + Standard volume, else CA-MTL-1 HIGH_PERFORMANCE** (§9). |
| Added | The history-source statement for D, in this protocol and later in the paper (§1, §3, §12, §13). |

## 16. Implementation notes (Phase 3, 2026-11-11, before the lock)

These fix details the text above leaves open. They change no hypothesis, unit, arm, sample size, ceiling or decision rule.

1. **Phase 2 facts, from pinned bytes** (`tools/v5_build_manifest.py`, `v5/V5_MODEL_MANIFEST.json`):
   - Qwen3-0.6B-Base has 28 `q_norm` and 28 `k_norm` tensors (QK-norm in every layer); 0.596 B parameters, 0.440 B in transformer linears.
   - SmolLM2-1.7B: 32/32 heads, head_dim 64 (no GQA); 1.711 B parameters, 1.611 B in linears.
   - Both are Apache-2.0 and not gated, and their tokenizers do not prepend BOS (Qwen2.5: no BOS; Mistral: BOS).
   - The local shard SHA-256 of both equals the HF-reported hash at the pinned revision.
   - WikiText-2 parquet at the pinned revision: test `5f1bea06…`, train `e83889ba…` (SHA-256); card licence CC BY-SA 3.0 and GFDL.
   - `hqq 0.2.8.post1` installs and runs (laptop venv). The v4 entries of Qwen2.5-0.5B and Mistral are copied unchanged.
2. **WikiText rules** (`rv5.py`):
   - a paragraph is a parquet row that is non-empty after `strip()` and not a heading; its text is the stripped row;
   - the prompt cut is the last whitespace character at or before position 600, and any whitespace just before it moves to the continuation;
   - the sample (`v5/v5_wikitext_sample.json`, built by `run_v5.py plan`) has 177 eligible test and 1,446 eligible train paragraphs, of which
     it takes the first 100 test, then 50 calibration and 3 timing paragraphs in the seeded orders;
   - it stores row indices, lengths and SHA-256 hashes only (text, prompt, continuation, prompt ids and history ids per D model).
3. **Record form (D-V5-5):** for U3 and U5, `top` is replaced by `top_sha256` and the history record by `{index, n, ids_sha256}`. Every record carries
   `unit`, `quantizer` and `task`. GW and G5 rebuild every sample paragraph on the pod and compare all its hashes.
4. **Tags and authorization:** `GM`, `U0`, `DESIGN` (Mistral download, the four probes, the design), `U1`…`U6`. Downloads are per model and per
   dataset, recorded in `v5_runs/_downloads/`.
5. **Order:** GM → U0 → anchor (computed on the laptop, `analyze_v5.py anchor`; `ANCHOR:` line) → DESIGN (`DESIGN:` line) → U1 … U6. U0 runs
   before the design, so it has no drift reference, as v4's Step 0 models had none; its cost is bounded by the ledger stop.
6. **GH (HQQ units):**
   - `hqq.json` holds the version, the configuration, both state hashes, the frozen flag, bits per weight and the oracle result;
   - `HQQ <unit>:` is appended to the lock before calibration;
   - every later load re-quantizes (or reads the frozen state) and must reproduce the recorded hash, else `hqq_state_failure.json` and stop.
   - Under the BF16 rule GH is redone in BF16 before G3 is repeated.
   - The frozen state file stays on the volume; checkpoints carry its hash, not the file.
7. **Design point:**
   - probes `qwen-nf4`, `qwen-hqq`, `mistral-nf4`, `mistral-hqq`: 3 timing problems (v4's G6 ids), six arms, no records kept;
   - the v4 G6 fit from the two NF4 probes projects U1 and U2, and the probes give U3–U6 directly; every unit's time is × 1.48, and the
     WikiText units are also × 64/61.1;
   - gates take 0.35 h plus two HQQ quantizations for HQQ units; billed = (GPU h × 1.222 + 0.25 h per new model) × 1.05; storage is prorated
     at the cost config's tier;
   - the reduction steps are none → R1 → R1+R2 → R1+R2+R3 → R4 (stop);
   - U0's measured six-arm time per problem is reported next to the probe's, × 1.48.
8. **Unit start check:** at a unit's download and gates, if used + projected cost > $14, that unit and every later one are recorded as not run,
   `STUDY_STOPPED.json` is written, and the study closes out. Units cut by the design never start.
9. **Operations:**
   - one study GPU process at a time (`gpu_process.lock`, O_EXCL, a dead holder taken over and logged);
   - the on-pod driver runs one phase (`anchor`, `design`, `units`), holds an atomic lock directory, refuses while a GPU process runs, never calls
     RunPod and never writes an authorization file;
   - the watcher exits on done, failure, protocol stop, driver gone, a failed fetch, or an unreachable pod;
   - `tools/v5_remote.py` has a local-pod mode used only by the rehearsal.
10. **Rehearsal mode:** `V5_REHEARSAL=1` (laptop only) is refused on an A40 and in `research_v2/v5_runs`, and needs authorization files that carry
    the word `REHEARSAL`, which a real run refuses; `V5_REHEARSAL_N` sets n. Probe files written as rehearsal stand-ins (Mistral does not fit
    in 8 GB) are refused by a real design.
11. **Analysis:**
    - each summary uses its own stream `default_rng([20261111, key...])`;
    - the A-X one-sided sign-flip test uses 10,000 flips in the direction of the observed mean of θ − b;
    - the references come from the locked v4 report, whose SHA-256 is checked;
    - a unit below its design n is unresolved and leaves its families;
    - the anchor's bitwise share compares every v4 record field except wall time, the NAT slot arrays and the histories.
12. **D-V5-8 at the lock:** A40 Secure stock only in CA-MTL-1 (LOW), which offers HIGH_PERFORMANCE volumes only; no A40 + Standard
    datacenter exists. The fallback applies (CA-MTL-1, HIGH_PERFORMANCE, $0.14/GB-month), re-read before creation (`cloud/RUNBOOK_V5.md`).
13. **Amendment r3 (pre-data, 2026-11-11, found by the rehearsal):** GM's environment record read the CUDA version only from an `nvidia-smi`
    banner of the form `CUDA Version: x.y`. Driver 621.14 on the rehearsal laptop prints `CUDA UMD Version: 13.5`, so GM failed on that item
    alone (16/16 tests and 66/66 mutants passed on CUDA). The A40 hosts now report CUDA 13.2 drivers and may print the new form. `v5_ops.py` reads
    both forms; test o21 and one mutant guard it. Nothing else changes.
14. **Amendment r4 (pre-data, 2026-11-11, found by the rehearsal):** the laptop watcher took the previous phase's `done` in the on-pod
    `status.json` as the end of the new phase, and a watcher restarted while a relaunched driver had not yet rewritten `status.json` could read a
    dead driver's `running` as "driver gone". `watch_v5.sh` now takes the phase as a second argument and counts only that phase's status (exit 6
    when its driver has not started after 3 polls), and confirms "driver gone" with a second look before exit 2; `run_v5_pod.sh` writes
    `starting` as soon as it holds the driver lock. Test o22 and one mutant guard it. Science, units, samples, gates and ceiling unchanged.
