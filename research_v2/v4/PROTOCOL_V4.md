# PROTOCOL_V4: scale study of weight-induced cache error on rented cloud GPUs (r8)

Deviation D1, 2026-11-08 (post-data, corresponding author's decision): drift reference = G6 projection x 1.48; resume check; GM again (§15, D1 table; V4-A17, V4-D1). The lock id is unchanged.
Revision r8, 2026-11-08: tests that passed without their inputs now fail; the locked v3 records the Step 0 tests need are copied to the pod and verified by GM (§15, r8 table; V4-A12).
Revision r7, 2026-11-07: three CUDA-only device bugs in the core test code fixed after GM failed on them (§15, r7 table; V4-A9).
Revision r6, 2026-11-07: a pre-data fix of the lock tooling after the first contact with the pod (§15, r6 table; `v4/DEVIATION_LOG_V4.md`
V4-A6, V4-A7). No study record exists; hypotheses, arms, sample, analysis and ceiling are unchanged.

Revision r5, 2026-11-07, branch `v3-novelty`: a pre-data amendment of the r4 lock, for a budget of $30 in total (§15, r5 table; `v4/DEVIATION_LOG_V4.md` V4-A2). r4, the same day, moved the study to the A40 (§15, r4 table; V4-A1). r3 was dated 2026-11-03. r2 applied the author decisions D1–D9 on draft r1. r3 applies the Phase 2 record
(`v4/phase2/PHASE2_LITERATURE.md` §5, approved by the corresponding author on 2026-11-03) and three author additions (§15, r3 table).
**No runs and no cloud actions yet.** The lock was written at the end of Phase 3 (§14) and rewritten by each amendment.
- New, separately locked study, with the v3 discipline:
  - a dated private SHA-256 lock (**not** a public pre-registration);
  - gates and a CUDA mutation check;
  - an independent recomputation;
  - a deviation log;
  - two ledgers: a GPU ledger (hours) and a cost ledger (USD).
- Hypotheses H1–H5 come from the v3 results (locked tests and X13), so they are exploratory in origin. v4 tests whether these
  results hold at larger scale and in other models.
- A cloud run requires authorization files written by the corresponding author (§11), **one per model**: `v4_runs/AUTHORIZED_V4_<model>.txt`,
  containing `PROTOCOL_V4`, the lock id and the model id. Claude Code works inside the pod over SSH only while the file for that
  model exists. The corresponding author starts and stops the pod. I never create an authorization file, and I have no `runpodctl`
  or billing access.
- Every number below comes from `v4/plan/v4_plan_numbers.py` → `v4/plan/v4_plan_numbers.json`. The script reads the v3 records
  read-only, does no GPU work and uses no network.
  - Model geometry: Qwen2.5-1.5B/3B/7B and Mistral-7B-v0.3 come from their HF `config.json`. Llama-3.2-3B and Llama-3.1-8B are
    gated, so they come from the model cards. Gate G7 re-reads every geometry from the pinned snapshot.
  - r3: the Llama geometry is confirmed from Meta's `llama-models/models/sku_list.py` (Phase 2 §3), and the Phase 3 snapshot
    checks (§2) fix tied embeddings and vocab size from pinned bytes.
  - Prices: runpod.io/pricing, read 2026-11-02. The A40's secure price and stock were confirmed by the corresponding author on 2026-11-07.
- Nothing from v3, X9, X13, E1/E1L or Stage 1/2 is edited.

## 0. Read first

1. **Licences.** Qwen2.5-3B uses the *Qwen Research License* (non-commercial, research or evaluation only, "Built with Qwen"
   attribution). It is not Apache-2.0 like the other Qwen sizes. Research use is permitted. The model is kept (D2) and its licence
   goes into the manuscript's licence paragraph (§13). H4 needs all seven Qwen and Llama models (r5, §7.4).
2. **FP16 risk.** Every model in the ladder is stored in bfloat16. v3 ran the 0.5B/1B models in FP16 without problems, but larger
   models can overflow in FP16. Gate G3 (finite values) decides this per model, before calibration. The rule for a failure is fixed
   in advance (D6, §5):
   - the model runs **both weight copies in BF16** and is **flagged**;
   - every hypothesis is reported **with and without the flagged models**.
3. **Head dimension changes with scale.** Qwen 0.5B and Llama 1B have head_dim 64; every larger model has 128. Within each family,
   head_dim is therefore confounded with size between the smallest rung and the rest. C1 draws from the unit sphere of R^head_dim.
   **head_dim is reported per model** in every results table and figure.
4. **Power is not the constraint for H1–H3 and H5, even at n = 100.** It is for H4. v3's effects are 0.74–1.23 log units, and the per-model
   MDE is at most 0.206 (n = 100, Holm worst case; §7.6). H4 is a test across models: with 4 Qwen and 3 Llama models, the exact
   permutation test can reach p < 0.05 only if the two families separate completely (§7.4).
5. **The budget is tight, and the main uncertainty is speed, not price.**
   - Projection (r5): about 27.9 GPU-hours, about 36.4 billed pod-hours, about **$22.27** on one A40 (secure cloud, on demand), storage included.
   - v3 ran host-launch-bound, so time depends on the cloud host CPU, which is unknown. The $30 ceiling allows only about **1.34×** the projected time
     with every model and about 1.56× without Mistral (§10), far less than the 4× of r4. G6 replaces the projection with measurements before any new
     model, by which point about $5.47 is committed.

## 1. Questions

Do v3's core findings on weight-induced (NF4) cache error hold across scale (0.5B–8B) and in a third model family?
1. **Path provenance (H1, H2):** natural K error is milder than isotropic noise of its size, and natural V error is harsher.
2. **Natural-size path ranking (H3):** natural V error is more disruptive than natural K error.
3. **Family dependence (H4):** the sign of NAT − C1 depends on family.
4. **Structure (H5):** the cross-position common mode is far above isotropic.

**Scope.** v4 has the core arms only. Nothing from QA/QO, CV, C2, the energy-matched arms, LOC/INH or the predictor tests (H-A, H-B,
H-C) is repeated.

Novelty wording, as in v3: "to our knowledge"; never "first". A literature update on scale studies of weight-quantization error in
the KV cache is part of Phase 1 (§14).

**Scale context (r3; each entry added only after its full PDF was read on 2026-11-03).**
- **arXiv 2609.26173** (Dewage, Pensky, De Silva; accepted at IEEE ICMLA 2026, per the PDF). They quantize one attention projection
  at a time (RTN, and GPTQ for 7 models, 3 and 4 bits, group 128) in nine models from 1.3B to 8B, including Qwen2.5-1.5B/7B,
  LLaMA-3-8B and Mistral-7B, and measure the WikiText-2 ΔPPL.
  - At 3-bit RTN, V is the dominant component in 7 of 9 models, with 38–51 % of the positive ΔPPL.
  - V's share exceeds K's in 8 of 9 models; the exception is OPT-1.3B.
  - This is the closest scale precedent for **H3** (V more disruptive than K). It works at the weight-projection and perplexity
    level, with no cache or decoding measurement and no norm-matched random control. v4 adds both.
- **arXiv 2605.23901** (Ouyang et al., "LLMs as Noisy Channels"). They fit a Shannon-type scaling law to WikiText-2 loss under
  three perturbations on Pythia (to 12B) and OLMo2 (to 32B):
  - additive Gaussian weight noise set by an SNR relative to the weight power;
  - GPTQ at 2–4 bits (and, in an appendix, AWQ, bitsandbytes 4-bit and quanto 2-bit);
  - supervised fine-tuning.
  - The paper does **not** compare quantization error and Gaussian noise at equal norm, and does not touch the KV cache. It is
    scale context only, and is cited as such.
- **arXiv 2608.08188** (Zhou et al., "Quantization Degradation in LLMs: A Signal-Noise Perspective"). They split a source SNR into
  three parts: the weight-error magnitude, the signal strength, and the **alignment** of the error with task activations. The
  quantization method changes the alignment, not only the size.
  - Across Qwen3 0.6B–14B under GPTQ-W3, larger models have a higher source SNR, though not strictly monotonically.
  - Larger models also show weaker cross-layer error amplification.
  - This supports reading the H1/H2 effects as alignment ("which direction") effects, and the expectation that their size may change
    with scale. The paper has no cache measurement and no norm-matched control.
- The gap statement of Phase 2 §4 stands: no work found measures weight-induced KV-cache error against norm-matched isotropic
  controls, by path, across a size ladder in several families.

## 2. Models: base checkpoints throughout

**Choice: base models for the whole ladder.** Reasons:
- the prompt is the raw 3-shot completion prompt, with no chat template (v3 L1). Instruct models would run off-format;
- v3's base-vs-instruct effects were small and unresolved (X12);
- base checkpoints exist for every size and family on the list.

The one instruct model is Llama-3.2-1B-Instruct, which appears **only in Step 0**, because it has v3/E1L estimates for the
cross-hardware check.

| Id | HF id | Layers | Q/KV heads | head_dim | Params | Licence | Dual-weight memory (projected) |
|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b | `Qwen/Qwen2.5-0.5B` | 24 | 14/2 | 64 | 0.49 B | apache-2.0 | 2.7 GB |
| qwen2.5-1.5b | `Qwen/Qwen2.5-1.5B` | 28 | 12/2 | 128 | 1.54 B | apache-2.0 | 5.3 GB |
| qwen2.5-3b | `Qwen/Qwen2.5-3B` | 36 | 16/2 | 128 | 3.09 B | **qwen-research** | 9.1 GB |
| qwen2.5-7b | `Qwen/Qwen2.5-7B` | 28 | 28/4 | 128 | 7.62 B | apache-2.0 | 20.1 GB |
| llama3.2-1b | `meta-llama/Llama-3.2-1B` | 16 | 32/8 | 64 | 1.24 B | Llama 3.2 Community (gated) | 4.5 GB |
| llama3.2-3b | `meta-llama/Llama-3.2-3B` | 28 | 24/8 | 128 | 3.21 B | Llama 3.2 Community (gated) | 9.4 GB |
| llama3.1-8b | `meta-llama/Llama-3.1-8B` | 32 | 32/8 | 128 | 8.03 B | Llama 3.1 Community (gated) | 21.2 GB |
| mistral-7b-v0.3 | `mistralai/Mistral-7B-v0.3` | 32 | 32/8 | 128 | 7.25 B | apache-2.0 | 19.6 GB |
| *Step 0 only:* llama3.2-1b-instruct | `meta-llama/Llama-3.2-1B-Instruct` | 16 | 32/8 | 64 | 1.24 B | Llama 3.2 Community (gated) | 4.5 GB |

- **Dual-weight memory** is FP16 linear weights plus embeddings and an untied head, plus NF4 linear weights at 0.516 bytes per
  parameter, plus a 1.5 GB allowance. It is the same `DynamicLinear` construction as v3. Every model fits a 48 GB GPU with room to
  spare. The 7–8B models would fit a 24 GB GPU only tightly, so 24 GB cards are not used.
- **Revisions:** each snapshot is pinned to a commit hash, listed in the lock, together with the SHA-256 of its safetensors files.

**Pinned revisions (r3).** Read from the HF API on 2026-11-03. Each shard's SHA-256 is the HF-reported LFS hash at that revision,
recorded in `v4/V4_MODEL_MANIFEST.json`. G7 re-hashes the downloaded shards against it.

| Id | Revision | Shards | Source of geometry |
|---|---|---|---|
| qwen2.5-0.5b | `060db649…` (= v3 qwen-base) | 1 | config.json |
| qwen2.5-1.5b | `8faed761…` | 1 | config.json |
| qwen2.5-3b | `3aab1f19…` | 2 | config.json |
| qwen2.5-7b | `d1497293…` | 4 | config.json |
| llama3.2-1b | `4e20de36…` (= v3 llama-base) | 1 | config.json |
| llama3.2-3b | `13afe512…` | 2 | Meta `sku_list.py`, plus the snapshot checks below |
| llama3.1-8b | `d04e592b…` | 4 | Meta `sku_list.py`, plus the snapshot checks below |
| mistral-7b-v0.3 | `caa1feb0e54d415e2df31207e5f4e273e33509b1` | 3 (HF format) | config.json |
| llama3.2-1b-instruct | `92131767…` (= v3 llama-instruct) | 1 | config.json |

The full hashes are in the manifest. The three Step 0 revisions are those of the v3 lock, and the shard hashes agree with it.

**Snapshot checks (r3, author addition), recorded in the manifest.** The corresponding author's laptop HF account has no access to
the two gated Llama repos that v3 did not use (HTTP 401 on 2026-11-03). Their `config.json` was therefore not readable in Phase 3.
- Two pinned-byte checks are made instead, and recorded with their method:
  1. **Tied embeddings:** from `model.safetensors.index.json`. Its bytes at the pinned revision are proven identical through the
     git blob id, which the HF API lists without access, compared with a public copy.
     - **Llama-3.2-3B: tied** (no `lm_head.weight`; 254 tensors). This confirms the r2 memory assumption.
     - **Llama-3.1-8B: untied.**
  2. **Vocab size:** from the safetensors header of the shard that holds `model.embed_tokens.weight`. The header was range-read
     from a public copy whose shard SHA-256 equals the pinned shard's HF-reported SHA-256, so the bytes are the same file.
     - **Llama-3.2-3B: embed_tokens [128256, 3072], BF16.**
     - **Llama-3.1-8B: embed_tokens [128256, 4096], BF16.**
- G7 re-reads `config.json` on the pod. Its `tie_word_embeddings` and `vocab_size` must equal these values, or the model stops.
- The other models' tied flag and vocab come from their pinned `config.json`, and are also in the manifest:
  - Qwen 0.5/1.5/3B: tied, vocab 151936;
  - Qwen 7B: untied, vocab 152064;
  - Llama 1B: tied, vocab 128256;
  - Mistral: untied, vocab 32768.

**Third family: Mistral-7B-v0.3, recommended.**
- *For:*
  - Apache-2.0, and **not gated** (`"gated": false`, verified in Phase 2).
  - Its architecture is close to Llama's: GQA 32/8, RoPE, SwiGLU, no attention biases, no QK-norm, head_dim 128.
  - A Mistral–Llama difference would therefore point to training, data or tokenizer rather than architecture. A Mistral–Qwen
    difference would leave that question open.
  - It is a widely used and well documented base model at the top of the size range.
  - **Download (r3):** revision `caa1feb0e54d415e2df31207e5f4e273e33509b1`. The download filter (`allow_patterns`) takes only the
    HF-format shards `model-0000{1..3}-of-00003.safetensors`, their index, the config and the tokenizer files. It **excludes
    `consolidated.safetensors`** (the mistral_inference copy of the same weights, about 14.5 GB).
  - `sliding_window: null`, so attention is full and there is no special case in the cache. RoPE base 1e6 (Llama 3.x: 5e5,
    scaled). This is one more within-"family" difference, carried into L2.
- *Against:* it has one size only, so it cannot enter the H4 family test, which needs at least two models per family (§7.4). In H4 it is
  an out-of-sample observation with no prediction.
- *Alternative (D3):* a third family with two or more permissively licensed sizes would let that family enter H4. Example: OLMo-2
  1B and 7B (Apache-2.0, open data). Its known differences would add confounds: QK-norm on the projected keys, full multi-head
  attention at 7B and a different normalization placement. These architecture facts would be verified in Phase 1 before any
  choice.

## 3. Arms, protocol and calibration (v3 rules, restricted)

**Unchanged from v3 (PROTOCOL_V3 §3–§5, with amendments 1 and 2):**
- teacher forcing;
- steps s = 1…S_i, with S_i = min(64, history length − 1);
- the FP16 prefill is never perturbed;
- perturbations go only to the newly written entry, at write time;
- JS in float64, with m_i floored at 1e-9;
- θ(a, b) = mean_i [log m_i^a − log m_i^b];
- seeds are SHA-256 labels, with the new master label `rv4-scale-2026-11-02`;
- amendment 2's float64 construction check and bitwise correctly-rounded FP16 check;
- the exactly-once records and torn-line rule of amendment 1.

**Problems and prompt:**
- r5: the **first 100** of the same 200 test ids (`e1_sample.json` tierA_ids, in their fixed seeded order; seed `rv2-E1-test-order-2026-10-27`). A problem's draws are keyed by its id, so they are the draws it would have had in the 200;
- r5: the **first 50** of the same 100 GSM8K-train calibration ids, in their fixed seeded order. The G6 timing problems stay the last 3 of the 100, so they are not calibration problems;
- the raw 3-shot prompt;
- each model's in-session FP16 greedy history, ≤ 65 tokens, with the drift-marker stop.

**Arms (6 + reference):**

| Arm | Stored K | Stored V |
|---|---|---|
| FP16 | κ^F | κ^F (reference pass) |
| NAT | κ^N | κ^N |
| C1 | κ^F + ‖Δ‖u | κ^F + ‖Δ‖u |
| NAT-K | κ^N_K | κ^F_V |
| C1-K | κ^F + ‖Δ_K‖u | κ^F_V |
| NAT-V | κ^F_K | κ^N_V |
| C1-V | κ^F_K | κ^F + ‖Δ_V‖u |

Here u is uniform on the unit sphere of R^head_dim, and Δ = κ^N − κ^F per slot.

**Calibration, per model, before its first test problem:**
- arms: FP16, NAT, and C1 at α ∈ {0.5, 1, 2};
- 50 train problems (r5), 64 steps;
- frozen outputs: ρ_K, ρ_V (descriptive), b̂, c and margin_m = |b̂|·ln(1 + c), by the E1/v3 rules. Equivalence can be claimed only if
  b̂ ≥ 0.25;
- the hash is appended to the lock before the model's first test problem;
- not computed, because no v4 arm needs them: M, U₈, σ_c and the outlier channels.

**X13a logging (NAT, per problem):** per slot, n_i, S1_σ = Σ_j δ_jσ, S2_σ = Σ_j ‖δ_jσ‖², and the per-entry norms (for the empirical
null). The record schema is that of X13 §4.

## 4. Step 0: cross-hardware check (before any new model)

Rerun on the cloud GPU, with the full v4 pipeline (gates, calibration, test):
- **qwen2.5-0.5b** (base; also a ladder rung);
- **llama3.2-1b-instruct** (Step 0 only);
- **llama3.2-1b** (base; a ladder rung). It also has v3 estimates, so it is checked at no extra cost.

Bitwise identity with the laptop (RTX 4060, Ada sm_89) is **not** expected on the A40 (Ampere sm_86): the GPU architecture, cuBLAS
heuristics and kernels differ. v3 also ran without the r4 deterministic settings (§5). The criteria below are unchanged by r4. The pre-declared agreement criteria, per model:
1. **Contrast agreement.** For each of θ(NAT, C1), θ(NAT-K, C1-K), θ(NAT-V, C1-V) and θ(NAT-K, NAT-V):
   - δ = θ_cloud − θ_v3, paired by problem id over the 100 ids (r5);
   - the 90 % problem-clustered bootstrap CI of δ must lie within ±margin_m^v3. The locked v3 margins are 0.2347 (qwen-base),
     0.2096 (llama-instruct) and 0.2095 (llama-base);
   - **and** the point estimate |δ̂| ≤ 0.10.
2. **Verdict agreement (D9), only for contrasts with |θ_v3| ≥ 0.2.** Each such contrast gets the same verdict category as in v3:
   "difference" with the same sign, or "equivalent". For NAT-K − NAT-V, which was descriptive in v3, the category is read from its
   v3 CI. Below 0.2, a verdict can flip from small numerical differences alone, so only criterion 1 applies there.

   This gives the following contrasts:

   | Model | NAT − C1 (v3) | NAT-K − C1-K | NAT-V − C1-V | NAT-K − NAT-V |
   |---|---|---|---|---|
   | qwen2.5-0.5b (v3 qwen-base) | −0.241 → criterion applies | −1.233 → applies | +0.831 → applies | −0.932 → applies |
   | llama3.2-1b-instruct | +0.079 → criterion 1 only | −0.882 → applies | +0.890 → applies | −1.008 → applies |
   | llama3.2-1b (v3 llama-base) | +0.013 → criterion 1 only | −0.932 → applies | +0.977 → applies | −0.958 → applies |
   Checked on v3's own records (r5): every contrast this criterion applies to keeps its v3 category on the first 100 ids. Dropping to 100 therefore does not by itself flip a verdict.
3. **Calibration agreement.** ρ_K and ρ_V within ±5 % relative of v3, and b̂ within ±0.25 of v3. r5: v3's values are recomputed on the same first 50 problems by the same rules, so that this compares like with like. They differ from the locked 100-problem values by at most 0.3 % in ρ and 0.042 in b̂.
4. **Reported, not gated:**
   - the share of problems whose FP16 history is token-identical to v3's;
   - X13a L^K and L^V against the X13 values (which came from 40 problems);
   - the per-step JS correlation with v3.

**What n = 100 changes (r5).** The criteria are unchanged, but criterion 1's confidence interval is about √2 wider, so it is harder to meet. It still passes if the per-problem SD of δ is at most about 0.67, with the point estimate at its 0.10 limit and the smallest locked margin.

**Decision.** Step 0 passes if criteria 1–3 hold for all three models. If any criterion fails: **stop and report**, and the authors
decide. The options would then be:
- proceed with all models on the cloud, with every comparison to v3 labelled "cross-hardware, not agreed";
- or stop the study.

Step 0's records are the v4 records of the two base rungs: they are not run twice.

## 5. Gates (per model; any failure ⇒ stop and report; nothing relaxed)

| Gate | Check |
|---|---|
| GM | CUDA mutation check on the cloud GPU, once before Step 0. It covers v3's core mutants restricted to the v4 arms, plus the new v4 mutants (§12). r4: it also logs the `nvidia-smi` driver version, CUDA version and ECC mode, torch's CUDA runtime, the deterministic settings and the image digest record; GM fails if any is missing or the settings are off. No ECC mode is required |
| G1 | α = 0: every constructed arm at scale 0 equals pure FP16 bitwise (10 train problems × 64 steps). For NAT-K and NAT-V, the FP16 pass after measurement gives logits bitwise equal to pure FP16 |
| G2 | amendment 2: float64 construction check (≤ 1e-4) and bitwise rounding check, in every constructed arm |
| G3 | finite: all logits and entries are finite and step-1 JS is 0, on 6 train problems, in FP16. **On failure, the BF16 rule below applies** |
| G4 | seeds: same label ⇒ identical draws; different label ⇒ different |
| G5 | format: prompt round-trip, BOS, EOS ids and drift-marker detection, per tokenizer (the pilot's procedure) |
| G6 | runtime: measured on the cloud (§10). The design is fixed once, before the first test problem of the first new model |
| G7 | geometry: layers, heads, KV heads and head_dim read from the pinned snapshot equal §2. The snapshot hash and safetensors SHA-256 equal the lock |
| G8 | determinism: the first 3 test problems of each model are run twice in-session, and the records must be bitwise equal |

**Deterministic settings (r4).** Every driver process runs with `torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`, set before torch loads. Every pod command refuses to start if they are not active.

**Per-problem integrity** (v3 rules): step-1 JS is 0, values are finite and non-negative, the construction and rounding checks pass,
the routing overrides are reset, and the first token is shared. Hardware stop at 88 °C, read through `nvidia-smi` on the pod.

**BF16 rule for a G3 failure (D6, pre-declared).** It is decided at G3, before the model's calibration, and the model is never
switched back.
- **What changes for the flagged model:**
  - the high-precision weight copy is BF16 instead of FP16;
  - the 4-bit compute dtype and activations are BF16;
  - the cache is BF16-typed;
  - the reference pass, histories and calibration all run in BF16.
- **Unchanged:** the NF4 codes, the arms and the estimands.
- **Amendment 2's rounding check becomes a bitwise check against correctly rounded BF16** (round to nearest, ties to even), with
  its own independent oracle. The float64 construction check is unchanged.
- **G3 rerun.** G3 is rerun in BF16. If it fails again, the model stops and is reported as not run.
- **Flagging.** The flag is written to the lock appendix and to the model's manifest.
- **Reporting.** Every hypothesis (F1–F3, F5, H4) is computed **twice**: once with all models, with Holm over all of them, and
  once without the flagged models, with Holm over the rest. Both are reported side by side, with the flagged models marked in every
  table.
  - If the two disagree on a verdict, the abstract-level statement is the one **without** the flagged models, and the disagreement
    is reported.
  - In H4, removing models changes the permutation set, and the smallest attainable p is stated for each version.
- **Step 0 models** have v3 FP16 estimates. If one of them fails G3, Step 0 fails (§4).

## 6. Outcomes

- **Per problem:** m_i^a, the mean JS over steps 1…S_i. Contrasts are θ(a, b).
- **Per problem, NAT:** the X13a statistic x_i^P = ln(n_i C_i^P) − ν_i^P for P ∈ {K, V}, where C_i^P = Σ_σ‖S1_σ/n_i‖² / Σ_σ(S2_σ/n_i)
  and ν_i^P is the empirical isotropic null (200 seeded draws).
- **Descriptive:** ρ_K, ρ_V and ρ_V/ρ_K per model, and the natural relative error by path.

## 7. Hypotheses, decision rules, falsification

**Common rules (E1/v3).**
- **Difference:** the 95 % CI excludes 0.
- **Equivalent:** the 90 % CI lies within ±margin_m, and b̂_m ≥ 0.25.
- **Inconclusive:** otherwise.
- Equivalence is never inferred from non-significance.
- p-values: two-sided paired sign-flip permutation, 10,000.
- CIs: problem-clustered percentile bootstrap, 10,000, locked-analysis seed 20261105.

**"In every model".** A universal hypothesis is **supported in every model** only if every model's own test is supported after Holm
within that hypothesis's family. It is **falsified** if any one model meets the falsification rule. Otherwise the result is "not
established in all models", and each model's verdict is listed.

**Families (Holm within each).** The model set is the 8 ladder models, minus any model that was not run (a G3 failure in BF16 as
well, or Mistral dropped, §10) or is unresolved after truncation. Without Mistral the families are F1–F3 with 7 tests each and F5 with 14. Every family is also computed without the BF16-flagged models (§5).
- **F1:** H1, one test per model (8).
- **F2:** H2, per model (8).
- **F3:** H3, per model (8).
- **F5:** H5, per model and path (16).
- **H4:** one test, outside the Holm families. It has its own decision rule (§7.4).
- Step 0's llama3.2-1b-instruct is in no family.

### 7.1 H1: NAT-K milder than C1-K, in every model

- **Source:** v3 provenance-K, −0.882 to −1.233, every CI excluding 0, in four models.
- **Estimand:** θ₁ = θ(NAT-K, C1-K).
- **Prediction:** θ₁ < 0.
- **Supported** (per model): Holm p < 0.05 and the CI lies entirely below 0.
- **Falsified** (per model, which falsifies the universal claim): the CI lies entirely above 0, or θ₁ is equivalent to 0.

### 7.2 H2: NAT-V harsher than C1-V, in every model

- **Source:** v3 provenance-V, +0.740 to +0.977, in four models.
- **Estimand:** θ₂ = θ(NAT-V, C1-V).
- **Prediction:** θ₂ > 0.
- **Supported:** Holm p < 0.05 and the CI lies entirely above 0.
- **Falsified:** the CI lies entirely below 0, or θ₂ is equivalent to 0.

### 7.3 H3: NAT-K vs NAT-V at natural size

- **Source:** v3 descriptive, −0.852 to −1.008, every CI excluding 0. ρ_V/ρ_K = 2.9–3.5 in v3.
- **Estimand:** θ₃ = θ(NAT-K, NAT-V).
- **Prediction:** θ₃ < 0 (V more disruptive at natural size). The test is two-sided.
- **Supported:** Holm p < 0.05 and the CI lies entirely below 0.
- **Falsified:** the CI lies entirely above 0, or θ₃ is equivalent to 0.
- **Pre-stated caution:** θ₃ compares two paths at their own natural sizes, so it depends on ρ_V/ρ_K. That ratio may change with scale,
  and is reported next to θ₃. The energy-matched comparison (v3 H-D) is not repeated.

### 7.4 H4: the sign of NAT − C1 follows family

- **Source:** v3 θ(NAT, C1) was −0.241 and −0.323 in the two Qwen models, and +0.079 and +0.013 in the two Llama models (the last one
  equivalent to 0). The v3 H-F family contrast was supported.
- **Prediction, per family:** **Qwen θ < 0; Llama θ ≥ 0** (not negative).
- **Test**, on the ladder's Qwen (4) and Llama (3) models; θ̂_m is the per-model estimate:
  1. **Model-level exact permutation test.** The statistic is T = mean_{Llama} θ̂_m − mean_{Qwen} θ̂_m. The test is exact over all
     C(7, 3) = 35 assignments of the family labels, one-sided in the predicted direction. Models are the unit, so between-model
     variation within a family is the error term. The smallest attainable p is 1/35 = 0.029, so **p ≤ 0.05 requires complete
     separation** (every Qwen θ̂ below every Llama θ̂).
  2. **Per-model sign readings** (95 % CIs): every Qwen model has a CI entirely below 0, and no Llama model has a CI entirely below 0.
- **Supported:** both 1 and 2 hold.
- **Falsified:** any Qwen model has a CI entirely above 0, or is equivalent to 0; **or** any Llama model has a CI entirely below 0.
- **Inconclusive:** otherwise, for example incomplete separation without a falsifying model.
- **Mistral-7B-v0.3:** reported with its θ̂ and CI as an out-of-sample observation, with no prediction. It does not enter the test. If it is dropped (§10), the test is untouched.
- **All seven are required (r5).** If any Qwen or Llama model is not run or is unresolved, H4 is reported as **not estimable**, and the per-model θ̂ stay descriptive. r4 allowed 3 + 2 models. The one exception is the without-flagged version (§5), which uses the remaining models and states its smallest attainable p.
- **Descriptive:** θ̂_m against log parameter count within each family (Spearman, no test). The Step 0 llama-instruct estimate is shown
  alongside.
- **Scope:** "family" here means these specific checkpoints. Family is confounded with tokenizer, data, architecture details and
  head_dim (v3 L2).

### 7.5 H5: the common-mode share is above isotropic, in every model and on both paths

- **Source:** X13a. L^K = 2.29–2.63 and L^V = 2.23–2.54 (about 10–14× isotropic), supported in all four v3 models.
- **Estimand:** L^P = mean_i x_i^P, per model and path.
- **Supported:** the 95 % lower bound is above ln 4 and the Holm p (F5) is below 0.05. These are the X13a thresholds, unchanged.
- **Falsified:** the 95 % upper bound is below ln 2.
- **Inconclusive:** otherwise.
- **Descriptive:** C^P per layer.

### 7.6 Power

All from `v4/plan/v4_plan_numbers.json` (`power_r5`, `mde_log_units`, `subset_v3`). r5 lowers n from 200 to 100.

- **Assumption:** the largest per-problem SD of the four contrasts in v3 is 0.575 (computed from the v3 records; the range is 0.39–0.58).
- **MDE** at 80 % power, two-sided, at that SD:

  | n | α = 0.05 | α = 0.05/8 (F1–F3, Holm worst case) | α = 0.05/7 (F1–F3 without Mistral) | α = 0.05/16 (F5) | α = 0.05/14 (F5 without Mistral) |
  |---|---|---|---|---|---|
  | 200 (r4) | 0.114 | 0.145 | 0.144 | 0.154 | 0.153 |
  | **100 (r5)** | **0.161** | **0.206** | **0.203** | **0.218** | **0.216** |

- **Per hypothesis at n = 100** (each with its own largest v3 SD, Holm worst case over 8):

  | Hypothesis | Largest SD | Smallest v3 effect | MDE | Effect / MDE |
  |---|---|---|---|---|
  | H1 | 0.575 | 0.882 | 0.206 | 4.3 |
  | H2 | 0.523 | 0.740 | 0.187 | 4.0 |
  | H3 | 0.568 | 0.852 | 0.203 | 4.2 |

- **H5** has its own SD, backed out of the X13 CIs at n = 40: at most 0.305. At n = 100 the 95 % half-width is 0.060 and the MDE over 16 tests is 0.116. Support needs a lower bound above ln 4 = 1.386, so a true L of about 1.446 or more; v3's smallest is 2.231.
- H1–H3 therefore keep about 4× the MDE (about 5–8× at n = 200), and v3's smallest H5 estimate clears the level support needs by about 0.8. A failure at scale would still show up as a real sign change or shrinkage, not as power loss. Smaller true effects at scale are less protected than before.
- **Checked on v3's own records:** on the first 100 ids, all 12 H1–H3 estimates in the four v3 models move by at most 0.059 from their 200-problem values and keep CIs that exclude 0.
- **Equivalence:** the 90 % CI half-width is about 0.095 at n = 100 (0.067 at 200). With margins of about 0.21, equivalence is attainable only when |θ| ≲ 0.115 (0.14 at 200).
- **H4.**
  - **The permutation test is unaffected by n.** It is model-level: 35 labellings, smallest attainable p = 1/35 = 0.029, requiring complete separation.
  - **Its other inputs do depend on n.** Each θ̂_m is estimated with more noise (SE up to 0.051, from 0.036), which makes complete separation less likely when the families are close. The per-model sign readings are also part of "supported" (§7.4), and their 95 % half-width grows to about 0.10 (from 0.07), with SDs of 0.43–0.51. A Qwen model must have |θ| of about 0.10 for its CI to exclude 0. v3's Qwen estimates were −0.241 and −0.323, and −0.295 and −0.346 on the first 100 ids, so this is enough at v3's sizes. The larger Qwen models have no estimate yet. v3 llama-base, at +0.013 (+0.001 on the first 100 ids), stays unresolved, which is acceptable for its "not negative" prediction.
  - With 4 + 3 models, H4 is a coarse test by construction, and the reported between-model spread is part of the result.

## 8. Descriptive and sensitivity analyses

- **Descriptive:** all four θ contrasts and the X13a statistics per model; ρ_K, ρ_V; all of these against size within family, plotted.
- **Sensitivity** (none replaces a primary):
  - excluding the exposed test ids (2 of the first 100, ids 45 and 172; 4 of the 200);
  - steps 2–8 and 9–64;
  - per-problem medians;
  - amendment 2's truncation sensitivity (slots above 5 %).
- **Independent recomputation** (as in v3 Phase 5, V3-D08):
  - a script with no project imports recomputes every estimate, CI, Holm decision, the H4 permutation and X13a from the raw records;
  - tolerances declared before it runs: point estimates exact to 1e-12; CI bounds within the Monte-Carlo tolerance of
    v3's `RECOMPUTATION_TOLERANCES.md`; every verdict must agree;
  - any verdict disagreement is reported as unresolved.

## 9. Second quantizer: dropped (r5)

The optional HQQ 4-bit arm (D4) is dropped entirely, to fit the $30 ceiling. It would have cost about 9 GPU-hours even at the smaller n. Its code, tests, mutants, gate branch, lock line and package are removed. The study has one quantizer, bitsandbytes NF4 (L8). The r3 decisions on HQQ remain in §15 as history. The section number is kept so that references stay stable.

## 10. GPU choice, time, cost, ceiling and reductions

**GPU (r4): one A40 48 GB pod for the whole study, secure cloud, on demand** ($0.59/h; community cloud $0.35/h).
- **No GPU-type change mid-study (D5).** If an A40 is unavailable when a model is due, the study **waits**, or **stops and
  reports**. It never moves to another SKU. Waiting is unbilled once the pod is stopped; only network-volume storage accrues (about $0.23 a day).
- **Software (r4):** torch stays 2.13.0+cu126. The wheel's architecture list includes sm_86, read from the installed wheel on 2026-11-07. GM is the first run on the A40 itself.
- **The SKU check is exact (r4):** `NVIDIA A40` or `A40`. A substring test would also accept an RTX A4000 or RTX A400.
- **Why not others:**
  - the RTX A6000, the r3 choice, was not available for network volumes (corresponding author's check, 2026-11-07). It has the same secure price, and the projection on it is about the same;
  - the L40S is 1.85× the price for about 24 % more bandwidth;
  - the A100 80 GB is 3× the price;
  - the RTX 5090 would need another torch build: the cu126 wheel has no sm_120;
  - the run is host-bound for most models, so a faster GPU buys little.
- **Community cloud:** cheaper (about $13.56 for the same plan), but its hosts are less reliable. Interruptions are survivable (§11) but cost billed time. D5 stays Secure Cloud.

**Projection model** (assumptions stated; G6 replaces them). Per arm-problem time is the sum of:
- a host term, 0.286 s per layer. This is v3's laptop NAT time per layer, including v3's predictor overhead, which v4 does not have,
  so it is conservative;
- a device term: 64 steps × (bytes read by the FP16 pass at 60 % of peak bandwidth, plus bytes read by the NF4 pass at 30 %).

Arm counts (r5):
- test: 7 arm-equivalents × (100 + the 3 G8 problems run twice) (the FP16 reference counted as a full arm);
- calibration: 5 × 50;
- plus 0.35 h of gates per model, and 0.5 h of GM once.

Billing: billed pod time = 1.2 × GPU-ledger time (idle and setup), plus 0.25 h per model for download and checkpointing, plus 0.5 h of first-boot setup once, plus a 5 %
margin. Rate: $0.59/h (A40, Secure Cloud, On-Demand).

| Model | s per arm-problem | GPU-h |
|---|---|---|
| qwen2.5-0.5b | 7.2 | 2.3 |
| qwen2.5-1.5b | 8.8 | 2.7 |
| qwen2.5-3b | 11.9 | 3.6 |
| qwen2.5-7b | 11.5 | 3.5 |
| llama3.2-1b | 5.3 | 1.8 |
| llama3.2-3b | 9.7 | 3.0 |
| llama3.1-8b | 12.9 | 3.8 |
| mistral-7b-v0.3 | 12.5 | 3.7 |
| llama3.2-1b-instruct (Step 0) | 5.3 | 1.8 |

**Projection (r5, at $0.59/h):**

| | GPU-h | Billed h | Pods | Storage (prorated) | **Total** | Margin under $30 | Left before the $29 stop | Margin under $40 |
|---|---|---|---|---|---|---|---|---|
| **Every model** | 27.9 | 36.4 | $21.45 | $0.82 | **$22.27** | **$7.73** | $6.73 | $17.73 |
| Without Mistral | 24.0 | 31.4 | $18.53 | $0.77 | **$19.30** | $10.70 | $9.70 | $20.70 |

**If everything runs slower than projected:**

| Speed relative to the projection | Every model | Without Mistral |
|---|---|---|
| 1× | $22.27 | $19.30 |
| 1.25× | $27.29 | $23.61 |
| 1.5× | $32.30 (over) | $27.93 |
| 2× | $42.34 (over) | $36.56 (over) |
| **Limit at the $29 stop** | **1.34×** | **1.56×** |

- **Storage (r5):** a 100 GB network volume at $0.07/GB/month is about $0.233 a day, prorated to the expected duration: the billed hours plus an assumed 48 h with no pod (Step 0 decision, G6 stop, gaps between models), about 3.5 days. Each further day adds $0.23. **The corresponding author deletes the volume right after the last download is verified** (§11).
- **Volume size (r5):** the pinned shards are 67.4 GB. With about 12 GB of venv and 3 GB of records (both assumed) the study uses about 82 GB of the 100.
- **Committed before G6:** GM, first-boot setup and the Step 0 models cost about $5.47 before the measured design exists.
- **Hard cost ceiling: $30 USD in total (r5), storage included.** That is about 1.35× the projected $22.27. The corresponding author's r5 request also names "the $40 ceiling"; this protocol uses $30, because the prepaid balance makes it the binding figure (V4-A2). The margin under $40 is reported above for that reason.
- **Close-out reserve (r5): the run stops at $29.** Running up to $30 would leave nothing for the last download or for the volume until it is deleted, and a balance at zero could cost the records still on the volume. It is enforced three ways:
  1. **Cost ledger:** `v4_runs/cost_ledger.jsonl`. Each pod session has two kinds of line:
     - **in-pod lines**, written by the driver: pod session start from `/proc/uptime` at driver start, a 15 s heartbeat, the declared
       $/h, the SKU from `nvidia-smi`, and cumulative estimated USD. For the ceiling check, a session's estimated cost runs from **pod
       boot**, not from driver start, plus the author's idle auto-stop window (counted as billed). This is conservative;
     - **billing lines**, entered by the corresponding author from the RunPod billing page at each checkpoint. These are the authoritative
       USD figures. Each difference from the in-pod estimate is logged.

     The driver refuses to start a problem whose projected cost would cross the $29 stop, and uses the larger of the two figures. A
     watchdog closes the ledgers and stops the run 60 s before the stop.
  2. **Prepaid balance:** RunPod is prepaid. The corresponding author loads exactly $30 (D7), so the balance is a hard
     stop outside the code.
  3. **Idle auto-stop:** the corresponding author starts and stops the pod and sets an idle auto-stop on it (D8). Pods are stopped
     between models unless the next model starts at once.

  The GPU ledger (`v4_runs/gpu_ledger.jsonl`, v3 heartbeat semantics) is kept separately, for compute-hours comparable with v3.
- **G6 runtime gate:**
  - measured during Step 0 on qwen2.5-0.5b, plus a 3-problem probe of llama3.1-8b (the largest model) for the host and device terms;
  - the remaining study is re-projected in USD, with storage prorated to its expected duration;
  - if it exceeds the money left below the $29 stop, **the reduction order (r5) is:**
    1. **drop Mistral-7B-v0.3.** It is out of sample in H4, so H4 is untouched;
    2. still over: **stop and report.**
  - **There are no other automatic cuts.** n stays at 100 and 50, and no other model is dropped, because H4 needs all seven Qwen and Llama models.
  - The chosen design is written to the lock appendix before the first new model, and not revisited. A per-model drift of more than
    25 % above projection triggers a stop-and-report.
  - **Mistral is also checked at its own start (r5).** It runs last, so if the design kept it but the money left no longer covers its projected cost, it is recorded as not run, and the study closes out. No money is spent on a model that could not reach n = 100.
- **Run order:**
  1. GM;
  2. Step 0: qwen2.5-0.5b → llama3.2-1b-instruct → llama3.2-1b, then the Step 0 decision and G6;
  3. the Qwen and Llama ladder: qwen2.5-1.5b → llama3.2-3b → qwen2.5-3b → qwen2.5-7b → llama3.1-8b. The order alternates families;
  4. mistral-7b-v0.3, last and first to drop.
- **Truncation:** a model with fewer than 100 test problems is unresolved. r5: that is the full design, so a partial model is always unresolved. H4 requires all seven Qwen and Llama models, or it is reported as not estimable (§7.4).
- **No analysis of test records until all models finish.** During the run, I report only gates, calibration constants, G6,
  GPU-hours, USD and integrity events (the v3 Phase 4 rule).

## 11. Cloud operations plan

**Who does what (D8).**
- **The corresponding author:**
  - holds the RunPod account, the billing and the HF token. **The token's account must have accepted the licences of all four gated
    Llama repos** (Llama-3.2-1B, Llama-3.2-1B-Instruct, Llama-3.2-3B, Llama-3.1-8B). The laptop account had no access to the last
    two on 2026-11-03 (§2);
  - loads the prepaid balance (equal to the $30 ceiling);
  - creates the network volume and the pod, and sets the HF token as a RunPod secret;
  - sets the idle auto-stop, and starts and stops the pod;
  - writes one authorization file per model on the volume: `/workspace/v4_runs/AUTHORIZED_V4_<model>.txt`, containing `PROTOCOL_V4`,
    the lock id and the model id. A copy goes to the laptop's `research_v2/v4_runs/`. There is a separate file for the GM check
    (`AUTHORIZED_V4_GM.txt`).
- **Claude Code:**
  - works **inside the pod over SSH only**, and only while the authorization file for the current model exists. The driver also
    refuses to run without it;
  - has **no `runpodctl` access, no RunPod API key and no billing access**;
  - prepares the bundle, the scripts and the run instructions;
  - runs the gates and the model's run;
  - packs the checkpoint, and reads the downloaded records;
  - never creates or edits an authorization file.
- **Superseded document.** The repo-root `CLOUD_SETUP.md` (the original project's Qwen2.5-14B note) does **not** apply to v4. It allows
  `huggingface-cli login`, spot instances and terminating the instance, all of which conflict with this section. It is not edited (it
  is an original). Its pinned `requirements.txt` (torch 2.13.0+cu126, bitsandbytes 0.49.2) is the starting point for the v4 package pin.

**Pod and storage.**
- **Network volume (r5):** 100 GB, Standard tier, in one datacenter that has A40 capacity, mounted at `/workspace`. It holds:
  - `/workspace/hf` (`HF_HOME`: model snapshots, about 65 GB at pinned revisions);
  - `/workspace/venv`;
  - `/workspace/repo` (the locked bundle);
  - `/workspace/v4_runs` (records, ledgers, logs).
- **Image:** the official RunPod PyTorch template, pinned to an exact tag. Python packages are pinned to the laptop's versions where
  available (torch 2.13.0, transformers 5.13.0, bitsandbytes 0.49.2) to keep the software stack equal across hardware. The full
  `pip freeze`, CUDA and driver versions, ECC mode and the `nvidia-smi` output go into every record manifest.
- **Image digest (r4):** the tag stays `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`. At first boot `pod_setup.sh` writes `v4_runs/image_digest.json` once, with the digest Docker Hub reports for that tag. GM refuses to start without it. A later read that differs is logged for a deviation row and never replaces the record. A container cannot see its own image's digest, so this records what the tag pointed at, not what the host pulled.

**Code transfer.**
- The repo is local-only and never pushed, so the locked commit goes over as a `git bundle`, copied with `scp` over the pod's SSH.
- On the pod, the driver checks the lock hash of the checked-out tree before any GPU work, and refuses to run on a mismatch.

**Secrets.**
- The HF token is stored as a RunPod secret and exposed to the pod only as the environment variable `HF_TOKEN`. Code reads it only
  through `os.environ` and passes it to `huggingface_hub`.
- `huggingface-cli login` is not used, because it writes a token file. A start-up check verifies that no token file exists under
  `HF_HOME`.
- Logging uses a redaction filter for the patterns `hf_[A-Za-z0-9]{20,}` and the RunPod API-key format.
- **Download gate:** before any checkpoint download, `tools/v4_secret_scan.py` scans every file to be downloaded for token patterns
  and for the literal value of `HF_TOKEN`. A hit blocks the download and is reported, with the value withheld. The same scan runs
  before any commit on the laptop.
- No secret is ever written to the repo, the records, the ledgers or the manifests. `.gitignore` covers `v4_runs/`, as for `v3_runs/`.

**Checkpoint and download after each model.**
1. The model's records, calibration file, gate logs, both ledgers and the manifest are packed into a tar, with a `SHA256SUMS` file.
2. Secret scan.
3. Download to the laptop with `scp` over SSH.
4. `sha256sum -c` on the laptop (with LF line endings).
5. Only then is the model marked "checkpointed" in `v4_runs/status.json`.
6. Nothing is deleted from the volume until the study has ended and every checkpoint is verified.
7. The next model may start while the download runs.

**Deleting the volume (r5).** Right after the last download, `run_v4.py closeout` on the laptop checks that every model has a verified checkpoint there, that its tar is unchanged, and that its merged records show a final state. Only then does the corresponding author delete the volume, which cannot be undone, terminate the pod, and record `volume_deleted_utc` and a final billing line. Claude Code has no access to do this. A model the locked design dropped has nothing to fetch. If the study stops early, the check lists what is missing, and the corresponding author decides.

**Text encoding (r5).** The laptop's piped stdout is cp1252 and strict, and the pod's output is UTF-8. `tools/v4_remote.py` escapes any character the console lacks instead of raising, and every captured child process is decoded as UTF-8 with replacement (`O.TEXT`). Before r5, the first progress bar or warning sign in the pod's output would have killed the stream of a paid run (V4-A2b).

**Resume behaviour.**
- The amendment-1 rules apply unchanged: exactly-once records, a driver stop on a torn line, repair only by script with a deviation
  row, and a ledger heartbeat every 15 s on the volume, so that a killed pod's time is charged at the next start.
- **After an interruption** (host failure, an idle auto-stop, a manual stop), the corresponding author restarts the pod, or starts a
  new one on the same volume, always an **A40**:
  1. the driver checks the lock hash;
  2. it charges the open progress file;
  3. it re-runs G8 on the next problem (determinism after a restart);
  4. it continues from the first missing (model, arm, problem).
- If no A40 is available: wait, or stop and report (§10, D5). The study never moves to another SKU.
- **Idle auto-stop and long models.** The auto-stop window must be longer than the longest gap between driver heartbeats. Since the
  driver keeps the GPU busy, an auto-stop during a run indicates a hang. On restart, the open progress file is charged and the
  event is logged as a deviation.
- Each pod session adds a cost-ledger line, with billed time taken from pod start and stop. The authors confirm it against the RunPod
  billing page at each checkpoint, and differences are logged.

## 12. Implementation-phase validation (planned; nothing exists yet)

- **New modules only:** `v4/rv4.py` (core arms on top of the locked `stored_entry_v3`, X13a sums, the head_dim generalization),
  `v4/run_v4.py`, `v4/analyze_v4.py`, `v4/recompute_v4_independent.py`, `tools/lock_v4.py` and
  `tools/v4_secret_scan.py`. Locked v3, X13 and E1 code is imported unchanged and hash-checked.
- **Tests:**
  - tiny Qwen2 and Llama models with head_dim 64 **and 128**, a tiny GQA model with 4 KV heads, and a tiny Mistral config;
  - real-model CPU checks for qwen2.5-0.5b and llama3.2-1b, and for the larger models on CPU at 2 steps where memory allows;
  - the CPU harness of the driver's cloud code paths, including the ledger, cost-ledger, watchdog, resume and secret-scan paths;
  - the analysis on simulated truth: the "every model" logic, the H4 permutation (exact 35 assignments; separation/p mapping), the
    falsification readings and the H5 thresholds.
- **Mutants (all must be caught):**
  - the v3 core mutants restricted to the six arms;
  - C1 drawn on the 64-sphere when head_dim is 128;
  - the slot norm taken over the KV heads;
  - an X13a sum with ‖S1‖² instead of ‖S1/n‖²;
  - an H4 permutation over problems instead of models;
  - Holm over the wrong family;
  - the cost-ledger rate ignored;
  - a ceiling check after the problem instead of before;
  - a secret scan that misses a token in a gzip-compressed log;
  - G7 accepting an unpinned revision.
- **r3 additions** (the HQQ tests and mutants were removed by r5):
  - **Snapshot checks:** the Llama-3.2-3B tied flag and vocab size, and those of every model, as in §2. They are recorded in the
    manifest, and G7 compares them with the pod's `config.json`.
- **r4 additions:**
  - **Reloads:** when the next load starts, the previous model is released, at the reload site in `gates()` (r5: one site is left, FP16 → BF16; tested on the real function with cyclic fakes) and with real dual-weight backends on CPU, and on CUDA in GM, including the memory figure.
  - **Mutants:** the caller's reference kept at the reload site; reload without garbage collection; a substring SKU check; the deterministic settings not applied, or not required; GM ignoring the environment record or the image record; the image record overwritten or of another tag; an amendment that does not preserve the old lock, runs after data, or ignores an old lock that fails its own commit; verify ignoring amendments or `HEAD`.
- **r5 additions:**
  - **Tests:** the $30 ceiling and $29 stop; storage prorated and ending at deletion; the sample being exactly the first 100 and 50; draws keyed by problem id; Mistral last, dropped by the design or at its start; the design's whole reduction order; H4 not estimable without all seven; Step 0 on v3's first 100 and first 50; the stream on a strict cp1252 console with a real child process; the close-out check; no HQQ left in the study code.
  - **Mutants:** the ceiling left at $150; no close-out reserve; storage ignoring deletion; a full month of storage; n left at 200; calibration left at 100; G6 problems inside calibration; Mistral not last; Mistral not checked at its start; the design never dropping Mistral; the design cutting another model; G8 left out of the projection; H4 estimated without all seven; Step 0 compared with v3's 100 problems; the strict stream; the close-out always ready; the close-out ignoring the final state.
- **Lock manifest:** this protocol, the code, the tests, the sample, the planning JSON, the pinned model revisions and safetensors
  hashes, and the pinned package list. Lock file `research_v2/V4_PROTOCOL_LOCK.txt`.

## 13. Records, deviations, AI use, limitations

- **Records:** `v4_runs/<model>/{calib,tierA}/<arm>.jsonl` (v3 schema, restricted to the core arms, plus the X13a fields), the
  per-model `calibration.json`, the gate logs and the manifest.
- **Deviations:** every change after the lock goes to `v4/DEVIATION_LOG_V4.md` with its date and reason. Only the pre-declared
  reductions, the G3 drop rule and the Step 0 author decision may change what is tested.
- **AI use:** `v4/AI_USE_LOG_V4.md`. This draft: Claude Opus 5.5 (claude-opus-5-5) in Claude Code, 2026-11-02.
- **Limitations, carried into the manuscript:**
  - **L1:** teacher-forced, label-free outcomes; no accuracy (v3 L3).
  - **L2:** one checkpoint per (family, size); family is confounded with tokenizer, data, architecture and head_dim. Within the
    Llama-like architectures, Mistral-7B-v0.3 also differs in RoPE base (1e6 against Llama 3.x's 5e5 with scaling) and vocabulary
    (32768 against 128256).
  - **L3:** head_dim changes with size (64 → 128).
  - **L4:** cross-hardware: the v3 and v4 numbers come from different GPUs (Ada sm_89 and Ampere sm_86) and, from r4, different cuBLAS settings. Step 0 bounds the difference; it does not remove it.
  - **L5:** no chat template.
  - **L6:** the 100 test problems were used before (outcome-free here).
  - **L7:** private lock, exploratory origin.
  - **L8:** one quantizer (bitsandbytes NF4); the HQQ arm was dropped before any data for budget reasons (§9), so nothing here speaks to other quantizers.
  - **L9:** BF16-flagged models, if any (§5): their reference precision differs from the rest, so they are reported with and without.
  - **L10 (r5):** n = 100 per model and 50 for calibration, for budget reasons, chosen before any data. Effects below about 0.21 per model are not reliably detected, and equivalence is attainable only below about 0.115. If Mistral is dropped, the third family is absent.
- **Manuscript items fixed now:**
  - **Licence paragraph:** add Qwen2.5-3B under the Qwen Research License (non-commercial research and evaluation; "Built with
    Qwen" attribution if a derived model is distributed), next to the Apache-2.0 (Qwen 0.5/1.5/7B, Mistral-7B-v0.3) and Llama
    3.1/3.2 Community License entries. H4's need for all seven models, Qwen2.5-3B included (§7.4), goes in the H4 methods text.
  - **Funding statement:** add that the rented cloud GPU time was paid personally by the corresponding author. The amount (USD) is
    filled in at the end from the cost ledger's billing lines, through a `make_*` macro, never typed by hand.
  - **head_dim** is a column in every per-model table.

## 14. Phases (a STOP after each)

1. Draft r1 → author decisions (done 2026-11-02) → r2 (this text).
2. Literature update: scale studies, Mistral-7B-v0.3 facts, and the HQQ release and API from primary sources
   (`v4/phase2/PHASE2_LITERATURE.md`) → STOP. Approved 2026-11-03; r3 applies it.
3. CPU implementation, validation and lock (on the laptop); r4 and r5 re-locks 2026-11-07. The corresponding author then writes the authorization files and starts
   the pod. ← *current*
4. GM and Step 0 → Step 0 decision → G6 design lock → STOP.
5. Ladder run under the reporting rule → STOP.
6. Locked analysis, independent recomputation and report → STOP.
7. Manuscript integration (to be decided after the results) → STOP.

## 15. Decision record (author decisions on r1, 2026-11-02)

| # | Decision | Implemented in |
|---|---|---|
| D1 | Base checkpoints throughout; instruct (Llama-3.2-1B-Instruct) only in Step 0 | §2, §4 |
| D2 | Keep Qwen2.5-3B. Its research licence goes into the licence paragraph. r5: H4 needs all seven models | §0, §7.4, §13 |
| D3 | Third family: Mistral-7B-v0.3 (out of sample in H4) | §2, §7.4 |
| D4 (changed by r5) | **HQQ dropped entirely** (r1 decision: keep it on 3 models, first in the reduction order) | §9 |
| D5 (changed by r4) | One A40 pod for the whole study (r3: RTX A6000). No GPU-type change mid-study. If none is available, wait, or stop and report. No parallel pods | §10, §11 |
| D6 (changed) | A model failing G3 runs both weight copies in BF16 and is flagged. Every hypothesis is reported with and without the flagged models (pre-declared) | §0, §5, §13 L9 |
| D7 (changed by r5) | Ceiling **$30 USD in total, storage included** (before r5: $150); prepaid balance equal to the ceiling | §10 |
| D8 | The corresponding author starts and stops the pod and sets an idle auto-stop. Claude Code works inside the pod over SSH only under a written authorization file per model, with no `runpodctl` or billing access | header, §10, §11 |
| D9 (changed) | The Step 0 verdict-match criterion applies only to contrasts with \|θ_v3\| ≥ 0.2. The paired-difference CI criterion and the \|point difference\| ≤ 0.10 criterion apply to all contrasts | §4 |
| — | head_dim reported per model | §0, §13 |
| — | Funding statement: the rented GPU time was paid personally; the amount comes from the cost ledger at the end | §13 |

**r3 (corresponding author, 2026-11-03): Phase 2 approved; all §5 edits of the Phase 2 record applied, plus three additions.** The HQQ rows were superseded by r5.

| # | Change | Implemented in |
|---|---|---|
| r3-1 | HQQ pinned to 0.2.8.post1, defaults 4-bit / group 64 / axis 1, PYTORCH backend, no tuning; code `dropbox/hqq`; cited through the Dropbox republication | §9 |
| r3-2 | Mistral-7B-v0.3: not gated; revision `caa1feb0…`; the download filter excludes `consolidated.safetensors`; RoPE difference in L2 | §2, §13 |
| r3-3 | Geometry sources: Meta `sku_list.py` for Llama; all revisions pinned with shard SHA-256 | header, §2 |
| r3-4 | Novelty and scale context: 2609.26173, 2605.23901, 2608.08188, each added only after its full PDF was read | §1 |
| r3-5 | Storage note (about 65 GB of snapshots) | §10 |
| r3-6 (addition) | HQQ ≈ 4.5 vs NF4 ≈ 4.13 bits/weight: provenance tests stay norm-matched within each quantizer; any cross-quantizer comparison of raw divergence is labelled descriptive and not like-for-like | §9, §13 L8 |
| r3-7 (addition) | Phase 3 HQQ determinism test (quantize twice, bitwise); if not bitwise on the pod, quantize once, hash and freeze into the lock appendix before calibration | §9, §12 |
| r3-8 (addition) | Snapshot checks: Llama-3.2-3B tied embeddings and vocab size (and every model's), recorded in the manifest and compared at G7 | §2, §12 |

**r4 (corresponding author, 2026-11-07): pre-data amendment and re-lock. No cloud action and no study data before it. Hypotheses, arms, estimands, gates G1–G8, thresholds, reductions and the ceiling are unchanged.**

| # | Change | Implemented in |
|---|---|---|
| r4-1 | D5 GPU is the A40 (Secure Cloud, On-Demand) for the whole study, because the RTX A6000 was not available for network volumes. torch stays 2.13.0+cu126 (sm_86 covered). SKU checks, plan numbers, runbook and tests updated. Ceiling stays $150 | §10, §11, `v4_ops`, `pod_setup.sh`, `v4_plan_numbers` |
| r4-2 | The reload double-load is fixed at all three sites, with a no-double-load test and mutants | `run_v4.ModelSlot`, §12, §16 |
| r4-3 | Deterministic settings added. GM logs driver version, CUDA version and ECC mode | §5, `rv4`, `tools/v4_gpu_mutation_check.py` |
| r4-4 | Container image digest recorded at first boot; the tag stays the locked one | §11, `v4_ops`, `pod_setup.sh` |
| r4-5 | Amendment path in the lock tool (old lock preserved, amendment record). `DEVIATION_LOG_V4.md` created, with this amendment recorded | `tools/lock_v4.py`, §16 |
| r4-6 (found while amending) | `--verify` now also compares `HEAD` with the lock; before, a later commit that changed a locked file still verified. The runbook pushes the commit that contains the lock file; the locked commit itself does not contain it | `tools/lock_v4.py`, `RUNBOOK_V4.md` |
| r4-7 (found while amending) | The ceiling's stated multiple is corrected to about 2.5× of $61, and the L40S bandwidth comparison is restated against the A40. The ceiling itself is unchanged | §10 |

**r5 (corresponding author, 2026-11-07): pre-data amendment and re-lock, for a hard ceiling of $30 in total. No cloud action and no study data before it. Hypotheses, arms, estimands, gates G1–G8 and thresholds are unchanged.**

| # | Change | Implemented in |
|---|---|---|
| r5-1 | Hard ceiling $30 in total, storage included; prepaid balance $30. The request also says "the $40 ceiling", so the margin is reported under both | §10, `v4_ops` |
| r5-2 | D4: HQQ dropped entirely, with its code, tests, mutants, gate branch, lock line and package | §9, `rv4`, `run_v4`, `analyze_v4`, `lock_v4` |
| r5-3 | Test n = 100 (first 100 of the seeded 200), calibration n = 50; power recomputed for H1–H3 and H5. H4's permutation test is unaffected (model-level) | §3, §7.6, `run_v4`, `v4_sample.json` |
| r5-4 | Network volume 100 GB; storage prorated to the expected duration; deletion right after the verified download | §10, §11, `v4_ops`, `RUNBOOK_V4.md` |
| r5-5 | Run order: Step 0, then the Qwen/Llama ladder, Mistral last and first to drop | §10, `run_v4` |
| r5-6 | Reduction order: drop Mistral, then stop and report. No other automatic cuts, because H4 needs all seven Qwen/Llama models | §7.4, §10, `run_v4`, `analyze_v4`, `recompute_v4_independent` |
| r5-7 | Projection at $0.59/h with billed-hour overhead; margin reported | §0, §10, `v4_plan_numbers` |
| r5-8 | The `UnicodeDecodeError` answered: it came from the validation launch, and the harness never runs on the pod. A real encoding failure was found in `v4_remote.py` and fixed | §11, §16, `v4_ops`, `tools/v4_remote.py`, V4-A2b |
| r5-9 (needed to make r5-1 and r5-4 work) | $1 close-out reserve: the run stops at $29. `run_v4.py closeout` gates the deletion. The storage ledger stops at `volume_deleted_utc` | §10, §11, `v4_ops`, `run_v4` |
| r5-10 (needed by r5-3) | Step 0 criterion 3 compares with v3 on the same first 50 problems | §4, `analyze_v4` |
| r5-11 (needed by the tight budget) | The projection counts GM, first-boot setup and the G8 reruns, which r4 left out | §10, `v4_plan_numbers`, `run_v4` |

**r6 (corresponding author, 2026-11-07): pre-data fix of the lock tooling, found when `pod_setup.sh` could not verify lock ae5bd6db5ce48b7a on the pod (V4-A7). GM had not run; no study data existed. Hypotheses, arms, estimands, sample, gates, thresholds and ceiling are unchanged.**

| # | Change | Where |
|---|---|---|
| r6-1 | The previous locks (V3, E1, E1L) are hashed with `sha_lf`, on writing and on verifying, through one helper `file_sha`. Raw bytes differed between the laptop's CRLF checkout and the pod's LF checkout of the same committed text | `tools/lock_v4.py` |
| r6-2 | The same helper hashes the appended run files (GM, Step 0, design, flag, calibration), on appending and verifying, and `run_v4.locked_line` uses the same LF-normalized hash. These JSON files travel between the laptop and the pod (`step0.json` is written on the laptop); before r6 they matched only because they were copied byte for byte | `tools/lock_v4.py`, `run_v4` |
| r6-3 | The known_hosts option of `v4_remote.py` is quoted. OpenSSH splits an unquoted `UserKnownHostsFile` at whitespace, so with the repository path's spaces the pin went to a stray file (V4-A6) | `tools/v4_remote.py` |
| r6-4 | Command syntax in the runbook and the tool's usage text: options before the subcommand; `MSYS_NO_PATHCONV=1` for Git Bash | `RUNBOOK_V4.md`, `tools/v4_remote.py` |
| r6-5 | Search of all v4 tooling for other raw-byte hashes of text files that could differ between Windows and Linux: none other. Locked files are hashed from committed blobs (`git show`); model shards, checkpoint tars and record files are binary or copied byte for byte; the lock id is computed from the decoded text; `e1_sample.json` already uses `lf_sha`; the Step 0 author-decision file hash is recorded, not compared | — |
| r6-6 | Tests: o38 (CRLF and LF copies verify, for previous locks and appended files; `locked_line` agrees), o39 (a path with a double space stays one file under every ssh found, through `ssh -G`). Mutants: raw-byte hashing in `lock_v4`, raw-byte hashing in `run_v4.locked_line`, the unquoted known_hosts path | `tests/test_ops_v4.py`, `tests/mutation_check_v4.py` |

**r7 (corresponding author, 2026-11-07): pre-data fix of three test-code device bugs, found when GM failed 15/18 on the A40 under lock 349d0acad9712e91 (V4-A9). No study record exists. The study code, hypotheses, arms, estimands, sample, gates, thresholds and ceiling are unchanged.**

| # | Change | Where |
|---|---|---|
| r7-1 | t08 builds its BF16 input on the model's device (`device=DEV`); before, a CPU tensor reached the CUDA BF16 forward | `tests/test_rv4.py` |
| r7-2 | t12 moves the manual sums to the CPU before `.numpy()` | `tests/test_rv4.py` |
| r7-3 | t13 compares the accumulator state and the manual ratios on the CPU | `tests/test_rv4.py` |
| r7-4 | Checked on the laptop's RTX 4060 (CUDA) before the pod: `test_rv4` 18/18 on CUDA, the full mutation check 87/87 on CUDA, and a smoke test of the run path for qwen2.5-0.5b through `run_v4`'s own functions (snapshot check; gates G1, G2, G3, G4, G5, G7 all passed; 3 calibration and 3 test problems with the G8 rerun bitwise equal; no integrity failure). The smoke test wrote only to a scratch directory outside `v4_runs`, marked as not data, and was deleted; the pod-only guards (authorization, A40 SKU, GM, order, token files) were not exercised there. No further device bug was found | — |
| r7-5 | From r7 on, every change to the GPU path or its tests is run on a local CUDA GPU before the pod | procedure |

**r8 (corresponding author, 2026-11-08): pre-data fix after GM failed 83/87 under lock e5e53c243e044bad (V4-A12): the four Step 0 mutants survived because test a16 returned without checking when `research_v2/v3_runs` was absent, and `v3_runs` is not tracked, so it never reaches the pod. No study record exists. Hypotheses, arms, estimands, sample, gates, thresholds and ceiling are unchanged.**

| # | Change | Where |
|---|---|---|
| r8-1 | Tests that passed without their inputs now fail loudly. The search covered every conditional `return`/`continue` in the five test files and the GM harness (the hits inside mutants are copies of production code, not skips). Three found: a16 (returned when `v3_runs/qwen-base/tierA/NAT.jsonl` was absent), a04 (skipped a model whose `v3_runs/<label>/calib` was absent), o39 (checked nothing when no ssh executable was found). a04 and a16 now assert their records exist; o39 asserts at least one ssh was found | `tests/test_analyze_v4.py`, `tests/test_ops_v4.py` |
| r8-2 | The locked v3 records the Step 0 tests read (for qwen-base, llama-instruct and llama-base: `tierA/*`, `calib/*`, `calibration.json`; 63 files) are listed with their SHA-256 in the committed `v4/v3_records_manifest.json`, packed by `tools/v4_v3_records.py pack`, copied to the pod and extracted into `/workspace/repo/research_v2/v3_runs` (the push does not clean untracked files). `v4_v3_records.py verify` checks every file against the manifest and each `calibration.json` against the `CALIBRATION <label>:` hash appended to `V3_PROTOCOL_LOCK.txt`. The V3 lock hashes only `calibration.json`; the other 60 files are bound by the manifest, which is under the v4 lock | `tools/v4_v3_records.py`, `v4/v3_records_manifest.json` |
| r8-3 | GM verifies the v3 records first; if they do not verify, the suites do not run and GM fails. The GM pass rule now also requires the verified records | `tools/v4_gpu_mutation_check.py` |
| r8-4 | Tests: o25 (the GM rule fails without verified records), o40 (verification fails on a changed file, a missing file, a missing manifest, a calibration hash other than the V3 lock's, and a lock without CALIBRATION lines; the repository manifest verifies on the laptop records), o41 (a04 and a16 fail without `v3_runs`). Mutants: GM ignores the records, hash not checked, V3 lock not checked, a16 returns without records, a04 skips missing records | `tests/test_ops_v4.py`, `tests/mutation_check_v4.py` |
| r8-6 | (r8b) The manifest `v4/v3_records_manifest.json` and `tools/v4_v3_records.py` are added to the locked file list; the first r8 lock (3ed7b28e64c3b636) omitted them although r8-2 says the manifest is under the v4 lock. That lock authorized nothing and was replaced before any use. Test o40 checks both are locked; mutant: both dropped from the list | `tools/lock_v4.py`, `tests/test_ops_v4.py`, `tests/mutation_check_v4.py` |
| r8-5 | Standing rule (corresponding author): before every pod run, the full GM equivalent runs on the laptop RTX 4060 in a clean clone of the bundle commit (LF, as on the pod) plus exactly the files the pod receives (the same v3 records tar, the authorization copies and the cost config), through GM's own harness; the pod starts only if every test passes and every mutant is caught (the first-boot image record exists only on the pod) | procedure |

**Deviation D1 (corresponding author, 2026-11-08): post-data, after the drift stop on qwen2.5-1.5b (V4-A17). Decided from timing data only (the Step 0 per-problem times and the G6 formula), before any test result of a ladder model was seen. Hypotheses, arms, estimands, sample, gates, thresholds, the $29 ledger stop and the author's $27 condition are unchanged. A pre-data amendment is no longer possible (`--amend` refuses after data), so D1 is recorded with the new post-data path below; the lock id stays b0b191df9bee180e.**

| # | Change | Where |
|---|---|---|
| D1-1 | The drift reference of every remaining ladder model is the G6-design projection times 1.48, the ratio of the real per-problem time to the G6 formula measured in Step 0 (qwen2.5-0.5b 28.3 / 19.0 s, llama3.2-1b 19.9 / 13.4 s). The tolerance stays 25 %: a model stops if its mean time exceeds 1.25 x 1.48 x projection. The design itself (design.json, appended) is not revisited | `run_v4.py` (`DRIFT_REF_FACTOR`) |
| D1-2 | Resume check: before a model's test resumes after a restart, its last completed problem is rerun (no record written) and must be bitwise equal to its stored arm records (wall time ignored), FP16 history and NAT slots; logged in `resume_check.jsonl`, a difference writes `resume_check_failure.json` (a stop file) and stops. qwen2.5-1.5b keeps its records and resumes under the exactly-once rules; its `drift_stop.json` from the old reference is renamed `drift_stop_superseded_D1.json` on the pod | `run_v4.py`, `v4_ops.py` |
| D1-3 | Post-data deviation records: `lock_v4.py --deviation ID TEXT` (laptop, after the changed files are committed; refused before any record exists) writes `v4/deviations/V4-<ID>.json` with the new SHA-256 of every locked file that differs from the lock, the commit and the reason, and adds the deviation-log row; `--append-deviation ID` (pod, before any further GPU use) appends the record's hash to the lock. `--verify` then holds each listed file at its deviation hash only, and requires the record and its log row | `tools/lock_v4.py` |
| D1-4 | GM again after a deviation: GM writes `gm_<ID>.json` (the first GM record stays), `--append-gm` adds one `GM <ID>:` line, and the pod commands require that GM | `tools/lock_v4.py`, `tools/v4_gpu_mutation_check.py`, `run_v4.py` |
| D1-5 | Tests o42 (deviation record, append and verify), o43 (drift reference and GM after a deviation), o44 (resume check, and run_test calls it first); seven mutants | `tests/test_ops_v4.py`, `tests/mutation_check_v4.py` |
| D1-6 | Before the pod: the r8-5 rehearsal on the laptop RTX 4060 in a clean clone of the deviation commit with the deviation appended to the clone's lock, plus exactly the files the pod receives; then GM on the pod | procedure |

## 16. Phase 3 implementation notes (pre-lock, 2026-11-03; r4 and r5 rows 2026-11-07)

These notes fix details that §3–§13 leave to the implementation. They are part of the locked text.

| Item | As implemented |
|---|---|
| FP16 path | Exactly v3's: the locked `DynamicLinear`, `stored_entry_v3`, `to_fp16` and FP16 oracle. A test checks every core arm against `rv3.core.teacher_forced_arm_v3` bitwise. |
| BF16 rule (§5) | Both weight copies, the activations and the cache are BF16. The entry is built in float64 with v3's draws and stored as the correctly rounded BF16, against an independent integer-arithmetic oracle. Ties, subnormals and overflow are tested; rounding through float32 first is shown to differ. |
| Measurement | The 4-bit pass uses the global routing flag (NF4). No predictors, no query capture. |
| X13a | Per problem: an npz with the per-entry norms and S1 for NAT, float64. The isotropic null uses the model's own head_dim, from the X13 functions unchanged. The per-layer C^P is descriptive. |
| Seeds | Arms: `rv3.core.seed_for` with the master label `rv4-scale-2026-11-02`, tiers V4calib, V4A, V4gate and V4g6. Analysis: one stream per summary, `default_rng([20261105, key…])`. |
| Prompt pin | `v4_sample.json` holds the SHA-256 of every test and calibration prompt. G5 checks them on the pod, along with round trip, BOS, EOS and the drift marker. |
| G7 | The snapshot directory must be the pinned revision, every shard must hash to the manifest, `consolidated.safetensors` must be absent, the config must equal the manifest geometry, tied flag and vocab, and the cache geometry and dtype must match. |
| G8 | The first 3 test problems, and the next problem after any restart, are run twice. The records, excluding wall time, and the X13a arrays must be bitwise equal. |
| Pod start (§10) | The start time of the container's PID 1 (`/proc/1/stat` over `/proc/stat` btime). **Correction:** inside a container, `/proc/uptime` gives the host's boot, so it is only a fallback. |
| Cost ledger | `cost_config.json` is written by the corresponding author: rate, idle auto-stop window, volume size and creation time, and from r5 the optional deletion time. The ceiling of $30 and the $1 reserve are fixed in code. The latest billing line wins a tie. A session still open at a billing time keeps its auto-stop allowance. |
| Secrets | RunPod passes pod variables to PID 1 only, so `HF_TOKEN` is read from `/proc/1/environ`, in memory. Patterns `hf_…`, `rpa_…` and `rps_…`, plus the literal values, are redacted and scanned for. Scans open tar, gzip and zip/npz members. |
| G6 design (r5) | `t_arm = a·layers + b·bytes`, solved from the qwen2.5-0.5b measurement and the llama3.1-8b probe. Remaining USD = (GPU-h × 1.2 + 0.25 h) × rate × 1.05, with GPU-h counting the 3 G8 reruns. The budget is $29 less the money used less the storage prorated to the remaining billed hours plus 48 h. First the full ladder, then without Mistral, then stop and report. |
| Step 0 | The v3 verdict categories are read from `v3/phase5/v3_report_locked.json`, and the v3 margins from `v3_runs/<label>/calibration.json`. r5: v3's ρ and b̂ for criterion 3 are recomputed on the first 50 problems from `v3_runs/<label>/calib/`, by the v4 calibration rules; the per-slot ratios are cut per problem, checked against the per-step records. |
| Reload (r4; r5) | `ModelSlot` holds the only reference to a process's model and releases it, with garbage collection and `empty_cache`, before the next load. Before r4, `free(be)` removed only its own argument, so the caller's reference kept the old model on the GPU during the reload. With about 21 GB per large model this would have put about 42 GB on the card, a tight fit in 48 GB. r5 removed the two HQQ reload sites, which leaves FP16 → BF16 after a G3 failure. |
| Determinism (r4) | `import run_v4` already initializes CUDA, through an import, but creates no cuBLAS context. The variable is therefore set at the top of each entry point before torch loads, and `apply_determinism()` runs first in `main()`. v3 did not use these settings. |
| Amendment (r4) | `lock_v4.py --amend REV TEXT`, refused once any appended line or study record exists, or if the old lock fails its own commit. The old lock goes to `v4/lock_history/V4_PROTOCOL_LOCK_<old id>.txt`. An `AMENDMENT` line enters the hashed text, and earlier ones are carried forward. A row from the old id to the new id goes to the deviation log. `--verify` checks all of these. The deviation log is not a locked file, because it grows after the lock. |
| Gated Llama access | Not possible from the laptop for 3.2-3B and 3.1-8B. Their geometry and snapshot checks use pinned bytes (§2). Llama-3.2-3B's tokenizer files are byte-identical to 3.2-1B's (G5 on the laptop). Llama-3.1-8B's G5 runs on the pod only. |
