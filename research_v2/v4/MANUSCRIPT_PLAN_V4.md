# Manuscript plan: main_v3 → main_v4 (proposal for author review; no manuscript file edited)

Date: 2026-11-09. Sources: `v4_runs/v4_report.json` (locked, sha256 df57a314…), `v4_runs/v4_recomputation.json` (214/214 within
tolerance), `v4/exploratory/X_RESULTS_V4.md` (X1-X4, plan 1409b94, results b4eecc6), `paper_journal/main_v3.tex` and `sections_v3/`.

## 1. Proposed title

**Key–value asymmetry of weight-quantization error in the KV cache holds from 0.5B to 8B parameters; the family effect does not**

Alternatives:
- (a) *Weight-quantization error in the KV cache across scale: a robust key–value asymmetry and a corrected family effect in eight language models*
- (b) keep v3's opening, *Weight-quantization error in the KV cache is not isotropic noise*, and add *: a pre-specified (privately locked) scale study across three model families*

## 2. Abstract (draft, 249 words)

> Decoding with NF4-quantized weights writes key–value (KV) cache entries that differ from full-precision ones. We ask whether the structure of this weight-induced cache error, first characterized in four small models, holds
> across scale and in a third family. In eight base checkpoints (Qwen2.5 0.5B–7B, Llama-3.2/3.1 1B–8B, Mistral-7B-v0.3) we
> stored, at every decode step, either the natural cache entry or a substitute whose error had the same norm per layer, KV head
> and path. We then measured next-token Jensen–Shannon divergence on 100 GSM8K problems per model. Hypotheses, decision rules and
> analysis code were locked before any test data. One post-data operational deviation, to a runtime-drift stop reference, is
> reported. In every model, natural value error was more disruptive than isotropic noise of the same norm, values-only error was
> more disruptive than keys-only error at natural size, and the error shared a component across decode positions far above the
> isotropic share. Natural key error was milder than isotropic noise in seven models. This advantage shrank with size in both
> multi-size families and was inconclusive in Mistral-7B. The pre-specified (privately locked) prediction that the sign of the
> natural-versus-isotropic contrast follows model family was falsified. The contrast was negative only in the smallest Qwen
> model, near zero in the smallest Llama, and positive in every larger model, including Mistral. The family difference reported
> in our earlier four-model study is therefore confounded with size, and with head dimension, which changes from 64 to 128 at the
> same step. An independent recomputation reproduced all 214 locked quantities.

## 3. Highlights (each ≤ 85 characters, Elsevier)

The JAIR template has no highlights; these are kept for other venues. Revised 2026-11-11 after the fresh-eyes review.

1. NF4 value-cache error is harsher than equal-norm noise in all eight checkpoints.
2. Values-only cache error exceeds keys-only error at natural size in every model.
3. The error shares a cross-position component far above isotropic in every model.
4. Key-path mildness held in 7 of 8 models; its size trend is confounded with head_dim.
5. The predicted family-dependent sign was falsified; a family offset is unresolved.

## 4. The framing change

- **Main result:** the locked v4 study, with 8 models, 3 families and one shared protocol.
- **v3** becomes the detailed small-model mechanism study: 2 models × base and instruct. It is the source of the questions v4
  retested, and of mechanism results v4 did not retest:
  - origin (K/V projections against the other weights);
  - X13 common-mode removal;
  - first-order logit energy;
  - the predictors H-A, H-B and H-C;
  - the 2 × 2 tuning design.
- **Explicit correction (new paragraph, in the Results H4 subsection and again in the Discussion).** Proposed wording:

  > *Correction of our earlier reading.* In the four-model study (Section 6), NAT−C1 was negative in both Qwen-0.5B models and
  > non-negative in both Llama-1B models, and we read the difference as tracking family, not tuning. With four Qwen and three
  > Llama sizes, the pre-specified (privately locked) family prediction (Qwen < 0, Llama ≥ 0) was falsified. Three of the four Qwen models had
  > NAT−C1 entirely above 0, and the permutation test gave p = 0.257 with no separation. The negative sign appears only at the
  > smallest Qwen size. It rose with size in both families at similar rates (exploratory: +0.52 and +0.63 log units per decade
  > of parameters). The earlier family contrast compared one size per family. Its sign difference is better described as a
  > small-model effect, and these data cannot separate it from the 64 → 128 change in head dimension. The tuning reading (no
  > base-to-instruct effect at 0.5B/1B) is not retested here, because v4 uses base checkpoints only.
- **What stays from v3, stated as is:** H1, H2 and H3 directions and H5 in four small models, all now replicated at scale.
  Rounding-corrected controls. The origin result. The X13 mechanism (a third of the V-path gap). The first-order benchmark.
  Step 0 shows that v3's three base and instruct estimates reproduce on the A40.

## 5. Section outline (main_v4)

1. **Introduction.**
   - The question: does the error structure hold across scale and family?
   - The four contributions:
     - (i) H2, H3 and H5 hold in all 8 models;
     - (ii) H1 holds in 7 of 8 and shrinks with size;
     - (iii) the family prediction is falsified, which corrects v3;
     - (iv) a fully locked cloud protocol: a $30 ceiling, $12.08 spent, recomputation, every deviation logged.
   - "To our knowledge" wording; no "first".
2. **Related work.** v3's section, plus the §1 literature update the protocol requires on scale studies of weight-quantization
   error in the KV cache: 2609.26173, 2605.23901 and 2608.08188, all read in full during Phase 1.
3. **Methods.**
   - 3.1 Models: an 8 + 1 table with family, size, layers, KV heads, head_dim, revision hash and licence. Llama-3.2-1B-Instruct
     is Step 0 only.
   - 3.2 Quantization and the controlled step: NF4, the teacher-forced history, the arms NAT, C1, NAT-K, C1-K, NAT-V and C1-V.
     Condensed from v3 §3.1–3.2.
   - 3.3 Calibration and margins: b̂, c, ρK and ρV per model, from v3's rules, n = 50 calibration problems.
   - 3.4 Hypotheses and decision rules H1–H5: Holm families, the H4 permutation rule, Mistral out of sample, the with- and
     without-flagged reporting.
   - 3.5 Step 0: the cross-hardware check against v3's records.
   - 3.6 Locks, the deviation, verification and recomputation: lock b0b191df9bee180e plus D1, the tolerances declared beforehand,
     214 compared quantities.
   - 3.7 Compute and operation: one A40 at a time, the cost, determinism.
   - 3.8 AI use.
4. **Results of the locked tests.**
   - 4.1 Step 0, all three models passed.
   - 4.2 Value path and natural-size ranking (H2, H3): a table and a forest plot of 8 models × 3 contrasts.
   - 4.3 Common mode (H5).
   - 4.4 Key path (H1): 7 of 8 supported, Mistral inconclusive (Holm p 0.054).
   - 4.5 Family test (H4), falsified, with Mistral out of sample, plus the correction paragraph.
   - 4.6 BF16 flags (none) and the with- and without-flagged versions (identical).
   - 4.7 Locked sensitivity analyses: excluded exposed problems, steps 2–8 and 9–64, the per-problem median, rounding truncation.
5. **Exploratory analyses (v4, not blind, plan 1409b94).** Every subsection is labelled exploratory:
   - X1, size trends;
   - X3, H1 against NAT-C1;
   - X4, head_dim confounding;
   - X2, per-layer common mode, briefly, with tables in the supplement.
6. **The small-model mechanism study (v3).** Condensed from v3 §4–5, framed as the detailed study in Qwen2.5-0.5B and
   Llama-3.2-1B, base and instruct:
   - 6.1 design and the locked v3 results, in a table;
   - 6.2 origin, K/V weights against the other weights;
   - 6.3 common mode and norm-corrected rounding (X13);
   - 6.4 the first-order benchmark and the predictors (H-A, H-B, H-C);
   - 6.5 the 2 × 2, with the v4 correction cross-referenced.
7. **Discussion.**
   - What is robust: the value-path harshness and the common mode.
   - What depends on scale: the key-path advantage.
   - The correction and why one size per family misled us.
   - The head_dim and size confound.
   - What base-only v4 says, and does not say, about tuning.
8. **Limitations.**
   - One quantizer (NF4; HQQ dropped in r5) and one task (GSM8K, teacher-forced).
   - n = 100 per model.
   - Three families, one of them a single model. With 7 models the H4 test is coarse: its smallest attainable p is 0.029.
   - Size and head_dim are not separable.
   - Exploratory analyses are not blind.
   - D1 is post-data. It is operational and touches no outcome or decision rule.
9. **Conclusion.**
10. **Declarations.** Data and code availability: the lock, records, hashes and recomputation.

## 6. What moves to the appendix and supplement

| Material | v3 location | v4 location |
|---|---|---|
| v4 per-model tables (CI90, raw p, sensitivity, ρ, margins) | none | Appendix A (new) |
| v4 protocol, lock lines, D1 record, deviation log V4-A1…A20 | none | Appendix B (new), condensed table; full log in the supplement |
| Step 0 table (12 rows) | none | Appendix C (new) |
| Cloud operation and cost | none | Appendix D (new) |
| Recomputation tolerances and the result | v3 B (v3 recomputation) | Appendix E: v4 recomputation, with v3's kept |
| v3 locked results detail (§4.1–4.6) | main text | §6 (condensed) and Appendix F |
| v3 exploratory (§5: absolute divergence, X13, origin shares, tuning, predictors, equal-relative matching, first order) | main text | Appendix G; §6 keeps X13 and first order |
| v3 H_v3extra, I_xdetails | appendix | Appendix F/G (merged) |
| Background on the earlier studies (G_background) | appendix | Supplement (keep a short paragraph in §6) |
| C_protocols (v3 locks) | appendix | Appendix B, with v4 |
| X2 per-layer C^P tables | none | Supplement |
| J_aiuse | appendix | Appendix H, updated with v4 roles (pod lifecycle via MCP, analysis, recomputation) |

**New figures:**
- (1) a forest plot of H1, H2 and H3 per model, ordered by family and size;
- (2) NAT−C1 against log10 params by family, with markers for head_dim 64 and 128, and Mistral shown separately;
- (3) H5 K and V per model.

They are generated from the locked report by a `make_v4.py` in the style of `make_v3.py`.

## 7. Full v4 deviation and operation summary (for Appendix B and D)

**Amendments before any data:**

| Entry | Revision | What changed |
|---|---|---|
| V4-A1 / A1b | r4 | GPU set to the A40 (Secure, On-Demand), torch 2.13.0+cu126, ceiling $150; verify and push rules |
| V4-A2 / A2b | r5 | Ceiling $30 including storage, stop at $29; HQQ dropped; test n = 100, calibration n = 50; volume 100 GB; Mistral last; H4 needs all seven; the decode-error question answered |
| — | r6 | Lock hashes made line-ending-neutral (CRLF/LF); quoted known_hosts path |
| — | r7 | CUDA device fixes in `test_rv4.py` |
| — | r8 / r8b | Tests fail loudly without their inputs; v3 records manifest, verified by GM |

The final lock is **b0b191df9bee180e**.

**Operational and authorization entries:**

| Entry | What it records |
|---|---|
| A3 | D8 changed: Claude Code ran the pod lifecycle through the RunPod MCP, each action approved by the author; volume deletion denied in settings; HF token held only as a RunPod secret |
| A4–A7 | GM attempt 1: volume and pod created; the lock failed on CRLF |
| A8 | r6 GM authorization |
| A9 | GM 15/18 (CUDA test bugs) |
| A10, A11 | r7 GM and Step 0 authorizations; pod replaced after its GPU was lost |
| A12 | GM 18/18, mutants 83/87, because inputs were absent on the pod (led to r8) |
| A13–A15 | Rehearsal on the RTX 4060; pod replaced; **GM passed** (18/18, 93/93); **Step 0 passed** (3 models, FP16) |
| A16 | Ladder authorizations; G6 design at a $11.62 projection |
| A17 | **Drift stop** on qwen2.5-1.5b: 31.9 s against 23.0 s × 1.25. The G6 fit underpredicted real time by about 1.37–1.48× |
| A18 | D1 authorizations |
| A19 | Ladder complete, and how it was operated |
| A20 | Closeout fetch |

**Deviation D1** (post-data, operational, record `v4/deviations/V4-D1.json`, commit 7c4f558, record hash f6c95590…):
- The drift reference is the locked design projection × 1.48, with 25 % tolerance.
- Any resumed test must first rerun its last completed problem bitwise.
- GM was rerun under D1 (18/18, 100/100), and the `DEVIATION D1:` and `GM D1:` lines were appended to the lock.
- No outcome, arm, sample, hypothesis or decision rule changed.
- qwen2.5-1.5b's resume check was bitwise equal.
- Every model then ran inside the D1 limit. Measured s/problem:

  | qwen 1.5b | llama 3b | qwen 3b | qwen 7b | llama 8b | mistral |
  |---|---|---|---|---|---|
  | 28.2 | 40.7 | 37.0 | 35.1 | 44.1 | 45.8 |

**How it was operated:**
- One A40 Secure pod at a time in CA-MTL-1, on a 100 GB high-performance network volume.
- Pods were replaced when a restart could not get its GPU back, each with the author's approval: 0o2qt1zg8uenfh →
  1gh2wczbuxheeo → wletc98r6v348k → pnlub2js003kue, then fetch-only q28xtpej8k1sdv (A20).
- The ladder was driven from the laptop over SSH (`tools/v4_remote.py`). Each model's checkpoint was fetched and verified as it
  finished. There was no SSH drop, and the on-pod fallback driver (`v4/ops_fallback/`, 070ee8c) was prepared but never launched.
- Before each pod run, a rehearsal on the RTX 4060 in a clean LF clone.
- **Closeout:** 1,107 of 1,107 volume files verified on the laptop by sha256. The author then deleted the volume and pods.

**Cost:**
- 15.6 GPU-hours.
- RunPod billing total $12.08 (pods $11.31, storage $0.78), against the $30 ceiling.
- The in-pod ledger read $11.62 at the end of the ladder.

**Analysis:**
- The locked `analyze_v4.py` (seed 20261105) and the independent `recompute_v4_independent.py`: 214 of 214 quantities within the
  tolerances declared beforehand, with identical decisions.
- Exploratory X1–X4: plan committed before computing (1409b94), results in b4eecc6. Not blind and descriptive only.

## 8. Decisions for the author

1. The title: the main proposal or alternative (a) or (b).
2. Whether to keep v3's four-model results in the main text as §6, as proposed, or move all of them to an appendix with a short summary.
3. Whether the correction should also go out as a note or erratum on any v3 version that has been circulated or submitted.
4. The authors and affiliations stay as in v3, unless you change them.
5. The target journal: stay with Neurocomputing, or choose another venue now that the paper is a scale study.
6. How much of the cloud operation and cost belongs in the main text (proposal: §3.7, one paragraph; details in Appendix D).
