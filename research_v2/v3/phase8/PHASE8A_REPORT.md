# Phase 8a report (2026-11-01): declarations and text revision of main_v3

Text-only revision of `paper_journal/main_v3.tex` in response to the second internal review. No new analysis was run on
the experimental records. The only new numbers are arithmetic on locked inputs: the share of NF4 linear weights in the
k/v projections (from the locked `config.json` files, sha256-checked against `V3_PROTOCOL_LOCK.txt`) and its ratio to the X8
share, a sign flip of the locked Gamma_0, and the unclamped calibration c. All three are generated macros in `make_v3.py`.

## 1. Author-supplied declarations

The texts were inserted verbatim. Provenance is in `v3/AI_USE_LOG_V3.md`, section "Provenance of author-supplied declaration
text":
- competing interests, funding (corresponding author's personal laptop GPU), the new "Approval" paragraph, and the CRediT
  roles of Srikar Reddy Gunupati, all in `09_declarations.tex`;
- the co-authors' CRediT roles are kept as [UNRESOLVED: roles pending from corresponding author];
- both [UNRESOLVED] markers of the responsibility statement (Section 3.8) were removed.

Two cautions:
- The approval statement attests for all three authors, and it was given before the Phase 8a changes.
- The removed second marker asked for each author's confirmation and a signed verification record. Neither is on file.

## 2. Phase 8a changes (review sections 2, 3 and 5, plus the extra items)

**Claims (review section 2)**
- Highlights: the "inherited" and matching highlights were replaced. The new ones cover natural-size values, family not
  tuning, and k/v versus other weights (74-78 characters).
- Abstract, intro and conclusion:
  - "inherited ... exceeded error" became divergence reached by INH versus LOC, defined as k/v weights versus all other
    weights.
  - The X9a deficit is now 0.7-5.3% (macro precision raised).
  - Matching claims are labelled exploratory and limited to "layer 0 part tested, inconclusive".
  - "Changed nothing" became "at most 0.004".
  - "Opposite signs" became "opposite-signed point estimates, neither resolved". It now appears once per section, not
    verbatim five times.
- The Holm statement in Section 3.5 now names the reproduction as the exception.
- The Llama REL arms are described as within 0.81-1.27 of a rough benchmark, not "as the dose slope predicts".
- The equivalence margins are stated to be set by the 0.10 clamp; the unclamped c was 0.063-0.099. This is in Section 3.3,
  Table 2, Section 4.2 and Limitations.
- The Qwen rounding caveat is now in the intro and Section 4.1.

**Readability (review section 3)**
- Terms defined at first use:
  - slot, perturbation-matched and privately locked (intro);
  - runtime gate (Section 3.7), Tier B (Appendices C and D), exposed problems (Section 4.6 and Appendix D);
  - P_out's X;
  - "v3" was removed.
- Section 2 now forward-references the arm and predictor definitions.
- Unique symbols:
  - decode step t and cache C_{t-1};
  - scored steps T_i;
  - Spearman Sp(.);
  - subspace share phi;
  - cross-model gaps Gamma_0/Gamma_1 and fraction f_x;
  - families F_m, F_A, F_2x2 (the label "X" is gone);
  - Gaussian draw z;
  - geometric mean GM.
- NAT-C1 means only theta(NAT, C1). beta_0 is called the step-level mean log ratio.
- Cross-model gaps in the text use Llama-instruct minus Qwen-base. The locked Gamma_0 sign is explained.
- Verdict vocabulary: difference / equivalent / inconclusive for contrasts, and supported / contradicted for predictions.
  H-A now reads "contradicted", not "falsified".
- Protocol arm names (NAT-K* etc.) moved from the Table 1 caption to Appendix D.
- "200 of 200" now reads "200 of 200 problem records (50 per model)".
- The Figure 1 legend is outside the axes.
- Authorization, manifest and tool detail moved from Sections 3.6-3.8 to Appendix D and a new Appendix E (`J_aiuse.tex`).

**Tuning, H-D, H-C and exploratory language (review section 5)**
- Section 4 was restructured in this order: provenance and H-D, four case studies ("family, not tuning"), origin, predictors,
  H-C, robustness.
- The tuning prediction is stated as contradicted (Table 5 has a new "Pre-declared prediction" column). Family-not-tuning is
  a headline result in the abstract, highlights, intro and conclusion.
- H-D now says its direction largely follows from rho_V/rho_K and the dose slope.
- H-C was removed from Table 4 and Figure 1 discussion. Its bands and p-values appear only in Table A.14 and Appendix A.
- X4, X5, X8 and X9a-d use non-confirmatory wording and carry [exploratory] labels in the abstract, intro and conclusion.
- X9a now notes that the top-8 share is coarse (r80).
- X8 gains the per-parameter reading: LOC holds 1.5% (Qwen) and 3.4% (Llama) of the NF4 linear weights. Its share of NAT's
  divergence is 18x (Qwen) and 7x (Llama) its weight share.
- "Practical implications" became "Hypotheses for practice", with no deployment claims.
- The in-session reproduction now states what it rules out (the particular isotropic draws, pipeline nondeterminism between
  sessions; the master seed is new) and what it does not (independent replication, a model-specific accident).

**Related work**
- Added IntactKV, Quant-Noise, DiffQ and Widrow & Kollar, each verified from primary sources (see `PHASE8A_LITERATURE.md`).
- The uncited "common stand-in" sentence was replaced by a sourced statement.
- "What we did not find" now acknowledges IntactKV.
- The "venue verified" notes moved to a `verified` bib field that does not print.

## 3. q/k/v projection biases (finding only; not added to the paper)

- Qwen2.5-0.5B (both checkpoints) has bias terms on the q, k and v projections. They are hard-coded `bias=True` in
  transformers' `modeling_qwen2.py`, lines 200-202, and present as tensors in the checkpoint (e.g.
  `model.layers.0.self_attn.k_proj.bias`, shape [128]). o_proj and the MLP have no bias.
- Llama-3.2-1B (both checkpoints) has `attention_bias: false` and `mlp_bias: false` in config.json, and no bias tensors.
- In the dual-weight implementation (`src/dynamic_linear.py`), bitsandbytes quantizes only the weight. The FP16 bias is
  shared by the FP16 and NF4 branches.
- Consequence: Qwen's k/v biases add the same FP16 term to kappa^F and kappa^M. They cancel in Delta but enlarge
  ||kappa^F_K||, which lowers the relative error rho_K. This is one candidate for the large-norm, small-relative-error key
  slots behind the REL-K inflation, and a family difference that the Discussion does not list. It is untested.

## 4. Build and audit

- `make_v3.py`: 1167 macros.
- `main_v3.pdf`: 55 pp, 0 LaTeX errors, 0 overfull boxes, 0 undefined references.
- `audit_v3_numbers.py`:
  - 46/46 summary claims hold (9 new Phase 8a claims);
  - 0 undefined macros and 0 unclassified literals;
  - 1 recomputation item, the known Monte-Carlo endpoint `hc.lb.fci`.
- The supplement was not rebuilt (no source change).
- Nothing is committed.

## 4b. Acceptance edits (author decision, 2026-11-01)

1. **Verification wording.** The per-author verification wording in Section 3.8 and in the generative-AI declaration was
   replaced, verbatim, by: "The corresponding author reviewed the methodology, implementation, results and claims; the
   co-authors reviewed and approved the final manuscript." This removed the "each author to confirm" marker. AI_USE_LOG_V3.md
   notes that the co-authors' approval must be re-confirmed on the final PDF.
2. **Bias sentence.** One untested candidate was added to the Discussion (precedents paragraph): Qwen2.5 has q/k/v projection
   biases and Llama-3.2 none. The citations are the Qwen2.5 technical report, the configurations and Transformers
   (`wolf2020transformers`, verified).
3. **Licences.** These were checked from primary sources (PHASE8A_LITERATURE.md) and drafted as a new "Licences" paragraph:
   Qwen Apache-2.0, Llama 3.2 Community License with its notice, and GSM8K MIT. One marker remains for the release plan,
   because the Llama obligations apply only if Llama materials are distributed. The Ethics marker is reduced to "re-confirm".
4. **Earlier-study Claude Code versions.** Read from the commit trailers and stated in Appendix E: Opus 5.5 on
   2026-10-22..23, Sonnet 5.5 on 2026-10-23..28. Sonnet 5 appears only on 2026-09-29/30, in the routing work. The marker was
   removed.

Build after these edits: 56 pp, 0 errors, 0 overfull boxes. Audit: 46/46 claims, 0 unclassified literals, 1 known
Monte-Carlo item.

## 5. Remaining [UNRESOLVED] fields (before the acceptance edits; see 4b for the four that were resolved or changed)

Appendix D = `C_protocols.tex`; Appendix E = `J_aiuse.tex`.

**`09_declarations.tex`**

| Line | Paragraph | Marker |
|---|---|---|
| 1 | Author-review notice | generic `\unres{...}` legend |
| 3 | Declaration of generative AI | ChatGPT product/model/version/dates; "each author to confirm" |
| 5 | Authors and affiliation | order, spelling, corresponding author, affiliation, e-mails, ORCIDs |
| 7 | CRediT | roles of Shrinivas Anavekar and Alen K Varghese |
| 15 | Ethics | re-confirm; model and GSM8K licence terms. The paragraph still begins "Proposed text" |
| 17 | Data and code availability | release decision |
| 19 | Previous submissions | verify the statement; journal declaration |
| 21 | Acknowledgements | authors |

**`C_protocols.tex` (Appendix D)**
- Line 86, "Previous submissions": not under consideration elsewhere; ARR #283 status; TMLR #13386 decision date; no
  preprint.
- Line 94, "Artifacts": public release.

**`J_aiuse.tex` (Appendix E)**
- Lines 7 and 9: ChatGPT version/dates, and which uses applied.
- Line 13: Claude Code versions and date range in the earlier studies.
