# Phase 7c — manuscript revision report (2026-11-01)

**Status: done; stopped for author review.** No GPU was used in 7c. Every new number in the manuscript is a generated
macro.

## Build and audit

| Item | Result |
|---|---|
| `main_v3.pdf` | 52 pp (49 before). Main text pp. 1–32, declarations to p. 34, references to p. 37, appendices pp. 37–52. No LaTeX error, undefined reference, undefined citation or overfull box. |
| `supplement_v3.pdf` | 32 pp (title updated). Six overfull boxes, all in the carried-over earlier-study material; none comes from the title. |
| Abstract | 187 words (limit 200), counted from the rendered PDF |
| Highlights | 78, 76, 75, 74 and 74 characters (limit 85) |
| Number audit (`audit/audit_v3_numbers.py`) | 1,144 macros, 866 used, 0 used but undefined. 281 values re-checked against the independent recomputation; 1 failure, the known Monte-Carlo-sensitive Llama-base H-C fraction CI, which enters no decision. 168 literals, 0 unclassified. **38/38 prose summary claims hold**, 23 of them new: X1, X4, X5, X7, X9a–d, X10, X11, X12, the X9 reproduction, the NAT arithmetic JS = E1/E1L, Holm < 0.05 for every locked test cited in the abstract/conclusions, and the interaction Holm in both analyses. |
| `??` macros in the PDF | none |

## New exploratory analysis added in 7c

X12, the per-family tuning effects, was required by the review item "per-family tuning effects". Its plan entry
(`fcdc74d`) went in before the script (`7b25a27`) and the results (`6aec144`). The results are Qwen −0.082
[−0.175, +0.011] and Llama +0.066 [−0.017, +0.150], Holm 0.169 for both. Neither is resolved; the locked interaction rests
on their opposite signs. The check against the locked interaction is about 0 (1e-17).

## Framing and structure

- **Title:** *Weight-quantization error in the KV cache is not isotropic noise: key–value asymmetry in four small language
  models*. The supplement's title was changed to match.
- **Lead order** in the abstract, introduction and conclusion:
  1. key error milder and value error harsher than isotropic noise of equal norm, in all four models, with the mechanism
     open (the X9a subspace account is real but explains almost none of it);
  2. value dominance at natural size;
  3. inherited error greater than local error (X8: 67–77 % and 22–28 %);
  4. in-session reproduction of the Qwen/Llama difference, read as four case studies with a crossover interaction and the
     per-family tuning effects (X12).

  The predictor results (X4, X5, X9c) come next, then the matching caution (X10, X9d, X9b).
- **Sections:**
  - §4 holds the locked tests only, in the order of Table 3: provenance, H-D, H-E, NAT−C1 / H-F, H-A, H-B, H-C, then
    robustness.
  - §5 holds the exploratory analyses X1–X12. Table 6 lists each one's plan-entry commit, kind and section, with the
    not-blind statements.
  - Appendix B holds X1/X2, the full X6 and X7 (H-C identification).
  - Appendix D holds the lock, amendment, deviation, gate, run, verification and X9 apparatus. Methods §3.7 summarizes it
    in one paragraph.
- **Prose over bullets** throughout the main text. The remaining lists are the tables.

## Review items (sections 2 and 3 of the review), item by item

| Item | Where it is handled |
|---|---|
| "replicated" → in-session reproduction, with what it does and does not rule out | §4.3; Appendix C |
| "theory" → first-order heuristics / predictors | title, §1, §2 ("not … a theory"), §6 |
| Provenance: null rejected, not "falsified" | Table 3 ("null rejected if …"), §4.1 |
| INH sentence + X8 numbers; INH = cache component, logits/queries FP16 | §3.2, §5.3, §6 |
| QA/QO matched per arm (+0.011 to +0.032 log) | §3.2 (macro), §7 |
| Token-metric claim removed | §6 no longer has it; §5.1 relates JS to the earlier top-1 flip rates (3.43 % E1, 2.56 % E1L, same NAT arm and histories; macros from the E1/E1L records) |
| H-B as a positive control, with the analytic benchmark and dilution indicator | §4.4, §5.7 (X11), Table 8 |
| H-D both normalizations; "falsifies [8]" dropped | §4.1 (natural size and equal relative size), §5.6, §6 ("consistent with both") |
| H-C not identified; X7b/X7c as illustration only, in an appendix | §4.5, Appendix B, Table B.17 |
| Four case studies, per-family tuning effects, crossover | §4.3, §5.4 (X12), §6 |
| Table 8 sign criterion defined (now the X1/X2 table in Appendix B) | caption of Table B.15 (X1/X2) |
| X1–X11 (now X12) naming; E1 means only the earlier study | everywhere; Appendix D records the renaming |
| Distinct names for NAT-K* vs NAT-K | REL-K / REL-V (the caption of Table 1 gives the protocol names) |
| Definitions before use | Tier A removed; drift marker defined (newline + "Question:"); margin, c and b̂ ≥ 0.25 explained in §3.3; arms defined before use |
| Results in hypothesis order | Table 3 order = §4 order = Table 4 rows; descriptive names, with protocol labels kept |
| One symbol per meaning | Δ natural error, δ stored perturbation; θ contrasts; β for the H-C intercepts only; ρ_K/ρ_V/ρ* relative sizes, ρ_S Spearman |
| Unadjusted vs Holm labels | §3.5 paragraph; captions of Tables 4 and 3 |
| Absolute mean JS (X6) | §5.1, Table 7, Table B.16 |
| Layer-0 hypothesis removed from the Discussion candidates | §6 (now "removed by the exploratory analyses: X5, X9c") |
| GPU accounting | §3.7 and Appendix D: 23.75 h locked + 1.175 h X9 = 24.92 h |
| Related work | attention sinks, massive activations, KIVI/KVQuant, QuaRot/SpinQuant, QServe/Atom, and QSpec stated plainly. Every venue was verified on a proceedings page (PHASE7C_LITERATURE.md); the bib entries carry the verification notes. |

## Choices made that the authors may want to revise

1. **Hypothesis naming.** The protocol labels H-A to H-F are kept, for traceability to the locked documents, but they are
   now ordered as in §4 (provenance, D, E, NAT−C1, F, A, B, C). The alternative is to re-letter them in the paper with a
   mapping table.
2. **"falsified" / "supported"** stay as the quoted pre-declared H-A verdict words. §5.5 and §6 explain that they rest on a
   weak signal.
3. **The Methods statement** "Every locked test cited in the abstract and the conclusions also has a Holm-adjusted p < 0.05".
   The audit checks it for provenance, H-E, NAT−C1 and the family contrast. The interaction (Holm 0.043) is cited in the
   conclusion as "with tuning changes of opposite sign" and is also Holm < 0.05.
4. **X9b reading.** The excess over the benchmark that remains (1.22–1.43 log units) is stated as "consistent with the
   other inflated layers of X9d, but no arm tested them".
5. **AI-use section.** It now names the internal hostile review, the X9 authorization file written at the corresponding
   author's instruction, and the claude.ai chat as the source of the exploratory proposals of this revision.
6. **The supplement** is unchanged apart from its title. It still says "v3 experiment" in its opening paragraph.

## Unchanged and still open

All `[UNRESOLVED]` author fields (declarations, AI use, CRediT, funding, licences, data availability, previous
submissions) are kept as they were.
