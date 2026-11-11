# v3 Phase 0 — fixes to a copy of the current manuscript (no GPU)

Branch `v3-novelty`, 2026-10-29. Status: **approved by the authors 2026-10-29** (decisions below).

**Base file.** The flagged errors (Table B.5/B.6/B.9/D.19 numbering, `Nresauthors`, `[UNRESOLVED]` fields) are in the
author-review version `paper_journal/main_review.tex` + `sections_review/`, so v3 starts from that:

| v3 file | copied from (unchanged) |
|---|---|
| `paper_journal/main_v3.tex` | `main_review.tex` (only edits: header comment, `\input{sections_v3/...}`, `\bibliography{refs_v3}`) |
| `paper_journal/sections_v3/` | `sections_review/` |
| `paper_journal/refs_v3.bib` | `refs.bib` |
| `paper_journal/main_v3.pdf` | built with latexmk; 47 pp; 0 undefined references |

`tables/` and `tables/numbers.tex` are shared and were **not** edited. `git status` shows no modified tracked file;
all 62 entries of `FROZEN_RECORDS_MANIFEST.sha256` verify OK after the edits.

**Sources of every number below.** `v3/phase0/recompute_phase0.py` reads the raw per-problem records directly and
imports none of the existing analysis/audit code. Its output is `v3/phase0/phase0_results.json`. The edits are made by
`v3/phase0/apply_phase0_edits.py`, where each replacement must match exactly once.

## Change table

| # | Location (PDF label → source file) | Old | New | Evidence |
|---|---|---|---|---|
| 1a | Table B.5 caption → `sections_v3/B_statistics.tex` | "Stage 1 primary family [exploratory]." | Same, plus: the H5 90% CI is from the locked analysis; H5 lies on a 1/n lattice (0.11 pt); D.19 uses an independent stream (+0.43); exact bootstrap limits [−2.04, +0.54]. **Table values unchanged.** | Exact B→∞ bootstrap gives [−2.043, +0.538]. Over 1,000 streams × 10,000 resamples the upper limit is +0.54 in 878 streams and +0.43 in 109 (+0.44 in 13). The locked JSON has +0.538 and the audit JSON +0.430. |
| 1b | Table B.6 caption → same file | "…excluded (n = 342)." | Same, plus the lattice note (0.29 pt); D.19 gives +2.63; exact limits [−2.05, +2.63]. **Table values unchanged.** | Exact upper limit +2.632 (CDF at +2.34 = 0.9493 < 0.95). Streams: +2.63 in 596, +2.34 in 390 (+2.35 in 14). |
| 1c | Table D.19 caption → `sections_v3/E_stage12_details.tex` | caption ended "…Appendix Table B.12." | Adds: independent stream; one-step differences (+0.43 vs +0.54; +2.63 vs +2.34); exact limits; δ* and all conclusions unchanged. **Table values unchanged.** | As 1a/1b. δ*(exact) = 2.04 (Stage 1) and 2.63 (342). Both are < 3, so the ±3-point criterion is unaffected. |
| 2 | App. D.1 text → `sections_v3/E_stage12_details.tex` | "(no #### in 6.1% … 8.1% token-only; …)" | "(unparseable output, i.e. no #### or #### not followed by a number, in 6.1% … 8.4% token-only; …)" | Token-only, Stage 1: 75 with no #### (8.06%) + 3 with #### but no number = **78 (8.39%)**. Text and Table B.9 used different definitions. FP16 57 (6.13%), NF4 115 (12.37%) and cache-only 105 (11.29%) are identical under both. |
| 3a | Sec 3.2 Prefill → `sections_v3/03_intervention.tex` | "(on Qwen, about 108 generated against about 431 prompt positions on average)" | "(on Qwen in Stage 1, n = 930: on average about 108 decode-written positions per FP16 generation and 112–119 in the other conditions, against about 431 prompt positions; the first generated token comes from the prefill)" | Stage 1 mean `steps`: FP16 107.56, NF4 118.78, cache-only 114.83, token-only 111.79, oracle 115.35. Prompt 430.84. `len(ids)` = steps + 1. |
| 3b | Sec 4.3 + App. B.7 → `sections_v3/04_design.tex`, `sections_v3/B_statistics.tex` | "(mean length 112 tokens, at most 256)" | "(mean length 112 generated tokens over the 200 Tier A problems, at most 256)" | 112.115 = mean `ref_len` over the 200 E1 Tier A problems = mean `len(ids)` of the stored FP16 records (equal for all 200). **The two figures are not in conflict:** different population (930 vs 200 problems) and different count (decode steps vs generated tokens). |
| 4a | Ref. [2] `yuan2025nondeterminism` → `refs_v3.bib` | `@article`, journal "arXiv preprint arXiv:2506.09501", note "NeurIPS 2025" (rendered fused "2506.09501NeurIPS 2025"); title with "(earlier title: …)" | `@inproceedings`, booktitle "39th Conference on Neural Information Processing Systems (NeurIPS 2025)", note "arXiv:2506.09501"; title = current arXiv title only | Verified from the v2 PDF first page (arxiv.org/pdf/2506.09501v2), footer verbatim: "39th Conference on Neural Information Processing Systems (NeurIPS 2025)". The abs page has no venue. The earlier title was not checked and was dropped as note text. |
| 4b | Ref. [3] `du2026greedy` → `refs_v3.bib` | note "Published in TMLR, September 2026 (per the paper's own header and the arXiv journal reference)" | note removed; journal "Transactions on Machine Learning Research (arXiv:2609.26621)", year 2026 | arxiv.org/abs/2609.26621: Journal-ref "Transactions on Machine Learning Research, 2026"; Comments "Accepted by Transactions on Machine Learning Research (TMLR), 2026". **"September" is not stated there and was removed.** |
| 4c | Ref. [22] `lotfi2026quantizedthink` | arXiv preprint 2606.00206, 2026 | **Updated after author decision:** `@inproceedings`, booktitle "Conference on Neural Information Processing Systems (NeurIPS 2026)", note "Oral; arXiv:2606.00206" | Confirmed on https://neurips.cc/virtual/2026/oral/151669: title, the four authors, "Oral", "NeurIPS 2026". Earlier evidence: | The abs page (v1, 29 May 2026) has no Comments or Journal-ref. The PDF says "Date: June 2, 2026" and gives no venue. NeurIPS 2026 status appears only on an aggregator post (digg.com) → **[UNVERIFIED]**; the official NeurIPS listing did not render for me. Title, authors and ID are verified. |
| 5 | App. C Artifacts → `sections_v3/C_protocols.tex` | "…pending the authors' decision Nres{authors}." | "…pending the authors' decision \unres{authors}." (renders as the red [UNRESOLVED: authors] field) | — |
| 6 | Sec 5 Tier B → `sections_v3/05_results.tex` | "(power of about 5--6 points)" | "(minimum detectable paired difference of about 5.5--6.8 points at 80% power, n = 400)" | See deviation V3-D01: 5–6 was the protocol's projection for n = 500 (discordance 0.16–0.20). From the realized Tier B records (n = 400), MDE = 2.80·√(disc/n): NAT−C1 6.75 (disc 0.2325), NAT−C2 6.79, C2−C1 6.46, NAT−FP16 5.47, C1−FP16 6.19, C2−FP16 6.23. |
| — | `main_v3.tex` | — | header comment pointing here; `\input` paths → `sections_v3/`; `\bibliography{refs_v3}` | — |

## Points for author decision

1. **H5 intervals (items 1a–1c).** I explained the difference in the captions and did not change the tables. B.5/B.6
   reproduce the locked analysis output, and D.19 reproduces a separate exploratory stream. The alternative is to report
   the exact bootstrap limits in all three tables, which replaces locked-analysis values. In that case B.6's upper limit
   would become +2.63. Also note that **the held-out-342 lower limit is itself stream-unstable**: −2.05 in 515 of 1,000
   streams and −1.75 in 471. Both tables show −2.05 only by chance agreement. The exact value is −2.05.
2. **Item 6** uses the realized n = 400 range (5.5–6.8), not the literal wording "about 5–6" that was suggested.
   Choose one: (a) the realized range (as applied), or (b) "the protocol's projected minimum detectable effect of about
   5–6 points (n = 500)", which is accurate as a statement about the protocol but not about the run.
3. **Lotfi et al.** If the NeurIPS 2026 oral is real, an author should confirm it on neurips.cc or OpenReview. Until then
   it stays an arXiv citation.
4. **Not fixed, because outside the request:** the same "power of about 5–6 points" and "about 108" wording appears in
   the non-review `sections/` (main.tex). It is untouched, as required.
   The uncited `kurtic2024bf16` entry has the same `@article` + `note` pattern; it does not render.

## Author decisions (2026-10-29) and follow-up

1. H5: keep the locked table values with the caption explanations. Follow-up check `h5_upper_bound_check.py` →
   `h5_upper_bound_check.json`. It ran 1,000 streams × 10,000 resamples per sample and recorded the 90% upper limit:

   | Sample | n | upper-limit distribution over streams | fraction ≥ 3.0 | TOST p (±3, normal) |
   |---|---|---|---|---|
   | Stage 1 | 930 | +0.54 ×877, +0.44 ×14, +0.43 ×109 | 0.000 | 0.0017 |
   | Held-out | 389 | +2.83 ×1000 | 0.000 | 0.035 |
   | Held-out-342 | 342 | +2.63 ×605, +2.35 ×16, +2.34 ×379 | 0.000 | 0.022 |

   No stream reaches 3.0, so the conditional rewording of Sec 5.1 and App. D.2 was **not triggered** and no deviation
   is logged. Note: for held-out-389 the next lattice point above +2.83 is +3.08 (step 0.26), so it sits one step below
   the margin. All 1,000 streams agree on +2.83, and the main text already calls this equivalence marginal (TOST p 0.035).
2.–5. Accepted. 4c: see the updated row. 6: 5.5–6.8 kept (V3-D01 accepted).

## Addition during Phase 2 (author request, 2026-10-29)

| # | Location | Old | New | Evidence |
|---|---|---|---|---|
| 7 | App. D.2 audit paragraph → `sections_v3/E_stage12_details.tex` | "(upper 90% bound 2.83 of 3.0; TOST p = 0.035 …)." | adds: "Because H5 moves in steps of one problem (100/389 = 0.26 points), this bound (11/389) lies one problem below the margin: the next attainable value is 12/389 = +3.08 points." | `h5_upper_bound_check.json`: held-out-389 upper limit 2.8278 = 11/389 in all 1,000 streams; 12/389 = 3.0848. Rebuilt main_v3.pdf, 0 undefined references. |
