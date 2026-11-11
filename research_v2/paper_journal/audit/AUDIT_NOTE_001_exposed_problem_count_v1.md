# Audit note 001 (v1): number of exposed test problems in the Qwen E1 sensitivity analyses (4 vs 13)

Date: 2026-10-28. Status: correction note for a historical report; **the original report and every locked file are unchanged** (see "Integrity").
Applies to: `research_v2/E1_REPORT.md` section 5, row "excluding the 13 exposed test problems | −0.252 [−0.328, −0.176]", listed under "Tier A".

## Finding
The label "13" is wrong for the Tier A row. Correct values, from the locked data files:

| Quantity | Value | Source |
|---|---|---|
| Exposed test problems in the E1 **Tier B** sample (400) | **13** (indices 4, 38, 45, 69, 71, 94, 100, 132, 151, 156, 160, 165, 172) | `e1_sample.json` → `exposed_test_ids_in_sample` |
| Exposed test problems in the E1 **Tier A** sample (200, a subset of Tier B) | **4** (45, 71, 100, 172) | intersection of the previous list with `tierA_ids` |
| Llama E1L Tier A exposed problems (same 200 ids) | **4** (45, 71, 100, 172); `E1L_REPORT.md` says 4 | `e1l_sample.json` |
| Sensitivity analysis n, Tier A / Tier B | **196 / 387** | `e1_runs/e1_report.json` → `sensitivity.tierA|tierB["excluding exposed test problems"].n` |
| Tier A estimate excluding exposed | −0.2515 [−0.3277, −0.1764] | same |
| Tier B estimate excluding exposed | +3.587 [+1.142, +6.049] | same |

Second, smaller discrepancy in the same row: the report text prints the point estimate as −0.252, the stored value −0.25147 rounds to **−0.251**. The interval printed (−0.328, −0.176) is correct.

## Cause
A transcription slip when the sensitivity table was written: the count of exposed problems in the Tier B sample (13) was attached to the Tier A row. The numerical estimate in the row is the correct Tier A value; only the label (and the third decimal of the estimate) is affected. No conclusion changes.

## Where the correct values are used
* Manuscript (`paper_journal`): "excluding exposed problems (n = 196)" for Tier A (Table 8; macro `\QSensExp` = −0.251), "13 of the 400 Tier B problems" and "four of the 200 Tier A problems" in §4.2 and Appendix B; independent audit `audit_journal.py` does not depend on the label.
* `reports/D2…`, `D5…` carry the 4-vs-13 note (written 2026-10-28).

## Integrity
* `E1_REPORT.md` was **not edited** (sha256 `6b033884…8748`); this note is the correction record. `e1_runs/e1_report.json` (the machine-readable locked report) is correct and was used for all manuscript numbers via `make_numbers.py`.
* Frozen records manifest (62 entries) re-verified unchanged by `audit_journal.py` on 2026-10-28 (see its results file).

## Versions
v1 (this note) — first record of the correction. Any later change must be a new version (`…_v2.md`) that supersedes this one without deleting it.
