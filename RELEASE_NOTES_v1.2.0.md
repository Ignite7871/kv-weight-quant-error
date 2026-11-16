# v1.2.0: the logging study (v6)

Code, locks and per-problem records for the article *Weight-Quantization Error in the KV Cache Is Aligned with the Value
Signal: Direction, Not Common Mode, Carries Most of the Value-Path Excess* (Gunupati, Anavekar and Varghese; prepared for
the *Journal of Machine Learning Research*).

## What's new

v1.2.0 adds the third locked study, **v6** (Sections 3 and 6 of the article). The v3, v4 and v5 material of v1.1.0 is
unchanged (two v4 audit outputs are updated from the research repository).

- A first-order model of weight-induced KV-cache error on the head output, and its tests on fresh GSM8K problems in
  Qwen2.5-0.5B (100 problems, 13 arms), Llama-3.2-1B and Qwen2.5-1.5B (70 problems, 10 arms each, deviation V6-D1), run on a
  laptop RTX 4060 (13.6 GPU-hours in the ledger, cap 16).
- Logged attention internals, centered arms (common mode removed), per-token INT4 storage arms, and two alignment
  statistics of the value error.
- Verdicts under the criteria fixed before the rehearsal: L1, L5, L8a, L9a, L9b agreement; L2, L6, L7 neither; L8b
  disagreement (the prediction was falsified).
- The R block: pre-specified secondary analyses of the v4/v5 records.
- Lock `00369b8d99cefd69` (`research_v2/V6_PROTOCOL_LOCK.txt`); deviations V6-D1 (cut level re-set at the true speed) and
  V6-D2 (independent recomputation added before the analysis) in `research_v2/v6/DEVIATION_LOG_V6.md`.
- The locked report `v6_report.json` (SHA-256 `98626e3e…`) and the independent recomputation `v6_recomputation.json`
  (`ec169a0d…`; 264 quantities, 0 outside the declared tolerances, 0 verdict disagreements), both in the git tree under
  `research_v2/v6_runs/`; their hashes are fixed in `research_v2/v6/results/V6_RESULTS_HASHES.txt`.
- `make_v6.py` (every v6 number, table and figure of the article) and `audit/audit_v6_numbers.py` with its output.
- `verify_release.py` now also checks the v6 lock and the two v6 reports.

## Assets

The six archives of v1.1.0, unchanged, plus `v6_runs.tar.gz` (unpacks to `research_v2/v6_runs/`; SHA-256
`28ced3f6b6b59b968620df70ee2eb127c2fb75aaac6d28ffa63b59c990b2508d`). See `README.md` for the full table and the unpacking
commands; `RELEASE_SHA256SUMS.txt` lists every file of the repository and of the unpacked records.
