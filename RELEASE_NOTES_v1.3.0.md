# v1.3.0: the split follow-up (v6b)

Code, locks and per-problem records for the article *Weight-Quantization Error in the KV Cache Is Not Isotropic Noise: Value
Error Is Consistently Harsher, and Its Common-Mode and Direction Parts Are Not Separable* (Gunupati, Anavekar and Varghese;
prepared for the *Journal of Machine Learning Research*). The article's title changed with this release, because v6b
rejected the headline of the v1.2.0 title.

## What's new

v1.3.0 adds the locked follow-up **v6b** (Section 6.3 of the article). The v3–v6 material of v1.2.0 is unchanged, except
that `make_v6.py` gained two macros and a caption sentence, and the v6 number audit (`audit_v6_numbers.py`) now also covers
v6b.

- The value-path split of v6 (the excess of natural NF4 value error over norm-matched isotropic noise, divided into a
  common-mode part and a direction part) rerun in both orders and with and without re-norming, on the first 70 v6 test
  problems of Qwen2.5-0.5B, Llama-3.2-1B and Qwen2.5-1.5B, with three new value arms (NAT-V-cen0, C1-V+mu, C1-V@cen).
- Before any new arm of a problem ran, NAT-V, C1-V and NAT-V-cen reproduced that problem's v6 records bitwise (all 70 problems
  of every model).
- Run on the laptop RTX 4060: 2.7 GPU-hours in the ledger including the rehearsal (cap 9), no cut, no deviation, no record with
  a failed integrity flag.
- Verdict under the rule locked before the data: the headline "the direction part carries most of the value-path excess" is
  **rejected** (along the path that adds the common mode to isotropic noise at the centered norm, the direction share's upper
  95% bound is below one half in Qwen2.5-0.5B and Llama-3.2-1B). The direction share ranges from 0.36 to 0.87 across the four
  paths, and the difference between the two orders excludes 0 in every model.
- Lock `cba09c444c6de782` (`research_v2/V6B_PROTOCOL_LOCK.txt`); empty deviation log `research_v2/v6b/DEVIATION_LOG_V6B.md`.
- The locked report `v6b_report.json` (SHA-256 `a7668747…`) and the independent recomputation `v6b_recomputation.json`
  (`2903910c…`; 85 quantities, 0 outside the declared tolerances, same verdict), both in the git tree under
  `research_v2/v6b_runs/`; their hashes are fixed in `research_v2/v6b/results/V6B_RESULTS_HASHES.txt`.
- `make_v6b.py` (every v6b number and the v6b table of the article); `audit/audit_v6_numbers.py` and its output.
- `verify_release.py` now also checks the v6b lock and the two v6b reports.

## Assets

The seven archives of v1.2.0, unchanged, plus `v6b_runs.tar.gz` (unpacks to `research_v2/v6b_runs/`; SHA-256
`1dcb2653b592a1213974b3e04101bece83b9f127267ad94712c83f4f715c5006`). See `README.md` for the full table and the unpacking
commands; `RELEASE_SHA256SUMS.txt` lists every file of the repository and of the unpacked records.
