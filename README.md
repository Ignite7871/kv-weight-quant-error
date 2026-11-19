# Weight-quantization error in the KV cache: code, locks and per-problem records

This repository accompanies the manuscript *Weight-Quantization Error in the KV Cache Is Not Isotropic Noise* by Srikar Reddy Gunupati, Shrinivas Anavekar and
Alen K Varghese (manuscript under review). It holds the code, the protocol locks and their history, the manifests, and the per-problem
records of the locked studies:

- **v6b, the split follow-up** (Section 6.3 of the article; added in v1.3.0): the value-path split of v6 rerun on the first
  70 v6 test problems of each v6 model, in both orders and with and without re-norming (three new value arms), after a
  bitwise reproduction of the v6 records. It ran on the laptop RTX 4060 under lock `cba09c444c6de782`, with no deviation;
  the pre-stated headline that the direction part carries most of the value-path excess was rejected.
- **v6, the logging study** (Sections 3 and 6 of the article; added in v1.2.0): a first-order model of the head output and
  its tests on fresh GSM8K problems in Qwen2.5-0.5B, Llama-3.2-1B and Qwen2.5-1.5B, with logged attention internals,
  centered arms, storage-error arms and two direction (alignment) statistics. It ran on a laptop RTX 4060 under lock
  `00369b8d99cefd69` plus the deviations V6-D1 (cut level at the true speed: n = 70 and 10 arms for the second and third
  models) and V6-D2 (independent recomputation added before the analysis). Also the R block, pre-specified secondary analyses
  of the v4/v5 records (`v6/results/`).
- **v4, the scale study** (Section 5 of the article): eight base models, 0.5B–8B parameters, three families, on a rented NVIDIA A40.
  Lock `b0b191df9bee180e` plus the post-data deviation D1.
- **v5, the generality study** (Section 5 and Appendix B of the article; added in v1.1.0): size against head dimension
  (Qwen3-0.6B-Base, SmolLM2-1.7B), a second 4-bit quantizer (HQQ) and a second task (WikiText-2), with an anchor that reruns a
  v4 model. It ran on a rented NVIDIA A40 under lock `14d2e1bf669417b8` plus the post-data deviations D1–D3.
- **v3, the four-model pilot** (Appendix C.2 of the article): Qwen2.5-0.5B and Llama-3.2-1B, base and instruct, on a laptop
  RTX 4060. Lock `7325f563ce2978d8`, with the follow-ups X9 and X13.

No model weights are included. Every model is downloaded from Hugging Face at the revision pinned in the manifests.

## Getting the records

The per-problem records are too large for the git history and are attached to the GitHub Release
[v1.3.0](https://github.com/Ignite7871/kv-weight-quant-error/releases/tag/v1.3.0) as eight archives (the seven of v1.2.0,
unchanged, and `v6b_runs.tar.gz`). The two v6 reports and the two v6b reports are also in the git tree
(`research_v2/v6_runs/`, `research_v2/v6b_runs/`). Download the archives
and unpack each inside `research_v2/`:

| Asset | Unpacks to | SHA-256 |
|---|---|---|
| `v6b_runs.tar.gz` | `research_v2/v6b_runs/` | `1dcb2653b592a1213974b3e04101bece83b9f127267ad94712c83f4f715c5006` |
| `v6_runs.tar.gz` | `research_v2/v6_runs/` | `28ced3f6b6b59b968620df70ee2eb127c2fb75aaac6d28ffa63b59c990b2508d` |
| `v5_runs.tar.gz` | `research_v2/v5_runs/` | `4c95eec74dae1d0bb25d16cd0f2887da7364f08ff34bad2a873893c36a63c49f` |
| `v4_runs.tar.gz` | `research_v2/v4_runs/` | `0e064c7b3d57ca6f5d3c5869d40c4e7e681f501fd619c97eb697d6897de41623` |
| `v3_runs.tar.gz` | `research_v2/v3_runs/` | `9cd1250b8991dcf07d7a9749d92db3729e2fe09682deaf5bf6f4caa808df2999` |
| `v3_runs_x9.tar.gz` | `research_v2/v3_runs_x9/` | `3cafe7ef28c84e3e9723156c794bcbfe33bd504abbdee12bbc3c32092ab745ff` |
| `v3_runs_x13.tar.gz` | `research_v2/v3_runs_x13/` | `3e561d053869cd92e49a79c9b744b66ce8bdc510be36b8a7bed403ef5c873cae` |
| `e1_e1l_runs.tar.gz` | `research_v2/e1_runs/`, `research_v2/e1l_runs/` | `df6e94d44df06c0324356de4b65f0cc00a102c95249631b7d71b72b560eb870d` |

```
cd research_v2
for f in v6b_runs v6_runs v5_runs v4_runs v3_runs v3_runs_x9 v3_runs_x13 e1_e1l_runs; do tar -xzf ../$f.tar.gz; done
cd .. && python verify_release.py
```

`RELEASE_SHA256SUMS.txt` lists every file of the repository and of the unpacked records.

## Layout

| Path | Contents |
|---|---|
| `src/` | `dynamic_linear.py`, `dynamic_model.py`: the dual FP16/NF4 linear layers used by every study |
| `research_v2/v6b/` | v6b protocol (`FOLLOWUP_V6B.md`), arms and driver (`rv6b.py`, `run_v6b.py`), test launcher, locked analysis (`analyze_v6b.py`), independent recomputation (`recompute_v6b_independent.py`) and its tolerances, deviation log (empty), AI-use log, tests and mutation check with its output, and the SHA-256 of the two accepted reports (`results/V6B_RESULTS_HASHES.txt`) |
| `research_v2/V6B_PROTOCOL_LOCK.txt` | the v6b lock (it also hashes the V6 lock and the v6/v4/v3 code it reuses) |
| `research_v2/v6b_runs/` | v6b records: per model the six arm JSONL files and the reproduction log `g1.jsonl`, for the rehearsal and the test phase; the GPU ledger, design, launcher status, heartbeat and logs; the authorization files; a design file left by a CPU test before the launch and renamed aside (`STRAY_design_from_cpu_test_*.json`, see the AI-use log); the locked report `v6b_report.json` and its recomputation `v6b_recomputation.json` |
| `research_v2/v6/` | v6 protocol (`PROTOCOL_V6.md`) and theory plan (`THEORY_PLAN_V6.md`), novelty check, logging and driver (`rv6.py`, `run_v6.py`), test launcher and power-throttling watcher, locked analysis (`analyze_v6.py`), R-block plan and analysis (`R_ANALYSIS_PLAN_V6.md`, `analyze_r_v6.py`), independent recomputation (`recompute_v6_independent.py`) and its tolerances, deviation log, AI-use log, sample, tests, RoPE-config provenance, rehearsal gate records, R-block report and the SHA-256 of the two accepted reports (`results/V6_RESULTS_HASHES.txt`) |
| `research_v2/V6_PROTOCOL_LOCK.txt` | the v6 lock |
| `research_v2/v6_runs/` | v6 records: per model the arm JSONL files and per-problem `.npz` arrays of the logged quantities, for the rehearsal and the test phase; the GPU ledger, design (before and after V6-D1), launcher status and logs; the authorization files; the locked report `v6_report.json` and its recomputation `v6_recomputation.json` |
| `research_v2/v4/` | v4 protocol (`PROTOCOL_V4.md`), driver (`run_v4.py`, `rv4.py`, `v4_ops.py`), locked analysis (`analyze_v4.py`), independent recomputation (`recompute_v4_independent.py`) and its tolerances, exploratory plan and code, deviation log, deviation record (`deviations/V4-D1.json`), AI-use log, lock history, model manifest, sample, tests, cloud runbook, fallback driver, close-out records |
| `research_v2/V4_PROTOCOL_LOCK.txt` | the v4 lock, with every appended line (GM, Step 0, design, calibrations, deviation D1, GM D1) |
| `research_v2/v4_runs/` | v4 records: per model `calib/` and `test/` (one JSONL per arm, one `.npz` of common-mode statistics per problem), `calibration.json`, gate logs, manifests; the GPU and cost ledgers; the gate-check (GM) outputs; Step 0; the locked report `v4_report.json`, the recomputation and the exploratory output; the authorization files of every GPU run |
| `research_v2/v5/` | v5 protocol (`PROTOCOL_V5.md`), on-pod driver (`run_v5.py`, `run_v5_pod.sh`, `rv5.py`, `rv5_hqq.py`, `v5_ops.py`), laptop watcher (`watch_v5.sh`), locked analysis (`analyze_v5.py`), independent recomputation (`recompute_v5_independent.py`) and its tolerances, deviation log, deviation records (`deviations/V5-D1.json` to `V5-D3.json`), AI-use log, lock history, model manifest, WikiText-2 sample (indices and hashes only), tests, runbook and the D3 plan, rehearsal and validation records |
| `research_v2/V5_PROTOCOL_LOCK.txt` | the v5 lock, with every appended line (GM, anchor, design, calibrations, HQQ states, deviations D1–D3, GM D2, GM D3) |
| `research_v2/v5_runs/` | v5 records: per unit `calib/` and `test/`, `calibration.json`, gate logs, HQQ state records, manifests; the GPU and cost ledgers; the gate-check outputs; the anchor, design and probes; the locked report `v5_report.json` and its recomputation, and the first analysis before U1's re-execution (`_pre_d3_analysis/`); U1's D3 checks and the surviving first-run problems (`U1/d3_reference/`); the authorization files |
| `research_v2/v3/`, `research_v2/run_v3.py`, `research_v2/analyze_v3.py`, `research_v2/rv3/` | v3 protocol, driver, analysis, phases 0–8 (locked report, recomputation, exploratory analyses, X9, X13) |
| `research_v2/v3_runs/`, `v3_runs_x9/`, `v3_runs_x13/` | v3 records, and those of its two follow-ups |
| `research_v2/e1_runs/`, `e1l_runs/`, earlier `*_SPEC.md`, `*_REPORT.md`, `*_LOCK.txt` | the earlier studies that v3 builds on (background; used by the v3 generator) |
| `research_v2/tools/` | lock tools (`lock_v3.py` to `lock_v6b.py`), mutation checks, remote-execution tools, secret scan, manifest builders, the D3 restore-set builder and the WikiText-2 release scan |
| `research_v2/paper_journal/` | the generators of every number, table and figure in the article (`make_v6b.py`, `make_v6.py`, `make_v5.py`, `make_v4.py`, `make_v3.py`), their outputs, and the number audits (`audit/`) |

## Reproducing the analyses (CPU only)

Python 3.12 with the packages in `research_v2/v4/V4_PACKAGES.txt` (torch 2.13.0+cu126 for the GPU runs; the analyses need
numpy only). From `research_v2/`:

```
python ../verify_release.py                          # every locked file and run file hashes to the v3-v6b locks; the six reports
python v6b/analyze_v6b.py                            # rewrites v6b_runs/v6b_report.json
python v6b/recompute_v6b_independent.py              # v6b independent recomputation (85 quantities)
python paper_journal/make_v6b.py                     # every v6b number macro and the v6b table of the article
python v6/analyze_v6.py                              # rewrites v6_runs/v6_report.json
python v6/recompute_v6_independent.py                # v6 independent recomputation (264 quantities)
python v6/analyze_r_v6.py                            # R block on the v4/v5 records
python paper_journal/make_v6.py                      # every v6 number macro, table and the figure of the article
python paper_journal/audit/audit_v6_numbers.py       # re-derives every v6 and v6b macro, checks them against the recomputations
python v4/analyze_v4.py analyze                      # rewrites v4_runs/v4_report.json (about 80 min on CPU)
python v4/recompute_v4_independent.py                # independent recomputation, compared under the declared tolerances
python v4/exploratory_v4.py                          # exploratory X1-X4
python paper_journal/make_v4.py                      # every v4 number macro, table and figure of the article
python paper_journal/audit/audit_v4_numbers.py       # re-derives every macro from its source
python v5/analyze_v5.py analyze                      # rewrites v5_runs/v5_report.json
python v5/recompute_v5_independent.py                # v5 independent recomputation (230 quantities)
python paper_journal/make_v5.py                      # every v5 number macro and table of the article
python paper_journal/audit/audit_v5_numbers.py       # re-derives every v5 macro, checks it against the recomputation
```

`recompute_v6_independent.py` reads each v6 model's RoPE parameters from its pinned `config.json` in the local Hugging Face
cache (the config file alone suffices; its git-blob hash is checked against `v4/V4_MODEL_MANIFEST.json`).
`v6_runs/v6_report.json` must hash to `98626e3ec4a26da93fc6e45ac0a104264ad9fd914f8b14bb26b36c2ab246acc1` and
`v6_runs/v6_recomputation.json` to `ec169a0d628cb2ea875406806afdc7d320333395ba08978f92a248ca804b5be6` (checked by `make_v6.py`);
`v6b_runs/v6b_report.json` to `a766874757f178caa46af4d28873c39b27b039f25e20330b16d321a14226d98f` and
`v6b_runs/v6b_recomputation.json` to `2903910cb3cbb9d6048ca9583697725bbca811236c9118d85626268f5366e55f` (checked by `make_v6b.py`).
`v5_runs/v5_report.json` must hash to `2f00f5113aec05f67990fac55e9150340780f31273a18080d2e276db0f493706` (checked by
`make_v5.py`). The audit scripts also read the manuscript sources, which are not included, so in this repository they
exit at once with a missing-file error. Their outputs, run on the full sources, are included (`audit/V4_NUMBER_AUDIT.md`,
`audit/V5_NUMBER_AUDIT.md`, `audit/V6_NUMBER_AUDIT.md`, the last covering v6 and v6b), and each macro's declared source is in
the `numbers_*_sources.json` file next to it. Rerunning `make_v4.py`, `make_v5.py`, `make_v6.py` and `make_v6b.py` on the
unpacked records reproduces every table and macro file byte for byte, apart from line endings; the figure PDFs differ only
in their embedded metadata.

`tools/lock_v6b.py --verify`, `tools/lock_v6.py --verify`, `tools/lock_v5.py --verify`, `tools/lock_v4.py --verify` and
`tools/lock_v3.py --verify` need the original git history;
`verify_release.py` checks the same hashes without it. `v4_runs/v4_report.json` must hash to `df57a31483160e62351378576d14123893081249e06525ed5be5f5099758dccb` (checked by
`make_v4.py`). The v3 analyses run the same way from `analyze_v3.py`, `v3/phase5/` and `paper_journal/make_v3.py`.

Re-running the GPU studies needs the models, a CUDA GPU and an authorization file per run, as described in
`v4/cloud/RUNBOOK_V4.md` and `v3/PROTOCOL_V3.md`.

## What is not included, and why

- Model weights and Hugging Face caches (licences; downloaded at pinned revisions).
- `v4_runs/checkpoints/`: the per-model archives used to move records off the cloud volume. Their contents are included
  unpacked. The hash of every file on the cloud volume at close-out is in `v4/closeout/volume_sha256sums.txt`; the
  archives and the pods' host-key and connection files (`known_hosts*`, `pod_ssh.json`) are the only entries listed there
  that are not shipped; every other listed file is included with a matching hash.
- `v5_runs/checkpoints/` (the per-unit archives; their contents are included unpacked), the D3 restore set
  (`v5_runs/_d3_restore_*`, a copy of laptop records staged for U1's re-execution), and the pods' host-key and connection
  files.
- `gm_D2.json`, the GM record of v5 deviation D2: it was never checkpointed and was lost with the first cloud volume. V5-D3
  declares it lost and keeps its hash; `verify_release.py` reports it as recorded lost.
- WikiText-2 text and token ids (see `NOTICE`).
- The manuscript sources, the submission plan and internal review documents.
- Access tokens: the model-download token existed only as a cloud-provider secret and was never written to disk.

## Licences

Code: MIT (`LICENSE`). Records, reports and manifests: CC BY 4.0 (`LICENSE-DATA`). Model and dataset attributions,
including the Llama 3.1 and 3.2 notices: `NOTICE`. Citation: `CITATION.cff`.

## Integrity

`RELEASE_SHA256SUMS.txt` lists the SHA-256 of every file in this package.
