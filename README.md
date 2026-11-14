# Weight-quantization error in the KV cache: code, locks and per-problem records

This repository accompanies the article *Weight-Quantization Error in the KV Cache Is Not Isotropic Noise: Value Error Is
Consistently Harsher in Ten 0.5B–8B Base Language Models* (Gunupati, Anavekar and Varghese; submitted to the Journal of
Artificial Intelligence Research). It holds the code, the protocol locks and their history, the manifests, and the
per-problem records of the locked studies:

- **v4, the scale study** (main result): eight base models, 0.5B–8B parameters, three families, on a rented NVIDIA A40.
  Lock `b0b191df9bee180e` plus the post-data deviation D1.
- **v5, the second locked study** (Section 6 of the article; added in v1.1.0): size against head dimension (Qwen3-0.6B-Base,
  SmolLM2-1.7B), a second 4-bit quantizer (HQQ) and a second task (WikiText-2), with an anchor that reruns a v4 model. It ran on a
  rented NVIDIA A40 under lock `14d2e1bf669417b8` plus the post-data deviations D1–D3.
- **v3, the four-model mechanism study** (Section 6 of the article): Qwen2.5-0.5B and Llama-3.2-1B, base and instruct, on a
  laptop RTX 4060. Lock `7325f563ce2978d8`, with the follow-ups X9 and X13.

No model weights are included. Every model is downloaded from Hugging Face at the revision pinned in the manifests.

## Getting the records

The per-problem records are too large for the git history and are attached to the GitHub Release
[v1.1.0](https://github.com/Ignite7871/kv-weight-quant-error/releases/tag/v1.1.0) as six archives (the five of v1.0.0,
unchanged, and `v5_runs.tar.gz`). Download them and unpack each inside `research_v2/`:

| Asset | Unpacks to | SHA-256 |
|---|---|---|
| `v5_runs.tar.gz` | `research_v2/v5_runs/` | `4c95eec74dae1d0bb25d16cd0f2887da7364f08ff34bad2a873893c36a63c49f` |
| `v4_runs.tar.gz` | `research_v2/v4_runs/` | `0e064c7b3d57ca6f5d3c5869d40c4e7e681f501fd619c97eb697d6897de41623` |
| `v3_runs.tar.gz` | `research_v2/v3_runs/` | `9cd1250b8991dcf07d7a9749d92db3729e2fe09682deaf5bf6f4caa808df2999` |
| `v3_runs_x9.tar.gz` | `research_v2/v3_runs_x9/` | `3cafe7ef28c84e3e9723156c794bcbfe33bd504abbdee12bbc3c32092ab745ff` |
| `v3_runs_x13.tar.gz` | `research_v2/v3_runs_x13/` | `3e561d053869cd92e49a79c9b744b66ce8bdc510be36b8a7bed403ef5c873cae` |
| `e1_e1l_runs.tar.gz` | `research_v2/e1_runs/`, `research_v2/e1l_runs/` | `df6e94d44df06c0324356de4b65f0cc00a102c95249631b7d71b72b560eb870d` |

```
cd research_v2
for f in v5_runs v4_runs v3_runs v3_runs_x9 v3_runs_x13 e1_e1l_runs; do tar -xzf ../$f.tar.gz; done
cd .. && python verify_release.py
```

`RELEASE_SHA256SUMS.txt` lists every file of the repository and of the unpacked records.

## Layout

| Path | Contents |
|---|---|
| `src/` | `dynamic_linear.py`, `dynamic_model.py`: the dual FP16/NF4 linear layers used by every study |
| `research_v2/v4/` | v4 protocol (`PROTOCOL_V4.md`), driver (`run_v4.py`, `rv4.py`, `v4_ops.py`), locked analysis (`analyze_v4.py`), independent recomputation (`recompute_v4_independent.py`) and its tolerances, exploratory plan and code, deviation log, deviation record (`deviations/V4-D1.json`), AI-use log, lock history, model manifest, sample, tests, cloud runbook, fallback driver, close-out records |
| `research_v2/V4_PROTOCOL_LOCK.txt` | the v4 lock, with every appended line (GM, Step 0, design, calibrations, deviation D1, GM D1) |
| `research_v2/v4_runs/` | v4 records: per model `calib/` and `test/` (one JSONL per arm, one `.npz` of common-mode statistics per problem), `calibration.json`, gate logs, manifests; the GPU and cost ledgers; the gate-check (GM) outputs; Step 0; the locked report `v4_report.json`, the recomputation and the exploratory output; the authorization files of every GPU run |
| `research_v2/v5/` | v5 protocol (`PROTOCOL_V5.md`), on-pod driver (`run_v5.py`, `run_v5_pod.sh`, `rv5.py`, `rv5_hqq.py`, `v5_ops.py`), laptop watcher (`watch_v5.sh`), locked analysis (`analyze_v5.py`), independent recomputation (`recompute_v5_independent.py`) and its tolerances, deviation log, deviation records (`deviations/V5-D1.json` to `V5-D3.json`), AI-use log, lock history, model manifest, WikiText-2 sample (indices and hashes only), tests, runbook and the D3 plan, rehearsal and validation records |
| `research_v2/V5_PROTOCOL_LOCK.txt` | the v5 lock, with every appended line (GM, anchor, design, calibrations, HQQ states, deviations D1–D3, GM D2, GM D3) |
| `research_v2/v5_runs/` | v5 records: per unit `calib/` and `test/`, `calibration.json`, gate logs, HQQ state records, manifests; the GPU and cost ledgers; the gate-check outputs; the anchor, design and probes; the locked report `v5_report.json` and its recomputation, and the first analysis before U1's re-execution (`_pre_d3_analysis/`); U1's D3 checks and the surviving first-run problems (`U1/d3_reference/`); the authorization files |
| `research_v2/v3/`, `research_v2/run_v3.py`, `research_v2/analyze_v3.py`, `research_v2/rv3/` | v3 protocol, driver, analysis, phases 0–8 (locked report, recomputation, exploratory analyses, X9, X13) |
| `research_v2/v3_runs/`, `v3_runs_x9/`, `v3_runs_x13/` | v3 records, and those of its two follow-ups |
| `research_v2/e1_runs/`, `e1l_runs/`, earlier `*_SPEC.md`, `*_REPORT.md`, `*_LOCK.txt` | the earlier studies that v3 builds on (background; used by the v3 generator) |
| `research_v2/tools/` | lock tools (`lock_v3.py`, `lock_v4.py`, `lock_v5.py`), mutation checks, remote-execution tools, secret scan, manifest builders, the D3 restore-set builder and the WikiText-2 release scan |
| `research_v2/paper_journal/` | the generators of every number, table and figure in the article (`make_v5.py`, `make_v4.py`, `make_v3.py`), their outputs, and the number audits (`audit/`) |

## Reproducing the analyses (CPU only)

Python 3.12 with the packages in `research_v2/v4/V4_PACKAGES.txt` (torch 2.13.0+cu126 for the GPU runs; the analyses need
numpy only). From `research_v2/`:

```
python ../verify_release.py                          # every locked file and run file hashes to the v3, v4 and v5 locks
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

`v5_runs/v5_report.json` must hash to `2f00f5113aec05f67990fac55e9150340780f31273a18080d2e276db0f493706` (checked by
`make_v5.py`). The two audit scripts also read the manuscript sources, which are not included, so in this repository they
exit at once with a missing-file error. Their outputs, run on the full sources, are included (`audit/V4_NUMBER_AUDIT.md`,
`audit/V5_NUMBER_AUDIT.md`), and each macro's declared source is in `tables_v4/numbers_v4_sources.json` and
`tables_v5/numbers_v5_sources.json`. Rerunning `make_v4.py` and `make_v5.py` on the unpacked records reproduces every table
and macro file byte for byte, apart from line endings; the figure PDFs differ only in their embedded metadata.

`tools/lock_v5.py --verify`, `tools/lock_v4.py --verify` and `tools/lock_v3.py --verify` need the original git history;
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
- The manuscript sources and internal review documents.
- Access tokens: the model-download token existed only as a cloud-provider secret and was never written to disk.

## Licences

Code: MIT (`LICENSE`). Records, reports and manifests: CC BY 4.0 (`LICENSE-DATA`). Model and dataset attributions,
including the Llama 3.1 and 3.2 notices: `NOTICE`. Citation: `CITATION.cff`.

## Integrity

`RELEASE_SHA256SUMS.txt` lists the SHA-256 of every file in this package.
