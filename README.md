# Weight-quantization error in the KV cache: code, locks and per-problem records

This repository accompanies the article *Weight-Quantization Error in the KV Cache Is Not Isotropic Noise: Value Error Is
Consistently Harsher across 0.5B–8B Language Models* (Gunupati, Anavekar and Varghese; submitted to the Journal of
Artificial Intelligence Research). It holds the code, the protocol locks and their history, the manifests, and the
per-problem records of both locked studies:

- **v4, the scale study** (main result): eight base models, 0.5B–8B parameters, three families, on a rented NVIDIA A40.
  Lock `b0b191df9bee180e` plus the post-data deviation D1.
- **v3, the four-model mechanism study** (Section 6 of the article): Qwen2.5-0.5B and Llama-3.2-1B, base and instruct, on a
  laptop RTX 4060. Lock `7325f563ce2978d8`, with the follow-ups X9 and X13.

No model weights are included. Every model is downloaded from Hugging Face at the revision pinned in the manifests.

## Getting the records

The per-problem records are too large for the git history and are attached to the GitHub Release
[v1.0.0](https://github.com/Ignite7871/kv-weight-quant-error/releases/tag/v1.0.0) as five archives. Download them and
unpack each inside `research_v2/`:

| Asset | Unpacks to | SHA-256 |
|---|---|---|
| `v4_runs.tar.gz` | `research_v2/v4_runs/` | `0e064c7b3d57ca6f5d3c5869d40c4e7e681f501fd619c97eb697d6897de41623` |
| `v3_runs.tar.gz` | `research_v2/v3_runs/` | `9cd1250b8991dcf07d7a9749d92db3729e2fe09682deaf5bf6f4caa808df2999` |
| `v3_runs_x9.tar.gz` | `research_v2/v3_runs_x9/` | `3cafe7ef28c84e3e9723156c794bcbfe33bd504abbdee12bbc3c32092ab745ff` |
| `v3_runs_x13.tar.gz` | `research_v2/v3_runs_x13/` | `3e561d053869cd92e49a79c9b744b66ce8bdc510be36b8a7bed403ef5c873cae` |
| `e1_e1l_runs.tar.gz` | `research_v2/e1_runs/`, `research_v2/e1l_runs/` | `df6e94d44df06c0324356de4b65f0cc00a102c95249631b7d71b72b560eb870d` |

```
cd research_v2
for f in v4_runs v3_runs v3_runs_x9 v3_runs_x13 e1_e1l_runs; do tar -xzf ../$f.tar.gz; done
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
| `research_v2/v3/`, `research_v2/run_v3.py`, `research_v2/analyze_v3.py`, `research_v2/rv3/` | v3 protocol, driver, analysis, phases 0–8 (locked report, recomputation, exploratory analyses, X9, X13) |
| `research_v2/v3_runs/`, `v3_runs_x9/`, `v3_runs_x13/` | v3 records, and those of its two follow-ups |
| `research_v2/e1_runs/`, `e1l_runs/`, earlier `*_SPEC.md`, `*_REPORT.md`, `*_LOCK.txt` | the earlier studies that v3 builds on (background; used by the v3 generator) |
| `research_v2/tools/` | lock tools (`lock_v3.py`, `lock_v4.py`), mutation checks, remote-execution tool, secret scan, manifest builders |
| `research_v2/paper_journal/` | the generators of every number, table and figure in the article (`make_v4.py`, `make_v3.py`), their outputs, and the number audits (`audit/`) |

## Reproducing the analyses (CPU only)

Python 3.12 with the packages in `research_v2/v4/V4_PACKAGES.txt` (torch 2.13.0+cu126 for the GPU runs; the analyses need
numpy only). From `research_v2/`:

```
python ../verify_release.py                          # every locked file and run file hashes to the v3 and v4 locks
python v4/analyze_v4.py analyze                      # rewrites v4_runs/v4_report.json (about 80 min on CPU)
python v4/recompute_v4_independent.py                # independent recomputation, compared under the declared tolerances
python v4/exploratory_v4.py                          # exploratory X1-X4
python paper_journal/make_v4.py                      # every v4 number macro, table and figure of the article
python paper_journal/audit/audit_v4_numbers.py       # re-derives every macro from its source
```

`tools/lock_v4.py --verify` and `tools/lock_v3.py --verify` need the original git history; `verify_release.py` checks the
same hashes without it. `v4_runs/v4_report.json` must hash to `df57a31483160e62351378576d14123893081249e06525ed5be5f5099758dccb` (checked by
`make_v4.py`). The v3 analyses run the same way from `analyze_v3.py`, `v3/phase5/` and `paper_journal/make_v3.py`.

Re-running the GPU studies needs the models, a CUDA GPU and an authorization file per run, as described in
`v4/cloud/RUNBOOK_V4.md` and `v3/PROTOCOL_V3.md`.

## What is not included, and why

- Model weights and Hugging Face caches (licences; downloaded at pinned revisions).
- `v4_runs/checkpoints/`: the per-model archives used to move records off the cloud volume. Their contents are included
  unpacked. The hash of every file on the cloud volume at close-out is in `v4/closeout/volume_sha256sums.txt`; the
  archives and the pods' host-key and connection files (`known_hosts*`, `pod_ssh.json`) are the only entries listed there
  that are not shipped; every other listed file is included with a matching hash.
- The manuscript sources and internal review documents.
- Access tokens: the model-download token existed only as a cloud-provider secret and was never written to disk.

## Licences

Code: MIT (`LICENSE`). Records, reports and manifests: CC BY 4.0 (`LICENSE-DATA`). Model and dataset attributions,
including the Llama 3.1 and 3.2 notices: `NOTICE`. Citation: `CITATION.cff`.

## Integrity

`RELEASE_SHA256SUMS.txt` lists the SHA-256 of every file in this package.
