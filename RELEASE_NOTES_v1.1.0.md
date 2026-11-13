# v1.1.0: second locked study (v5)

Release accompanying *Weight-Quantization Error in the KV Cache Is Not Isotropic Noise: Value
Error Is Consistently Harsher across 0.5B–8B Language Models* (Gunupati, Anavekar, and Varghese;
submitted to the *Journal of Artificial Intelligence Research*).

## What's new

v1.1.0 adds the second locked study, **v5** (Section 6 and Appendix C of the article). The v3
and v4 material from v1.0.0 is unchanged.

v5 asks three questions:

- **Size or head dimension?** A crossed pair: Qwen3-0.6B-Base and SmolLM2-1.7B.
- **A second 4-bit quantizer:** HQQ.
- **A second task:** WikiText-2.

It ran on one rented A40 under lock `14d2e1bf669417b8`, with post-data deviations D1–D3, each
recorded in `research_v2/v5/deviations/`.

## Added to the repository

- `research_v2/v5/`: protocol, code, tests, mutation checks, tolerances, deviation log and
  records, AI-use log, lock history, model manifest, WikiText-2 sample (indices and hashes
  only), runbook, rehearsal and validation records.
- `research_v2/V5_PROTOCOL_LOCK.txt`, with every appended line.
- v5 tools in `research_v2/tools/`: `lock_v5.py`, `v5_remote.py`, `v5_build_manifest.py`,
  `v5_d3_prepare.py`, `v5_gpu_mutation_check.py`, `v5_wikitext_release_scan.py`.
- The article's v5 number generator and audit: `paper_journal/make_v5.py`, `tables_v5/`,
  `audit/audit_v5_numbers.py`.
- `verify_release.py`, now also checking the V5 lock.

## Downloads

Unpack each archive inside `research_v2/`, then run `python verify_release.py` from the
repository root (see README).

| File | Contents | Size |
|---|---|---|
| `v5_runs.tar.gz` | **New.** v5 (anchor, A, B, D; 7 units): records, ledgers, gate checks, locked report, recomputation, D3 checks | 358 MB |
| `v4_runs.tar.gz` | v4 scale study (8 models + Step 0) | 311 MB |
| `v3_runs.tar.gz` | v3 four-model study | 135 MB |
| `v3_runs_x9.tar.gz` | v3 follow-up X9 | 128 MB |
| `v3_runs_x13.tar.gz` | v3 follow-up X13 | 97 MB |
| `e1_e1l_runs.tar.gz` | earlier studies E1 and E1L (background) | 13 MB |

SHA-256 checksums:

    4c95eec74dae1d0bb25d16cd0f2887da7364f08ff34bad2a873893c36a63c49f  v5_runs.tar.gz
    0e064c7b3d57ca6f5d3c5869d40c4e7e681f501fd619c97eb697d6897de41623  v4_runs.tar.gz
    9cd1250b8991dcf07d7a9749d92db3729e2fe09682deaf5bf6f4caa808df2999  v3_runs.tar.gz
    3cafe7ef28c84e3e9723156c794bcbfe33bd504abbdee12bbc3c32092ab745ff  v3_runs_x9.tar.gz
    3e561d053869cd92e49a79c9b744b66ce8bdc510be36b8a7bed403ef5c873cae  v3_runs_x13.tar.gz
    df6e94d44df06c0324356de4b65f0cc00a102c95249631b7d71b72b560eb870d  e1_e1l_runs.tar.gz

## WikiText-2

No released file contains WikiText-2 text or token ids. The WikiText units (U3, U5) store
argmax ids and histories only as SHA-256 hashes.
`research_v2/tools/v5_wikitext_release_scan.py` rebuilt every sample paragraph from the pinned
dataset files and found no copied text in any released file or record.

## Licences

Code: MIT. Records: CC BY 4.0. Model and dataset attributions are in `NOTICE`. No model weights
or dataset text are included.