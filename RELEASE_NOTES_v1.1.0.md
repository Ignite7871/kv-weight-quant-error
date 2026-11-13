# v1.1.0: adds the second locked study (v5)

Release accompanying *Weight-Quantization Error in the KV Cache Is Not Isotropic Noise: Value Error Is Consistently Harsher across 0.5B–8B Language Models* (Gunupati, Anavekar, and Varghese; submitted to the Journal of Artificial Intelligence Research).

v1.1.0 adds the second locked study (v5, Section 6 and Appendix C of the article) to the v3 and v4 material of v1.0.0, which is unchanged. v5 covers three questions: size against head dimension (Qwen3-0.6B-Base and SmolLM2-1.7B), a second 4-bit quantizer (HQQ) and a second task (WikiText-2). It ran on one rented A40 under lock `14d2e1bf669417b8`, with post-data deviations D1–D3.

New in the repository:

- `research_v2/v5/`: protocol, code, tests, mutation checks, tolerances, deviation log, deviation records `V5-D1.json` to `V5-D3.json`, AI-use log, lock history, model manifest, WikiText-2 sample (indices and hashes only), runbook, rehearsal and validation records.
- `research_v2/V5_PROTOCOL_LOCK.txt`, with every appended line.
- The v5 tools (`lock_v5.py`, `v5_remote.py`, `v5_build_manifest.py`, `v5_d3_prepare.py`, `v5_gpu_mutation_check.py`, `v5_wikitext_release_scan.py`).
- The article's v5 number generator and audit (`paper_journal/make_v5.py`, `tables_v5/`, `audit/audit_v5_numbers.py`).
- `verify_release.py`, extended to the V5 lock.

Attach the new archive below together with the five v1.0.0 archives, which are unchanged:

| File | Contents | Size |
|---|---|---|
| `v5_runs.tar.gz` | v5 (anchor, A, B, D; 7 units): records, ledgers, gate checks, anchor, design, locked report and recomputation, D3 checks, authorization files | 358 MB |

SHA-256:

    4c95eec74dae1d0bb25d16cd0f2887da7364f08ff34bad2a873893c36a63c49f  v5_runs.tar.gz

The WikiText-2 hash-only rule: no released file holds WikiText-2 text or token ids. The WikiText units (U3, U5) store argmax ids and histories as SHA-256 hashes. `research_v2/tools/v5_wikitext_release_scan.py` rebuilt every sample paragraph from the pinned parquet files and found no copied text in any released file or record.

Code: MIT. Records: CC BY 4.0. Model and dataset attributions: `NOTICE`. No model weights or dataset text are included.
