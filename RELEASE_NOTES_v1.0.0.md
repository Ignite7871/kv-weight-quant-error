# v1.0.0: code, locks, and per-problem records

Release accompanying *Weight-Quantization Error in the KV Cache Is Not Isotropic Noise: Value Error Is Consistently Harsher across 0.5B–8B Language Models* (Gunupati, Anavekar, and Varghese; submitted to the Journal of Artificial Intelligence Research).

The repository holds the code, protocols, locks, manifests, and documentation. The per-problem records of both locked studies are attached below as archives; unpack each inside `research_v2/` and run `python verify_release.py` from the repository root (see README).

| File | Contents | Size |
|---|---|---|
| `v4_runs.tar.gz` | v4 scale study (8 models + Step 0): records, ledgers, gate checks, locked report, recomputation | 311 MB |
| `v3_runs.tar.gz` | v3 four-model study records | 135 MB |
| `v3_runs_x9.tar.gz` | v3 follow-up X9 records | 128 MB |
| `v3_runs_x13.tar.gz` | v3 follow-up X13 records | 97 MB |
| `e1_e1l_runs.tar.gz` | earlier studies E1 and E1L (background) | 13 MB |

SHA-256 checksums:

    0e064c7b3d57ca6f5d3c5869d40c4e7e681f501fd619c97eb697d6897de41623  v4_runs.tar.gz
    9cd1250b8991dcf07d7a9749d92db3729e2fe09682deaf5bf6f4caa808df2999  v3_runs.tar.gz
    3cafe7ef28c84e3e9723156c794bcbfe33bd504abbdee12bbc3c32092ab745ff  v3_runs_x9.tar.gz
    3e561d053869cd92e49a79c9b744b66ce8bdc510be36b8a7bed403ef5c873cae  v3_runs_x13.tar.gz
    df6e94d44df06c0324356de4b65f0cc00a102c95249631b7d71b72b560eb870d  e1_e1l_runs.tar.gz

Code: MIT. Records: CC BY 4.0. Model attributions, including the Llama 3.1 and 3.2 notices: `NOTICE`. No model weights
are included.
