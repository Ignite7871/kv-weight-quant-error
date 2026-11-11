# research_v2 — research-recovery workspace (branch `research-recovery`, local only)

Work on the post-TMLR (#13386) recovery. Nothing here modifies the rejected submission:
- `paper_tmlr/` is untracked and untouched;
- `src/` and `scripts/` are imported read-only;
- the archive of the rejected package is at `D:\Projects\Dynamic Precision Scaling\archive_tmlr13386_2026-10-22\` (read-only, SHA-256 manifest).

| Path | What |
|---|---|
| `RESEARCH_RECOVERY_PLAN.md` | assessment, novelty comparison, roadmap (2026-10-22) |
| `PILOT_SPEC.md` | implementation-ready feasibility pilot specification and open questions |
| `diagnostics/` | exploratory analyses of historical data (`lag_check.py`, `power_check.py`) and the independent alignment check (`alignment_check.py`) |
| `rv2/` | instrumentation: policy-pluggable dual-precision generation, answer boundary, budget accounting |
| `tests/` | CPU validation suite (`run_cpu_tests.py`, 23 tests with `--real`) and mutation check (5 injected bugs) |
| `run_pilot.py`, `analyze_pilot.py` | pilot driver (GPU needs `--authorized-gpu-run`; enforces the GPU-hour cap via `pilot_runs/gpu_ledger.jsonl`) and report |
| `PROTOCOL_LOCK.txt`, `COAUTHOR_LOCK_NOTE.md` | binding private protocol lock for the pilot and the note for coauthor review |
| `validation_logs/` | outputs of the CPU validation runs |
| `tools/lock_protocol.py` | private dated SHA-256 protocol lock |

Status labels: everything derived from historical data is **exploratory**. Confirmatory status applies only to analyses specified in a locked protocol before the corresponding runs.

Environment: the project venv (`venv/`, torch 2.13.0+cu126, transformers 5.13.0, bitsandbytes 0.49.2) for everything except `diagnostics/lag_check.py` and `diagnostics/power_check.py`, which need scipy and scikit-learn (they were run with `D:\Projects\RAG\.venv312`).
