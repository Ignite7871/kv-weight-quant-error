# Protocol lock for coauthor review — research_v2 Stage 1 (token-source × KV-source)

- **Lock file:** `research_v2/STAGE1_PROTOCOL_LOCK.txt`
- **SHA-256 of the lock file:** `2344f57b05f3b420b4d0cfd471b974eed691cd3177ad346065fc82cd1250eb11`
- **Commit containing the locked files:** `4b0ff0b602fc40b61a7bd3aa282e1121ec7c131c` (branch `research-recovery`, local)
- **Created:** 2026-10-23 14:03 UTC, after all correctness gates and **before** any evaluation problem was run.
- **Sharing:** this note and the lock file are for the coauthors, privately. They have not been sent anywhere by the assistant; sharing is the authors' action.

## What this lock is and is not

- It is a **private, dated hash** of the Stage 1 specification (`STAGE1_SPEC.md`), the driver, gates, analysis, tests, the instrumentation code and the historical code the run imports.
- It is **not an independent or public pre-registration.** Describe it as "a privately locked protocol, informed by prior exploration".
- **The protocol is exploratory in origin.** It follows earlier looks at the same data:
  - the 2026-10-22 pilot, where the flip oracle diverged from FP16 on 50/50 train problems (median first divergence at token 20);
  - the historical outcomes on the 930-problem test pool (FP16 336 vs NF4 243 solved) and several routing comparisons run on it;
  - 30 pool problems re-run under several policies in the pilot.
- The 930-problem evaluation is therefore an **exploratory replication with pre-declared estimands on previously examined data.** It is not confirmation of any hypothesis those looks suggested. The 389 test problems outside the pool are reserved for a separately reported validation and are not touched.

## What was verified before the lock (and what changed afterwards)

| Gate | Result | Evidence |
|---|---|---|
| G1 pilot CPU suite + mutation check | 24/24 (`--real`), 5/5 mutants killed | `validation_logs/stage1_g1_*.txt` |
| G2 Stage 1 CPU suite + mutation check + analysis tests | 11/11 (`--real` adds one), 6/6 mutants killed, 8/8 | `validation_logs/stage1_g2_*.txt` |
| G3 GPU equivalence | pure cells ≡ static ≡ untouched `DynamicQwen.generate` (10/10, caches bitwise); hybrids ≡ independent deep-copy reference (6/6, caches bitwise, ≡ single-precision replay); early stop identical (14/14) | `validation_logs/stage1_gates/gate_equiv.*` |
| G4 feasibility (10 train problems) | token = declared-source argmax at every step, labels consistent at step 1, both hybrids diverge from FP16, flip oracle ≡ cache-only until its first flip. **Passed after amendment A1** (see below) | `validation_logs/stage1_gates/feasibility_*` |
| G5 speed re-check | about 9% faster than 2026-10-23, tokens deterministic across repetitions | `validation_logs/stage1_gates/speed.json` |
| G7 projection | main run 9.8 h central (7.7–12.0), cumulative 10.18 h central vs the 12 h cap; the upper estimate (12.33 h) does not fit | `validation_logs/stage1_gates/projection.json` |

**Amendment A1 (disclosed in `STAGE1_SPEC.md` section 10):** the first feasibility run reported a failure only because an auxiliary sanity bound on the JS divergence used 0.0 as lower limit and one value was −9.7e-9 (float32 rounding at a near-identical step). The bound became −1e-6 after that observation; the failing report is preserved. No criterion in section 4 of the spec was changed.

**Code changes between the GPU gates (commit `fec1a19`) and the lock (`4b0ff0b`):** the generation code (`rv2/`), the analysis, `src/` and `scripts/` are **byte-identical**. Changed: amendment text, the gate tolerance, a driver stop rule (stop and request approval when the live projection exceeds the cap), crash-safe ledger recovery, and their tests.

## Verifying

```
git show 4b0ff0b:research_v2/STAGE1_SPEC.md | sha256sum    # compare with the line in STAGE1_PROTOCOL_LOCK.txt
sha256sum research_v2/STAGE1_PROTOCOL_LOCK.txt              # compare with the hash above
```

Any change to a locked file after this point is a protocol amendment. It must be listed with its reason, and analyses affected by it are labelled exploratory.

## Addendum, 2026-10-24: amendment A2 (after the lock, during the main run)

The locked driver stopped itself at 155/930 problems because it extrapolated the *current process's* mean time (46.7 s/problem over 40 problems) to a 12.15 h cumulative projection. The author then instructed (Option 1): judge completion on the pooled time over all sessions, keep the **absolute 12.0 h cumulative ceiling** (2.099 h already used), and accept a truncated sample rather than exceed it. **A2 changes only the runtime projection and stopping procedure** (plus runtime monitoring, per-problem integrity checks, hardware-safety telemetry and checkpoint verification), **not experimental policies, sampling, outcomes, hypotheses or analysis.** Details: `STAGE1_SPEC.md` section 10, A2.

- The lock file `STAGE1_PROTOCOL_LOCK.txt` (sha256 `2344f57b...eb11`) and the locked commit `4b0ff0b` are unchanged and still describe the protocol *as locked*. `run_stage1.py` and `STAGE1_SPEC.md` now differ from the locked versions; the A2 commit hash is the authoritative record of the difference (`git diff 16b2ed8 <A2 commit> -- research_v2/run_stage1.py research_v2/STAGE1_SPEC.md`). `rv2/*.py` (generation code), `analyze_stage1.py`, `stage1_sample.json`, `src/` and `scripts/` are byte-identical to the locked commit.
- The 155 problems completed before A2 are kept unchanged and are verified by hash before and after the resumed run.
- Any analysis of the resumed run is labelled exploratory in the same sense as the rest of Stage 1.
