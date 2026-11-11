# Protocol lock for coauthor review — research_v2 Stage 2 (held-out validation)

- **Lock file:** `research_v2/STAGE2_PROTOCOL_LOCK.txt`
- **SHA-256 of the lock file:** `8241c4dc1693eec80872839c081386cce3182af9fbdd84d854847fe71a950523`
- **Commit containing the locked files:** `4dd8ebb` (branch `research-recovery`, local); see the first lines of the lock file for the full hash and UTC time.
- **Created:** 2026-10-25, **before any GPU use in Stage 2 and before any held-out problem was evaluated.**
- **Sharing:** this note and the lock file are for the coauthors, privately. The assistant has not sent them anywhere (it has no channel to do so); sharing is the authors' action.

## What this lock is and is not

- A **private, dated hash** of the Stage 2 specification, driver, analysis, tests, the frozen sample, the pre-run checks and every Stage 1 file the run imports. It is **not a public or independent pre-registration.**
- The five hypotheses (`STAGE2_SPEC.md` §1) are **exploratory in origin**: they are the five primary contrasts of Stage 1 with the directions that Stage 1 found on the 930-problem pool. The held-out sample (389 problems, the complement of the pool) is the first sample on which they are tested.
- **The held-out set is not completely untouched.** Two problems (test indices 4, 5) appeared in an earlier eight-problem console probe; 47 (test index < 200, including those two) had their gold solutions token-labelled by a teacher-forced FP16/INT4 argmax comparison (`results/eqf_spike_dataset_*.csv`). No generated-answer correctness outcome was retained for any of them. All 389 stay in the primary validation, as the author approved; the pre-declared sensitivity analysis excludes the 47 (342 problems).
- **Power is low for H1–H3** (design power from Stage 1 data only: P(Holm < 0.05) ≈ 0.10, 0.21, 0.18; H4 ≈ 0.85). Non-significant held-out results for H1–H3 will be reported as inconclusive, not as failure to replicate.

## Frozen

389 problem ids and seeded order (`stage2_sample.json`); hypotheses H1–H5 and directions; the Holm family (raw and adjusted p reported); equivalence margin ±3 pp (H5); replication categories (§6.3); secondary combined (n = 1,319) and sensitivity (n = 342) analyses; the code (Stage 1 / A2 generation and integrity code unchanged); the 5.5 h absolute GPU ceiling for all Stage 2 GPU use; monitoring limited to integrity, accounting and hardware telemetry.

## Verifying

```
sha256sum research_v2/STAGE2_PROTOCOL_LOCK.txt            # compare with the hash above
git show 4dd8ebb:research_v2/STAGE2_SPEC.md | sha256sum    # compare with the line in the lock file
```

Any change to a locked file after this point is an amendment, listed in `STAGE2_SPEC.md` §9 with its reason; affected analyses are labelled exploratory.
