# Protocol lock for coauthor review — research_v2 GPU feasibility pilot

- **Lock file:** `research_v2/PROTOCOL_LOCK.txt`
- **SHA-256 of the lock file:** `47bd6caea51c81083c05082925df98a57749a21d384b9a336eb1c9c632e73935`
- **Commit containing the locked files:** recorded inside the lock file.
- **Created:** 2026-10-22, before any GPU run of the pilot.

## What this lock is and is not

- It is a **private, dated hash** of the pilot protocol (`PILOT_SPEC.md`), the code, the sample and the historical code the pilot imports. It is meant to be shared with the coauthors before the GPU pilot.
- It is **not an independent or public pre-registration**. Describe it as "a privately locked protocol informed by prior exploratory analyses".
- The protocol is **informed by prior exploratory findings** on the historical data:
  - lag-1 vs same-step entropy as predictors of quantization fragility;
  - power calculations from the stored paired outcomes;
  - the observation that historical precision budgets include unscored post-answer steps.

  These findings remain exploratory.
- The pilot is a **feasibility study**: correctness, budget comparability, answer-boundary reliability, runtime. Its accuracy differences (50 problems per arm) are descriptive only, and no method is selected on them.

## Verifying

```
git show <commit>:research_v2/PILOT_SPEC.md | sha256sum     # compare with the line in PROTOCOL_LOCK.txt
sha256sum research_v2/PROTOCOL_LOCK.txt                      # compare with the hash above
```

Any change to a locked file after this point is a protocol amendment. It must be listed with its reason, and analyses affected by it are labelled exploratory.
