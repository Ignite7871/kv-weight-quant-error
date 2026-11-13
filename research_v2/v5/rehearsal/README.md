# v5 rehearsal on the laptop RTX 4060 (PROTOCOL_V5 §10, DEVIATION_LOG_V5 rows V5-P1 and V5-RH)

Rehearsal mode (`V5_REHEARSAL=1`, `V5_REHEARSAL_N=3`) runs the locked code in a clean clone of the lock commit, against a local "pod"
directory, with a laptop watcher. The authorization files used say REHEARSAL; outside rehearsal mode `run_v5.py` refuses them. **Nothing
here is study data.** These records are 3-problem runs on a different GPU, kept only as evidence that the pipeline works end to end. They
contain hashes only, no WikiText text and no token ids. Local paths have been replaced with placeholders. There was no cloud action.

## Final rehearsal: lock 14d2e1bf669417b8 (r4), lock commit 17b7c66, 2026-11-11 15:28Z to 2026-11-12 03:36Z

| Step | Result |
|---|---|
| GM | 16/16 core tests and 68/68 mutants on CUDA; environment record complete (the nvidia-smi "CUDA UMD Version" banner was read, r3) |
| U0 anchor | gates passed; calibration appended; test 3/3, G8 bitwise 3/3; watcher fetched and verified the checkpoint |
| Anchor decision | criterion 1 holds for all four contrasts; criterion 2 fails for H2 and NAT−C1 (expected at n = 3); criterion 3 holds (ρK 0.0734 vs v4 0.0724, ρV 0.2117 vs 0.2095, b̂ 2.43 vs 2.63). Rehearsal decision "proceed" written to the scratch directory and appended (`ANCHOR:`) |
| Probes | qwen-nf4 4.342 s per arm-problem, qwen-hqq 4.900 s, **HQQ/NF4 = 1.13** (1.27 in the r3-lock run), HQQ quantization 1.45 s. Mistral probes are stand-ins (qwen × 1.9, marked `rehearsal_synthetic`, which a real design refuses) |
| Design | R1 (n = 60 for B/D), no unit cut, fit ×1.48 (v4 G6) and WikiText ×1.047; appended (`DESIGN:`) |
| r4 check, phase change | the design watcher saw the anchor's `done` and reported "design driver has not started yet (1)" instead of exiting; it exited 0 on the design's own `done` |
| Phase units | U1 to U4: gates passed (GW on U3, GH on U4 with state hash 8b93f27d…, not frozen), FLAG/HQQ/CALIBRATION lines appended, 3 test problems each, checkpoints verified |
| Second driver | `REFUSED: another on-pod driver is running` |
| Forced stop | the driver and GPU process were killed during U1's test (after 1 problem). The watcher exited 2 (`DRIVER GONE while state=running`) only after its confirming second look |
| Resume | the stale driver lock and the stale GPU-process lock were taken over and logged; `[RESUME CHECK] U1 problem 172: bitwise equal to the stored records`; U1 completed |
| Author pause | paused by the author during U4's gates (process stopped while it was still importing torch, before it took the GPU lock). On relaunch, U1 to U3 were skipped as complete and checkpointed, and U4 ran |
| U5, U6 | Mistral-7B does not fit in 8 GB: written by hand as rehearsal `not_run` stand-ins and skipped by the driver |
| Close-out | `ready: false`, only for U5 and U6 (no checkpoint: the stand-ins never went through the driver), so the volume-deletion gate stays closed as intended. U0 to U4 verify |
| GPU time | 2.06 h on the RTX 4060 (gates 84 min, test 22 min, calibration 9 min, GM 5 min, probes 3 min) |

The rehearsal USD in the cost ledger is inflated, and that is an artifact of rehearsing on Windows: each process counts as a new pod session
there. On the pod, the session start is PID 1's start time.

## Defects found by the rehearsals (both fixed before any study data, by re-lock)

- **r3** (lock 57998073077c7cd7 → cfe92ef11264ef8c): GM read the CUDA version only from the banner form `CUDA Version:`. Driver 621.14 prints
  `CUDA UMD Version:`, so GM failed. Fixed together with test o21 and one mutant.
- **r4** (lock cfe92ef11264ef8c → 14d2e1bf669417b8): the watcher took the previous phase's `done` as the end of the new phase, and it could
  read a dead driver's `running` as "driver gone" before a restarted driver had rewritten the status. Fixed: the watcher now takes a phase
  argument, a second look confirms "driver gone", the driver writes `starting`, and test o22 and one mutant were added.

## Folders

- `r4_lock_14d2e1bf/`: the final rehearsal. It contains GM, anchor, design, probes, the GPU ledger, the GPU-lock log, per-unit gates,
  calibration, G8 and HQQ records, the U5/U6 stand-ins, and the filtered driver and watcher logs.
- `r3_lock_cfe92ef1/`: the earlier full run, used for comparison: GM, anchor, design (R1+R2, U6 cut) and probes (HQQ/NF4 1.27). Its units
  phase ran on the r4 working-tree watcher, before r4 was committed.

## D2 rehearsal: lock 14d2e1bf669417b8 + D1 + D2, commit 2b12b9a, 2026-11-12 07:25Z to 07:40Z (DEVIATION_LOG_V5 row V5-RH2)

The rehearsal clone (from the r4 rehearsal above, U0 to U4 complete with n = 3) was taken through the same path the pod takes:

| Step | Result |
|---|---|
| Deviations | `--append-deviation D1` at 2b12b9a is refused ("run_v5.py is not at the deviation's hash"), which is correct because D2 changed that file after D1. In the pod's order (checkout 3916f2f, append D1, checkout 2b12b9a, append D2) both lines append and the lock verifies. Between the second checkout and the D2 append the lock does not verify, as expected; no driver runs then |
| Rehearsal-only fix | files checked out after the clone had CRLF line endings, because the laptop's global `core.autocrlf=true` overrode the clone-time `-c core.autocrlf=false`, and append-deviation compares raw bytes. Fixed by `core.autocrlf=false` in the clone's own config and a fresh checkout. Cannot happen on the Linux pod |
| GM D2 | 16/16 core tests and 73/73 mutants on CUDA (RTX 4060); `GM D2:` appended (gm_D2.json) |
| U1 resume after a drift stop | stand-in: U1's test_complete.json and checkpoint marker removed, a copy of the real U1 drift_stop.json placed, rehearsal n raised from 3 to 5. The driver skipped download, gates and calibration; `[RESUME CHECK] U1 problem 334: bitwise equal`; G8 on the next problem bitwise equal; U1 completed 5/5 and was checkpointed; drift_stop.json unchanged byte for byte; U2 to U4 skipped as complete, U5/U6 as rehearsal not-run stand-ins; watcher exit 0 |
| U2's first lock check | `run_v5.py download --tag U2` passed `require` (lock verify, GM D2, order rules, unit start check), exit 0 |
