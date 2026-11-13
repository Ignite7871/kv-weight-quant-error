# RUNBOOK_V5: cloud setup and run sequence (PROTOCOL_V5 §9, §10, §14)

**Roles.**
- **The corresponding author** owns the RunPod account and billing, approves every create, start, stop and delete in the permission prompt,
  writes every authorization file and `cost_config.json`, and deletes the network volume at the end.
- **Claude Code** operates the pod lifecycle through the official RunPod MCP (each call approved by the author) and the pod's shell only
  through `tools/v5_remote.py`, while the authorization file of the current tag exists. It never writes an authorization file.
- **No secrets.** Every model and the dataset are ungated, so there is **no HF token and no RunPod secret**. The pod must not have
  `HF_TOKEN` set; `pod_setup_v5.sh` and every pod command refuse if it does.

## A40 stock check at the lock (D-V5-8)

Read with the RunPod MCP on 2026-11-11 (`get-gpu-type`, `list-data-centers`):
- A40, Secure Cloud: $0.59/h on demand. Stock only in **CA-MTL-1**, availability LOW.
- CA-MTL-1 offers **HIGH_PERFORMANCE** network volumes only ($0.14/GB-month).
- No datacenter has A40 stock together with Standard volumes.
- **Hence the D-V5-8 fallback applies: CA-MTL-1 with a HIGH_PERFORMANCE volume** (as in v4), unless a re-read right before creation shows an
  A40 + Standard datacenter.
- The A40 hosts report CUDA 13.2 drivers (12.8 hosts unavailable). The image's CUDA 12.8 runtime and the venv's torch 2.13.0+cu126 run on that
  driver. GM logs the driver and CUDA versions.

## A. One-time setup by the corresponding author (before GM)

1. **Balance:** it stays at about $17.87 (D-V5-7). Turn off automatic top-up. The run stops at **$14** in the ledger (ceiling $15 including
   storage).
2. **SSH key:** register the laptop's public key in the RunPod account before the pod is created; keys are injected at boot.
3. **Network volume:** 50 GB in the chosen datacenter (CA-MTL-1, HIGH_PERFORMANCE unless the re-read says otherwise), mounted at `/workspace`.
   Claude Code creates it through the MCP after stating the price, with your approval.
4. **Pod:**

   | Setting | Value |
   |---|---|
   | GPU | 1 × A40 |
   | Cloud and pricing | Secure Cloud, On-Demand (not spot / interruptible) |
   | Image | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (the v4 tag; its digest is recorded at first boot). The study's torch 2.13.0+cu126 lives in `/workspace/venv` |
   | Container disk | 30 GB |
   | Volume | the network volume at `/workspace` |
   | Ports | expose **22/tcp** (public IP; scp does not work through the proxied SSH endpoint) |
   | Environment | **none**: no `HF_TOKEN`, no secret |

5. **Idle stop:** the RunPod docs available here describe no native idle auto-stop for pods. The watcher's exit tells Claude Code to stop the
   pod through the MCP; declare `autostop_minutes` (30) as the ledger's conservative window.
6. **Files on the volume** (`/workspace/v5_runs/`), written by you (or by Claude Code at your explicit instruction, logged in
   `DEVIATION_LOG_V5.md`):
   - `cost_config.json`:

     ```json
     {"rate_usd_per_h": 0.59, "autostop_minutes": 30, "volume_gb": 50, "volume_created_utc": "YYYY-MM-DDTHH:MM:SSZ",
      "volume_tier": "high_performance", "storage_usd_per_gb_month": 0.14}
     ```

     Use `"standard"` and `0.07` only if the volume is created as Standard.
   - The authorization files (section B).
7. **Files on the laptop** (`research_v2/v5_runs/`, never committed):
   - `pod_ssh.json`: `{"host": "<public IP>", "port": <external port for 22>, "user": "root", "identity_file": "C:/Users/<you>/.ssh/<key>"}`;
   - a copy of every authorization file.

## B. Authorization files (only the corresponding author creates them)

Each file is named `AUTHORIZED_V5_<tag>.txt` and contains `PROTOCOL_V5`, the lock id (`python research_v2/tools/lock_v5.py --lock-id`)
and `<tag>` as a separate word, for example `PROTOCOL_V5 <lock id> U1`. It goes in **both** `/workspace/v5_runs/` on the pod and
`research_v2/v5_runs/` on the laptop. A file for another lock id, or one containing the word `REHEARSAL`, authorizes nothing.

| Phase (§14) | Tags |
|---|---|
| 4a: setup, GM, U0 | `GM` (also covers push and `pod_setup_v5.sh`), `U0` |
| 4b: after the anchor decision | `DESIGN` (Mistral download, the four probes, the design) |
| 5: after the Phase 4 STOP | `U1`, `U2`, `U3`, `U4`, `U5`, `U6` (one at a time or all at once; cut units never start) |
| Only after an anchor failure, if you decide | `ANCHOR_AUTHOR_DECISION.txt` with `PROTOCOL_V5`, the lock id and `proceed` or `stop` |

## C. Run sequence (Claude Code)

`R` = `python research_v2/tools/v5_remote.py`; `X <tag>` = `R --tag <tag> exec --`. Options come **before** the subcommand. From Git
Bash, set `MSYS_NO_PATHCONV=1`.

0. **Before the pod:** the rehearsal on the laptop RTX 4060 in a clean clone of the lock commit (§10; done at the lock, see
   `v5/validation/`). Repeat it only if anything changes.
1. **Create** (MCP, after stating the price, each approved): the network volume, then the pod (table A4). Then write `pod_ssh.json`, pin the
   host key, and check: `R --tag GM check`.
2. **Push and set up:** `R --tag GM --commit <lock commit> push`, then `X GM bash research_v2/v5/cloud/pod_setup_v5.sh` (GPU, no token,
   image digest, venv, hqq 0.2.8.post1, lock verifies).
   The lock commit is `git log -1 --format=%H -- research_v2/V5_PROTOCOL_LOCK.txt`.
3. **Phase anchor (on-pod driver):**
   - launch: `X GM "setsid -f nohup bash research_v2/v5/run_v5_pod.sh anchor > /workspace/v5_runs/driver_anchor.log 2>&1 < /dev/null"`;
   - watch: `bash research_v2/v5/watch_v5.sh GM anchor` in the background (the second argument is the phase: only that phase's status counts);
   - the driver runs GM (+ append), then U0: download, gates, calibration (+ append), test, checkpoint. The watcher fetches and verifies.
4. **Anchor decision (laptop):** `python research_v2/v5/analyze_v5.py anchor` (U0 against the locked v4 records), then
   `R --tag U0 put research_v2/v5_runs/anchor.json /workspace/v5_runs/anchor.json` and `X U0 python research_v2/tools/lock_v5.py --append-anchor`.
   If it fails: **STOP and report**; you decide (section B). Stop the pod while you decide.
5. **Phase design:** `X DESIGN "setsid -f nohup bash research_v2/v5/run_v5_pod.sh design > /workspace/v5_runs/driver_design.log 2>&1 < /dev/null"`,
   then the watcher with `DESIGN design`. It downloads Mistral, runs the four probes, writes the design with the reduction order (R1 → R2 → R3, else
   R4 stop), appends it and checkpoints. **STOP (end of Phase 4):** I report the probes, the HQQ ratio, the design, the GPU hours and the USD.
   The pod is stopped.
6. **Phase units (Phase 5):** after your `U1`…`U6` files:
   `X U1 "setsid -f nohup bash research_v2/v5/run_v5_pod.sh units > /workspace/v5_runs/driver_units.log 2>&1 < /dev/null"` and the
   watcher with `U1 units`. Each unit: download, gates (+ FLAG / HQQ lines), calibration (+ append), test, checkpoint. A unit that no longer fits
   under $14 is recorded as not run with every later unit, and the study closes out.
7. **On every watcher exit:**
   - done: run close-out (step 9) while the pod is up, then stop the pod through the MCP and report;
   - failed or protocol stop: stop the pod and report, with no restart;
   - driver gone (exit 2, confirmed by a second look) or unreachable (exit 5): stop the pod, report, and on restart relaunch the same phase
     (the resume check runs first);
   - exit 6 (the phase's driver never started) or 4 (a fetch failed verification): stop the pod and report.
8. **Billing:** after each phase you enter a billing line (`python research_v2/v5/run_v5.py billing --usd <cumulative> --utc <time>` on the pod
   through `X`), or send me the figure to enter as yours.
9. **Close-out:**
   - on the laptop, `python research_v2/v5/run_v5.py closeout --tag <T>` **while the pod is still running** (V5-D3): it first takes a fresh
     manifest of the pod's checkpoints (`tools/v5_remote.py --tag <T> manifest`) and is ready only if the laptop holds the pod's newest
     checkpoint of every tag, by name and SHA-256 (never by the "checkpointed" flag alone: that is how U1's completed checkpoint was lost,
     V5-A10). If it is not ready, `tools/v5_remote.py --tag <T> sync` and run close-out again; then stop the pod;
   - if the pod was stopped before close-out, start it again for the manifest before any deletion;
   - only when it reports every checkpoint verified do you delete the network volume and terminate the pod (cannot be undone);
   - then add `"volume_deleted_utc"` to the laptop's `cost_config.json` and enter the final billing line.
10. **V5-D3, U1 re-execution:** phase `u1d3` (`run_v5_pod.sh u1d3`, watcher `U1 u1d3`) on a new Standard volume, after the restore of
    `tools/v5_d3_prepare.py` (see `v5/deviations/V5-D3.json` and DEVIATION_LOG_V5 V5-D3).

During the run I report only gates, calibration constants, the anchor decision, the design, GPU hours, USD and integrity events; no test record
is analysed (apart from the anchor) before every unit has finished (§8).
