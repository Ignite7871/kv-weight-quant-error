# RUNBOOK_V4: cloud setup and run sequence (PROTOCOL_V4 §10, §11, §14)

Who does what follows D8. **The corresponding author** owns the RunPod account, the billing, the HF token, the pod's start and stop, and
every authorization file. **Claude Code** works only inside the pod, over SSH, through `tools/v4_remote.py`, while the authorization
file for the current tag exists. It has no `runpodctl`, no RunPod API key and no billing access.

**Changed before GM (deviation log):** V4-A3 changed D8: Claude Code operates the pod lifecycle through the official RunPod MCP, and the
corresponding author approves each create, start, stop and delete in the permission prompt; volume and secret deletion are denied in the
Claude Code settings. V4-A4: the volume is HIGH_PERFORMANCE in CA-MTL-1 ($0.14/GB-month), because no datacenter had A40 stock and Standard
volumes; the ledger keeps $0.07, so enter billing lines (step 8). V4-A5: the GM authorization file was written at the author's instruction.

## A. One-time setup by the corresponding author (before GM)

1. **Prepaid balance:** load exactly **$30** (D7, r5). Turn off any automatic top-up. The run stops at **$29**, so that $1 stays for the last download and for the volume until you delete it. The RunPod documentation available here does not say what happens to a stopped pod or a network volume when the balance nears zero: please check this before starting, because the volume holds any records not yet fetched.
2. **Hugging Face:**
   - The token's account must have **accepted the licences** of `meta-llama/Llama-3.2-1B`, `meta-llama/Llama-3.2-1B-Instruct`,
     `meta-llama/Llama-3.2-3B` and `meta-llama/Llama-3.1-8B`. On 2026-11-03 the laptop account had no access to the last two.
   - Create a **read-only** token and store it as a RunPod **Secret**.
3. **Network volume:** 100 GB, Standard, in a **Secure Cloud** datacenter with A40 capacity. It is mounted at `/workspace`.
4. **SSH key:** register the laptop's public key in the RunPod account **before** creating the pod; keys are injected at boot.
5. **Pod:**

   | Setting | Value |
   |---|---|
   | GPU | 1 × A40 |
   | Cloud and pricing | Secure Cloud, On-Demand (not spot / interruptible) |
   | Image | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, pinned by tag; use exactly this tag. `pod_setup.sh` records the digest Docker Hub reports for it at first boot. The study's own torch 2.13.0 + cu126 lives in `/workspace/venv`. |
   | Volume | the network volume at `/workspace` |
   | Ports | expose **22/tcp** (public IP; scp does not work through the proxied SSH endpoint) |
   | Environment | `HF_TOKEN` = the RunPod secret; nothing else secret |

   The driver reads secrets from PID 1's environment, because RunPod does not pass pod variables to SSH shells.
6. **Idle auto-stop (D8):** set it if your account offers one, with a window of **30 min or more**. The RunPod documentation available
   here describes no native idle stop for pods. If there is none, stop the pod by hand between models and tell me; the in-pod ledger then
   assumes the window you declare below.
7. **Files on the volume** (`/workspace/v4_runs/`), written by you:
   - `cost_config.json`:

     ```json
     {"rate_usd_per_h": 0.59, "autostop_minutes": 30, "volume_gb": 100, "volume_created_utc": "YYYY-MM-DDTHH:MM:SSZ"}
     ```

     Use the rate RunPod shows for the pod, and the volume's creation time.
   - The authorization files (section B).
8. **Files on the laptop** (`research_v2/v4_runs/`, never committed):
   - `pod_ssh.json`:

     ```json
     {"host": "<public IP>", "port": <external port for 22>, "user": "root", "identity_file": "C:/Users/<you>/.ssh/<key>"}
     ```

     The port can change after a stop and start; update the file then.
   - A copy of every authorization file.

## B. Authorization files (only the corresponding author creates them)

Each file is named `AUTHORIZED_V4_<tag>.txt` and contains the words `PROTOCOL_V4`, the current lock id (`python research_v2/tools/lock_v4.py --lock-id`)
and `<tag>` as a separate word. A file written for an earlier lock id authorizes nothing. Example content: `PROTOCOL_V4 <lock id> qwen2.5-0.5b`. Each file goes in **both** places: the pod's
`/workspace/v4_runs/` and the laptop's `research_v2/v4_runs/`.

| When | Tag(s) |
|---|---|
| Before setup and GM | `GM` (it also covers the bundle push and `pod_setup.sh`) |
| Step 0 | `qwen2.5-0.5b`, `llama3.2-1b-instruct`, `llama3.2-1b`, and `llama3.1-8b` (download and the 3-problem G6 probe only, before its ladder turn) |
| Ladder, one at a time or all at once | `qwen2.5-1.5b`, `llama3.2-3b`, `qwen2.5-3b`, `qwen2.5-7b` (`llama3.1-8b` already exists) |
| Last, only if the locked design keeps it | `mistral-7b-v0.3` |
| After a Step 0 failure, only if you decide | `STEP0_AUTHOR_DECISION.txt` containing `PROTOCOL_V4`, the lock id and `proceed` or `stop` |

## C. Run sequence (Claude Code, over SSH, under the files above)

`R` stands for `python research_v2/tools/v4_remote.py`; `X` stands for `R --tag <tag> exec --`. The options always come **before** the
subcommand (r6). From Git Bash, run every call with `MSYS_NO_PATHCONV=1` set, or `/workspace/...` arguments are rewritten as Windows paths.

0. (r8) Before the pod starts: the GM rehearsal on the laptop's CUDA GPU in a clean clone of the lock commit plus the files the pod receives (PROTOCOL_V4 §15 r8-5). Start the pod only if every test passes and every mutant is caught.
1. `R --tag GM check`, then `R --tag GM --commit <lock commit> push`, then (r8) the v3 records: `python research_v2/tools/v4_v3_records.py pack <tar>` on the laptop, `R --tag GM put <tar> /workspace/v3_records.tar`, `X tar -xf /workspace/v3_records.tar -C /workspace/repo/research_v2`, `X python research_v2/tools/v4_v3_records.py verify`; then `X bash research_v2/v4/cloud/pod_setup.sh`. The lock commit is the one that contains the lock file: `git log -1 --format=%H -- research_v2/V4_PROTOCOL_LOCK.txt`. `pod_setup.sh` checks the GPU, records the image digest and verifies the lock.
2. `X python research_v2/v4/run_v4.py gm --authorized-gpu-run`, then `X python research_v2/tools/lock_v4.py --append-gm`. GM logs the driver version, CUDA version and ECC mode.
3. For each Step 0 model M (`qwen2.5-0.5b` → `llama3.2-1b-instruct` → `llama3.2-1b`):
   1. `download`, then `gates` (if G3 fails in FP16: `lock_v4.py --append-flag M`);
   2. `calibrate`, then `analyze_v4.py calibrate`, then `lock_v4.py --append-calibration M`;
   3. for qwen2.5-0.5b only: `g6`;
   4. `test`, then `checkpoint`, then on the laptop `R --tag M fetch`.
4. `download --model llama3.1-8b`, then `g6-probe --model llama3.1-8b`.
5. On the laptop: `analyze_v4.py step0` (it compares with the locked v3 records). Then `R --tag <tag> put` the file `step0.json` and run
   `lock_v4.py --append-step0` on the pod. If Step 0 fails: **STOP and report**; the authors decide (section B).
6. `run_v4.py g6-design`, then `lock_v4.py --append-design`. **STOP (Phase 4).**
7. Ladder (Phase 5): each model as in step 3, in the fixed order: `qwen2.5-1.5b` → `llama3.2-3b` → `qwen2.5-3b` → `qwen2.5-7b` → `llama3.1-8b`, then `mistral-7b-v0.3`.
   Mistral is first to drop: the design may drop it, and its `download` refuses and records it as not run if the money left no longer covers it. Nothing else is cut automatically. If the design fails even without Mistral, **STOP and report.**
7a. (D1) After a post-data deviation: push the commit with the record, then on the pod `lock_v4.py --append-deviation <ID>` before any GPU command, then GM again (`run_v4.py gm`, `lock_v4.py --append-gm`, which writes `GM <ID>:`); a model whose test resumes runs the resume check first.
8. After every checkpoint: you enter a billing line from the RunPod billing page
   (`run_v4.py billing --usd <cumulative> --utc <time>`), or send me the figure to enter as yours.

9. **Close-out, right after the last download** (also after any stop you decide is final). On the laptop run `run_v4.py closeout`. It lists anything not verified on the laptop. Only when it says every checkpoint is verified:
   - you delete the network volume (this cannot be undone, and storage costs about $0.23 a day until you do) and terminate the pod;
   - you add `"volume_deleted_utc": "YYYY-MM-DDTHH:MM:SSZ"` to the laptop's `cost_config.json` and enter the final billing line.

The pod commands (`download` through `checkpoint`) are `X python research_v2/v4/run_v4.py <cmd> --model M --authorized-gpu-run`.

During the run I report only gates, calibration constants, G6, GPU hours, USD and integrity events. No test record is analysed before all
models finish (§10).
