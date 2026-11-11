# Fallback: switch to the on-pod driver after an SSH drop (prepared 2026-11-08, not launched)

Author's instruction (2026-11-08): leave the laptop driver running; if the SSH session drops and the ladder stops, switch to the
on-pod driver on restart (resume check as usual) and log how the run was operated; stop the pod immediately if a drop leaves it idle.

Files (scratchpad, not in the repo):
- `ladder_pod_d1.sh`: pod-side driver. Same steps and skip rules as `ladder_d1.sh`, minus fetch. It writes
  `/workspace/v4_runs/ladder_pod/{status.json,driver.pid,<model>.checkpointed}` and logs to `/workspace/v4_runs/ladder_pod_d1.log`.
  It refuses to start if another driver holds its flock or any run_v4 GPU command is running. It never calls RunPod and never
  writes an authorization file.
- `watch_pod_d1.sh`: laptop watcher. Polls every 5 min, fetches and verifies each checkpointed model with `v4_remote fetch`, and
  exits with one `WATCH …` line on done, failed, driver gone, or pod unreachable for 3 polls.

Launch check (2026-11-08 17:3x UTC, harmless probes, no GPU): `setsid -f nohup … > log 2>&1 < /dev/null` through `v4_remote exec`
(allowed by `command_allowed`) returns in about 4 s, the same as a plain exec. The child gets its own session and finishes after
SSH closes, and it inherits POD_ENV (V4_RUNS, venv, HF_HOME). A plain `setsid nohup … &` does NOT detach: the `&` binds to the
whole `POD_ENV && …` chain, so SSH waits. `flock` and `pgrep` exist on the pod.

## Steps when the laptop driver stops

Notation: `R` = `../venv/Scripts/python.exe tools/v4_remote.py`, `X T cmd` = `R --tag T exec -- cmd`; Git Bash with `MSYS_NO_PATHCONV=1`.

1. Read the tail of `ladder_d1.log`. A gate, drift, ledger ($29) or resume-check stop is a protocol stop, not a drop:
   stop the pod via MCP and report. Do not restart.
2. A drop (ssh error or exit 255, laptop sleep or network loss, with no protocol stop on the pod): check the pod with
   `X GM "pgrep -af run_v4.py; ls /workspace/v4_runs/*/test/../*stop*.json /workspace/v4_runs/*/resume_check_failure.json 2>/dev/null"`.
   - If a run_v4 process is still running, wait for it to finish or die (it loses its stdout reader). Never start a second one.
   - If the pod is idle (no run_v4 process), **stop the pod via MCP right away** (pod-action stop), then report the cost.
   - If the pod can't be reached, stop it via MCP (the MCP doesn't depend on the pod's SSH).
3. Restart (pod start via MCP). If its A40 isn't available, report and don't create a new pod: the earlier one-off approval
   doesn't carry over. Then:
   1. update `v4_runs/pod_ssh.json` with the new IP and port, re-pin the host key (direct ssh-keyscan fingerprint = proxy
      `ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub`), and append it to `v4_runs/known_hosts`;
   2. `X GM bash research_v2/v4/cloud/pod_setup.sh` (GPU, digest, lock verifies), and
      `X GM python research_v2/tools/lock_v4.py --verify` (must include D1 and `GM D1:`);
   3. `R --tag GM put <scratchpad>/ladder_pod_d1.sh /workspace/ladder_pod_d1.sh`, and compare its sha256 laptop vs pod;
   4. launch: `X GM "setsid -f nohup bash /workspace/ladder_pod_d1.sh > /workspace/v4_runs/ladder_pod_d1.log 2>&1 < /dev/null"`;
   5. confirm: `X GM "cat /workspace/v4_runs/ladder_pod/status.json; tail -5 /workspace/v4_runs/ladder_pod_d1.log"`.
      The interrupted model's test resumes via `run_test`, which runs the resume check first (last completed problem,
      bitwise). Then check `resume_check.jsonl` shows `bitwise_equal: true`;
   6. start `bash <scratchpad>/watch_pod_d1.sh` in the background. On its exit: done means stop the pod and write the final
      report; failed, gone or unreachable means stop the pod via MCP and report.
4. Log it: DEVIATION_LOG_V4 row (operational, no protocol change): drop time, what was running, idle minutes, pod stop
   and restart, the driver hash, the resume-check result, and that the rest of the ladder ran on the pod-side driver.
   Scan and commit (v4 files only).
