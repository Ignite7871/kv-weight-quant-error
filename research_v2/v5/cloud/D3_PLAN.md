# V5-D3: U1 re-execution plan (awaiting the corresponding author's go)

Lock 14d2e1bf669417b8 with D1, D2, D3. Code commit d361d43 (D3 record committed after it). Measurement code byte-identical to 27e32b7.
No cloud action has been taken. Every RunPod step goes through the MCP with the author's approval.

## Steps (each pod step only after the go)

1. **Volume and pod** (MCP, approval each): a new Standard network volume in EU-SE-1, 50 GB ($3.50/month, as before); a pod on it with the
   same image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, 1 × A40 Secure ($0.59/h), port 22/tcp, the same SSH public keys, no HF token.
   If no A40 is available in EU-SE-1: report and wait.
2. **Billing line at volume creation** (author): enter the cumulative RunPod figure with that time
   (`run_v5.py billing --usd <cumulative> --utc <time>`). Needed: the ledger counts storage from the last billing line (01:13Z on 13 November),
   about $0.117 per day, and the U1 start check leaves only about $0.84 for everything before the gates.
3. **SSH**: host key read through the RunPod proxy and pinned; `pod_ssh.json` updated.
4. **Restore before setup**: `tools/v5_d3_prepare.py --out <dir> --pod-id <P> --volume-id <V> --volume-created-utc <T>`; `put` the tar and
   its .sha256; on the pod: check the .sha256, extract into `/workspace/v5_runs`, check SHA256SUMS. Then `cloud/pod_setup_v5.sh` (it keeps the
   first volume's image record and logs a changed digest, which would need a deviation row).
5. **Code**: push the D3 commit (HEAD with the V5-D3 record); `lock_v5.py --append-deviation D3`; `--verify` must print "lock verifies"
   with `recorded lost: gm_D2.json`.
6. **Authorization files** GM and U1 on the volume, written at the author's instruction (laptop copies already exist).
7. **Phase u1d3**: `setsid -f nohup bash research_v2/v5/run_v5_pod.sh u1d3 > /workspace/v5_runs/driver_u1d3.log 2>&1 < /dev/null`, laptop
   watcher `watch_v5.sh U1 u1d3`. The driver: GM D3 (+ append), U1 download, gates, calibration, d3-calibration-check, test (100 problems,
   13 checked against the first run), checkpoint.
8. **On watcher exit**: done → `run_v5.py closeout --tag U1` with the pod up (fresh manifest; the laptop must hold the newest U1 checkpoint),
   then stop the pod; any stop or failure → stop the pod, records untouched, report. Volume deletion is the author's.
9. **Then**: the locked analysis and the independent recomputation are run again on the author's word (U1 back in the A families).

## Stops (beyond the locked ones: gates, D2 drift reference 49.9 s/problem with 25 %, $14 ledger stop, unit start check)

- the recomputed U1 calibration differs from the locked hash ad6dfa7a… (d3_calibration_mismatch.json);
- any of the 13 surviving problems (172, 811, 334, 24, 606, 916, 335, 350, 786, 1102, 406, 242, 1302) differs bitwise (d3_reference_failure.json);
- a BF16 flag at the U1 gates;
- a unit-start refusal records U1 as not run (study stays without U1).

## Projection (A40, $0.59/h; times from the first U1 run)

| Item | h |
|---|---|
| Pod start to GM (setup, restore, push, D3 append) | 0.28 |
| GM D3 | 0.13 |
| U1 download | 0.10 |
| Gates | 0.27 |
| Calibration + D3 check | 0.54 |
| Test: 100 problems + 3 G8 reruns at about 47.5 s | 1.36 |
| Checkpoint, close-out with the pod up, stop latency | 0.15 |
| **Pod total** | **2.83 h = $1.67** |

- Storage, 50 GB Standard: $0.25 if the volume lives 48 h after the run (the locked allowance), about $0.02 if deleted at once.
- **D3 total about $1.92; cumulative about $13.61** against the $14 stop and the $15 ceiling (spent $11.692; balance about $6.18).
- Ledger checks, with the billing line of step 2: unit start check at the gates about $12.29 + $1.47 = $13.76 ≤ $14 (margin about $0.24,
  about 24 minutes of extra delay before the gates); before the last test problem about $13.61 (margin about $0.39). The ledger adds a
  30-minute allowance per pod session, so a second pod session (a restart) costs $0.30 of margin by itself.
- Not projected: a slower A40 host. U1's first run went from 36.2 s projected to about 47.5–50 s measured; the D2 drift stop is at 62.4 s.
