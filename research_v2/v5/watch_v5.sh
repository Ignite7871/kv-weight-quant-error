#!/usr/bin/env bash
# PROTOCOL_V5 section 10: the laptop watcher for the on-pod driver (run_v5_pod.sh). Polls every V5_WATCH_POLL seconds (default 300) over short
# tools/v5_remote.py calls:
#  - fetches and verifies (sha256, secret scan, SHA256SUMS) every checkpoint the pod holds that the laptop does not (V5-D3: by the pod's
#    checkpoint list and hashes, tools/v5_remote.py sync), every poll and before it reads a terminal state;
#  - EXITS with one WATCH line on every terminal state, so the pod can be stopped through the MCP right away:
#    0 phase done; 1 driver failed; 3 protocol stop (state "stopped"); 2 driver gone while "running"; 4 a fetch failed; 5 pod unreachable 3 polls;
#    6 the driver of <phase> has not started after 3 polls.
# r4: only a status of the expected phase counts (an earlier phase's "done" is not this phase's end).
# It never accepts an unverified checkpoint: a fetch that fails verification is exit 4, not a marked tag.
# Run in the background (Bash run_in_background); its exit is the notification. Usage: watch_v5.sh <exec tag, e.g. GM> <phase: anchor|design|units>
export MSYS_NO_PATHCONV=1 PYTHONIOENCODING=utf-8
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE" || exit 9
PY="${V5_LAPTOP_PY:-../venv/Scripts/python.exe}"
TAG="${1:?usage: watch_v5.sh <exec tag> <phase>}"
PHASE="${2:?usage: watch_v5.sh <exec tag> <phase>}"
POLL="${V5_WATCH_POLL:-300}"
LRUNS="${V5_LOCAL_RUNS:-v5_runs}"
R() { "$PY" tools/v5_remote.py "$@"; }
miss=0
wait_start=0
gone_confirm=0
while true; do
  s=$(R --tag "$TAG" exec -- 'cat "$V5_RUNS/driver/status.json" 2>/dev/null; echo; ls "$V5_RUNS/driver" 2>/dev/null | grep checkpointed; P=$(cat "$V5_RUNS/driver/driver.lockdir/pid" 2>/dev/null || echo 999999); kill -0 $P 2>/dev/null && echo DRIVER_ALIVE || echo DRIVER_GONE' 2>/dev/null)
  if [ -z "$s" ]; then miss=$((miss+1)); echo "$(date -u +%TZ) poll failed ($miss)"; [ $miss -ge 3 ] && { echo "WATCH UNREACHABLE: pod not reachable for 3 polls"; exit 5; }; sleep "$POLL"; continue; fi
  miss=0
  # V5-D3: compare the pod's checkpoint list and hashes with the laptop's and fetch every tag whose newest pod checkpoint differs (never the
  # 'checkpointed' flag alone: U1's second checkpoint was skipped that way, V5-A10)
  R --tag "$TAG" sync || { echo "WATCH FETCH FAILED (sync)"; exit 4; }
  state=$(echo "$s" | head -1 | "$PY" -c "import json,sys; d=json.loads(sys.stdin.readline() or '{}'); print(d.get('state','none') if d.get('phase')=='$PHASE' else 'other-phase')" 2>/dev/null)
  if [ "$state" = "other-phase" ] || [ "$state" = "none" ]; then
    wait_start=$((wait_start+1)); echo "$(date -u +%TZ) the $PHASE driver has not started yet ($wait_start)"
    [ $wait_start -ge 3 ] && { echo "WATCH NOT STARTED: no $PHASE driver after 3 polls: $(echo "$s" | head -1)"; exit 6; }
    sleep "$POLL"; continue
  fi
  case "$state" in
    done)    echo "WATCH PHASE DONE: $(echo "$s" | head -1)"; exit 0 ;;
    failed)  echo "WATCH DRIVER FAILED: $(echo "$s" | head -1)"; exit 1 ;;
    stopped) echo "WATCH PROTOCOL STOP: $(echo "$s" | head -1)"; exit 3 ;;
  esac
  if echo "$s" | grep -q DRIVER_GONE; then                 # r4: confirmed V5_WATCH_CONFIRM (30) s later (a restarted driver may not have rewritten status.json yet)
    if [ "$gone_confirm" = 1 ]; then echo "WATCH DRIVER GONE while state=$state: $(echo "$s" | head -1)"; exit 2; fi
    gone_confirm=1; sleep "${V5_WATCH_CONFIRM:-30}"; continue
  fi
  gone_confirm=0
  sleep "$POLL"
done
