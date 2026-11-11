#!/usr/bin/env bash
# FALLBACK laptop watcher for the on-pod driver (ladder_pod_d1.sh). Polls every 5 min over short SSH calls:
#  - fetches (and verifies) each model's checkpoint once the pod marks it checkpointed;
#  - EXITS with one WATCH line on every terminal state so the pod can be stopped via MCP right away:
#    driver done, driver failed, driver process gone while "running" (crash), pod unreachable 3 polls in a row.
# Run in the background (Bash run_in_background); its exit is the notification. An SSH drop here does not touch the pod run.
export MSYS_NO_PATHCONV=1 PYTHONIOENCODING=utf-8
cd "D:/Projects/Dynamic Precision Scaling/entropy-dps-project/research_v2" || exit 9
PY=../venv/Scripts/python.exe
R() { $PY tools/v4_remote.py "$@"; }
X() { local t="$1"; shift; R --tag "$t" exec -- "$@"; }
ST=/workspace/v4_runs/ladder_pod
miss=0
while true; do
  s=$(X GM "cat $ST/status.json 2>/dev/null; echo; ls $ST 2>/dev/null | grep checkpointed; kill -0 \$(cat $ST/driver.pid 2>/dev/null || echo 999999) 2>/dev/null && echo DRIVER_ALIVE || echo DRIVER_GONE" 2>/dev/null)
  if [ -z "$s" ]; then miss=$((miss+1)); echo "$(date -u +%TZ) poll failed ($miss)"; [ $miss -ge 3 ] && { echo "WATCH UNREACHABLE: pod not reachable for 3 polls"; exit 3; }; sleep 300; continue; fi
  miss=0
  for m in $(echo "$s" | sed -n 's/\.checkpointed$//p'); do
    if ! $PY -c "import json,sys; d=json.load(open('v4_runs/status.json')); sys.exit(0 if d.get('$m',{}).get('checkpointed') else 1)" 2>/dev/null; then
      echo "$(date -u +%TZ) fetching $m"; R --tag $m fetch || { echo "WATCH FETCH FAILED $m"; exit 4; }
    fi
  done
  state=$(echo "$s" | head -1 | $PY -c "import json,sys; print(json.loads(sys.stdin.readline() or '{}').get('state','none'))" 2>/dev/null)
  case "$state" in
    done)   echo "WATCH LADDER DONE"; exit 0 ;;
    failed) echo "WATCH LADDER FAILED: $(echo "$s" | head -1)"; exit 1 ;;
  esac
  if echo "$s" | grep -q DRIVER_GONE; then echo "WATCH DRIVER GONE while state=$state: $(echo "$s" | head -1)"; exit 2; fi
  sleep 300
done
