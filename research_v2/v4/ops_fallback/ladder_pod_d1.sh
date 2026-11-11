#!/usr/bin/env bash
# FALLBACK ladder driver under deviation D1, POD side (not in the repo; copied to /workspace/ladder_pod_d1.sh).
# Same steps and skip rules as the laptop driver ladder_d1.sh, run directly on the pod so an SSH drop cannot kill it.
# Differences: no fetch (the laptop watcher fetches each checkpoint), and a marker per checkpointed model.
# It never calls RunPod (no runpodctl, no API), never writes an authorization file, and cannot stop the pod:
# stopping stays with the laptop side (MCP, approved by the author).
# Launch (laptop):  X GM "setsid -f nohup bash /workspace/ladder_pod_d1.sh > /workspace/v4_runs/ladder_pod_d1.log 2>&1 < /dev/null"
set -o pipefail
cd /workspace/repo || exit 9
[ -f /workspace/venv/bin/activate ] && . /workspace/venv/bin/activate
export V4_RUNS=/workspace/v4_runs HF_HOME=/workspace/hf V4_POD=1 HF_HUB_DISABLE_TELEMETRY=1 PYTHONUNBUFFERED=1
RUNS=$V4_RUNS
LOCKF=/workspace/repo/research_v2/V4_PROTOCOL_LOCK.txt
ST=$RUNS/ladder_pod
mkdir -p "$ST"
# one driver at a time: refuse if another holds the lock or any run_v4 GPU command is already running
exec 9> "$ST/driver.lock"
flock -n 9 || { echo "REFUSED: another on-pod driver holds $ST/driver.lock"; exit 7; }
if pgrep -f "run_v4.py (gates|calibrate|test|checkpoint|download|gm)" > /dev/null; then echo "REFUSED: a run_v4 command is already running"; pgrep -af run_v4.py; exit 7; fi
echo $$ > "$ST/driver.pid"
status() { printf '{"state": "%s", "step": "%s", "utc": "%s", "pid": %s}\n' "$1" "$2" "$(date -u +%FT%TZ)" "$$" > "$ST/status.json"; }
run() { local name="$1"; shift; echo "=== $(date -u +%FT%TZ) START $name"; status running "$name"; "$@"; local rc=$?; echo "=== $(date -u +%FT%TZ) END $name exit $rc"
        if [ $rc -ne 0 ]; then echo "LADDER FAILED AT $name (exit $rc)"; status failed "$name"; exit 1; fi; }
echo "=== $(date -u +%FT%TZ) on-pod driver start, pid $$, HEAD $(git rev-parse --short HEAD)"
python research_v2/tools/lock_v4.py --verify | tail -1 | grep -q "lock verifies" || { echo "LADDER FAILED AT lock verify"; status failed verify; exit 1; }

for M in qwen2.5-1.5b llama3.2-3b qwen2.5-3b qwen2.5-7b llama3.1-8b mistral-7b-v0.3; do
  if [ -f $RUNS/$M/test_complete.json ] && ls $RUNS/checkpoints/${M}_*.tar > /dev/null 2>&1; then echo "=== SKIP $M (test complete, checkpoint exists)"; touch "$ST/$M.checkpointed"; continue; fi
  if [ ! -f $RUNS/$M/test_complete.json ]; then
    if [ -f $RUNS/$M/download.json ]; then
      echo "=== SKIP download $M (download.json exists)"
    else
      echo "=== $(date -u +%FT%TZ) START download $M"; status running "download $M"
      python research_v2/v4/run_v4.py download --model $M --authorized-gpu-run; rc=$?
      echo "=== $(date -u +%FT%TZ) END download $M exit $rc"
      if [ $rc -ne 0 ]; then
        if [ $M = mistral-7b-v0.3 ] && [ -f $RUNS/$M/not_run.json ]; then echo "MISTRAL DROPPED (locked rule, not_run.json)"; break; fi
        echo "LADDER FAILED AT download $M (exit $rc)"; status failed "download $M"; exit 1
      fi
    fi
    if python -c "import json,sys; sys.exit(0 if json.load(open('$RUNS/$M/gates/gates.json')).get('all_passed') else 1)" 2>/dev/null; then
      echo "=== SKIP gates $M (passed)"
    else
      run "gates $M" python research_v2/v4/run_v4.py gates --model $M --authorized-gpu-run
    fi
    if [ -f $RUNS/$M/precision.json ] && ! grep -q "^FLAG $M:" $LOCKF; then
      echo "D6: G3 failed in FP16 for $M, BF16 fallback"
      run "append-flag $M" python research_v2/tools/lock_v4.py --append-flag $M
    fi
    if grep -q "^CALIBRATION $M:" $LOCKF; then
      echo "=== SKIP calibration $M (appended)"
    else
      run "calibrate $M"          python research_v2/v4/run_v4.py calibrate --model $M --authorized-gpu-run
      run "analyze calibrate $M"  python research_v2/v4/analyze_v4.py calibrate --model $M
      run "append-calibration $M" python research_v2/tools/lock_v4.py --append-calibration $M
    fi
    run "test $M"               python research_v2/v4/run_v4.py test --model $M --authorized-gpu-run
  fi
  run "checkpoint $M"         python research_v2/v4/run_v4.py checkpoint --model $M --authorized-gpu-run
  touch "$ST/$M.checkpointed"
  run "ledger after $M"       python research_v2/v4/run_v4.py ledger
done
echo "LADDER DONE"; status done ladder
