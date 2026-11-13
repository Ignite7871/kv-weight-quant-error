#!/usr/bin/env bash
# PROTOCOL_V5 section 10: the on-pod driver (v4's prepared fallback, made primary). It runs one phase of the fixed order on the pod itself, so an SSH
# drop or a sleeping laptop does not stop the run. It never calls RunPod, never writes an authorization file and cannot stop the pod: stopping stays
# with the laptop side (RunPod MCP, approved by the corresponding author). Every protocol stop (gate, drift, ledger stop at $14, resume check, unit
# start check, design R4) ends the phase with state "stopped" or "failed"; nothing is retried.
#
#   phase anchor : GM (+ append) -> U0 (download, gates, calibration + append, test, checkpoint)            -> state "done"; STOP (anchor on the laptop)
#   phase design : Mistral download -> probes qwen-nf4, qwen-hqq, mistral-nf4, mistral-hqq -> design (+ append) -> checkpoint DESIGN -> "done"; STOP
#   phase units  : U1 -> U2 -> U3 -> U4 -> U5 -> U6, each: download, gates (+ FLAG / HQQ append), calibration (+ append), test, checkpoint -> "done"
#   phase u1d3   : V5-D3 only (U1 re-executed on a new volume after its completed checkpoint was lost): DEVIATION D3 must be appended; GM for D3
#                  (+ append); U1 download, gates (a BF16 flag stops: the first run had none), calibration, d3-calibration-check (the recomputed
#                  calibration must equal the locked CALIBRATION U1 hash, else stop), test with the bitwise check of the 13 surviving first-run
#                  problems (a difference stops), checkpoint -> "done"
#
# Launch on the pod (laptop):  X <tag> "setsid -f nohup bash research_v2/v5/run_v5_pod.sh <phase> > /workspace/v5_runs/driver_<phase>.log 2>&1 < /dev/null"
# Files: $V5_RUNS/driver/{status.json, driver.lockdir/pid, <tag>.checkpointed}. Environment: V5_RUNS (set by tools/v5_remote.py exec).
set -o pipefail
PHASE="${1:?usage: run_v5_pod.sh anchor|design|units|u1d3}"
RUNS="${V5_RUNS:?V5_RUNS is not set}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO" || exit 9
PY="${V5_PY:-python}"
LOCKF="$REPO/research_v2/V5_PROTOCOL_LOCK.txt"
ST="$RUNS/driver"
mkdir -p "$ST"
status() { printf '{"state": "%s", "step": "%s", "phase": "%s", "utc": "%s", "pid": %s}\n' "$1" "$2" "$PHASE" "$(date -u +%FT%TZ)" "$$" > "$ST/status.json.tmp" \
           && mv -f "$ST/status.json.tmp" "$ST/status.json"; }
# one driver at a time: an atomic lock directory with the owner's pid; a dead owner's lock is taken over (logged)
if ! mkdir "$ST/driver.lockdir" 2>/dev/null; then
  OWNER="$(cat "$ST/driver.lockdir/pid" 2>/dev/null)"
  if [ -n "$OWNER" ] && kill -0 "$OWNER" 2>/dev/null; then echo "REFUSED: another on-pod driver is running (pid $OWNER)"; exit 7; fi
  echo "=== $(date -u +%FT%TZ) stale driver lock (pid ${OWNER:-none}) taken over"
fi
echo $$ > "$ST/driver.lockdir/pid"
trap 'rm -rf "$ST/driver.lockdir"' EXIT
status starting lock                                         # r4: this phase owns status.json from now on
# refuse while any study GPU process holds v5_runs/gpu_process.lock
gpu_busy() { "$PY" - "$RUNS" <<'EOF'
import json, sys
from pathlib import Path
sys.path.insert(0, "research_v2/v5")
import v5_ops as O
p = Path(sys.argv[1]) / "gpu_process.lock"
try:
    h = json.loads(p.read_text(encoding="utf-8"))
except (OSError, ValueError):
    sys.exit(1)
sys.exit(0 if O.pid_alive(h.get("pid", -1)) else 1)
EOF
}
if gpu_busy; then echo "REFUSED: a study GPU process is running (v5_runs/gpu_process.lock)"; exit 7; fi
R() { "$PY" research_v2/v5/run_v5.py "$@"; }
run() { local name="$1"; shift; echo "=== $(date -u +%FT%TZ) START $name"; status running "$name"; "$@"; local rc=$?
        echo "=== $(date -u +%FT%TZ) END $name exit $rc"; return $rc; }
fail() { echo "DRIVER FAILED AT $1"; status failed "$1"; exit 1; }
stopped() { echo "DRIVER STOPPED: $1"; status stopped "$1"; exit 3; }
locked() { grep -q "^$1" "$LOCKF"; }
json_true() { "$PY" -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get(sys.argv[2]) else 1)" "$1" "$2" 2>/dev/null; }

echo "=== $(date -u +%FT%TZ) on-pod driver phase $PHASE, pid $$, HEAD $(git rev-parse --short HEAD 2>/dev/null)"
"$PY" research_v2/tools/lock_v5.py --verify | tail -1 | grep -q "lock verifies" || fail "lock verify"

checkpoint() { run "checkpoint $1" R checkpoint --tag "$1" || fail "checkpoint $1"; touch "$ST/$1.checkpointed"; }

run_unit() {
  local U="$1"
  if [ -f "$RUNS/$U/not_run.json" ]; then echo "=== SKIP $U (not run: $(cat "$RUNS/$U/not_run.json"))"; return 0; fi
  if [ -f "$RUNS/$U/test_complete.json" ] && [ -f "$ST/$U.checkpointed" ]; then echo "=== SKIP $U (complete, checkpointed)"; return 0; fi
  if [ ! -f "$RUNS/$U/test_complete.json" ]; then
    if [ -f "$RUNS/$U/download.json" ]; then echo "=== SKIP download $U"; else
      run "download $U" R download --tag "$U" --authorized-gpu-run
      if [ $? -ne 0 ]; then [ -f "$RUNS/$U/not_run.json" ] && { checkpoint "$U"; stopped "unit start check at $U (study closes out)"; }; fail "download $U"; fi
    fi
    if json_true "$RUNS/$U/gates/gates.json" all_passed; then echo "=== SKIP gates $U (passed)"; else
      run "gates $U" R gates --tag "$U" --authorized-gpu-run
      if [ $? -ne 0 ]; then [ -f "$RUNS/$U/not_run.json" ] && { checkpoint "$U"; stopped "unit start check at $U (study closes out)"; }; checkpoint "$U"; fail "gates $U"; fi
    fi
    if [ -f "$RUNS/$U/precision.json" ] && ! locked "FLAG $U:"; then run "append-flag $U" "$PY" research_v2/tools/lock_v5.py --append-flag "$U" || fail "append-flag $U"; fi
    if [ -f "$RUNS/$U/hqq.json" ] && ! locked "HQQ $U:"; then run "append-hqq $U" "$PY" research_v2/tools/lock_v5.py --append-hqq "$U" || fail "append-hqq $U"; fi
    if locked "CALIBRATION $U:"; then echo "=== SKIP calibration $U (appended)"; else
      run "calibrate $U" R calibrate --tag "$U" --authorized-gpu-run || { checkpoint "$U"; fail "calibrate $U"; }
      if [ ! -f "$RUNS/$U/calibration.json" ]; then run "analyze calibrate $U" "$PY" research_v2/v5/analyze_v5.py calibrate --unit "$U" || fail "analyze calibrate $U"; fi
      run "append-calibration $U" "$PY" research_v2/tools/lock_v5.py --append-calibration "$U" || fail "append-calibration $U"
    fi
    run "test $U" R test --tag "$U" --authorized-gpu-run || { checkpoint "$U"; fail "test $U"; }
  fi
  checkpoint "$U"
  run "ledger after $U" R ledger
}

case "$PHASE" in
  anchor)
    if locked "GM:"; then echo "=== SKIP GM (appended)"; else
      run "gm" R gm --authorized-gpu-run || fail "gm"
      run "append-gm" "$PY" research_v2/tools/lock_v5.py --append-gm || fail "append-gm"
    fi
    run_unit U0 ;;
  design)
    [ -f "$RUNS/_downloads/mistral-7b-v0.3.json" ] || run "download DESIGN" R download --tag DESIGN --authorized-gpu-run || fail "download DESIGN"
    for P in qwen-nf4 qwen-hqq mistral-nf4 mistral-hqq; do
      [ -f "$RUNS/probes/$P.json" ] && { echo "=== SKIP probe $P"; continue; }
      run "probe $P" R probe --probe "$P" --authorized-gpu-run || fail "probe $P"
    done
    if ! locked "DESIGN:"; then
      if [ ! -f "$RUNS/design.json" ]; then
        run "design" R design; rc=$?
        if [ $rc -eq 2 ]; then checkpoint DESIGN; stopped "design R4: the projection does not fit even with R1-R3"; fi
        [ $rc -eq 0 ] || fail "design"
      fi
      run "append-design" "$PY" research_v2/tools/lock_v5.py --append-design || fail "append-design"
    fi
    checkpoint DESIGN ;;
  units)
    for U in U1 U2 U3 U4 U5 U6; do run_unit "$U"; done ;;
  u1d3)
    locked "DEVIATION D3:" || fail "u1d3 needs DEVIATION D3 appended to the lock"
    if locked "GM D3:"; then echo "=== SKIP GM D3 (appended)"; else
      run "gm" R gm --authorized-gpu-run || fail "gm"
      run "append-gm" "$PY" research_v2/tools/lock_v5.py --append-gm || fail "append-gm"
    fi
    U=U1
    if [ -f "$RUNS/$U/test_complete.json" ]; then echo "=== SKIP $U (complete)"; else
      if [ -f "$RUNS/$U/download.json" ]; then echo "=== SKIP download $U"; else
        run "download $U" R download --tag "$U" --authorized-gpu-run || { [ -f "$RUNS/$U/not_run.json" ] && stopped "unit start check at $U"; fail "download $U"; }
      fi
      if json_true "$RUNS/$U/gates/gates.json" all_passed; then echo "=== SKIP gates $U (passed)"; else
        run "gates $U" R gates --tag "$U" --authorized-gpu-run || { checkpoint "$U"; fail "gates $U"; }
      fi
      [ -f "$RUNS/$U/precision.json" ] && { checkpoint "$U"; stopped "D3: the U1 gates set the BF16 flag; the first run had none"; }
      if json_true "$RUNS/$U/d3_calibration_check.json" equal; then echo "=== SKIP calibration $U (reproduced)"; else
        run "calibrate $U" R calibrate --tag "$U" --authorized-gpu-run || { checkpoint "$U"; fail "calibrate $U"; }
        run "d3-calibration-check $U" R d3-calibration-check --tag "$U" --authorized-gpu-run \
          || { checkpoint "$U"; [ -f "$RUNS/$U/d3_calibration_mismatch.json" ] && stopped "D3: the U1 calibration does not reproduce the locked hash"; fail "d3-calibration-check $U"; }
      fi
      run "test $U" R test --tag "$U" --authorized-gpu-run \
        || { checkpoint "$U"; [ -f "$RUNS/$U/d3_reference_failure.json" ] && stopped "D3: a re-executed U1 problem differs from the surviving first-run records"; fail "test $U"; }
    fi
    checkpoint "$U"
    run "ledger after $U" R ledger ;;
  *) echo "unknown phase $PHASE"; exit 2 ;;
esac
echo "PHASE $PHASE DONE"; status done "$PHASE"
