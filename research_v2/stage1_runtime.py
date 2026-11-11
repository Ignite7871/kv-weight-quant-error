"""Amendment A2 runtime layer for the Stage 1 main run (STAGE1_SPEC.md section 10, A2).

A2 changes ONLY the runtime projection and stopping procedure plus monitoring. It does not touch experimental policies,
sampling, outcomes, hypotheses or the analysis. Everything here is pure or injectable so it can be unit-tested on CPU.

  * pooled pace statistics: the projected completion uses ALL completed problems across ALL sessions; the current-session
    and last-20 rates are recorded separately so a sustained slowdown stays visible (it is reported, never acted on);
  * the ONLY stopping rule is the absolute cumulative ceiling (12.0 h, counting every earlier session): the run stops before
    any problem whose worst-case duration could cross it, and a watchdog finalizes the ledger and exits before the ceiling
    even if something hangs. A run cut short by the ceiling is TRUNCATED, never extended;
  * per-problem integrity checks (token = argmax of its declared source, branch labels, KV letters, JS range, flip-oracle
    consistency, identical-history outcome identity, first-token identity, memory) stop the run immediately on any violation;
  * GPU telemetry every 10 problems with a hardware-safety stop on over-temperature;
  * checkpoint verification: previously completed records must be byte-identical and the order must be the seeded run order
    with no repeats or skips.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import threading
import time
from pathlib import Path

ARMS = ["fp16", "nf4", "cache_only", "token_only", "oracle_flip"]
HARD_CAP_HOURS = 12.0
WORST_FACTOR = 1.25          # worst-case problem duration = 1.25 x the longest problem seen so far (all sessions)
HARD_MARGIN_S = 60.0         # reserved for the ledger write and process exit
WATCHDOG_MARGIN_S = 20.0     # the watchdog fires this long before the ceiling
TEMP_STOP_C = 88.0           # hardware safety: stop at or above this GPU temperature
MAX_MEM_GB = 6.0             # an 8 GB card; the run uses about 1.4 GB
SLOWDOWN_RATIO = 1.15        # last-20-problem rate / pooled rate above which a sustained slowdown is flagged (reporting only)
DEFAULT_OVERHEAD = 1.03      # process seconds per problem / sum of arm wall seconds, until measured


# ----------------------------------------------------------------------------- pace statistics (reporting only)
def per_problem_totals(records_by_arm: dict, arms=ARMS) -> dict:
    """Sum of the arms' wall seconds per problem, for problems complete on every arm."""
    idx = set.intersection(*[set(records_by_arm[a]) for a in arms]) if all(records_by_arm.get(a) for a in arms) else set()
    return {i: sum(records_by_arm[a][i]["wall_seconds"] for a in arms) for i in idx}


def pace_stats(totals_all, totals_session, overhead, used_s, remaining, cap_s):
    """Pooled (all sessions) and recent (this session, last 20) pace and the cumulative-hours projections they imply.
    Reporting only: no stopping rule reads the projections (A2)."""
    pooled = (sum(totals_all) / len(totals_all)) * overhead if totals_all else None
    sess = (sum(totals_session) / len(totals_session)) * overhead if totals_session else None
    last20 = (sum(totals_session[-20:]) / len(totals_session[-20:])) * overhead if totals_session else None
    out = {"pooled_s_per_problem": pooled, "session_s_per_problem": sess, "last20_s_per_problem": last20,
           "n_pooled": len(totals_all), "n_session": len(totals_session), "remaining": remaining,
           "used_h": used_s / 3600, "cap_h": cap_s / 3600}
    if pooled is not None:
        out["projected_cumulative_h_pooled"] = (used_s + pooled * remaining) / 3600
    if sess is not None:
        out["projected_cumulative_h_session"] = (used_s + sess * remaining) / 3600
        out["projected_cumulative_h_last20"] = (used_s + last20 * remaining) / 3600
    out["sustained_slowdown"] = bool(pooled and len(totals_session) >= 20 and last20 / pooled > SLOWDOWN_RATIO)
    out["problems_still_fitting_under_ceiling_at_pooled_rate"] = (
        int(max(0.0, cap_s - HARD_MARGIN_S - used_s) // pooled) if pooled else None)
    return out


# ----------------------------------------------------------------------------- the only stopping rule: the absolute ceiling
def worst_problem_seconds(totals_all, overhead, factor=WORST_FACTOR):
    return factor * max(totals_all) * overhead if totals_all else 0.0


def stop_decision(used_s, worst_problem_s, cap_s, margin_s=HARD_MARGIN_S):
    """A2: stop (return 'cap') iff starting one more problem could cross the absolute ceiling, i.e. iff
    used + worst-case problem duration + margin > cap. No projection-based stop exists."""
    return "cap" if used_s + worst_problem_s + margin_s > cap_s else None


class Watchdog(threading.Thread):
    """Calls on_expire() once the wall clock reaches `deadline_epoch`, unless stopped. Used so that the ceiling holds even if a
    generation hangs: on_expire finalizes the ledger and terminates the process."""

    def __init__(self, deadline_epoch, on_expire, poll_s=1.0, on_tick=None, tick_s=15.0):
        super().__init__(daemon=True)
        self.deadline, self.on_expire, self.poll = deadline_epoch, on_expire, poll_s
        self.on_tick, self.tick_s, self._last_tick = on_tick, tick_s, 0.0
        self._stop_evt = threading.Event()
        self.fired = False

    def run(self):
        while not self._stop_evt.wait(self.poll):
            now = time.time()
            if now >= self.deadline:
                self.fired = True
                self.on_expire()
                return
            if self.on_tick and now - self._last_tick >= self.tick_s:
                self._last_tick = now
                try:
                    self.on_tick()
                except Exception:  # noqa: BLE001  (a heartbeat failure must never kill the watchdog)
                    pass

    def stop(self):
        self._stop_evt.set()


# ----------------------------------------------------------------------------- hardware telemetry
TELEMETRY_QUERY = "temperature.gpu,clocks.sm,clocks.mem,power.draw,pstate,utilization.gpu,clocks_event_reasons.active"


def parse_telemetry(line: str) -> dict:
    keys = ["temp_c", "sm_mhz", "mem_mhz", "power_w", "pstate", "util_pct", "throttle_reasons"]
    parts = [p.strip() for p in line.strip().split(",")]
    d = dict(zip(keys, parts))
    for k in ("temp_c", "sm_mhz", "mem_mhz", "power_w", "util_pct"):
        try:
            d[k] = float(d[k])
        except (KeyError, ValueError):
            d[k] = None
    return d


def telemetry_now() -> dict:
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={TELEMETRY_QUERY}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
        return parse_telemetry(out)
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def hardware_stop(tele: dict, limit_c=TEMP_STOP_C):
    t = tele.get("temp_c") if isinstance(tele, dict) else None
    return f"GPU temperature {t:.0f} C >= {limit_c:.0f} C" if t is not None and t >= limit_c else None


# ----------------------------------------------------------------------------- per-problem integrity
def first_divergence(a, b):
    n = min(len(a), len(b))
    for k in range(n):
        if a[k] != b[k]:
            return k
    return None if len(a) == len(b) else n


def problem_integrity_errors(R: dict, max_mem_gb=MAX_MEM_GB) -> list:
    """Correctness / cache-consistency / data-integrity invariants for one problem (R: arm -> record). An empty list
    means all pass. Any entry stops the run (A2 item 7). None of these reads outcomes for any purpose other than the
    identical-history identity check, which compares statuses only for equality."""
    errs = []
    for a in ARMS:
        if a not in R:
            return [f"missing record for arm {a}"]
    i = R["fp16"]["index"]
    ref = R["fp16"]["ids"]
    for a in ARMS:
        r = R[a]
        if r["index"] != i:
            errs.append(f"{a}: index {r['index']} != {i}")
        if r["status"] not in ("SOLVED", "WRONG", "EXTRACTION_FAILED"):
            errs.append(f"{a}: unexpected status {r['status']!r}")
        if not (1 <= len(r["ids"]) <= 256):
            errs.append(f"{a}: {len(r['ids'])} emitted tokens")
        if r["steps"] != len(r["ids"]) - 1 or len(r["routes"]) != r["steps"]:
            errs.append(f"{a}: steps/ids/routes inconsistent")
        if (r.get("peak_mem_allocated") or 0) > max_mem_gb * 1e9:
            errs.append(f"{a}: peak memory {r['peak_mem_allocated'] / 1e9:.2f} GB")
        if r["ids"][0] != ref[0]:
            errs.append(f"{a}: first token differs from the FP16 prefill token")
        if r.get("truncated_extraction_identical") is not True:
            errs.append(f"{a}: truncated-text extraction differs from full-text extraction")
        if not (r["wall_seconds"] > 0 and math.isfinite(r["wall_seconds"])):
            errs.append(f"{a}: bad wall time")
    for arm, tk, kv in (("cache_only", "F", "N"), ("token_only", "N", "F")):
        r = R[arm]
        ids, tF, tN, js = r["ids"], r["tF"], r["tN"], r["js"]
        if not (len(tF) == len(tN) == len(js) == r["steps"]):
            errs.append(f"{arm}: trace lengths inconsistent")
            continue
        if r.get("token_source") != tk or r.get("kv_source") != kv:
            errs.append(f"{arm}: wrong declared sources {r.get('token_source')}/{r.get('kv_source')}")
        if r["steps"] and set(r["routes"]) != {kv}:
            errs.append(f"{arm}: KV letters {set(r['routes'])} != {{{kv}}}")
        for k in range(r["steps"]):
            if ids[k + 1] != (tF[k] if tk == "F" else tN[k]):
                errs.append(f"{arm}: step {k + 1} token {ids[k + 1]} != argmax of its declared {tk} source")
                break
        if r["steps"] and (tF[0] != ref[1] or tN[0] != R["nf4"]["ids"][1]):
            errs.append(f"{arm}: step-1 branch tokens disagree with the static arms")
        if any((x != x) or not (-1e-6 <= x <= math.log(2) + 1e-6) for x in js):
            errs.append(f"{arm}: JS divergence NaN or out of range")
    fo, co = R["oracle_flip"], R["cache_only"]
    j = fo["routes"].find("F")
    if j >= 0 and fo["ids"][: j + 2] != co["ids"][: j + 2]:
        errs.append("oracle_flip differs from cache_only before its first flip step")
    for a in ("nf4", "cache_only", "token_only", "oracle_flip"):
        d = first_divergence(R[a]["ids"], ref)
        if d is None and R[a]["status"] != R["fp16"]["status"]:
            errs.append(f"{a}: identical emitted history to FP16 but a different outcome")
    return errs


# ----------------------------------------------------------------------------- checkpoint verification
def verify_checkpoint(eval_dir: Path, snapshot_path: Path, run_order: list, arms=ARMS) -> dict:
    """(1) The first n_lines records of every arm file are byte-identical to the snapshot; (2) every arm's index sequence is
    exactly the seeded run order (no repeats, no skips) and all arms have the same completed set."""
    snap = json.loads(snapshot_path.read_text()) if snapshot_path else None
    rep = {"ok": True, "problems": []}
    seqs = {}
    for a in arms:
        raw = (eval_dir / f"{a}.jsonl").read_bytes()
        lines = raw.split(b"\n")
        if lines and lines[-1] == b"":
            lines = lines[:-1]
        seqs[a] = [json.loads(l)["index"] for l in lines]
        if snap:
            n0 = snap["arms"][a]["n_lines"]
            if len(lines) < n0 or [hashlib.sha256(l).hexdigest() for l in lines[:n0]] != snap["arms"][a]["line_sha256"]:
                rep["ok"] = False
                rep["problems"].append(f"{a}: first {n0} records differ from the snapshot")
        if seqs[a] != run_order[: len(seqs[a])]:
            rep["ok"] = False
            rep["problems"].append(f"{a}: index sequence is not the seeded run-order prefix (repeat/skip/reorder)")
        if len(set(seqs[a])) != len(seqs[a]):
            rep["ok"] = False
            rep["problems"].append(f"{a}: duplicate problem indices")
    sets = [set(v) for v in seqs.values()]
    rep["lengths"] = {a: len(v) for a, v in seqs.items()}
    # a single in-flight problem may be missing from some arms after an unclean exit; anything larger is an error
    n_min, n_max = min(rep["lengths"].values()), max(rep["lengths"].values())
    if n_max - n_min > 1:
        rep["ok"] = False
        rep["problems"].append("arm files differ in length by more than one problem")
    rep["n_complete_all_arms"] = n_min
    return rep


def arm_totals(records_by_arm: dict, idxs=None, arms=ARMS) -> dict:
    """Per arm: (sum of wall seconds, sum of decode steps) over the given problems (default: all the arm's records)."""
    out = {}
    for a in arms:
        recs = records_by_arm[a]
        keys = recs if idxs is None else [i for i in idxs if i in recs]
        out[a] = (sum(recs[i]["wall_seconds"] for i in keys), sum(recs[i]["steps"] for i in keys))
    return out


def append_jsonl(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, default=float) + "\n")
