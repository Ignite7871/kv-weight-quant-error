"""Detached launcher for the v6 test phase (operational; not a locked file; it does not change any locked behaviour).

  python research_v2/v6/launch_test_v6.py      runs `run_v6.py test` for qwen2.5-0.5b, llama3.2-1b, qwen2.5-1.5b in that order

Exactly-once: before each model it checks the record files. A duplicate (model, arm, problem), a problem present in some arm files but not
in all, or a complete problem with a missing array file -> STOP (no automatic repair; author decision). Resume = rerun this launcher; the
locked driver skips completed problems.
Ledger: a heartbeat every 60 s; if a previous launcher died mid-problem, the interrupted GPU time (last heartbeat minus the later of the
last ledger line and the child start) is ledgered on restart, so the cap counts it.
Cuts (PROTOCOL_V6 s9): before each model, projected total = ledger + remaining problems x per-problem seconds (rehearsal value x the
observed test/rehearsal ratio of completed models). Over 16 h -> the pre-declared cut order is applied via v6_runs/design.json.
Status after each model: v6_runs/status_<model>.json. Log: v6_runs/test_run.log.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
RV = HERE.parent
sys.path.insert(0, str(HERE))
import unthrottle_v6 as UT  # noqa: E402  (power-throttling opt-out; operational)
OUT = RV / "v6_runs"
PY = RV.parent / "venv" / "Scripts" / "python.exe"
MODELS = ["qwen2.5-0.5b", "llama3.2-1b", "qwen2.5-1.5b"]
REH_S = {"qwen2.5-0.5b": 148.5, "llama3.2-1b": 104.8, "qwen2.5-1.5b": 169.2}      # PROTOCOL_V6 r3 section 9 (measured)
ARMS = ["NAT-K", "NAT-V", "C1-K", "C1-V", "NAT-K-cen", "NAT-V-cen", "ST-K", "ST-V", "ST-K@nat", "ST-V@nat", "ST-V@nat-cen", "ST-Kc", "ST-Kc@nat"]
FLAGS = ["exact_ok", "rounding_ok", "finite_ok", "routing_reset", "natural_ok", "construction_ok"]
CAP_H, N = 16.0, 100
LOG = OUT / "test_run.log"
HB = OUT / "heartbeat.json"


def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ts(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"[{utc()}] {msg}\n")


def ledger_lines():
    p = OUT / "gpu_ledger.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def used_h():
    return sum(r["seconds"] for r in ledger_lines()) / 3600


def design():
    p = OUT / "design.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def stop(msg):
    log("STOP: " + msg)
    (OUT / "launcher_stop.json").write_text(json.dumps({"utc": utc(), "reason": msg}), encoding="utf-8")
    sys.exit(msg)


def scan(m, arms):
    """index sets per arm; raises STOP on duplicates, partial problems or missing arrays."""
    d = OUT / "test" / m
    if not d.exists():
        return set(), 0
    per = {}
    bad_flags = 0
    for a in arms:
        f = d / f"{a}.jsonl"
        ids = []
        if f.exists():
            for l in f.read_text(encoding="utf-8").splitlines():
                if l.strip():
                    r = json.loads(l)
                    ids.append(r["index"])
                    bad_flags += sum(1 for k in FLAGS if not r.get(k, False))
        if len(ids) != len(set(ids)):
            stop(f"{m}/{a}: duplicate problem records")
        per[a] = set(ids)
    union, full = set().union(*per.values()), set.intersection(*per.values())
    if union - full:
        stop(f"{m}: partial problem(s) {sorted(union - full)} (present in some arm files only); author decision needed")
    for i in full:
        for a in arms:
            if not (d / "arrays" / f"{a}_{i}.npz").exists():
                stop(f"{m}: array file missing for {a} problem {i}")
    return full, bad_flags


def recover_interrupted():
    if not HB.exists():
        return
    hb = json.loads(HB.read_text(encoding="utf-8"))
    if not hb.get("running"):
        return
    last = max([ts(r["utc"]) for r in ledger_lines()] + [ts(hb["child_start_utc"])])
    lost = max(0.0, ts(hb["utc"]) - last)
    with open(OUT / "gpu_ledger.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"utc": utc(), "block": f"interrupted:{hb['model']} (heartbeat recovery)", "seconds": lost, "completed": False}) + "\n")
    log(f"recovered interrupted run of {hb['model']}: {lost:.0f} s ledgered")
    HB.write_text(json.dumps({**hb, "running": False}), encoding="utf-8")


def project_and_cut(k):
    """Projected total hours with the current design; apply the cut order until it fits. Returns the design."""
    des = design() or {"n": N, "arms": ARMS, "cuts": []}
    ratio = []
    for m in MODELS[:k]:
        secs = [r["seconds"] for r in ledger_lines() if r["block"].startswith(f"test:{m}:")]
        if secs:
            ratio.append((sum(secs) / len(secs)) / REH_S[m])
    rho = max(ratio) if ratio else 1.0
    def proj(d):
        tot = used_h()
        for m in MODELS[k:]:
            done, _ = scan(m, d["arms"])
            rem = max(0, d["n"] - len(done))
            tot += rem * REH_S[m] * rho * len(d["arms"]) / len(ARMS) / 3600
        return tot
    steps = [("drop ST-Kc, ST-Kc@nat", lambda d: {**d, "arms": [a for a in d["arms"] if a not in ("ST-Kc", "ST-Kc@nat")]}),
             ("n = 70", lambda d: {**d, "n": min(d["n"], 70)}), ("drop NAT-K-cen", lambda d: {**d, "arms": [a for a in d["arms"] if a != "NAT-K-cen"]}),
             ("n = 50", lambda d: {**d, "n": min(d["n"], 50)})]
    p = proj(des)
    applied = list(des.get("cuts", []))
    for name, f in steps:
        if p <= CAP_H:
            break
        if name in applied:
            continue
        des = f(des)
        applied.append(name)
        des["cuts"] = applied
        p = proj(des)
        log(f"cut applied: {name} (projected {p:.2f} h, ratio {rho:.2f})")
    if p > CAP_H:
        stop(f"projection {p:.2f} h exceeds the cap after all cuts")
    des.update(projected_total_h=p, utc=utc())
    (OUT / "design.json").write_text(json.dumps(des, indent=1), encoding="utf-8")
    return des


def guard(root, stop_ev, done):
    """Opt every descendant of `root` (the venv python.exe stub and the real interpreter it starts) out of Windows power
    throttling, every 5 s while the child runs (the 2026-11-15 fault: windowless processes were throttled ~2.5x)."""
    while not stop_ev.is_set():
        for p in UT.descendants(root):
            if p not in done and UT.unthrottle(p):
                done.add(p)
                log(f"unthrottled pid {p} {UT.query(p)}")
        stop_ev.wait(5)


def spawn(args, f):
    p = subprocess.Popen(args, cwd=str(RV), stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))   # no console window to close
    UT.unthrottle(p.pid)
    return p


def run_model(m, des):
    start = utc()
    stop_hb = threading.Event()
    def beat():
        while not stop_hb.is_set():
            HB.write_text(json.dumps({"utc": utc(), "model": m, "child_start_utc": start, "running": True}), encoding="utf-8")
            stop_hb.wait(60)
    th = threading.Thread(target=beat, daemon=True)
    th.start()
    with open(LOG, "a", encoding="utf-8") as f:
        p = spawn([str(PY), str(HERE / "run_v6.py"), "test", "--model", m], f)
        done = {p.pid}
        tg = threading.Thread(target=guard, args=(os.getpid(), stop_hb, done), daemon=True)
        tg.start()
        rc = p.wait()
    stop_hb.set()
    th.join()
    tg.join()
    HB.write_text(json.dumps({"utc": utc(), "model": m, "child_start_utc": start, "running": False}), encoding="utf-8")
    return rc


def keep_awake():
    """While this process runs, ask Windows not to idle-sleep (ES_CONTINUOUS | ES_SYSTEM_REQUIRED); released when the process exits."""
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        return True
    except Exception:  # noqa: BLE001
        return False


def selftest():
    """No GPU: spawn a child through the same venv stub and flags, then check self + every descendant is opted out (1, 0)."""
    stop_ev, res = threading.Event(), {}
    with open(OUT / "selftest.log", "w", encoding="utf-8") as f:
        p = spawn([str(PY), "-c", "import time; time.sleep(20)"], f)
        done = {p.pid}
        tg = threading.Thread(target=guard, args=(os.getpid(), stop_ev, done), daemon=True)
        tg.start()
        time.sleep(12)
        res = {pid: UT.query(pid) for pid in [os.getpid()] + UT.descendants(os.getpid())}
        stop_ev.set()
        p.wait()
    ok = len(res) >= 3 and all(v == (1, 0) for v in res.values())     # self + stub + real interpreter
    log(f"selftest unthrottle: {res} -> {'PASS' if ok else 'FAIL'}")
    (OUT / "selftest.json").write_text(json.dumps({"utc": utc(), "exe": sys.executable, "states": {str(k): v for k, v in res.items()}, "pass": ok}), encoding="utf-8")
    sys.exit(0 if ok else 1)


def main():
    OUT.mkdir(exist_ok=True)
    if not UT.unthrottle(os.getpid()):
        stop("could not opt the launcher out of power throttling")
    if "--selftest" in sys.argv:
        selftest()
    log(f"launcher start (keep-awake {keep_awake()}, unthrottled {UT.query(os.getpid())})")
    recover_interrupted()
    for k, m in enumerate(MODELS):
        for f in ("cap_stop.json",):
            if (OUT / f).exists():
                stop(f"{f} present")
        if any((OUT / "test" / x / "integrity_failure.json").exists() for x in MODELS):
            stop("integrity_failure.json present")
        des = project_and_cut(k)
        done, _ = scan(m, des["arms"])
        if len(done) < des["n"]:
            log(f"{m}: start ({len(done)} of {des['n']} done)")
            rc = run_model(m, des)
            if rc != 0:
                done, bad = scan(m, des["arms"])
                (OUT / f"status_{m}.json").write_text(json.dumps({"utc": utc(), "model": m, "exit_code": rc, "problems_completed": len(done),
                                                                   "records_with_a_failed_flag": bad, "gpu_h_cumulative": used_h(), "cap_h": CAP_H,
                                                                   "cuts": des.get("cuts", [])}, indent=1), encoding="utf-8")
                stop(f"{m}: driver exit code {rc} (integrity failure or cap; see test_run.log)")
        done, bad = scan(m, des["arms"])
        st = {"utc": utc(), "model": m, "exit_code": 0, "problems_completed": len(done), "design_n": des["n"], "arms": len(des["arms"]),
              "records_with_a_failed_flag": bad, "gpu_h_cumulative": used_h(), "cap_h": CAP_H, "cuts": des.get("cuts", [])}
        (OUT / f"status_{m}.json").write_text(json.dumps(st, indent=1), encoding="utf-8")
        log(f"{m}: complete {st}")
    log("all models complete")
    (OUT / "test_complete.json").write_text(json.dumps({"utc": utc(), "gpu_h": used_h()}), encoding="utf-8")


if __name__ == "__main__":
    main()
