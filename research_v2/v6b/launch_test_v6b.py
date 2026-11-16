"""Detached launcher for the v6b test phase (locked with the protocol; it changes no computed value).

  python research_v2/v6b/launch_test_v6b.py              runs `run_v6b.py test` for qwen2.5-0.5b, llama3.2-1b, qwen2.5-1.5b in that order
  python research_v2/v6b/launch_test_v6b.py --selftest   no GPU: checks that the launcher and its children are opted out of power throttling

Start it so that it survives a closed terminal (FOLLOWUP_V6B s10) the way the v6 test phase finally ran (v6 AI-use log, 2026-11-15:
a WMI start died after ~22 min; a console task died when its console closed): a Windows scheduled task 'v6b_test_launcher' that runs
venv/Scripts/pythonw.exe (no console) on this file, battery allowed, no time limit; the driver child is started with CREATE_NO_WINDOW and
every descendant is opted out of power throttling.

Exactly-once: before each model it checks the record files. A duplicate (model, arm, problem); a problem present in some of NAT-V, A, B, C
but not all; a C1-V or NAT-V-cen record for a problem without the four; or a g1.jsonl that does not list exactly the completed problems
-> STOP (no automatic repair; author decision). Resume = rerun this launcher; the driver skips completed problems.
Ledger: a heartbeat every 60 s; if a previous launcher died mid-problem, the interrupted GPU time is ledgered on restart (as v6).
Cuts (FOLLOWUP_V6B s8): before each model, projected total = ledger + remaining problems x per-problem seconds (rehearsal value x the
observed test/rehearsal ratio of completed models, guard 1.25 before any test model completes). Over 9 h -> cut 1 (rerun only NAT-V after
10 bitwise G1 problems; per-problem cost x 4/6), then n = 50, via v6b_runs/design.json.
Reported while running: integrity counts and GPU-hours only (status_<model>.json); no estimate is computed.
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
sys.path.insert(0, str(RV / "v6"))
import unthrottle_v6 as UT  # noqa: E402  (process helpers only: descendants, unthrottle, query)

OUT = Path(os.environ.get("V6B_RUNS", RV / "v6b_runs"))
PY = RV.parent / "venv" / "Scripts" / "python.exe"
MODELS = ["qwen2.5-0.5b", "llama3.2-1b", "qwen2.5-1.5b"]
REQ = ["NAT-V", "NAT-V-cen0", "C1-V+mu", "C1-V@cen"]
OPT = ["C1-V", "NAT-V-cen"]
FLAGS = ["exact_ok", "rounding_ok", "finite_ok", "routing_reset", "natural_ok", "construction_ok", "b_construction_ok"]
CAP_H, N, GUARD = 9.0, 70, 1.25
CUT1 = "rerun NAT-V only after 10 bitwise G1 problems"


def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ts(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()


def log(msg):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "test_run.log", "a", encoding="utf-8") as f:
        f.write(f"[{utc()}] {msg}\n")


def ledger_lines():
    p = OUT / "gpu_ledger.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def used_h():
    return sum(r["seconds"] for r in ledger_lines()) / 3600


def stop(msg):
    log("STOP: " + msg)
    (OUT / "launcher_stop.json").write_text(json.dumps({"utc": utc(), "reason": msg}), encoding="utf-8")
    sys.exit(msg)


def ids_of(f):
    if not f.exists():
        return []
    return [json.loads(l)["index"] for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]


def scan(m):
    """completed problem ids and failed-flag count; STOP on duplicates, partial problems or a G1 log mismatch."""
    d = OUT / "test" / m
    if not d.exists():
        return set(), 0
    per, bad = {}, 0
    for a in REQ + OPT:
        f = d / f"{a}.jsonl"
        ids = ids_of(f)
        if len(ids) != len(set(ids)):
            stop(f"{m}/{a}: duplicate problem records")
        per[a] = set(ids)
        if f.exists():
            for l in f.read_text(encoding="utf-8").splitlines():
                if l.strip():
                    r = json.loads(l)
                    bad += sum(1 for k in FLAGS if not r.get(k, False))
    union, full = set().union(*[per[a] for a in REQ]), set.intersection(*[per[a] for a in REQ])
    if union - full:
        stop(f"{m}: partial problem(s) {sorted(union - full)}; author decision needed")
    extra = (per["C1-V"] | per["NAT-V-cen"]) - full
    if extra:
        stop(f"{m}: C1-V / NAT-V-cen records for incomplete problems {sorted(extra)}")
    g = ids_of(d / "g1.jsonl")
    if len(g) != len(set(g)) or set(g) != full:
        stop(f"{m}: g1.jsonl does not list exactly the completed problems")
    return full, bad


def recover_interrupted():
    if not (OUT / "heartbeat.json").exists():
        return
    hb = json.loads((OUT / "heartbeat.json").read_text(encoding="utf-8"))
    if not hb.get("running"):
        return
    last = max([ts(r["utc"]) for r in ledger_lines()] + [ts(hb["child_start_utc"])])
    lost = max(0.0, ts(hb["utc"]) - last)
    with open(OUT / "gpu_ledger.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"utc": utc(), "block": f"interrupted:{hb['model']} (heartbeat recovery)", "seconds": lost, "completed": False}) + "\n")
    log(f"recovered interrupted run of {hb['model']}: {lost:.0f} s ledgered")
    (OUT / "heartbeat.json").write_text(json.dumps({**hb, "running": False}), encoding="utf-8")


def reh_seconds():
    s = json.loads((OUT / "rehearsal" / "rehearsal_summary.json").read_text(encoding="utf-8"))
    return {m: s[m]["block_seconds_per_problem"] for m in MODELS}


def project_and_cut(k):
    p = OUT / "design.json"
    des = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"n": N, "cuts": []}
    reh = reh_seconds()
    ratio = []
    for m in MODELS[:k]:
        secs = [r["seconds"] for r in ledger_lines() if r["block"].startswith(f"test:{m}:") and r["completed"]]
        if secs:
            ratio.append((sum(secs) / len(secs)) / reh[m])
    rho = max(ratio) if ratio else GUARD

    def proj(d):
        tot = used_h()
        for m in MODELS[k:]:
            done, _ = scan(m)
            rem = max(0, d["n"] - len(done))
            f = (4 / 6) if CUT1 in d["cuts"] else 1.0
            tot += rem * reh[m] * rho * f / 3600
        return tot
    steps = [(CUT1, lambda d: d), ("n = 50", lambda d: {**d, "n": min(d["n"], 50)})]
    pr = proj(des)
    for name, f in steps:
        if pr <= CAP_H:
            break
        if name in des["cuts"]:
            continue
        des = f(des)
        des["cuts"] = des["cuts"] + [name]
        pr = proj(des)
        log(f"cut applied: {name} (projected {pr:.2f} h, ratio {rho:.2f})")
    if pr > CAP_H:
        stop(f"projection {pr:.2f} h exceeds the cap after all cuts")
    des.update(projected_total_h=pr, utc=utc())
    p.write_text(json.dumps(des, indent=1), encoding="utf-8")
    return des


def guard(root, stop_ev, done):
    while not stop_ev.is_set():
        for p in UT.descendants(root):
            if p not in done and UT.unthrottle(p):
                done.add(p)
                log(f"unthrottled pid {p} {UT.query(p)}")
        stop_ev.wait(5)


def spawn(args, f):
    p = subprocess.Popen(args, cwd=str(RV), stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    UT.unthrottle(p.pid)
    return p


def run_model(m):
    start = utc()
    stop_hb = threading.Event()

    def beat():
        while not stop_hb.is_set():
            (OUT / "heartbeat.json").write_text(json.dumps({"utc": utc(), "model": m, "child_start_utc": start, "running": True}), encoding="utf-8")
            stop_hb.wait(60)
    th = threading.Thread(target=beat, daemon=True)
    th.start()
    with open(OUT / "test_run.log", "a", encoding="utf-8") as f:
        p = spawn([str(PY), str(HERE / "run_v6b.py"), "test", "--model", m], f)
        done = {p.pid}
        tg = threading.Thread(target=guard, args=(os.getpid(), stop_hb, done), daemon=True)
        tg.start()
        rc = p.wait()
    stop_hb.set()
    th.join()
    tg.join()
    (OUT / "heartbeat.json").write_text(json.dumps({"utc": utc(), "model": m, "child_start_utc": start, "running": False}), encoding="utf-8")
    return rc


def keep_awake():
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        return True
    except Exception:  # noqa: BLE001
        return False


def status(m, rc):
    done, bad = scan(m)
    st = {"utc": utc(), "model": m, "exit_code": rc, "problems_completed": len(done), "records_with_a_failed_flag": bad,
          "gpu_h_cumulative": used_h(), "cap_h": CAP_H, "cuts": (json.loads((OUT / "design.json").read_text(encoding="utf-8")).get("cuts", [])
                                                                  if (OUT / "design.json").exists() else [])}
    (OUT / f"status_{m}.json").write_text(json.dumps(st, indent=1), encoding="utf-8")
    return st


def selftest():
    stop_ev = threading.Event()
    with open(OUT / "selftest.log", "w", encoding="utf-8") as f:
        p = spawn([str(PY), "-c", "import time; time.sleep(20)"], f)
        done = {p.pid}
        tg = threading.Thread(target=guard, args=(os.getpid(), stop_ev, done), daemon=True)
        tg.start()
        time.sleep(12)
        res = {pid: UT.query(pid) for pid in [os.getpid()] + UT.descendants(os.getpid())}
        stop_ev.set()
        p.wait()
    ok = len(res) >= 3 and all(v == (1, 0) for v in res.values())
    log(f"selftest unthrottle: {res} -> {'PASS' if ok else 'FAIL'}")
    (OUT / "selftest.json").write_text(json.dumps({"utc": utc(), "states": {str(k): v for k, v in res.items()}, "pass": ok}), encoding="utf-8")
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
        if any((OUT / "test" / x / n).exists() for x in MODELS for n in ("integrity_failure.json", "g1_failure.json")):
            stop("integrity_failure.json or g1_failure.json present")
        des = project_and_cut(k)
        done, _ = scan(m)
        if len(done) < des["n"]:
            log(f"{m}: start ({len(done)} of {des['n']} done)")
            rc = run_model(m)
            if rc != 0:
                st = status(m, rc)
                stop(f"{m}: driver exit code {rc} (integrity, G1 or cap; see test_run.log); {st['problems_completed']} done")
        st = status(m, 0)
        log(f"{m}: complete {st}")
    log("all models complete")
    (OUT / "test_complete.json").write_text(json.dumps({"utc": utc(), "gpu_h": used_h()}), encoding="utf-8")


if __name__ == "__main__":
    main()
