"""Operational helper for the v6 test phase (not a locked file; no effect on any computed value).

Windows 11 power-throttles (EcoQoS) windowless background processes, which slowed the test driver ~2.5x on 2026-11-15.
Every 30 s this opts every running python*.exe process out of execution-speed throttling (PROCESS_POWER_THROTTLING_EXECUTION_SPEED
in the control mask, cleared in the state mask). It exits when v6_runs/test_complete.json exists or when no launcher has
heartbeat for 15 min. Log: v6_runs/unthrottle.log (one line per newly unthrottled pid).
"""
from __future__ import annotations

import ctypes
import time
from ctypes import wintypes as W
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "v6_runs"
LOG = OUT / "unthrottle.log"
k = ctypes.WinDLL("kernel32", use_last_error=True)
k.OpenProcess.restype = W.HANDLE
k.CreateToolhelp32Snapshot.restype = W.HANDLE


class PPTS(ctypes.Structure):
    _fields_ = [("Version", W.ULONG), ("ControlMask", W.ULONG), ("StateMask", W.ULONG)]


class PE32(ctypes.Structure):
    _fields_ = [("dwSize", W.DWORD), ("cntUsage", W.DWORD), ("th32ProcessID", W.DWORD), ("th32DefaultHeapID", ctypes.c_size_t),
                ("th32ModuleID", W.DWORD), ("cntThreads", W.DWORD), ("th32ParentProcessID", W.DWORD), ("pcPriClassBase", ctypes.c_long),
                ("dwFlags", W.DWORD), ("szExeFile", ctypes.c_wchar * 260)]


def procs():
    """[(pid, parent pid, exe name)] of all processes."""
    snap = k.CreateToolhelp32Snapshot(0x2, 0)
    e = PE32()
    e.dwSize = ctypes.sizeof(e)
    out = []
    ok = k.Process32FirstW(snap, ctypes.byref(e))
    while ok:
        out.append((e.th32ProcessID, e.th32ParentProcessID, e.szExeFile))
        ok = k.Process32NextW(snap, ctypes.byref(e))
    k.CloseHandle(snap)
    return out


def pythons():
    return [(p, n) for p, _, n in procs() if n.lower().startswith("python")]


def descendants(root):
    """pids of all descendants of root (children, grandchildren, ...)."""
    kids, out, todo = {}, [], [root]
    for p, pp, _ in procs():
        kids.setdefault(pp, []).append(p)
    while todo:
        for c in kids.get(todo.pop(), []):
            if c not in out and c != root:
                out.append(c)
                todo.append(c)
    return out


def unthrottle(pid):
    h = k.OpenProcess(0x1000 | 0x0200, False, pid)
    if not h:
        return False
    s = PPTS(1, 0x1, 0)
    ok = bool(k.SetProcessInformation(h, 4, ctypes.byref(s), ctypes.sizeof(s)))
    k.CloseHandle(h)
    return ok


def query(pid):
    """(ControlMask, StateMask) of the process's power-throttling setting; (1, 0) = opted out of execution-speed throttling."""
    h = k.OpenProcess(0x1000, False, pid)
    if not h:
        return None
    s = PPTS(1, 0, 0)
    ok = k.GetProcessInformation(h, 4, ctypes.byref(s), ctypes.sizeof(s))
    k.CloseHandle(h)
    return (s.ControlMask, s.StateMask) if ok else None


def main():
    seen = set()
    while True:
        if (OUT / "test_complete.json").exists():
            break
        hb = OUT / "heartbeat.json"
        if hb.exists() and time.time() - hb.stat().st_mtime > 900:
            break
        for pid, name in pythons():
            if pid not in seen and unthrottle(pid):
                seen.add(pid)
                with open(LOG, "a", encoding="utf-8") as f:
                    f.write(f"[{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}] unthrottled {pid} {name}\n")
        time.sleep(30)


if __name__ == "__main__":
    main()
