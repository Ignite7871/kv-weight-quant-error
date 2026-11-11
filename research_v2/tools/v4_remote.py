"""Laptop-side SSH / scp tooling for the v4 pod (PROTOCOL_V4.md section 11, D8). Claude Code reaches the pod ONLY through this tool, over the pod's
SSH (public IP and exposed TCP port; RunPod's proxied ssh.runpod.io endpoint does not support scp). There is no runpodctl, no RunPod API key and
no billing access here, and the tool refuses commands that would use them or that would write an authorization file.

Every pod action needs the laptop copy of the corresponding author's authorization file, research_v2/v4_runs/AUTHORIZED_V4_<tag>.txt, containing
'PROTOCOL_V4', the lock id and <tag> (the pod's driver checks its own copy on the volume again). All output is passed through the secret
redaction filter.

Connection details: research_v2/v4_runs/pod_ssh.json, written by the corresponding author:
  {"host": "<public IP>", "port": <exposed TCP port for 22>, "user": "root", "identity_file": "C:/Users/<you>/.ssh/<key>"}
Host keys are pinned per pod in research_v2/v4_runs/known_hosts (accept-new on first contact).

  python research_v2/tools/v4_remote.py --tag T check                        ssh echo + GPU name (no other effect)
  python research_v2/tools/v4_remote.py --tag T --commit <sha> push          git bundle of the branch, scp, clone/fetch, detached checkout of <sha>
  python research_v2/tools/v4_remote.py --tag T put <local> <remote>         scp one file to the pod (setup script, step0.json, ...)
  python research_v2/tools/v4_remote.py --tag T exec -- <command ...>        run in /workspace/repo with the venv and V4_RUNS/HF_HOME set
  python research_v2/tools/v4_remote.py --tag T fetch                        newest checkpoint of T: scp, sha256 check, secret scan, extract, status
The options come before the subcommand (r6). From Git Bash, set MSYS_NO_PATHCONV=1, or /workspace/... arguments are rewritten as Windows paths.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

RV = Path(__file__).resolve().parents[1]
ROOT = RV.parent
sys.path.insert(0, str(RV / "v4"))
sys.path.insert(0, str(RV / "tools"))
import v4_ops as O  # noqa: E402
import v4_secret_scan as SS  # noqa: E402

LOCAL_RUNS = RV / "v4_runs"
POD_ROOT = "/workspace"
POD_RUNS = f"{POD_ROOT}/v4_runs"
POD_REPO = f"{POD_ROOT}/repo"
POD_ENV = (f"cd {POD_REPO} && {{ [ ! -f {POD_ROOT}/venv/bin/activate ] || . {POD_ROOT}/venv/bin/activate; }} && export V4_RUNS={POD_RUNS} HF_HOME={POD_ROOT}/hf V4_POD=1 "
           f"HF_HUB_DISABLE_TELEMETRY=1 PYTHONUNBUFFERED=1 && ")
FORBIDDEN = ("runpodctl", "api.runpod.io", "rest.runpod.io", "huggingface-cli login", "hf auth login")
AUTH_WRITERS = (">", "tee", "cp ", "mv ", "touch", "rm ", "sed ", "dd ", "install ", "truncate", "chmod", "ln ")


def lock_id():
    r = subprocess.run([sys.executable, str(RV / "tools" / "lock_v4.py"), "--lock-id"], capture_output=True, **O.TEXT)
    return r.stdout.strip() if r.returncode == 0 else None


def require_auth(tag):
    ok, why = O.check_auth(LOCAL_RUNS, tag, lock_id())
    if not ok:
        sys.exit(f"Refusing: laptop copy of the authorization file: {why}")


def command_allowed(cmd: str):
    low = cmd.lower()
    for f in FORBIDDEN:
        if f in low:
            return False, f"'{f}' is not permitted (D8: no runpodctl, API or billing access; no token login)"
    if "authorized_v4" in low and any(w in low for w in AUTH_WRITERS):
        return False, "commands that could create or change an authorization file are refused"
    return True, "ok"


def conf():
    p = LOCAL_RUNS / "pod_ssh.json"
    if not p.exists():
        sys.exit(f"{p} is missing (written by the corresponding author after starting the pod)")
    c = json.loads(p.read_text(encoding="utf-8"))
    return c


def known_hosts_option():
    """r6: quoted. OpenSSH splits an unquoted UserKnownHostsFile at whitespace, and the repository path has spaces, so before r6 the pin went to
    a stray file named after the first word of the path."""
    return f'UserKnownHostsFile="{(LOCAL_RUNS / "known_hosts").as_posix()}"'


def ssh_base(c, tool="ssh"):
    port_flag = "-p" if tool == "ssh" else "-P"
    return [tool, "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", "-o", known_hosts_option(),
            "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30", port_flag, str(c["port"]), "-i", c["identity_file"]]


def run_stream(argv):
    lits = O.secret_literals()
    p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    for line in p.stdout:
        sys.stdout.write(O.redact(line, lits))
        sys.stdout.flush()
    return p.wait()


def ssh(c, remote_cmd):
    return run_stream(ssh_base(c) + [f"{c['user']}@{c['host']}", remote_cmd])


def scp(c, src, dst):
    return run_stream(ssh_base(c, "scp") + [src, dst])


def remote(c, path):
    return f"{c['user']}@{c['host']}:{path}"


def cmd_push(c, commit):
    branch = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, check=True, **O.TEXT).stdout.strip()
    full = subprocess.run(["git", "-C", str(ROOT), "rev-parse", commit], capture_output=True, check=True, **O.TEXT).stdout.strip()
    with tempfile.TemporaryDirectory() as td:
        b = Path(td) / "v4.bundle"
        subprocess.run(["git", "-C", str(ROOT), "bundle", "create", str(b), branch], check=True)
        if scp(c, str(b), remote(c, f"{POD_ROOT}/v4.bundle")) != 0:
            sys.exit("scp of the bundle failed")
    script = (f"set -e; if [ ! -d {POD_REPO}/.git ]; then git clone -q {POD_ROOT}/v4.bundle {POD_REPO}; fi; "
              f"git -C {POD_REPO} fetch -q {POD_ROOT}/v4.bundle {branch}; git -C {POD_REPO} -c advice.detachedHead=false checkout -q {full}; "
              f"git -C {POD_REPO} rev-parse HEAD")
    return ssh(c, script)


def safe_extract(tar: Path, dest: Path):
    with tarfile.open(tar) as tf:
        for m in tf.getmembers():
            p = (dest / m.name).resolve()
            if not str(p).startswith(str(dest.resolve())) or m.issym() or m.islnk():
                sys.exit(f"refusing to extract {m.name}: unsafe path")
        tf.extractall(dest)


def cmd_fetch(c, tag):
    LOCAL_RUNS.mkdir(parents=True, exist_ok=True)
    listing = subprocess.run(ssh_base(c) + [f"{c['user']}@{c['host']}", f"ls -1 {POD_RUNS}/checkpoints/{tag}_*.tar"], capture_output=True, **O.TEXT)
    tars = sorted(l.strip() for l in listing.stdout.splitlines() if l.strip().endswith(".tar"))
    if not tars:
        sys.exit(f"no checkpoint for {tag} on the pod (run: run_v4.py checkpoint --model ...)")
    src = tars[-1]
    dst = LOCAL_RUNS / "checkpoints"
    dst.mkdir(exist_ok=True)
    for s in (src, src + ".sha256"):
        if scp(c, remote(c, s), str(dst / Path(s).name)) != 0:
            sys.exit(f"scp of {s} failed")
    tar = dst / Path(src).name
    if not O.verify_sha256_file(Path(str(tar) + ".sha256")):
        sys.exit("SHA-256 of the downloaded checkpoint does not match; not extracted")
    hits = SS.scan_path(tar)
    if hits:
        for where, name, off in hits:
            print(f"SECRET SCAN HIT: {where}: {name} at byte {off} (value withheld)")
        sys.exit("secret scan hit in the downloaded checkpoint; not extracted, reported")
    with tempfile.TemporaryDirectory() as td:
        safe_extract(tar, Path(td))
        if not O.verify_sha256_file(Path(td) / "SHA256SUMS"):
            sys.exit("a member's SHA-256 does not match SHA256SUMS; not merged")
        for p in Path(td).rglob("*"):
            if p.is_file() and p.name != "SHA256SUMS":
                q = LOCAL_RUNS / p.relative_to(td)
                q.parent.mkdir(parents=True, exist_ok=True)
                q.write_bytes(p.read_bytes())
    st_p = LOCAL_RUNS / "status.json"
    st = json.loads(st_p.read_text()) if st_p.exists() else {}
    st[tag] = {"checkpointed": True, "tar": tar.name, "sha256": O.sha256_file(tar), "fetched_utc": O.utc(time.time())}
    st_p.write_text(json.dumps(st, indent=1))
    print(f"checkpoint {tar.name} verified, scanned and merged into {LOCAL_RUNS}; {tag} marked checkpointed")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "push", "put", "exec", "fetch"])
    ap.add_argument("--tag", required=True, help="the authorization tag: GM or a model id")
    ap.add_argument("--commit")
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    O.install_redaction()
    require_auth(a.tag)
    c = conf()
    if a.cmd == "check":
        return ssh(c, "echo pod reachable; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader")
    if a.cmd == "push":
        if not a.commit:
            sys.exit("--commit is required (the locked commit)")
        return cmd_push(c, a.commit)
    if a.cmd == "put":
        rest = [r for r in a.rest if r != "--"]
        if len(rest) != 2:
            sys.exit("put <local> <remote>")
        if "authorized_v4" in rest[1].lower():
            sys.exit("refusing to write an authorization file")
        return scp(c, rest[0], remote(c, rest[1]))
    if a.cmd == "exec":
        rest = a.rest[1:] if a.rest and a.rest[0] == "--" else a.rest
        cmd = " ".join(shlex.quote(x) for x in rest) if len(rest) > 1 else (rest[0] if rest else "")
        ok, why = command_allowed(cmd)
        if not ok:
            sys.exit(f"Refusing: {why}")
        return ssh(c, POD_ENV + cmd)
    return cmd_fetch(c, a.tag)


if __name__ == "__main__":
    sys.exit(main())
