"""Laptop-side SSH / scp tooling for the v5 pod (PROTOCOL_V5.md section 10), derived from the locked tools/v4_remote.py. Claude Code reaches the
pod's shell ONLY through this tool, over the pod's SSH (public IP and exposed TCP port). Pod lifecycle (create, start, stop) goes through the
official RunPod MCP with the corresponding author's approval in the permission prompt; this tool never calls RunPod, holds no API key, and refuses
commands that would use one, log in to Hugging Face, or write an authorization file.

Every pod action needs the laptop copy of the corresponding author's authorization file, <local runs>/AUTHORIZED_V5_<tag>.txt, containing
'PROTOCOL_V5', the lock id and <tag> (the pod's driver checks its own copy on the volume again). All output is passed through the redaction filter.

Connection details: research_v2/v5_runs/pod_ssh.json, written after the pod is created:
  {"host": "<public IP>", "port": <exposed TCP port for 22>, "user": "root", "identity_file": "C:/Users/<you>/.ssh/<key>"}
Host keys are pinned per pod in research_v2/v5_runs/known_hosts (accept-new on first contact).

Local "pod" (rehearsal, section 10): with V5_LOCAL_POD=<directory>, 'exec' runs the command with bash in <directory>/repo, 'put' and 'fetch' copy
files, and the pod's runs directory is <directory>/v5_runs; V5_LOCAL_RUNS sets the laptop-side runs directory. Everything else is the same code.

  python research_v2/tools/v5_remote.py --tag T check                        ssh echo + GPU name (no other effect)
  python research_v2/tools/v5_remote.py --tag T --commit <sha> push          git bundle of the branch, scp, clone/fetch, detached checkout of <sha>
  python research_v2/tools/v5_remote.py --tag T put <local> <remote>         copy one file to the pod
  python research_v2/tools/v5_remote.py --tag T exec -- <command ...>        run in the pod's repo with the venv and V5_RUNS/HF_HOME set
  python research_v2/tools/v5_remote.py --tag T fetch                        newest checkpoint of T: copy, sha256 check, secret scan, extract, status
  python research_v2/tools/v5_remote.py --tag T manifest                     V5-D3: the pod's checkpoint list and hashes -> <local runs>/pod_checkpoints.json
  python research_v2/tools/v5_remote.py --tag T sync                         V5-D3: manifest, then fetch every tag whose newest pod checkpoint is not
                                                                             the laptop's (name or hash differs), whatever the 'checkpointed' flag says
The options come before the subcommand. From Git Bash, set MSYS_NO_PATHCONV=1, or /workspace/... arguments are rewritten as Windows paths.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

RV = Path(__file__).resolve().parents[1]
ROOT = RV.parent
sys.path.insert(0, str(RV / "v5"))
sys.path.insert(0, str(RV / "tools"))
import v5_ops as O  # noqa: E402
import v4_secret_scan as SS  # noqa: E402  (locked v4 tool; same patterns)

LOCAL_POD = os.environ.get("V5_LOCAL_POD")
LOCAL_RUNS = Path(os.environ.get("V5_LOCAL_RUNS", RV / "v5_runs"))
POD_ROOT = Path(LOCAL_POD).as_posix() if LOCAL_POD else "/workspace"
POD_RUNS = f"{POD_ROOT}/v5_runs"
POD_REPO = f"{POD_ROOT}/repo"
POD_ENV = (f"cd '{POD_REPO}' && {{ [ ! -f '{POD_ROOT}/venv/bin/activate' ] || . '{POD_ROOT}/venv/bin/activate'; }} && export V5_RUNS='{POD_RUNS}' "
           f"HF_HOME='{POD_ROOT}/hf' HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_TELEMETRY=1 PYTHONUNBUFFERED=1 && ")
FORBIDDEN = ("runpodctl", "api.runpod.io", "rest.runpod.io", "huggingface-cli login", "hf auth login", "hf_token=", "hugging_face_hub_token=")
AUTH_WRITERS = (">", "tee", "cp ", "mv ", "touch", "rm ", "sed ", "dd ", "install ", "truncate", "chmod", "ln ")


def lock_id():
    r = subprocess.run([sys.executable, str(RV / "tools" / "lock_v5.py"), "--lock-id"], capture_output=True, **O.TEXT)
    return r.stdout.strip() if r.returncode == 0 else None


def require_auth(tag):
    ok, why = O.check_auth(LOCAL_RUNS, tag, lock_id())
    if not ok:
        sys.exit(f"Refusing: laptop copy of the authorization file: {why}")


def command_allowed(cmd: str):
    low = cmd.lower()
    for f in FORBIDDEN:
        if f in low:
            return False, f"'{f}' is not permitted (section 10: no RunPod CLI or API from the pod shell, no Hugging Face token)"
    if "authorized_v5" in low and any(w in low for w in AUTH_WRITERS):
        return False, "commands that could create or change an authorization file are refused"
    return True, "ok"


def conf():
    if LOCAL_POD:
        return {"local": True}
    p = LOCAL_RUNS / "pod_ssh.json"
    if not p.exists():
        sys.exit(f"{p} is missing (written after the pod is created)")
    return json.loads(p.read_text(encoding="utf-8"))


def known_hosts_option():
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


def bash():
    return shutil.which("bash") or "bash"


def ssh(c, remote_cmd):
    if c.get("local"):
        return run_stream([bash(), "-c", remote_cmd])
    return run_stream(ssh_base(c) + [f"{c['user']}@{c['host']}", remote_cmd])


def ssh_capture(c, remote_cmd):
    if c.get("local"):
        return subprocess.run([bash(), "-c", remote_cmd], capture_output=True, **O.TEXT)
    return subprocess.run(ssh_base(c) + [f"{c['user']}@{c['host']}", remote_cmd], capture_output=True, **O.TEXT)


def copy(c, src, dst):
    """scp from or to the pod; in local mode a plain copy (remote paths are local paths)."""
    if c.get("local"):
        s, d = src.split(":", 1)[1] if src.startswith("pod:") else src, dst.split(":", 1)[1] if dst.startswith("pod:") else dst
        Path(d).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(s, d)
        return 0
    return run_stream(ssh_base(c, "scp") + [src, dst])


def remote(c, path):
    return f"pod:{path}" if c.get("local") else f"{c['user']}@{c['host']}:{path}"


def cmd_push(c, commit):
    branch = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, check=True, **O.TEXT).stdout.strip()
    full = subprocess.run(["git", "-C", str(ROOT), "rev-parse", commit], capture_output=True, check=True, **O.TEXT).stdout.strip()
    with tempfile.TemporaryDirectory() as td:
        b = Path(td) / "v5.bundle"
        subprocess.run(["git", "-C", str(ROOT), "bundle", "create", str(b), branch], check=True)
        if copy(c, str(b), remote(c, f"{POD_ROOT}/v5.bundle")) != 0:
            sys.exit("copy of the bundle failed")
    script = (f"set -e; if [ ! -d '{POD_REPO}/.git' ]; then git -c core.autocrlf=false clone -q '{POD_ROOT}/v5.bundle' '{POD_REPO}'; fi; "
              f"git -C '{POD_REPO}' fetch -q '{POD_ROOT}/v5.bundle' {branch}; git -C '{POD_REPO}' -c advice.detachedHead=false checkout -q {full}; "
              f"git -C '{POD_REPO}' rev-parse HEAD")
    return ssh(c, script)


def safe_extract(tar: Path, dest: Path):
    with tarfile.open(tar) as tf:
        for m in tf.getmembers():
            p = (dest / m.name).resolve()
            if not str(p).startswith(str(dest.resolve())) or m.issym() or m.islnk():
                sys.exit(f"refusing to extract {m.name}: unsafe path")
        tf.extractall(dest)


def cmd_manifest(c):
    """V5-D3: every finished checkpoint on the pod (a tar counts once its .sha256 exists; pack_checkpoint writes it last), with its hash."""
    listing = ssh_capture(c, f"for f in '{POD_RUNS}'/checkpoints/*.tar.sha256; do [ -f \"$f\" ] && cat \"$f\"; done; echo MANIFEST_OK")
    if listing.returncode != 0 or "MANIFEST_OK" not in (listing.stdout or ""):
        sys.exit("could not read the pod's checkpoint list")
    man = {"utc": O.utc(time.time()), "tars": O.parse_pod_sums(listing.stdout)}
    LOCAL_RUNS.mkdir(parents=True, exist_ok=True)
    (LOCAL_RUNS / "pod_checkpoints.json").write_text(json.dumps(man, indent=1))
    return man


def cmd_sync(c):
    """V5-D3: fetch every stale tag (its newest pod checkpoint differs from the laptop's), each through the verifying fetch."""
    man = cmd_manifest(c)
    st_p = LOCAL_RUNS / "status.json"
    st = json.loads(st_p.read_text()) if st_p.exists() else {}
    newest = O.newest_by_tag(man["tars"])
    for tag in O.stale_tags(man["tars"], st):
        ok, why = O.check_auth(LOCAL_RUNS, tag, lock_id())
        if not ok:
            sys.exit(f"Refusing to fetch {tag}: laptop copy of the authorization file: {why}")
        print(f"{O.utc(time.time())[11:]} fetching {tag} ({newest[tag][0]})", flush=True)
        cmd_fetch(c, tag, newest[tag][0])
    return 0


def cmd_fetch(c, tag, name=None):
    LOCAL_RUNS.mkdir(parents=True, exist_ok=True)
    if name is None:
        listing = ssh_capture(c, f"ls -1 '{POD_RUNS}/checkpoints/'{tag}_*.tar.sha256")
        tars = sorted(l.strip()[:-len(".sha256")] for l in listing.stdout.splitlines() if l.strip().endswith(".tar.sha256"))
        if not tars:
            sys.exit(f"no checkpoint for {tag} on the pod (run: run_v5.py checkpoint --tag ...)")
        src = tars[-1]
    else:
        if O.checkpoint_tag(name) != tag:
            sys.exit(f"{name} is not a checkpoint of {tag}")
        src = f"{POD_RUNS}/checkpoints/{name}"
    dst = LOCAL_RUNS / "checkpoints"
    dst.mkdir(exist_ok=True)
    for s in (src, src + ".sha256"):
        if copy(c, remote(c, s), str(dst / Path(s).name)) != 0:
            sys.exit(f"copy of {s} failed")
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
    ap.add_argument("cmd", choices=["check", "push", "put", "exec", "fetch", "manifest", "sync"])
    ap.add_argument("--tag", required=True, help="the authorization tag: GM, U0, DESIGN, U1 ... U6")
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
            sys.exit("--commit is required (the lock commit)")
        return cmd_push(c, a.commit)
    if a.cmd == "put":
        rest = [r for r in a.rest if r != "--"]
        if len(rest) != 2:
            sys.exit("put <local> <remote>")
        if "authorized_v5" in rest[1].lower():
            sys.exit("refusing to write an authorization file")
        return copy(c, rest[0], remote(c, rest[1]))
    if a.cmd == "exec":
        rest = a.rest[1:] if a.rest and a.rest[0] == "--" else a.rest
        cmd = " ".join(shlex.quote(x) for x in rest) if len(rest) > 1 else (rest[0] if rest else "")
        ok, why = command_allowed(cmd)
        if not ok:
            sys.exit(f"Refusing: {why}")
        return ssh(c, POD_ENV + cmd)
    if a.cmd == "manifest":
        man = cmd_manifest(c)
        print(json.dumps(man, indent=1))
        return 0
    if a.cmd == "sync":
        return cmd_sync(c)
    return cmd_fetch(c, a.tag)


if __name__ == "__main__":
    sys.exit(main())
