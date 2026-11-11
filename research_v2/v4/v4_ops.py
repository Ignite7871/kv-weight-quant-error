"""v4 cloud operations (PROTOCOL_V4.md sections 10 and 11): authorization files, the cost ledger (USD) and the GPU ledger (hours), the ceiling
check and watchdog, secret redaction, the token-file check, the SKU check and checkpoint packing. No torch; every clock and path is injectable so
the CPU harness can drive each path.

Cost ledger (v4_runs/cost_ledger.jsonl). In-pod lines are written by the driver: 'session_open' (pod start, declared $/h, SKU), 'session_close',
and 'recovered_close' for a session whose driver died (from the atomic heartbeat file cost_heartbeat.json, rewritten every 15 s). 'billing' lines
are entered by the corresponding author from the RunPod billing page (authoritative USD). The used amount is the LARGER of
  in-pod:   sum over pod sessions of (last seen - pod start + idle auto-stop window) x rate, plus volume storage since the volume was created;
  billing:  the latest billing figure, plus the in-pod accrual after it (same rule) and storage since it.
Pod start is the start time of PID 1 of the pod's container (/proc/1/stat field 22 over /proc/stat btime). /proc/uptime would give the HOST's
boot inside a container; it is used only as a (conservative, earlier) fallback.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

CEILING_USD = 30.0                          # D7 (r5): hard ceiling, total, including storage
CLOSEOUT_RESERVE_USD = 1.0                  # r5: kept back from the run for the final download and the storage until the volume is deleted
AUTHOR_WAIT_H = 48.0                        # r5: expected hours with the volume alive and no pod (decisions, gaps between models); storage allowance only
STORAGE_USD_PER_GB_MONTH = 0.07             # network volume, Standard tier (section 10)
HEARTBEAT_S = 15.0
WATCHDOG_MARGIN_S = 60.0                    # the watchdog closes the ledgers and stops the run this long before the ceiling
SKU = "A40"                                 # D5 (r4): the only permitted GPU type; also the key of the plan-numbers projection
SKU_NAMES = ("NVIDIA A40", "A40")           # exact nvidia-smi names: a substring test would also accept RTX A4000, RTX A400 and vGPU profiles
IMAGE_REPO = "runpod/pytorch"               # section 11: the locked image; its digest is recorded at first boot (r4)
IMAGE_TAG = "1.0.2-cu1281-torch280-ubuntu2404"
AUTH_TEXT = "PROTOCOL_V4"
MONTH_S = 30.0 * 86400.0
SECRET_PATTERNS = [("hf_token", re.compile(rb"hf_[A-Za-z0-9]{20,}")), ("runpod_api_key", re.compile(rb"rpa_[A-Za-z0-9]{20,}")),
                   ("runpod_s3_key", re.compile(rb"rps_[A-Za-z0-9]{20,}"))]
SECRET_ENV = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "RUNPOD_API_KEY")
TEXT = {"text": True, "encoding": "utf-8", "errors": "replace"}      # r5: child output never depends on the platform's default text encoding
STOP_FILES = ("ceiling_stop.json", "drift_stop.json", "integrity_failure.json", "g8_failure.json", "resume_check_failure.json")   # V4-D1


def stop_usd() -> float:
    """r5: the run stops here, not at the ceiling itself: the rest pays for fetching the last checkpoint and for the volume until it is deleted."""
    return CEILING_USD - CLOSEOUT_RESERVE_USD


def utc(t: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def parse_utc(s: str) -> float:
    import calendar
    return float(calendar.timegm(time.strptime(s, "%Y-%m-%dT%H:%M:%SZ")))


def atomic_write(p: Path, text: str):
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, p)


def append_jsonl(p: Path, obj: dict):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(obj, default=float) + "\n")


def read_jsonl(p: Path) -> list:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


# ----------------------------------------------------------------------------- authorization
def auth_path(out: Path, tag: str) -> Path:
    return out / f"AUTHORIZED_V4_{tag}.txt"


def check_auth(out: Path, tag: str, lock_id: str | None):
    """(ok, reason). The file must exist and contain 'PROTOCOL_V4', the lock id, and the tag as a whole whitespace-separated token (so that
    'llama3.2-1b' is not authorized by a file written for 'llama3.2-1b-instruct')."""
    if lock_id is None:
        return False, "no v4 lock (tools/lock_v4.py)"
    p = auth_path(out, tag)
    if not p.exists():
        return False, f"{p.name} does not exist (only the corresponding author creates it)"
    txt = p.read_text(encoding="utf-8", errors="replace")
    if AUTH_TEXT not in txt:
        return False, f"{p.name} does not contain '{AUTH_TEXT}'"
    if lock_id not in txt:
        return False, f"{p.name} does not contain the lock id {lock_id}"
    if tag not in txt.split():
        return False, f"{p.name} does not name '{tag}' as a separate token"
    return True, "ok"


# ----------------------------------------------------------------------------- pod start, SKU, token files
def pod_start_epoch(proc: str = "/proc", now=time.time, clk_tck: int | None = None) -> float:
    try:
        stat = Path(proc, "1", "stat").read_text()
        fields = stat[stat.rindex(")") + 2:].split()
        start_ticks = int(fields[19])                                  # field 22 of /proc/<pid>/stat (1-based, counting pid and comm)
        btime = next(int(l.split()[1]) for l in Path(proc, "stat").read_text().splitlines() if l.startswith("btime "))
        return btime + start_ticks / (clk_tck or os.sysconf("SC_CLK_TCK"))
    except Exception:  # noqa: BLE001
        try:
            return now() - float(Path(proc, "uptime").read_text().split()[0])
        except Exception:  # noqa: BLE001
            return now()                                               # not on Linux (CPU tests): the process start


def gpu_name() -> str | None:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, timeout=10, **TEXT)
        return r.stdout.strip().splitlines()[0] if r.returncode == 0 and r.stdout.strip() else None
    except Exception:  # noqa: BLE001
        return None


def sku_ok(name: str | None) -> bool:
    return name is not None and name.strip() in SKU_NAMES


def gpu_environment(run=subprocess.run) -> dict:
    """Name, driver version, ECC mode and total memory from the nvidia-smi query, and the CUDA version from its banner (the highest the driver
    supports; the study's own runtime is torch's cu126). A value that cannot be read is None. Logged at GM and in every record manifest."""
    env = {"name": None, "driver_version": None, "ecc_mode": None, "memory_total": None, "cuda_version": None}
    try:
        r = run(["nvidia-smi", "--query-gpu=name,driver_version,ecc.mode.current,memory.total", "--format=csv,noheader"],
                capture_output=True, timeout=20, **TEXT)
        if r.returncode == 0 and r.stdout.strip():
            parts = [p.strip() for p in r.stdout.strip().splitlines()[0].split(",")]
            if len(parts) == 4:
                env.update(zip(("name", "driver_version", "ecc_mode", "memory_total"), parts))
        r = run(["nvidia-smi"], capture_output=True, timeout=20, **TEXT)
        m = re.search(r"CUDA Version:\s*([0-9][0-9.]*)", r.stdout or "")
        env["cuda_version"] = m.group(1) if m else None
    except Exception:  # noqa: BLE001
        pass
    return env


def gpu_environment_complete(env: dict) -> bool:
    """Every field was read. ECC may be 'Enabled', 'Disabled' or '[N/A]'; it is recorded, not judged."""
    return all(env.get(k) for k in ("name", "driver_version", "ecc_mode", "cuda_version"))


# ----------------------------------------------------------------------------- container image digest (r4)
def fetch_image_digest(repo: str = IMAGE_REPO, tag: str = IMAGE_TAG, opener=None) -> str:
    """The digest Docker Hub reports for the tag, read anonymously. A container cannot see the digest of its own image, so this is the digest the
    locked tag points at when it is read, not proof of what the host pulled."""
    import urllib.request
    url = f"https://hub.docker.com/v2/repositories/{repo}/tags/{tag}"
    with (opener or urllib.request.urlopen)(url, timeout=30) as r:
        digest = json.loads(r.read().decode("utf-8")).get("digest")
    if not (isinstance(digest, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", digest)):
        raise ValueError(f"Docker Hub returned no valid digest for {repo}:{tag}")
    return digest


def record_image_digest(out: Path, fetch=fetch_image_digest, now=time.time) -> dict:
    """Write-once: v4_runs/image_digest.json is written at first boot and never replaced. A later read that differs goes to
    image_digest_changes.jsonl for a deviation row."""
    p = out / "image_digest.json"
    digest = fetch()
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        if d["digest"] != digest:
            append_jsonl(out / "image_digest_changes.jsonl", {"recorded": d["digest"], "read": digest, "utc": utc(now())})
            print(f"[IMAGE] the tag now points at {digest}; {d['digest']} stays on record. Add a deviation row.", flush=True)
        return d
    d = {"image": IMAGE_REPO, "tag": IMAGE_TAG, "digest": digest, "source": "Docker Hub tag record, read inside the pod at first boot", "utc": utc(now())}
    out.mkdir(parents=True, exist_ok=True)
    atomic_write(p, json.dumps(d, indent=1))
    return d


def image_record(out: Path):
    """The first-boot record if it names the locked image and tag, else None."""
    p = out / "image_digest.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    return d if (d.get("image"), d.get("tag")) == (IMAGE_REPO, IMAGE_TAG) and d.get("digest") else None


def token_files(hf_home: str | None = None, home: str | None = None) -> list:
    """Token files that must NOT exist on the pod (huggingface-cli login is never used; the token is only the HF_TOKEN variable)."""
    hf_home = hf_home or os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    home = home or str(Path.home())
    cands = [Path(hf_home) / "token", Path(hf_home) / "stored_tokens", Path(home) / ".cache" / "huggingface" / "token",
             Path(home) / ".huggingface" / "token"]
    return [str(p) for p in cands if p.exists()]


# ----------------------------------------------------------------------------- secrets
def pid1_environ(proc: str = "/proc") -> dict:
    """RunPod passes pod environment variables (and secrets) to the container's PID 1 only, not to SSH login shells, so the driver reads them
    from /proc/1/environ, in memory; nothing is written. Empty off Linux."""
    try:
        raw = Path(proc, "1", "environ").read_bytes()
    except OSError:
        return {}
    out = {}
    for item in raw.split(b"\0"):
        if b"=" in item:
            k, v = item.split(b"=", 1)
            out[k.decode(errors="replace")] = v.decode(errors="replace")
    return out


def pod_env(name: str, proc: str = "/proc"):
    return os.environ.get(name) or pid1_environ(proc).get(name)


def secret_literals(env=None, proc: str = "/proc") -> list:
    envs = [os.environ, pid1_environ(proc)] if env is None else [env]
    vals = {e[k] for e in envs for k in SECRET_ENV if e.get(k) and len(e[k]) >= 8}
    return [v.encode() for v in sorted(vals)]


def find_secrets(b: bytes, literals=()) -> list:
    """[(pattern name, offset)]; the value itself is never returned."""
    hits = [(name, m.start()) for name, rx in SECRET_PATTERNS for m in rx.finditer(b)]
    for lit in literals:
        i = b.find(lit)
        if i >= 0:
            hits.append(("literal environment secret", i))
    return hits


def redact(text: str, literals=()) -> str:
    for lit in literals:
        text = text.replace(lit.decode(errors="ignore"), "[REDACTED]")
    for name, rx in SECRET_PATTERNS:
        text = re.sub(rx.pattern.decode(), f"[REDACTED {name}]", text)
    return text


class RedactingStream(io.TextIOBase):
    """Wraps stdout / stderr: every write is redacted before it reaches the terminal or a log."""

    def __init__(self, inner, literals=None):
        self.inner = inner
        self.literals = secret_literals() if literals is None else literals

    def write(self, s):
        return self.inner.write(redact(s, self.literals))

    def flush(self):
        self.inner.flush()


def safe_stream(stream):
    """r5: a character the console encoding lacks is escaped instead of raising. On the laptop a piped stdout is cp1252, and the pod's progress bars and
    warnings are not: without this, `v4_remote.py exec` dies on the first such character and takes the SSH stream of a paid run with it."""
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(errors="backslashreplace")
    return stream


def install_redaction():
    if not isinstance(sys.stdout, RedactingStream):
        sys.stdout = RedactingStream(safe_stream(sys.stdout))
        sys.stderr = RedactingStream(safe_stream(sys.stderr))


# ----------------------------------------------------------------------------- cost ledger
def load_cost_config(out: Path) -> dict:
    p = out / "cost_config.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} is missing: the corresponding author writes it (rate_usd_per_h, autostop_minutes, volume_gb, volume_created_utc)")
    c = json.loads(p.read_text(encoding="utf-8"))
    for k in ("rate_usd_per_h", "autostop_minutes", "volume_gb", "volume_created_utc"):
        if k not in c:
            raise ValueError(f"cost_config.json lacks {k}")
    if not (c["rate_usd_per_h"] > 0 and c["autostop_minutes"] > 0 and c["volume_gb"] > 0):
        raise ValueError("cost_config.json: rate, auto-stop window and volume size must be positive")
    return c


class CostLedger:
    def __init__(self, out: Path, cfg: dict, now=time.time, pod_start=None, sku: str | None = None):
        self.out, self.cfg, self.now = out, cfg, now
        self.path, self.hb = out / "cost_ledger.jsonl", out / "cost_heartbeat.json"
        self.boot = pod_start if pod_start is not None else pod_start_epoch(now=now)
        self.sku = sku
        self.open = False

    # --- lines
    def recover(self) -> bool:
        """A heartbeat file left by a driver that did not close its session is turned into a 'recovered_close' line."""
        if not self.hb.exists():
            return False
        try:
            d = json.loads(self.hb.read_text(encoding="utf-8"))
        except ValueError:
            d = None
        if d:
            append_jsonl(self.path, {"kind": "recovered_close", "pod_start_utc": utc(d["boot"]), "boot": d["boot"], "last": d["last"],
                                     "utc": utc(d["last"]), "rate_usd_per_h": d.get("rate", self.cfg["rate_usd_per_h"])})
        self.hb.unlink(missing_ok=True)
        return True

    def open_session(self, block: str):
        self.recover()
        t = self.now()
        append_jsonl(self.path, {"kind": "session_open", "block": block, "pod_start_utc": utc(self.boot), "boot": self.boot, "last": t, "utc": utc(t),
                                 "rate_usd_per_h": self.cfg["rate_usd_per_h"], "sku": self.sku, "pid": os.getpid()})
        self.open = True
        self.heartbeat()

    def heartbeat(self):
        atomic_write(self.hb, json.dumps({"boot": self.boot, "last": self.now(), "rate": self.cfg["rate_usd_per_h"]}))

    def close(self, completed: bool, extra=None):
        if not self.open:
            return
        t = self.now()
        est = self.estimate(t)
        append_jsonl(self.path, {"kind": "session_close", "pod_start_utc": utc(self.boot), "boot": self.boot, "last": t, "utc": utc(t),
                                 "rate_usd_per_h": self.cfg["rate_usd_per_h"], "completed": bool(completed), "used_usd_estimate": est["used_usd"],
                                 **(extra or {})})
        self.hb.unlink(missing_ok=True)
        self.open = False

    # --- the estimate
    def sessions(self, now: float) -> dict:
        s = {}
        for l in read_jsonl(self.path):
            if l["kind"] in ("session_open", "session_close", "recovered_close"):
                b = l["boot"]
                s[b] = {"last": max(s.get(b, {}).get("last", b), l["last"]), "rate": l.get("rate_usd_per_h", self.cfg["rate_usd_per_h"])}
        if self.open:
            s[self.boot] = {"last": max(s.get(self.boot, {}).get("last", self.boot), now), "rate": self.cfg["rate_usd_per_h"]}
        return s

    def storage_usd(self, t0: float, t1: float) -> float:
        deleted = self.cfg.get("volume_deleted_utc")                   # r5: storage stops when the corresponding author deletes the volume
        if deleted:
            t1 = min(t1, parse_utc(deleted))
        return max(0.0, t1 - t0) / MONTH_S * self.cfg["volume_gb"] * STORAGE_USD_PER_GB_MONTH

    def estimate(self, now: float | None = None) -> dict:
        now = self.now() if now is None else now
        auto = self.cfg["autostop_minutes"] * 60.0
        sess = self.sessions(now)
        vol0 = parse_utc(self.cfg["volume_created_utc"])
        pods = sum((v["last"] - b + auto) * v["rate"] / 3600.0 for b, v in sess.items())
        inpod = pods + self.storage_usd(vol0, now)
        bills = [l for l in read_jsonl(self.path) if l["kind"] == "billing"]
        billing = None
        if bills:
            last = max(enumerate(bills), key=lambda il: (parse_utc(il[1]["utc"]), il[0]))[1]      # latest time; on a tie, the later line
            tb = parse_utc(last["utc"])
            acc = sum((v["last"] - max(b, tb) + auto) * v["rate"] / 3600.0 for b, v in sess.items() if v["last"] >= tb)
            billing = float(last["usd"]) + acc + self.storage_usd(tb, now)
        used = max(inpod, billing) if billing is not None else inpod
        return {"inpod_usd": inpod, "billing_based_usd": billing, "used_usd": used, "ceiling_usd": CEILING_USD, "stop_usd": stop_usd(), "n_pod_sessions": len(sess)}

    def can_start(self, projected_seconds: float) -> tuple:
        """Checked BEFORE a problem: refuse if the used amount plus the projected cost of the problem (and the watchdog margin) would cross
        the stop level (the ceiling less the close-out reserve)."""
        est = self.estimate()
        add = (projected_seconds + WATCHDOG_MARGIN_S) * self.cfg["rate_usd_per_h"] / 3600.0
        return est["used_usd"] + add <= stop_usd(), est

    def at_watchdog_limit(self) -> bool:
        return self.estimate()["used_usd"] + WATCHDOG_MARGIN_S * self.cfg["rate_usd_per_h"] / 3600.0 >= stop_usd()


def add_billing_line(out: Path, usd: float, when_utc: str, note: str = ""):
    """Entered by the corresponding author from the RunPod billing page (cumulative USD spent on this study at that time)."""
    append_jsonl(out / "cost_ledger.jsonl", {"kind": "billing", "usd": float(usd), "utc": when_utc, "entered_by": "corresponding author", "note": note})


# ----------------------------------------------------------------------------- GPU ledger (v3 heartbeat semantics)
class GpuLedger:
    def __init__(self, out: Path, now=time.time):
        self.out, self.now = out, now
        self.path, self.progress = out / "gpu_ledger.jsonl", out / "current_process.json"
        self.t0 = None
        self.block = None
        self.done = False

    def recover(self) -> float:
        if not self.progress.exists():
            return 0.0
        try:
            d = json.loads(self.progress.read_text(encoding="utf-8"))
        except ValueError:
            d = {}
        sec = float(d.get("seconds_so_far", 0.0)) if d.get("open") else 0.0
        if d.get("open"):
            append_jsonl(self.path, {"start_utc": d.get("start_utc"), "seconds": sec, "block": d.get("block"), "completed": False,
                                     "recovered_after_unclean_exit": True})
        self.progress.unlink()
        return sec

    def seconds(self) -> float:
        return sum(float(l["seconds"]) for l in read_jsonl(self.path))

    def start(self, block: str):
        self.recover()
        self.t0, self.block, self.done = self.now(), block, False
        self.heartbeat()

    def heartbeat(self):
        atomic_write(self.progress, json.dumps({"open": True, "block": self.block, "start_utc": utc(self.t0), "seconds_so_far": self.now() - self.t0,
                                                "last_utc": utc(self.now())}))

    def close(self, completed: bool, extra=None):
        if self.done or self.t0 is None:
            return
        self.done = True
        append_jsonl(self.path, {"start_utc": utc(self.t0), "seconds": self.now() - self.t0, "block": self.block, "completed": bool(completed), **(extra or {})})
        self.progress.unlink(missing_ok=True)


class Ledgers:
    """Both ledgers around one GPU process, with a 15 s heartbeat thread that also acts as the cost watchdog."""

    def __init__(self, gpu: GpuLedger, cost: CostLedger, block: str, on_expire=None, period=HEARTBEAT_S):
        self.gpu, self.cost, self.block, self.period = gpu, cost, block, period
        self.completed = False
        self.on_expire = on_expire or (lambda: os._exit(3))
        self.lock = threading.Lock()
        self.stop_ev = threading.Event()
        self.fired = False

    def __enter__(self):
        self.gpu.start(self.block)
        self.cost.open_session(self.block)
        self.th = threading.Thread(target=self._beat, daemon=True)
        self.th.start()
        return self

    def tick(self):
        with self.lock:
            if self.fired:
                return
            self.gpu.heartbeat()
            self.cost.heartbeat()
            if self.cost.at_watchdog_limit():
                self.fired = True
                self.gpu.close(False, {"watchdog_fired": True})
                self.cost.close(False, {"watchdog_fired": True})
                print(f"[WATCHDOG] the ${stop_usd():.2f} stop level of the ${CEILING_USD:.0f} ceiling is {WATCHDOG_MARGIN_S:.0f} s away; ledgers closed, stopping.", flush=True)
                self.on_expire()

    def _beat(self):
        while not self.stop_ev.wait(self.period):
            self.tick()

    def close(self, completed: bool):
        self.stop_ev.set()
        with self.lock:
            self.gpu.close(completed)
            self.cost.close(completed)

    def __exit__(self, et, ev, tb):
        self.close(et is None and self.completed)
        return False


# ----------------------------------------------------------------------------- checkpoint packing
def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def pack_checkpoint(out: Path, tag: str, extra_files=(), dest: Path | None = None):
    """tar of v4_runs/<tag>/ and the shared ledgers and run files, with SHA256SUMS (LF) inside and <tar>.sha256 beside it."""
    dest = dest or (out / "checkpoints")
    dest.mkdir(parents=True, exist_ok=True)
    tar_path = dest / f"{tag}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.tar"
    shared = ["gpu_ledger.jsonl", "cost_ledger.jsonl", "cost_config.json", "gm.json", "design.json", "step0.json", "status.json", "g6_measure.json",
              "g6_probe.json", "image_digest.json", "image_digest_changes.jsonl"]
    files = sorted(p for p in (out / tag).rglob("*") if p.is_file())
    files += [out / n for n in shared if (out / n).exists()]
    files += sorted(out.glob("AUTHORIZED_V4_*.txt"))
    sums = []
    with tarfile.open(tar_path, "w") as tf:
        for p in files:
            arc = p.relative_to(out).as_posix()
            tf.add(p, arcname=arc)
            sums.append(f"{sha256_file(p)}  {arc}")
        for p in extra_files:
            p = Path(p)
            arc = f"_extra/{p.name}"
            tf.add(p, arcname=arc)
            sums.append(f"{sha256_file(p)}  {arc}")
        b = ("\n".join(sums) + "\n").encode()
        ti = tarfile.TarInfo("SHA256SUMS")
        ti.size = len(b)
        tf.addfile(ti, io.BytesIO(b))
    Path(str(tar_path) + ".sha256").write_bytes(f"{sha256_file(tar_path)}  {tar_path.name}\n".encode())
    return tar_path


def verify_sha256_file(sum_file: Path) -> bool:
    """`sha256sum -c` semantics on a file written with LF endings (CRLF tolerated on read)."""
    ok = True
    for line in sum_file.read_bytes().replace(b"\r\n", b"\n").decode().splitlines():
        if not line.strip():
            continue
        h, name = line.split("  ", 1)
        p = sum_file.parent / name
        ok = ok and p.exists() and sha256_file(p) == h
    return ok


def storage_allowance(volume_gb: float, billed_h: float) -> float:
    """r5: the volume's cost prorated to the expected duration (billed hours plus the expected hours with no pod), not a whole month."""
    return volume_gb * STORAGE_USD_PER_GB_MONTH * (billed_h + AUTHOR_WAIT_H) * 3600.0 / MONTH_S


def closeout_report(out: Path, tags, drop_first=None) -> dict:
    """r5, on the laptop, before the corresponding author deletes the volume (which cannot be undone): every tag has a verified checkpoint here, the tar
    is unchanged, and the merged records show the tag's final state. A model the locked design dropped has nothing to fetch."""
    st_p, dz_p = out / "status.json", out / "design.json"
    st = json.loads(st_p.read_text(encoding="utf-8")) if st_p.exists() else {}
    dz = json.loads(dz_p.read_text(encoding="utf-8")) if dz_p.exists() else {}
    missing = []
    for t in tags:
        if t == drop_first and dz.get("drop_mistral"):
            continue
        e = st.get(t)
        if not (e and e.get("checkpointed")):
            missing.append(f"{t}: no verified checkpoint on the laptop")
            continue
        tar = out / "checkpoints" / e["tar"]
        if not tar.exists() or sha256_file(tar) != e["sha256"]:
            missing.append(f"{t}: the checkpoint tar is missing or changed")
            continue
        if not any((out / t / n).exists() for n in ("test_complete.json", "not_run.json") + STOP_FILES):
            missing.append(f"{t}: the merged records show no final state (fetch again after the model ends)")
    return {"ready": not missing, "missing": missing}


def gz_bytes(b: bytes) -> bytes:
    return gzip.compress(b)
