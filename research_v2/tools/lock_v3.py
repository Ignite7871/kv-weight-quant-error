"""Write / verify the dated SHA-256 lock of v3 (PROTOCOL_V3.md). Private dated hash for the coauthors; NOT a public pre-registration.
  python research_v2/tools/lock_v3.py                          create V3_PROTOCOL_LOCK.txt from COMMITTED content (commit first, then lock)
  python research_v2/tools/lock_v3.py --verify                 exit 0 iff every locked file matches the commit and is unmodified in the working tree,
                                                               the E1 / E1L lock files are unchanged, and appended calibration/design hashes match
  python research_v2/tools/lock_v3.py --verify-model M         as --verify, and re-hash model M's weight files (slow: reads the safetensors)
  python research_v2/tools/lock_v3.py --lock-id                print the lock id (first 16 hex of the SHA-256 of the lock text before any appended line)
  python research_v2/tools/lock_v3.py --append-calibration M   append the hash of v3_runs/M/calibration.json (once, before any test problem of M)
  python research_v2/tools/lock_v3.py --append-design          append the hash of v3_runs/design.json (once, before any test problem of any model)
  python research_v2/tools/lock_v3.py --amend "<text>" --preserved <path under research_v2>   re-create the lock after a logged amendment
"""
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

RV = Path(__file__).resolve().parents[1]
ROOT = RV.parent
LOCK = RV / "V3_PROTOCOL_LOCK.txt"
OUT = RV / "v3_runs"
PREV_LOCKS = [RV / "E1_PROTOCOL_LOCK.txt", RV / "E1L_PROTOCOL_LOCK.txt"]
MODELS = {"qwen-base": "Qwen/Qwen2.5-0.5B", "llama-instruct": "meta-llama/Llama-3.2-1B-Instruct",
          "qwen-instruct": "Qwen/Qwen2.5-0.5B-Instruct", "llama-base": "meta-llama/Llama-3.2-1B"}
FILES = ([RV / "v3" / "PROTOCOL_V3.md", RV / "run_v3.py", RV / "analyze_v3.py", RV / "v3_sample.json",
          RV / "tests" / "run_v3_tests.py", RV / "tests" / "test_analyze_v3.py", RV / "tests" / "mutation_check_v3.py", RV / "tests" / "cpu_harness_v3_paths.py",
          RV / "tools" / "lock_v3.py", RV / "tools" / "v3_gpu_mutation_check.py", RV / "tools" / "v3_cpu_validation.py",
          RV / "tools" / "v3_repair_torn_line.py", RV / "tests" / "test_v3_repair.py", RV / "tests" / "test_v3_resume.py",
          RV / "validation_logs" / "v3_resume_tests.json",
          RV / "v3" / "phase2" / "project_budget.py", RV / "v3" / "phase2" / "budget_projection.json",
          RV / "validation_logs" / "v3_cpu_validation.json", RV / "validation_logs" / "v3_mutation_check.json",
          RV / "validation_logs" / "v3_cpu_harness.json", RV / "validation_logs" / "v3_validation_summary.json"]
         + sorted((RV / "rv3").glob("*.py"))
         + [RV / f for f in ("run_controls.py", "run_stage1.py", "stage1_runtime.py", "e1l_support.py", "e1_sample.json", "stage1_sample.json",
                             "pilot_sample.json", "stage2_sample.json")]
         + sorted((RV / "rv2").glob("*.py"))
         + [ROOT / "src" / f for f in ("dynamic_linear.py", "dynamic_model.py", "routing.py", "fragility.py")]
         + [ROOT / "scripts" / f for f in ("gsm8k_gate_check.py", "probe_gsm8k_routing.py")])
PROVENANCE = [
    "Private dated hash for coauthors. NOT a public pre-registration.",
    "PROVENANCE: v3 is EXPLORATORY IN ORIGIN (it follows E1, E1L and a literature check). The 200 test ids and 100 calibration ids are copied from the",
    "locked E1 sample. No v3 outcome exists at lock time. This lock is NOT an authorization to run: the authors must create",
    "research_v2/v3_runs/AUTHORIZED_BY_AUTHORS_V3.txt containing 'PROTOCOL_V3' and the lock id printed by `tools/lock_v3.py --lock-id`."]


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, check=True).stdout


def sha(b):
    return hashlib.sha256(b).hexdigest()


def file_sha(p: Path):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def model_files(hf_id):
    base = Path.home() / ".cache" / "huggingface" / "hub" / ("models--" + hf_id.replace("/", "--")) / "snapshots"
    snaps = sorted(base.iterdir())
    if len(snaps) != 1:
        sys.exit(f"expected exactly one cached snapshot for {hf_id}, found {len(snaps)}")
    snap = snaps[0]
    return snap.name, [snap / n for n in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json")]


def create(amendment=None):
    """amendment: (text, preserved original lock path). The original must already have been moved to its preserved path."""
    if LOCK.exists():
        sys.exit("V3_PROTOCOL_LOCK.txt exists; a new lock needs a logged amendment (preserve the old one first)")
    head = git("rev-parse", "HEAD").decode().strip()
    rels = [f.relative_to(ROOT).as_posix() for f in FILES]
    missing = [r for r in rels if not (ROOT / r).exists()]
    if missing:
        sys.exit(f"locked files missing: {missing}")
    dirty = git("status", "--porcelain", "--", *rels).decode().strip()
    if dirty:
        sys.exit(f"Refusing binding lock: uncommitted or modified:\n{dirty}")
    lines = ["protocol lock -- research_v2 v3 (origin and transfer of weight-induced cache error)", f"utc: {datetime.now(timezone.utc).isoformat()}",
             f"git commit containing the locked files: {head}", "Hashes are SHA-256 of the committed file contents (git show <commit>:<path>)."]
    for p in PREV_LOCKS:
        lines.append(f"PREVIOUS LOCK: {sha(p.read_bytes())}  {p.relative_to(ROOT).as_posix()}")
    if amendment:
        text, orig = amendment
        if not orig.exists():
            sys.exit("the preserved original lock is missing")
        lines.append(f"AMENDMENT: {text} | original lock file sha256 {sha(orig.read_bytes())} preserved at {orig.relative_to(ROOT).as_posix()}")
    for m, hf in MODELS.items():
        rev, fs = model_files(hf)
        for f in fs:
            lines.append(f"MODEL {m} {hf}@{rev}: {file_sha(f)}  {f.name}")
    lines += PROVENANCE + [""]
    for r in rels:
        lines.append(f"{sha(git('show', f'{head}:{r}'))}  {r}")
    LOCK.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(LOCK.read_text())
    print("lock id:", lock_id())


def base_text():
    out = []
    for l in LOCK.read_text(encoding="utf-8").splitlines():
        if l.startswith("CALIBRATION ") or l.startswith("DESIGN:"):
            break
        out.append(l)
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out) + "\n"


def lock_id():
    return sha(base_text().encode("utf-8"))[:16]


def verify(model=None):
    if not LOCK.exists():
        print("no lock file")
        return 1
    text = LOCK.read_text(encoding="utf-8").splitlines()
    head = next(l.split(": ")[1].strip() for l in text if l.startswith("git commit containing"))
    bad = []
    for l in text:
        if l.startswith("PREVIOUS LOCK: "):
            h, rel = l[len("PREVIOUS LOCK: "):].split("  ", 1)
            if sha((ROOT / rel).read_bytes()) != h:
                bad.append(f"previous lock changed: {rel}")
            continue
        if l.startswith("MODEL "):
            m = l.split()[1]
            if model == m:
                hf = MODELS[m]
                rev, fs = model_files(hf)
                h, name = l.split(": ", 1)[1].split("  ", 1)
                if f"@{rev}:" not in l or file_sha(next(f for f in fs if f.name == name)) != h:
                    bad.append(f"model file differs from lock: {m} {name}")
            continue
        if l.startswith("CALIBRATION "):
            m = l.split()[1].rstrip(":")
            h = l.split(": ")[-1].split("  ")[0]
            p = OUT / m / "calibration.json"
            if not p.exists() or sha(p.read_bytes()) != h:
                bad.append(f"calibration of {m} differs from the hash in the lock")
            continue
        if l.startswith("DESIGN:"):
            h = l.split(": ")[-1].split("  ")[0]
            p = OUT / "design.json"
            if not p.exists() or sha(p.read_bytes()) != h:
                bad.append("design differs from the hash in the lock")
            continue
        parts = l.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            continue
        h, rel = parts
        if sha(git("show", f"{head}:{rel}")) != h:
            bad.append(f"committed content differs from lock: {rel}")
        if git("status", "--porcelain", "--", rel).decode().strip():
            bad.append(f"working tree differs from commit: {rel}")
    for b in bad:
        print(b)
    print("lock verifies" if not bad else "LOCK DOES NOT VERIFY")
    return 1 if bad else 0


def append(line):
    with open(LOCK, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line)


def append_calibration(m):
    p = OUT / m / "calibration.json"
    if not p.exists():
        sys.exit("no calibration file")
    if f"CALIBRATION {m}:" in LOCK.read_text(encoding="utf-8"):
        sys.exit("calibration already appended")
    if (OUT / m / "tierA").exists():
        sys.exit("test records exist for this model: refusing")
    append(f"CALIBRATION {m}: (appended {datetime.now(timezone.utc).isoformat()}, before any test problem of {m}): {sha(p.read_bytes())}  "
           f"research_v2/v3_runs/{m}/calibration.json")


def append_design():
    p = OUT / "design.json"
    if not p.exists():
        sys.exit("no design file")
    if "DESIGN:" in LOCK.read_text(encoding="utf-8"):
        sys.exit("design already appended")
    if any((OUT / m / "tierA").exists() for m in MODELS):
        sys.exit("test records exist: refusing")
    d = json.loads(p.read_text())
    append(f"DESIGN: (appended {datetime.now(timezone.utc).isoformat()}, before any test problem; dropped arms {d.get('dropped_arms')}, n_test {d.get('n_test')}): "
           f"{sha(p.read_bytes())}  research_v2/v3_runs/design.json")


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--lock-id" in a:
        if not LOCK.exists():
            sys.exit(1)
        print(lock_id())
    elif "--verify-model" in a:
        sys.exit(verify(a[a.index("--verify-model") + 1]))
    elif "--verify" in a:
        sys.exit(verify())
    elif "--append-calibration" in a:
        append_calibration(a[a.index("--append-calibration") + 1])
    elif "--append-design" in a:
        append_design()
    elif "--amend" in a:
        create((a[a.index("--amend") + 1], RV / a[a.index("--preserved") + 1]))
    else:
        create()
