"""Write / verify the dated SHA-256 lock of the E1 protocol (E1_SPEC.md section 11). Private dated hash for the coauthors; NOT a public pre-registration.

  python research_v2/tools/lock_e1.py            create E1_PROTOCOL_LOCK.txt from COMMITTED content (commit first, then lock)
  python research_v2/tools/lock_e1.py --verify   exit 0 iff every locked file still matches its recorded hash and is unmodified in the working tree
  python research_v2/tools/lock_e1.py --append-calibration   append the SHA-256 of e1_runs/e1_calibration.json (done once, after calibration, BEFORE any test problem is evaluated)
"""

import hashlib
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

RV = Path(__file__).resolve().parents[1]
ROOT = RV.parent
LOCK = RV / "E1_PROTOCOL_LOCK.txt"
HIST_SRC = [ROOT / "src" / f for f in ("dynamic_linear.py", "dynamic_model.py", "routing.py", "fragility.py")]
FILES = ([RV / "E1_SPEC.md", RV / "run_controls.py", RV / "analyze_controls.py", RV / "e1_sample.json", RV / "rv2" / "controls.py",
          RV / "tests" / "run_controls_tests.py", RV / "tests" / "mutation_check_controls.py", RV / "tests" / "test_analyze_controls.py",
          RV / "tools" / "lock_e1.py", RV / "validation_logs" / "e1_cpu_validation.json",
          RV / "run_stage1.py", RV / "stage1_runtime.py", RV / "stage1_sample.json", RV / "stage2_sample.json", RV / "pilot_sample.json"]
         + sorted((RV / "rv2").glob("*.py")) + HIST_SRC
         + [ROOT / "scripts" / f for f in ("gsm8k_gate_check.py", "probe_gsm8k_routing.py")])
EXTERNAL = [RV / "stage1_runs" / "eval" / "fp16.jsonl", RV / "stage2_runs" / "eval" / "fp16.jsonl",
            RV / "stage1_runs" / "eval" / "cache_only.jsonl", RV / "stage2_runs" / "eval" / "cache_only.jsonl"]
PROVENANCE = [
    "Private dated hash for coauthors. NOT a public pre-registration.",
    "PROVENANCE: E1 is EXPLORATORY IN ORIGIN. Its hypotheses follow from the limitations of Stages 1-2 (no magnitude-matched perturbation control) and from the",
    "full-text novelty review; all 1,319 GSM8K test problems were used in earlier work, but the control arms have no prior outcomes and the primary outcomes use no",
    "gold label. This lock is created before ANY E1 GPU use and before any E1 outcome exists. It is NOT an authorization to run: the E1 GPU evaluation needs",
    "further author approval (run_controls.py refuses without e1_runs/AUTHORIZED_BY_AUTHORS.txt)."]


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, check=True).stdout


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def create():
    head = git("rev-parse", "HEAD").decode().strip()
    rels = [f.relative_to(ROOT).as_posix() for f in FILES]
    missing = [r for r in rels if not (ROOT / r).exists()]
    if missing:
        sys.exit(f"locked files missing: {missing}")
    dirty = git("status", "--porcelain", "--", *rels).decode().strip()
    if dirty:
        sys.exit(f"Refusing binding lock: locked files are uncommitted or modified:\n{dirty}")
    lines = ["protocol lock -- research_v2 E1 (perturbation-matched cache controls)", f"utc: {datetime.now(timezone.utc).isoformat()}",
             f"git commit containing the locked files: {head}", "Hashes are SHA-256 of the committed file contents (git show <commit>:<path>)."] + PROVENANCE + [""]
    for r in rels:
        lines.append(f"{sha(git('show', f'{head}:{r}'))}  {r}")
    lines += ["", "EXTERNAL (untracked raw records; SHA-256 of the working files at lock time)"]
    for p in EXTERNAL:
        lines.append(f"{sha(p.read_bytes())}  {p.relative_to(ROOT).as_posix()}")
    LOCK.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(LOCK.read_text())
    print("sha256 of lock file:", sha(LOCK.read_bytes()))


def verify() -> int:
    if not LOCK.exists():
        print("no lock file")
        return 1
    text = LOCK.read_text(encoding="utf-8").splitlines()
    head = next(l.split(": ")[1].strip() for l in text if l.startswith("git commit containing"))
    bad = []
    external = False
    for l in text:
        if l.startswith("EXTERNAL"):
            external = True
            continue
        if l.startswith("CALIBRATION"):
            h = l.split(": ")[-1].split("  ")[0]
            cp = RV / "e1_runs" / "e1_calibration.json"
            if not cp.exists() or sha(cp.read_bytes()) != h:
                bad.append("calibration constants differ from the hash recorded in the lock")
            continue
        parts = l.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            continue
        h, rel = parts
        if external:
            if sha((ROOT / rel).read_bytes()) != h:
                bad.append(f"external file changed: {rel}")
        else:
            if sha(git("show", f"{head}:{rel}")) != h:
                bad.append(f"committed content differs from lock: {rel}")
            if git("status", "--porcelain", "--", rel).decode().strip():
                bad.append(f"working tree differs from commit: {rel}")
    for b in bad:
        print(b)
    print("lock verifies" if not bad else "LOCK DOES NOT VERIFY")
    return 1 if bad else 0


def append_calibration():
    p = RV / "e1_runs" / "e1_calibration.json"
    if not p.exists():
        sys.exit("no calibration file")
    line = f"CALIBRATION (appended {datetime.now(timezone.utc).isoformat()}, before any test problem was evaluated): {sha(p.read_bytes())}  research_v2/e1_runs/e1_calibration.json"
    with open(LOCK, "a", encoding="utf-8") as f:
        f.write("\n" + line + "\n")
    print(line)


if __name__ == "__main__":
    if "--verify" in sys.argv:
        sys.exit(verify())
    if "--append-calibration" in sys.argv:
        append_calibration()
    else:
        create()
