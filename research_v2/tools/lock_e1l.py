"""Write / verify the dated SHA-256 lock of the Llama E1 Tier A replication (E1L_SPEC.md). Private dated hash for the coauthors; NOT a public pre-registration.
  python research_v2/tools/lock_e1l.py            create E1L_PROTOCOL_LOCK.txt from COMMITTED content (commit first, then lock)
  python research_v2/tools/lock_e1l.py --verify   exit 0 iff every locked file matches and is unmodified in the working tree
  python research_v2/tools/lock_e1l.py --append-calibration   append the hash of e1l_runs/e1l_calibration.json (once, BEFORE any Llama test problem is evaluated)
"""
import hashlib
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

RV = Path(__file__).resolve().parents[1]
ROOT = RV.parent
LOCK = RV / "E1L_PROTOCOL_LOCK.txt"
QWEN_LOCK = RV / "E1_PROTOCOL_LOCK.txt"
FILES = ([RV / "E1L_SPEC.md", RV / "run_e1l.py", RV / "analyze_e1l.py", RV / "e1l_support.py", RV / "e1l_sample.json",
          RV / "tests" / "run_e1l_tests.py", RV / "tests" / "mutation_check_e1l.py", RV / "tools" / "e1l_gpu_mutation_check.py", RV / "tools" / "lock_e1l.py",
          RV / "validation_logs" / "e1l_cpu_validation.json",
          RV / "E1_SPEC.md", RV / "run_controls.py", RV / "analyze_controls.py", RV / "e1_sample.json", RV / "rv2" / "controls.py",
          RV / "tests" / "run_controls_tests.py", RV / "tests" / "mutation_check_controls.py", RV / "run_stage1.py", RV / "stage1_runtime.py",
          RV / "stage1_sample.json", RV / "pilot_sample.json"]
         + sorted((RV / "rv2").glob("*.py")) + [ROOT / "src" / f for f in ("dynamic_linear.py", "dynamic_model.py", "routing.py", "fragility.py")]
         + [ROOT / "scripts" / f for f in ("gsm8k_gate_check.py", "probe_gsm8k_routing.py")])
PROVENANCE = [
    "Private dated hash for coauthors. NOT a public pre-registration.",
    "PROVENANCE: the Llama Tier A replication is EXPLORATORY IN ORIGIN (it follows the completed Qwen E1; its design is the Qwen E1 Tier A design applied to a second",
    "model family). The 200 test ids and 100 calibration ids are copied from the locked Qwen E1 sample; no Llama outcome existed at lock time except the 100-problem",
    "feasibility pilot (accuracy/gates only; no perturbation-matched outcome). This lock is created before ANY Llama-E1 GPU use. It is NOT an authorization to run:",
    "the corresponding author must create research_v2/e1l_runs/AUTHORIZED_BY_AUTHORS_LLAMA.txt (non-empty, containing the word 'Llama') after reviewing it."]


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, check=True).stdout


def sha(b):
    return hashlib.sha256(b).hexdigest()


def create():
    head = git("rev-parse", "HEAD").decode().strip()
    rels = [f.relative_to(ROOT).as_posix() for f in FILES]
    missing = [r for r in rels if not (ROOT / r).exists()]
    if missing:
        sys.exit(f"locked files missing: {missing}")
    dirty = git("status", "--porcelain", "--", *rels).decode().strip()
    if dirty:
        sys.exit(f"Refusing binding lock: uncommitted or modified:\n{dirty}")
    lines = ["protocol lock -- research_v2 Llama E1 Tier A replication", f"utc: {datetime.now(timezone.utc).isoformat()}",
             f"git commit containing the locked files: {head}", "Hashes are SHA-256 of the committed file contents (git show <commit>:<path>).",
             f"Qwen E1 lock file (whose hash pins the Qwen protocol): {sha(QWEN_LOCK.read_bytes())}  research_v2/E1_PROTOCOL_LOCK.txt"] + PROVENANCE + [""]
    for r in rels:
        lines.append(f"{sha(git('show', f'{head}:{r}'))}  {r}")
    LOCK.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(LOCK.read_text())
    print("sha256 of lock file:", sha(LOCK.read_bytes()))


def verify():
    if not LOCK.exists():
        print("no lock file")
        return 1
    text = LOCK.read_text(encoding="utf-8").splitlines()
    head = next(l.split(": ")[1].strip() for l in text if l.startswith("git commit containing"))
    bad = []
    for l in text:
        if l.startswith("CALIBRATION"):
            h = l.split(": ")[-1].split("  ")[0]
            cp = RV / "e1l_runs" / "e1l_calibration.json"
            if not cp.exists() or sha(cp.read_bytes()) != h:
                bad.append("calibration constants differ from the hash recorded in the lock")
            continue
        if l.startswith("Qwen E1 lock file"):
            if sha(QWEN_LOCK.read_bytes()) != l.split(": ")[1].split("  ")[0]:
                bad.append("the Qwen E1 lock file changed")
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


def append_calibration():
    p = RV / "e1l_runs" / "e1l_calibration.json"
    if not p.exists():
        sys.exit("no calibration file")
    line = f"CALIBRATION (appended {datetime.now(timezone.utc).isoformat()}, before any Llama test problem was evaluated): {sha(p.read_bytes())}  research_v2/e1l_runs/e1l_calibration.json"
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
