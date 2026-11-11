"""Write a dated SHA-256 lock of a protocol (PILOT_SPEC.md section 11 / STAGE1_SPEC.md section 9).

Usage: python research_v2/tools/lock_protocol.py [--stage1] [--draft]
  (default)  the 2026-10-22 pilot lock -> PROTOCOL_LOCK.txt
  --stage1   the Stage 1 lock             -> STAGE1_PROTOCOL_LOCK.txt
  --stage2   the Stage 2 held-out lock    -> STAGE2_PROTOCOL_LOCK.txt
The lock file is meant to be shared PRIVATELY with the coauthors. It is not a public pre-registration.
"""

import hashlib
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

RV = Path(__file__).resolve().parents[1]
ROOT = RV.parent
HIST_SRC = [ROOT / "src" / f for f in ("dynamic_linear.py", "dynamic_model.py", "routing.py", "fragility.py")]
PILOT_FILES = ([RV / "PILOT_SPEC.md", RV / "run_pilot.py", RV / "analyze_pilot.py", RV / "pilot_sample.json"]
               + sorted((RV / "rv2").glob("*.py")) + HIST_SRC
               + [ROOT / "scripts" / f for f in ("gsm8k_gate_check.py", "probe_gsm8k_routing.py",
                                                 "gsm8k_random_matched_budget_n930.py")])
STAGE1_FILES = ([RV / "STAGE1_SPEC.md", RV / "run_stage1.py", RV / "stage1_gates.py", RV / "analyze_stage1.py",
                 RV / "stage1_sample.json", RV / "tests" / "run_stage1_tests.py", RV / "tests" / "mutation_check_stage1.py",
                 RV / "tests" / "run_cpu_tests.py", RV / "tests" / "test_analyze_stage1.py"]
                + sorted((RV / "rv2").glob("*.py")) + HIST_SRC
                + [ROOT / "scripts" / f for f in ("gsm8k_gate_check.py", "probe_gsm8k_routing.py")])

STAGE2_FILES = ([RV / "STAGE2_SPEC.md", RV / "run_stage2.py", RV / "analyze_stage2.py", RV / "stage2_sample.json",
                 RV / "stage1_runtime.py", RV / "tools" / "check_heldout_clean.py", RV / "tools" / "design_power_stage2.py",
                 RV / "tools" / "verify_checkpoint.py", RV / "validation_logs" / "stage2_heldout_clean_check.json",
                 RV / "validation_logs" / "stage2_design_power.json", RV / "stage1_analysis" / "stage1_report.json",
                 RV / "STAGE1_REPORT.md", RV / "tests" / "run_stage2_tests.py", RV / "tests" / "test_analyze_stage2.py",
                 RV / "tests" / "mutation_check_stage2.py", RV / "tests" / "mutation_check_a2.py"]
                + STAGE1_FILES)

STAGE2_PROVENANCE = [
    "Private dated hash for coauthors. NOT a public pre-registration.",
    "PROVENANCE: this is a PROTOCOL-LOCKED HELD-OUT VALIDATION of hypotheses that are EXPLORATORY IN ORIGIN. The five hypotheses (STAGE2_SPEC.md",
    "section 1) were drawn from the Stage 1 results on the 930-problem pool (STAGE1_REPORT.md), which itself followed the 2026-10-22 pilot and",
    "earlier routing experiments on that pool. The 389 test problems evaluated here are the complement of the pool; no generated-answer correctness",
    "outcome was retained for any of them, but they are NOT completely untouched: two of them (test indices 4, 5) appeared in an 8-problem console probe and 47 (test index < 200,",
    "including those two) had their gold solutions token-labelled by a teacher-forced FP16/INT4 argmax comparison (disclosed in the spec;",
    "a sensitivity analysis excluding them is pre-declared). This lock is created before ANY GPU use in Stage 2 and before any held-out",
    "problem is evaluated. Design power from Stage 1 data is low for H1-H3 (STAGE2_SPEC.md 6.7); non-significant results will be reported as inconclusive."]

PROVENANCE = {
    False: ["Private dated hash for coauthors. NOT a public pre-registration."],
    True: ["Private dated hash for coauthors. NOT a public pre-registration.",
           "PROVENANCE: this protocol is EXPLORATORY IN ORIGIN. It is informed by earlier looks at these data: the 2026-10-22 pilot",
           "(flip oracle diverged from FP16 on 50/50 train problems, median first divergence token 20), the historical outcomes on the",
           "930-problem test pool (FP16 336 / NF4 243 solved), several routing comparisons run on that pool, and 30 of its problems",
           "re-run under several policies in the pilot. The 930-problem evaluation is therefore an exploratory replication with",
           "pre-declared estimands on data that have been examined before, not an independent confirmation. The 389 test problems",
           "outside the pool are reserved for a separately reported validation and are not touched in Stage 1."]}


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, check=True).stdout


def main():
    """Hashes the COMMITTED content (git show HEAD:path), so the lock does not depend on how a checkout renders
    line endings. A binding lock requires every locked file to be committed and unchanged in the working tree;
    commit first, then lock."""
    draft = "--draft" in sys.argv
    stage1 = "--stage1" in sys.argv
    stage2 = "--stage2" in sys.argv
    files = STAGE2_FILES if stage2 else STAGE1_FILES if stage1 else PILOT_FILES
    head = git("rev-parse", "HEAD").decode().strip()
    rels = [f.relative_to(ROOT).as_posix() for f in files if f.exists()]
    missing = [f.relative_to(ROOT).as_posix() for f in files if not f.exists()]
    if missing:
        sys.exit(f"locked files missing: {missing}")
    dirty = git("status", "--porcelain", "--", *rels).decode().strip()
    if dirty and not draft:
        sys.exit(f"Refusing binding lock: locked files are uncommitted or modified:\n{dirty}")
    label = ("research_v2 Stage 2 (protocol-locked held-out validation of the Stage 1 token-source x KV-source experiment)" if stage2
             else "research_v2 Stage 1 (token-source x KV-source)" if stage1 else "research_v2 pilot")
    lines = [f"{'DRAFT (not binding) ' if draft else ''}protocol lock -- {label}",
             f"utc: {datetime.now(timezone.utc).isoformat()}",
             f"git commit containing the locked files: {head}",
             "Hashes are SHA-256 of the committed file contents (git show <commit>:<path>)."] + (STAGE2_PROVENANCE if stage2 else PROVENANCE[stage1]) + [""]
    for rel in rels:
        lines.append(f"{hashlib.sha256(git('show', f'{head}:{rel}')).hexdigest()}  {rel}")
    if dirty:
        lines += ["", "WARNING (draft only): working tree differs from the commit for:", dirty]
    name = ("STAGE2_PROTOCOL_LOCK" if stage2 else "STAGE1_PROTOCOL_LOCK" if stage1 else "PROTOCOL_LOCK") + ("_DRAFT" if draft else "") + ".txt"
    out = RV / name
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out.read_text())
    print("sha256 of lock file:", hashlib.sha256(out.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
