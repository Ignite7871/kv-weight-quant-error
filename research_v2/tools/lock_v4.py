"""Write / verify the dated SHA-256 lock of v4 (PROTOCOL_V4.md). Private dated hash for the coauthors; NOT a public pre-registration.

  python research_v2/tools/lock_v4.py                         create V4_PROTOCOL_LOCK.txt from COMMITTED content (commit first, then lock)
  python research_v2/tools/lock_v4.py --verify                exit 0 iff every locked file matches the commit and is unmodified in the working tree,
                                                              the previous locks are unchanged, every appended hash matches its file, and every
                                                              amendment's preserved lock and deviation-log row are in place
  python research_v2/tools/lock_v4.py --lock-id               the lock id: first 16 hex of the SHA-256 of the lock text before any appended line
  python research_v2/tools/lock_v4.py --append-gm             append the hash of v4_runs/gm.json (GM passed; once, before Step 0)
  python research_v2/tools/lock_v4.py --append-flag M         append the BF16 flag of M (v4_runs/M/precision.json; G3 failed in FP16, before calibration)
  python research_v2/tools/lock_v4.py --append-calibration T  append the hash of v4_runs/T/calibration.json (T = model; before its first test problem)
  python research_v2/tools/lock_v4.py --append-step0          append the hash and decision of v4_runs/step0.json (before any non-Step-0 model)
  python research_v2/tools/lock_v4.py --append-design         append the hash of v4_runs/design.json (G6 design; before the first new model)
  python research_v2/tools/lock_v4.py --amend REV "TEXT"      pre-data amendment REV (for example r5): commit first, then re-lock. The old lock is kept at
                                                              v4/lock_history/, an AMENDMENT line enters the hashed text (so the lock id changes), and a row
                                                              linking the two ids goes to v4/DEVIATION_LOG_V4.md. Refused once any appended line or study
                                                              record exists.
  python research_v2/tools/lock_v4.py --deviation ID "TEXT"   (V4-D1) post-data deviation ID (for example D1), decided by the corresponding author: commit the
                                                              changed locked files first. Writes v4/deviations/V4-<ID>.json with the new SHA-256 of every
                                                              locked file that differs from the lock (and from earlier deviations), the commit and the
                                                              reason, and adds a row to v4/DEVIATION_LOG_V4.md. The lock id does not change. Refused before
                                                              any study record exists (use --amend then). Commit the record, then on the pod:
  python research_v2/tools/lock_v4.py --append-deviation ID   append the record's hash to the lock (before any further GPU use). From then on --verify accepts
                                                              each listed file at its deviation hash only, and GM may be appended once more for that
                                                              deviation (v4_runs/gm_<ID>.json, line "GM <ID>:").

Model weights are not hashed here (they live on the pod): the lock records the pinned revision and each shard's HF-reported SHA-256 from
v4/V4_MODEL_MANIFEST.json, and G7 re-hashes the downloaded shards against them. V4_RUNS (environment) relocates v4_runs (the pod uses
/workspace/v4_runs).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

RV = Path(__file__).resolve().parents[1]
ROOT = RV.parent
LOCK = RV / "V4_PROTOCOL_LOCK.txt"
OUT = Path(os.environ.get("V4_RUNS", RV / "v4_runs"))
V4 = RV / "v4"
PREV_LOCKS = [RV / "V3_PROTOCOL_LOCK.txt", RV / "E1_PROTOCOL_LOCK.txt", RV / "E1L_PROTOCOL_LOCK.txt"]
APPENDIX = ("GM:", "FLAG ", "CALIBRATION ", "STEP0:", "DESIGN:", "DEVIATION ", "GM ")
DEV_DIR = V4 / "deviations"                       # V4-D1: post-data deviation records (committed; their hashes are appended to the lock)
DEV_ID_RX = re.compile(r"D[0-9]+")
HISTORY = V4 / "lock_history"
DEVIATION_LOG = V4 / "DEVIATION_LOG_V4.md"       # not a locked file: it keeps growing after the lock (section 13)
AMENDMENT_RX = re.compile(r"AMENDMENT (\S+): from lock ([0-9a-f]{16}) \(commit ([0-9a-f]{40})\); old lock sha256 ([0-9a-f]{64}) preserved at (\S+); (.*)")
RECORD_DIRS = ("calib", "test", "gates")
RECORD_FILES = ("gm.json", "gm_test_rv4.json", "gm_mutation.json", "step0.json", "design.json", "g6_measure.json", "g6_probe.json")


def locked_files():
    v4 = [V4 / "PROTOCOL_V4.md", V4 / "phase2" / "PHASE2_LITERATURE.md", V4 / "V4_MODEL_MANIFEST.json", V4 / "v4_sample.json", V4 / "V4_PACKAGES.txt",
          V4 / "rv4.py", V4 / "v4_ops.py", V4 / "run_v4.py", V4 / "analyze_v4.py", V4 / "recompute_v4_independent.py",
          V4 / "RECOMPUTATION_TOLERANCES_V4.md", V4 / "plan" / "v4_plan_numbers.py", V4 / "plan" / "v4_plan_numbers.json",
          V4 / "cloud" / "pod_setup.sh", V4 / "cloud" / "RUNBOOK_V4.md", V4 / "v3_records_manifest.json"]   # r8b: the v3 records manifest
    v4 += sorted((V4 / "tests").glob("*.py")) + sorted((V4 / "validation").glob("*.json"))
    tools = [RV / "tools" / f for f in ("lock_v4.py", "v4_secret_scan.py", "v4_remote.py", "v4_build_manifest.py", "v4_gpu_mutation_check.py",
                                        "v3_repair_torn_line.py", "v4_v3_records.py")]
    imported = (sorted((RV / "rv3").glob("*.py")) + sorted((RV / "rv2").glob("*.py"))
                + [RV / f for f in ("run_controls.py", "run_stage1.py", "stage1_runtime.py", "e1l_support.py", "analyze_v3.py", "e1_sample.json")]
                + [RV / "v3" / "phase8" / "analyze_x13.py", RV / "v3" / "phase5" / "v3_report_locked.json"]
                + [ROOT / "src" / f for f in ("dynamic_linear.py", "dynamic_model.py", "routing.py", "fragility.py")]
                + [ROOT / "scripts" / f for f in ("gsm8k_gate_check.py", "probe_gsm8k_routing.py")])
    return v4 + tools + imported


PROVENANCE = [
    "Private dated hash for coauthors. NOT a public pre-registration.",
    "PROVENANCE: v4 is EXPLORATORY IN ORIGIN: H1-H5 come from the v3 locked results and X13. The 100 test ids and 50 calibration ids are the first",
    "of the 200 and 100 of the locked E1/v3 sample. No v4 outcome exists at lock time. This lock is NOT an authorization to run: the corresponding author creates one",
    "file per model, v4_runs/AUTHORIZED_V4_<model>.txt (and AUTHORIZED_V4_GM.txt), containing 'PROTOCOL_V4',",
    "the lock id printed by `tools/lock_v4.py --lock-id` and the model id."]


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, check=True).stdout


def sha(b):
    return hashlib.sha256(b).hexdigest()


def sha_lf(b):
    """The hash of a text file, whatever line endings a checkout gave it."""
    return sha(b.replace(b"\r\n", b"\n"))


def file_sha(p):
    """r6: the hash of a text file on disk (a previous lock, an appended run file), the same on a Windows checkout (CRLF) and on the pod (LF).
    Before r6 these were raw bytes, so a lock written on the laptop could not verify on the pod."""
    return sha_lf(Path(p).read_bytes())


def lock_text(amendments=()):
    """The text of a binding lock from the committed tree. Exits, with nothing written, if a locked file is missing or uncommitted."""
    head = git("rev-parse", "HEAD").decode().strip()
    rels = [f.relative_to(ROOT).as_posix() for f in locked_files()]
    missing = [r for r in rels if not (ROOT / r).exists()]
    if missing:
        sys.exit(f"locked files missing: {missing}")
    dirty = git("status", "--porcelain", "--", *rels).decode().strip()
    if dirty:
        sys.exit(f"Refusing binding lock: uncommitted or modified:\n{dirty}")
    lines = ["protocol lock -- research_v2 v4 (scale study of weight-induced cache error on rented cloud GPUs)",
             f"utc: {datetime.now(timezone.utc).isoformat()}",
             f"git commit containing the locked files: {head}", "Hashes are SHA-256 of the committed file contents (git show <commit>:<path>)."]
    for p in PREV_LOCKS:
        lines.append(f"PREVIOUS LOCK: {file_sha(p)}  {p.relative_to(ROOT).as_posix()}")
    lines += list(amendments)
    man = json.loads(git("show", f"{head}:research_v2/v4/V4_MODEL_MANIFEST.json"))["models"]
    for m, e in man.items():
        for shard, s in sorted(e["shards"].items()):
            lines.append(f"MODEL {m} {e['hf_id']}@{e['revision']}: {s['sha256']}  {shard}")
    lines += PROVENANCE + [""]
    for r in rels:
        lines.append(f"{sha(git('show', f'{head}:{r}'))}  {r}")
    return "\n".join(lines) + "\n"


def create():
    if LOCK.exists():
        sys.exit("V4_PROTOCOL_LOCK.txt exists; a new lock needs a logged amendment (--amend)")
    LOCK.write_text(lock_text(), encoding="utf-8", newline="\n")
    print(LOCK.read_text(encoding="utf-8"))
    print("lock id:", lock_id())


def data_records():
    """Anything that shows the study has started: appended lock lines, GM / Step 0 / design / G6 files, or a model's gate, calibration or test files."""
    found = [l.split(":")[0] for l in LOCK.read_text(encoding="utf-8").splitlines() if l.startswith(APPENDIX)]
    found += [n for n in RECORD_FILES if (OUT / n).exists()]
    if OUT.exists():
        found += [(d / r).relative_to(OUT).as_posix() for d in sorted(OUT.iterdir()) if d.is_dir() for r in RECORD_DIRS if (d / r).exists()]
    return found


def committed_differences(text, head):
    """Locked hashes of a lock text that no longer match the files of its own commit."""
    bad = []
    for l in text.splitlines():
        parts = l.split("  ", 1)
        if len(parts) == 2 and len(parts[0]) == 64 and sha(git("show", f"{head}:{parts[1]}")) != parts[0]:
            bad.append(parts[1])
    return bad


def preserve(path, old_bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(old_bytes)


def amend(rev, text):
    """A pre-data amendment. Every check runs before anything is written, so a refusal leaves the old lock in place."""
    if not LOCK.exists():
        sys.exit("no lock to amend")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", rev) or not text.strip() or "\n" in text:
        sys.exit("--amend REV TEXT: REV is one word and TEXT one line")
    data = data_records()
    if data:
        sys.exit(f"Refusing a pre-data amendment: the study has started ({', '.join(data)}). A later change needs a logged deviation and an author decision.")
    old_bytes = LOCK.read_bytes()
    old_text = old_bytes.decode("utf-8")
    amendments = [l for l in old_text.splitlines() if l.startswith("AMENDMENT ")]
    if any(AMENDMENT_RX.fullmatch(l).group(1) == rev for l in amendments):
        sys.exit(f"amendment {rev} is already in the lock")
    old_id = lock_id()
    old_head = next(l.split(": ")[1].strip() for l in old_text.splitlines() if l.startswith("git commit containing"))
    bad = committed_differences(old_text, old_head)
    if bad:
        sys.exit(f"the old lock does not match its own commit {old_head}: {bad}")
    if not DEVIATION_LOG.exists():
        sys.exit(f"{DEVIATION_LOG.name} is missing")
    preserved = HISTORY / f"V4_PROTOCOL_LOCK_{old_id}.txt"
    if preserved.exists():
        sys.exit(f"{preserved.name} already exists")
    rel = preserved.relative_to(ROOT).as_posix()
    line = f"AMENDMENT {rev}: from lock {old_id} (commit {old_head}); old lock sha256 {sha_lf(old_bytes)} preserved at {rel}; {text.strip()}"
    new_text = lock_text(amendments + [line])
    preserve(preserved, old_bytes)
    LOCK.write_text(new_text, encoding="utf-8", newline="\n")
    new_id = lock_id()
    with open(DEVIATION_LOG, "a", encoding="utf-8", newline="\n") as f:
        f.write(f"| V4-{rev} (re-lock) | {datetime.now(timezone.utc).date().isoformat()} | pre-data | Lock {old_id} | **Re-lock:** from lock {old_id} to lock {new_id}. "
                f"Old lock preserved unchanged at `{rel}`. {text.strip()} | Amendment {rev}, made before any cloud action | No study data existed |\n")
    print(new_text)
    print("lock id:", new_id)


def base_text(path=None):
    out = []
    for l in (path or LOCK).read_text(encoding="utf-8").splitlines():
        if l.startswith(APPENDIX):
            break
        out.append(l)
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out) + "\n"


def lock_id(path=None):
    return sha(base_text(path).encode("utf-8"))[:16]


def appendix_file(line):
    if line.startswith("GM:"):
        return OUT / "gm.json"
    if line.startswith("GM "):
        return OUT / f"gm_{line.split()[1].rstrip(':')}.json"
    if line.startswith("DEVIATION "):
        return deviation_record(line.split()[1].rstrip(":"))
    if line.startswith("STEP0:"):
        return OUT / "step0.json"
    if line.startswith("DESIGN:"):
        return OUT / "design.json"
    tag = line.split()[1].rstrip(":")
    if line.startswith("FLAG "):
        return OUT / tag / "precision.json"
    return OUT / tag / "calibration.json"


def verify_amendments(lines):
    """Each AMENDMENT line: the preserved lock has the recorded hash and id, and the deviation log has a row from that id to the next one."""
    rows = [AMENDMENT_RX.fullmatch(l) for l in lines if l.startswith("AMENDMENT ")]
    if None in rows:
        return ["an AMENDMENT line is malformed"]
    bad = []
    log = DEVIATION_LOG.read_text(encoding="utf-8") if DEVIATION_LOG.exists() else ""
    to_ids = [r.group(2) for r in rows[1:]] + [lock_id()]
    for r, to_id in zip(rows, to_ids):
        rev, from_id, _, h, rel = r.groups()[:5]
        p = ROOT / rel
        if not p.exists() or sha_lf(p.read_bytes()) != h:
            bad.append(f"preserved lock of amendment {rev} is missing or changed: {rel}")
        elif lock_id(p) != from_id:
            bad.append(f"preserved lock of amendment {rev} does not have the id {from_id}")
        if f"from lock {from_id} to lock {to_id}" not in log:
            bad.append(f"{DEVIATION_LOG.name} has no row from lock {from_id} to lock {to_id}")
    return bad


def deviation_record(dev_id):
    return DEV_DIR / f"V4-{dev_id}.json"


def appended_deviations(lines=None):
    """IDs of the DEVIATION lines appended to the lock, in order."""
    lines = lines if lines is not None else (LOCK.read_text(encoding="utf-8").splitlines() if LOCK.exists() else [])
    return [l.split()[1].rstrip(":") for l in lines if l.startswith("DEVIATION ")]


def active_deviation():
    """The latest appended deviation, or None (V4-D1)."""
    d = appended_deviations()
    return d[-1] if d else None


def base_hashes(text=None):
    """rel -> SHA-256 of the locked files in the lock text (before any deviation)."""
    out = {}
    for l in (text or LOCK.read_text(encoding="utf-8")).splitlines():
        parts = l.split("  ", 1)
        if len(parts) == 2 and re.fullmatch(r"[0-9a-f]{64}", parts[0]):
            out[parts[1]] = parts[0]
    return out


def deviation_hashes(ids):
    """rel -> (sha256, commit, id) after applying the deviation records ids in order."""
    eff = {}
    for i in ids:
        rec = json.loads(deviation_record(i).read_text(encoding="utf-8"))
        for rel, h in rec["files"].items():
            eff[rel] = (h, rec["commit"], i)
    return eff


def deviation(dev_id, text):
    """V4-D1: a post-data deviation decided by the corresponding author. Every check runs before anything is written."""
    if not LOCK.exists():
        sys.exit("no lock")
    if not DEV_ID_RX.fullmatch(dev_id) or not text.strip() or "\n" in text:
        sys.exit("--deviation ID TEXT: ID is D<number> and TEXT one line")
    if not data_records():
        sys.exit("no study record exists: a change now is a pre-data amendment (--amend)")
    rec_path = deviation_record(dev_id)
    if rec_path.exists():
        sys.exit(f"{rec_path.name} already exists")
    known = sorted(DEV_DIR.glob("V4-D*.json")) if DEV_DIR.exists() else []
    earlier = [json.loads(q.read_text(encoding="utf-8"))["id"] for q in known]
    eff = dict(base_hashes(base_text()))
    eff.update({rel: v[0] for rel, v in deviation_hashes(earlier).items()})
    head = git("rev-parse", "HEAD").decode().strip()
    files = {}
    for rel, h in eff.items():
        if git("status", "--porcelain", "--", rel).decode().strip():
            sys.exit(f"{rel} has uncommitted changes: commit the deviation's files first")
        now = sha(git("show", f"HEAD:{rel}"))
        if now != h:
            files[rel] = now
    if not files:
        sys.exit("no locked file differs from the lock: nothing to record")
    if not DEVIATION_LOG.exists():
        sys.exit(f"{DEVIATION_LOG.name} is missing")
    rec = {"id": dev_id, "lock_id": lock_id(), "commit": head, "reason": text.strip(), "utc": datetime.now(timezone.utc).isoformat(),
           "earlier_deviations": earlier, "files": files}
    DEV_DIR.mkdir(parents=True, exist_ok=True)
    rec_path.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8", newline="\n")
    listing = "; ".join(f"{rel} {h}" for rel, h in sorted(files.items()))
    with open(DEVIATION_LOG, "a", encoding="utf-8", newline="\n") as f:
        f.write(f"| V4-{dev_id} | {datetime.now(timezone.utc).date().isoformat()} | post-data deviation | Lock {lock_id()} | **Deviation {dev_id} "
                f"(corresponding author's decision):** {text.strip()} Changed locked files at commit {head}, SHA-256: {listing} | Record "
                f"`{rec_path.relative_to(ROOT).as_posix()}`; appended to the lock on the pod before any further GPU use |\n")
    print(json.dumps(rec, indent=1))


def append_deviation(dev_id):
    """On the pod: append the committed deviation record's hash. The record must name this lock, and every listed file must be at its hash."""
    if not LOCK.exists():
        sys.exit("no lock")
    if dev_id in appended_deviations():
        sys.exit(f"DEVIATION {dev_id} is already appended; it is never replaced")
    p = deviation_record(dev_id)
    if not p.exists():
        sys.exit(f"{p} does not exist")
    rel_rec = p.relative_to(ROOT).as_posix()
    if git("status", "--porcelain", "--", rel_rec).decode().strip() or sha(git("show", f"HEAD:{rel_rec}")) != sha(p.read_bytes()):
        sys.exit(f"{p.name} is not committed at HEAD")
    rec = json.loads(p.read_text(encoding="utf-8"))
    if rec.get("id") != dev_id or rec.get("lock_id") != lock_id():
        sys.exit(f"{p.name} is not a record of deviation {dev_id} under lock {lock_id()}")
    if f"| V4-{dev_id} |" not in (DEVIATION_LOG.read_text(encoding="utf-8") if DEVIATION_LOG.exists() else ""):
        sys.exit(f"{DEVIATION_LOG.name} has no row V4-{dev_id}")
    for rel, h in rec["files"].items():
        if sha(git("show", f"HEAD:{rel}")) != h or git("status", "--porcelain", "--", rel).decode().strip():
            sys.exit(f"{rel} is not at the deviation's hash")
    append(f"DEVIATION {dev_id}: {file_sha(p)}  {rel_rec}  files {len(rec['files'])} utc {datetime.now(timezone.utc).isoformat()}")


def verify():
    if not LOCK.exists():
        print("no lock file")
        return 1
    text = LOCK.read_text(encoding="utf-8").splitlines()
    head = next(l.split(": ")[1].strip() for l in text if l.startswith("git commit containing"))
    bad = verify_amendments(text)
    devs = appended_deviations(text)
    log = DEVIATION_LOG.read_text(encoding="utf-8") if DEVIATION_LOG.exists() else ""
    for i in devs:
        p = deviation_record(i)
        if not p.exists():
            bad.append(f"deviation record missing: {p.name}")
            continue
        rec = json.loads(p.read_text(encoding="utf-8"))
        if rec.get("id") != i or rec.get("lock_id") != lock_id():
            bad.append(f"{p.name} is not a record of deviation {i} under this lock")
        if f"| V4-{i} |" not in log:
            bad.append(f"{DEVIATION_LOG.name} has no row V4-{i}")
    eff = deviation_hashes([i for i in devs if deviation_record(i).exists()])
    for l in text:
        if l.startswith("AMENDMENT "):
            continue
        if l.startswith("PREVIOUS LOCK: "):
            h, rel = l[len("PREVIOUS LOCK: "):].split("  ", 1)
            if file_sha(ROOT / rel) != h:
                bad.append(f"previous lock changed: {rel}")
            continue
        if l.startswith("MODEL "):
            continue
        if l.startswith(APPENDIX):
            p = appendix_file(l)
            h = l.split(": ", 1)[1].split()[0]
            if not p.exists() or file_sha(p) != h:
                bad.append(f"appended file differs from the hash in the lock: {p.name} ({l.split(':')[0]})")
            continue
        parts = l.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            continue
        h, rel = parts
        commit = head
        if rel in eff:                                                 # V4-D1: the file is held at its latest deviation hash only
            h, commit, _ = eff[rel]
        if sha(git("show", f"{commit}:{rel}")) != h:
            bad.append(f"committed content differs from lock: {rel}")
        if sha(git("show", f"HEAD:{rel}")) != h:
            bad.append(f"checked-out commit differs from lock: {rel}")
        if git("status", "--porcelain", "--", rel).decode().strip():
            bad.append(f"working tree differs from commit: {rel}")
    for b in bad:
        print(b)
    print("lock verifies" if not bad else "LOCK DOES NOT VERIFY")
    return 1 if bad else 0


def appended(prefix):
    return [l for l in LOCK.read_text(encoding="utf-8").splitlines() if l.startswith(prefix)] if LOCK.exists() else []


def append(line):
    with open(LOCK, "a", encoding="utf-8", newline="\n") as f:
        f.write(line + "\n")
    print(line)


def do_append(kind, tag=None):
    if not LOCK.exists():
        sys.exit("no lock")
    prefix = {"gm": "GM:", "step0": "STEP0:", "design": "DESIGN:", "flag": f"FLAG {tag}:", "cal": f"CALIBRATION {tag}:"}[kind]
    if kind == "gm" and active_deviation():                           # V4-D1: GM once more after each deviation, into gm_<ID>.json
        prefix = f"GM {active_deviation()}:"
    if appended(prefix):
        sys.exit(f"{prefix} already appended; it is never replaced")
    p = appendix_file(prefix + " x" if kind in ("gm", "step0", "design") else prefix)
    if not p.exists():
        sys.exit(f"{p} does not exist")
    d = json.loads(p.read_text(encoding="utf-8"))
    note = ""
    if kind == "gm" and not d.get("all_passed"):
        sys.exit("GM has not passed")
    if kind == "flag":
        if d.get("precision") != "bf16":
            sys.exit("precision.json does not record the BF16 flag")
        note = "BF16 rule (D6): G3 failed in FP16"
    if kind == "step0":
        if d.get("decision") not in ("pass", "author: proceed", "author: stop"):
            sys.exit("step0.json has no decision")
        note = f"decision {d['decision']}"
    if kind == "design":
        note = f"reductions {d.get('reductions')}"
    append(f"{prefix} {file_sha(p)}  {p.relative_to(OUT).as_posix()}  {note} utc {datetime.now(timezone.utc).isoformat()}")


def main():
    a = sys.argv[1:]
    if not a:
        create()
    elif a[0] == "--verify":
        sys.exit(verify())
    elif a[0] == "--lock-id":
        if not LOCK.exists():
            sys.exit(1)
        print(lock_id())
    elif a[0] == "--amend":
        if len(a) != 3:
            sys.exit(__doc__)
        amend(a[1], a[2])
    elif a[0] == "--deviation":
        if len(a) != 3:
            sys.exit(__doc__)
        deviation(a[1], a[2])
    elif a[0] == "--append-deviation":
        if len(a) != 2:
            sys.exit(__doc__)
        append_deviation(a[1])
    elif a[0] == "--append-gm":
        do_append("gm")
    elif a[0] == "--append-step0":
        do_append("step0")
    elif a[0] == "--append-design":
        do_append("design")
    elif a[0] == "--append-flag":
        do_append("flag", a[1])
    elif a[0] == "--append-calibration":
        do_append("cal", a[1])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
