"""Text-overlap check between the Research V2 manuscript and the earlier ARR #283 / TMLR #13386 manuscripts (CPU, local).
Extracts text with pdftotext, strips reference lists and line numbers, and reports shared word n-grams (default n=8) and the longest shared passages.
  venv/Scripts/python.exe research_v2/paper/audit/text_overlap.py [path/to/new.pdf]
Writes paper/audit/TEXT_OVERLAP_RESULTS.md. Not a plagiarism detector; a transparency aid for the submission-history disclosure."""

import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]                       # entropy-dps-project (audit -> paper_journal -> research_v2 -> project)
NEW = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "main.pdf"  # paper_journal/main.pdf
OLD = {
    "ARR #283 (July 27 PDF, likely submitted)": ROOT / "archive_papers_2026-10-28" / "paper_acl" / "Dynamic_Precision_Routing_via_Distributional_Fragility_in_Large_Language_Models.pdf",
    "ARR-format revision (Sept 30 main.pdf)": ROOT / "archive_papers_2026-10-28" / "paper_acl" / "main.pdf",
    "TMLR #13386 package main.pdf": ROOT / "archive_papers_2026-10-28" / "paper_tmlr" / "main.pdf",
}
N = 8


def text(pdf):
    t = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
    parts = re.split(r"\n\s*References\s*\n", t, maxsplit=1)                    # drop the reference list, keep appendices after it (Research V2 layout)
    t = parts[0]
    if len(parts) == 2 and "Implementation and verification" in parts[1]:
        t += parts[1][parts[1].index("Implementation and verification"):]
    t = re.sub(r"(?m)^\s*\d{3}\s*$", " ", t)                                     # ACL line numbers
    t = re.sub(r"[^A-Za-z0-9\s\-\.%]", " ", t.lower())
    return t.split()


def shingles(w, n=N):
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def passages(a, b, n=N):
    """Maximal shared word runs (>= n words) between token lists a and b."""
    sb = shingles(b, n)
    runs, i = [], 0
    while i < len(a) - n + 1:
        if tuple(a[i:i + n]) in sb:
            j = i + n
            while j < len(a) and tuple(a[j - n + 1:j + 1]) in sb:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


new = text(NEW)
out = [f"# Text overlap: Research V2 manuscript vs earlier manuscripts", "",
       f"New text: `{NEW.name}` ({len(new)} words after dropping references). Shared word {N}-grams (lower-cased, punctuation stripped). Local check only.", "",
       "| Earlier manuscript | words | shared 8-grams | share of V2 8-grams | words in shared runs (V2) | % of V2 words | longest run (words) |", "|---|---|---|---|---|---|---|"]
detail = []
for name, p in OLD.items():
    old = text(p)
    sh = shingles(new) & shingles(old)
    runs = passages(new, old)
    covered = sum(j - i for i, j in runs)
    longest = max(((j - i, i, j) for i, j in runs), default=(0, 0, 0))
    out.append(f"| {name} | {len(old)} | {len(sh)} | {100 * len(sh) / max(1, len(shingles(new))):.1f}% | {covered} | {100 * covered / len(new):.1f}% | {longest[0]} |")
    detail.append((name, sorted(runs, key=lambda r: r[0] - r[1])[:12]))
out += ["", "## Longest shared runs (V2 wording)", ""]
for name, rs in detail:
    out.append(f"**{name}**")
    for i, j in rs:
        out.append(f"- ({j - i} words) \"{' '.join(new[i:j])}\"")
    out.append("")
(HERE / "JOURNAL_TEXT_OVERLAP_RESULTS.md").write_text("\n".join(out), encoding="utf-8")
print("\n".join(out[:12]))
