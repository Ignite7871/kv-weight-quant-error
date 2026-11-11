"""Prints every occurrence, with context, of phrases that the manuscript must use only in a qualified or negated way, and checks that required statements exist.
Human review of the printed contexts is the point; the script only gathers them and fails if a required statement is missing.
  venv/Scripts/python.exe research_v2/paper/audit/claims_guard.py
Writes paper/audit/MAIN_TEXT_CLAIMS_GUARD.md."""

import re
import sys
from pathlib import Path

P = Path(__file__).resolve().parents[1]  # paper_journal
files = ["00_abstract", "01_introduction", "02_related", "03_intervention", "04_design", "05_results", "06_discussion", "07_limitations", "08_conclusion"]
SEC = sys.argv[1] if len(sys.argv) > 1 else "sections"
OUTNAME = sys.argv[2] if len(sys.argv) > 2 else "JOURNAL_CLAIMS_GUARD.md"
text = {f: (P / SEC / f"{f}.tex").read_text(encoding="utf-8") for f in files}
WATCH = [r"super-?additiv", r"pre-?regist", r"untouched", r"equivalen", r"FP16[ -]cache", r"more important", r"matters more", r"dominant", r"mediat",
         r"accounts? for", r"\d+(\.\d+)?\?%\s+of the (loss|gap|accuracy)", r"generali[sz]", r"storage quantization", r"proves?\b", r"confirm", r"replicat", r"significan", r"safer", r"universal", r"explains?", r"caus", r"speed", r"faster", r"architecture"]
REQUIRED = {
    "two pathways defined (token source / cache source)": r"token source.*cache source|token-source",
    "no super-additivity claim (explicit)": r"neither support super-additivity|not claim the interaction replicated|does not replicate",
    "oracle: close to, equivalence not firmly established": r"not firmly established",
    "held-out low power stated": r"[Pp]ower",
    "dtype vs value: FP16-typed but NF4-written": r"FP16-typed",
    "two models and one benchmark": r"Two small models.*one benchmark",
    "exploratory vs held-out vs pooled tags": r"tagE.*tagC|tagC.*tagE",
    "not completely untouched": r"not completely untouched",
    "private lock, not pre-registration": r"not\} a public pre-registration|not (a )?public pre-registration|not public pre-registrations",
    "raw and Holm both reported": r"raw and Holm",
    "no ranking of pathways": r"no basis for ranking|either pathway matters|do not rank the pathways",
    "no universal-safety claim": r"safer than random noise in general|NF4 perturbations are safer",
    "signs matter on Llama / not irrelevant": r"signs of the perturbation have no effect|signs of the perturbation are irrelevant",
    "size alone does not explain": r"size alone|size fully explains",
    "architecture not established as cause": r"architecture causes the difference|architecture alone",
    "non-significance is not equivalence": r"not inferred from non-significance|nonsignificant|non-significant contrast",
    "no speed claim": r"inference-speed improvement|no deployable method",
    "no causal claim for Stage 1/2 from E1": r"establish the causes",
    "reference-history difference disclosed": r"in-session|stored",
    "Tier B untested on Llama": r"untested outside Qwen|not run|untested on Llama",
    "calibration not transferred on Llama": r"did \\emph\{not\} transfer|did not transfer",
}
out = ["# Main-text claims guard (journal manuscript)", "", "Occurrences of watched phrases with context (for human review), then required-statement checks.", ""]
for pat in WATCH:
    hits = []
    for f, t in text.items():
        for m in re.finditer(pat, t, re.I):
            s = t[max(0, m.start() - 90): m.end() + 90].replace("\n", " ")
            hits.append(f"- `{f}`: …{s}…")
    out.append(f"## `{pat}` — {len(hits)} occurrence(s)")
    out += hits[:40]
    out.append("")
whole = "\n".join(text.values())
missing = []
out.append("## Required statements")
for name, pat in REQUIRED.items():
    ok = re.search(pat, whole, re.S) is not None
    out.append(f"- {'OK' if ok else 'MISSING'}: {name}")
    if not ok:
        missing.append(name)
(P / "audit" / OUTNAME).write_text("\n".join(out), encoding="utf-8")
print(f"{len(REQUIRED) - len(missing)}/{len(REQUIRED)} required statements present")
for m in missing:
    print("  MISSING:", m)
sys.exit(1 if missing else 0)
