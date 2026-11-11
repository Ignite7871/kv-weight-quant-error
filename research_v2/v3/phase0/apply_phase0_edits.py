"""Applies the Phase 0 fixes to the v3 copies only (sections_v3/, refs_v3.bib, main_v3.tex). Each replacement must match exactly once."""
from pathlib import Path
P = Path(__file__).resolve().parents[2] / "paper_journal"
E = [
 ("sections_v3/B_statistics.tex",
  r"\caption{Stage 1 primary family \tagE{}.}",
  r"\caption{Stage 1 primary family \tagE{}. The 90\% CI (H5) is from the locked analysis. H5 takes values on a lattice of step $1/n$ (0.11 points at $n=930$), so its percentile-bootstrap limits can move by one step between bootstrap streams: Table~\ref{tab:oracle}, from an independent stream, gives an upper limit of $+0.43$. The exact bootstrap limits ($B\to\infty$, recomputed from the raw records) are $[-2.04,+0.54]$.}"),
 ("sections_v3/B_statistics.tex",
  r"excluded ($n=342$).}",
  r"excluded ($n=342$). The 90\% CI (H5) is from the locked analysis; H5 lies on a lattice of step $1/n$ (0.29 points at $n=342$), and Table~\ref{tab:oracle}, from an independent bootstrap stream, gives an upper limit of $+2.63$. The exact bootstrap limits ($B\to\infty$, recomputed from the raw records) are $[-2.05,+2.63]$.}"),
 ("sections_v3/E_stage12_details.tex",
  r"Margin-sensitivity curve in Appendix Table~\ref{tab:margin}.}",
  r"Margin-sensitivity curve in Appendix Table~\ref{tab:margin}. The 90\% intervals here come from a bootstrap stream independent of the locked analysis (Tables~\ref{tab:fam1} and~\ref{tab:fam342}); because H5 lies on a lattice of step $1/n$, the two streams differ by one step in the upper limit for Stage~1 ($+0.43$ here, $+0.54$ there) and for held-out-342 ($+2.63$ here, $+2.34$ there). Exact bootstrap limits: Stage~1 $[-2.04,+0.54]$, held-out-342 $[-2.05,+2.63]$; $\delta^*$ and every conclusion are unchanged.}"),
 ("sections_v3/E_stage12_details.tex",
  r"(no \texttt{\#\#\#\#} in 6.1\% of Stage 1 FP16 outputs, 12.4\% NF4, 11.3\% cache-only, 8.1\% token-only;",
  r"(unparseable output, i.e.\ no \texttt{\#\#\#\#} or \texttt{\#\#\#\#} not followed by a number, in 6.1\% of Stage 1 FP16 outputs, 12.4\% NF4, 11.3\% cache-only, 8.4\% token-only;"),
 ("sections_v3/03_intervention.tex",
  r"(on Qwen, about 108 generated against about 431 prompt positions on average)",
  r"(on Qwen in Stage~1, $n=930$: on average about 108 decode-written positions per FP16 generation and 112--119 in the other conditions, against about 431 prompt positions; the first generated token comes from the prefill)"),
 ("sections_v3/C_protocols.tex", r"Nres{authors}", r"\unres{authors}"),
 ("sections_v3/05_results.tex",
  r"(power of about 5--6 points)",
  r"(minimum detectable paired difference of about 5.5--6.8 points at 80\% power, $n=400$)"),
 ("refs_v3.bib",
  "@article{yuan2025nondeterminism,\n  title   = {Understanding and Mitigating Numerical Sources of Nondeterminism in {LLM} Inference (earlier title: Give Me {FP32} or Give Me Death? Challenges and Solutions for Reproducible Reasoning)},\n  author  = {Yuan, Jiayi and others},\n  journal = {arXiv preprint arXiv:2506.09501},\n  year    = {2025},\n  note    = {NeurIPS 2025}\n}",
  "@inproceedings{yuan2025nondeterminism,\n  title     = {Understanding and Mitigating Numerical Sources of Nondeterminism in {LLM} Inference},\n  author    = {Yuan, Jiayi and others},\n  booktitle = {39th Conference on Neural Information Processing Systems (NeurIPS 2025)},\n  year      = {2025},\n  note      = {arXiv:2506.09501}\n}"),
 ("refs_v3.bib",
  "  journal = {Transactions on Machine Learning Research (arXiv:2609.26621)},\n  year    = {2026},\n  note    = {Published in TMLR, September 2026 (per the paper's own header and the arXiv journal reference)}\n}",
  "  journal = {Transactions on Machine Learning Research},\n  year    = {2026},\n  note    = {arXiv:2609.26621}\n}"),
 ("main_v3.tex",
  "% FOR AUTHOR REVIEW. NOT SUBMITTED.",
  "% v3 working copy (branch v3-novelty) of main_review.tex; Phase 0 fixes: research_v2/v3/phase0/PHASE0_CHANGES.md.\n% FOR AUTHOR REVIEW. NOT SUBMITTED."),
]
import sys
for f, old, new in ([] if any(a.startswith('--pass') for a in sys.argv) else E):
    p = P / f
    s = p.read_text(encoding="utf-8")
    assert s.count(old) == 1, (f, s.count(old), old[:60])
    p.write_text(s.replace(old, new), encoding="utf-8")
    print("ok", f)

# second pass: make the population behind \QGenMean explicit (200 Tier A problems, generated tokens incl. the prefill token)
E2 = [("sections_v3/04_design.tex", r"(mean length \QGenMean\ tokens, at most 256)", r"(mean length \QGenMean\ generated tokens over the 200 Tier~A problems, at most 256)"),
      ("sections_v3/B_statistics.tex", r"(mean length \QGenMean\ tokens, at most 256)", r"(mean length \QGenMean\ generated tokens over the 200 Tier~A problems, at most 256)")]
if __name__ == "__main__" and len(__import__("sys").argv) > 1 and __import__("sys").argv[1] == "--pass2":
    for f, old, new in E2:
        p = P / f
        s = p.read_text(encoding="utf-8")
        assert s.count(old) == 1, (f, s.count(old))
        p.write_text(s.replace(old, new), encoding="utf-8")
        print("ok", f)

# third pass: @article renders `note` fused to the journal in elsarticle-num; keep the arXiv id inside the journal field instead
E3 = [("refs_v3.bib", "  journal = {Transactions on Machine Learning Research},\n  year    = {2026},\n  note    = {arXiv:2609.26621}\n}",
       "  journal = {Transactions on Machine Learning Research (arXiv:2609.26621)},\n  year    = {2026}\n}")]
if "--pass3" in sys.argv:
    for f, old, new in E3:
        p = P / f
        s = p.read_text(encoding="utf-8")
        assert s.count(old) == 1, (f, s.count(old))
        p.write_text(s.replace(old, new), encoding="utf-8")
        print("ok", f)

# fourth pass (author decision 4c): Lotfi et al. confirmed on https://neurips.cc/virtual/2026/oral/151669 (title, four authors, "Oral", "NeurIPS 2026")
E4 = [("refs_v3.bib",
       "@article{lotfi2026quantizedthink,\n  title   = {Quantized Reasoning Models Think They Need to Think Longer, but They Do Not},\n  author  = {Lotfi, Sanae and Kirichenko, Polina and Li, Steven and Liu, Zechun},\n  journal = {arXiv preprint arXiv:2606.00206},\n  year    = {2026}\n}",
       "@inproceedings{lotfi2026quantizedthink,\n  title     = {Quantized Reasoning Models Think They Need to Think Longer, but They Do Not},\n  author    = {Lotfi, Sanae and Kirichenko, Polina and Li, Steven and Liu, Zechun},\n  booktitle = {Conference on Neural Information Processing Systems (NeurIPS 2026)},\n  year      = {2026},\n  note      = {Oral; arXiv:2606.00206}\n}")]
if "--pass4" in sys.argv:
    for f, old, new in E4:
        p = P / f
        s = p.read_text(encoding="utf-8")
        assert s.count(old) == 1, (f, s.count(old))
        p.write_text(s.replace(old, new), encoding="utf-8")
        print("ok", f)
