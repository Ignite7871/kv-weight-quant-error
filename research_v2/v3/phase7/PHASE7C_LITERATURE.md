# Phase 7c — literature verification (2026-11-01)

These are the references added or completed for the revised related-work section. Each venue was read on the proceedings
page listed below, which states the title, the full author list and the venue. Pages were read through the Claude Code
fetch tool. Following the project rule, a venue is cited only if a proceedings page confirms it.

| Key | Work | Venue as cited | Proceedings page checked |
|---|---|---|---|
| `xiao2024attentionsinks` | Xiao, Tian, Chen, Han, Lewis. *Efficient Streaming Language Models with Attention Sinks* | ICLR 2024 | proceedings.iclr.cc/paper_files/paper/2024/hash/5e5fd18f863cbe6d8ae392a93fd271c9-Abstract-Conference.html |
| `sun2024massive` | Sun, Chen, Kolter, Liu. *Massive Activations in Large Language Models* | COLM 2024 | OpenReview note F7aAhfitX6, venue id `colmweb.org/COLM/2024/Conference` (from the OpenReview API search). COLM publishes its proceedings on OpenReview. |
| `ashkboos2024quarot` | Ashkboos et al. *QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs* | NeurIPS 2024 (Advances in NeurIPS 37) | proceedings.neurips.cc/paper_files/paper/2024/hash/b5b939436789f76f08b9d0da5e81af7c-Abstract.html |
| `liu2025spinquant` | Liu et al. *SpinQuant: LLM Quantization with Learned Rotations* | ICLR 2025 | proceedings.iclr.cc/paper_files/paper/2025/hash/e5b1c0d4866f72393c522c8a00eed4eb-Abstract-Conference.html |
| `lin2025qserve` | Lin, Tang, Yang, Zhang, Xiao, Gan, Han. *QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving* | Proceedings of MLSys 7 (2025) | proceedings.mlsys.org/paper_files/paper/2025 (listed; hash fbe2b2f74a2ece8070d8fb073717bda6) |
| `zhao2024atom` | Zhao et al. *Atom: Low-Bit Quantization for Efficient and Accurate LLM Serving* | Proceedings of MLSys 6 (2024) | proceedings.mlsys.org/paper_files/paper/2024/hash/5edb57c05c81d04beb716ef1d542fe9e-Abstract-Conference.html |
| `liu2024kivi` (completed) | Liu et al. *KIVI* | ICML 2024, PMLR 235:32332–32344 | proceedings.mlr.press/v235/liu24bz.html |
| `hooper2024kvquant` (completed) | Hooper et al. *KVQuant* | NeurIPS 2024 (Advances in NeurIPS 37) | proceedings.neurips.cc/paper_files/paper/2024/hash/028fcbcf85435d39a40c4d61b42c99a4-Abstract-Conference.html |
| `zhao2025qspec` (completed) | Zhao, Lu, Wang, Kong, Wu. *QSpec* | EMNLP 2025, pp. 4779–4795, doi 10.18653/v1/2025.emnlp-main.240 | aclanthology.org/2025.emnlp-main.240 |

**What the paper says about each work, and its basis.** The content claims rest on the abstracts on these pages and on
the earlier reading recorded in the supplement (S6).
- Attention sinks: early tokens receive large attention regardless of their semantic importance.
- Massive activations: a very few activations are vastly larger than the rest, stay nearly constant across inputs, act as
  bias terms and concentrate attention on specific tokens.
- KIVI: keys are best quantized per channel and values per token.
- KVQuant: per-channel and pre-rotary key quantization, and per-vector outlier isolation.
- QuaRot: rotations remove outliers from the hidden state without changing the output, and weights, activations and the KV
  cache are quantized to 4 bits.
- SpinQuant: learned rotations.
- QServe: 4-bit weights, 8-bit activations and a 4-bit KV cache.
- Atom: 4-bit weights and activations, with a quantized KV cache.
- QSpec: a low-precision draft with quantized weights and activations writes cache entries that verification overwrites
  for accepted tokens. The supplement's S6 reading found no ablation of overwriting against keeping the entries.

The paper does **not** claim that the Qwen layer-0 key norms are attention sinks or massive activations ("may be related;
we did not test this"). Nor does it claim that rotations change the key–value asymmetry ("suggests … we did not test this
either").
