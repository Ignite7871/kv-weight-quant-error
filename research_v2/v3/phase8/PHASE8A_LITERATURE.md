# Phase 8a literature verification (2026-11-01)

The four references added to `paper_journal/refs_v3.bib` in Phase 8a. Each fact used in the manuscript was read from the
primary source named here, through Claude Code's WebFetch/WebSearch tools (the page-summarization model behind WebFetch is not
identified to the agent) and, for two PDFs, local `pdftotext`. The extracted texts are in the session scratchpad only; they are not
committed.

| Key | Primary source checked | Facts used in the manuscript |
|---|---|---|
| `liu2024intactkv` | aclanthology.org/2024.findings-acl.460 (title, authors, Findings of ACL 2024, pp. 7716-7741, DOI 10.18653/v1/2024.findings-acl.460); Anthology PDF, Sections 1, 2 and 4 | IntactKV generates the KV cache of "pivot tokens" (initial tokens that receive most attention) "losslessly from the full-precision model" and uses it in the quantized model; the experiments mainly use weight-only quantization (RTN, GPTQ, AWQ, OmniQuant; INT3/INT4), with extensions to KV-cache and activation quantization |
| `fan2021quantnoise` | iclr.cc/virtual/2021/poster/2772 (ICLR 2021 poster); arXiv:2004.07320 abstract | Quant-Noise quantizes a different random subset of the weights in each forward pass, so the other weights get unbiased gradients; the "noise" is the quantizer itself applied to a subset, not additive random noise |
| `defossez2022diffq` | arXiv:2104.09987v3 PDF (header: "Published in Transactions on Machine Learning Research (09/2022)"; "Reviewed on OpenReview ... DijnKziche"); method section | DiffQ adds independent pseudo quantization noise to the parameters during training to approximate a quantizer; the noise is an independent uniform sample per weight in the derivation, and Gaussian noise is used in the experiments; the pseudo-quantization-noise assumption is attributed to Widrow et al. The OpenReview page itself could not be opened (browser check), so the venue rests on the TMLR header of the arXiv v3 PDF |
| `widrow2008quantization` | cambridge.org/core product page (DOI 10.1017/CBO9780511754661): Cambridge University Press, print 03 July 2008, ISBN 9780521886710 | Classical statistical theory of quantization, in which quantization error is modelled as additive noise ("pseudo quantization noise") independent of the signal |

## Added with the 8a author edits (2026-11-01)

| Key / item | Primary source checked | Facts used |
|---|---|---|
| `qwen2024qwen25` (already cited) | arXiv:2412.15115 PDF, architecture paragraph | Qwen2.5 uses "QKV bias (Su, 2023) in the attention mechanism" |
| `wolf2020transformers` | aclanthology.org/2020.emnlp-demos.6 (EMNLP 2020 System Demonstrations, pp. 38-45, DOI 10.18653/v1/2020.emnlp-demos.6) | cited for the model implementation: `modeling_qwen2.py` hard-codes `bias=True` for q/k/v (lines 200-202 of the installed version); `modeling_llama.py` uses `config.attention_bias` |
| Model configs | locked `config.json` of the four checkpoints (hashes in V3_PROTOCOL_LOCK.txt); checkpoint tensor names | Llama-3.2-1B and -Instruct: `attention_bias: false`, no bias tensors; Qwen2.5-0.5B: `k_proj.bias` etc. present |

## Licences (checked 2026-11-01 for the declarations)

| Item | Primary source | Finding |
|---|---|---|
| Qwen2.5-0.5B, Qwen2.5-0.5B-Instruct | huggingface.co API (`cardData.license` = apache-2.0, `license_link` .../blob/main/LICENSE) and the repository LICENSE files (Apache License, Version 2.0, January 2004) | Apache-2.0 |
| Llama-3.2-1B, Llama-3.2-1B-Instruct | huggingface.co API (`cardData.license` = llama3.2; gated; repository files LICENSE.txt, USE_POLICY.md); llama.com/llama3_2/license (Llama 3.2 Community License Agreement, release date September 25, 2024), section 1.b | On distribution of Llama Materials or derivatives: (A) provide a copy of the Agreement; (B) prominently display "Built with Llama"; a model trained with Llama outputs and distributed must start its name with "Llama"; (iii) retain the notice "Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved." in a "Notice" text file; (iv) comply with the Acceptable Use Policy; section 2: separate licence above 700 million monthly active users |
| GSM8K | github.com/openai/grade-school-math LICENSE (MIT License, Copyright (c) 2021 OpenAI); local Hugging Face dataset card `openai/gsm8k` (license: mit) | MIT |

## Effect on the "isotropic noise as a stand-in" statement

No source was found that uses isotropic noise of the per-slot norm as a stand-in for weight-induced cache error. The Discussion
sentence that called it "a common stand-in for quantization error" was replaced by a sourced statement: independent additive
noise is the classical model of quantization error (Widrow and Kollar), is used as a training-time proxy for weight quantization
(DiffQ), and underlies the white-noise analysis of attention-aware KV distortion (AATC, already cited). The isotropic control is
described as the norm-matched version of that model for a cache entry; the paper no longer claims that this exact control is common.
