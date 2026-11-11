"""Perplexity evaluation on the wikitext-2 test split."""

import torch
from datasets import load_dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase

# Sliding-window PPL evaluation window, in tokens. This intentionally does
# NOT use the model's full RoPE-extended context (Qwen2.5-0.5B reports
# max_position_embeddings=131072): a 131k-token window would make attention
# cost O(131072^2) per forward pass, which is impractical on an 8GB laptop
# GPU and is not what quantization papers actually run. 2048 matches the
# window used for wikitext-2 PPL in GPTQ (Frantar et al. 2022), AWQ, and
# llama.cpp's perplexity tool. VERIFY this against whichever specific
# paper(s) you cite in the writeup — conventions vary and a mismatched
# window/stride will produce PPL numbers that aren't directly comparable.
DEFAULT_PPL_MAX_LENGTH = 2048


def load_wikitext2_test_text() -> str:
    """Load and concatenate the wikitext-2 (raw) test split into one string."""
    # The legacy unnamespaced "wikitext" repo was retired from the Hub;
    # "Salesforce/wikitext" is the current canonical mirror with identical
    # config names and splits.
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    return "\n\n".join(dataset["text"])


@torch.no_grad()
def compute_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int | None = None,
    stride: int = 512,
) -> float:
    """Compute perplexity of `model` on wikitext-2 test using a sliding window.

    Text is tokenized once, then scored in overlapping windows of
    `max_length` tokens advancing by `stride`. Only the non-overlapping
    "new" tokens in each window contribute to the loss (via -100 masking
    of the overlap region), which is the standard strided-perplexity
    recipe for causal LMs on long documents. stride=512 is a common choice
    in quantization-eval literature but is NOT universal — cross-check it
    against the specific paper(s) this is being compared to.

    If `max_length` is None, it defaults to DEFAULT_PPL_MAX_LENGTH (2048)
    rather than the model's own max_position_embeddings, since modern
    long-context models report context lengths (e.g. 131072 for
    Qwen2.5-0.5B) that are far too large for a full sliding-window pass to
    be tractable. Pass an explicit value to override.

    Model weights may be FP16 or bitsandbytes 4-bit; in both cases the
    logits returned by `model(...)` are upcast internally to FP16/FP32
    for the loss computation, so this function is precision-agnostic.
    """
    if max_length is None:
        max_length = DEFAULT_PPL_MAX_LENGTH

    text = load_wikitext2_test_text()
    encodings = tokenizer(text, return_tensors="pt")
    input_ids_full = encodings.input_ids.to(model.device)
    seq_len = input_ids_full.size(1)

    nll_sum = 0.0
    n_tokens = 0
    prev_end = 0

    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        trg_len = end - prev_end  # number of tokens not yet scored

        input_ids = input_ids_full[:, begin:end]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100  # mask overlapping tokens

        outputs = model(input_ids, labels=target_ids)

        # Loss is averaged over non-masked tokens; recover the token count
        # to properly weight this window's contribution to the running sum.
        num_valid = (target_ids != -100).sum().item()
        nll_sum += outputs.loss.item() * num_valid
        n_tokens += num_valid

        prev_end = end
        if end == seq_len:
            break

    return float(torch.exp(torch.tensor(nll_sum / n_tokens)))
