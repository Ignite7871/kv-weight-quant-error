"""Diagnostic probe: does the CURRENT entropy-gated router (unmodified, at
the existing p50 calibrated threshold) treat digit/operator tokens
differently than other tokens during math reasoning generation?

Motivation: this directly tests the apparent conflict with ReQAT's finding
that quantization failures concentrate on low-entropy digit/operator
tokens. If digit/operator tokens are systematically LOW entropy under our
model, our router would route them to 4-bit (the cheap, less-precise path)
-- the opposite of what ReQAT's finding would suggest is safe.

This is observation only: no routing logic, threshold, or model code is
changed. We reuse the existing median-entropy calibrated threshold (p50,
tau=1.943) from results/calibration_thresholds.json as-is.

NOT in scope here: grading GSM8K answer correctness. A full accuracy-vs-
routing evaluation (GSM8K solve rate under dynamic routing vs. pure FP16
vs. pure 4-bit) is a separate, larger follow-up, not part of this probe.
"""

import re
import sys
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from dynamic_model import DynamicQwen
from model_utils import DEFAULT_MODEL_NAME, load_tokenizer

MEDIAN_THRESHOLD = 1.943  # results/calibration_thresholds.json p50, unchanged
NUM_FEWSHOT = 3
NUM_PROBLEMS = 8
MAX_NEW_TOKENS = 150
NUM_ANNOTATED_EXAMPLES = 3

# Digits 0-9, arithmetic/comparison operators, and common symbols GSM8K
# reasoning chains use (decimal points, commas in numbers, currency,
# percent, parentheses). A token counts as DIGIT_OP only if its STRIPPED
# text consists entirely of these characters.
DIGIT_OP_PATTERN = re.compile(r"^[0-9+\-*/=.,()$%<>]+$")


def build_fewshot_prefix(train_examples) -> str:
    """Standard GSM8K few-shot format: Question/Answer pairs ending in '#### <ans>'."""
    blocks = []
    for ex in train_examples:
        blocks.append(f"Question: {ex['question']}\nAnswer: {ex['answer']}")
    return "\n\n".join(blocks) + "\n\n"


def classify_token(token_text: str) -> str:
    stripped = token_text.strip()
    if stripped and DIGIT_OP_PATTERN.match(stripped):
        return "DIGIT_OP"
    return "OTHER"


def main() -> None:
    print(f"Loading GSM8K (openai/gsm8k, main config)...")
    train_ds = load_dataset("openai/gsm8k", "main", split="train")
    test_ds = load_dataset("openai/gsm8k", "main", split="test")

    # Few-shot exemplars come from TRAIN, evaluated problems from TEST --
    # avoids leaking evaluated problems into their own prompt.
    fewshot_prefix = build_fewshot_prefix(train_ds.select(range(NUM_FEWSHOT)))
    problems = test_ds.select(range(NUM_PROBLEMS))

    print(f"Building DynamicQwen({DEFAULT_MODEL_NAME!r}) at threshold={MEDIAN_THRESHOLD} (p50, unchanged)...")
    dyn = DynamicQwen(DEFAULT_MODEL_NAME, entropy_threshold=MEDIAN_THRESHOLD)
    tokenizer = load_tokenizer(DEFAULT_MODEL_NAME)

    # Aggregate counts across all problems.
    counts = {
        "DIGIT_OP": {"fp16": 0, "4bit": 0},
        "OTHER": {"fp16": 0, "4bit": 0},
    }

    annotated_examples = []

    for i, problem in enumerate(problems):
        prompt = fewshot_prefix + f"Question: {problem['question']}\nAnswer:"
        generated_text, stats, routing_sequence = dyn.generate(prompt, tokenizer, MAX_NEW_TOKENS)

        generated_ids = dyn.last_generated_ids
        # generated_ids[0] is the fixed-FP16 prefill token (no routing
        # decision -- excluded from the DIGIT_OP/OTHER aggregate, same
        # convention generate()'s own stats dict uses). generated_ids[j]
        # for j>=1 aligns with routing_sequence[j-1].
        tagged_tokens = []  # (token_text, tag) for the annotated printout
        first_token_text = tokenizer.decode([generated_ids[0]])
        tagged_tokens.append((first_token_text, "FP16-prefill"))

        for j in range(1, len(generated_ids)):
            token_text = tokenizer.decode([generated_ids[j]])
            route = routing_sequence[j - 1]  # "fp16" or "4bit"
            category = classify_token(token_text)
            counts[category][route] += 1
            tagged_tokens.append((token_text, route))

        print(f"  problem {i}: {stats}  ({len(routing_sequence)} routed steps)")

        if i < NUM_ANNOTATED_EXAMPLES:
            annotated_examples.append((problem["question"], tagged_tokens))

    # --- Aggregate report ---
    print("\n" + "=" * 78)
    print("AGGREGATE: DIGIT_OP vs OTHER routing (excludes the fixed-FP16 prefill token per problem)")
    print("=" * 78)
    for category in ("DIGIT_OP", "OTHER"):
        fp16_n = counts[category]["fp16"]
        bit4_n = counts[category]["4bit"]
        total = fp16_n + bit4_n
        if total == 0:
            print(f"{category}: no tokens observed")
            continue
        pct_fp16 = 100.0 * fp16_n / total
        pct_4bit = 100.0 * bit4_n / total
        print(
            f"{category:<10} n={total:<5}  {pct_fp16:5.1f}% routed FP16   {pct_4bit:5.1f}% routed 4-bit   "
            f"(raw: {fp16_n} fp16 / {bit4_n} 4bit)"
        )
    print("=" * 78)

    # --- Annotated examples ---
    print("\n" + "=" * 78)
    print(f"ANNOTATED EXAMPLES (first {NUM_ANNOTATED_EXAMPLES} problems)")
    print("=" * 78)
    for question, tagged_tokens in annotated_examples:
        print(f"\nQuestion: {question}")
        print("Generated (tagged):")
        line = ""
        for token_text, tag in tagged_tokens:
            line += f"[{tag}]{token_text}"
        print(line)


if __name__ == "__main__":
    main()
