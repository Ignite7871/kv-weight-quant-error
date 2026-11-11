"""Gate check: does Qwen2.5-0.5B have enough raw GSM8K capability, in pure
FP16, for a dynamic-routing comparison to be informative? If pure FP16
solve rate is near zero, a routing comparison against it can't tell us
anything -- there's no headroom for degradation to show up in.

This is deliberately NOT a routing experiment: routing_state.use_4bit is
forced False for the entire generation (via entropy_threshold=-1.0, the
same "forced_fp16" convention used throughout this project -- e.g.
scripts/test_generate_routing.py's Test A2, scripts/test_sequential_ppl.py's
Pass 1 -- since entropy is always >= 0, a threshold of -1.0 makes the
`entropy <= threshold` routing condition always False). No dynamic-routing
GSM8K comparison happens here; that is a separate, later step gated on
this script's result.
"""

import re
import sys
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dynamic_model import DynamicQwen
from model_utils import DEFAULT_MODEL_NAME, load_tokenizer
# Reuse the EXACT same few-shot prompt construction as the earlier routing
# probe, rather than redesigning it, so results stay comparable.
from probe_gsm8k_routing import NUM_FEWSHOT, build_fewshot_prefix

FORCE_FP16_THRESHOLD = -1.0  # see module docstring
MAX_NEW_TOKENS = 256
NUM_GATE_PROBLEMS = 30
PROBE_TEST_INDICES_USED = range(0, 8)  # scripts/probe_gsm8k_routing.py's NUM_PROBLEMS=8, test indices 0-7
NUMERIC_TOLERANCE = 1e-2

NUMBER_RE = re.compile(r"[-+]?\$?[\d][\d,]*\.?\d*")


def clean_number(raw: str):
    """Parse a number-like string ('$1,234.5', '18', '-3') to float, or None."""
    cleaned = raw.replace("$", "").replace(",", "").strip().rstrip(".")
    if not cleaned or cleaned in ("-", "+"):
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def extract_ground_truth(answer_field: str) -> float:
    """GSM8K ground truth is the numeric target after '####' in the answer field."""
    tail = answer_field.split("####")[-1]
    value = clean_number(tail.strip())
    if value is None:
        raise ValueError(f"Could not parse ground-truth answer from: {tail!r}")
    return value


CONTINUATION_DRIFT_RE = re.compile(r"\n\s*Question\s*:", re.IGNORECASE)


def truncate_at_continuation_drift(generated_text: str) -> str:
    """Cut off text at the first sign the model fabricated a NEW question.

    With a generous max_new_tokens, the few-shot-prompted 0.5B model
    routinely finishes answering the target question and then keeps going,
    inventing its own next "Question: ... Answer: ... ####" block (a
    continuation of the few-shot pattern it was shown). Any extraction
    strategy run on text past that point would grab an answer to a
    HALLUCINATED problem, not the target one -- this must be cut off
    before any strategy runs, not patched per-strategy.
    """
    match = CONTINUATION_DRIFT_RE.search(generated_text)
    return generated_text[: match.start()] if match else generated_text


def extract_candidate_answer(generated_text: str):
    """Multi-strategy candidate-answer extraction. Returns (value, strategy_name)."""
    relevant_text = truncate_at_continuation_drift(generated_text)

    # Strategy A: "####" pattern, in case the 0.5B model imitates the GSM8K format.
    # Within relevant_text there should be at most one "####" (the model's
    # real answer); take the first to be robust even if that assumption
    # is violated.
    if "####" in relevant_text:
        tail = relevant_text.split("####")[1]
        match = NUMBER_RE.search(tail)
        if match:
            value = clean_number(match.group())
            if value is not None:
                return value, "hash_pattern"

    # Strategy B: common natural-language final-answer phrasings, last occurrence wins
    # (models often restate the answer near the end after working through the problem).
    phrase_pattern = re.compile(
        r"(?:answer is|answer:|answer =|=)\s*\$?([\-+]?[\d][\d,]*\.?\d*)", re.IGNORECASE
    )
    phrase_matches = list(phrase_pattern.finditer(relevant_text))
    if phrase_matches:
        value = clean_number(phrase_matches[-1].group(1))
        if value is not None:
            return value, "phrase_pattern"

    # Strategy C: last resort -- the final standalone number anywhere in the
    # (already continuation-truncated) text.
    all_numbers = list(NUMBER_RE.finditer(relevant_text))
    if all_numbers:
        value = clean_number(all_numbers[-1].group())
        if value is not None:
            return value, "last_number_fallback"

    return None, "extraction_failed"


def numeric_match(a: float, b: float) -> bool:
    return abs(a - b) < NUMERIC_TOLERANCE


def main() -> None:
    print("Loading GSM8K (openai/gsm8k, main config)...")
    train_ds = load_dataset("openai/gsm8k", "main", split="train")
    test_ds = load_dataset("openai/gsm8k", "main", split="test")

    fewshot_prefix = build_fewshot_prefix(train_ds.select(range(NUM_FEWSHOT)))

    # Distinct from the earlier probe's test indices (0-7): start right after them.
    start = max(PROBE_TEST_INDICES_USED) + 1
    gate_indices = range(start, start + NUM_GATE_PROBLEMS)
    problems = test_ds.select(gate_indices)
    print(
        f"Using test indices {gate_indices.start}-{gate_indices.stop - 1} "
        f"({NUM_GATE_PROBLEMS} problems, distinct from probe indices {PROBE_TEST_INDICES_USED.start}-{PROBE_TEST_INDICES_USED.stop - 1})"
    )

    print(f"\nBuilding DynamicQwen({DEFAULT_MODEL_NAME!r}) forced to pure FP16 (entropy_threshold={FORCE_FP16_THRESHOLD})...")
    dyn = DynamicQwen(DEFAULT_MODEL_NAME, entropy_threshold=FORCE_FP16_THRESHOLD)
    tokenizer = load_tokenizer(DEFAULT_MODEL_NAME)

    results = []  # list of dicts per problem
    for i, problem in enumerate(problems):
        prompt = fewshot_prefix + f"Question: {problem['question']}\nAnswer:"
        generated_text, stats, _routing_sequence = dyn.generate(prompt, tokenizer, MAX_NEW_TOKENS)

        assert stats["4bit_tokens"] == 0, (
            f"Expected pure FP16 (entropy_threshold={FORCE_FP16_THRESHOLD}) but saw "
            f"{stats['4bit_tokens']} 4bit-routed tokens on problem {i} -- forced-FP16 invariant violated."
        )

        ground_truth = extract_ground_truth(problem["answer"])
        candidate, strategy = extract_candidate_answer(generated_text)

        if strategy == "extraction_failed":
            status = "EXTRACTION_FAILED"
        elif numeric_match(candidate, ground_truth):
            status = "SOLVED"
        else:
            status = "WRONG"

        results.append(
            {
                "index": gate_indices.start + i,
                "question": problem["question"],
                "generated_text": generated_text,
                "ground_truth": ground_truth,
                "candidate": candidate,
                "strategy": strategy,
                "status": status,
            }
        )
        print(f"  problem {gate_indices.start + i}: {status:<18} strategy={strategy:<20} candidate={candidate}  truth={ground_truth}")

    # --- Summary ---
    n_total = len(results)
    n_solved = sum(1 for r in results if r["status"] == "SOLVED")
    n_wrong = sum(1 for r in results if r["status"] == "WRONG")
    n_failed = sum(1 for r in results if r["status"] == "EXTRACTION_FAILED")
    solve_rate = 100.0 * n_solved / n_total if n_total else 0.0

    print("\n" + "=" * 78)
    print("GATE CHECK SUMMARY (pure FP16, no dynamic routing)")
    print("=" * 78)
    print(f"Total problems:      {n_total}")
    print(f"Solved:              {n_solved}  ({solve_rate:.1f}%)")
    print(f"Wrong answer:        {n_wrong}  ({100.0 * n_wrong / n_total:.1f}%)")
    print(f"Extraction failed:   {n_failed}  ({100.0 * n_failed / n_total:.1f}%)")
    print("=" * 78)
    print(
        f"\nSOLVE RATE: {solve_rate:.1f}%  "
        f"({'above floor -- proceed to routing comparison' if n_solved >= 3 else 'AT/NEAR FLOOR -- routing comparison would not be informative'})"
    )

    # --- Full transcripts for every non-SOLVED case, so failures can be
    # eyeballed as genuine reasoning errors vs. extraction artifacts. ---
    print("\n" + "=" * 78)
    print("FULL TRANSCRIPTS: WRONG and EXTRACTION_FAILED cases")
    print("=" * 78)
    for r in results:
        if r["status"] == "SOLVED":
            continue
        print(f"\n--- problem {r['index']} [{r['status']}, strategy={r['strategy']}] ---")
        print(f"Question: {r['question']}")
        print(f"Ground truth: {r['ground_truth']}")
        print(f"Extracted candidate: {r['candidate']}")
        print(f"Generated text:\n{r['generated_text']}")


if __name__ == "__main__":
    main()
