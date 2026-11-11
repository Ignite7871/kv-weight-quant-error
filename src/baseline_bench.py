"""Phase 1 (hardened) baseline: static FP16 vs static 4-bit (NF4) generation.

Runs a fixed set of prompts through a model loaded in each precision,
measuring TTFT (prefill), TPOT (decode-only per-token latency), peak VRAM,
and wikitext-2 perplexity, across multiple independent trials to report
mean +/- std. No dynamic per-token precision switching happens here — that
is Phase 2 scope.

Results are written to results/baseline_results_rigorous.csv and .json.
"""

import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model_utils import (
    DEFAULT_MODEL_NAME,
    FALLBACK_MODEL_NAME,
    load_model_4bit,
    load_model_fp16,
    load_tokenizer,
)
from eval import compute_perplexity

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Small fixed prompt set spanning short/simple to longer/more complex inputs,
# so latency is measured over a representative mix rather than one shape.
PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Explain the theory of relativity in simple terms:",
    "Once upon a time, in a distant galaxy,",
    "The main causes of the French Revolution were",
    "To bake a chocolate cake, you first need to",
    "In machine learning, overfitting occurs when",
    "Write a short poem about the ocean.",
]

MAX_NEW_TOKENS = 50
NUM_TRIALS = 3
# VRAM budget for the OOM pre-flight warning, matching the laptop's 8GB card.
VRAM_BUDGET_GB = 8.0


def warn_if_oom_risk() -> None:
    """Print a warning if current free VRAM looks tight for this benchmark.

    This is a heuristic pre-flight check, not a guarantee: actual usage
    depends on model size and generation length. An OutOfMemoryError during
    the run is still caught and reported explicitly rather than silently
    handled.
    """
    if not torch.cuda.is_available():
        return
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gb = free_bytes / 1024**3
    total_gb = total_bytes / 1024**3
    print(f"[VRAM check] {free_gb:.2f} GB free / {total_gb:.2f} GB total")
    if free_gb < 2.0:
        print(
            f"[WARNING] Only {free_gb:.2f} GB free VRAM. Close other GPU "
            "applications before running the benchmark, or expect possible OOM."
        )


@torch.no_grad()
def generate_with_timing(
    model, tokenizer, prompt: str, max_new_tokens: int
) -> tuple[float, list[float]]:
    """Greedy-decode `prompt` with a manual loop, timing prefill vs decode.

    This bypasses model.generate() so prefill and each individual decode
    step can be timed separately with torch.cuda.synchronize() bracketing
    each one — a black-box call to generate() cannot be split this way.

    Boundary definition (matches vLLM/TensorRT-LLM serving-benchmark
    convention):
      - TTFT (returned as `ttft_s`): wall-clock time for the prefill
        forward pass over the full prompt, which produces the logits used
        to pick token #1 (the FIRST generated token). Token #1's own
        argmax/sampling cost is negligible and is included in TTFT, not
        TPOT — i.e. TTFT covers "time until token #1 exists".
      - TPOT (returned as `decode_times_s`, one entry per token): wall-clock
        time for each subsequent forward pass (tokens #2..N), each
        conditioned on the growing KV-cache via `past_key_values`. Token #1
        is NOT included in this list.

    Greedy (argmax) decoding is used throughout, matching the do_sample=False
    behavior of the original model.generate() baseline, so trial-to-trial
    differences reflect timing noise, not sampling randomness.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]

    # --- Prefill: produces token #1 -> this is TTFT. ---
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outputs = model(input_ids, use_cache=True)
    torch.cuda.synchronize()
    ttft_s = time.perf_counter() - t0

    next_token_logits = outputs.logits[:, -1, :]
    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
    past_key_values = outputs.past_key_values

    # --- Decode: tokens #2..N, one forward pass each -> these are TPOT samples. ---
    decode_times_s: list[float] = []
    cur_input = next_token
    for _ in range(max_new_tokens - 1):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = model(cur_input, past_key_values=past_key_values, use_cache=True)
        torch.cuda.synchronize()
        decode_times_s.append(time.perf_counter() - t0)

        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        past_key_values = outputs.past_key_values
        cur_input = next_token

    return ttft_s, decode_times_s


def run_generation_trial(model, tokenizer, prompts: list[str]) -> tuple[float, float, float]:
    """Run one generation trial over all prompts.

    Returns (ttft_ms_mean, tpot_ms_mean, tokens_per_sec), where tpot_ms_mean
    and tokens_per_sec are derived ONLY from decode-phase steps (excluding
    prefill), since the eventual dynamic router only operates during decode.
    """
    all_ttft_s: list[float] = []
    all_decode_times_s: list[float] = []

    for prompt in prompts:
        ttft_s, decode_times_s = generate_with_timing(model, tokenizer, prompt, MAX_NEW_TOKENS)
        all_ttft_s.append(ttft_s)
        all_decode_times_s.extend(decode_times_s)

    ttft_ms_mean = float(np.mean(all_ttft_s)) * 1000
    tpot_ms_mean = float(np.mean(all_decode_times_s)) * 1000
    tokens_per_sec = 1000.0 / tpot_ms_mean
    return ttft_ms_mean, tpot_ms_mean, tokens_per_sec


def run_precision_trials(model_name: str, precision: str, num_trials: int = NUM_TRIALS) -> list[dict]:
    """Load `model_name` at `precision` once, then run `num_trials` independent trials."""
    print(f"\n=== {model_name} [{precision}] ===")
    tokenizer = load_tokenizer(model_name)

    try:
        if precision == "fp16":
            model = load_model_fp16(model_name)
        elif precision == "4bit":
            model = load_model_4bit(model_name)
        else:
            raise ValueError(f"Unknown precision: {precision}")
    except torch.cuda.OutOfMemoryError as e:
        raise RuntimeError(
            f"CUDA OOM while loading {model_name} in {precision}. "
            f"Consider FALLBACK_MODEL_NAME ({FALLBACK_MODEL_NAME}) instead."
        ) from e

    # Warm-up call (excluded from all trials) to pay for one-time CUDA kernel
    # compilation / cuBLAS handle setup so it doesn't skew trial 1.
    generate_with_timing(model, tokenizer, PROMPTS[0], 8)
    torch.cuda.synchronize()

    trial_rows = []
    try:
        for trial in range(1, num_trials + 1):
            # Reset before each trial so VRAM and timing aren't contaminated
            # by the previous trial's allocations.
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            ttft_ms, tpot_ms, tokens_per_sec = run_generation_trial(model, tokenizer, PROMPTS)
            perplexity = compute_perplexity(model, tokenizer)

            # Peak VRAM here covers both the generation and perplexity phases
            # of this trial (both ran since the last reset above).
            peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2

            row = {
                "model": model_name,
                "precision": precision,
                "trial": trial,
                "ttft_ms": round(ttft_ms, 3),
                "tpot_ms_mean": round(tpot_ms, 3),
                "tokens_per_sec": round(tokens_per_sec, 2),
                "peak_vram_mb": round(peak_vram_mb, 2),
                "perplexity": round(perplexity, 4),
            }
            print(f"  trial {trial}: {row}")
            trial_rows.append(row)
    except torch.cuda.OutOfMemoryError as e:
        raise RuntimeError(
            f"CUDA OOM during benchmarking of {model_name} in {precision}."
        ) from e
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()

    return trial_rows


def summarize(trial_rows: list[dict]) -> dict:
    """Aggregate mean +/- std across trials for one (model, precision) group."""
    metrics = ["ttft_ms", "tpot_ms_mean", "tokens_per_sec", "peak_vram_mb", "perplexity"]
    df = pd.DataFrame(trial_rows)
    summary = {"model": df["model"].iloc[0], "precision": df["precision"].iloc[0], "trial": "summary"}
    for metric in metrics:
        summary[f"{metric}_mean"] = round(float(df[metric].mean()), 4)
        # ddof=1 (sample std) since these are 3 independent trial measurements.
        summary[f"{metric}_std"] = round(float(df[metric].std(ddof=1)), 4)
    return summary


def print_summary_table(summaries: dict[str, dict]) -> None:
    """Print a side-by-side FP16 vs 4-bit summary table to the console."""
    fp16, four_bit = summaries["fp16"], summaries["4bit"]
    print("\n" + "=" * 72)
    print(f"{'metric':<24}{'fp16':<24}{'4bit (NF4)':<24}")
    print("=" * 72)
    rows = [
        ("TTFT (ms)", "ttft_ms"),
        ("TPOT (ms/token)", "tpot_ms_mean"),
        ("Tokens/sec (TPOT-based)", "tokens_per_sec"),
        ("Peak VRAM (MB)", "peak_vram_mb"),
        ("Perplexity (wikitext-2)", "perplexity"),
    ]
    for label, key in rows:
        fp16_str = f"{fp16[f'{key}_mean']} +/- {fp16[f'{key}_std']}"
        bit4_str = f"{four_bit[f'{key}_mean']} +/- {four_bit[f'{key}_std']}"
        print(f"{label:<24}{fp16_str:<24}{bit4_str:<24}")
    print("=" * 72)
    print(
        "Note: perplexity is expected to have near-zero std across trials "
        "since greedy forward passes are deterministic modulo minor "
        "GPU floating-point nondeterminism (kernel reduction order)."
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This benchmark requires a CUDA GPU; "
            "refusing to silently fall back to CPU."
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    warn_if_oom_risk()

    model_name = DEFAULT_MODEL_NAME
    all_trial_rows = []
    summaries = {}
    for precision in ("fp16", "4bit"):
        trial_rows = run_precision_trials(model_name, precision)
        all_trial_rows.extend(trial_rows)
        summaries[precision] = summarize(trial_rows)

    print_summary_table(summaries)

    csv_path = RESULTS_DIR / "baseline_results_rigorous.csv"
    json_path = RESULTS_DIR / "baseline_results_rigorous.json"

    trials_df = pd.DataFrame(all_trial_rows)
    summary_df = pd.DataFrame(list(summaries.values()))
    # Append summary rows (trial="summary") beneath the per-trial rows, in
    # the same CSV, so both are visible together per the task spec.
    combined_df = pd.concat([trials_df, summary_df], ignore_index=True)
    combined_df.to_csv(csv_path, index=False)

    with open(json_path, "w") as f:
        json.dump({"trials": all_trial_rows, "summary": list(summaries.values())}, f, indent=2)

    print(f"\nResults written to {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
