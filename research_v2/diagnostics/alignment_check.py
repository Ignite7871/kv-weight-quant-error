"""Independent CPU verification of EQF-dataset row alignment (research_v2, exploratory diagnostics).

Question: in results/eqf_spike_dataset_<model>.csv, is
  - row i's `entropy` the entropy of the FP16 distribution that PREDICTS gold token i
    (lag-1 relative to the step whose JS is stored in row i), and
  - row i+1's `entropy` the entropy of the FP16 distribution produced by the SAME step
    whose JS is stored in row i (the step that consumes gold token i and predicts token i+1)?
And is there label leakage, i.e. does either quantity depend on the token it predicts?

Method (deliberately different from the original builder, scripts/build_eqf_spike.py, which runs
incremental single-token FP16 steps on GPU with a growing KV cache):
  * one PARALLEL teacher-forced forward pass over prompt + " " + gold answer, in FLOAT32 on CPU,
    no KV cache, no DynamicLinear wrapper -- plain HF model;
  * the prompt and answer are tokenized exactly as the builder does (prompt with special tokens,
    answer with add_special_tokens=False and a leading space), then concatenated;
  * compare our entropies at positions P+i-1 (lag-1) and P+i (same step) with stored values, and
    our argmax at P+i with the stored fp16_top1 of row i;
  * leakage test: replace gold token i+1 with a different token and recompute; the distributions at
    positions <= P+i must be bit-identical (causal masking), so neither the lag-1 nor the same-step
    entropy of row i can encode the label it is used to predict.

Expected tolerance: stored values are FP16-on-GPU, ours FP32-on-CPU, so entropies agree to ~1e-2
nats, not bit-exactly; argmax can differ only at near-ties.

Usage: venv/Scripts/python.exe research_v2/diagnostics/alignment_check.py [n_problems]
"""

import sys
from pathlib import Path

# HF imports before torch (project convention on this Windows machine).
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "results" / "eqf_spike_dataset_Qwen2.5-0.5B.csv"
MODEL = "Qwen/Qwen2.5-0.5B"
N_FEWSHOT = 3


def entropy(logits: torch.Tensor) -> torch.Tensor:
    lp = torch.log_softmax(logits.float(), dim=-1)
    return -(lp.exp() * lp).sum(-1)


def build_fewshot_prefix(train_examples) -> str:  # verbatim copy of the builder's function
    blocks = [f"Question: {ex['question']}\nAnswer: {ex['answer']}" for ex in train_examples]
    return "\n\n".join(blocks) + "\n\n"


@torch.no_grad()
def main() -> None:
    n_problems = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    torch.manual_seed(0)
    df = pd.read_csv(CSV)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    train = load_dataset("openai/gsm8k", "main", split="train")
    test = load_dataset("openai/gsm8k", "main", split="test")
    prefix = build_fewshot_prefix(train.select(range(N_FEWSHOT)))

    report = []
    for pid in range(n_problems):
        rows = df[df.problem_id == pid].sort_values("token_idx").reset_index(drop=True)
        prob = test[pid]
        prompt_ids = tok(prefix + f"Question: {prob['question']}\nAnswer:", return_tensors="pt").input_ids
        ans_ids = tok(" " + prob["answer"], add_special_tokens=False, return_tensors="pt").input_ids
        assert ans_ids.size(1) == len(rows), f"problem {pid}: {ans_ids.size(1)} answer tokens vs {len(rows)} rows"
        # token identity check: stored token text must be the decode of gold token i
        for i in range(len(rows)):
            assert tok.decode(ans_ids[0, i : i + 1]) == rows.token[i], f"token text mismatch p{pid} i{i}"

        P = prompt_ids.size(1)
        full = torch.cat([prompt_ids, ans_ids], dim=1)
        logits = model(full).logits[0]  # [T, V]
        H = entropy(logits)
        n = len(rows)
        H_lag1 = H[P - 1 : P - 1 + n].numpy()   # position P+i-1 predicts gold token i
        H_same = H[P : P + n].numpy()           # position P+i consumes token i, predicts i+1
        top1_same = logits[P : P + n].argmax(-1)

        stored_H = rows.entropy.to_numpy()
        stored_H_next = rows.entropy.shift(-1).to_numpy()  # what lag_check.py used as "same step"
        top1_match = np.mean([tok.decode([int(t)]) == s for t, s in zip(top1_same, rows.fp16_top1)])

        # leakage test: change gold token i+1 for a mid-answer i and check positions <= P+i unchanged
        i = n // 2
        perturbed = full.clone()
        old = int(perturbed[0, P + i + 1])
        perturbed[0, P + i + 1] = (old + 12345) % tok.vocab_size
        logits_p = model(perturbed).logits[0]
        causal_ok = torch.equal(logits_p[: P + i + 1], logits[: P + i + 1])
        changed_after = not torch.equal(logits_p[P + i + 1], logits[P + i + 1])

        report.append({
            "problem": pid, "n_tokens": n,
            "max|H_lag1 - stored entropy|": float(np.max(np.abs(H_lag1 - stored_H))),
            "max|H_same - stored entropy(next row)|": float(np.nanmax(np.abs(H_same[:-1] - stored_H_next[:-1]))),
            "max|H_same - stored entropy(same row)| (should be LARGE)": float(np.max(np.abs(H_same - stored_H))),
            "fp16_top1 match (same-step argmax)": float(top1_match),
            "causal: logits<=P+i unchanged when token i+1 altered": bool(causal_ok),
            "sanity: logits at P+i+1 do change": bool(changed_after),
        })

    out = pd.DataFrame(report)
    pd.set_option("display.width", 250)
    print(out.to_string(index=False))
    ok = (
        (out["max|H_lag1 - stored entropy|"] < 0.05).all()
        and (out["max|H_same - stored entropy(next row)|"] < 0.05).all()
        and (out["fp16_top1 match (same-step argmax)"] >= 0.97).all()
        and out["causal: logits<=P+i unchanged when token i+1 altered"].all()
        and out["sanity: logits at P+i+1 do change"].all()
    )
    print("\nALIGNMENT CHECK", "PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
