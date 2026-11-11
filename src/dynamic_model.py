"""Dynamic per-token precision switching model wrapper (Phase 2).

__init__ is now implemented: it loads a base FP16 model and replaces every
attention/MLP projection nn.Linear with a DynamicLinear (dynamic_linear.py)
holding both FP16 and 4-bit weights, sharing one RoutingState instance.
generate() is still a design-only scaffold (raises NotImplementedError) --
the autoregressive per-token routing loop is the next step, pending review
of this construction step.

Phase 2 goal recap: switch a causal LM's WEIGHTS between FP16 and 4-bit
NF4 at the token level, using the entropy of the previous token's logits
(see routing.calculate_entropy) as the routing signal. The KV-cache must
remain FP16 at all times regardless of which weight precision produced it
(enforced today at the DynamicLinear.forward() output-cast level; the
KV-cache itself is not yet touched since generate() isn't implemented).

Research framing note (locked in after the Phase 1 hardened baseline):
bitsandbytes 4-bit on this GPU is a software dequantization path and was
measurably SLOWER than FP16 at batch size 1 (see
results/baseline_results_rigorous.json) -- true 4-bit hardware execution
(NPUs, native INT4 kernels) is out of scope here. This project therefore
does NOT claim empirical wall-clock speedup from dynamic routing. Instead
it claims reduced theoretical memory-bandwidth pressure / compute
workload, with the fp16:4bit routed-token ratio (see self.stats below) as
the primary proxy metric -- not tokens/sec.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase

from dynamic_linear import DynamicLinear, RoutingState
from routing import calculate_entropy

# Named children targeted for DynamicLinear replacement: the attention and
# MLP projection layers of Qwen2.5's decoder blocks (model.layers[i].
# self_attn.{q,k,v,o}_proj and model.layers[i].mlp.{gate,up,down}_proj).
# Deliberately excludes embed_tokens (nn.Embedding, not nn.Linear), the
# final norm, and lm_head -- lm_head is tied to embed_tokens for
# Qwen2.5-0.5B (tie_word_embeddings=True), so routing it dynamically would
# desync the tied weights; it stays FP16-only.
DYNAMIC_LINEAR_TARGET_NAMES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}


def replace_linear_with_dynamic(model: nn.Module, routing_state: RoutingState) -> list[str]:
    """Replace every targeted nn.Linear submodule in `model` with a DynamicLinear.

    Walks named_modules() looking for direct children whose name matches
    DYNAMIC_LINEAR_TARGET_NAMES and whose type is nn.Linear, and swaps
    each one in-place via setattr on its parent module. Returns the list
    of fully-qualified module names that were replaced, so the caller can
    print/inspect exactly what changed.
    """
    replaced_names = []
    for parent_name, parent_module in list(model.named_modules()):
        for child_name, child_module in list(parent_module.named_children()):
            if child_name in DYNAMIC_LINEAR_TARGET_NAMES and isinstance(child_module, nn.Linear):
                dynamic_layer = DynamicLinear(child_module, routing_state)
                setattr(parent_module, child_name, dynamic_layer)
                full_name = f"{parent_name}.{child_name}" if parent_name else child_name
                replaced_names.append(full_name)
    return replaced_names


class DynamicQwen:
    """Wraps a base Qwen model with simultaneously-loaded FP16 and 4-bit weights.

    Construction loads the model once in FP16, then replaces attention/MLP
    projection layers with DynamicLinear in-place -- both weight
    representations are seeded from the SAME loaded FP16 weights (the
    4-bit representation is quantized from them at construction time),
    rather than loading the model twice via two separate code paths.
    """

    def __init__(self, model_name: str, entropy_threshold: float):
        """Load the base model once, then wire in DynamicLinear everywhere.

        Args:
            model_name: HF model id, e.g. "Qwen/Qwen2.5-0.5B".
            entropy_threshold: entropy cutoff (in nats) used by generate()
                to decide FP16 vs 4-bit routing per token. Above threshold
                -> FP16 (uncertain, spend the accuracy budget); at or
                below -> 4-bit (confident, use the cheap path). TODO: this
                is a single global scalar for now; whether it should be
                per-layer, calibrated per-model, or dynamically annealed
                during generation is still an open design question, not
                decided by this step.
        """
        self.model_name = model_name
        self.entropy_threshold = entropy_threshold
        self.routing_state = RoutingState()

        # Reset peak-memory tracking before construction so the
        # memory_summary()/max_memory_allocated() printed below reflect
        # ONLY this DynamicQwen's construction, not whatever else has run
        # in this process.
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        print(f"[DynamicQwen] Loading base FP16 model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16).to("cuda")
        self.model.eval()

        print("[DynamicQwen] Replacing attention/MLP nn.Linear layers with DynamicLinear...")
        self.replaced_module_names = replace_linear_with_dynamic(self.model, self.routing_state)
        print(f"[DynamicQwen] Replaced {len(self.replaced_module_names)} modules:")
        for name in self.replaced_module_names:
            print(f"  {name}")

        torch.cuda.synchronize()
        print("\n[DynamicQwen] torch.cuda.memory_summary() after construction:")
        print(torch.cuda.memory_summary(abbreviated=True))
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        print(f"[DynamicQwen] torch.cuda.max_memory_allocated(): {peak_mb:.2f} MB")

        # Reset per Task 2's design: populated by generate() during the
        # decode loop once it's implemented; not yet meaningful here since
        # no tokens have been generated.
        self.stats = {"fp16_tokens": 0, "4bit_tokens": 0}

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        tokenizer: PreTrainedTokenizerBase,
        max_new_tokens: int,
    ) -> tuple[str, dict, list[str]]:
        """Autoregressively generate text, switching weight precision per token.

        Per-token decode loop:
          1. Prefill the full prompt through FP16 only
             (routing_state.use_4bit = False). This is a FIXED rule, not a
             routing decision -- there is no previous-token entropy signal
             yet. The prefill's last-position logits both (a) argmax to
             token #1 and (b) supply the entropy that gates the very next
             forward pass (the one consuming token #1 to produce token
             #2's logits).
          2. For each subsequent decode step (producing tokens #2..N):
             a. Compute entropy of the PREVIOUS step's output logits via
                routing.calculate_entropy().
             b. Set self.routing_state.use_4bit = (entropy <=
                self.entropy_threshold). One shared RoutingState means
                every DynamicLinear in the model reads the same decision
                for this step -- routing is global-per-token, not per-layer.
             c. Run the forward pass with use_cache=True, passing
                past_key_values from the previous step (manual incremental
                decoding -- NOT model.generate(), since routing_state must
                be mutated between calls).
             d. Greedy-decode (argmax) the next token, matching
                baseline_bench.py's decoding strategy exactly (confirmed:
                baseline_bench.py uses torch.argmax throughout, no
                do_sample) so this stays comparable to the Phase 1
                baseline.
             e. Record the routing decision in self.stats and
                routing_sequence, then stop at max_new_tokens or EOS.

        Token #1's generation is NOT counted in self.stats or
        routing_sequence: it was produced by the fixed-FP16 prefill rule,
        not an entropy-gated routing decision, exactly mirroring how
        baseline_bench.py's TTFT/TPOT split excludes token #1 from
        decode-phase measurements (see generate_with_timing there).
        self.stats is reset to {"fp16_tokens": 0, "4bit_tokens": 0} at the
        start of this call.

        Returns:
            (generated_text, stats, routing_sequence) where stats is a
            dict copy of self.stats and routing_sequence is a list of
            "fp16"/"4bit" strings, one per decode step (tokens #2..N),
            in generation order.
        """
        self.stats = {"fp16_tokens": 0, "4bit_tokens": 0}
        routing_sequence: list[str] = []
        # Diagnostic-only (not part of the design contract's return value):
        # the raw entropy seen at each decode step, exposed so callers can
        # inspect the actual entropy distribution when picking a threshold.
        self.last_entropies: list[float] = []

        inputs = tokenizer(prompt, return_tensors="pt").to(self.model.device)
        eos_token_id = tokenizer.eos_token_id

        # --- Step 1: prefill, FP16 only (fixed rule, not a routing decision). ---
        self.routing_state.use_4bit = False
        outputs = self.model(inputs["input_ids"], use_cache=True)
        past_key_values = outputs.past_key_values
        prev_logits = outputs.logits[:, -1, :]

        generated_ids: list[int] = []
        next_token_id = torch.argmax(prev_logits, dim=-1, keepdim=True)
        generated_ids.append(next_token_id.item())

        # --- Step 2: decode loop for tokens #2..max_new_tokens. ---
        if next_token_id.item() != eos_token_id:
            cur_input = next_token_id
            for _ in range(1, max_new_tokens):
                # 2a/2b: entropy of the PREVIOUS step's logits gates this step's routing.
                entropy = calculate_entropy(prev_logits)  # shape [batch]=[1]
                use_4bit = bool((entropy <= self.entropy_threshold).item())
                self.routing_state.use_4bit = use_4bit
                self.last_entropies.append(entropy.item())

                route_label = "4bit" if use_4bit else "fp16"
                routing_sequence.append(route_label)
                self.stats["4bit_tokens" if use_4bit else "fp16_tokens"] += 1

                # 2c: incremental forward pass via the KV-cache.
                outputs = self.model(cur_input, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values
                prev_logits = outputs.logits[:, -1, :]

                # 2d: greedy decode.
                next_token_id = torch.argmax(prev_logits, dim=-1, keepdim=True)
                generated_ids.append(next_token_id.item())
                cur_input = next_token_id

                # 2e: stop at EOS.
                if eos_token_id is not None and next_token_id.item() == eos_token_id:
                    break

        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        self.last_past_key_values = past_key_values  # exposed for KV-cache dtype verification
        # Diagnostic-only (not part of the design contract's return value),
        # same pattern as self.last_entropies: exposes the per-token ids
        # already built above so callers can align individual tokens with
        # routing_sequence (generated_ids[0] is the fixed-FP16 prefill
        # token with no routing decision; generated_ids[i] for i>=1
        # corresponds to routing_sequence[i-1]).
        self.last_generated_ids = generated_ids
        return generated_text, dict(self.stats), routing_sequence
