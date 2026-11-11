"""Stage 1 interventions: decoupled token source and KV-cache source (STAGE1_SPEC.md).

A decode step runs the model on the previously emitted token x_{s-1}. That one forward pass yields BOTH
the next-token logits and the K/V entries of position L (the position of x_{s-1}). A *cell* chooses
independently
    token_source  in {F, N}   which pass's logits select the emitted token x_s (greedy argmax), and
    kv_source     in {F, N}   which pass's K/V entries for position L stay in the cache,
where F = FP16 weights and N = NF4 weights. The prompt prefill is FP16 in every condition (historical
protocol), so "NF4 cache" means NF4-written DECODE positions only.

Both passes of a step start from the SAME cache (length L) and consume the SAME emitted token, so the
cache is never combined across incompatible histories:
    1. run the precision that is NOT kept (appends its entry at L), keep its logits;
    2. crop the cache back to L (a view; bitwise restoration of the pre-step state);
    3. run the kept precision (appends the entry that stays), keep its logits;
    4. the token is the argmax of the token-source logits.
No snapshot/restore is needed (unlike the flip oracle in engine.py, which keeps NF4 entries chosen
retrospectively); the cache after every step is exactly what a single-precision teacher-forced replay
of the emitted history would hold (tested bitwise).

Cells (token, kv): FF = static FP16, NN = static NF4, FN = "cache-only" (FP16 token selection, NF4
cache), NF = "token-only" (NF4 token selection, FP16 cache). FF/NN are implemented here too (both
passes computed, one kept) so that the machinery can be tested against the static arms; production
runs use the cheaper static policies for FF/NN.

Logging (never fed back): per step the argmax of both branches, the JS divergence between them and the
token source/KV source letters. Early stopping, EOS and the 256-token cap follow engine.generate.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from . import PROJECT_ROOT  # noqa: F401
from fragility import js_divergence
from .engine import Backend, GenResult


@dataclass(frozen=True)
class Cell:
    token_source: str      # "F" or "N"
    kv_source: str         # "F" or "N"
    name: str = ""
    kind: str = "factorial"
    deployable: bool = False

    def __post_init__(self):
        assert self.token_source in ("F", "N") and self.kv_source in ("F", "N")
        if not self.name:
            object.__setattr__(self, "name", f"tok{self.token_source}_kv{self.kv_source}")

    def describe(self) -> dict:
        return {"name": self.name, "kind": self.kind, "deployable": False,
                "token_source": self.token_source, "kv_source": self.kv_source}


CACHE_ONLY = Cell("F", "N", "cache_only")     # quantization reaches the output only through stored KV
TOKEN_ONLY = Cell("N", "F", "token_only")     # quantization reaches the output only through token choice


# Three one-line helpers carry the whole intervention semantics. They are module-level functions so that
# tests/mutation_check_stage1.py can inject a wrong version of each and confirm the suite notices.
def _unkept_precision(kv_source: str) -> str:
    """The precision whose pass is computed first and whose KV entry is discarded."""
    return "N" if kv_source == "F" else "F"


def _restore_cache(cache, L: int) -> None:
    """Return the cache to its exact pre-step state (crop is a view; entries below L are untouched)."""
    cache.crop(L)


def _token_logits(token_source: str, kv_source: str, lg_kept, lg_unkept):
    """Logits whose argmax is emitted: the kept pass's if the sources agree, else the other pass's."""
    return lg_kept if token_source == kv_source else lg_unkept


def _branch_logits(kv_source: str, lg_kept, lg_unkept):
    """(FP16-branch logits, NF4-branch logits) for logging: the kept pass is the kv_source precision."""
    return (lg_kept, lg_unkept) if kv_source == "F" else (lg_unkept, lg_kept)


@torch.no_grad()
def generate_cell(backend: Backend, prompt_ids, cell: Cell, max_new_tokens: int, eos_token_id,
                  early_stop_fn=None, log: bool = True, step_hook=None) -> GenResult:
    """Greedy generation for one cell. Returns the engine's GenResult with extra trace keys:
    tF, tN (argmax of the FP16 / NF4 branch at each routed step), js (JS divergence between the
    branches), tok_src / kv_src (letters actually used). routes = kv_src per step (the precision whose
    K/V entry was kept); passes = executed pass order, e.g. "NF" = NF4 pass then FP16 pass.

    step_hook(step, cache_len_before, input_token, pass_letter) is called before every decode pass and
    exists only for tests (it must not mutate anything)."""
    kv, tk = cell.kv_source, cell.token_source
    other = _unkept_precision(kv)
    timing = {"prefill": 0.0, "nf4_pass": 0.0, "fp16_pass": 0.0, "cache_ops": 0.0, "decide": 0.0, "logging": 0.0}
    counts = {"nf4_passes": 0, "fp16_passes": 0, "recomputes": 0, "crops": 0, "restores": 0}
    trace = {k: [] for k in ("tF", "tN", "js", "tok_src", "kv_src")}
    routes, passes = [], []

    t_wall0 = time.perf_counter()
    logits, cache = backend.run(prompt_ids, None, use_4bit=False)         # FP16 prefill, every condition
    ids = [int(torch.argmax(logits, dim=-1).item())]
    stopped_early = False

    if ids[0] != eos_token_id:
        for s in range(1, max_new_tokens):
            L = cache.get_seq_length()
            cur = torch.tensor([[ids[-1]]], device=prompt_ids.device)
            if step_hook:
                step_hook(s, L, ids[-1], other)
            lg_other, _ = backend.run(cur, cache, other == "N")           # 1. precision that is not kept
            _restore_cache(cache, L)                                       # 2. restore the pre-step cache
            counts["crops"] += 1
            if step_hook:
                step_hook(s, cache.get_seq_length(), ids[-1], kv)
            lg_kv, _ = backend.run(cur, cache, kv == "N")                  # 3. precision that is kept
            assert cache.get_seq_length() == L + 1, "cache length invariant violated"
            counts["nf4_passes"] += 1
            counts["fp16_passes"] += 1
            lg_tok = _token_logits(tk, kv, lg_kv, lg_other)                # 4. token-source logits
            lg_f, lg_n = _branch_logits(kv, lg_kv, lg_other)
            top = torch.argmax(lg_tok, dim=-1)
            if log:
                # one device->host transfer carries the token and all logged statistics
                row = torch.stack([top.double().reshape(()), torch.argmax(lg_f, dim=-1).double().reshape(()),
                                   torch.argmax(lg_n, dim=-1).double().reshape(()),
                                   js_divergence(lg_f, lg_n).double().reshape(())]).tolist()
                tok, tf, tn, js = int(row[0]), int(row[1]), int(row[2]), row[3]
                trace["tF"].append(tf); trace["tN"].append(tn); trace["js"].append(js)
                trace["tok_src"].append(tk); trace["kv_src"].append(kv)
            else:
                tok = int(top.item())
            routes.append(kv)
            passes.append(other + kv)
            ids.append(tok)
            if eos_token_id is not None and ids[-1] == eos_token_id:
                break
            if early_stop_fn is not None and early_stop_fn(ids):
                stopped_early = True
                break

    timing["total"] = time.perf_counter() - t_wall0
    return GenResult(ids, routes, passes, trace, timing, counts, stopped_early, cell.describe())
