"""Mutation check for the Llama E1 replication: the 15 Qwen E1 mutants plus Llama-geometry mutants, all run against the Llama suite (tiny GQA Llama).
  venv/Scripts/python.exe research_v2/tests/mutation_check_e1l.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

import mutation_check_controls as M
import run_e1l_tests as E
import rv2.controls as C


def mut_write_kv_swapped():
    def f(cache, L, stacked):
        for l, lay in enumerate(cache.layers):
            lay.keys[:, :, L:L + 1, :] = stacked[2 * l + 1].unsqueeze(0).unsqueeze(2)
            lay.values[:, :, L:L + 1, :] = stacked[2 * l].unsqueeze(0).unsqueeze(2)
    C._write_back = f


def mut_write_head_rolled():
    def f(cache, L, stacked):
        for l, lay in enumerate(cache.layers):
            lay.keys[:, :, L:L + 1, :] = torch.roll(stacked[2 * l], 1, dims=0).unsqueeze(0).unsqueeze(2)
            lay.values[:, :, L:L + 1, :] = torch.roll(stacked[2 * l + 1], 1, dims=0).unsqueeze(0).unsqueeze(2)
    C._write_back = f


def mut_stored_as_float32_view():
    o = M.ORIG["_stored_entry"]
    C._stored_entry = lambda arm, alpha, kF, kN, gen: o(arm, alpha, kF, kN, gen).float().half().float().half() * (1.0 + 2 ** -9 if arm != "NAT" else 1.0)
    # a tiny systematic scaling of every perturbed entry (stored values no longer kF + delta)


def mut_norm_over_heads_not_per_head():
    o = M.ORIG["_slot_norm"]
    C._slot_norm = lambda x: x.float().norm(dim=(-2, -1), keepdim=True).expand(*x.shape[:-1], 1)


M.ORIG["_stack_slices"] = C._stack_slices
extra = [("write_kv_swapped", mut_write_kv_swapped), ("write_head_rolled", mut_write_head_rolled),
         ("stored_value_scaled_by_epsilon", mut_stored_as_float32_view), ("norm_pooled_over_kv_heads", mut_norm_over_heads_not_per_head)]


def main():
    E.install()
    names = E.register_into_T()
    M.T = E.T
    M.NAMES = names
    M.MUTANTS = list(M.MUTANTS) + extra
    base = M.run_suite()
    print(f"baseline (unmutated) failures: {len(base)}")
    if base:
        for n, m in base:
            print("  ", n, m)
        sys.exit(2)
    missed = []
    for name, mut in M.MUTANTS:
        for k, v in M.ORIG.items():
            setattr(C, k, v)
        mut()
        fails = M.run_suite()
        for k, v in M.ORIG.items():
            setattr(C, k, v)
        print(f"{'CAUGHT' if fails else 'MISSED':7} {name:34} failing: {', '.join(n for n, _ in fails)[:110] or '-'}")
        if not fails:
            missed.append(name)
    print(f"\n{len(M.MUTANTS) - len(missed)}/{len(M.MUTANTS)} mutants caught")
    sys.exit(1 if missed else 0)


if __name__ == "__main__":
    main()
