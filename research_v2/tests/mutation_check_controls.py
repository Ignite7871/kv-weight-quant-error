"""Mutation check for the E1 controls: inject known bugs and confirm tests/run_controls_tests.py fails for each. A suite that passes on correct
code proves little unless it fails on wrong code.   Run: venv/Scripts/python.exe research_v2/tests/mutation_check_controls.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

import run_controls_tests as T
import rv2.controls as C

NAMES = [k for k in dir(T) if k.startswith("c") and k[1:3].isdigit() and callable(getattr(T, k))]
ORIG = {k: getattr(C, k) for k in ("_slot_norm", "_unit_gaussian", "_rademacher", "_stored_entry", "_write_back", "seed_for", "js64",
                                   "controlled_step", "reference_logits")}


def run_suite():
    T.RESULTS.clear()
    for n in NAMES:
        getattr(T, n)()
    return [(n, m) for n, s, m in T.RESULTS if s != "PASS"]


def mut_norm_x1_1():
    C._slot_norm = lambda x: 1.1 * x.float().norm(dim=-1, keepdim=True)


def mut_same_noise_all_slots():
    def f(shape, gen, device):
        u = torch.randn((1,) + tuple(shape[1:]), generator=gen, device=device)
        u = u.expand(shape).clone()
        return u / u.norm(dim=-1, keepdim=True)
    C._unit_gaussian = f


def mut_c2_no_sign_randomization():
    C._rademacher = lambda shape, gen, device: torch.ones(shape)


def mut_c1_direction_equals_D():
    o = ORIG["_stored_entry"]

    def f(arm, alpha, kF, kN, gen):
        if arm == "C1":
            D = kN.float() - kF.float()
            return (kF.float() + alpha * D).to(kF.dtype)
        return o(arm, alpha, kF, kN, gen)
    C._stored_entry = f


def mut_alpha_ignored():
    o = ORIG["_stored_entry"]
    C._stored_entry = lambda arm, alpha, kF, kN, gen: o(arm, 1.0, kF, kN, gen)


def mut_nat_stores_fp16_entry():
    o = ORIG["_stored_entry"]
    C._stored_entry = lambda arm, alpha, kF, kN, gen: kF.clone() if arm == "NAT" else o(arm, alpha, kF, kN, gen)


def mut_write_wrong_layer():
    def f(cache, L, stacked):
        for l, lay in enumerate(cache.layers):
            j = (l + 1) % len(cache.layers)
            lay.keys[:, :, L:L + 1, :] = stacked[2 * j].unsqueeze(0).unsqueeze(2)
            lay.values[:, :, L:L + 1, :] = stacked[2 * j + 1].unsqueeze(0).unsqueeze(2)
    C._write_back = f


def mut_write_keys_only():
    def f(cache, L, stacked):
        for l, lay in enumerate(cache.layers):
            lay.keys[:, :, L:L + 1, :] = stacked[2 * l].unsqueeze(0).unsqueeze(2)
    C._write_back = f


def mut_constant_seed():
    C.seed_for = lambda *a, **k: 0


def mut_revisit_previous_entry():
    o = ORIG["_write_back"]

    def f(cache, L, stacked):
        o(cache, L, stacked)
        if L >= 1:
            for lay in cache.layers:
                lay.keys[:, :, L - 1:L, :] += 1e-2
    C._write_back = f


def mut_perturb_prefill():
    o = ORIG["_write_back"]

    def f(cache, L, stacked):
        o(cache, L, stacked)
        for lay in cache.layers:
            lay.keys[:, :, 0:1, :] += 1e-2
    C._write_back = f


def mut_unkept_pass_is_fp16():
    def f(backend, cur, cache, arm, alpha, gen, diag=True):
        L = cache.get_seq_length()
        backend.run(cur, cache, False)                                   # BUG: the "NF4" pass runs FP16 weights
        kN = C._stack_slices(C._snapshot_last(cache, L))
        cache.crop(L)
        lg_F, _ = backend.run(cur, cache, False)
        kF = C._stack_slices(C._snapshot_last(cache, L))
        C._write_back(cache, L, C._stored_entry(arm, alpha, kF, kN, gen))
        return lg_F, ({"norm_dev_max": torch.tensor(0.0), "norm_dev_mean": torch.tensor(0.0), "rel_scale_mean": torch.tensor(0.0), "target_norm_mean": torch.tensor(0.0)} if diag else {})
    C.controlled_step = f


def mut_token_from_nf4_logits():
    def f(backend, cur, cache, arm, alpha, gen, diag=True):
        L = cache.get_seq_length()
        lg_N, _ = backend.run(cur, cache, True)
        kN = C._stack_slices(C._snapshot_last(cache, L))
        cache.crop(L)
        _, _ = backend.run(cur, cache, False)
        kF = C._stack_slices(C._snapshot_last(cache, L))
        C._write_back(cache, L, C._stored_entry(arm, alpha, kF, kN, gen))
        return lg_N, ({"norm_dev_max": torch.tensor(0.0), "norm_dev_mean": torch.tensor(0.0), "rel_scale_mean": torch.tensor(0.0), "target_norm_mean": torch.tensor(0.0)} if diag else {})
    C.controlled_step = f


def mut_reference_off_by_one():
    o = ORIG["reference_logits"]
    C.reference_logits = lambda backend, prompt_ids, ref_ids, S: o(backend, prompt_ids, [ref_ids[0]] + list(ref_ids[:-1]), S)


def mut_js_is_kl():
    def f(a, b):
        p = torch.softmax(a.double(), -1)
        q = torch.softmax(b.double(), -1)
        return torch.special.xlogy(p, p / q.clamp_min(1e-300)).sum(-1).reshape(())
    C.js64 = f


MUTANTS = [(k[4:], v) for k, v in sorted(globals().items()) if k.startswith("mut_")]


def main():
    base = run_suite()
    print(f"baseline (unmutated) failures: {len(base)}")
    if base:
        for n, m in base:
            print("  ", n, m)
        sys.exit(2)
    missed = []
    for name, mut in MUTANTS:
        for k, v in ORIG.items():
            setattr(C, k, v)
        mut()
        fails = run_suite()
        for k, v in ORIG.items():
            setattr(C, k, v)
        status = "CAUGHT" if fails else "MISSED"
        print(f"{status:7} {name:34} failing tests: {', '.join(n for n, _ in fails) or '-'}")
        if not fails:
            missed.append(name)
    print(f"\n{len(MUTANTS) - len(missed)}/{len(MUTANTS)} mutants caught")
    sys.exit(1 if missed else 0)


if __name__ == "__main__":
    main()
