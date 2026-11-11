"""Real-model CPU checks (same as tests/run_v3_tests.py real_checks) for the remaining models, using the locked code unchanged."""
import json, sys
from pathlib import Path
RV = Path(r"D:\Projects\Dynamic Precision Scaling\entropy-dps-project\research_v2"); sys.path.insert(0, str(RV)); sys.path.insert(0, str(RV / "tests"))
import run_v3 as R
import e1l_support as LS
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, rv3.core as V
rep = {}
for label in ("qwen-instruct", "llama-base", "llama-instruct"):
    mid, fam = R.MODELS[label], R.family(label)
    b = LS.build_backend(AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16), "cpu")
    tok = AutoTokenizer.from_pretrained(mid)
    p = tok("Question: Tom has 3 apples and buys 4 more. How many apples does he have?\nAnswer:", return_tensors="pt").input_ids
    ref = R.fp16_history(b, tok, p)[:5]; Sr = len(ref) - 1
    geo = LS.cache_geometry(b, p); ctx = V.synthetic_ctx(b.model.config, "cpu"); cap = V.QueryCapture(b.model)
    rl = V.reference_logits_v3(b, p, ref, Sr); cF = V.fp16_teacher_forced_cache(b, p, ref, Sr)
    a0, errs = [], []
    for arm in V.TEST_ARMS:
        r0, c0 = V.teacher_forced_arm_v3(b, p, ref, Sr, arm, 0.0, V.arm_seeds("real", arm, 0.0, 0, "T"), rl, ctx, cap, False, return_cache=True)
        a0.append(all(x == 0 for x in r0["js"]) and all(torch.equal(x.keys, y.keys) and torch.equal(x.values, y.values) for x, y in zip(c0.layers, cF.layers)))
        errs += R.integrity_errors({arm: V.teacher_forced_arm_v3(b, p, ref, Sr, arm, 1.0, V.arm_seeds("real", arm, 1.0, 0, "T"), rl, ctx, cap, True)})
    mp = {mode: V.routing_map(b.model, mode) for mode in ("LOC", "INH")}
    n_dyn = len(V.dynamic_modules(b.model))
    rep[label] = {"hf_id": mid, "n_layers": geo["n_layers"], "n_kv_heads": geo["n_kv_heads"], "n_q_heads": b.model.config.num_attention_heads,
                  "head_dim": geo["head_dim"], "cache_fp16": geo["all_fp16"], "routed_modules": n_dyn,
                  "LOC_nf4_modules": sum(mp["LOC"].values()), "INH_nf4_modules": sum(mp["INH"].values()),
                  "alpha0_bitwise_all_14_arms": all(a0), "integrity_errors_alpha1": errs,
                  "pass": bool(all(a0) and not errs and n_dyn == R.N_DYNAMIC[fam] and (geo["n_layers"], geo["n_kv_heads"], geo["head_dim"]) == R.GEOMETRY[fam] and geo["all_fp16"])}
    cap.remove(); del b
print(json.dumps(rep, indent=1))
