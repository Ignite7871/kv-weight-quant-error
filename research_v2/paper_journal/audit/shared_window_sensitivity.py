"""Exploratory, CPU-only: compare the Qwen (E1) and Llama (E1L) Tier A reference histories and re-estimate the Tier A contrasts on windows that are
shared by both models. Uses only stored records; does not touch the locked files (read-only) and replaces no original result.

  venv/Scripts/python.exe research_v2/paper_journal/audit/shared_window_sensitivity.py
Writes audit/SHARED_WINDOW_SENSITIVITY.json and .md.

Why 'shared window' and not 'shared token prefix': the two models have different tokenizers and generate different texts, so a token-level
prefix common to both histories does not exist. What differs between the experiments is the number of scored steps S_i per problem
(Qwen: first 64 steps of a stored free-running generation; Llama: first 64 steps of a capped in-session generation). The sensitivity analyses therefore
restrict both models to the same step windows / problem sets (defined from stored S_i only, never from outcomes).
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RV = HERE.parents[1]
FLOOR = 1e-9
B = 10_000
SEED = 20261201      # new stream, distinct from the locked seed 20261101 and the audit seed 777


def load(root, tag):
    out = {}
    for line in open(RV / root / "tierA" / f"{tag}.jsonl", encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            out[r["index"]] = r
    return out


ARMS = {"NAT": "NAT", "C1": "C1_a1.0", "C2": "C2_a1.0"}
data = {m: {a: load(root, tag) for a, tag in ARMS.items()} for m, root in (("Qwen", "e1_runs"), ("Llama", "e1l_runs"))}
idx = sorted(set(data["Qwen"]["NAT"]) & set(data["Llama"]["NAT"]))
assert len(idx) == 200, len(idx)
S = {m: np.array([data[m]["NAT"][i]["S"] for i in idx]) for m in data}
reflen = {m: np.array([data[m]["NAT"][i]["ref_len"] for i in idx]) for m in data}
for m in data:   # all arms of a model share the same history, hence the same S
    for a in ARMS:
        assert all(data[m][a][i]["S"] == S[m][k] for k, i in enumerate(idx))


def logm(m, arm, k_i):
    """log of per-problem mean JS over steps 1..k_i[i] (floored), for model m."""
    return np.array([np.log(max(float(np.mean(data[m][arm][i]["js"][:int(k)])), FLOOR)) for i, k in zip(idx, k_i)])


def contrasts(m, k_i, mask):
    nat, c1, c2 = (logm(m, a, k_i) for a in ("NAT", "C1", "C2"))
    return {"NAT-C1": (nat - c1)[mask], "NAT-C2": (nat - c2)[mask], "C2-C1": (c2 - c1)[mask]}


def boot(x, rng_idx):
    return x[rng_idx].mean(axis=1)


def summarize_pair(k_q, k_l, mask, label):
    """Paired problem-clustered bootstrap, with the same resampled problems for both models (so the between-model difference is paired)."""
    n = int(mask.sum())
    cq, cl = contrasts("Qwen", k_q, mask), contrasts("Llama", k_l, mask)
    rng = np.random.default_rng(SEED)
    ridx = rng.integers(0, n, size=(B, n))
    res = {"label": label, "n": n}
    for c in ("NAT-C1", "NAT-C2", "C2-C1"):
        for m, cc in (("Qwen", cq), ("Llama", cl)):
            bm = boot(cc[c], ridx)
            res[f"{m} {c}"] = {"estimate": float(cc[c].mean()), "ci95": [float(x) for x in np.percentile(bm, [2.5, 97.5])]}
    for c in ("NAT-C1", "NAT-C2"):
        d = cl[c] - cq[c]
        bm = boot(d, ridx)
        res[f"Llama-Qwen {c}"] = {"estimate": float(d.mean()), "ci95": [float(x) for x in np.percentile(bm, [2.5, 97.5])]}
    return res


results = []
full = np.ones(len(idx), bool)
# 0. original windows (reproduces the locked primary; paired bootstrap for the between-model difference)
results.append(summarize_pair(S["Qwen"], S["Llama"], full, "original windows (each model's own S_i): reproduces the locked estimates"))
# 1. shared window per problem: steps 1..min(S_q, S_l)
K = np.minimum(S["Qwen"], S["Llama"])
results.append(summarize_pair(K, K, K >= 1, "shared per-problem window: steps 1..min(S_Qwen, S_Llama)"))
# 2. fixed 32-step window on problems with >= 32 scored steps in both
m32 = (S["Qwen"] >= 32) & (S["Llama"] >= 32)
results.append(summarize_pair(np.full(len(idx), 32), np.full(len(idx), 32), m32, "steps 1-32, problems with S>=32 in both models"))
# 3. full 64-step window on problems with S=64 in both
m64 = (S["Qwen"] == 64) & (S["Llama"] == 64)
results.append(summarize_pair(np.full(len(idx), 64), np.full(len(idx), 64), m64, "steps 1-64, problems with S=64 in both models"))
# 4. fixed 16-step window (all problems with S>=16 in both)
m16 = (S["Qwen"] >= 16) & (S["Llama"] >= 16)
results.append(summarize_pair(np.full(len(idx), 16), np.full(len(idx), 16), m16, "steps 1-16, problems with S>=16 in both models"))

# history comparison (descriptive)
hist = {}
for m in data:
    hist[m] = {"S_mean": float(S[m].mean()), "S_median": float(np.median(S[m])), "share_S64": float((S[m] == 64).mean()), "S_min": int(S[m].min()),
               "ref_len_mean": float(reflen[m].mean())}
ranks = lambda x: np.argsort(np.argsort(x))   # noqa: E731
hist["spearman_S_Qwen_Llama"] = float(np.corrcoef(ranks(S["Qwen"]), ranks(S["Llama"]))[0, 1])
hist["n_both_S64"] = int(m64.sum())
hist["n_both_S_ge32"] = int(m32.sum())
hist["n_llama_window_shorter"] = int((S["Llama"] < S["Qwen"]).sum())
hist["n_equal_window"] = int((S["Llama"] == S["Qwen"]).sum())
hist["n_qwen_window_shorter"] = int((S["Qwen"] < S["Llama"]).sum())

# text-level look at shared prefixes (descriptive; needs both tokenizers locally; skipped if unavailable)
text = {"available": False}
try:
    from transformers import AutoTokenizer
    tq = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", local_files_only=True)
    tl = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", local_files_only=True)
    fq = {}
    for line in open(RV / "e1_runs" / "tierB" / "FP16.jsonl", encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            fq[r["index"]] = r["ids"][:65]
    fl = {json.loads(l)["index"]: json.loads(l)["ids"][:65] for l in open(RV / "e1l_runs" / "tierA" / "FP16_ref.jsonl", encoding="utf-8") if l.strip()}
    cps = []
    for i in idx:
        a = tq.decode(fq[i]).strip()
        b = tl.decode(fl[i]).strip()
        n = 0
        while n < min(len(a), len(b)) and a[n] == b[n]:
            n += 1
        cps.append(n)
    cps = np.array(cps)
    text = {"available": True, "common_character_prefix_mean": float(cps.mean()), "median": float(np.median(cps)),
            "share_identical_first_20_chars": float((cps >= 20).mean()), "share_identical_first_60_chars": float((cps >= 60).mean())}
except Exception as e:   # noqa: BLE001
    text = {"available": False, "reason": f"{type(e).__name__}: {str(e)[:160]}"}

out = {"seed": SEED, "B": B, "history": hist, "text_prefix": text, "analyses": results}
(HERE / "SHARED_WINDOW_SENSITIVITY.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


def f(v):
    return f"{v['estimate']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]"


md = ["# Exploratory shared-window sensitivity analysis (CPU only; stored records; replaces no original result)", "",
      f"Seed {SEED} (new stream), {B} paired problem-clustered resamples (the same resampled problems for both models, so between-model differences are paired). Log mean-JS contrasts, as in the locked analysis (floor {FLOOR}).", "",
      "## Reference histories", "",
      f"* Qwen: mean S {hist['Qwen']['S_mean']:.1f}, median {hist['Qwen']['S_median']:.0f}, S=64 in {100 * hist['Qwen']['share_S64']:.1f}% of problems, mean stored-generation length {hist['Qwen']['ref_len_mean']:.1f} tokens (capped at 65 in Tier A).",
      f"* Llama: mean S {hist['Llama']['S_mean']:.1f}, median {hist['Llama']['S_median']:.0f}, S=64 in {100 * hist['Llama']['share_S64']:.1f}%, mean capped length {hist['Llama']['ref_len_mean']:.1f}.",
      f"* Per problem: Llama window shorter in {hist['n_llama_window_shorter']}, equal in {hist['n_equal_window']}, longer in {hist['n_qwen_window_shorter']} problems; Spearman correlation of S across models {hist['spearman_S_Qwen_Llama']:+.2f}; both S=64 in {hist['n_both_S64']} problems; both S>=32 in {hist['n_both_S_ge32']}.",
      "* A token-level shared prefix between the two histories is **not defined** (different tokenizers, different generating models).",
      ("* Text level (decoded, common character prefix of the first 65 tokens of each history): mean {:.1f} characters, median {:.0f}; identical for >=20 characters in {:.0f}% and >=60 characters in {:.0f}% of problems.".format(
          text["common_character_prefix_mean"], text["median"], 100 * text["share_identical_first_20_chars"], 100 * text["share_identical_first_60_chars"])
       if text.get("available") else f"* Text-level prefix comparison not run: {text.get('reason')}"), "",
      "## Contrasts on shared windows (NAT-C1 is the locked primary; the others are secondary)", ""]
cols = ["Qwen NAT-C1", "Llama NAT-C1", "Llama-Qwen NAT-C1", "Qwen NAT-C2", "Llama NAT-C2", "Qwen C2-C1", "Llama C2-C1"]
md.append("| Window / problem set | n | " + " | ".join(cols) + " |")
md.append("|---|---|" + "---|" * len(cols))
for r in results:
    md.append(f"| {r['label']} | {r['n']} | " + " | ".join(f(r[c]) for c in cols) + " |")
md += ["", "Reading: see the manuscript's exploratory appendix. These analyses are exploratory, were defined after the original results were known, and none replaces a locked result."]
(HERE / "SHARED_WINDOW_SENSITIVITY.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))
