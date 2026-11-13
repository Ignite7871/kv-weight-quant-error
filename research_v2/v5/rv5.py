"""v5 core (PROTOCOL_V5.md sections 2, 3 and 5): the seven units, their seeds, the NF4 and HQQ dual-weight backends on the v5 model manifest,
the WikiText-2 task layer, and the record form that keeps WikiText text and token ids out of every released file (D-V5-5).

The step, the six core arms, the BF16 rule, the X13a sums and the calibration accumulator are the locked v4 ones (rv4.py, imported unchanged and
hash-checked against the v4 lock by the tests). HQQ is the v4 r3 wrapper, restored byte-for-byte as rv5_hqq.py.

Seeds. U0 (the anchor) uses the v4 master label and the v4 tier labels, so its C1 draws equal v4's and its records can be compared with v4
bitwise. Every other unit uses the master label 'rv5-2026-11-11|<quantizer>|<task>' and the v5 tier labels.

WikiText-2 (section 3). Paragraph: a row of the pinned parquet split whose text is non-empty after strip() and is not a heading (' = ... = ').
The paragraph text is the stripped row. Prompt: the first 600 characters cut back to the last whitespace at or before position 600 (any run of
whitespace before that cut stays with the continuation). History: the first 65 tokens of the continuation (it starts with whitespace),
tokenized separately with no special tokens. Eligible: at least 1,200 characters and at least 65 continuation tokens under the tokenizer of
every D model. Order: SHA-256 of '<label>|<row index>'. Every module-level helper can be replaced by the mutation check.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RV = HERE.parent
sys.path.insert(0, str(RV))
sys.path.insert(0, str(RV / "v4"))
sys.path.insert(0, str(HERE))
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"              # before anything imports torch (v4 r4)

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402  (HF before torch)

import torch  # noqa: E402

import rv3.core as V  # noqa: E402
import rv4  # noqa: E402  (locked v4 core, unchanged)
from dynamic_linear import RoutingState  # noqa: E402  (historical, unmodified)
from rv2.engine import Backend  # noqa: E402

MANIFEST = HERE / "V5_MODEL_MANIFEST.json"
V5_MASTER = "rv5-2026-11-11"
ANCHOR = "U0"
UNITS = {  # unit: (model, quantizer, task, component)
    "U0": ("qwen2.5-0.5b", "nf4", "gsm8k", "anchor"),
    "U1": ("qwen3-0.6b-base", "nf4", "gsm8k", "A"),
    "U2": ("smollm2-1.7b", "nf4", "gsm8k", "A"),
    "U3": ("qwen2.5-0.5b", "nf4", "wikitext2", "D"),
    "U4": ("qwen2.5-0.5b", "hqq", "gsm8k", "B"),
    "U5": ("mistral-7b-v0.3", "nf4", "wikitext2", "D"),
    "U6": ("mistral-7b-v0.3", "hqq", "gsm8k", "B"),
}
ORDER = list(UNITS)
NEVER_CUT = ("U0", "U1", "U2")                                # section 9: A and the anchor are never cut
TIERS_V4 = {"calibration": "V4calib", "test": "V4A", "gates": "V4gate", "probe": "V4g6"}
TIERS_V5 = {"calibration": "V5calib", "test": "V5A", "gates": "V5gate", "probe": "V5probe"}
CORE_ARMS = rv4.CORE_ARMS
CAL_ARMS = rv4.CAL_ARMS
DTYPES = rv4.DTYPES

WIKI_MIN_CHARS, WIKI_PROMPT_CHARS, WIKI_HIST = 1200, 600, 65
WIKI_ORDER_LABELS = {"test": "rv5-wikitext2-test-order-2026-11-11", "train": "rv5-wikitext2-train-order-2026-11-11"}
D_MODELS = ("qwen2.5-0.5b", "mistral-7b-v0.3")


def manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def models() -> dict:
    return manifest()["models"]


def dataset_entry() -> dict:
    return manifest()["datasets"]["wikitext2"]


def unit(u: str) -> dict:
    m, q, task, comp = UNITS[u]
    return {"unit": u, "model": m, "quantizer": q, "task": task, "component": comp}


# ----------------------------------------------------------------------------- seeds
def master_label(u: str) -> str:
    if u == ANCHOR:
        return rv4.MASTER_SEED
    _, q, task, _ = UNITS[u]
    return f"{V5_MASTER}|{q}|{task}"


def tiers(u: str) -> dict:
    return TIERS_V4 if u == ANCHOR else TIERS_V5


def arm_seeds(u: str, arm: str, alpha: float, index: int, tier: str):
    """(main, value-side) seeds by label only, as v4. For the anchor this is exactly rv4.arm_seeds (tested)."""
    m, mas = UNITS[u][0], master_label(u)
    return V.seed_for(m, arm, alpha, index, tier, master=mas), V.seed_for(m, f"{arm}-V", alpha, index, tier, master=mas)


def probe_seeds(model_id: str, quantizer: str, arm: str, alpha: float, index: int):
    mas = f"{V5_MASTER}|probe|{quantizer}"
    return V.seed_for(model_id, arm, alpha, index, TIERS_V5["probe"], master=mas), V.seed_for(model_id, f"{arm}-V", alpha, index, TIERS_V5["probe"], master=mas)


# ----------------------------------------------------------------------------- backends
def build_backend(model, device: str, precision: str = "fp16", quantizer: str = "nf4", hqq_frozen=None) -> Backend:
    """NF4: exactly rv4.build_backend (FP16 + NF4 is the v3 construction). HQQ: the v4 r3 construction (model cast and moved, then every target
    linear replaced by HQQDynamicLinear in the locked walk order)."""
    if quantizer == "nf4":
        if hqq_frozen is not None:
            raise ValueError("a frozen HQQ state with the NF4 quantizer")
        return rv4.build_backend(model, device, precision, "nf4")
    if quantizer != "hqq":
        raise ValueError(quantizer)
    import rv5_hqq as H
    dt = DTYPES[precision]
    model = model.to(dt).to(device).eval()
    state = RoutingState()
    names = H.replace_linear_with_hqq(model, state, dt, device, frozen=hqq_frozen)
    assert names, "no layers replaced"
    be = Backend(model, state, device)
    be.precision, be.quantizer, be.n_dynamic = precision, quantizer, len(names)
    return be


def load_tokenizer(model_id: str, local_only: bool = True):
    e = models()[model_id]
    return AutoTokenizer.from_pretrained(e["hf_id"], revision=e["revision"], local_files_only=local_only)


def load_model(model_id: str, device: str, precision: str = "fp16", quantizer: str = "nf4", local_only: bool = True, hqq_frozen=None):
    """The pinned snapshot of a v5 manifest model, as a dual-weight backend, and its tokenizer."""
    e = models()[model_id]
    model = AutoModelForCausalLM.from_pretrained(e["hf_id"], revision=e["revision"], dtype=DTYPES[precision], local_files_only=local_only)
    return build_backend(model, device, precision, quantizer, hqq_frozen), load_tokenizer(model_id, local_only)


def attention_snapshot(model) -> dict:
    """G7 (section 5): per-head query/key RMSNorm modules (Qwen3's QK-norm) and the KV-head count of the first attention block."""
    names = [n for n, _ in model.named_modules()]
    att = next(m for n, m in model.named_modules() if n.endswith("layers.0.self_attn"))
    return {"q_norm": any(n.endswith("self_attn.q_norm") for n in names), "k_norm": any(n.endswith("self_attn.k_norm") for n in names),
            "n_q_norm": sum(n.endswith("self_attn.q_norm") for n in names), "n_k_norm": sum(n.endswith("self_attn.k_norm") for n in names),
            "k_proj_out": int(att.k_proj.out_features if hasattr(att.k_proj, "out_features") else att.k_proj.weight.shape[0])}


def attention_matches(snap: dict, geom: dict) -> bool:
    L = geom["layers"]
    want = geom["qk_norm"]
    return bool(snap["q_norm"] == want and snap["k_norm"] == want and snap["n_q_norm"] == (L if want else 0)
                and snap["n_k_norm"] == (L if want else 0) and snap["k_proj_out"] == geom["kv_heads"] * geom["head_dim"])


# ----------------------------------------------------------------------------- ids and hashes
def sha_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def ids_sha(ids) -> str:
    return hashlib.sha256(json.dumps([int(x) for x in ids]).encode()).hexdigest()


# ----------------------------------------------------------------------------- WikiText-2 task layer
def is_heading(s: str) -> bool:
    return s.startswith("=") and s.endswith("=")


def paragraphs(lines) -> list:
    """[(row index, paragraph text)] of a parquet split: rows non-empty after strip() that are not headings."""
    out = []
    for i, line in enumerate(lines):
        s = line.strip()
        if s and not is_heading(s):
            out.append((i, s))
    return out


def split_prompt(text: str):
    """(prompt, continuation). The cut is the last whitespace character at or before position 600; whitespace just before it moves to the
    continuation, so the prompt never ends with whitespace and the continuation always starts with it."""
    head = text[:WIKI_PROMPT_CHARS + 1]
    ws = [i for i, c in enumerate(head) if c.isspace()]
    if not ws or ws[-1] == 0:
        raise ValueError("no whitespace to cut at within the first 600 characters")
    j = ws[-1]
    while j > 0 and text[j - 1].isspace():
        j -= 1
    if j == 0:
        raise ValueError("empty prompt")
    return text[:j], text[j:]


def history_ids(tok, rest: str) -> list:
    return [int(x) for x in tok(rest, add_special_tokens=False).input_ids[:WIKI_HIST]]


def prompt_token_ids(tok, prompt: str) -> list:
    """The prompt as the model sees it: default special tokens (BOS where the tokenizer adds one), as the GSM8K prompts in v4."""
    return [int(x) for x in tok(prompt).input_ids]


def order_key(split: str, idx: int) -> str:
    return hashlib.sha256(f"{WIKI_ORDER_LABELS[split]}|{idx}".encode()).hexdigest()


def ordered(paras, split: str) -> list:
    return sorted(paras, key=lambda p: order_key(split, p[0]))


def parquet_path(split: str, local_only: bool = True) -> Path:
    from huggingface_hub import hf_hub_download
    d = dataset_entry()
    return Path(hf_hub_download(d["repo"], d["files"][split]["path"], revision=d["revision"], repo_type="dataset", local_files_only=local_only,
                                token=False))


def wikitext_lines(split: str, local_only: bool = True) -> list:
    import pyarrow.parquet as pq
    return pq.read_table(parquet_path(split, local_only)).column("text").to_pylist()


def eligible(text: str, toks: dict):
    """(eligible, per-tokenizer details). toks: model id -> tokenizer of every D model."""
    if len(text) < WIKI_MIN_CHARS:
        return False, None
    prompt, rest = split_prompt(text)
    det = {}
    for m, tok in toks.items():
        h = history_ids(tok, rest)
        if len(h) < WIKI_HIST:
            return False, None
        p = prompt_token_ids(tok, prompt)
        det[m] = {"prompt_ids_sha256": ids_sha(p), "prompt_len": len(p), "history_ids_sha256": ids_sha(h), "history_len": len(h)}
    return True, det


def sample_item(idx: int, text: str, det: dict) -> dict:
    prompt, rest = split_prompt(text)
    return {"index": idx, "text_sha256": sha_text(text), "chars": len(text), "prompt_chars": len(prompt), "prompt_sha256": sha_text(prompt),
            "continuation_sha256": sha_text(rest), "tok": det}


def build_wikitext_sample(lines_by_split: dict, toks: dict, n_test=100, n_cal=50, n_timing=3) -> dict:
    """The locked WikiText sample: hashes and row indices only (D-V5-5). lines_by_split: {'test': rows, 'train': rows}."""
    d = dataset_entry()
    out = {"what": "PROTOCOL_V5 section 3 WikiText-2 sample: row indices and SHA-256 hashes only; no text, no token ids (D-V5-5)",
           "repo": d["repo"], "config": d["config"], "revision": d["revision"], "files": d["files"],
           "rules": {"min_chars": WIKI_MIN_CHARS, "prompt_chars": WIKI_PROMPT_CHARS, "history_tokens": WIKI_HIST, "order_labels": WIKI_ORDER_LABELS,
                     "d_models": list(toks)},
           "counts": {}}
    for split, take in (("test", n_test), ("train", n_cal + n_timing)):
        paras = paragraphs(lines_by_split[split])
        chosen, n_long, n_elig = [], 0, 0
        for idx, text in ordered(paras, split):
            if len(text) >= WIKI_MIN_CHARS:
                n_long += 1
            ok, det = eligible(text, toks)
            if not ok:
                continue
            n_elig += 1
            if len(chosen) < take:
                chosen.append(sample_item(idx, text, det))
        if len(chosen) < take:
            raise ValueError(f"{split}: only {len(chosen)} eligible paragraphs")
        out["counts"][split] = {"rows": len(lines_by_split[split]), "paragraphs": len(paras), "at_least_min_chars": n_long, "eligible": n_elig}
        if split == "test":
            out["test"] = chosen
        else:
            out["calibration"], out["timing"] = chosen[:n_cal], chosen[n_cal:]
    return out


class WikiData:
    """The pinned parquet splits, read once. problem(split, idx) -> (text, prompt, continuation)."""

    def __init__(self, local_only: bool = True):
        self.lines = {s: wikitext_lines(s, local_only) for s in ("test", "train")}

    def problem(self, split: str, idx: int):
        text = self.lines[split][idx].strip()
        prompt, rest = split_prompt(text)
        return text, prompt, rest


# ----------------------------------------------------------------------------- the record form (D-V5-5)
def public_record(rec: dict, task: str) -> dict:
    """WikiText units: the per-step argmax token ids are replaced by their SHA-256 (they can reproduce the text under teacher forcing)."""
    if task != "wikitext2":
        return rec
    out = dict(rec)
    out["top_sha256"] = ids_sha(out.pop("top"))
    return out


def history_record(idx: int, ids, task: str) -> dict:
    if task == "wikitext2":
        return {"index": idx, "n": len(ids), "ids_sha256": ids_sha(ids)}
    return {"index": idx, "ids": [int(x) for x in ids]}


def history_matches(stored: dict, ids, task: str) -> bool:
    return stored == history_record(stored["index"], ids, task)


# ----------------------------------------------------------------------------- HQQ checks (GH)
def hqq_oracle_ok(model) -> dict:
    """Every HQQ layer: the independent dequantization oracle equals HQQLinear.dequantize() bitwise."""
    import rv5_hqq as H
    bad = [n for n, m in H.hqq_modules(model) if not torch.equal(H.dequant_oracle(m), m.hqq.dequantize())]
    return {"n_layers": len(H.hqq_modules(model)), "mismatches": bad[:10], "pass": not bad and len(H.hqq_modules(model)) > 0}


def hqq_config_ok(model) -> bool:
    import rv5_hqq as H
    mods = H.hqq_modules(model)
    return bool(mods) and all(m.hqq.meta["nbits"] == 4 and m.hqq.meta["group_size"] == 64 and m.hqq.meta["axis"] == 1 for _, m in mods)


def hqq_version() -> str:
    from importlib.metadata import version
    return version("hqq")
