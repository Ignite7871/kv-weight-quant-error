"""Build v4/V4_MODEL_MANIFEST.json (PROTOCOL_V4 r3 sections 2 and 12): pinned revisions, shard SHA-256, geometry, tied flag and vocab of every
v4 model, from the HF API and pinned bytes only. Network; no weights are downloaded (configs, tokenizers and index files only).

  venv/Scripts/python.exe research_v2/tools/v4_build_manifest.py            write the manifest
  venv/Scripts/python.exe research_v2/tools/v4_build_manifest.py --check    rebuild in memory and compare with the committed manifest

Sources, per model:
  * revision, gated flag, licence, and every file's size / LFS SHA-256 / git blob id: HfApi.model_info(files_metadata=True) (no access needed);
  * readable repos: config.json and tokenizer files at the pinned revision;
  * the two gated Llama repos the corresponding author's account cannot read (Llama-3.2-3B, Llama-3.1-8B): the geometry constants of Meta's
    llama-models/models/sku_list.py (Phase 2 section 3), plus two pinned-byte snapshot checks:
      - model.safetensors.index.json from a public copy whose git blob id equals the pinned file's blob id (identical bytes): tied flag
        (no lm_head.weight) and the layer count;
      - the safetensors header of the shard holding model.embed_tokens.weight, range-read from a public copy whose LFS SHA-256 equals the
        pinned shard's: vocab size and hidden size, and the layer-0 q/k/v projection shapes (head and KV-head counts).
  G7 re-reads config.json on the pod and must agree with every recorded value.
The token is never read or printed by this script; huggingface_hub uses the machine's own credentials for readable gated repos.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

RV = Path(__file__).resolve().parents[1]
OUT = RV / "v4" / "V4_MODEL_MANIFEST.json"

# id -> (hf id, revision, family, licence note, geometry source)
MODELS = {
    "qwen2.5-0.5b": ("Qwen/Qwen2.5-0.5B", "060db6499f32faf8b98477b0a26969ef7d8b9987", "qwen", "apache-2.0"),
    "qwen2.5-1.5b": ("Qwen/Qwen2.5-1.5B", "8faed761d45a263340a0528343f099c05c9a4323", "qwen", "apache-2.0"),
    "qwen2.5-3b": ("Qwen/Qwen2.5-3B", "3aab1f1954e9cc14eb9509a215f9e5ca08227a9b", "qwen", "qwen-research (non-commercial research/evaluation)"),
    "qwen2.5-7b": ("Qwen/Qwen2.5-7B", "d149729398750b98c0af14eb82c78cfe92750796", "qwen", "apache-2.0"),
    "llama3.2-1b": ("meta-llama/Llama-3.2-1B", "4e20de362430cd3b72f300e6b0f18e50e7166e08", "llama", "Llama 3.2 Community (gated)"),
    "llama3.2-3b": ("meta-llama/Llama-3.2-3B", "13afe5124825b4f3751f836b40dafda64c1ed062", "llama", "Llama 3.2 Community (gated)"),
    "llama3.1-8b": ("meta-llama/Llama-3.1-8B", "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b", "llama", "Llama 3.1 Community (gated)"),
    "mistral-7b-v0.3": ("mistralai/Mistral-7B-v0.3", "caa1feb0e54d415e2df31207e5f4e273e33509b1", "mistral", "apache-2.0"),
    "llama3.2-1b-instruct": ("meta-llama/Llama-3.2-1B-Instruct", "9213176726f574b556790deb65791e0c5aa438b6", "llama", "Llama 3.2 Community (gated)"),
}
STEP0_ONLY = {"llama3.2-1b-instruct"}
V3_LABEL = {"qwen2.5-0.5b": "qwen-base", "llama3.2-1b": "llama-base", "llama3.2-1b-instruct": "llama-instruct"}
# Meta llama-models/models/sku_list.py (read 2026-11-02, Phase 2 section 3): dim, n_layers, n_heads, n_kv_heads, rope_theta
META_SKU = {"meta-llama/Llama-3.2-3B": (3072, 28, 24, 8, 500000.0), "meta-llama/Llama-3.1-8B": (4096, 32, 32, 8, 500000.0)}
MIRRORS = {"meta-llama/Llama-3.2-3B": ["unsloth/Llama-3.2-3B"], "meta-llama/Llama-3.1-8B": ["unsloth/Meta-Llama-3.1-8B", "unsloth/Llama-3.1-8B"]}
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.model", "vocab.json", "merges.txt"]
BASE_PATTERNS = ["config.json", "generation_config.json", "model.safetensors.index.json"] + TOKENIZER_FILES


def git_blob(b: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def file_meta(api, hf_id, rev):
    info = api.model_info(hf_id, revision=rev, files_metadata=True)
    if info.sha != rev:
        sys.exit(f"{hf_id}: API returned revision {info.sha}, expected the pinned {rev}")
    files = {s.rfilename: {"size": s.size, "lfs_sha256": s.lfs.sha256 if s.lfs else None, "git_blob": s.blob_id} for s in info.siblings}
    card = info.card_data.to_dict() if info.card_data else {}
    return info, files, card.get("license")


def allow_patterns(files):
    """The HF-format weight shards, their index, config, generation config and tokenizer files. Never consolidated.safetensors
    (Mistral's mistral_inference copy of the same weights)."""
    pats = [p for p in BASE_PATTERNS if p in files]
    pats += sorted(f for f in files if f.startswith("model") and f.endswith(".safetensors"))
    return pats


def read_header(url: str) -> dict:
    import httpx
    with httpx.Client(follow_redirects=True, timeout=120) as c:
        n = struct.unpack("<Q", c.get(url, headers={"Range": "bytes=0-7"}).content[:8])[0]
        return json.loads(c.get(url, headers={"Range": f"bytes=8-{7 + n}"}).content)


def config_geometry(cfg: dict) -> dict:
    hd = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    rope = cfg.get("rope_theta") or (cfg.get("rope_parameters") or {}).get("rope_theta")
    return {"layers": cfg["num_hidden_layers"], "q_heads": cfg["num_attention_heads"], "kv_heads": cfg["num_key_value_heads"], "head_dim": hd,
            "hidden": cfg["hidden_size"], "intermediate": cfg["intermediate_size"], "vocab": cfg["vocab_size"],
            "tied_embeddings": bool(cfg.get("tie_word_embeddings", False)), "rope_theta": rope, "stored_dtype": cfg.get("torch_dtype") or cfg.get("dtype"),
            "sliding_window": cfg.get("sliding_window") if cfg.get("use_sliding_window", True) else None, "model_type": cfg["model_type"]}


def gated_snapshot_checks(api, hf_id, files):
    """Pinned-byte checks for a gated repo this account cannot read (see the module docstring)."""
    from huggingface_hub import hf_hub_download, hf_hub_url
    out = {"method": []}
    idx_meta = files["model.safetensors.index.json"]
    index = None
    for mi in MIRRORS[hf_id]:
        minfo = api.model_info(mi, files_metadata=True)
        msib = {s.rfilename: s for s in minfo.siblings}
        s = msib.get("model.safetensors.index.json")
        if s is None or s.blob_id != idx_meta["git_blob"]:
            continue
        p = hf_hub_download(mi, "model.safetensors.index.json", revision=minfo.sha)
        b = Path(p).read_bytes()
        if git_blob(b) != idx_meta["git_blob"]:
            continue
        index = json.loads(b)
        wm = index["weight_map"]
        shard = wm["model.embed_tokens.weight"]
        ms = msib.get(shard)
        if ms is None or ms.lfs is None or ms.lfs.sha256 != files[shard]["lfs_sha256"]:
            continue
        hdr = read_header(hf_hub_url(mi, shard, revision=minfo.sha))
        layers = 1 + max(int(k.split(".")[2]) for k in wm if k.startswith("model.layers."))
        e = hdr["model.embed_tokens.weight"]
        q, k, v = (hdr[f"model.layers.0.self_attn.{n}_proj.weight"]["shape"] for n in "qkv")
        out.update({"mirror": mi, "mirror_revision": minfo.sha, "index_git_blob": idx_meta["git_blob"], "embed_shard": shard,
                    "embed_shard_sha256": files[shard]["lfs_sha256"], "tied_embeddings": "lm_head.weight" not in wm,
                    "index_n_tensors": len(wm), "layers_from_index": layers, "vocab": e["shape"][0], "hidden": e["shape"][1],
                    "embed_dtype": e["dtype"], "q_proj_shape": q, "k_proj_shape": k, "v_proj_shape": v})
        out["method"] = [f"model.safetensors.index.json of {mi}@{minfo.sha} has the pinned file's git blob id {idx_meta['git_blob']} (identical bytes)",
                         f"safetensors header of {shard} range-read from {mi}@{minfo.sha}, whose LFS SHA-256 equals the pinned shard's"]
        return out
    sys.exit(f"{hf_id}: no public copy with identical pinned bytes was found; the snapshot checks cannot be made")


def build():
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    man = {"what": "PROTOCOL_V4 r3 model manifest: pinned revisions, shard SHA-256 (HF-reported LFS hashes at the pinned revision), geometry, "
                   "tied flag and vocab. Built by tools/v4_build_manifest.py; G7 checks every value on the pod.",
           "models": {}}
    for mid, (hf_id, rev, fam, lic) in MODELS.items():
        info, files, card_lic = file_meta(api, hf_id, rev)
        shards = {f: {"size": m["size"], "sha256": m["lfs_sha256"]} for f, m in files.items() if f.startswith("model") and f.endswith(".safetensors")}
        pats = allow_patterns(files)
        e = {"hf_id": hf_id, "revision": rev, "family": fam, "licence": lic, "card_licence": card_lic, "gated": info.gated or False,
             "step0_only": mid in STEP0_ONLY, "v3_label": V3_LABEL.get(mid), "shards": shards,
             "small_files": {f: {"size": files[f]["size"], "git_blob": files[f]["git_blob"], "lfs_sha256": files[f]["lfs_sha256"]}
                             for f in pats if f in files and f not in shards},
             "allow_patterns": pats, "excluded_files": sorted(f for f in files if f.endswith(".safetensors") and f not in shards)}
        try:
            cfg = json.loads(Path(hf_hub_download(hf_id, "config.json", revision=rev)).read_bytes())
            e["geometry"] = config_geometry(cfg)
            e["geometry_source"] = "config.json at the pinned revision"
            e["snapshot_checks"] = {"tied_embeddings": e["geometry"]["tied_embeddings"], "vocab": e["geometry"]["vocab"],
                                    "method": ["config.json at the pinned revision"]}
        except Exception as ex:                                          # gated repo this account cannot read
            if hf_id not in META_SKU:
                raise
            dim, nl, nh, nkv, rope = META_SKU[hf_id]
            sc = gated_snapshot_checks(api, hf_id, files)
            hd = dim // nh
            consistent = (sc["layers_from_index"] == nl and sc["hidden"] == dim and sc["q_proj_shape"] == [nh * hd, dim]
                          and sc["k_proj_shape"] == [nkv * hd, dim] and sc["v_proj_shape"] == [nkv * hd, dim])
            if not consistent:
                sys.exit(f"{hf_id}: pinned bytes disagree with Meta sku_list geometry: {sc}")
            e["geometry"] = {"layers": nl, "q_heads": nh, "kv_heads": nkv, "head_dim": hd, "hidden": dim, "intermediate": None, "vocab": sc["vocab"],
                             "tied_embeddings": sc["tied_embeddings"], "rope_theta": rope, "stored_dtype": "bfloat16" if sc["embed_dtype"] == "BF16" else sc["embed_dtype"],
                             "sliding_window": None, "model_type": "llama"}
            e["geometry_source"] = (f"Meta llama-models sku_list.py, confirmed by pinned bytes (layer count from the index; hidden, vocab and the layer-0 "
                                    f"q/k/v shapes from the shard header); config.json not readable with this account ({type(ex).__name__})")
            e["snapshot_checks"] = sc
        man["models"][mid] = e
    return man


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    man = build()
    text = json.dumps(man, indent=1, sort_keys=False) + "\n"
    if a.check:
        same = OUT.exists() and json.loads(OUT.read_text(encoding="utf-8")) == man
        print("manifest matches a fresh build" if same else "MANIFEST DIFFERS FROM A FRESH BUILD")
        sys.exit(0 if same else 1)
    OUT.write_text(text, encoding="utf-8", newline="\n")
    for mid, e in man["models"].items():
        g = e["geometry"]
        print(f"{mid:22s} {e['revision'][:8]} L{g['layers']} {g['q_heads']}/{g['kv_heads']} hd{g['head_dim']} vocab {g['vocab']} tied {g['tied_embeddings']} "
              f"shards {len(e['shards'])} excluded {e['excluded_files']}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
