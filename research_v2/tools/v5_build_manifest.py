"""Build v5/V5_MODEL_MANIFEST.json (PROTOCOL_V5 sections 2 and 11): the pinned revisions, shard SHA-256, geometry, tokenizer behaviour and
parameter counts of the v5 models, and the pinned WikiText-2 parquet files. Network (HF API) plus the local pinned snapshots.

  venv/Scripts/python.exe research_v2/tools/v5_build_manifest.py            write the manifest
  venv/Scripts/python.exe research_v2/tools/v5_build_manifest.py --check    rebuild in memory and compare with the committed manifest

Sources:
  * qwen2.5-0.5b and mistral-7b-v0.3: the v4 manifest entries, copied unchanged (the v4 pins, section 2), plus the v5 fields below;
  * qwen3-0.6b-base and smollm2-1.7b: HfApi.model_info(files_metadata=True) at the pinned revision (sizes, LFS SHA-256, git blob ids, gated
    flag, card licence) and config.json at the pinned revision (tools/v4_build_manifest.py functions, unchanged);
  * v5 fields of every model, from the pinned local snapshot (G7 re-checks them on the pod): 'qk_norm' (q_norm / k_norm tensors in the
    safetensors header), 'params' (total and transformer-linear parameter counts from the header shapes; Mistral, whose 14.5 GB of shards are
    not on the laptop, from the header range-read of each pinned shard), and 'expects_bos' (whether the pinned tokenizer prepends BOS);
  * WikiText-2: HfApi.dataset_info(files_metadata=True) at the pinned revision (parquet sizes and LFS SHA-256).
The local snapshot's shard SHA-256 must equal the HF-reported one, or the build stops.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

RV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RV / "tools"))
import v4_build_manifest as B  # noqa: E402  (locked v4 tool, unchanged)

OUT = RV / "v5" / "V5_MODEL_MANIFEST.json"
V4_MAN = RV / "v4" / "V4_MODEL_MANIFEST.json"
REUSED = ("qwen2.5-0.5b", "mistral-7b-v0.3")
NEW = {"qwen3-0.6b-base": ("Qwen/Qwen3-0.6B-Base", "da87bfb608c14b7cf20ba1ce41287e8de496c0cd", "qwen3"),
       "smollm2-1.7b": ("HuggingFaceTB/SmolLM2-1.7B", "effd688a12921b4cc83e3312b6feb579f70f9c71", "smollm2")}
WIKI = {"repo": "Salesforce/wikitext", "config": "wikitext-2-raw-v1", "revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
        "splits": {"test": "wikitext-2-raw-v1/test-00000-of-00001.parquet", "train": "wikitext-2-raw-v1/train-00000-of-00001.parquet"}}
LINEAR = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def local_header(p: Path) -> dict:
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def remote_header(hf_id, rev, shard) -> dict:
    from huggingface_hub import hf_hub_url
    return B.read_header(hf_hub_url(hf_id, shard, revision=rev))


def header_facts(headers: list) -> dict:
    """Parameter counts and QK-norm tensors from the safetensors headers of every pinned shard."""
    tot = lin = 0
    qn = kn = 0
    for h in headers:
        for k, v in h.items():
            if k == "__metadata__":
                continue
            n = 1
            for d in v["shape"]:
                n *= d
            tot += n
            if k.endswith(".weight") and any(f".{x}." in k for x in LINEAR) and ".layers." in k:
                lin += n
            qn += k.endswith("self_attn.q_norm.weight")
            kn += k.endswith("self_attn.k_norm.weight")
    return {"params": {"total": tot, "linear": lin}, "qk_norm_tensors": {"q_norm": qn, "k_norm": kn}}


def expects_bos(hf_id, rev) -> bool:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_id, revision=rev, local_files_only=True)
    ids = tok("Question: test").input_ids
    return tok.bos_token_id is not None and ids[0] == tok.bos_token_id


def snapshot(hf_id, rev, pats) -> Path:
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(hf_id, revision=rev, allow_patterns=pats, local_files_only=True))


def v5_fields(e: dict, local_weights: bool) -> dict:
    if local_weights:
        snap = snapshot(e["hf_id"], e["revision"], e["allow_patterns"])
        for shard, s in e["shards"].items():
            if sha256_file(snap / shard) != s["sha256"]:
                sys.exit(f"{e['hf_id']}: local {shard} differs from the pinned SHA-256")
        heads = [local_header(snap / s) for s in sorted(e["shards"])]
        src = "safetensors headers of the local pinned snapshot (shard SHA-256 verified)"
    else:
        heads = [remote_header(e["hf_id"], e["revision"], s) for s in sorted(e["shards"])]
        src = "safetensors headers range-read from the pinned revision"
    f = header_facts(heads)
    L = e["geometry"]["layers"]
    qk = f["qk_norm_tensors"]
    if qk["q_norm"] not in (0, L) or qk["q_norm"] != qk["k_norm"]:
        sys.exit(f"{e['hf_id']}: unexpected q_norm/k_norm tensor counts {qk}")
    return {"params": f["params"], "params_source": src, "qk_norm_tensors": qk, "expects_bos": expects_bos(e["hf_id"], e["revision"])}


def build():
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    v4m = json.loads(V4_MAN.read_text(encoding="utf-8"))["models"]
    man = {"what": "PROTOCOL_V5 model and dataset manifest: pinned revisions, shard SHA-256 (HF-reported LFS hashes), geometry, QK-norm, "
                   "parameter counts, tokenizer BOS behaviour; the WikiText-2 parquet files. Built by tools/v5_build_manifest.py; G7 and GW check it on the pod.",
           "models": {}, "datasets": {}}
    for mid in REUSED:
        e = json.loads(json.dumps(v4m[mid]))
        e["source"] = "v4/V4_MODEL_MANIFEST.json entry, unchanged (v4 pins)"
        e["geometry"]["qk_norm"] = False
        e.update(v5_fields(e, local_weights=(mid == "qwen2.5-0.5b")))
        man["models"][mid] = e
    for mid, (hf_id, rev, fam) in NEW.items():
        info, files, card_lic = B.file_meta(api, hf_id, rev)
        shards = {f: {"size": m["size"], "sha256": m["lfs_sha256"]} for f, m in files.items() if f.startswith("model") and f.endswith(".safetensors")}
        pats = B.allow_patterns(files)
        cfg = json.loads(Path(hf_hub_download(hf_id, "config.json", revision=rev)).read_bytes())
        geo = B.config_geometry(cfg)
        e = {"hf_id": hf_id, "revision": rev, "family": fam, "licence": card_lic, "card_licence": card_lic, "gated": info.gated or False,
             "shards": shards, "small_files": {f: {"size": files[f]["size"], "git_blob": files[f]["git_blob"], "lfs_sha256": files[f]["lfs_sha256"]}
                                               for f in pats if f in files and f not in shards},
             "allow_patterns": pats, "excluded_files": sorted(f for f in files if f.endswith(".safetensors") and f not in shards),
             "geometry": geo, "geometry_source": "config.json at the pinned revision", "architectures": cfg.get("architectures"),
             "source": "HF API at the pinned revision"}
        e.update(v5_fields(e, local_weights=True))
        e["geometry"]["qk_norm"] = e["qk_norm_tensors"]["q_norm"] == geo["layers"]
        man["models"][mid] = e
    info = api.dataset_info(WIKI["repo"], revision=WIKI["revision"], files_metadata=True)
    if info.sha != WIKI["revision"]:
        sys.exit(f"wikitext: API returned {info.sha}, expected {WIKI['revision']}")
    sib = {s.rfilename: s for s in info.siblings}
    card = info.card_data.to_dict() if info.card_data else {}
    man["datasets"]["wikitext2"] = {"repo": WIKI["repo"], "config": WIKI["config"], "revision": WIKI["revision"], "gated": info.gated or False,
                                    "card_licence": card.get("license"),
                                    "files": {s: {"path": p, "size": sib[p].size, "sha256": sib[p].lfs.sha256} for s, p in WIKI["splits"].items()}}
    return man


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    man = build()
    if a.check:
        same = OUT.exists() and json.loads(OUT.read_text(encoding="utf-8")) == man
        print("manifest matches a fresh build" if same else "MANIFEST DIFFERS FROM A FRESH BUILD")
        sys.exit(0 if same else 1)
    OUT.write_text(json.dumps(man, indent=1) + "\n", encoding="utf-8", newline="\n")
    for mid, e in man["models"].items():
        g = e["geometry"]
        print(f"{mid:18s} {e['revision'][:8]} L{g['layers']} {g['q_heads']}/{g['kv_heads']} hd{g['head_dim']} vocab {g['vocab']} tied {g['tied_embeddings']} "
              f"qk_norm {g['qk_norm']} bos {e['expects_bos']} params {e['params']} gated {e['gated']} lic {e['card_licence']}")
    print(json.dumps(man["datasets"], indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
