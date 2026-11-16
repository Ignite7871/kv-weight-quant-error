"""Reconstruct candidate official Llama configs from the unsloth mirrors and test them against the pinned git-blob SHA-1s."""
import hashlib, itertools, json, sys
D = sys.argv[1]
TARGET = {"l3b": ("47d4a5aa69cdef91a53b77f5c5583647a578ca0e", 844), "l8b": ("cccf055d6f8f210387a248c91dc40e0c7a4bafab", 826)}
ORDERS = [["factor", "high_freq_factor", "low_freq_factor", "original_max_position_embeddings", "rope_type"],
          ["factor", "low_freq_factor", "high_freq_factor", "original_max_position_embeddings", "rope_type"]]
VERS = [f"4.{m}.{p}{s}" for m in range(38, 50) for p in range(0, 4) for s in ("", ".dev0")]


def blob(b):
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


for key in ("l3b", "l8b"):
    mirror = json.load(open(f"{D}/{key}.json"))
    want, size = TARGET[key]
    found = None
    for drop_name, keep_head, keep_pad, order, ver, nl in itertools.product(
            (True, False), (True, False), (False, True), ORDERS, VERS, ("", "\n")):
        c = dict(mirror)
        c.pop("unsloth_fixed", None)
        if drop_name:
            c.pop("_name_or_path", None)
        if not keep_head:
            c.pop("head_dim", None)
        if not keep_pad:
            c.pop("pad_token_id", None)
        c["transformers_version"] = ver
        c["rope_scaling"] = {k: mirror["rope_scaling"][k] for k in order}
        b = (json.dumps(c, indent=2) + nl).encode()
        if len(b) == size and blob(b) == want:
            found = c
            open(f"{D}/{key}_official_reconstructed.json", "wb").write(b)
            break
    if found:
        diff = {k: (mirror.get(k), found.get(k)) for k in set(mirror) | set(found) if mirror.get(k) != found.get(k)}
        rope_same = all(mirror[k] == found[k] for k in ("rope_theta", "rope_scaling")) and mirror.get("head_dim", found.get("head_dim")) == found.get("head_dim", mirror.get("head_dim"))
        print(key, "MATCH pinned blob", want[:12], "| RoPE fields identical:", rope_same, "| fields that differ:", diff)
    else:
        print(key, "no candidate matched the pinned blob")
