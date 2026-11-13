#!/usr/bin/env bash
# PROTOCOL_V5 section 10: one-time environment setup on the network volume, run INSIDE the pod from /workspace/repo after `tools/v5_remote.py push`
# has placed the lock commit there, and only while AUTHORIZED_V5_GM.txt exists (it is run through
# `tools/v5_remote.py --tag GM exec -- bash research_v2/v5/cloud/pod_setup_v5.sh`).
# v5 has no secret: every model and the dataset are ungated. The setup refuses if a Hugging Face token variable or token file exists.
set -euo pipefail

W=/workspace
export HF_HOME=$W/hf HF_HUB_DISABLE_IMPLICIT_TOKEN=1
mkdir -p "$HF_HOME" "$W/v5_runs"

for f in "$HF_HOME/token" "$HF_HOME/stored_tokens" "$HOME/.cache/huggingface/token"; do
  if [ -e "$f" ]; then echo "REFUSING: a Hugging Face token file exists ($f); v5 runs with no token"; exit 2; fi
done
if tr '\0' '\n' < /proc/1/environ 2>/dev/null | grep -q -E '^(HF_TOKEN|HUGGING_FACE_HUB_TOKEN)='; then
  echo "REFUSING: the pod has a HF_TOKEN / HUGGING_FACE_HUB_TOKEN variable; v5 runs with no token (remove it from the pod)"; exit 2
fi

nvidia-smi --query-gpu=name,memory.total,driver_version,ecc.mode.current --format=csv,noheader
case "$(nvidia-smi --query-gpu=name --format=csv,noheader)" in
  "NVIDIA A40"|"A40") ;;
  *) echo "REFUSING: section 9 permits only an A40"; exit 2 ;;
esac

PY=${PY:-python3}
$PY -c 'import sys; assert sys.version_info >= (3, 10), sys.version; print("python", sys.version.split()[0])'
# First boot: record the digest of the image tag (write-once; v5_ops uses only the standard library). Image: runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
V5_RUNS=$W/v5_runs $PY - <<'EOF'
import json, os, sys
from pathlib import Path
sys.path.insert(0, "research_v2/v5")
import v5_ops as O
print(json.dumps(O.record_image_digest(Path(os.environ["V5_RUNS"]))))
EOF
if [ ! -x "$W/venv/bin/python" ]; then
  $PY -m venv "$W/venv"
fi
. "$W/venv/bin/activate"
python -m pip install --quiet --upgrade "pip==26.1.2"
python -m pip install --quiet --index-url https://download.pytorch.org/whl/cu126 "torch==2.13.0"
python -m pip install --quiet -r research_v2/v5/V5_PACKAGES.txt
# HQQ (section 3): the pinned version, pure-PyTorch backend, without its dependencies so the pinned torch stack is untouched
DISABLE_CUDA=1 python -m pip install --quiet --no-deps "hqq==0.2.8.post1"
python - <<'EOF'
import torch, transformers, bitsandbytes
from importlib.metadata import version
assert torch.__version__.startswith("2.13.0"), torch.__version__
assert torch.cuda.is_available(), "CUDA not available"
assert version("hqq") == "0.2.8.post1", version("hqq")
import hqq.core.quantize  # noqa: F401
print("torch", torch.__version__, "cuda", torch.version.cuda, "|", torch.cuda.get_device_name(0))
print("transformers", transformers.__version__, "bitsandbytes", bitsandbytes.__version__, "hqq", version("hqq"))
EOF
python research_v2/tools/lock_v5.py --verify
python research_v2/tools/lock_v5.py --lock-id
echo "pod setup complete"
