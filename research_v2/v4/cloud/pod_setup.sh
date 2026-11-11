#!/usr/bin/env bash
# PROTOCOL_V4 section 11: one-time environment setup on the network volume, run INSIDE the pod from /workspace/repo after
# `tools/v4_remote.py push` has placed the locked commit there, and only while an authorization file exists (it is run through
# `tools/v4_remote.py exec --tag GM -- bash research_v2/v4/cloud/pod_setup.sh`).
# No secret is read or written here. HF_TOKEN stays an environment variable set by the corresponding author as a RunPod secret.
set -euo pipefail

W=/workspace
export HF_HOME=$W/hf
mkdir -p "$HF_HOME" "$W/v4_runs"

for f in "$HF_HOME/token" "$HF_HOME/stored_tokens" "$HOME/.cache/huggingface/token"; do
  if [ -e "$f" ]; then echo "REFUSING: a Hugging Face token file exists ($f); the token must only be the HF_TOKEN variable"; exit 2; fi
done

nvidia-smi --query-gpu=name,memory.total,driver_version,ecc.mode.current --format=csv,noheader
case "$(nvidia-smi --query-gpu=name --format=csv,noheader)" in
  "NVIDIA A40"|"A40") ;;
  *) echo "REFUSING: D5 permits only an A40"; exit 2 ;;
esac

PY=${PY:-python3}
$PY -c 'import sys; assert sys.version_info >= (3, 10), sys.version; print("python", sys.version.split()[0])'
# First boot: record the digest of the locked image tag (write-once; v4_ops uses only the standard library).
V4_RUNS=$W/v4_runs $PY - <<'EOF'
import json, os, sys
from pathlib import Path
sys.path.insert(0, "research_v2/v4")
import v4_ops as O
print(json.dumps(O.record_image_digest(Path(os.environ["V4_RUNS"]))))
EOF
if [ ! -x "$W/venv/bin/python" ]; then
  $PY -m venv "$W/venv"
fi
. "$W/venv/bin/activate"
python -m pip install --quiet --upgrade "pip==26.1.2"
python -m pip install --quiet --index-url https://download.pytorch.org/whl/cu126 "torch==2.13.0"
python -m pip install --quiet -r research_v2/v4/V4_PACKAGES.txt
python - <<'EOF'
import torch, transformers, bitsandbytes
assert torch.__version__.startswith("2.13.0"), torch.__version__
assert torch.cuda.is_available(), "CUDA not available"
print("torch", torch.__version__, "cuda", torch.version.cuda, "|", torch.cuda.get_device_name(0))
print("transformers", transformers.__version__, "bitsandbytes", bitsandbytes.__version__)
EOF
python research_v2/tools/lock_v4.py --verify
python research_v2/tools/lock_v4.py --lock-id
echo "pod setup complete"
