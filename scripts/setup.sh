#!/usr/bin/env bash
# Bootstrap a reproducible Python 3.10/3.11 venv for PerSAM + SegGPT experiments.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if command -v python3.11 >/dev/null 2>&1; then
    PYTHON=python3.11
  elif command -v python3.10 >/dev/null 2>&1; then
    PYTHON=python3.10
  else
    echo "ERROR: Need python3.10 or python3.11 on PATH." >&2
    exit 1
  fi
fi

echo "==> Using $PYTHON ($($PYTHON -V))"
if ! "$PYTHON" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] in {(3,10),(3,11)} else 1)'; then
  echo "ERROR: Unsupported Python. Use 3.10 or 3.11." >&2
  exit 1
fi

if [[ ! -d .venv ]]; then
  echo "==> Creating .venv"
  "$PYTHON" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -U pip wheel

echo "==> Installing PyTorch (CUDA 12.8 wheels; needs NVIDIA driver)"
pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128

echo "==> Installing remaining requirements"
# Drop torch pins that require the cu128 index; already installed above.
grep -vE '^(torch|torchvision)==' requirements.txt > /tmp/soilnail-req-notorch.txt
pip install -r /tmp/soilnail-req-notorch.txt

mkdir -p weights outputs
SAM_CKPT="weights/sam_vit_h_4b8939.pth"
SAM_URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
if [[ ! -f "$SAM_CKPT" ]]; then
  echo "==> Downloading SAM ViT-H (~2.4 GB) -> $SAM_CKPT"
  if command -v wget >/dev/null 2>&1; then
    wget -c -O "$SAM_CKPT" "$SAM_URL"
  else
    curl -L -C - -o "$SAM_CKPT" "$SAM_URL"
  fi
else
  echo "==> SAM ViT-H already present"
fi

MOBILE_SRC="third_party/Personalize-SAM/weights/mobile_sam.pt"
if [[ -f "$MOBILE_SRC" && ! -f weights/mobile_sam.pt ]]; then
  cp "$MOBILE_SRC" weights/mobile_sam.pt
  echo "==> Copied MobileSAM -> weights/mobile_sam.pt"
fi

echo "==> Verifying imports / CUDA"
python - <<'PY'
import torch
from segment_anything import sam_model_registry
from transformers import SegGptForImageSegmentation, SegGptImageProcessor
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    free, total = torch.cuda.mem_get_info(0)
    print(f"vram_free_gib {free/1024**3:.2f} / {total/1024**3:.2f}")
print("SAM registry keys", list(sam_model_registry.keys()))
print("SegGPT OK", SegGptForImageSegmentation.__name__, SegGptImageProcessor.__name__)
PY

echo ""
echo "Setup complete. Activate with:  source .venv/bin/activate"
echo "Smoke tests:                 python scripts/smoke_all.py"
