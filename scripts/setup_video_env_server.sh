#!/usr/bin/env bash
# Cree et valide l'environnement GPU des etapes video 1 a 4.
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

env_name="${RECEPTION_RAG_VIDEO_ENV:-reception-rag-video}"
if [ -n "${RECEPTION_RAG_HF_HOME:-}" ]; then
  hf_home="$RECEPTION_RAG_HF_HOME"
elif [ -d "/data/$USER" ] && [ -w "/data/$USER" ]; then
  # Evite le quota relativement faible du dossier /users pour les poids VLM.
  hf_home="/data/$USER/ia/huggingface"
else
  hf_home="$HOME/checkpoints/huggingface"
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda est introuvable dans PATH" >&2
  exit 2
fi

conda_base="$(conda info --base)"
# shellcheck disable=SC1091
source "$conda_base/etc/profile.d/conda.sh"

if ! conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fxq "$env_name"; then
  conda create -y -n "$env_name" \
    --override-channels \
    --channel conda-forge \
    python=3.11 pip
fi

conda activate "$env_name"
mkdir -p "$hf_home" logs
export HF_HOME="$hf_home"
export PIP_DISABLE_PIP_VERSION_CHECK=1

python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  torch==2.8.0 \
  torchvision==0.23.0 \
  torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-video.txt
python -m pip check

python - <<'PY'
import importlib.metadata

import torch
import whisperx
from transformers import Qwen2_5_VLForConditionalGeneration

print(f"Python/PyTorch : {torch.__version__}")
print(f"CUDA runtime   : {torch.version.cuda}")
print(f"WhisperX       : {importlib.metadata.version('whisperx')}")
print(f"Transformers   : {importlib.metadata.version('transformers')}")
print(f"CUDA disponible: {torch.cuda.is_available()}")
print(f"GPU detectees  : {torch.cuda.device_count()}")

if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
    raise SystemExit("Les deux GPU CUDA attendues ne sont pas disponibles")

for index in range(torch.cuda.device_count()):
    print(
        f"GPU {index}: {torch.cuda.get_device_name(index)} "
        f"capability={torch.cuda.get_device_capability(index)}"
    )

probe = torch.ones(1, device="cuda:1")
print(f"Test GPU 1     : {probe.item()}")
del probe
torch.cuda.empty_cache()

# Garantit que la classe utilisee par l'etape VLM est presente.
assert Qwen2_5_VLForConditionalGeneration is not None
assert whisperx is not None
PY

python -m pytest -q tests/test_schema.py tests/test_video_pipeline.py
python -m pip freeze | sort > logs/requirements-video-freeze.txt

echo
echo "Environnement valide: $env_name"
echo "Cache Hugging Face   : $hf_home"
echo "Versions figees      : $project_dir/logs/requirements-video-freeze.txt"
