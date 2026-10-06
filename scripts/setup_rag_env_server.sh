#!/usr/bin/env bash
# Crée l'environnement séparé des embeddings et de Qdrant local.
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

env_name="${RECEPTION_RAG_RAG_ENV:-reception-rag-rag}"
if [ -n "${RECEPTION_RAG_HF_HOME:-}" ]; then
  hf_home="$RECEPTION_RAG_HF_HOME"
elif [ -d "/data/$USER" ] && [ -w "/data/$USER" ]; then
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
mkdir -p "$hf_home" data/qdrant_storage logs
export HF_HOME="$hf_home"
export PIP_DISABLE_PIP_VERSION_CHECK=1

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-rag.txt
python -m pip check

python - <<'PY'
import importlib.metadata
import torch
from qdrant_client import QdrantClient
from FlagEmbedding import BGEM3FlagModel

print(f"PyTorch        : {torch.__version__}")
print(f"CUDA runtime   : {torch.version.cuda}")
print(f"CUDA disponible: {torch.cuda.is_available()}")
print(f"Qdrant client  : {importlib.metadata.version('qdrant-client')}")
print(f"FlagEmbedding  : {importlib.metadata.version('FlagEmbedding')}")
assert QdrantClient is not None
assert BGEM3FlagModel is not None
PY

python -m pytest -q \
  tests/test_schema.py \
  tests/test_normalization.py \
  tests/test_chunking.py \
  tests/test_vector_store.py \
  tests/test_retrieval_eval.py
python -m pip freeze | sort > logs/requirements-rag-freeze.txt

echo
echo "Environnement valide : $env_name"
echo "Cache Hugging Face    : $hf_home"
echo "Stockage Qdrant       : $project_dir/data/qdrant_storage"
echo "Versions figées       : $project_dir/logs/requirements-rag-freeze.txt"
