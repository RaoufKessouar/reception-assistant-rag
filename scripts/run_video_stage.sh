#!/usr/bin/env bash
# Usage : bash scripts/run_video_stage.sh 1 [--all | --video "Categorie/fichier.mp4"]
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <stage 1..4> [arguments video]" >&2
  exit 2
fi

stage="$1"
shift
if [[ ! "$stage" =~ ^[1-4]$ ]]; then
  echo "Etape invalide: $stage" >&2
  exit 2
fi

if [ "$#" -eq 0 ]; then
  set -- --all
fi

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
mkdir -p logs

expected_env="${RECEPTION_RAG_VIDEO_ENV:-reception-rag-video}"
active_env="${CONDA_DEFAULT_ENV:-}"
active_prefix_name=""
if [ -n "${CONDA_PREFIX:-}" ]; then
  active_prefix_name="$(basename "$CONDA_PREFIX")"
fi
if [ "$active_env" != "$expected_env" ] && [ "$active_prefix_name" != "$expected_env" ]; then
  echo "Activez d'abord l'environnement Conda: conda activate $expected_env" >&2
  exit 5
fi

if [ -n "${RECEPTION_RAG_HF_HOME:-}" ]; then
  export HF_HOME="$RECEPTION_RAG_HF_HOME"
elif [ -d "/data/$USER" ] && [ -w "/data/$USER" ]; then
  # Les poids VLM depassent souvent le quota du dossier /users.
  export HF_HOME="/data/$USER/ia/huggingface"
else
  export HF_HOME="$HOME/checkpoints/huggingface"
fi
mkdir -p "$HF_HOME"

if [ "$stage" = "1" ] || [ "$stage" = "3" ]; then
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 | tr -d ' ')"
  required_mib=10000
  if [ "$stage" = "3" ]; then
    required_mib=20000
  fi
  if [ "$free_mib" -lt "$required_mib" ]; then
    echo "GPU 1 insuffisamment libre: ${free_mib} MiB, minimum ${required_mib} MiB" >&2
    exit 3
  fi
fi

if [ "$stage" = "4" ]; then
  llm_backend="$(python -c 'from src.config import settings; print(settings()["hw"].get("llm_backend", "vllm"))')"
  if [ "$llm_backend" = "vllm" ]; then
    curl --fail --silent http://127.0.0.1:8000/v1/models >/dev/null || {
      echo "Le serveur LLM vLLM n'est pas joignable sur 127.0.0.1:8000" >&2
      exit 4
    }
  elif [ "$llm_backend" = "transformers" ]; then
    free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 | tr -d ' ')"
    if [ "$free_mib" -lt 32000 ]; then
      echo "GPU 1 insuffisamment libre pour le LLM: ${free_mib} MiB" >&2
      exit 3
    fi
  else
    echo "Backend LLM inconnu: $llm_backend" >&2
    exit 6
  fi
fi

timestamp="$(date +%Y%m%d-%H%M%S)"
log="logs/video-stage-${stage}-${timestamp}.log"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m src.cli video "$@" --stage "$stage" --stop-after "$stage" 2>&1 | tee "$log"
echo "Log: $log"
