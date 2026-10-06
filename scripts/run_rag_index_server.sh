#!/usr/bin/env bash
# Attend le modèle et une GPU libre, puis reconstruit l’index.
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

gpu_index="${RECEPTION_RAG_EMBEDDING_GPU:-1}"
max_used_mb="${RECEPTION_RAG_GPU_MAX_USED_MB:-8000}"
max_util="${RECEPTION_RAG_GPU_MAX_UTIL:-20}"
required_checks="${RECEPTION_RAG_GPU_FREE_CHECKS:-3}"
poll_seconds="${RECEPTION_RAG_GPU_POLL_SECONDS:-30}"
model_exit="logs/bge-m3-download.exit"
status_file="logs/rag-index.status"
exit_file="logs/rag-index.exit"

mkdir -p logs
rm -f "$exit_file"

write_status() {
  printf '%s | %s\n' "$(date --iso-8601=seconds)" "$1" > "$status_file"
}

finish() {
  code=$?
  printf '%s\n' "$code" > "$exit_file"
  if [ "$code" -eq 0 ]; then
    write_status "COMPLETED"
  else
    write_status "FAILED exit=$code"
  fi
}
trap finish EXIT

while [ ! -f "$model_exit" ]; do
  write_status "WAITING_MODEL"
  sleep "$poll_seconds"
done
if [ "$(tr -d '[:space:]' < "$model_exit")" != "0" ]; then
  echo "Le téléchargement BGE-M3 a échoué" >&2
  exit 2
fi

stable=0
while [ "$stable" -lt "$required_checks" ]; do
  IFS=, read -r used_mb util < <(
    nvidia-smi \
      --id="$gpu_index" \
      --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits
  )
  used_mb="$(echo "$used_mb" | xargs)"
  util="$(echo "$util" | xargs)"
  if [ "$used_mb" -le "$max_used_mb" ] && [ "$util" -le "$max_util" ]; then
    stable=$((stable + 1))
  else
    stable=0
  fi
  write_status "WAITING_GPU gpu=$gpu_index used=${used_mb}MiB util=${util}% stable=${stable}/${required_checks}"
  [ "$stable" -ge "$required_checks" ] || sleep "$poll_seconds"
done

write_status "INDEXING gpu=$gpu_index"
export HF_HOME="/data/$USER/ia/huggingface"
export HF_HUB_DISABLE_XET=1
conda run -n reception-rag-rag python -m src.cli index --reset
