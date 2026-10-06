#!/usr/bin/env bash
# Lance le corpus video complet, une etape a la fois, de facon reprenable.

set -Eeuo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
mkdir -p logs

lock_file="logs/video-batch.lock"
status_file="logs/video-batch.status"
exec 9>"$lock_file"
if ! flock -n 9; then
  echo "Un traitement video global est deja actif." >&2
  exit 9
fi

run_id="$(date +%Y%m%d-%H%M%S)"
main_log="logs/video-batch-${run_id}.log"
current_stage="initialisation"

write_status() {
  local state="$1"
  local stage="$2"
  local exit_code="${3:-0}"
  local temporary="${status_file}.tmp.$$"
  {
    printf 'state=%s\n' "$state"
    printf 'stage=%s\n' "$stage"
    printf 'exit_code=%s\n' "$exit_code"
    printf 'run_id=%s\n' "$run_id"
    printf 'pid=%s\n' "$$"
    printf 'updated_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'log=%s\n' "$main_log"
  } > "$temporary"
  mv "$temporary" "$status_file"
}

on_error() {
  local exit_code="$?"
  trap - ERR
  write_status "failed" "$current_stage" "$exit_code"
  echo "ECHEC pendant l'etape $current_stage (code $exit_code)."
  echo "Relancer ce script apres correction reprendra les resultats deja calcules."
  exit "$exit_code"
}
trap on_error ERR

exec > >(tee -a "$main_log") 2>&1

conda_init="$HOME/miniconda3/etc/profile.d/conda.sh"
if [ ! -f "$conda_init" ]; then
  echo "Initialisation Conda introuvable: $conda_init" >&2
  exit 10
fi
# shellcheck source=/dev/null
source "$conda_init"
conda activate "${RECEPTION_RAG_VIDEO_ENV:-reception-rag-video}"

export RECEPTION_RAG_HF_HOME="${RECEPTION_RAG_HF_HOME:-/data/$USER/ia/huggingface}"
export PYTHONUNBUFFERED=1

data_root="$(readlink -f data)"
if [ "$(hostname -s)" = "server-b" ] && [[ "$data_root" != /data/* ]]; then
  echo "Sur server-b, data doit pointer vers /data pour eviter le quota /users." >&2
  echo "Emplacement actuel: $data_root" >&2
  exit 11
fi

write_status "running" "$current_stage"
echo "Traitement global $run_id"
echo "Projet : $project_dir"
echo "Donnees : $data_root"
echo "Cache HF : $RECEPTION_RAG_HF_HOME"
echo "Mode : reprise sans --force; aucun export automatique"

for stage in 1 2 3 4; do
  current_stage="$stage"
  write_status "running" "$current_stage"
  echo "Debut etape $stage"
  bash scripts/run_video_stage.sh "$stage" --all
  echo "Fin etape $stage"
done

current_stage="review_required"
write_status "completed" "$current_stage"
echo "Pipeline termine. Les nouvelles etapes restent en review_required."
echo "Aucun video-export n'a ete lance automatiquement."
