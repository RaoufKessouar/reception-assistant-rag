#!/usr/bin/env bash
# Diagnostic en lecture seule. A executer avant toute installation.
set -u

section() {
  echo
  echo "===== $1 ====="
}

section "SYSTEME"
hostname
date --iso-8601=seconds 2>/dev/null || date
uname -a
if [ -r /etc/os-release ]; then
  cat /etc/os-release
fi

section "RESSOURCES"
free -h 2>/dev/null || true
df -h "$HOME" 2>/dev/null || true

section "GPU"
nvidia-smi
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true

gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sort -nu | paste -sd, -)"
if [ -n "$gpu_pids" ]; then
  echo
  echo "Proprietaires des processus GPU :"
  ps -o user=,pid=,etime=,cmd= -p "$gpu_pids" 2>/dev/null || true
fi

section "CONDA ET PYTHON"
command -v conda || true
conda --version 2>/dev/null || true
conda env list 2>/dev/null || true
command -v python || true
python --version 2>/dev/null || true

section "OUTILS"
for tool in ffmpeg ffprobe git rsync tmux curl docker nvcc gcc; do
  printf '%-10s ' "$tool"
  command -v "$tool" 2>/dev/null || echo "ABSENT"
done
ffmpeg -version 2>/dev/null | head -n 2 || true
nvcc --version 2>/dev/null | tail -n 4 || true

section "PORTS LOCAUX"
ss -ltnp 2>/dev/null | grep -E ':(6333|8000|8001)\b' || echo "Aucun service sur 6333/8000/8001"

section "DOSSIERS CANDIDATS"
for directory in "$HOME/reception-assistant-rag" "$HOME/venvs" "$HOME/checkpoints"; do
  if [ -e "$directory" ]; then
    du -sh "$directory" 2>/dev/null || true
    find "$directory" -maxdepth 2 -type f \( -name 'pyproject.toml' -o -name 'requirements*.txt' -o -name 'environment*.yml' \) -print 2>/dev/null | head -n 30
  fi
done
