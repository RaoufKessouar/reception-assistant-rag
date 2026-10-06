#!/usr/bin/env bash
# Compatibilite : l'etape 4 server-b n'a plus besoin d'un serveur vLLM.
set -euo pipefail

echo "server-b utilise le backend Transformers integre."
echo "Lancez directement: bash scripts/run_video_stage.sh 4 --video ID_VIDEO"
