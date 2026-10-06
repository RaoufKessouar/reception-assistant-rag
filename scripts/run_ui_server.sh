#!/usr/bin/env bash
# Interface privée : accès depuis le PC via un tunnel SSH, jamais exposée au réseau.
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

port="${RECEPTION_RAG_UI_PORT:-8501}"
exec conda run --no-capture-output -n reception-rag-rag \
  streamlit run interface_custom/app.py \
  --server.address 127.0.0.1 \
  --server.port "$port" \
  --server.headless true \
  --browser.gatherUsageStats false
