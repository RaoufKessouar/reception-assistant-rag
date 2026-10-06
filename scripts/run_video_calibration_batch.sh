#!/usr/bin/env bash
set -uo pipefail

PROJECT_DIR="${RECEPTION_RAG_PROJECT_DIR:-$HOME/reception-assistant-rag}"
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/video-calibration-batch.log"
STATUS_FILE="$LOG_DIR/video-calibration-batch.status"

mkdir -p "$LOG_DIR"
printf 'RUNNING\nstarted_at=%s\narguments=%s\n' \
    "$(date --iso-8601=seconds)" "$*" > "$STATUS_FILE"

cd "$PROJECT_DIR" || exit 1
if conda run -n reception-rag-video python -m src.cli video-calibration-propose "$@" \
    > "$LOG_FILE" 2>&1; then
    printf 'SUCCEEDED\nfinished_at=%s\narguments=%s\n' \
        "$(date --iso-8601=seconds)" "$*" > "$STATUS_FILE"
else
    code=$?
    printf 'FAILED\nexit_code=%s\nfinished_at=%s\narguments=%s\n' \
        "$code" "$(date --iso-8601=seconds)" "$*" > "$STATUS_FILE"
    exit "$code"
fi
