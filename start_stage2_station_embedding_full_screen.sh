#!/usr/bin/env bash

set -euo pipefail

ROOT="/home/gms/kalem_seismic_paper_repro"
LOG_DIR="$ROOT/ML/autoencoder/experiments/PaperRepro/runs/nohup_logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SESSION_NAME="paperrepro_stage2_semb_${TIMESTAMP}"
LOG_PATH="$LOG_DIR/stage2_semb_full_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

cd "$ROOT"

screen -dmS "$SESSION_NAME" bash -lc "bash '$ROOT/run_stage2_station_embedding_full.sh' 2>&1 | tee '$LOG_PATH'"

echo "Started Stage-2 station-embedding training in screen."
echo "Session: $SESSION_NAME"
echo "Log: $LOG_PATH"
echo "Attach: screen -r $SESSION_NAME"
