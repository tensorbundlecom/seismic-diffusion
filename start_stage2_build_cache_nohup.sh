#!/usr/bin/env bash

set -euo pipefail

ROOT="/home/gms/kalem_seismic_paper_repro"
LOG_DIR="$ROOT/ML/autoencoder/experiments/PaperRepro/results/stage2_cache_logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="$LOG_DIR/stage2_cache_${TIMESTAMP}.log"
PID_PATH="$LOG_DIR/stage2_cache_${TIMESTAMP}.pid"

mkdir -p "$LOG_DIR"

cd "$ROOT"

nohup bash "$ROOT/run_stage2_build_cache.sh" >"$LOG_PATH" 2>&1 &
PID=$!

echo "$PID" >"$PID_PATH"

echo "Started Stage-2 latent cache build."
echo "PID: $PID"
echo "Log: $LOG_PATH"
echo "PID file: $PID_PATH"
