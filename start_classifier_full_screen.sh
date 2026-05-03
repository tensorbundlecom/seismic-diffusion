#!/usr/bin/env bash

set -euo pipefail

ROOT="/home/gms/kalem_seismic_paper_repro"
LOG_DIR="$ROOT/ML/autoencoder/experiments/PaperRepro/runs/nohup_logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SESSION_NAME="paperrepro_classifier_${TIMESTAMP}"
LOG_PATH="$LOG_DIR/classifier_full_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

cd "$ROOT"

screen -dmS "$SESSION_NAME" bash -lc "/home/gms/miniconda3/bin/python -u ML/autoencoder/experiments/PaperRepro/training/train_paper_metrics_classifier.py --config ML/autoencoder/experiments/PaperRepro/configs/frozen_paper_repro_v1.yaml --device cuda 2>&1 | tee '$LOG_PATH'"

echo "Started PaperRepro classifier training in screen."
echo "Session: $SESSION_NAME"
echo "Log: $LOG_PATH"
echo "Attach: screen -r $SESSION_NAME"
