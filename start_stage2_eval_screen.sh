#!/usr/bin/env bash

set -euo pipefail

ROOT="/home/gms/kalem_seismic_paper_repro"
LOG_DIR="$ROOT/ML/autoencoder/experiments/PaperRepro/results/stage2_eval_logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SESSION_NAME="paperrepro_stage2_eval_${TIMESTAMP}"
LOG_PATH="$LOG_DIR/stage2_eval_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

cd "$ROOT"

CLASSIFIER_CKPT="${CLASSIFIER_CKPT:-$(find ML/autoencoder/experiments/PaperRepro/runs -path '*classifier*/checkpoints/best_val_accuracy.pt' | sort | tail -n 1)}"

if [[ -z "${CLASSIFIER_CKPT}" ]]; then
  echo "No classifier checkpoint found. Train the paper metrics classifier first." >&2
  exit 1
fi

screen -dmS "$SESSION_NAME" bash -lc "/home/gms/miniconda3/bin/python -u ML/autoencoder/experiments/PaperRepro/evaluation/evaluate_stage2_generation.py --config ML/autoencoder/experiments/PaperRepro/configs/frozen_paper_repro_v1.yaml --stage1-checkpoint ML/autoencoder/experiments/PaperRepro/runs/run_20260424_1940_s1_ae_hh100_ori4064_logspec128_lat8x32x32_evt801010_s42_v1/checkpoints/best_recon.ckpt --stage2-checkpoint ML/autoencoder/experiments/PaperRepro/runs/run_20260426_1728_s2_ledm_hh100_ori4064_logspec128_lat8x32x32_evt801010_s42_v1/checkpoints/best_val_loss.ckpt --classifier-checkpoint '${CLASSIFIER_CKPT}' --cache-root ML/autoencoder/experiments/PaperRepro/results/stage2_cache/run_20260424_1940_s1_ae_hh100_ori4064_logspec128_lat8x32x32_evt801010_s42_v1 --device cuda --batch-size 32 --num-workers 4 --waveform-eval-samples 32 --visual-samples-per-split 6 2>&1 | tee '$LOG_PATH'"

echo "Started Stage-2 evaluation in screen."
echo "Session: $SESSION_NAME"
echo "Log: $LOG_PATH"
echo "Attach: screen -r $SESSION_NAME"
