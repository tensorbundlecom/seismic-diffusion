#!/usr/bin/env bash
set -euo pipefail

ROOT="ML/autoencoder/experiments/WSpaceVAE"
PY="/home/gms/.pyenv/shims/python"

PID_FILE="$ROOT/logs/train/train_true_wspace_vae_external.pid"
TRAIN_LOG_GLOB="$ROOT/logs/train/train_true_wspace_vae_external_*.log"
EVAL_LOG="$ROOT/logs/eval/eval_true_wspace_post_training_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$ROOT/logs/eval" "$ROOT/results"

if [[ ! -f "$PID_FILE" ]]; then
  echo "[ERROR] PID file not found: $PID_FILE"
  exit 1
fi

PID="$(cat "$PID_FILE")"
echo "[INFO] Waiting training PID=$PID to finish..."
while kill -0 "$PID" 2>/dev/null; do
  sleep 30
done
echo "[INFO] Training completed. Starting evaluation..."

"$PY" -u "$ROOT/evaluation/evaluate_true_wspace_post_training_custom_ood.py" \
  --config "$ROOT/configs/train_true_wspace_vae_external.json" \
  --checkpoint "$ROOT/checkpoints/true_wspace_vae_best.pt" \
  --mode reconstruct \
  --output_json "$ROOT/results/post_training_custom_ood_metrics_true_wspace_vae.json" \
  >"$EVAL_LOG" 2>&1

echo "[INFO] Evaluation log: $EVAL_LOG"

"$PY" -u "$ROOT/evaluation/compare_true_wspace_vs_wbaseline.py" \
  --true_wspace_json "$ROOT/results/post_training_custom_ood_metrics_true_wspace_vae.json" \
  --legacy_wbaseline_json "ML/autoencoder/experiments/LegacyCondBaseline/results/post_training_custom_ood_metrics.json" \
  --output_md "$ROOT/results/compare_true_wspace_vs_wbaseline.md" \
  >>"$EVAL_LOG" 2>&1

echo "[INFO] Comparison written to $ROOT/results/compare_true_wspace_vs_wbaseline.md"
