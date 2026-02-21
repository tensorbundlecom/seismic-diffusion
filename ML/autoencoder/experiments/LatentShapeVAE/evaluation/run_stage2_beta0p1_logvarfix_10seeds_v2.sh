#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$ROOT_DIR"

EXP_DIR="ML/autoencoder/experiments/LatentShapeVAE"
EVAL_DIR="$EXP_DIR/evaluation"
RES_DIR="$EXP_DIR/results"
PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/bin/python}"

SEEDS=(42 43 44 45 46 47 48 49 50 51)
RUN_NAMES=()
CHECKPOINTS=()
for seed in "${SEEDS[@]}"; do
  run_name="lsv_stage2_vae_base_ld64_b0p1_s${seed}_logvfixv2"
  ckpt="$EXP_DIR/checkpoints/${run_name}_best.pt"
  RUN_NAMES+=("$run_name")
  CHECKPOINTS+=("$ckpt")
done

for ck in "${CHECKPOINTS[@]}"; do
  if [[ ! -f "$ck" ]]; then
    echo "[ERROR] Missing checkpoint: $ck"
    exit 1
  fi
done

echo "[$(date '+%F %T')] START stage2 beta0.1 logvarfix 10-seed eval v2"
echo "[$(date '+%F %T')] PYTHON_BIN=$PYTHON_BIN"

"$PYTHON_BIN" "$EVAL_DIR/analyze_latent_shape.py" \
  --checkpoints "${CHECKPOINTS[@]}" \
  --split test \
  --output_dir "$RES_DIR/latent_shape_test_stage2_beta0p1_logvarfix_10seeds_v2"

"$PYTHON_BIN" "$EVAL_DIR/analyze_latent_shape.py" \
  --checkpoints "${CHECKPOINTS[@]}" \
  --split ood_event \
  --output_dir "$RES_DIR/latent_shape_ood_event_stage2_beta0p1_logvarfix_10seeds_v2"

"$PYTHON_BIN" "$EVAL_DIR/evaluate_prior_sampling.py" \
  --checkpoints "${CHECKPOINTS[@]}" \
  --split test \
  --num_real_samples 2000 \
  --num_generated_samples 2000 \
  --output_dir "$RES_DIR/prior_sampling_test_stage2_beta0p1_logvarfix_10seeds_v2"

"$PYTHON_BIN" "$EVAL_DIR/evaluate_prior_sampling.py" \
  --checkpoints "${CHECKPOINTS[@]}" \
  --split ood_event \
  --num_real_samples 2000 \
  --num_generated_samples 2000 \
  --output_dir "$RES_DIR/prior_sampling_ood_event_stage2_beta0p1_logvarfix_10seeds_v2"

"$PYTHON_BIN" "$EVAL_DIR/audit_latent_var_outliers.py" \
  --checkpoints "${CHECKPOINTS[@]}" \
  --splits test ood_event \
  --var_thresholds 10,1000,100000 \
  --top_k 100 \
  --output_dir "$RES_DIR/latent_var_outlier_audit_stage2_beta0p1_logvarfix_10seeds_v2"

"$PYTHON_BIN" "$EVAL_DIR/summarize_stage2_beta0p1_logvarfix_10seeds_v2.py" \
  --latent_test_csv "$RES_DIR/latent_shape_test_stage2_beta0p1_logvarfix_10seeds_v2/latent_shape_summary.csv" \
  --latent_ood_csv "$RES_DIR/latent_shape_ood_event_stage2_beta0p1_logvarfix_10seeds_v2/latent_shape_summary.csv" \
  --prior_test_csv "$RES_DIR/prior_sampling_test_stage2_beta0p1_logvarfix_10seeds_v2/prior_sampling_realism_summary.csv" \
  --prior_ood_csv "$RES_DIR/prior_sampling_ood_event_stage2_beta0p1_logvarfix_10seeds_v2/prior_sampling_realism_summary.csv" \
  --audit_csv "$RES_DIR/latent_var_outlier_audit_stage2_beta0p1_logvarfix_10seeds_v2/audit_summary.csv" \
  --output_dir "$RES_DIR/stage2_beta0p1_logvarfix_10seeds_v2"

echo "[$(date '+%F %T')] DONE stage2 beta0.1 logvarfix 10-seed eval v2"
