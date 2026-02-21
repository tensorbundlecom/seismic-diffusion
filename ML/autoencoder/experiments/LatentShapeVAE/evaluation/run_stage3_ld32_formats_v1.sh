#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$ROOT_DIR"

EXP_DIR="ML/autoencoder/experiments/LatentShapeVAE"
EVAL_DIR="$EXP_DIR/evaluation"
RES_DIR="$EXP_DIR/results"
PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/bin/python}"

SEEDS=(42 43 44)
FORMATS=(
  "fmtA_b0p1_lmax8"
  "fmtB_b0p1_lmax6"
  "fmtC_b0p03_anneal_lmax6"
)

CHECKPOINTS=()
for fmt in "${FORMATS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    rn="lsv_stage3_vae_base_ld32_${fmt}_s${seed}"
    ck="$EXP_DIR/checkpoints/${rn}_best.pt"
    CHECKPOINTS+=("$ck")
  done
done

for ck in "${CHECKPOINTS[@]}"; do
  if [[ ! -f "$ck" ]]; then
    echo "[ERROR] Missing checkpoint: $ck"
    exit 1
  fi
done

echo "[$(date '+%F %T')] START stage3 ld32 format eval v1"
echo "[$(date '+%F %T')] PYTHON_BIN=$PYTHON_BIN"

"$PYTHON_BIN" "$EVAL_DIR/analyze_latent_shape.py" \
  --checkpoints "${CHECKPOINTS[@]}" \
  --split test \
  --output_dir "$RES_DIR/latent_shape_test_stage3_ld32_formats_v1"

"$PYTHON_BIN" "$EVAL_DIR/analyze_latent_shape.py" \
  --checkpoints "${CHECKPOINTS[@]}" \
  --split ood_event \
  --output_dir "$RES_DIR/latent_shape_ood_event_stage3_ld32_formats_v1"

"$PYTHON_BIN" "$EVAL_DIR/evaluate_prior_sampling.py" \
  --checkpoints "${CHECKPOINTS[@]}" \
  --split test \
  --num_real_samples 2000 \
  --num_generated_samples 2000 \
  --output_dir "$RES_DIR/prior_sampling_test_stage3_ld32_formats_v1"

"$PYTHON_BIN" "$EVAL_DIR/evaluate_prior_sampling.py" \
  --checkpoints "${CHECKPOINTS[@]}" \
  --split ood_event \
  --num_real_samples 2000 \
  --num_generated_samples 2000 \
  --output_dir "$RES_DIR/prior_sampling_ood_event_stage3_ld32_formats_v1"

"$PYTHON_BIN" "$EVAL_DIR/audit_latent_var_outliers.py" \
  --checkpoints "${CHECKPOINTS[@]}" \
  --splits test ood_event \
  --var_thresholds 10,1000,100000 \
  --top_k 100 \
  --output_dir "$RES_DIR/latent_var_outlier_audit_stage3_ld32_formats_v1"

"$PYTHON_BIN" "$EVAL_DIR/summarize_stage3_ld32_formats_v1.py" \
  --latent_test_csv "$RES_DIR/latent_shape_test_stage3_ld32_formats_v1/latent_shape_summary.csv" \
  --latent_ood_csv "$RES_DIR/latent_shape_ood_event_stage3_ld32_formats_v1/latent_shape_summary.csv" \
  --prior_test_csv "$RES_DIR/prior_sampling_test_stage3_ld32_formats_v1/prior_sampling_realism_summary.csv" \
  --prior_ood_csv "$RES_DIR/prior_sampling_ood_event_stage3_ld32_formats_v1/prior_sampling_realism_summary.csv" \
  --audit_csv "$RES_DIR/latent_var_outlier_audit_stage3_ld32_formats_v1/audit_summary.csv" \
  --output_dir "$RES_DIR/stage3_ld32_formats_v1"

echo "[$(date '+%F %T')] DONE stage3 ld32 format eval v1"
