#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$ROOT_DIR"

EXP_DIR="ML/autoencoder/experiments/LatentShapeVAE"
EVAL_DIR="$EXP_DIR/evaluation"
RES_DIR="$EXP_DIR/results"
PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/bin/python}"

CKPT_S42_OLD="$EXP_DIR/checkpoints/lsv_stage2_vae_base_ld64_b0p1_s42_best.pt"
CKPT_S43_OLD="$EXP_DIR/checkpoints/lsv_stage2_vae_base_ld64_b0p1_s43_best.pt"
CKPT_S44_OLD="$EXP_DIR/checkpoints/lsv_stage2_vae_base_ld64_b0p1_s44_best.pt"
CKPT_S43_NEW="$EXP_DIR/checkpoints/lsv_stage2_vae_base_ld64_b0p1_s43_logvfixv1_best.pt"
CKPT_S44_NEW="$EXP_DIR/checkpoints/lsv_stage2_vae_base_ld64_b0p1_s44_logvfixv1_best.pt"

for ck in "$CKPT_S42_OLD" "$CKPT_S43_OLD" "$CKPT_S44_OLD" "$CKPT_S43_NEW" "$CKPT_S44_NEW"; do
  if [[ ! -f "$ck" ]]; then
    echo "[ERROR] Missing checkpoint: $ck"
    exit 1
  fi
done

echo "[$(date '+%F %T')] START stage2 beta0.1 logvarfix compare"
echo "[$(date '+%F %T')] PYTHON_BIN=$PYTHON_BIN"

"$PYTHON_BIN" "$EVAL_DIR/analyze_latent_shape.py" \
  --checkpoints "$CKPT_S42_OLD" "$CKPT_S43_OLD" "$CKPT_S44_OLD" "$CKPT_S43_NEW" "$CKPT_S44_NEW" \
  --split test \
  --output_dir "$RES_DIR/latent_shape_test_stage2_beta0p1_logvarfix_compare_v1"

"$PYTHON_BIN" "$EVAL_DIR/analyze_latent_shape.py" \
  --checkpoints "$CKPT_S42_OLD" "$CKPT_S43_OLD" "$CKPT_S44_OLD" "$CKPT_S43_NEW" "$CKPT_S44_NEW" \
  --split ood_event \
  --output_dir "$RES_DIR/latent_shape_ood_event_stage2_beta0p1_logvarfix_compare_v1"

"$PYTHON_BIN" "$EVAL_DIR/evaluate_prior_sampling.py" \
  --checkpoints "$CKPT_S42_OLD" "$CKPT_S43_OLD" "$CKPT_S44_OLD" "$CKPT_S43_NEW" "$CKPT_S44_NEW" \
  --split test \
  --num_real_samples 2000 \
  --num_generated_samples 2000 \
  --output_dir "$RES_DIR/prior_sampling_test_stage2_beta0p1_logvarfix_compare_v1"

"$PYTHON_BIN" "$EVAL_DIR/evaluate_prior_sampling.py" \
  --checkpoints "$CKPT_S42_OLD" "$CKPT_S43_OLD" "$CKPT_S44_OLD" "$CKPT_S43_NEW" "$CKPT_S44_NEW" \
  --split ood_event \
  --num_real_samples 2000 \
  --num_generated_samples 2000 \
  --output_dir "$RES_DIR/prior_sampling_ood_event_stage2_beta0p1_logvarfix_compare_v1"

"$PYTHON_BIN" "$EVAL_DIR/audit_latent_var_outliers.py" \
  --checkpoints "$CKPT_S42_OLD" "$CKPT_S43_OLD" "$CKPT_S44_OLD" "$CKPT_S43_NEW" "$CKPT_S44_NEW" \
  --splits test ood_event \
  --var_thresholds 10,1000,100000 \
  --top_k 100 \
  --output_dir "$RES_DIR/latent_var_outlier_audit_stage2_beta0p1_logvarfix_compare_v1"

echo "[$(date '+%F %T')] DONE stage2 beta0.1 logvarfix compare"
