#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$ROOT_DIR"

EXP_DIR="ML/autoencoder/experiments/LatentShapeVAE"
EVAL_DIR="$EXP_DIR/evaluation"
LOG_DIR="$EXP_DIR/logs"
RES_DIR="$EXP_DIR/results"
PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/bin/python}"

mkdir -p "$LOG_DIR"

CKPT_S42="$EXP_DIR/checkpoints/lsv_stage2_vae_base_ld64_b0p1_s42_best.pt"
CKPT_S43="$EXP_DIR/checkpoints/lsv_stage2_vae_base_ld64_b0p1_s43_best.pt"
CKPT_S44="$EXP_DIR/checkpoints/lsv_stage2_vae_base_ld64_b0p1_s44_best.pt"

for ck in "$CKPT_S42" "$CKPT_S43" "$CKPT_S44"; do
  if [[ ! -f "$ck" ]]; then
    echo "[ERROR] Missing checkpoint: $ck"
    exit 1
  fi
done

echo "[$(date '+%F %T')] START stage2 beta=0.1 outlier audit"
echo "[$(date '+%F %T')] PYTHON_BIN=$PYTHON_BIN"

"$PYTHON_BIN" "$EVAL_DIR/audit_latent_var_outliers.py" \
  --checkpoints "$CKPT_S42" "$CKPT_S43" "$CKPT_S44" \
  --splits test ood_event \
  --var_thresholds 10,1000,100000 \
  --top_k 100 \
  --output_dir "$RES_DIR/latent_var_outlier_audit_stage2_beta0p1_v1"

echo "[$(date '+%F %T')] DONE stage2 beta=0.1 outlier audit"
