#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/bin/python}"

LOG_DIR="$ROOT_DIR/logs"
CKPT_DIR="$ROOT_DIR/checkpoints"
mkdir -p "$LOG_DIR" "$CKPT_DIR"

COMMON_ARGS=(
  --ablation_mode vae
  --backbone base
  --latent_dim 64
  --beta 0.1
  --max_steps 12000
  --val_check_every_steps 1000
  --train_log_every_steps 100
  --batch_size 64
  --num_workers 8
  --lr 2e-4
  --grad_clip_norm 0.5
  --amp 0
)

run_if_needed() {
  local run_name="$1"
  local seed="$2"
  local ckpt_best="$CKPT_DIR/${run_name}_best.pt"

  if [[ -f "$ckpt_best" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] SKIP ${run_name} (checkpoint exists)"
    return 0
  fi

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] START ${run_name}"
  "$PYTHON_BIN" "$ROOT_DIR/training/train_single.py" \
    --run_name "$run_name" \
    --seed "$seed" \
    "${COMMON_ARGS[@]}"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE  ${run_name}"
}

run_if_needed "lsv_stage2_vae_base_ld64_b0p1_s42" 42
run_if_needed "lsv_stage2_vae_base_ld64_b0p1_s43" 43
run_if_needed "lsv_stage2_vae_base_ld64_b0p1_s44" 44

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ALL_STAGE2_BETA0P1_SEEDS_COMPLETED"
