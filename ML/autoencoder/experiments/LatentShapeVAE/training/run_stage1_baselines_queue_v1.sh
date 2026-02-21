#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/bin/python}"

LOG_DIR="$ROOT_DIR/logs"
CKPT_DIR="$ROOT_DIR/checkpoints"
mkdir -p "$LOG_DIR" "$CKPT_DIR"

COMMON_ARGS=(
  --backbone base
  --latent_dim 64
  --seed 42
  --max_steps 12000
  --val_check_every_steps 1000
  --train_log_every_steps 100
  --batch_size 64
  --num_workers 8
)

run_if_needed() {
  local run_name="$1"
  shift
  local ckpt_best="$CKPT_DIR/${run_name}_best.pt"

  if [[ -f "$ckpt_best" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] SKIP ${run_name} (checkpoint exists)"
    return 0
  fi

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] START ${run_name}"
  "$PYTHON_BIN" "$ROOT_DIR/training/train_single.py" \
    --run_name "$run_name" \
    "${COMMON_ARGS[@]}" \
    "$@"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE  ${run_name}"
}

run_if_needed "lsv_stage1_ae_base_ld64_s42" \
  --ablation_mode ae

run_if_needed "lsv_stage1_beta0_base_ld64_s42" \
  --ablation_mode beta0

run_if_needed "lsv_stage1_vae_base_ld64_b0p03_anneal_s42" \
  --ablation_mode vae \
  --beta 0.03 \
  --anneal_enabled 1 \
  --anneal_beta_start 0.0 \
  --anneal_beta_end 0.03 \
  --anneal_warmup_steps 4000

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ALL_RUNS_COMPLETED"
