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
  --lr 2e-4
  --grad_clip_norm 0.5
  --amp 0
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

# Previous run diverged to NaN after early stage.
run_if_needed "lsv_stage1_ae_base_ld64_s42_stablev1" \
  --ablation_mode ae

# Previous run had severe NaN/Inf with beta0. Deterministic beta0 path is patched in train_single.py.
run_if_needed "lsv_stage1_beta0_base_ld64_s42_stablev1" \
  --ablation_mode beta0

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ALL_STABLE_RERUNS_COMPLETED"
