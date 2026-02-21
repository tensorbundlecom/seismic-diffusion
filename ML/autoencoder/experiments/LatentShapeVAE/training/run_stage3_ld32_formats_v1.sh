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
  --latent_dim 32
  --logvar_mode bounded_sigmoid
  --logvar_min -12.0
  --max_steps 12000
  --val_check_every_steps 1000
  --train_log_every_steps 100
  --batch_size 64
  --num_workers 8
  --lr 2e-4
  --grad_clip_norm 0.5
  --amp 0
)

SEEDS=(42 43 44)

# fmt_name|beta|anneal_enabled|anneal_start|anneal_end|anneal_warmup|logvar_max
FORMATS=(
  "fmtA_b0p1_lmax8|0.1|0|0.0|0.1|0|8.0"
  "fmtB_b0p1_lmax6|0.1|0|0.0|0.1|0|6.0"
  "fmtC_b0p03_anneal_lmax6|0.03|1|0.0|0.03|4000|6.0"
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

for f in "${FORMATS[@]}"; do
  IFS='|' read -r fmt_name beta anneal_enabled anneal_start anneal_end anneal_warmup logvar_max <<< "$f"
  for seed in "${SEEDS[@]}"; do
    run_name="lsv_stage3_vae_base_ld32_${fmt_name}_s${seed}"
    run_if_needed "$run_name" \
      --seed "$seed" \
      --beta "$beta" \
      --anneal_enabled "$anneal_enabled" \
      --anneal_beta_start "$anneal_start" \
      --anneal_beta_end "$anneal_end" \
      --anneal_warmup_steps "$anneal_warmup" \
      --logvar_max "$logvar_max"
  done
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ALL_STAGE3_LD32_FORMATS_V1_COMPLETED"
