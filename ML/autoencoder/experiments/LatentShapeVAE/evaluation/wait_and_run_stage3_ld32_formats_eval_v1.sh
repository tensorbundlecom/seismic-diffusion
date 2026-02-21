#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$ROOT_DIR"

EXP_DIR="ML/autoencoder/experiments/LatentShapeVAE"
SEEDS=(42 43 44)
FORMATS=(
  "fmtA_b0p1_lmax8"
  "fmtB_b0p1_lmax6"
  "fmtC_b0p03_anneal_lmax6"
)

while true; do
  missing=0
  for fmt in "${FORMATS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      ck="$EXP_DIR/checkpoints/lsv_stage3_vae_base_ld32_${fmt}_s${seed}_best.pt"
      if [[ ! -f "$ck" ]]; then
        missing=$((missing + 1))
      fi
    done
  done

  echo "[$(date '+%F %T')] WAIT checkpoints missing=$missing"
  if [[ "$missing" -eq 0 ]]; then
    break
  fi
  sleep 120
done

echo "[$(date '+%F %T')] START eval batch"
"$EXP_DIR/evaluation/run_stage3_ld32_formats_v1.sh"
echo "[$(date '+%F %T')] DONE eval batch"
