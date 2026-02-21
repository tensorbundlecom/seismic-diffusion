#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$ROOT_DIR"

EXP_DIR="ML/autoencoder/experiments/LatentShapeVAE"
SEEDS=(42 43 44 45 46 47 48 49 50 51)

while true; do
  missing=0
  for s in "${SEEDS[@]}"; do
    ck="$EXP_DIR/checkpoints/lsv_stage2_vae_base_ld64_b0p1_s${s}_logvfixv2_best.pt"
    if [[ ! -f "$ck" ]]; then
      missing=$((missing + 1))
    fi
  done

  echo "[$(date '+%F %T')] WAIT checkpoints missing=$missing"
  if [[ "$missing" -eq 0 ]]; then
    break
  fi
  sleep 120
done

echo "[$(date '+%F %T')] START eval batch"
"$EXP_DIR/evaluation/run_stage2_beta0p1_logvarfix_10seeds_v2.sh"
echo "[$(date '+%F %T')] DONE eval batch"
