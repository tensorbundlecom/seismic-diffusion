#!/usr/bin/env bash

set -euo pipefail

cd /home/gms/kalem_seismic_paper_repro

/home/gms/miniconda3/bin/python ML/autoencoder/experiments/PaperRepro/training/train_stage1_autoencoder.py \
  --config ML/autoencoder/experiments/PaperRepro/configs/frozen_paper_repro_v1.yaml \
  --run-name "run_$(date +%Y%m%d_%H%M)_s1_ae_hh100_ori4064_logspec128_lat8x32x32_evt801010_s42_v1" \
  --device cuda \
  --epochs 200 \
  --batch-size 64 \
  --num-workers 16
