#!/usr/bin/env bash

set -euo pipefail

cd /home/gms/kalem_seismic_paper_repro

/home/gms/miniconda3/bin/python ML/autoencoder/experiments/PaperRepro/training/train_stage2_edm.py \
  --config ML/autoencoder/experiments/PaperRepro/configs/frozen_paper_repro_v1.yaml \
  --cache-root ML/autoencoder/experiments/PaperRepro/results/stage2_cache/run_20260424_1940_s1_ae_hh100_ori4064_logspec128_lat8x32x32_evt801010_s42_v1 \
  --device cuda \
  --epochs 200 \
  --batch-size 64 \
  --num-workers 8
