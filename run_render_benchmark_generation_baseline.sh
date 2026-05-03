#!/usr/bin/env bash

set -euo pipefail

cd /home/gms/kalem_seismic_paper_repro

/home/gms/miniconda3/bin/python ML/autoencoder/experiments/PaperRepro/evaluation/render_benchmark_generation.py \
  --config ML/autoencoder/experiments/PaperRepro/configs/frozen_paper_repro_v1.yaml \
  --stage1-checkpoint ML/autoencoder/experiments/PaperRepro/runs/run_20260424_1940_s1_ae_hh100_ori4064_logspec128_lat8x32x32_evt801010_s42_v1/checkpoints/best_recon.ckpt \
  --stage2-checkpoint ML/autoencoder/experiments/PaperRepro/runs/run_20260426_1728_s2_ledm_hh100_ori4064_logspec128_lat8x32x32_evt801010_s42_v1/checkpoints/best_val_loss.ckpt \
  --device cuda
