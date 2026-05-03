#!/usr/bin/env bash

set -euo pipefail

cd /home/gms/kalem_seismic_paper_repro

/home/gms/miniconda3/bin/python ML/autoencoder/experiments/PaperRepro/training/train_paper_metrics_classifier.py \
  --config ML/autoencoder/experiments/PaperRepro/configs/frozen_paper_repro_v1.yaml \
  --device cuda
