#!/usr/bin/env bash
set -euo pipefail

python create_embeddings.py \
  --data_dir ../../data/physical_waveforms_snr2_2-15hz \
  --waveform_domain physical_acceleration \
  --waveform_summary ../../data/waveform_summary.csv \
  --target_freq_bins 33 --target_time_bins 701 \
  --ae_checkpoint ../autoencoder/checkpoints/vae-global-physical-v2/best_model.pt

python fetch_station_locations.py \
  --embeddings_dir embeddings/vae-global-physical-v2 \
  --strict

python compute_station_vs30.py \
  --embeddings_dir embeddings/vae-global-physical-v2
