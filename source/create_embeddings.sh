python create_embeddings.py \
  --data_dir ../../data/physical_waveforms_snr2_2-15hz \
  --waveform_summary ../../data/waveform_summary.csv \
  --target_freq_bins 33 --target_time_bins 701 \
  --ae_checkpoint ../autoencoder/checkpoints/vae-global-physical-v2/best_model.pt
