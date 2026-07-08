python train.py --data_dir ../../data/filtered_waveforms_snr2_2-15hz \
  --nperseg 50 --noverlap 40 --nfft 64 \
  --use_phasenet_perceptual --phasenet_weight 100_000 --phasenet_pretrained stead \
  --beta 0.1 --latent_channels 16 --channels HH HN EH BH \
  --target_freq_bins 33 --target_time_bins 701 