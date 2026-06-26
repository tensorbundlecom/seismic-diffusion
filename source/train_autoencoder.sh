ython train.py --nperseg 512 --noverlap 448 --nfft 512 \
  --use_phasenet_perceptual --phasenet_weight 100_000 --phasenet_pretrained stead \
  --beta 0.1 --latent_channels 16 --channels HH HN EH BH \
  --target_freq_bins 64 --target_time_bins 128