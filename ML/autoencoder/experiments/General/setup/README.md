# OOD Setup (Post-Training, Custom Station Subset)

This folder contains data preparation scripts for post-training OOD waveforms.

## Custom OOD (HH-only, station subset)
Target stations:
`ADVT, ARMT, KCTX, YLV, GEML, GELI`

Output directories:
- `data/ood_waveforms/post_training_custom/raw/HH`
- `data/ood_waveforms/post_training_custom/filtered/HH`

Scripts:
- `download_post_training_ood_custom.py`
  - Downloads HH waveforms for `data/events/ood_catalog_post_training.txt`
  - Uses KOERI with IRIS fallback
- `preprocess_post_training_ood_custom.py`
  - Applies the same preprocessing as the post-training training pipeline
  - Detrend, taper, resample to `100 Hz`
  - **Bandpass: `0.5–45 Hz`** (matches external training data filter)
- `build_post_training_custom_ood_docs.py`
  - Builds event summary and station distance tables
  - Outputs to `ML/autoencoder/experiments/General/setup/docs`

Run order:
1. `python ML/autoencoder/experiments/General/setup/download_post_training_ood_custom.py`
2. `python ML/autoencoder/experiments/General/setup/preprocess_post_training_ood_custom.py`
3. `python ML/autoencoder/experiments/General/setup/build_post_training_custom_ood_docs.py`

Notes:
- Channel is **HH-only**.
- Station list for evaluation/visualization is `data/station_list_post_custom.json`.
