# NonDiagonel Evaluation Summary: `post_training_custom_geo_repi_s42_20260216_1942`

## Run Metadata

- Timestamp (UTC): `2026-02-16T16:41:11Z`
- Processed samples: `52`
- Skipped samples: `0`
- OOD data dir: `data/ood_waveforms/post_training_custom/filtered`
- OOD catalog: `data/events/ood_catalog_post_training.txt`
- Station subset file: `data/station_list_post_custom.json`
- Baseline checkpoint: `ML/autoencoder/experiments/NonDiagonel/checkpoints/baseline_geo_repi_external_s42_20260215_best.pt`
- FullCov checkpoint: `ML/autoencoder/experiments/NonDiagonel/checkpoints/fullcov_geo_repi_external_s42_20260215_best.pt`

## Metrics

| Model | SSIM ↑ | S-Corr ↑ | SC ↓ | STA/LTA Err ↓ | LSD ↓ | MR-LSD ↓ | Arias Err ↓ | Env Corr ↑ | DTW ↓ | XCorr ↑ |
|:--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| BaselineGeo | 0.6279 | 0.9688 | 0.2019 | 0.0765 | 1.7387 | 1.9418 | 0.3914 | 0.5863 | 12314.05 | 0.2071 |
| FullCovGeo | 0.6118 | 0.9619 | 0.2191 | 0.0748 | 1.7088 | 1.8029 | 0.5151 | 0.5747 | 11560.89 | 0.2043 |

## FullCov Posterior Correlation (Off-Diagonal)

- `mean_abs_corr_offdiag`: mean=0.0527, std=0.0035, min=0.0444, max=0.0630
- `max_abs_corr_offdiag`: mean=0.7660, std=0.0349, min=0.7018, max=0.8875
- `p95_abs_corr_offdiag`: mean=0.1376, std=0.0097, min=0.1137, max=0.1647
- `offdiag_energy_ratio`: mean=0.6467, std=0.0221, min=0.5936, max=0.7052

## Metric Definitions

- `SSIM`: Spectrogram structural similarity (higher is better).
- `S-Corr`: Spectral correlation (higher is better).
- `SC`: Spectral convergence (lower is better).
- `STA/LTA Err`: Onset energy ratio error (lower is better).
- `LSD`: Log-spectral distance (lower is better).
- `MR-LSD`: Multi-resolution log-spectral distance (lower is better).
- `Arias Err`: Arias intensity error (lower is better).
- `Env Corr`: Envelope correlation (higher is better).
- `DTW`: Dynamic time warping distance (lower is better).
- `XCorr`: Max cross-correlation (higher is better).

