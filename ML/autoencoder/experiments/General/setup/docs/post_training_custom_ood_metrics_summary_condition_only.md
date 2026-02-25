# Post-Training Custom OOD (HH-only) Metrics Summary

This summary compares all models on the custom post-training OOD set (HH-only, station subset).
Inference mode: `condition_only`.

## Metrics Table

| Model | SSIM | LSD | SC | S-Corr | STA/LTA Err | MR-LSD | Arias Err | Env Corr | DTW | XCorr |
|:--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline | 0.4257 | 2.5442 | 0.7819 | 0.5660 | 0.2089 | 3.1804 | 0.6927 | -0.1540 | 13606.01 | 0.1631 |
| FullCov | 0.4217 | 2.5244 | 0.7509 | 0.5756 | 0.2120 | 3.0764 | 0.8383 | -0.1268 | 13634.18 | 0.1419 |
| Flow | 0.2888 | 2.6118 | 0.6730 | 0.5933 | 0.1941 | 3.2959 | 0.9976 | -0.1449 | 13142.66 | 0.0841 |
| WBaseline | 0.4339 | 2.0347 | 0.7346 | 0.5514 | 0.1808 | 3.0564 | 0.8822 | -0.1314 | 13434.35 | 0.1390 |
| WFullCov | 0.4308 | 2.4770 | 0.7838 | 0.5920 | 0.2116 | 3.1976 | 0.7584 | -0.1402 | 13990.16 | 0.1683 |
| WFlow | 0.3416 | 2.5140 | 0.6686 | 0.6001 | 0.2181 | 3.1900 | 0.9895 | -0.1695 | 13313.21 | 0.1235 |
| CVAE_MRLoss | 0.4181 | 2.0035 | 0.7723 | 0.5558 | 0.1974 | 3.1407 | 0.8491 | -0.1453 | 13283.06 | 0.1319 |

## Metric Definitions

- `SSIM`: Spectrogram structural similarity (higher is better).
- `LSD`: Log-spectral distance (lower is better).
- `SC`: Spectral convergence (lower is better).
- `S-Corr`: Spectral correlation (higher is better).
- `STA/LTA Err`: Onset energy ratio error (lower is better).
- `MR-LSD`: Multi-resolution log-spectral distance (lower is better).
- `Arias Err`: Arias intensity error (lower is better).
- `Env Corr`: Envelope correlation (higher is better).
- `DTW`: Dynamic time warping distance (lower is better).
- `XCorr`: Maximum cross-correlation (higher is better).

## Overall Interpretation

Best-per-metric highlights:
- `ssim`: **WBaseline**
- `lsd`: **CVAE_MRLoss**
- `sc`: **WFlow**
- `s_corr`: **WFlow**
- `sta_lta_err`: **WBaseline**
- `mr_lsd`: **WBaseline**
- `arias_err`: **Baseline**
- `env_corr`: **FullCov**
- `dtw`: **Flow**
- `xcorr`: **WFullCov**
General interpretation:
The summary shows trade-offs between structural similarity (SSIM/S-Corr), spectral fidelity (LSD/MR-LSD), energy matching (Arias Err), and temporal alignment (DTW/XCorr).
Models that improve LSD or MR-LSD may not lead on SSIM, indicating sharper spectral detail at some cost to global structure.
Use DTW and XCorr together to judge timing vs phase alignment, and prioritize metrics based on the downstream objective (onset accuracy vs spectral texture vs waveform alignment).
