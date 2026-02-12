# Post-Training Custom OOD (HH-only) Metrics Summary

This summary compares all models on the custom post-training OOD set (HH-only, station subset).

## Metrics Table

| Model | SSIM | LSD | SC | S-Corr | STA/LTA Err | MR-LSD | Arias Err | Env Corr | DTW | XCorr |
|:--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline | 0.6349 | 2.2125 | 0.1911 | 0.9705 | 0.0639 | 1.6669 | 0.4178 | 0.5763 | 11212.07 | 0.2278 |
| FullCov | 0.6075 | 1.9000 | 0.2208 | 0.9607 | 0.0863 | 1.8472 | 0.4096 | 0.5729 | 11757.54 | 0.2124 |
| Flow | 0.6002 | 2.1242 | 0.1918 | 0.9701 | 0.0631 | 2.1895 | 0.3415 | 0.5846 | 12062.00 | 0.2036 |
| WBaseline | 0.6237 | 1.8123 | 0.1822 | 0.9731 | 0.0626 | 1.9670 | 0.3603 | 0.5996 | 11859.45 | 0.2179 |
| WFullCov | 0.6065 | 1.7589 | 0.2126 | 0.9639 | 0.0766 | 1.8381 | 0.5054 | 0.5693 | 11493.82 | 0.2003 |
| WFlow | 0.6221 | 1.8287 | 0.1991 | 0.9686 | 0.0931 | 1.7369 | 0.3094 | 0.5976 | 12190.66 | 0.2084 |
| CVAE_MRLoss | 0.5864 | 2.0513 | 0.1951 | 0.9694 | 0.0611 | 2.2237 | 0.4562 | 0.5885 | 11722.73 | 0.1912 |

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
- `ssim`: **Baseline**
- `lsd`: **WFullCov**
- `sc`: **WBaseline**
- `s_corr`: **WBaseline**
- `sta_lta_err`: **CVAE_MRLoss**
- `mr_lsd`: **Baseline**
- `arias_err`: **WFlow**
- `env_corr`: **WBaseline**
- `dtw`: **Baseline**
- `xcorr`: **Baseline**
General interpretation:
The summary shows trade-offs between structural similarity (SSIM/S-Corr), spectral fidelity (LSD/MR-LSD), energy matching (Arias Err), and temporal alignment (DTW/XCorr).
Models that improve LSD or MR-LSD may not lead on SSIM, indicating sharper spectral detail at some cost to global structure.
Use DTW and XCorr together to judge timing vs phase alignment, and prioritize metrics based on the downstream objective (onset accuracy vs spectral texture vs waveform alignment).
