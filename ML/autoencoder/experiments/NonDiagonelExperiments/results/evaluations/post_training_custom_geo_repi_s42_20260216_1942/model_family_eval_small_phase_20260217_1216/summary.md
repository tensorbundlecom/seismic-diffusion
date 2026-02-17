# NonDiagonel Model Family Evaluation

- Timestamp (UTC): `2026-02-17T09:25:15Z`
- Processed samples: `52`

## Metrics

| Model | SSIM ↑ | S-Corr ↑ | SC ↓ | STA/LTA Err ↓ | LSD ↓ | MR-LSD ↓ | Arias Err ↓ | Env Corr ↑ | DTW ↓ | XCorr ↑ |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BaselineGeo_large_ld128 | 0.6279 | 0.9688 | 0.2019 | 0.0765 | 1.7387 | 1.9421 | 0.3909 | 0.5845 | 12121.33 | 0.2142 |
| FullCovGeo_large_ld128 | 0.6118 | 0.9619 | 0.2191 | 0.0748 | 1.7088 | 1.7996 | 0.5150 | 0.5693 | 11668.84 | 0.2051 |
| BaselineGeo_small_ld128 | 0.6313 | 0.9692 | 0.1949 | 0.0658 | 1.6900 | 1.7827 | 0.4062 | 0.5892 | 12090.83 | 0.2116 |
| FullCovGeo_small_ld128 | 0.6126 | 0.9585 | 0.2266 | 0.0896 | 1.5877 | 1.8351 | 0.6674 | 0.5481 | 11634.71 | 0.1989 |
| BaselineGeo_small_ld64 | 0.6140 | 0.9649 | 0.2087 | 0.0709 | 3.2223 | 1.8249 | 0.3850 | 0.5662 | 11654.77 | 0.2149 |
| FullCovGeo_small_ld64 | 0.5941 | 0.9599 | 0.2226 | 0.0876 | 1.9718 | 1.8542 | 0.6903 | 0.5414 | 11711.58 | 0.1983 |

## FullCov Off-Diagonal Summary

| Model | mean | p95 | max | energy_ratio |
|:---|---:|---:|---:|---:|
| FullCovGeo_large_ld128 | 0.0527 | 0.1376 | 0.7660 | 0.6467 |
| FullCovGeo_small_ld128 | 0.0556 | 0.1414 | 0.7242 | 0.6496 |
| FullCovGeo_small_ld64 | 0.1095 | 0.2996 | 0.8438 | 0.7698 |
