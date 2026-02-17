# NonDiagonel Decision Report

## Scope

- Compare `BaselineGeo` vs `FullCovGeo` on paired OOD samples.
- Signed improvement convention: positive means **FullCov better**.
- Samples: `52`

## Bootstrap (Paired) Summary

| Metric | Baseline Mean | FullCov Mean | Signed Diff (FullCov better +) | 95% CI | p(diff>0) |
|:---|---:|---:|---:|:---:|---:|
| ssim | 0.6279 | 0.6118 | -0.0161 | [-0.0201, -0.0122] | 0.000 |
| lsd | 1.7387 | 1.7088 | +0.0299 | [-0.0748, 0.1321] | 0.719 |
| sc | 0.2019 | 0.2191 | -0.0172 | [-0.0202, -0.0142] | 0.000 |
| s_corr | 0.9688 | 0.9619 | -0.0069 | [-0.0079, -0.0059] | 0.000 |
| sta_lta_err | 0.0765 | 0.0748 | +0.0017 | [-0.0139, 0.0175] | 0.587 |
| mr_lsd | 1.9418 | 1.8029 | +0.1390 | [0.0850, 0.1930] | 1.000 |
| arias_err | 0.3914 | 0.5151 | -0.1237 | [-0.2005, -0.0413] | 0.002 |
| env_corr | 0.5863 | 0.5747 | -0.0117 | [-0.0267, 0.0035] | 0.064 |
| dtw | 12314.0453 | 11560.8915 | +753.1538 | [4.8958, 1820.1092] | 0.976 |
| xcorr | 0.2071 | 0.2043 | -0.0028 | [-0.0153, 0.0081] | 0.322 |

## Robust Metric Wins

- Decision: `favor_baseline`
- Robust FullCov wins: `['mr_lsd', 'dtw']`
- Robust Baseline wins: `['ssim', 'sc', 's_corr', 'arias_err']`
- Uncertain: `['lsd', 'sta_lta_err', 'env_corr', 'xcorr']`

## Station Breakdown (Primary Composite)

| Station | Count | Primary Z-Composite | FullCov Wins | Baseline Wins |
|:---|---:|---:|---:|---:|
| ADVT | 10 | +0.5384 | 2 | 2 |
| GELI | 10 | -0.6071 | 0 | 4 |
| YLV | 9 | +0.2695 | 3 | 1 |
| GEML | 9 | -0.4344 | 1 | 3 |
| ARMT | 8 | +0.2231 | 3 | 1 |
| KCTX | 6 | -0.1690 | 2 | 2 |

## Event Breakdown (Primary Composite)

| Event | Count | Primary Z-Composite | FullCov Wins | Baseline Wins |
|:---|---:|---:|---:|---:|
| OOD_POST_08 | 6 | -0.1128 | 3 | 1 |
| OOD_POST_10 | 6 | -0.1210 | 2 | 2 |
| OOD_POST_07 | 6 | -0.1271 | 2 | 2 |
| OOD_POST_02 | 6 | -0.1306 | 2 | 2 |
| OOD_POST_04 | 5 | +0.3419 | 3 | 1 |
| OOD_POST_01 | 5 | +0.1341 | 3 | 1 |
| OOD_POST_03 | 5 | +0.1089 | 2 | 2 |
| OOD_POST_09 | 5 | -0.2617 | 2 | 2 |
| OOD_POST_06 | 4 | +0.0298 | 2 | 2 |
| OOD_POST_05 | 4 | -0.0465 | 2 | 2 |

## Off-Diagonal vs Quality (Correlation)

| OffDiag Feature | Metric | Pearson | Spearman |
|:---|:---|---:|---:|
| offdiag_energy_ratio | mr_lsd | -0.5792 | -0.5830 |
| max_abs_corr_offdiag | lsd | -0.5414 | -0.5705 |
| mean_abs_corr_offdiag | mr_lsd | -0.5586 | -0.5680 |
| max_abs_corr_offdiag | mr_lsd | -0.5210 | -0.5572 |
| offdiag_energy_ratio | lsd | -0.4843 | -0.4858 |
| p95_abs_corr_offdiag | mr_lsd | -0.4815 | -0.4820 |
| mean_abs_corr_offdiag | lsd | -0.4657 | -0.4781 |
| max_abs_corr_offdiag | sta_lta_err | -0.2531 | -0.4409 |
| mean_abs_corr_offdiag | ssim | -0.3451 | -0.3940 |
| p95_abs_corr_offdiag | lsd | -0.3795 | -0.3799 |
| offdiag_energy_ratio | ssim | -0.2970 | -0.3500 |
| offdiag_energy_ratio | sc | -0.2919 | -0.3144 |

## Interpretation Guardrails

- Use primary metrics (`mr_lsd`, `dtw`, `lsd`, `ssim`) for direction.
- Keep guardrails (`s_corr`, `sc`, `arias_err`, `env_corr`, `xcorr`) from regressing.
- Prefer architecture change only if primary gains are robust and guardrail losses are limited.
- `Primary Z-Composite` averages primary signed improvements after per-metric std normalization.
