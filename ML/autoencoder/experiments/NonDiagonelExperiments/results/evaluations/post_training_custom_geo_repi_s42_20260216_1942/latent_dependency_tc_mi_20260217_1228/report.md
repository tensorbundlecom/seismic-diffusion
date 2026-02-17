# Latent Dependency Report (TC/MI + Basis Control)

- Timestamp (UTC): `2026-02-17T10:14:30Z`
- Processed samples: `52`
- Skipped samples: `0`
- Rotations per model: `24`

## Aggregated Dependency Summary

| Model | TC_agg ↓ | Offdiag Mean ↓ | Offdiag p95 ↓ | Pairwise MI Mean ↓ | Posterior TC Mean ↓ |
|:---|---:|---:|---:|---:|---:|
| BaselineGeo_large_ld128 | 69.0160 | 0.1733 | 0.4412 | 0.0279 | 0.0000 |
| FullCovGeo_large_ld128 | 42.5538 | 0.1404 | 0.3814 | 0.0181 | 45.8021 |
| BaselineGeo_small_ld128 | 72.4135 | 0.1844 | 0.4776 | 0.0314 | 0.0000 |
| FullCovGeo_small_ld128 | 38.3087 | 0.1372 | 0.3550 | 0.0171 | 52.1209 |
| BaselineGeo_small_ld64 | 42.7390 | 0.2453 | 0.5730 | 0.0525 | 0.0000 |
| FullCovGeo_small_ld64 | 28.5853 | 0.1994 | 0.4958 | 0.0359 | 30.8810 |

## Basis-Rotation Stress (Same Covariance, Rotated Coordinates)

| Model | Offdiag Mean Min | Offdiag Mean Max | TC Min | TC Max |
|:---|---:|---:|---:|---:|
| BaselineGeo_large_ld128 | 0.2160 | 0.2351 | 74.5023 | 76.4346 |
| FullCovGeo_large_ld128 | 0.1434 | 0.1568 | 43.2920 | 44.2515 |
| BaselineGeo_small_ld128 | 0.2128 | 0.2463 | 76.7700 | 79.3237 |
| FullCovGeo_small_ld128 | 0.1344 | 0.1480 | 38.9391 | 39.8273 |
| BaselineGeo_small_ld64 | 0.2568 | 0.2919 | 44.2421 | 46.2617 |
| FullCovGeo_small_ld64 | 0.1914 | 0.2162 | 29.2103 | 30.3139 |

## OOD + Dependency Combined View

| Model | Quality Score ↑ | Independence Score ↑ | Pareto Non-Dominated |
|:---|---:|---:|:---:|
| BaselineGeo_large_ld128 | 0.6949 | 0.5108 | yes |
| FullCovGeo_large_ld128 | 0.5975 | 0.8751 | yes |
| BaselineGeo_small_ld128 | 0.8731 | 0.3993 | yes |
| FullCovGeo_small_ld128 | 0.3426 | 0.9445 | yes |
| BaselineGeo_small_ld64 | 0.6699 | 0.1693 | no |
| FullCovGeo_small_ld64 | 0.2507 | 0.5618 | no |

## Artifacts

- `latent_dependency_summary.json`
- `rotation_stress_raw.json`
- `per_sample_posterior_tc.jsonl`
- `pareto_scores.json`
- `plots/tc_agg_bar.png`
- `plots/offdiag_mean_bar.png`
- `plots/pairwise_mi_mean_bar.png`
- `plots/rotation_stress_offdiag_mean_box.png`
- `plots/rotation_stress_tc_box.png`
- `plots/pareto_quality_vs_independence.png`
- `plots/corr_heatmaps/*.png`
