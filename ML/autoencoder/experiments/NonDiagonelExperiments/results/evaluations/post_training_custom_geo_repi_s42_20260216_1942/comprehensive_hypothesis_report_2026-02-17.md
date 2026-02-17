# NonDiagonel Comprehensive Hypothesis Report (2026-02-17)

## Scope

Goal: test the claim  
`"If FullCov off-diagonals are not close to zero, architecture is not optimal; an optimal architecture should exist."`

Minimum additional checks requested and completed:

1. `TC/MI` latent-dependency diagnostics
2. basis-rotation control (same covariance, rotated coordinate system)
3. combined OOD-quality vs independence decision view

---

## Models Compared

1. `BaselineGeo_large_ld128`
2. `FullCovGeo_large_ld128`
3. `BaselineGeo_small_ld128`
4. `FullCovGeo_small_ld128`
5. `BaselineGeo_small_ld64`
6. `FullCovGeo_small_ld64`

OOD set: post-training custom HH, `52` waveform samples.

---

## Training Results (Best Validation Loss)

| Model | Best Val Loss | Best Epoch |
|:---|---:|---:|
| BaselineGeo_large_ld128 | 173.7884 | 31 |
| FullCovGeo_large_ld128 | 176.5579 | 99 |
| BaselineGeo_small_ld128 | 174.1540 | 28 |
| FullCovGeo_small_ld128 | 179.3707 | 29 |
| BaselineGeo_small_ld64 | 176.8245 | 29 |
| FullCovGeo_small_ld64 | 178.4879 | 29 |

Key point: `small_ld128 baseline` stayed very close to large baseline, while small FullCov variants degraded more.

---

## OOD Results (Aggregated)

Source:  
`model_family_eval_small_phase_20260217_1216/summary.md`

| Model | SSIM ↑ | S-Corr ↑ | SC ↓ | STA/LTA Err ↓ | LSD ↓ | MR-LSD ↓ | Arias Err ↓ | Env Corr ↑ | DTW ↓ |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BaselineGeo_large_ld128 | 0.6279 | 0.9688 | 0.2019 | 0.0765 | 1.7387 | 1.9421 | 0.3909 | 0.5845 | 12121.33 |
| FullCovGeo_large_ld128 | 0.6118 | 0.9619 | 0.2191 | 0.0748 | 1.7088 | 1.7996 | 0.5150 | 0.5693 | 11668.84 |
| BaselineGeo_small_ld128 | 0.6313 | 0.9692 | 0.1949 | 0.0658 | 1.6900 | 1.7827 | 0.4062 | 0.5892 | 12090.83 |
| FullCovGeo_small_ld128 | 0.6126 | 0.9585 | 0.2266 | 0.0896 | 1.5877 | 1.8351 | 0.6674 | 0.5481 | 11634.71 |
| BaselineGeo_small_ld64 | 0.6140 | 0.9649 | 0.2087 | 0.0709 | 3.2223 | 1.8249 | 0.3850 | 0.5662 | 11654.77 |
| FullCovGeo_small_ld64 | 0.5941 | 0.9599 | 0.2226 | 0.0876 | 1.9718 | 1.8542 | 0.6903 | 0.5414 | 11711.58 |

Metric winners:

- most metrics leader: `BaselineGeo_small_ld128`
- LSD and DTW leader: `FullCovGeo_small_ld128`

---

## TC/MI + Dependency Results

Source:  
`latent_dependency_tc_mi_20260217_1228/report.md`

| Model | TC_agg ↓ | Offdiag Mean ↓ | Offdiag p95 ↓ | Pairwise MI Mean ↓ | Posterior TC Mean ↓ |
|:---|---:|---:|---:|---:|---:|
| BaselineGeo_large_ld128 | 69.0160 | 0.1733 | 0.4412 | 0.0279 | 0.0000 |
| FullCovGeo_large_ld128 | 42.5538 | 0.1404 | 0.3814 | 0.0181 | 45.8021 |
| BaselineGeo_small_ld128 | 72.4135 | 0.1844 | 0.4776 | 0.0314 | 0.0000 |
| FullCovGeo_small_ld128 | 38.3087 | 0.1372 | 0.3550 | 0.0171 | 52.1209 |
| BaselineGeo_small_ld64 | 42.7390 | 0.2453 | 0.5730 | 0.0525 | 0.0000 |
| FullCovGeo_small_ld64 | 28.5853 | 0.1994 | 0.4958 | 0.0359 | 30.8810 |

Observation:

- strict near-zero offdiag target is not met by any model.
- representative offdiag means are `0.137` to `0.245`, far from near-zero.

---

## Basis-Rotation Control

Same aggregated covariance, random orthogonal rotations (`24` draws):

| Model | Offdiag Mean Min | Offdiag Mean Max | TC Min | TC Max |
|:---|---:|---:|---:|---:|
| BaselineGeo_large_ld128 | 0.2160 | 0.2351 | 74.5023 | 76.4346 |
| FullCovGeo_large_ld128 | 0.1434 | 0.1568 | 43.2920 | 44.2515 |
| BaselineGeo_small_ld128 | 0.2128 | 0.2463 | 76.7700 | 79.3237 |
| FullCovGeo_small_ld128 | 0.1344 | 0.1480 | 38.9391 | 39.8273 |
| BaselineGeo_small_ld64 | 0.2568 | 0.2919 | 44.2421 | 46.2617 |
| FullCovGeo_small_ld64 | 0.1914 | 0.2162 | 29.2103 | 30.3139 |

Interpretation:

- dependency indicators are coordinate-sensitive.
- changing only basis (without changing covariance eigen-spectrum) changes offdiag/TC ranges.

---

## Strict-Hypothesis Check

Tested strict criterion example:

- `Offdiag mean <= 0.03`
- `Offdiag p95 <= 0.10`
- quality near best (within `2%` quality-score band)

Result:

- no model satisfies the offdiag thresholds.
- best-quality model (`BaselineGeo_small_ld128`) still has offdiag mean `0.1844`, p95 `0.4776`.

Therefore strict claim is **not supported** by current evidence.

---

## What Is Supported

1. FullCov can improve selected spectral/timing metrics (LSD/DTW) in some settings.
2. Baseline remains strongest on overall OOD metric balance.
3. Dependency does not collapse to near-zero under current architecture sweeps.
4. Offdiag-only criterion is insufficient as a sole optimality test in this setup.

---

## Artifacts

- OOD family comparison:
  - `ML/autoencoder/experiments/NonDiagonel/results/evaluations/post_training_custom_geo_repi_s42_20260216_1942/model_family_eval_small_phase_20260217_1216/summary.md`
- TC/MI + basis control:
  - `ML/autoencoder/experiments/NonDiagonel/results/evaluations/post_training_custom_geo_repi_s42_20260216_1942/latent_dependency_tc_mi_20260217_1228/report.md`
  - `ML/autoencoder/experiments/NonDiagonel/results/evaluations/post_training_custom_geo_repi_s42_20260216_1942/latent_dependency_tc_mi_20260217_1228/plots/`
