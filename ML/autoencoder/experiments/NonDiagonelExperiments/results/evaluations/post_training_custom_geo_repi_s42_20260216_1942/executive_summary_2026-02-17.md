# Executive Summary - NonDiagonel Program (2026-02-17)

## 1) Objective and Hypothesis

Primary hypothesis under test:

- `If FullCovariance posterior has non-negligible off-diagonal terms, the architecture is not optimal.`
- corollary: `there exists an architecture where near-optimal quality is achieved with near-zero non-diagonal dependence.`

This summary consolidates all completed runs and diagnostics for that claim.

---

## 2) What Was Executed

### A. Geometry-conditioned redesign (fixed)

Conditioning was standardized to:

- `magnitude`
- `log1p(epicentral_distance_km)`
- `depth_km`
- `sin(azimuth), cos(azimuth)`
- `station_embedding`

### B. Architecture axes compared

1. `large + ld128` (reference)
2. `small + ld128`
3. `small + ld64`

Each axis includes:

- `BaselineGeo` (diagonal posterior)
- `FullCovGeo` (full covariance posterior)

### C. Evaluation protocol

- OOD set: post-training custom HH (`52` samples)
- unified metric suite: SSIM, S-Corr, SC, STA/LTA Err, LSD, MR-LSD, Arias Err, Env Corr, DTW, XCorr
- additional latent diagnostics:
  - aggregated Gaussian `TC`
  - off-diagonal correlation summaries
  - pairwise Gaussian `MI`
  - basis-rotation stress test (orthogonal rotations)

---

## 3) Training Outcomes

| Model | Best Val Loss | Best Epoch |
|:---|---:|---:|
| BaselineGeo_large_ld128 | **173.7884** | 31 |
| BaselineGeo_small_ld128 | 174.1540 | 28 |
| BaselineGeo_small_ld64 | 176.8245 | 29 |
| FullCovGeo_large_ld128 | 176.5579 | 99 |
| FullCovGeo_small_ld128 | 179.3707 | 29 |
| FullCovGeo_small_ld64 | 178.4879 | 29 |

Key reading:

- `small_ld128 baseline` is close to `large baseline` (+0.3656).
- FullCov variants are consistently higher loss than baseline in matched scales.

---

## 4) OOD Outcomes (52 Samples)

Source: `model_family_eval_small_phase_20260217_1216/summary.md`

### Metric leadership

- strongest overall balance: **`BaselineGeo_small_ld128`**
  - best in SSIM, S-Corr, SC, STA/LTA Err, MR-LSD, Env Corr
- FullCov strengths:
  - `FullCovGeo_small_ld128` best in LSD and DTW
- special case:
  - `BaselineGeo_small_ld64` best XCorr and Arias Err but unstable overall (not competitive as a complete profile)

### Practical interpretation

- FullCov gives local gains on selected spectral/timing metrics.
- Baseline small (`ld128`) gives the best multi-metric balance.

---

## 5) Latent Dependency (TC/MI + Offdiag)

Source: `latent_dependency_tc_mi_20260217_1228/report.md`

| Model | TC_agg ↓ | Offdiag Mean ↓ | Offdiag p95 ↓ | Pairwise MI Mean ↓ |
|:---|---:|---:|---:|---:|
| BaselineGeo_large_ld128 | 69.0160 | 0.1733 | 0.4412 | 0.0279 |
| FullCovGeo_large_ld128 | 42.5538 | 0.1404 | 0.3814 | 0.0181 |
| BaselineGeo_small_ld128 | 72.4135 | 0.1844 | 0.4776 | 0.0314 |
| FullCovGeo_small_ld128 | 38.3087 | **0.1372** | **0.3550** | **0.0171** |
| BaselineGeo_small_ld64 | 42.7390 | 0.2453 | 0.5730 | 0.0525 |
| FullCovGeo_small_ld64 | **28.5853** | 0.1994 | 0.4958 | 0.0359 |

Critical fact for the strict hypothesis:

- none of the models are near zero in off-diagonal terms.
- observed offdiag means are `~0.137 to 0.245`, not in a near-zero regime.

---

## 6) Basis-Rotation Stress Result

Same covariance under random orthogonal basis rotations still changes coordinate off-diagonal summaries.

Implication:

- off-diagonal magnitude is basis-sensitive.
- offdiag alone is not a basis-invariant optimality certificate.
- therefore TC/MI + quality must be read jointly, not offdiag only.

---

## 7) Pareto Quality-Independent View (What It Means)

Source: `latent_dependency_tc_mi_20260217_1228/pareto_scores.json`

Construct:

- `Quality Score`: average min-max normalized OOD metrics (10 metrics).
- `Independence Score`: average min-max normalized dependency metrics (`TC_agg`, offdiag mean/p95, pairwise MI mean).
- Pareto nondominated = no other model is better in both scores at once.

Important caveat:

- this Pareto view is **relative to this 6-model set**, not an absolute physical scale.

---

## 8) Verdict Against Hypothesis

### Strict claim status

Claim:

- `Optimal architecture should push FullCov non-diagonal terms near zero while preserving quality.`

Current evidence:

- no tested model reaches near-zero offdiag.
- quality-optimal model (`BaselineGeo_small_ld128`) still has substantial offdiag in aggregated covariance space.
- reducing latent (`ld64`) did not drive dependence to near-zero; in multiple indicators it worsened stability.

Conclusion:

- **Strict hypothesis is not supported by current results.**

### What is supported

1. FullCov can help selected metrics (LSD/DTW).
2. Best overall OOD quality is currently baseline small (`ld128`).
3. Dependency structure remains non-trivial across all tested architectures.
4. Architecture optimality cannot be concluded from offdiag-only criteria in this setup.

---

## 9) Decision Recommendation (Current Program)

For production/next baseline:

- use `BaselineGeo_small_ld128` as primary reference.

For continued hypothesis program:

- keep strict thresholds explicit (example):
  - `offdiag_mean <= 0.03`
  - `offdiag_p95 <= 0.10`
  - OOD quality drop <= `1-2%` vs best
- continue only if a candidate approaches these thresholds without quality collapse.

---

## 10) Artifact Index

- OOD model family results:
  - `ML/autoencoder/experiments/NonDiagonel/results/evaluations/post_training_custom_geo_repi_s42_20260216_1942/model_family_eval_small_phase_20260217_1216/`
- TC/MI dependency analysis:
  - `ML/autoencoder/experiments/NonDiagonel/results/evaluations/post_training_custom_geo_repi_s42_20260216_1942/latent_dependency_tc_mi_20260217_1228/`
- Full technical report:
  - `ML/autoencoder/experiments/NonDiagonel/results/evaluations/post_training_custom_geo_repi_s42_20260216_1942/comprehensive_hypothesis_report_2026-02-17.md`
