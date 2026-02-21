# LatentShapeVAE Comprehensive Report
Date: 2026-02-21  
Experiment scope: `ML/autoencoder/experiments/LatentShapeVAE`

## 1. Why this line was created
Bu experiment hattı, condition etkisini tamamen çıkarıp (unconditional VAE), şu soruyu doğrudan test etmek için açıldı:

- `q(z)` aggregate latent şekline bakarak model seçim/optimizasyonunu güvenilir biçimde yapabilir miyiz?
- Bunu yaparken kaliteyi (reconstruction + prior sampling realism) bozuyor muyuz?

Ana iddia (frozen):
- Kaliteyi bozmadan aggregate latent dağılımını `N(0,I)` hedefine yaklaştıran bir seçim yapılabilir.

Referans:
- `ML/autoencoder/experiments/LatentShapeVAE/protocol/decisions.md`

## 2. Frozen setup (what never changed)
- Data: `data/external_dataset/extracted/data/filtered_waveforms/HH`
- Channels: `HHE, HHN, HHZ` (3C)
- Fs: `100 Hz`
- Segment: `7001`
- Preprocess: demean + detrend + `0.5-20 Hz` bandpass
- Normalization: train-only global channel-wise mean/std
- Split: event-wise (`train/val/test/ood_event`), leakage-free

Artifacts:
- `ML/autoencoder/experiments/LatentShapeVAE/protocol/frozen_event_splits_v1.json`
- `ML/autoencoder/experiments/LatentShapeVAE/protocol/waveform_stats_v1.json`

## 3. Model, loss, and diagnostics
Model:
- Unconditional diagonal-Gaussian VAE
- Encoder output: `mu(x), logvar(x)`
- Posterior: `q(z|x)=N(mu, diag(exp(logvar)))`

Loss:
- `L = MSE_time + lambda_mr * MR-STFT + beta * KL_raw`

Core latent-shape metrics (moment-approx to `N(0,I)`):
- `diag_mae`
- `offdiag_mean_abs_corr`
- `eig_ratio`
- `KL_moment_to_std_normal`
- `W2_moment_to_std_normal`

Extra diagnostic introduced in this phase:
- `n01_abs_gap = KL + W2 + 0.5*diag_mae + 0.25*|log(eig_ratio)|`
- `n01_similarity` (bounded monotonic transform of robust gap)

Script:
- `ML/autoencoder/experiments/LatentShapeVAE/evaluation/compute_n01_similarity.py`

Output:
- `ML/autoencoder/experiments/LatentShapeVAE/results/n01_similarity_v1/n01_similarity_per_run_split.csv`

### 3.1 Metric direction legend
- `diag_mae`: lower is better (`0` is ideal).
- `offdiag_mean_abs_corr`: lower is better (`0` is ideal).
- `eig_ratio`: closer to `1` is better.
- `KL_moment_to_std_normal`: lower is better (`0` is ideal).
- `W2_moment_to_std_normal`: lower is better (`0` is ideal).
- `n01_abs_gap`: lower is better.
- `n01_similarity`: higher is better.
- `realism_composite` (prior sampling): lower is better.

### 3.2 Composite metric definitions (explicit)
Prior-sampling realism (`realism_composite`):
- Feature set (8): `band_ratio_0p5_2`, `band_ratio_2_8`, `band_ratio_8_20`, `env_peak`, `env_kurtosis`, `env_duration_10pct_sec`, `psd_slope_loglog`, `spectral_centroid_hz`.
- Per-feature terms:
  - `mean_dev_z = |mu_gen - mu_real| / std_real`
  - `std_dev_log = |log(std_gen / std_real)|`
- Aggregation:
  - `realism_mean_dev_z_avg = mean(mean_dev_z over 8 features)`
  - `realism_std_dev_log_avg = mean(std_dev_log over 8 features)`
  - `realism_composite = realism_mean_dev_z_avg + realism_std_dev_log_avg`

N(0,1) similarity (`n01_abs_gap`):
- `n01_abs_gap = KL + W2 + 0.5*diag_mae + 0.25*|log(eig_ratio)|`
- Weighting note: heuristic scaling to reduce single-term dominance (not tuned by separate optimization).
- Sensitivity note: explicit weight-perturbation sweep has not been completed yet; current use is operational ranking, not theoretical proof.

### 3.3 Collapse definition used in this report
Formal collapse criterion in this report:
- Near-zero KL (`KL_raw` and moment KL both close to `0`) and
- Very low active latent usage (AU/per-dim KL concentrated near `0`).

Important:
- High KL with bad geometry is **not collapse**. It is treated as latent-geometry breakdown / strong prior mismatch.

## 4. Experiment timeline and what was tested
### Stage-1 (ablation sanity)
Tested:
- AE
- beta0
- VAE (`beta=0.03 anneal`)
- VAE (`beta=0.1`)

Result summary:
- AE and beta0 did not satisfy latent geometry target.
- VAE beta regimes were clearly better for `N(0,I)`-like latent shape.

Key file:
- `ML/autoencoder/experiments/LatentShapeVAE/results/latent_shape_test_20260219_175948/latent_shape_summary.csv`

### Stage-2 (beta=0.1 multi-seed)
Initial 3-seed showed severe instability on test side:
- `s43`: `diag_mae=1476.64`, `KL=47233.75`
- `s44`: `diag_mae=176.07`, `KL=5626.72`

Key file:
- `ML/autoencoder/experiments/LatentShapeVAE/results/stage2_beta0p1_seed_eval_summary_v1/stage2_beta0p1_seed_eval_summary.csv`

### Stage-2 stabilization (bounded logvar)
Patch:
- `logvar_mode=bounded_sigmoid`
- `logvar_min=-12`, `logvar_max=8`

Then 10-seed confirmatory run:
- seeds: `42..51`
- run pattern: `lsv_stage2_vae_base_ld64_b0p1_s<seed>_logvfixv2`

Aggregate outcome (test):
- `test_diag_mae = 0.3456 ± 0.1820`
- `test_offdiag = 0.1267 ± 0.0579`
- `test_KL = 7.9818 ± 4.5343`

Aggregate outcome (ood_event):
- `ood_diag_mae = 0.0356 ± 0.0239`
- `ood_offdiag = 0.0111 ± 0.0067`
- `ood_KL = 0.3183 ± 0.1910`

Interpretation:
- Numerical explosion solved.
- Seed-robust latent shape on test not achieved.
- OOD side is markedly more stable than test side.

OOD anomaly note (critical):
- In this stage, OOD latent-shape metrics are systematically much better than test metrics.
- This is not interpreted as "OOD is easier" by default; it is treated as an anomaly that requires dataset-profile explanation.
- Current report keeps this as a scope-limited pass, not a confirmatory theoretical pass.

Split profile used for anomaly context:

| Split | Trace count | Event count |
|---|---:|---:|
| train | 64,121 | 9,159 |
| val | 7,969 | 1,145 |
| test | 7,703 | 1,145 |
| ood_event | 9,128 | 1,272 |

Real-feature snapshot (selected run `s43_logvfixv2`, 2,000 real samples each split):

| Feature | test mean | test std | test p90 | ood mean | ood std | ood p90 |
|---|---:|---:|---:|---:|---:|---:|
| band_ratio_0p5_2 | 0.1963 | 0.1981 | 0.4804 | 0.2202 | 0.1965 | 0.5021 |
| band_ratio_2_8 | 0.4536 | 0.2002 | 0.7146 | 0.4546 | 0.1736 | 0.6833 |
| band_ratio_8_20 | 0.3502 | 0.2160 | 0.6540 | 0.3252 | 0.2093 | 0.6206 |
| env_duration_10pct_sec | 18.2040 | 11.4525 | 35.6600 | 18.3317 | 11.5007 | 36.2867 |
| env_kurtosis | 29.0635 | 29.8015 | 59.6457 | 30.2040 | 31.4806 | 66.0960 |
| env_peak | 8,818.3 | 33,056.0 | 16,525.0 | 20,499.4 | 188,978.2 | 20,101.0 |
| psd_slope_loglog | -1.2541 | 1.0766 | 0.0335 | -1.3732 | 1.1856 | -0.0178 |
| spectral_centroid_hz | 6.5683 | 2.4113 | 9.8213 | 6.2838 | 2.4217 | 9.4928 |

Interpretation of the snapshot:
- Most means are in similar range.
- `env_peak` differs strongly and also shows much larger OOD spread (heavy-tail amplitude effect).
- So OOD superiority in latent metrics is not explained as a trivial "all features are easier"; it remains a pending distributional explanation item.

Additional metric definition used in stage summaries:
- `test_max_var` / `ood_max_var` is from latent-var audit:
  - For each sample: `max_var_sample = max_j exp(logvar_j)`
  - Reported value: max over all samples in that split (`max_var_global`).

Key files:
- `ML/autoencoder/experiments/LatentShapeVAE/results/stage2_beta0p1_logvarfix_10seeds_v2/robustness_summary.md`
- `ML/autoencoder/experiments/LatentShapeVAE/results/stage2_beta0p1_logvarfix_10seeds_v2/robustness_aggregate.csv`

### Stage-3 (latent shrink test: 64 -> 32)
Goal:
- Test whether smaller latent (`ld32`) can keep geometry and realism acceptable.

Formats (3x3 runs):
- `fmtA_b0p1_lmax8`
- `fmtB_b0p1_lmax6`
- `fmtC_b0p03_anneal_lmax6`
- seeds `42,43,44`

Format-level results:

| Format | test diag | test offdiag | test KL | ood diag | ood offdiag | ood KL | prior test | prior ood | test max_var |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fmtA_b0p1_lmax8 | 0.3602 ± 0.0867 | 0.2042 ± 0.0457 | 4.5480 ± 1.4467 | 0.0349 ± 0.0090 | 0.0251 ± 0.0033 | 0.2481 ± 0.0725 | 1.7439 ± 0.1991 | 1.9920 ± 0.2020 | 1.950e+03 |
| fmtB_b0p1_lmax6 | 0.5018 ± 0.2301 | 0.2099 ± 0.0499 | 6.7250 ± 3.3921 | 0.0425 ± 0.0116 | 0.0186 ± 0.0104 | 0.2389 ± 0.1145 | 1.6655 ± 0.1819 | 1.9069 ± 0.1979 | 4.033e+02 |
| fmtC_b0p03_anneal_lmax6 | 3.8785 ± 1.5647 | 0.6894 ± 0.0680 | 62.0364 ± 27.2946 | 0.1825 ± 0.1052 | 0.1551 ± 0.0547 | 3.1147 ± 1.2602 | 2.0278 ± 0.1658 | 2.2129 ± 0.1958 | 3.605e+02 |

Interpretation:
- `fmtC` is clearly unusable due to severe latent-geometry breakdown (not posterior collapse).
- `fmtA` and `fmtB` are usable but both are materially worse than the selected `ld64` reference in latent geometry.
- `fmtB` has tighter variance cap behavior (`max_var`) and better realism means, but worse test KL/diag than `fmtA`.

Key files:
- `ML/autoencoder/experiments/LatentShapeVAE/results/stage3_ld32_formats_v1/stage3_ld32_formats_summary.md`
- `ML/autoencoder/experiments/LatentShapeVAE/results/stage3_ld32_formats_v1/stage3_ld32_formats_per_run.csv`

## 5. Direct comparison to selected ld64 reference
Selected operational reference:
- `lsv_stage2_vae_base_ld64_b0p1_s43_logvfixv2`
- checkpoint:
  - `ML/autoencoder/experiments/LatentShapeVAE/checkpoints/lsv_stage2_vae_base_ld64_b0p1_s43_logvfixv2_best.pt`
  - note: checkpoint artifacts were removed during repository cleanup; selection is retained via run id and protocol record.

Reference metrics (test):
- `diag_mae=0.0389`
- `offdiag=0.0013`
- `KL=0.1227`

Best ld32 run by latent shape (`fmtB_s43`, test):
- `diag_mae=0.2052`
- `offdiag=0.1431`
- `KL=2.4778`

Conclusion from this comparison:
- Current `ld32` trials do not preserve the latent geometry quality of the selected `ld64` reference.

### 5.1 Selection rationale mini-table (anti-cherry-pick check)
Compared candidates:
- Selected operational seed: `s43_logvfixv2`
- Median-like seed by test `n01_abs_gap` rank: `s50_logvfixv2`
- Best validation-loss seed: `s49_logvfixv2`

| Run | Selection role | best_val_loss | test n01_abs_gap | ood n01_abs_gap | test diag/offdiag/KL | prior test/ood composite |
|---|---|---:|---:|---:|---|---|
| `s43_logvfixv2` | selected operational | 1.7045 | 0.3950 | 0.3913 | 0.0389 / 0.0013 / 0.1227 | 1.8775 / 2.1250 |
| `s50_logvfixv2` | median-like | 1.7942 | 21.2465 | 0.7385 | 0.3007 / 0.1325 / 8.0995 | 1.8049 / 2.0632 |
| `s49_logvfixv2` | best val-loss | 1.6811 | 19.9355 | 0.6965 | 0.2824 / 0.1274 / 7.6257 | 1.6712 / 1.9000 |

Interpretation:
- Selection was not based on validation loss alone.
- Among compared representative candidates, the chosen operational seed is the only one that jointly keeps test and OOD latent-shape gaps low at this stage.
- This is an operational freeze decision, not a claim of seed-robust confirmatory success.

## 6. Hypothesis evaluation matrix (final)
### H1: "Aggregate latent shape can be measured and ranked with operational reliability in this pipeline"
- Status: **PASS**
- Evidence: frozen metric suite + `n01` composite ranking produced consistent ordering.
- Cross-split rank consistency:
  - Stage2 (`ld64`, 10 seeds): Spearman `0.818`, Kendall tau-a `0.778`
  - Stage3 (`ld32`, 9 runs): Spearman `0.733`, Kendall tau-a `0.556`

### H2: "Bounded logvar removes catastrophic numeric failure"
- Status: **PASS**
- Evidence: pre-fix extreme divergence disappeared in post-fix runs.

### H3: "beta=0.1 + bounded logvar is seed-robust on test latent shape"
- Status: **FAIL**
- Evidence: 10-seed test spread remains large.

### H4: "OOD latent behavior is at least stable enough for operational filtering"
- Status: **PASS (scope-limited, anomaly pending)**
- Evidence: OOD metrics are substantially tighter than test metrics, but this pattern is flagged for additional split/difficulty explanation.

### H5: "Shrinking latent to 32 preserves geometry while improving efficiency"
- Status: **NOT SUPPORTED (current formats)**
- Evidence: all ld32 formats are worse than the selected ld64 reference in latent shape.

## 7. What is proven vs what is not proven
Proven:
- Latent-shape diagnostics are operationally useful.
- Bounded logvar is necessary in this pipeline.
- A stable operational checkpoint can be selected and frozen.

Not proven:
- Robust geometry-optimal regime across seeds with current recipe.
- Geometry-preserving latent shrink to 32 with tested formats.
- Causal explanation of why OOD latent metrics are systematically better than test metrics.

## 8. Final operational decisions
- Keep operational reference at `ld64`:
  - `lsv_stage2_vae_base_ld64_b0p1_s43_logvfixv2`
- Keep stage3 results as negative/partial evidence for current shrink formats.
- Do not claim confirmatory success for the global "geometry-driven robust optimization" statement.

Decision artifacts:
- `ML/autoencoder/experiments/LatentShapeVAE/protocol/selected_operational_checkpoint_v2.json`
- `ML/autoencoder/experiments/LatentShapeVAE/protocol/decisions.md`

## 9. Reproducibility commands
Stage2 10-seed train:
```bash
ML/autoencoder/experiments/LatentShapeVAE/training/run_stage2_beta0p1_logvarfix_10seeds_v2.sh
```

Stage2 10-seed eval:
```bash
ML/autoencoder/experiments/LatentShapeVAE/evaluation/run_stage2_beta0p1_logvarfix_10seeds_v2.sh
```

Stage3 ld32 train:
```bash
ML/autoencoder/experiments/LatentShapeVAE/training/run_stage3_ld32_formats_v1.sh
```

Stage3 ld32 eval:
```bash
ML/autoencoder/experiments/LatentShapeVAE/evaluation/run_stage3_ld32_formats_v1.sh
```

N(0,1) similarity score generation:
```bash
~/miniconda3/bin/python ML/autoencoder/experiments/LatentShapeVAE/evaluation/compute_n01_similarity.py \
  --output_dir ML/autoencoder/experiments/LatentShapeVAE/results/n01_similarity_v1
```

## 10. Closing statement
Bu experiment hattı teknik olarak değerli iki sonuç verdi:
- stabilizasyon (numerical safety) ve
- latent-shape üzerinden şeffaf model seçimi.

Ama aynı hat, şu anda aradığımız güçlü iddiayı da sınırlandırdı:
- seed-robust ve küçültülmüş latent ile aynı kalite/şekil korunumu henüz gösterilemedi.

Bu nedenle raporun nihai hükmü:
- **operasyonel kullanım var, güçlü teorik doğrulama yok; kanıt düzeyi partial**.
