# NonDiagonel Experiment

## Scope

This experiment isolates a single design change for the baseline family:

- Keep architecture and training protocol aligned with current baseline.
- Replace raw location conditioning (`lat_norm, lon_norm, depth_norm`) with physically grounded geometry features.
- Evaluate whether improved conditioning drives FullCov posteriors toward near-diagonal covariance without losing output quality.

Target condition vector (numeric part):

- `magnitude`
- `log1p(epicentral_distance_km)`
- `event_depth_km`
- `sin(azimuth_rad)`
- `cos(azimuth_rad)`

Station embedding is kept.

## Why

Using geometry directly is intended to reduce burden on the network to infer distance/azimuth from raw coordinates and station identity.

## Core Research Claim

Working claim for this experiment:

- If a Full-Covariance posterior still needs strong non-diagonal covariance terms after condition quality is improved, latent representation may still be suboptimal or under-constrained.
- Practical target is not strict zero, but *near-diagonal unless quality clearly benefits from non-diagonal structure*.

This is treated as an empirical hypothesis, not a theorem.  
Decision is made jointly with quality metrics (reconstruction + OOD), not from covariance shape alone.

## Phases

1. Build geometry-conditioned baseline (`GeoBaselineCVAE`).
2. Build geometry-conditioned full-covariance counterpart (`GeoFullCovCVAE`).
3. Train both on external HH dataset with shared protocol.
4. Compare against existing baseline/fullcov after training.

## Directory Layout

- `core/`
  - `condition_utils.py`: distance/azimuth feature construction and normalization helpers
  - `stft_dataset_geo.py`: dataset returning STFT + geometry condition + station index
  - `model_baseline_geo.py`: baseline CVAE with geometry condition path
  - `model_full_cov_geo.py`: full-covariance CVAE with geometry condition path
  - `loss_utils.py`: loss helpers
- `setup/`
  - `build_station_geometry_cache.py`: fetch station coordinates and persist cache
- `training/`
  - `train_baseline_geo_external.py`
  - `train_fullcov_geo_external.py`
- `checkpoints/`: model checkpoints
- `logs/`: training and setup logs
- `results/`: condition stats, run configs, histories

## Reproducibility

- Condition normalization stats are fit on **train split only**.
- The same saved stats are reused for validation/inference.
- Station mapping is fixed with `data/station_list_external_full.json`.

## Outputs

- Station coordinate cache:
  - `ML/autoencoder/experiments/NonDiagonel/results/station_coords_external.json`
- Condition stats:
  - `ML/autoencoder/experiments/NonDiagonel/results/condition_stats_<run>.json`
- Training artifacts:
  - `ML/autoencoder/experiments/NonDiagonel/checkpoints/*.pt`
  - `ML/autoencoder/experiments/NonDiagonel/results/*_history.csv`
  - `ML/autoencoder/experiments/NonDiagonel/results/*_config.json`

## Evaluation Workflow

Use the dedicated script:

- `ML/autoencoder/experiments/NonDiagonel/evaluation/evaluate_post_training_custom_geo_ood.py`
- `ML/autoencoder/experiments/NonDiagonel/evaluation/visualize_fullcov_correlation_heatmaps.py`
- `ML/autoencoder/experiments/NonDiagonel/evaluation/analyze_non_diagonal_decision_report.py`

Default target is the current custom post-training HH OOD set and it compares:

- `BaselineGeo` (diagonal posterior)
- `FullCovGeo` (full covariance posterior)

Each evaluation run is isolated under:

- `ML/autoencoder/experiments/NonDiagonel/results/evaluations/<eval_name>/`

Run outputs:

- `manifest.json`: inputs, checkpoints, seed, sample counts
- `metrics_aggregate.json`: mean metrics per model
- `metrics_per_sample.jsonl`: per-sample metrics
- `fullcov_posterior_offdiag_summary.json`: off-diagonal posterior-correlation summary
- `summary.md`: compact human-readable report
- `heatmaps/` (from visualization script):
  - `heatmap_mean_signed_corr.png`
  - `heatmap_mean_abs_corr.png`
  - `heatmap_strongest_offdiag_sample.png`
  - `heatmap_summary.json`
- `decision/` (from decision-analysis script):
  - `decision_report.md`
  - `decision_report.json`
  - `bootstrap_metric_summary.csv`
  - `station_breakdown.csv`
  - `event_breakdown.csv`
  - `offdiag_metric_correlations.csv`
- `causal_tests/README.md`:
  - running status and outcomes for sampling/controls causal checks

Logs:

- `ML/autoencoder/experiments/NonDiagonel/logs/<eval_name>.log`

## Causal Test Pack (Off-Diagonal Hypothesis)

Goal: separate true off-diagonal benefit from architecture/optimization side-effects.

### Test A: Sampling Decomposition (No Retraining)

Script:

- `ML/autoencoder/experiments/NonDiagonel/evaluation/evaluate_fullcov_sampling_ablation.py`

Compares three decode modes on same FullCov checkpoint:

- `mu`: deterministic (`z=mu`)
- `diag_sampled`: stochastic with diagonal-only covariance
- `full_sampled`: stochastic with full covariance

Outputs:

- `sampling_ablation/manifest.json`
- `sampling_ablation/metrics_aggregate_by_mode.json`
- `sampling_ablation/paired_bootstrap.json`
- `sampling_ablation/per_sample_sampling_metrics.jsonl`
- `sampling_ablation/sampling_ablation_report.md`

### Test B/C: Parameter-Matched Controls (Retraining)

Script:

- `ML/autoencoder/experiments/NonDiagonel/training/train_fullcov_geo_control_external.py`

Modes:

- `diag_only`: same FullCov architecture, but reparameterization and KL use only diagonal `L`.
- `offdiag_penalty`: full covariance kept, with extra loss term `lambda * ||L_offdiag||^2`.

These runs isolate:

- capacity effect (same encoder/heads as FullCov)
- explicit off-diagonal pressure effect (penalty sweep)

### Family Evaluation

Script:

- `ML/autoencoder/experiments/NonDiagonel/evaluation/evaluate_non_diagonal_model_family.py`

Evaluates any model list defined in:

- `ML/autoencoder/experiments/NonDiagonel/results/model_family_specs.json`

and writes a unified summary under `model_family_eval/`.

## Small Architecture Phase (Large vs Small)

Objective:

- reduce only backbone capacity
- keep latent/condition/data/protocol fixed
- compare diagonal vs full-cov behavior in both scales

Small backbone definition:

- encoder channels: `24 -> 48 -> 96 -> 192`
- decoder channels: `192 -> 96 -> 48 -> 24`
- latent dim: unchanged (`128`)
- condition path: unchanged

New core models:

- `ML/autoencoder/experiments/NonDiagonel/core/model_baseline_geo_small.py`
- `ML/autoencoder/experiments/NonDiagonel/core/model_full_cov_geo_small.py`

New training scripts:

- `ML/autoencoder/experiments/NonDiagonel/training/train_baseline_geo_small_external.py`
- `ML/autoencoder/experiments/NonDiagonel/training/train_fullcov_geo_small_external.py`

Checkpoint/storage safeguards in small scripts:

- atomic checkpoint write (`.tmp` then replace)
- `--save_optimizer` (default `0`) to reduce checkpoint size
- `--save_epoch_every` (default `0`) to disable periodic heavy snapshots unless explicitly needed

Model family evaluator supports additional types:

- `baseline_geo_small`
- `fullcov_geo_small`

## Latent Dependency Diagnostics (TC/MI + Basis Control)

Script:

- `ML/autoencoder/experiments/NonDiagonel/evaluation/evaluate_latent_dependency_tc_mi.py`

What it computes on a fixed OOD set:

- aggregated Gaussian TC (`TC_agg`)
- aggregated correlation off-diagonal summaries
- pairwise Gaussian MI summaries
- posterior TC summary (`q(z|x)`) per model
- basis-rotation stress test (random orthogonal rotations of the same covariance)
- optional combined `quality vs independence` Pareto table using existing OOD metrics JSON

Main outputs:

- `latent_dependency_summary.json`
- `rotation_stress_raw.json`
- `per_sample_posterior_tc.jsonl`
- `report.md`
- `plots/` (bars, boxplots, Pareto scatter, correlation heatmaps)

## Current Status (Reset for Epi+Depth Design)

- Condition design is now:
  - `magnitude`
  - `log1p(epicentral_distance_km)`
  - `depth_km`
  - `sin(azimuth)`, `cos(azimuth)`
  - `station_embedding`
- Previous run artifacts based on `log1p(hypocentral_distance_km)` were cleared.
- Ready-to-use normalization stats for the new design:
  - `ML/autoencoder/experiments/NonDiagonel/results/condition_stats_external_seed42.json`
