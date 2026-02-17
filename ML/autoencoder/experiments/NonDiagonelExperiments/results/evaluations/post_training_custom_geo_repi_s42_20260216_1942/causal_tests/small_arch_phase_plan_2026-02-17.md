# Small Architecture Phase Plan (2026-02-17)

## Goal

Test whether FullCov-vs-Diagonal gap changes when only backbone capacity is reduced.

## Fixed Settings

- dataset/split: unchanged external HH setup
- latent dim: `128`
- condition dim: `64`
- numeric condition + station embedding path: unchanged
- optimizer/hyperparams: same defaults as current NonDiagonel external runs

## Changed Setting

- backbone channels only:
  - large: `32/64/128/256`
  - small: `24/48/96/192`

## Runs

1. `baseline_geo_small_external_s42_e30_20260217`
   - script: `train_baseline_geo_small_external.py`
2. `fullcov_geo_small_external_s42_e30_20260217`
   - script: `train_fullcov_geo_small_external.py`

## Evaluation

After training, compare four models in one table:

- `BaselineGeo` (large)
- `FullCovGeo_ref` (large)
- `BaselineGeo_small`
- `FullCovGeo_small`

using:

- `ML/autoencoder/experiments/NonDiagonel/evaluation/evaluate_non_diagonal_model_family.py`

## Execution Status (Live)

- launcher: detached `screen` session `nd_small_phase_20260217_0210`
- started: `2026-02-17 03:55 +03`
- monitor logs:
  - `ML/autoencoder/experiments/NonDiagonel/logs/baseline_geo_small_external_s42_e30_20260217.log`
  - `ML/autoencoder/experiments/NonDiagonel/logs/fullcov_geo_small_external_s42_e30_20260217.log`
- first confirmed progress (baseline-small):
  - `E001/030`: `val_loss=201.40`
  - `E002/030`: `val_loss=190.17`

## Overnight Queue (Added)

Rationale:

- keep backbone fixed at `small` and tighten latent bottleneck (`128 -> 64`)
- test whether off-diagonal reliance shrinks when representation pressure increases
- run sequentially after current small-pair completes (no GPU contention)

Queued runs:

1. `baseline_geo_small_ld64_external_s42_e30_20260217`
2. `fullcov_geo_small_ld64_external_s42_e30_20260217`

Queue monitor log:

- `ML/autoencoder/experiments/NonDiagonel/logs/nd_night_queue_20260217_0412.out`

## Error Recovery (ENOSPC)

Observed issue:

- overnight queue failed at `2026-02-17 04:59 +03` with `OSError: [Errno 28] No space left on device`
- `fullcov_geo_small_external_s42_e30_20260217` stopped at epoch 20 and left a truncated `epoch_020` checkpoint

Recovery actions:

- removed intermediate `*_epoch_*.pt` checkpoints under `NonDiagonel/checkpoints` (freed ~51 GB)
- removed zero-byte config artifact from failed ld64 run
- hardened small training scripts:
  - atomic checkpoint writes (`.tmp` + `os.replace`)
  - optional optimizer-state saving (`--save_optimizer`, default `0`)
  - optional periodic epoch checkpoints (`--save_epoch_every`, default `0`)
  - config-write failure no longer aborts training

Relaunched queue:

- session: `nd_recovery_20260217_1015`
- log: `ML/autoencoder/experiments/NonDiagonel/logs/nd_recovery_20260217_1015.out`
- run order:
  1. `fullcov_geo_small_external_s42_e30_20260217_r2`
  2. `baseline_geo_small_ld64_external_s42_e30_20260217_r2`
  3. `fullcov_geo_small_ld64_external_s42_e30_20260217_r2`

## OOD Family Comparison (Completed)

Model-spec file:

- `ML/autoencoder/experiments/NonDiagonel/results/model_family_specs_small_phase_20260217.json`

Evaluator run:

- script: `ML/autoencoder/experiments/NonDiagonel/evaluation/evaluate_non_diagonal_model_family.py`
- output:
  - `ML/autoencoder/experiments/NonDiagonel/results/evaluations/post_training_custom_geo_repi_s42_20260216_1942/model_family_eval_small_phase_20260217_1216/`
- log:
  - `ML/autoencoder/experiments/NonDiagonel/logs/nd_family_eval_smallphase_20260217_1216.out`

Key aggregates (Post-training custom HH OOD, 52 samples):

- best SSIM/S-Corr/EnvCorr among all compared models: `BaselineGeo_small_ld128`
- best LSD and DTW among compared models: `FullCovGeo_small_ld128`
- `small + ld64` setting increased off-diagonal correlation magnitude in FullCov:
  - mean abs offdiag corr: `0.0556 -> 0.1095` (`small_ld128 -> small_ld64`)

## Minimum Extra Checks (TC/MI + Basis Control) Completed

Script:

- `ML/autoencoder/experiments/NonDiagonel/evaluation/evaluate_latent_dependency_tc_mi.py`

Run output:

- `ML/autoencoder/experiments/NonDiagonel/results/evaluations/post_training_custom_geo_repi_s42_20260216_1942/latent_dependency_tc_mi_20260217_1228/`

Comprehensive write-up:

- `ML/autoencoder/experiments/NonDiagonel/results/evaluations/post_training_custom_geo_repi_s42_20260216_1942/comprehensive_hypothesis_report_2026-02-17.md`
