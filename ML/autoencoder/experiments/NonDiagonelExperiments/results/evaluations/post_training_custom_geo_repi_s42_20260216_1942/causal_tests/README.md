# Causal Test Pack Status (2026-02-16)

This folder tracks causal checks for the off-diagonal hypothesis.

## Test A: Sampling Decomposition (Completed)

Script:

- `ML/autoencoder/experiments/NonDiagonel/evaluation/evaluate_fullcov_sampling_ablation.py`

Output directory:

- `ML/autoencoder/experiments/NonDiagonel/results/evaluations/post_training_custom_geo_repi_s42_20260216_1942/sampling_ablation/`

Key result (paired bootstrap):

- `full_sampled` consistently beats `diag_sampled` on most metrics.
- both sampled modes are generally worse than deterministic `mu` mode in reconstruction-style evaluation.

Interpretation:

- off-diagonal structure helps *relative to diagonal-only stochastic sampling*.
- but deterministic `z=mu` remains strongest for reconstruction benchmarks.

## Test B/C: Parameter-Matched Control Trainings (Completed)

Script:

- `ML/autoencoder/experiments/NonDiagonel/training/train_fullcov_geo_control_external.py`

Suite session (completed):

- `screen`: `nd_ctrl_suite_20260216_2036`
- suite log: `ML/autoencoder/experiments/NonDiagonel/logs/nd_ctrl_suite_20260216_2036.out`

Runs in sequence:

1. `fullcov_geo_control_diag_only_external_s42_e30_20260216`
   - mode: `diag_only`
2. `fullcov_geo_control_offdiag_l1e4_external_s42_e30_20260216`
   - mode: `offdiag_penalty`, `offdiag_lambda=1e-4`
3. `fullcov_geo_control_offdiag_l5e4_external_s42_e30_20260216`
   - mode: `offdiag_penalty`, `offdiag_lambda=5e-4`

Each run writes:

- logs: `ML/autoencoder/experiments/NonDiagonel/logs/<run_name>.log`
- checkpoints: `ML/autoencoder/experiments/NonDiagonel/checkpoints/<run_name>_*.pt`
- history/config: `ML/autoencoder/experiments/NonDiagonel/results/<run_name>_*.csv|json`

Training outcomes (validation best):

- `diag_only`: `177.11`
- `offdiag_penalty (1e-4)`: `176.48`
- `offdiag_penalty (5e-4)`: `176.48`
- reference FullCov (100e): `176.56`

Observed caveat:

- with current penalty normalization, effective penalty contribution to total loss is tiny
  (`lambda * pen ~ 1e-7` scale), so stronger penalty scaling or reformulation is needed
  for a decisive off-diagonal suppression experiment.

## Phase-2 Penalty Calibration (Completed)

Updated strategy:

- keep penalty definition fixed
- increase `offdiag_lambda` by orders of magnitude
- monitor `pen_scaled = lambda * pen` explicitly in logs/history

Executed runs (30 epochs, same init checkpoint):

1. `offdiag_lambda=20`  (light-effective)
2. `offdiag_lambda=100` (medium-effective)
3. `offdiag_lambda=400` (strong-effective)

Validation-best summary:

- `l20`: `176.5075`
- `l100`: `176.5898`
- `l400`: `176.8018`

With references:

- reference FullCov (100e): `176.5579`
- diag-only control: `177.1075`

OOD off-diagonal trend (52 samples, encoder posterior corr):

- stronger lambda reduces off-diagonal stats monotonically
- example `mean_abs_corr_offdiag`:
  - ref: `0.05552`
  - l20: `0.05423`
  - l100: `0.05372`
  - l400: `0.05188`

Interpretation:

- mild suppression (`l20`) keeps quality near reference
- stronger suppression (`l100`, `l400`) starts to hurt validation loss
- this supports a trade-off: off-diagonal can be reduced, but aggressive suppression degrades fit

Success criterion:

- off-diagonal stats (`mean_abs`, `p95`, `energy_ratio`) reduce materially
- OOD metric degradation remains acceptable vs reference and diag-only controls

## Next Step

- update `ML/autoencoder/experiments/NonDiagonel/results/model_family_specs.json`
- run `ML/autoencoder/experiments/NonDiagonel/evaluation/evaluate_non_diagonal_model_family.py`
- compare:
  - BaselineGeo
  - FullCovGeo
  - DiagOnly control
  - Offdiag-penalty controls
