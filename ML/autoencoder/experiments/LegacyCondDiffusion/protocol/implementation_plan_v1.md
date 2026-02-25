# LegacyCondDiffusion - Implementation Plan (v1)

Date: 2026-02-24

## Goal

Generate conditional seismic outputs via latent diffusion on top of a legacy condition-embedding backbone, in a fully isolated experiment box.

## Scope Boundary

- All code under:
  - `ML/autoencoder/experiments/LegacyCondDiffusion/`
- No imports from:
  - `.../experiments/General/*`
  - `.../experiments/LegacyCondBaseline/*`
  - other experiment boxes
- Data access from existing `data/...` paths is allowed.

## Directory Layout

- `core/`
  - `dataset_stft.py`
  - `model_stage1_wbaseline.py`
  - `model_diffusion_resmlp.py`
  - `model_diffusion_unet1d.py`
  - `diffusion_utils.py`
  - `loss_utils.py`
- `training/`
  - `train_stage1_wbaseline.py`
  - `build_latent_cache.py`
  - `train_latent_diffusion.py`
  - `run_ablation_grid.py`
- `evaluation/`
  - `generate_from_conditions.py`
  - `evaluate_post_training_custom_ood.py`
  - `evaluate_diverse_ood.py` (phase-2)
  - `visualize_outputs.py`
- `configs/`
  - `stage1_default.json`
  - `diffusion_resmlp_default.json`
  - `diffusion_unet1d_default.json`
  - `ablation_grid_v1.json`
- `protocol/`
  - `decisions.md`
  - `implementation_plan_v1.md`
  - `runbook.md`
- `checkpoints/`, `logs/`, `results/`, `visualizations/`, `data_cache/`

## Phase Plan

### Phase 0 - Box bootstrap
- Create isolated package skeleton and entry scripts.
- Add runbook commands for detached execution and monitoring.

Exit criteria:
- `python .../train_stage1_wbaseline.py --help` works.
- `python .../train_latent_diffusion.py --help` works.

### Phase 1 - Stage-1 backbone (legacy condition local)
- Implement local model (`w_cond` mapping + station embedding + CVAE backbone).
- Train official Stage-1 in this folder.
- Save best/latest checkpoints.

Exit criteria:
- Stable train/val curves.
- Recon sanity outputs available.

### Phase 2 - Latent cache + normalization
- Build latent cache from Stage-1:
  - save `mu`, condition tensors, metadata keys.
- Compute train-only `z_mean/std`, normalize all splits.

Exit criteria:
- `results/latent_norm_stats.json` created.
- Cache reader returns consistent tensors and no NaNs.

### Phase 3 - Diffusion training (main: ResMLP)
- Train VE-style diffusion with Heun sampler settings.
- Conditioning mode: `w_cond + c` (primary).
- Log train/val losses and sampling sanity snapshots.

Exit criteria:
- No divergence.
- Decoded samples non-degenerate and metric pipeline runs.

### Phase 4 - Architecture ablation (1D U-Net)
- Same data/split/metrics, only denoiser changed.
- Compare ResMLP vs U-Net1D under equal budget.

Exit criteria:
- Completed comparative table in `results/`.

### Phase 5 - Conditioning ablation
- Run `w_cond-only`, `w_cond+c`, `c-only` with same denoiser.
- Report trade-offs and choose operational default.

Exit criteria:
- Ranking table with per-metric values and notes.

### Phase 6 - Evaluation
- Mandatory: post-training custom OOD.
- Optional phase-2: diverse OOD.
- Save:
  - machine-readable metrics JSON
  - markdown summary
  - selected visuals

Exit criteria:
- All target metrics computed successfully.
- Reproducible run command list documented.

## Experiment Matrix (minimum)

- Denoiser:
  - `resmlp`
  - `unet1d`
- Condition mode:
  - `w_plus_c` (default)
  - `w_only`
  - `c_only`
- Latent target:
  - `mu` (default)
  - `sampled_z` (single ablation)

## Logging and Long-Run Policy

- Use detached launch with PID + timestamped log files.
- Keep one log file per run.
- Periodically emit:
  - epoch, train/val loss
  - gradient norm (if enabled)
  - sample snapshot markers

## Risks and Controls

1. Condition shortcut risk (`c_only` dominates)
- Control with explicit conditioning ablation.

2. Latent scale instability
- Control with train-only per-dim normalization and clipping checks.

3. Overfitting to easier split
- Control with event-wise split and OOD evaluation.

4. Architecture mismatch
- Control with ResMLP main + U-Net1D ablation.
