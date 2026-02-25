# LegacyCondDiffusion Implementation Log

## 2026-02-24

### Entry 001 - Planning Freeze
- Added `protocol/decisions.md` with accepted decisions from discussion:
  - Stage-1 local official training
  - `z_target=mu` primary
  - train-only latent normalization
  - hybrid conditioning (`w_cond+c`) + ablations
  - ResMLP main denoiser + U-Net1D ablation
- Added `protocol/implementation_plan_v1.md` with phased execution plan.

### Entry 002 - Box Bootstrap
- Created isolated experiment directories under `LegacyCondDiffusion`.
- Added `README.md` and `protocol/runbook.md`.
- Confirmed no code import has been added yet from other experiment folders.

### Entry 003 - Core Implementation
- Added isolated core modules:
  - `core/dataset_stft.py`
  - `core/model_stage1_wbaseline.py`
  - `core/model_diffusion_resmlp.py`
  - `core/model_diffusion_unet1d.py`
  - `core/diffusion_utils.py`
  - `core/loss_utils.py`
  - `core/split_utils.py`
  - `core/latent_cache_dataset.py`
  - `core/config_utils.py`

### Entry 004 - Training/Evaluation Pipeline
- Added training scripts:
  - `training/train_stage1_wbaseline.py`
  - `training/build_latent_cache.py`
  - `training/train_latent_diffusion.py`
  - `training/run_ablation_grid.py`
- Added evaluation scripts:
  - `evaluation/inference_utils.py`
  - `evaluation/metrics.py`
  - `evaluation/evaluate_post_training_custom_ood.py`
  - `evaluation/evaluate_diverse_ood.py`
  - `evaluation/generate_from_conditions.py`
  - `evaluation/visualize_outputs.py`

### Entry 005 - Implementation Fixes
- Added root-path bootstrap in executable scripts to prevent `ModuleNotFoundError: ML`.
- Updated runbook commands to use `/home/gms/.pyenv/shims/python`.
- Updated default OOD paths to current repository layout:
  - `data/ood_waveforms/post_training_custom/filtered`
  - `data/station_list_post_custom.json`
- Stabilized Stage-1 encoder flatten size using adaptive pooling:
  - fixed shape mismatch across varying STFT time bins.

### Entry 006 - Smoke Validation + Cleanup
- Smoke-validated:
  - Dataset load + Stage-1 forward pass.
  - Diffusion training on synthetic cache.
  - End-to-end small run: Stage-1 (1 epoch) -> cache -> diffusion (2 epochs) -> OOD eval.
  - Generation + visualization scripts.
- Removed smoke artifacts/checkpoints/caches/results to keep box clean.
- Added local `.gitignore` and `.gitkeep` placeholders for runtime directories.

### Entry 007 - Official Run Start
- Launched official Stage-1 training with:
  - config: `configs/stage1_default.json`
  - detached PID file: `logs/train/stage1.pid`
  - log file: `logs/train/stage1_20260224_152632.log`
- First-time frozen event-wise split generated:
  - `protocol/frozen_event_splits_v1.json`

### Entry 008 - Stage-2 Preparation Start
- Confirmed Stage-1 completed (100 epochs).
- Final best checkpoint:
  - `checkpoints/stage1_wbaseline_best.pt`
- Started latent cache extraction (detached):
  - PID file: `logs/cache/cache.pid`
  - log file: `logs/cache/cache_20260224_165840.log`

### Entry 009 - Cache Completion and Diffusion Launch
- Latent cache extraction completed successfully.
- Produced artifacts:
  - `data_cache/train_latent_cache.pt`
  - `data_cache/val_latent_cache.pt`
  - `data_cache/test_latent_cache.pt`
  - `data_cache/latent_stats.pt`
  - `results/latent_norm_stats.json`
- Started diffusion training (ResMLP, `w_plus_c`) in detached mode:
  - PID file: `logs/diffusion/diff_resmlp.pid`
  - log file: `logs/diffusion/diff_resmlp_20260224_170704.log`

### Entry 010 - Stage-2 Completion + QA
- Diffusion run `diff_resmlp_w_plus_c_v1` completed (100 epochs).
- Best val loss observed at epoch 92 (`~0.01299`).
- Ran post-training custom OOD evaluation (full 52 samples):
  - `results/post_training_custom_ood_metrics_diff_resmlp_w_plus_c_v1.json`
- Ran oracle-vs-generated diagnostic:
  - `results/oracle_vs_generated_metrics_full.json`
- Detected critical latent usage issue:
  - Stage-1 latent collapse (near-zero `mu` scale).
  - Decoder is largely insensitive to latent changes.
  - Generated outputs become near-deterministic for fixed conditions.
- Added formal QA note:
  - `protocol/qa_checks_stage2_resmlp_v1.md`
