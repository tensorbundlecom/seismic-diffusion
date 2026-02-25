# LegacyCondDiffusion

Isolated experiment box for latent diffusion on top of a legacy condition-embedding Stage-1 backbone.

## Scope

- All implementation is inside this folder.
- No code imports from other `experiments/*` folders.
- Data input is read from existing `data/...` assets.

## Frozen Design (v1)

- Stage-1: local legacy-condition CVAE backbone trained in this box.
- Stage-2: latent diffusion on cached latents.
- Conditioning:
  - Main: `w_cond + raw physical c`
  - Ablations: `w_cond-only`, `c-only`
- Denoiser:
  - Main: `ResMLP`
  - Ablation: `U-Net1D`
- Split policy: event-wise.
- Evaluation: multi-metric OOD scoring.

See:
- `protocol/decisions.md`
- `protocol/implementation_plan_v1.md`

## Directory Layout

- `core/`: dataset, Stage-1 model, diffusion models, utils
- `training/`: Stage-1 train, latent cache build, diffusion train, ablation launcher
- `evaluation/`: generation, OOD evaluation, metrics, visualization
- `configs/`: dependency-free JSON configs
- `protocol/`: decisions, plan, runbook, implementation log
- `checkpoints/`, `logs/`, `data_cache/`, `results/`, `visualizations/`: runtime artifacts

## Run Order

1. Train Stage-1:
   - `training/train_stage1_wbaseline.py`
2. Build latent cache:
   - `training/build_latent_cache.py`
3. Train diffusion:
   - `training/train_latent_diffusion.py`
4. Evaluate OOD:
   - `evaluation/evaluate_post_training_custom_ood.py`
5. Optional ablations:
   - `training/run_ablation_grid.py`

Exact detached commands are in:
- `protocol/runbook.md`

## Notes

- Config format is JSON to keep the box dependency-free (no PyYAML requirement).
- Runtime artifacts are ignored by local `.gitignore`; directory placeholders are kept via `.gitkeep`.
