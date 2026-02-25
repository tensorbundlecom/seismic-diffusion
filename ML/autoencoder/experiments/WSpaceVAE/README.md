# WSpaceVAE (True W-Latent)

This experiment implements a true StyleGAN-like `W` latent workflow for VAE training.

## Key Difference vs legacy `W*` folders

- Legacy `W*` folders use `W` as deterministic condition embedding.
- This folder uses:
  - stochastic base latent `u ~ q(u|x,c)` during training
  - mapping network `w = M(u)`
  - decoder conditioned on `w` (and condition feature path)

So `w` here is a latent variable derived from stochastic `u`, not a pure condition embedding.

## Training / Evaluation

- Train:
  - `training/train_true_wspace_vae_external.py`
- Evaluate on post-training custom OOD:
  - `evaluation/evaluate_true_wspace_post_training_custom_ood.py`
  - Use `--mode reconstruct` for apples-to-apples comparison with `LegacyCondBaseline` OOD reports.
  - Use `--mode sample` for pure conditional generation from `u ~ N(0, I)`.
- Compare against `LegacyCondBaseline`:
  - `evaluation/compare_true_wspace_vs_wbaseline.py`

## Config

- `configs/train_true_wspace_vae_external.json`
