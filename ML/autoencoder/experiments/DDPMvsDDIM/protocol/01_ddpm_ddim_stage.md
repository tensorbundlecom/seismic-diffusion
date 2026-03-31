# DDPM vs DDIM Stage

## Frozen Frame
- Stage-1 backbone: localized `LegacyCondBaselineCVAE`
- Latent target: encoder posterior mean `z_mu`
- Diffusion objective: epsilon prediction on normalized latent `z`
- Condition for denoiser: `cond_embedding + raw_condition`
- Sampler comparison: same trained denoiser, two reverse processes
  - DDPM
  - DDIM

## Why this frame
- `DDPM` ve `DDIM` ayri egitimli iki model degil; ayni epsilon denoiser uzerinde iki farkli sampling proseduru.
- Bu nedenle karsilastirma adil olmak icin:
  - ayni Stage-1,
  - ayni latent cache,
  - ayni denoiser checkpoint
  kullanir.

## Runtime Order
1. `training/build_latent_cache.py`
2. `training/train_latent_diffusion.py`
3. `evaluation/run_sampler_comparison.py`
