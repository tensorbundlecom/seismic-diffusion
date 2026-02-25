# LegacyCondDiffusion Runbook

## 1) Stage-1 Training

Detached launch:

```bash
setsid bash -lc '/home/gms/.pyenv/shims/python -u ML/autoencoder/experiments/LegacyCondDiffusion/training/train_stage1_wbaseline.py \
  --config ML/autoencoder/experiments/LegacyCondDiffusion/configs/stage1_default.json \
  > ML/autoencoder/experiments/LegacyCondDiffusion/logs/train/stage1_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null & \
  echo $! > ML/autoencoder/experiments/LegacyCondDiffusion/logs/train/stage1.pid; disown' >/dev/null 2>&1
```

Monitor:

```bash
tail -f ML/autoencoder/experiments/LegacyCondDiffusion/logs/train/stage1_*.log
```

## 2) Build Latent Cache

```bash
setsid bash -lc '/home/gms/.pyenv/shims/python -u ML/autoencoder/experiments/LegacyCondDiffusion/training/build_latent_cache.py \
  --config ML/autoencoder/experiments/LegacyCondDiffusion/configs/stage1_default.json \
  --checkpoint ML/autoencoder/experiments/LegacyCondDiffusion/checkpoints/stage1_wbaseline_best.pt \
  > ML/autoencoder/experiments/LegacyCondDiffusion/logs/cache/cache_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null & \
  echo $! > ML/autoencoder/experiments/LegacyCondDiffusion/logs/cache/cache.pid; disown' >/dev/null 2>&1
```

## 3) Diffusion Training (ResMLP)

```bash
setsid bash -lc '/home/gms/.pyenv/shims/python -u ML/autoencoder/experiments/LegacyCondDiffusion/training/train_latent_diffusion.py \
  --config ML/autoencoder/experiments/LegacyCondDiffusion/configs/diffusion_resmlp_default.json \
  > ML/autoencoder/experiments/LegacyCondDiffusion/logs/diffusion/diff_resmlp_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null & \
  echo $! > ML/autoencoder/experiments/LegacyCondDiffusion/logs/diffusion/diff_resmlp.pid; disown' >/dev/null 2>&1
```

## 4) Diffusion Training (U-Net1D ablation)

```bash
setsid bash -lc '/home/gms/.pyenv/shims/python -u ML/autoencoder/experiments/LegacyCondDiffusion/training/train_latent_diffusion.py \
  --config ML/autoencoder/experiments/LegacyCondDiffusion/configs/diffusion_unet1d_default.json \
  > ML/autoencoder/experiments/LegacyCondDiffusion/logs/diffusion/diff_unet1d_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null & \
  echo $! > ML/autoencoder/experiments/LegacyCondDiffusion/logs/diffusion/diff_unet1d.pid; disown' >/dev/null 2>&1
```

## 5) OOD Evaluation

```bash
/home/gms/.pyenv/shims/python ML/autoencoder/experiments/LegacyCondDiffusion/evaluation/evaluate_post_training_custom_ood.py \
  --config ML/autoencoder/experiments/LegacyCondDiffusion/configs/diffusion_resmlp_default.json
```
