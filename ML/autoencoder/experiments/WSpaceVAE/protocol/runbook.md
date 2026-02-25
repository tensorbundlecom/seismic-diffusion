# WSpaceVAE Runbook

## Train (detached)

```bash
setsid bash -lc '/home/gms/.pyenv/shims/python -u ML/autoencoder/experiments/WSpaceVAE/training/train_true_wspace_vae_external.py \
  --config ML/autoencoder/experiments/WSpaceVAE/configs/train_true_wspace_vae_external.json \
  > ML/autoencoder/experiments/WSpaceVAE/logs/train/train_true_wspace_vae_external_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null & \
  echo $! > ML/autoencoder/experiments/WSpaceVAE/logs/train/train_true_wspace_vae_external.pid; disown' >/dev/null 2>&1
```

## Monitor

```bash
tail -f ML/autoencoder/experiments/WSpaceVAE/logs/train/train_true_wspace_vae_external_*.log
```

## Evaluate

```bash
/home/gms/.pyenv/shims/python ML/autoencoder/experiments/WSpaceVAE/evaluation/evaluate_true_wspace_post_training_custom_ood.py \
  --mode reconstruct
```

## Compare

```bash
/home/gms/.pyenv/shims/python ML/autoencoder/experiments/WSpaceVAE/evaluation/compare_true_wspace_vs_wbaseline.py
```

## Optional: Auto-run eval+compare after training

```bash
setsid bash -lc 'ML/autoencoder/experiments/WSpaceVAE/training/run_eval_after_training.sh \
  > ML/autoencoder/experiments/WSpaceVAE/logs/eval/orchestrator_true_wspace_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null & \
  echo $! > ML/autoencoder/experiments/WSpaceVAE/logs/eval/orchestrator_true_wspace.pid; disown' >/dev/null 2>&1
```
