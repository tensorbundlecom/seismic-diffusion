# WFullCov CVAE Experiment

Bu deney, FullCovariance CVAE'yi W-space conditioning ile yeniden kurar.

## Mimari
- Mapping input: `magnitude_norm + location_norm(3) + station_embedding`
- Mapping output: `w` (`w_dim=64` varsayilan)
- `w` kullanimi:
  - posterior tahmini (`mu`, `L`) icin encoder tarafinda
  - decoder kosullandirmasinda
- Full covariance posterior korunur (`L` Cholesky factor)
- Flow yok.

## Klasor Yapisi
- `core/model_wfullcov.py`: W-space conditioned full covariance model
- `core/loss_utils.py`: full covariance loss
- `training/train_wfullcov_external.py`: external dataset egitimi
- `evaluation/evaluate_wfullcov_post_training_ood.py`: post-training OOD metrikleri
- `evaluation/evaluate_wfullcov_diverse_ood.py`: diverse OOD metrikleri
- `checkpoints/`: model agirliklari
- `logs/`: train/eval detached loglari
- `results/`: ozet metrik json dosyalari

## Detached Egitim Baslatma
```bash
setsid bash -lc '/home/gms/.pyenv/shims/python -u ML/autoencoder/experiments/WFullCov/training/train_wfullcov_external.py > ML/autoencoder/experiments/WFullCov/logs/train/train_wfullcov_external_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null & echo $! > ML/autoencoder/experiments/WFullCov/logs/train/train_wfullcov_external.pid; disown' >/dev/null 2>&1
```

## Not
- Magnitude model icinde normalize edilir: `(m - mag_min) / (mag_max - mag_min)`
- Location datasetten normalize gelir, modelde tekrar `[0,1]` clamp edilir.
