# WBaseline CVAE Experiment

Bu deney, NormalizingW'deki W-space conditioning mantigini korur fakat normalizing flow kullanmaz.

## Mimari
- Mapping input: `magnitude_norm + location_norm(3) + station_embedding`
- Mapping output: `w` (`w_dim=64` varsayilan)
- `w` kullanimi:
  - posterior tahmini (`mu`, `logvar`) icin encoder tarafinda
  - decoder kosullandirmasinda
- Flow yok.

## Klasor Yapisi
- `core/model_wbaseline.py`: W-space conditioned base CVAE
- `core/loss_utils.py`: beta-CVAE loss
- `training/train_wbaseline_external.py`: external dataset egitimi
- `evaluation/evaluate_wbaseline_post_training_ood.py`: post-training OOD metrikleri
- `evaluation/evaluate_wbaseline_diverse_ood.py`: diverse OOD metrikleri
- `checkpoints/`: model agirliklari
- `logs/`: train/eval detached loglari
- `results/`: ozet metrik json dosyalari

## Detached Egitim Baslatma (terminal kopsa da surer)
```bash
setsid bash -lc '/home/gms/.pyenv/shims/python -u ML/autoencoder/experiments/WBaseline/training/train_wbaseline_external.py > ML/autoencoder/experiments/WBaseline/logs/train/train_wbaseline_external_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null & echo $! > ML/autoencoder/experiments/WBaseline/logs/train/train_wbaseline_external.pid; disown' >/dev/null 2>&1
```

## Not
- Magnitude model icinde normalize edilir: `(m - mag_min) / (mag_max - mag_min)`
- Location datasetten normalize gelir, modelde tekrar `[0,1]` clamp edilir.
