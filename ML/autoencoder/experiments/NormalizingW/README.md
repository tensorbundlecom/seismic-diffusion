# NormalizingW CVAE Experiment

Bu deney, condition bilgisini once bir `W-space` vektorune map edip bu vektoru modelin tum condition noktalarinda kullanir.

## Mimari
- Mapping input: `magnitude_norm + location_norm(3) + station_embedding`
- Mapping output: `w` (`w_dim=64` varsayilan)
- `w` kullanimi:
  - posterior tahmini (`mu`, `logvar`) icin encoder tarafinda
  - tum RealNVP coupling katmanlarinda
  - decoder kosullandirmasinda

## Klasor Yapisi
- `core/model_wflow.py`: W-space model
- `core/loss_utils.py`: loss fonksiyonu
- `training/train_wflow_external.py`: external dataset egitimi
- `evaluation/evaluate_wflow_post_training_ood.py`: post-training OOD metrikleri
- `evaluation/evaluate_wflow_diverse_ood.py`: diverse OOD metrikleri
- `checkpoints/`: model agirliklari
- `logs/`: train/eval nohup loglari
- `results/`: ozet metrik json dosyalari

## Egitim (Nohup)
```bash
nohup python3 ML/autoencoder/experiments/NormalizingW/training/train_wflow_external.py \
  > ML/autoencoder/experiments/NormalizingW/logs/train/train_wflow_external_$(date +%Y%m%d_%H%M%S).log 2>&1 &

echo $! > ML/autoencoder/experiments/NormalizingW/logs/train/train_wflow_external.pid
```

## Evaluation (Nohup)
```bash
nohup python3 ML/autoencoder/experiments/NormalizingW/evaluation/evaluate_wflow_post_training_ood.py \
  > ML/autoencoder/experiments/NormalizingW/logs/eval/eval_wflow_post_training_$(date +%Y%m%d_%H%M%S).log 2>&1 &

nohup python3 ML/autoencoder/experiments/NormalizingW/evaluation/evaluate_wflow_diverse_ood.py \
  > ML/autoencoder/experiments/NormalizingW/logs/eval/eval_wflow_diverse_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

## Not
- Magnitude model icinde normalize edilir: `(m - mag_min) / (mag_max - mag_min)`
- Location verisi dataset tarafinda normalize gelir, model icinde tekrar `[0,1]` clamp edilir.
