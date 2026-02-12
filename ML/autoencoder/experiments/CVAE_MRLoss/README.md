# CVAE_MRLoss Experiment

Bu deneyde model mimarisi ayni kalir (duz CVAE), fakat reconstruction hedefi tek MSE yerine
Multi-Resolution STFT tabanli composite loss ile zenginlestirilir.

## Hedef
- Kucuk pencere: onset/P dalgasi zamanlama hassasiyeti
- Buyuk pencere: S/coda frekans dokusu
- Orta pencere: genel denge

## Loss
Toplam:
`L = lambda_img * L_img + lambda_mr * L_mr + beta * KL`

- `L_img`: spectrogram-domain reconstruction (MSE)
- `L_mr`: multi-resolution STFT loss (waveform domain)
- `KL`: standart CVAE KL

## Teknik Not
- Mevcut pipeline STFT image uzerinde calistigi icin waveform reconstruction
  icin GT fazi kullanilan differentiable approximation uygulanir.
- Griffin-Lim training loop icine alinmaz.

## Klasor Yapisi
- `core/waveform_dataset.py`: mevcut dataset wrapper + waveform output
- `core/mr_loss_utils.py`: GT-faz reconstruction + MR-STFT loss
- `training/train_cvae_mrloss_external.py`: detayli loglu trainer
- `training/sweep_lambda_beta.py`: beta/lambda tarama launcher
- `checkpoints/`, `logs/`, `results/`

## Tek Kosu Detached Baslatma
```bash
setsid bash -lc '/home/gms/.pyenv/shims/python -u ML/autoencoder/experiments/CVAE_MRLoss/training/train_cvae_mrloss_external.py --run_name pilot_b0p1_l0p5 --epochs 20 --beta 0.1 --lambda_mr 0.5 --lambda_img 0.5 > ML/autoencoder/experiments/CVAE_MRLoss/logs/train/train_pilot_b0p1_l0p5.log 2>&1 < /dev/null & echo $! > ML/autoencoder/experiments/CVAE_MRLoss/logs/train/train_pilot_b0p1_l0p5.pid; disown' >/dev/null 2>&1
```

## Sweep (Runbook + Launch)
```bash
# Dry-run (sadece runbook)
/home/gms/.pyenv/shims/python ML/autoencoder/experiments/CVAE_MRLoss/training/sweep_lambda_beta.py

# Gercek launch
/home/gms/.pyenv/shims/python ML/autoencoder/experiments/CVAE_MRLoss/training/sweep_lambda_beta.py --launch
```
