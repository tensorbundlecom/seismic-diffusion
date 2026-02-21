# Training Guide

## 1) Frozen protokol artefaktlarini olustur

```bash
~/miniconda3/bin/python ML/autoencoder/experiments/LatentShapeVAE/protocol/freeze_event_splits_v1.py
~/miniconda3/bin/python ML/autoencoder/experiments/LatentShapeVAE/protocol/freeze_waveform_stats_v1.py
```

## 2) Tek run egitimi

```bash
~/miniconda3/bin/python ML/autoencoder/experiments/LatentShapeVAE/training/train_single.py \
  --run_name lsv_vae_base_ld64_s42 \
  --ablation_mode vae \
  --backbone base \
  --latent_dim 64 \
  --beta 0.1 \
  --max_steps 12000 \
  --val_check_every_steps 1000
```

## 3) Uzun run komutu (terminal bagimsiz)

```bash
mkdir -p ML/autoencoder/experiments/LatentShapeVAE/logs
setsid bash -lc '
~/miniconda3/bin/python ML/autoencoder/experiments/LatentShapeVAE/training/train_single.py \
  --run_name lsv_vae_base_ld64_s42 \
  --ablation_mode vae --backbone base --latent_dim 64 --beta 0.1
' > ML/autoencoder/experiments/LatentShapeVAE/logs/lsv_vae_base_ld64_s42.launch.log 2>&1 < /dev/null &
```

Canli log:

```bash
tail -f ML/autoencoder/experiments/LatentShapeVAE/logs/lsv_vae_base_ld64_s42.log
```

## 4) Sirali stage-1 baseline kuyrugu

Bu script asagidaki runlari sirayla calistirir:
- `lsv_stage1_ae_base_ld64_s42`
- `lsv_stage1_beta0_base_ld64_s42`
- `lsv_stage1_vae_base_ld64_b0p03_anneal_s42`

```bash
setsid bash -lc 'ML/autoencoder/experiments/LatentShapeVAE/training/run_stage1_baselines_queue_v1.sh' \
  > ML/autoencoder/experiments/LatentShapeVAE/logs/stage1_baselines_queue_v1.launch.log 2>&1 < /dev/null &
```

Kuyruk logu:

```bash
tail -f ML/autoencoder/experiments/LatentShapeVAE/logs/stage1_baselines_queue_v1.launch.log
```

## 5) Stable rerun kuyrugu (NaN gorulen AE/Beta0 icin)

Bu kuyruk sadece stabilite duzeltmesi gereken runlari tekrar calistirir:
- `lsv_stage1_ae_base_ld64_s42_stablev1`
- `lsv_stage1_beta0_base_ld64_s42_stablev1`

```bash
setsid bash -lc 'ML/autoencoder/experiments/LatentShapeVAE/training/run_stage1_stable_rerun_queue_v1.sh' \
  > ML/autoencoder/experiments/LatentShapeVAE/logs/stage1_stable_rerun_queue_v1.launch.log 2>&1 < /dev/null &
```

```bash
tail -f ML/autoencoder/experiments/LatentShapeVAE/logs/stage1_stable_rerun_queue_v1.launch.log
```

## 6) Stage-2 (Secili beta=0.1) multi-seed kuyrugu

Bu kuyruk secili rejimi (`beta=0.1`) 3 seed ile dogrular:
- `lsv_stage2_vae_base_ld64_b0p1_s42`
- `lsv_stage2_vae_base_ld64_b0p1_s43`
- `lsv_stage2_vae_base_ld64_b0p1_s44`

```bash
setsid bash -lc 'ML/autoencoder/experiments/LatentShapeVAE/training/run_stage2_beta0p1_seeds_v1.sh' \
  > ML/autoencoder/experiments/LatentShapeVAE/logs/stage2_beta0p1_seeds_v1.launch.log 2>&1 < /dev/null &
```

```bash
tail -f ML/autoencoder/experiments/LatentShapeVAE/logs/stage2_beta0p1_seeds_v1.launch.log
```

## 7) Stage-2 logvar-fix rerun (s43/s44)

Nadir posterior-variance outlierlarini bastirmak icin bounded-logvar parametrizasyonu:
- `logvar_mode=bounded_sigmoid`
- `logvar_min=-12`
- `logvar_max=8`

```bash
setsid bash -lc 'ML/autoencoder/experiments/LatentShapeVAE/training/run_stage2_beta0p1_logvarfix_s43s44_v1.sh' \
  > ML/autoencoder/experiments/LatentShapeVAE/logs/stage2_beta0p1_logvarfix_s43s44_v1.launch.log 2>&1 < /dev/null &
```

```bash
tail -f ML/autoencoder/experiments/LatentShapeVAE/logs/stage2_beta0p1_logvarfix_s43s44_v1.launch.log
```

## 8) Stage-2 logvar-fix robustness (10 seed, v2)

Sabit recipe (`beta=0.1`, bounded logvar) ile 10 seed:
- `s42 ... s51`
- run adlari: `lsv_stage2_vae_base_ld64_b0p1_s<seed>_logvfixv2`

```bash
setsid bash -lc 'ML/autoencoder/experiments/LatentShapeVAE/training/run_stage2_beta0p1_logvarfix_10seeds_v2.sh' \
  > ML/autoencoder/experiments/LatentShapeVAE/logs/stage2_beta0p1_logvarfix_10seeds_v2.launch.log 2>&1 < /dev/null &
```

```bash
tail -f ML/autoencoder/experiments/LatentShapeVAE/logs/stage2_beta0p1_logvarfix_10seeds_v2.launch.log
```

## 9) Stage-3 latent=32 format sweep (v1)

3 format x 3 seed:
- `fmtA_b0p1_lmax8`
- `fmtB_b0p1_lmax6`
- `fmtC_b0p03_anneal_lmax6`
- seeds: `42,43,44`

```bash
setsid bash -lc 'ML/autoencoder/experiments/LatentShapeVAE/training/run_stage3_ld32_formats_v1.sh' \
  > ML/autoencoder/experiments/LatentShapeVAE/logs/stage3_ld32_formats_v1.launch.log 2>&1 < /dev/null &
```

```bash
tail -f ML/autoencoder/experiments/LatentShapeVAE/logs/stage3_ld32_formats_v1.launch.log
```
