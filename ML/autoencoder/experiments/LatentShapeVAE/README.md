# LatentShapeVAE

Bu deney kutusu, **conditional etkileri tamamen disarida birakarak** sadece su soruyu test eder:

> Kalite korunurken, latent uzayin aggregate sekli `N(0, I)` hedefine daha cok yaklasacak sekilde model secimi/optimizasyonu yapilabilir mi?

## Kapsam

- Model: **Unconditional VAE** (`x -> q(z|x) -> x_hat`)
- Domain: **Waveform** (3C, 100 Hz)
- Loss: `time-domain MSE + lambda_mr * MR-STFT + beta * KL`
- Split: event-wise IID + event-wise OOD (unseen events)
- Karar: kalite kapisi -> latent-shape skoru -> model boyutu tie-break

## Dizinler

- `protocol/`: frozen kararlar, split/stats uretim scriptleri
- `configs/`: model/loss/budget gridleri
- `core/`: dataset, model, loss kodlari
- `training/`: tek-run/grid egitim scriptleri
- `evaluation/`: latent-shape, ELBO ve prior-sampling analizleri
- `results/`, `logs/`, `checkpoints/`: deney artefaktlari

## Ilk adimlar

1. Event-wise split freeze et:
```bash
~/miniconda3/bin/python ML/autoencoder/experiments/LatentShapeVAE/protocol/freeze_event_splits_v1.py
```

2. Train-only global waveform stats freeze et:
```bash
~/miniconda3/bin/python ML/autoencoder/experiments/LatentShapeVAE/protocol/freeze_waveform_stats_v1.py
```

3. Tek model smoke egitimi:
```bash
~/miniconda3/bin/python ML/autoencoder/experiments/LatentShapeVAE/training/train_single.py \
  --run_name lsv_smoke_beta0p1_ld64_s42 \
  --latent_dim 64 \
  --beta 0.1 \
  --max_steps 2000 \
  --val_check_every_steps 500
```

4. Latent shape analizi:
```bash
~/miniconda3/bin/python ML/autoencoder/experiments/LatentShapeVAE/evaluation/analyze_latent_shape.py \
  --checkpoints ML/autoencoder/experiments/LatentShapeVAE/checkpoints/lsv_smoke_beta0p1_ld64_s42_best.pt \
  --split test
```

## Guncel Robustness Turu

- Egitim kuyrugu (10 seed, bounded logvar):
  - `training/run_stage2_beta0p1_logvarfix_10seeds_v2.sh`
- Toplu degerlendirme:
  - `evaluation/run_stage2_beta0p1_logvarfix_10seeds_v2.sh`
- Ayrintili adimlar:
  - `training/README.md`
  - `evaluation/README.md`

## Final Closure (2026-02-20)

- Final rapor:
  - `docs/final_closure_report_2026-02-20.md`
- Kapsamli rapor (stage3 dahil):
  - `docs/comprehensive_report_2026-02-21.md`
- Operasyonel referans checkpoint:
  - run id: `lsv_stage2_vae_base_ld64_b0p1_s43_logvfixv2`
  - not: agir checkpoint dosyalari repo temizlik adiminda kaldirildi.
- Frozen secim kaydi:
  - `protocol/selected_operational_checkpoint_v2.json`

## Sonraki Mini Tur (latent=32)

- Plan/karar:
  - `protocol/decisions.md` (`D15`)
- Train:
  - `training/run_stage3_ld32_formats_v1.sh`
- Eval:
  - `evaluation/run_stage3_ld32_formats_v1.sh`
