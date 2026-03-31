# DDPMvsDDIM

Bu klasor, `LegacyCondBaseline` mimarisini izole edip daha sonra ayni kutu icinde `DDPM` ve `DDIM` latent diffusion denemelerini karsilastirmak icin acildi.

Ilk asama bilerek basit tutuldu:
- `LegacyCondBaseline` backbone'u lokal olarak kopyalandi.
- Ortak dataset kodu da lokal olarak kopyalandi.
- Bu kutu kendi klasoru disinda deney-kodu import etmez.
- Disaridan yalnizca veri dosyalari kullanilir.

## Mevcut Durum
- `core/stft_dataset.py`: `General/core/stft_dataset.py` tabanli lokal dataset kopyasi
- `core/model_legacy_cond_baseline.py`: `LegacyCondBaseline` backbone'unun lokal kopyasi
- `core/loss_utils.py`: lokal beta-CVAE loss
- `training/train_legacy_cond_baseline_external.py`: izole egitim entrypoint'i
- `evaluation/smoke_test_isolation.py`: import/forward/sample smoke testi

## Current Official Status

Frozen `v1` zinciri tamamlandi:

1. event-wise split olusturuldu
2. Stage-1 backbone yeniden egitildi
3. Stage-1 sanity gate gecti
4. event-wise latent cache uretildi
5. latent diffusion egitildi
6. full test set uzerinde `DDPM-200 vs DDIM-50` karsilastirildi
7. ek olarak 25 ornek uzerinde gorsel subset uretildi

Ana rapor:

- `ML/autoencoder/experiments/DDPMvsDDIM/docs/current_state_2026-03-29.md`

Iyilestirme takip dosyasi:

- `ML/autoencoder/experiments/DDPMvsDDIM/docs/improvement_tracking.md`
- final sampler suite plan ve instrumentation notlari:
  - `ML/autoencoder/experiments/DDPMvsDDIM/docs/final_sampler_suite_v3_2026-03-31.md`
- final neutral comparison report:
  - `ML/autoencoder/experiments/DDPMvsDDIM/docs/final_ddpm_vs_ddim_suite_report_2026-03-31.md`

## Tasarim Kurali
Bu kutu su an icin baska deney kutularindan kod import etmez:
- `General`
- `LegacyCondBaseline`
- `FullCovariance`
- `NormalizingFlow`
- diger `experiments/*`

Data path'leri repo seviyesinde kalir:
- `data/external_dataset/...`
- `data/station_list_external_full.json`

## Baseline Smoke Test
```bash
/home/gms/miniconda3/bin/python3.12 -m ML.autoencoder.experiments.DDPMvsDDIM.evaluation.smoke_test_isolation
```

## Baseline Smoke Training
```bash
/home/gms/miniconda3/bin/python3.12 -m ML.autoencoder.experiments.DDPMvsDDIM.training.train_legacy_cond_baseline_external \
  --epochs 1 \
  --batch-size 4 \
  --num-workers 0 \
  --limit-train-batches 1 \
  --limit-val-batches 1 \
  --run-name smoke_local_copy
```

## Lokal Checkpoint ile Baseline Sanity
```bash
/home/gms/miniconda3/bin/python3.12 -m ML.autoencoder.experiments.DDPMvsDDIM.evaluation.run_baseline_sanity \
  --max-samples 3
```

Lokal checkpoint:
- `ML/autoencoder/experiments/DDPMvsDDIM/checkpoints/legacy_cond_baseline_best.pt`

## Sonraki Adim
Bu baseline lokal kopya smoke-test ile dogrulandiktan sonra ayni kutu icinde:
1. stage-1 backbone freeze karari,
2. latent cache uretimi,
3. DDPM egitimi,
4. DDIM sampling,
5. ayni metriklerle karsilastirma
eklencek.

## DDPM vs DDIM Akisi
1. Frozen event-wise split uret:
```bash
/home/gms/miniconda3/bin/python3.12 -m ML.autoencoder.experiments.DDPMvsDDIM.training.freeze_eventwise_split
```

2. Stage-1 backbone'u event-wise split ile egit:
```bash
/home/gms/miniconda3/bin/python3.12 -m ML.autoencoder.experiments.DDPMvsDDIM.training.train_stage1_eventwise \
  --run-name stage1_eventwise_v1
```

3. Stage-1 latent sanity gate:
```bash
/home/gms/miniconda3/bin/python3.12 -m ML.autoencoder.experiments.DDPMvsDDIM.evaluation.run_stage1_sanity_gate \
  --checkpoint ML/autoencoder/experiments/DDPMvsDDIM/checkpoints/stage1_eventwise_v1_best.pt
```

4. Latent cache olustur:
```bash
/home/gms/miniconda3/bin/python3.12 -m ML.autoencoder.experiments.DDPMvsDDIM.training.build_latent_cache \
  --checkpoint ML/autoencoder/experiments/DDPMvsDDIM/checkpoints/stage1_eventwise_v1_best.pt
```

5. Diffusion denoiser egit:
```bash
/home/gms/miniconda3/bin/python3.12 -m ML.autoencoder.experiments.DDPMvsDDIM.training.train_latent_diffusion \
  --run-name diffusion_eventwise_v1
```

6. Ayni checkpoint ile DDPM/DDIM sampler karsilastir:
```bash
/home/gms/miniconda3/bin/python3.12 -m ML.autoencoder.experiments.DDPMvsDDIM.evaluation.run_sampler_comparison \
  --diffusion-checkpoint ML/autoencoder/experiments/DDPMvsDDIM/runs/diffusion/diffusion_eventwise_v1/checkpoints/best.pt
```

## Instrumented Final Sampler Suite

`run_sampler_comparison.py` artik yalnizca kalite metriklerini degil, ayni
zamanda runtime ve kaynak kullanimini da kaydeder.

Yeni `summary.json` bloklari:

- `runtime`
  - `evaluation_wall_time_sec`
  - `evaluation_wall_time_min`
  - `samples_per_sec`
  - `avg_oracle_decode_time_ms`
  - `avg_ddpm_sampling_time_ms`
  - `avg_ddpm_total_time_ms`
  - `avg_ddim_sampling_time_ms`
  - `avg_ddim_total_time_ms`
- `resources`
  - `cpu_percent_avg`
  - `cpu_percent_peak`
  - `rss_mb_peak_process`
  - `gpu_util_percent_avg`
  - `gpu_util_percent_peak`
  - `gpu_memory_used_mb_peak_poll`
  - `torch_peak_allocated_mb`
  - `torch_peak_reserved_mb`

Final `v3` sampler suite launcher:

```bash
/home/gms/miniconda3/bin/python3.12 -m ML.autoencoder.experiments.DDPMvsDDIM.evaluation.run_v3_final_sampler_suite
```

Bu launcher su adimlari ardarda kosar:

- `DDIM-25`
- `DDIM-50`
- `DDIM-100`

Her kosu icin:

- ayni `v3` checkpoint
- ayni event-wise test cache
- ayni sampler seed policy
- metrics-only evaluation
- runtime/resource instrumentation

Toplu ciktular:

- `ML/autoencoder/experiments/DDPMvsDDIM/results/final_sampler_suite_v3/suite_summary.json`
- `ML/autoencoder/experiments/DDPMvsDDIM/results/final_sampler_suite_v3/suite_summary.md`

## Frozen Runtime Defaults
- Stage-1 VAE training:
  - `batch_size=128`
  - `num_workers=24`
- Latent cache extraction:
  - `batch_size=512`
  - `num_workers=16`
- Latent diffusion training:
  - `batch_size=2048`
  - `num_workers=8`

Bu sayilar mevcut makine profiline gore secildi:
- GPU: `RTX A5000 24GB`
- CPU threads: `64`
- RAM: `~1.0 TiB`

## Frozen Protocol Docs

Ana frozen karar belgesi:
- `ML/autoencoder/experiments/DDPMvsDDIM/protocol/02_frozen_experiment_spec_v1.md`

Acik madde kapanis durumu:
- `ML/autoencoder/experiments/DDPMvsDDIM/protocol/03_open_items_status.md`

## Official Results Snapshot

Full test (`8838` sample) mean metrics:

| Method | spec_corr | LSD | MR-LSD |
| --- | ---: | ---: | ---: |
| Oracle mu | `0.9544` | `1.6058` | `1.4533` |
| DDPM | `0.8806` | `1.7623` | `1.6192` |
| DDIM | `0.8679` | `1.8094` | `1.6685` |

Kisa yorum:

- `DDPM > DDIM`
- ancak ana kalite boslugu sampler seciminden cok ortak diffusion modelinin
  gucunden geliyor

Resmi sonuc dosyalari:

- full metrics-only:
  - `ML/autoencoder/experiments/DDPMvsDDIM/results/sampler_comparison_eventwise_v1/summary.json`
  - `ML/autoencoder/experiments/DDPMvsDDIM/results/sampler_comparison_eventwise_v1/per_sample_metrics.csv`
- 25-sample visual subset:
  - `ML/autoencoder/experiments/DDPMvsDDIM/results/sampler_comparison_eventwise_v1_subset25/summary.json`
  - `ML/autoencoder/experiments/DDPMvsDDIM/results/sampler_comparison_eventwise_v1_subset25/specs/`
  - `ML/autoencoder/experiments/DDPMvsDDIM/results/sampler_comparison_eventwise_v1_subset25/waveforms/`
