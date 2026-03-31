# DDPMvsDDIM Current State Report (2026-03-29)

Bu belge, `DDPMvsDDIM` kutusunun 2026-03-29 itibariyle resmi durumunu
toplar. Amac, hangi artefaktin resmi oldugunu, hangi ciktinin smoke/stale
oldugunu ve ilk `DDPM vs DDIM` karsilastirmasinin ne gosterdigini tek yerde
toplamaktir.

## 1. Scope

Bu rapor yalnizca su frozen hatta aittir:

- Stage-1 backbone: `LegacyCondBaselineCVAE`
- split: hybrid magnitude-stratified `event-wise`
- latent target: `mu`
- diffusion objective: `epsilon prediction`
- denoiser: `ResMLPDenoiser`
- schedule: `cosine`
- train timesteps: `200`
- comparison:
  - `DDPM-200`
  - `DDIM-50`
- metrics:
  - `spec_corr`
  - `LSD`
  - `MR-LSD`

## 2. Official Artefacts

### 2.1 Frozen protocol

- `ML/autoencoder/experiments/DDPMvsDDIM/protocol/02_frozen_experiment_spec_v1.md`
- `ML/autoencoder/experiments/DDPMvsDDIM/protocol/eventwise_split_v1.json`
- `ML/autoencoder/experiments/DDPMvsDDIM/protocol/eventwise_split_summary_v1.json`

### 2.2 Stage-1

- best checkpoint:
  - `ML/autoencoder/experiments/DDPMvsDDIM/checkpoints/stage1_eventwise_v1_best.pt`
- latest checkpoint:
  - `ML/autoencoder/experiments/DDPMvsDDIM/checkpoints/stage1_eventwise_v1_latest.pt`
- training history:
  - `ML/autoencoder/experiments/DDPMvsDDIM/runs/stage1/stage1_eventwise_v1/metrics/history.csv`
- log:
  - `ML/autoencoder/experiments/DDPMvsDDIM/logs/stage1/stage1_eventwise_v1_20260327_223445.log`
- sanity gate:
  - `ML/autoencoder/experiments/DDPMvsDDIM/results/stage1_sanity_gate/summary.json`

### 2.3 Latent cache

- train cache:
  - `ML/autoencoder/experiments/DDPMvsDDIM/data_cache/latent_cache_eventwise_v1/train_latent_cache.pt`
- val cache:
  - `ML/autoencoder/experiments/DDPMvsDDIM/data_cache/latent_cache_eventwise_v1/val_latent_cache.pt`
- test cache:
  - `ML/autoencoder/experiments/DDPMvsDDIM/data_cache/latent_cache_eventwise_v1/test_latent_cache.pt`
- latent stats:
  - `ML/autoencoder/experiments/DDPMvsDDIM/data_cache/latent_cache_eventwise_v1/latent_stats.pt`
- latent stats summary:
  - `ML/autoencoder/experiments/DDPMvsDDIM/data_cache/latent_cache_eventwise_v1/latent_stats_summary.json`
- cache log:
  - `ML/autoencoder/experiments/DDPMvsDDIM/logs/cache/latent_cache_eventwise_v1_20260329_141702.log`

### 2.4 Diffusion

- best checkpoint:
  - `ML/autoencoder/experiments/DDPMvsDDIM/runs/diffusion/diffusion_eventwise_v1/checkpoints/best.pt`
- latest checkpoint:
  - `ML/autoencoder/experiments/DDPMvsDDIM/runs/diffusion/diffusion_eventwise_v1/checkpoints/latest.pt`
- training history:
  - `ML/autoencoder/experiments/DDPMvsDDIM/runs/diffusion/diffusion_eventwise_v1/metrics/history.json`
- log:
  - `ML/autoencoder/experiments/DDPMvsDDIM/logs/diffusion/diffusion_eventwise_v1_20260329_172305.log`

### 2.5 Evaluation

- full test metrics-only run:
  - `ML/autoencoder/experiments/DDPMvsDDIM/results/sampler_comparison_eventwise_v1/summary.json`
  - `ML/autoencoder/experiments/DDPMvsDDIM/results/sampler_comparison_eventwise_v1/per_sample_metrics.csv`
  - `ML/autoencoder/experiments/DDPMvsDDIM/logs/evaluation/sampler_comparison_eventwise_v1_full_20260329_175258.log`
- 25-sample visual subset:
  - `ML/autoencoder/experiments/DDPMvsDDIM/results/sampler_comparison_eventwise_v1_subset25/summary.json`
  - `ML/autoencoder/experiments/DDPMvsDDIM/results/sampler_comparison_eventwise_v1_subset25/per_sample_metrics.csv`
  - `ML/autoencoder/experiments/DDPMvsDDIM/results/sampler_comparison_eventwise_v1_subset25/specs/`
  - `ML/autoencoder/experiments/DDPMvsDDIM/results/sampler_comparison_eventwise_v1_subset25/waveforms/`

## 3. Stale / Non-Official Artefacts

Asagidakiler resmi event-wise final artefakt degildir:

- eski smoke sampler summary:
  - `ML/autoencoder/experiments/DDPMvsDDIM/results/sampler_comparison/summary.json`
- eski root-level cache dosyalari:
  - `ML/autoencoder/experiments/DDPMvsDDIM/data_cache/train_latent_cache.pt`
  - `ML/autoencoder/experiments/DDPMvsDDIM/data_cache/val_latent_cache.pt`
  - `ML/autoencoder/experiments/DDPMvsDDIM/data_cache/test_latent_cache.pt`
  - `ML/autoencoder/experiments/DDPMvsDDIM/data_cache/latent_stats.pt`
- smoke gate:
  - `ML/autoencoder/experiments/DDPMvsDDIM/results/stage1_sanity_gate_smoke/summary.json`

Bu dosyalar ya smoke ya da gecis asamasi artefaktidir. Nihai yorum bu
dosyalara dayandirilmamalidir.

## 4. Frozen Split Summary

Split ozeti:

| Split | Samples | Events | Magnitude bins |
| --- | ---: | ---: | --- |
| train | 71265 | 10176 | `lt3=9675`, `3to4=441`, `4to5=58`, `ge5=2` |
| val | 8818 | 1271 | `lt3=1209`, `3to4=55`, `4to5=7` |
| test | 8838 | 1274 | `lt3=1210`, `3to4=56`, `4to5=8` |

Not:

- `M >= 5` event'leri frozen policy geregi tamamen `train` split'indedir.
- Bu karar, test/val tarafinda 1-2 event'e dayali kirilgan buyuk-event
  yorumu yapmamak icin alinmistir.

## 5. Stage-1 Summary

### 5.1 Training

- epochs: `100`
- batch size: `128`
- workers: `24`
- beta: `0.1`
- lr: `1e-4`

Training sonucu:

- first val loss: `69850.8044`
- last val loss: `22376.7432`
- best epoch: `99`
- best val loss: `22333.4342`
- last-10 average epoch time: `27.34 s`

### 5.2 Sanity gate

Gate sonucu:

- `mean_spec_corr = 0.9582`
- `mean_mu_norm = 9.3320`
- `mean_logvar = -1.5611`
- `per_dim_mu_std_mean = 0.7910`
- `automatic_gate_pass = true`

Yorum:

- Stage-1 backbone operasyonel olarak diffusion'a gecmeye uygun kabul edildi.
- Belirlenen gate esikleri asildi.

## 6. Latent Cache Summary

Train latent cache ozeti:

- shape: `(71265, 128)`
- `z_mean_abs_mean = 0.0501`
- `z_std_mean = 0.8287`
- `z_std_min = 0.6823`
- `z_std_max = 1.3078`
- condition embedding shape: `(71265, 64)`
- raw condition shape: `(71265, 4)`

Yorum:

- `mu` latent dagilimi normalize edilebilir ve diffusable gorunuyor.
- Per-dim spread sifira cokmus degil.

## 7. Diffusion Training Summary

Training setup:

- denoiser: `ResMLPDenoiser`
- hidden dim: `512`
- depth: `6`
- cond mode: `embedding_plus_raw`
- objective: `epsilon prediction`
- schedule: `cosine`
- timesteps: `200`
- epochs: `100`
- batch size: `2048`
- workers: `8`

Training sonucu:

- epochs: `100`
- best val loss: `0.441806`
- last val loss: `0.449745`
- last-10 average epoch time: `0.547 s`

Yorum:

- Egitim cok hizli ve stabil tamamlandi.
- Val loss duzgun azaldi; ancak bu tek basina sampler kalitesini garanti etmez.

## 8. Full Test Evaluation (8838 samples)

Bu run metrics-only modda yapildi:

- `save_plots = none`
- `plot_count_written = 0`

### 8.1 Mean metrics

| Method | spec_corr | LSD | MR-LSD |
| --- | ---: | ---: | ---: |
| Oracle mu | `0.9544` | `1.6058` | `1.4533` |
| DDPM | `0.8806` | `1.7623` | `1.6192` |
| DDIM | `0.8679` | `1.8094` | `1.6685` |

### 8.2 Pairwise DDPM vs DDIM

`DDPM`'in `DDIM`'den daha iyi oldugu ornek sayisi:

- spec_corr: `5105 / 8838`
- LSD: `4978 / 8838`
- MR-LSD: `4956 / 8838`

Gap ozeti:

- mean `spec_corr` gap (`DDPM - DDIM`): `+0.0127`
- mean `LSD` gap (`DDIM - DDPM`): `+0.0471`
- mean `MR-LSD` gap (`DDIM - DDPM`): `+0.0493`

### 8.3 Interpretation

Ana bulgu:

- `DDPM > DDIM`
- ancak fark buyuk degil
- daha buyuk fark `oracle mu` ile diffusion sampler'lari arasinda

Bu su anlama geliyor:

- sampler secimi etkili ama ana darbo gaz sampler degil
- ortak score model yeterince guclu degil
- mevcut kurulumda `DDPM` fidelity tarafinda hafif ustun, `DDIM` ise bu
  backbone ile yeterince guclu degil

## 9. Visual Subset Evaluation (25 samples)

Bu run:

- `max_samples = 25`
- `selection_mode = evenly_spaced`
- `save_plots = all`

Yani ilk 25 degil, test set boyunca yayilmis 25 temsilci ornek secildi.

### 9.1 Mean metrics

| Method | spec_corr | LSD | MR-LSD |
| --- | ---: | ---: | ---: |
| Oracle mu | `0.9520` | `1.8183` | `1.6703` |
| DDPM | `0.8508` | `2.0335` | `1.8939` |
| DDIM | `0.7970` | `2.0789` | `1.9375` |

### 9.2 Output counts

- spectrogram plot count: `25`
- waveform plot count: `25`

### 9.3 Interpretation

Full test bulgusu subsette de korunuyor:

- `Oracle > DDPM > DDIM`

Bu subset ayni zamanda qualitative gozlem icin resmi gorsel klasordur.

## 10. Oracle Mu Nedir

`Oracle mu` bir sampler degildir.

Anlamı:

- gercek ornek `x`, Stage-1 encoder'dan gecirilir
- `mu(x)` alinir
- decoder, bu dogru latent ile reconstruction yapar

Kisa form:

- `oracle mu = decode(mu(x), c)`

Bu nedenle:

- `oracle mu`, generation sonucu degil
- `DDPM` ve `DDIM` icin ust sinir / referans tavan gibi kullanilir

## 11. Current Conclusion

Mevcut frozen v1 sonucuna gore:

1. pipeline teknik olarak calisiyor
2. event-wise split ile tam bir `Stage-1 -> cache -> diffusion -> evaluation`
   zinciri tamamlandi
3. `DDPM`, `DDIM`'den daha iyi
4. ancak ikisi de `oracle mu` kalitesine yeterince yakin degil
5. dolayisiyla bir sonraki arastirma odağı sampler degil, ortak diffusion
   modelini guclendirmek olmali

## 12. Next Improvement Track

Bir sonraki fazda condition seti ve split sabit tutulup sadece diffusion tarafi
iyilestirilecektir.

Ilk adaylar:

1. daha guclu condition-aware denoiser
2. `v-prediction`
3. `min-SNR` loss weighting
4. gerekirse self-conditioning

Bu iyilestirme fazi icin izleme dosyasi:

- `ML/autoencoder/experiments/DDPMvsDDIM/docs/improvement_tracking.md`
