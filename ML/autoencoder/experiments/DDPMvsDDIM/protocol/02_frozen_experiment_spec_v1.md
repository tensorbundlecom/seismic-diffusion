# Frozen Experiment Spec v1

Bu belge, `DDPMvsDDIM` kutusundaki ilk ciddi karsilastirma kosusu icin
frozen kararlarin tek kaynagidir. Bu noktadan sonra yeni run baslatmadan once
kodun bu belgeye uyumlu hale getirilmesi gerekir.

Mevcut smoke artefact'lari yalnizca pipeline-proof olarak kabul edilir;
nihai deney raporuna girmez.

## 1. Amaç

Amaç, ayni latent diffusion denoiser uzerinde iki sampler'i karsilastirmaktir:

- `DDPM`
- `DDIM`

Bu nedenle ana karsilastirma, sampler farki disindaki butun degiskenleri
sabitler:

- ayni Stage-1 VAE backbone
- ayni event-wise split
- ayni latent cache
- ayni latent normalization
- ayni denoiser checkpoint
- ayni condition
- ayni baslangic latent noise

## 2. Frozen Split Spec

### 2.1 Grouping Key

- Split anahtari: `event_id`
- `event_id`, `stft_dataset.py` icindeki `_extract_event_id_from_filename()` ile
  uretilir.
- Ayni `event_id` icindeki tum trace/sample'lar ayni split'e gider.

### 2.2 Split Ratios

- `train = 0.80`
- `val = 0.10`
- `test = 0.10`

### 2.3 Magnitude Bins

Magnitude bazli event bins:

- `lt3`: `M < 3.0`
- `3to4`: `3.0 <= M < 4.0`
- `4to5`: `4.0 <= M < 5.0`
- `ge5`: `M >= 5.0`

Magnitude alani:

- ana alan: `ML`
- `ML` bos ise fallback: `xM`

### 2.4 Hybrid Event-Wise Policy

- `lt3`, `3to4`, `4to5` bin'leri icinde event-level shuffle yapilir.
- Shuffle seed: `42`
- Sonrasinda event bazinda `80/10/10` bolunur.

### 2.5 Rare Large-Event Rule

`ge5` bin'inde sadece 2 event oldugu icin bu event'ler split'e dagitilmaz.

Frozen kural:

- `M >= 5.0` olan iki event'in ikisi de `train` split'ine konur.

Gerekce:

- Test/val split'ine 1 adet event dusmesi istatistiksel olarak zayif olur.
- Buyuk event genellemesi ana test yerine ikinci asamada OOD ile olculecek.

### 2.6 Split Artefacts

Frozen dosyalar:

- split dosyasi:
  - `ML/autoencoder/experiments/DDPMvsDDIM/protocol/eventwise_split_v1.json`
- split ozet dosyasi:
  - `ML/autoencoder/experiments/DDPMvsDDIM/protocol/eventwise_split_summary_v1.json`

Split summary minimum su alanlari icermelidir:

- split basina `num_events`
- split basina `num_samples`
- split basina magnitude-bin dagilimi
- split basina event listesi

## 3. Frozen Stage-1 Backbone Spec

### 3.1 Backbone

- Model: `LegacyCondBaselineCVAE`
- Dosya:
  - `ML/autoencoder/experiments/DDPMvsDDIM/core/model_legacy_cond_baseline.py`

### 3.2 Initialization

- Event-wise setup icin Stage-1 backbone sifirdan egitilir.
- Eski random-split checkpoint'ten fine-tune yapilmaz.

### 3.3 Training Hyperparameters

- epoch: `100`
- lr: `1e-4`
- beta: `0.1` sabit
- batch size: `128`
- num workers: `24`
- save every: `10 epoch`
- train seed: `42`

### 3.4 Checkpoint Selection Rule

Ana secim:

- `best validation loss`

Ek kapı:

- `latent sanity gate` gecilmeden diffusion asamasina gecilmez.

### 3.5 Frozen Stage-1 Run Name

- `stage1_eventwise_v1`

## 4. Latent Target and Cache Spec

### 4.1 Latent Target

Frozen latent target:

- `mu`

Yani diffusion input/target'i encoder posterior mean'idir.

### 4.2 Neden `mu`

- deterministik cache
- sampler farkini daha temiz okumak
- posterior sampling gurultusunu sampler kiyasina karistirmamak

### 4.3 Cache Payload

Cache su alanlari saklar:

- `z_mu`
- `z_logvar`
- `cond_embedding`
- `raw_condition`
- `station_idx`
- `magnitude`
- `location`
- `meta`

### 4.4 Raw Condition Tanimi

`raw_condition`:

- `magnitude_norm`
- `latitude_norm`
- `longitude_norm`
- `depth_norm`

Toplam boyut:

- `4`

### 4.5 Cache Runtime Defaults

- batch size: `512`
- num workers: `16`

### 4.6 Frozen Cache Artefacts

- `ML/autoencoder/experiments/DDPMvsDDIM/data_cache/train_latent_cache.pt`
- `ML/autoencoder/experiments/DDPMvsDDIM/data_cache/val_latent_cache.pt`
- `ML/autoencoder/experiments/DDPMvsDDIM/data_cache/test_latent_cache.pt`
- `ML/autoencoder/experiments/DDPMvsDDIM/data_cache/latent_stats.pt`
- `ML/autoencoder/experiments/DDPMvsDDIM/data_cache/latent_stats_summary.json`

## 5. Latent Sanity Gate

Bu gate, Stage-1 checkpoint'in diffusion icin kullanilmaya uygun olup
olmadigini kontrol eder. Bu gate bir paper-level kalite metriği degil,
operasyonel kabul/red kuralidir.

### 5.1 Sanity Subset

- subset: validation split icinden ilk `64` sample
- gorsel inceleme: bu subset icinden ilk `8` sample

### 5.2 Gate Conditions

Checkpoint ancak su kosullarin tumu saglanirsa kabul edilir:

1. `recon`, `mu`, `logvar` icinde `NaN` veya `Inf` yok.
2. Z-kanali reconstruction icin mean `spec_corr >= 0.88`
3. mean `||mu||_2` araligi:
   - `0.5 <= mean_mu_norm <= 25.0`
4. mean `logvar` araligi:
   - `-6.0 <= mean_logvar <= 2.0`
5. per-dim `mu` standard deviation ortalamasi:
   - `mean(std(mu_j)) >= 0.01`
6. Ilk `8` sample icin STFT sanity plotlari gozle kontrol edilir.

### 5.3 Failure Policy

Su durumda diffusion egitimi baslatilmaz:

- gate kosullarindan biri fail olursa
- veya gorsel incelemede acik reconstruction bozulmasi varsa

## 6. Frozen Diffusion Spec

### 6.1 Objective

- objective: `epsilon prediction`

### 6.2 Schedule

- beta schedule: `cosine`

### 6.3 Timesteps

- diffusion training timesteps: `200`

### 6.4 Denoiser

- model: `ResMLP`

### 6.5 Denoiser Condition

- condition mode: `embedding_plus_raw`

Yani denoiser girdisi:

- learned condition embedding
- ham fiziksel condition

### 6.6 Latent Normalization

Diffusion egitimi ve sampling su normalize latent uzayinda calisir:

[
z_\text{norm} = \frac{z_\mu - \mu_\text{train}}{\sigma_\text{train} + 1e{-}8}
]

### 6.7 CFG Policy

- classifier-free guidance: `kapali`

### 6.8 Diffusion Runtime Defaults

- batch size: `2048`
- num workers: `8`

### 6.9 Frozen Diffusion Run Name

- `diffusion_eventwise_v1`

## 7. Frozen Metric Definitions

Ana metrikler, decoder cikti STFT'sinin Z-kanali uzerinden hesaplanir.

Waveform reconstruction sadece ikincil gorsel/sanity aracidir;
primer skorlamada kullanilmaz.

### 7.1 `spec_corr`

Z-kanali normalized magnitude STFT'leri flatten edilip Pearson korelasyonu
hesaplanir.

Yorum:

- yuksek daha iyi

### 7.2 `LSD`

[
\mathrm{LSD}(S,\hat S)=\sqrt{\mathrm{mean}\left(\log(S+\epsilon)-\log(\hat S+\epsilon)\right)^2}
]

Frozen detaylar:

- log: natural log
- epsilon: `1e-8`
- kanal: yalnizca Z

Yorum:

- dusuk daha iyi

### 7.3 `MR-LSD`

`MR-LSD`, ayni Z-kanali spectrogram uzerinde uc farkli olcekte LSD ortalamasi
olarak tanimlanir.

Olcekler:

- scale `1x`: orijinal grid
- scale `2x`: freq ve time eksenlerinde `2x2 average pooling`
- scale `4x`: freq ve time eksenlerinde `4x4 average pooling`

[
\mathrm{MR\text{-}LSD} = \frac{\mathrm{LSD}_{1x} + \mathrm{LSD}_{2x} + \mathrm{LSD}_{4x}}{3}
]

Gerekce:

- ayni STFT temsilinde kalir
- waveform inversion hatasini ana metrikten uzak tutar
- hem yerel hem daha kaba olcekte spektral farki gorur

Yorum:

- dusuk daha iyi

### 7.4 Secondary Outputs

Asagidaki urunler tutulur ama primer ranking'de kullanilmaz:

- Griffin-Lim waveform gorselleri
- oracle waveform
- DDPM waveform
- DDIM waveform

## 8. Frozen Sampler Fairness Protocol

Bu bolum, `DDPM` ve `DDIM` sampler kiyasinin adil olmasini saglar.

### 8.1 Fixed Inputs

Her test sample icin su degiskenler sabittir:

- ayni Stage-1 decoder
- ayni diffusion checkpoint
- ayni condition
- ayni normalized latent noise `x_T`

### 8.2 Initial Noise Policy

- sampler base seed: `1234`
- her sample icin noise seed:
  - `1234 + dataset_index`

Bu sayede `DDPM` ve `DDIM` ayni baslangic noise'dan baslar.

### 8.3 Reverse Steps

- `DDPM`: tam `200` step
- `DDIM`: `50` step
- `DDIM eta`: `0.0`

### 8.4 Oracle Reference

Her sample icin bir referans daha uretilir:

- `oracle_mu = decode(mu, cond)`

Bu referans sampler degil, Stage-1 latent ceiling'idir.

### 8.5 Reporting Rule

Her sampler icin rapor iki seviyede yazilir:

1. sample-level tablo
2. split-level ortalama

Rapor sirasi:

1. `test` split
2. `OOD` split

## 9. Run Naming and Seeds

Frozen isimler:

- Stage-1 run: `stage1_eventwise_v1`
- Cache build label: `cache_eventwise_v1`
- Diffusion run: `diffusion_eventwise_v1`
- Sampler comparison report: `sampler_compare_eventwise_v1`

Frozen seedler:

- split seed: `42`
- Stage-1 train seed: `42`
- diffusion train seed: `42`
- sampler base seed: `1234`

## 10. Acceptance Rule

Asagidaki durumda deney bir sonraki asamaya gecer:

1. Event-wise Stage-1 egitimi tamamlanmis olmali
2. `best val loss` checkpoint alinmis olmali
3. latent sanity gate PASS olmali
4. cache PASS olmali
5. diffusion training PASS olmali
6. sampler comparison test split uzerinde tamamlanmis olmali

Asagidaki durumda durulur:

- Stage-1 sanity gate FAIL
- latent cache icinde NaN/Inf
- diffusion train loss patlamasi
- sampler fairness protokolunun bozulmasi
