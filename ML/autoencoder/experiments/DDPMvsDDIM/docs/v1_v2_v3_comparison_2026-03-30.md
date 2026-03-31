# DDPMvsDDIM v1-v2-v3 Comparison Report (2026-03-30)

Bu rapor, `DDPMvsDDIM` kutusundaki ilk uc resmi varyanti ayni frozen
event-wise protokol altinda karsilastirir:

- `v1`: baseline `ResMLP` + `epsilon`
- `v2`: `AdaLNResMLP` + `epsilon`
- `v3`: `AdaLNResMLP` + `v-prediction`

Bu raporun amaci iki soruya cevap vermektir:

1. sadece diffusion modelini guclendirerek kalite artiyor mu?
2. `DDPM` ve `DDIM` hangi varyantta nasil ayrisiyor?

## 1. Frozen Evaluation Frame

Tum varyantlar icin sabit tutulanlar:

- Stage-1 backbone:
  - `LegacyCondBaselineCVAE`
- split:
  - hybrid magnitude-stratified `event-wise`
- latent target:
  - `mu`
- latent cache:
  - `latent_cache_eventwise_v1`
- denoiser condition:
  - `cond_embedding + raw_condition`
- train timesteps:
  - `200`
- `DDIM` inference steps:
  - `50`
- metrics:
  - `spec_corr` (yuksek iyi)
  - `LSD` (dusuk iyi)
  - `MR-LSD` (dusuk iyi)

Dolayisiyla bu raporda gorulen farklar Stage-1 veya split kaynakli degil,
dogrudan diffusion model tarafindaki degisikliklerden geliyor.

## 2. Variant Definitions

### v1

- model: `ResMLPDenoiser`
- hidden dim: `512`
- depth: `6`
- objective: `epsilon`

Resmi sonuc:

- `ML/autoencoder/experiments/DDPMvsDDIM/results/sampler_comparison_eventwise_v1/summary.json`

### v2

- model: `AdaLNResMLPDenoiser`
- hidden dim: `768`
- depth: `10`
- objective: `epsilon`

Resmi sonuc:

- `ML/autoencoder/experiments/DDPMvsDDIM/results/sampler_comparison_eventwise_v2_adaln/summary.json`

### v3

- model: `AdaLNResMLPDenoiser`
- hidden dim: `768`
- depth: `10`
- objective: `v-prediction`

Resmi sonuc:

- `ML/autoencoder/experiments/DDPMvsDDIM/results/sampler_comparison_eventwise_v3_adaln_vpred/summary.json`

## 3. Full Test Results (8838 samples)

### 3.1 Mean metrics

| Variant | DDPM spec_corr | DDIM spec_corr | DDPM LSD | DDIM LSD | DDPM MR-LSD | DDIM MR-LSD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `v1` | `0.8806` | `0.8679` | `1.7623` | `1.8094` | `1.6192` | `1.6685` |
| `v2` | `0.8893` | `0.8850` | `1.7344` | `1.7552` | `1.5901` | `1.6115` |
| `v3` | `0.8963` | `0.8968` | `1.7194` | `1.7169` | `1.5743` | `1.5713` |

Oracle reference:

| Reference | spec_corr | LSD | MR-LSD |
| --- | ---: | ---: | ---: |
| `oracle mu` | `0.9544` | `1.6058` | `1.4533` |

### 3.2 Main ranking

Aggregate kalite siralamasi:

1. `v3`
2. `v2`
3. `v1`

Bu siralama hem `spec_corr` artisi hem de `LSD/MR-LSD` azalisiyla destekleniyor.

## 4. Improvement Over v1

### 4.1 DDPM side

`v1 -> v2`:

- `spec_corr`: `+0.0087`
- `LSD`: `-0.0280`
- `MR-LSD`: `-0.0291`

`v2 -> v3`:

- `spec_corr`: `+0.0071`
- `LSD`: `-0.0150`
- `MR-LSD`: `-0.0158`

`v1 -> v3`:

- `spec_corr`: `+0.0157`
- `LSD`: `-0.0429`
- `MR-LSD`: `-0.0449`

### 4.2 DDIM side

`v1 -> v2`:

- `spec_corr`: `+0.0171`
- `LSD`: `-0.0542`
- `MR-LSD`: `-0.0569`

`v2 -> v3`:

- `spec_corr`: `+0.0118`
- `LSD`: `-0.0383`
- `MR-LSD`: `-0.0403`

`v1 -> v3`:

- `spec_corr`: `+0.0290`
- `LSD`: `-0.0925`
- `MR-LSD`: `-0.0972`

Yorum:

- `DDIM` tarafi iyilestirmelerden daha fazla kazanc aldi.
- Ozellikle `v3`, `DDIM` icin belirgin bir sifirlayici etki yapti.

## 5. Gap to Oracle

### 5.1 Correlation gap (`oracle - sampler`)

| Variant | DDPM gap | DDIM gap |
| --- | ---: | ---: |
| `v1` | `0.0738` | `0.0865` |
| `v2` | `0.0651` | `0.0694` |
| `v3` | `0.0581` | `0.0576` |

### 5.2 LSD gap (`sampler - oracle`)

| Variant | DDPM gap | DDIM gap |
| --- | ---: | ---: |
| `v1` | `0.1566` | `0.2036` |
| `v2` | `0.1286` | `0.1495` |
| `v3` | `0.1136` | `0.1111` |

### 5.3 MR-LSD gap (`sampler - oracle`)

| Variant | DDPM gap | DDIM gap |
| --- | ---: | ---: |
| `v1` | `0.1659` | `0.2152` |
| `v2` | `0.1368` | `0.1583` |
| `v3` | `0.1210` | `0.1180` |

Yorum:

- `v3` iki sampler'i de `oracle mu` tavanina en cok yaklastiran varyant.
- `v1`'deki ana kalite boslugunun onemli bir kismi model-side iyilestirmelerle
  kapatilabiliyor.

## 6. DDPM vs DDIM Relationship

### 6.1 Aggregate mean difference

`DDPM - DDIM` farki:

| Variant | spec_corr gap | LSD gap (`DDIM - DDPM`) | MR-LSD gap (`DDIM - DDPM`) |
| --- | ---: | ---: | ---: |
| `v1` | `+0.0127` | `+0.0471` | `+0.0493` |
| `v2` | `+0.0043` | `+0.0209` | `+0.0214` |
| `v3` | `-0.0005` | `-0.0025` | `-0.0030` |

Bu tablo sunu gosteriyor:

- `v1`: net `DDPM` ustunlugu
- `v2`: `DDPM` ustunlugu azaliyor
- `v3`: aggregate ortalamada `DDIM` cok hafif onde

### 6.2 Per-sample win counts

`DDPM`'in `DDIM`'den daha iyi oldugu sample sayisi:

| Variant | spec_corr | LSD | MR-LSD |
| --- | ---: | ---: | ---: |
| `v1` | `5105 / 8838` | `4978 / 8838` | `4956 / 8838` |
| `v2` | `4822 / 8838` | `4712 / 8838` | `4682 / 8838` |
| `v3` | `4408 / 8838` | `4402 / 8838` | `4369 / 8838` |

Yorum:

- `v1` ve `v2`'de `DDPM` biraz daha sik kazaniyor.
- `v3`'te bu denge tersine yaklasiyor; `DDIM` sample bazinda da daha rekabetci
  hale geliyor.

Bu, `v-prediction` etkisinin sadece ortalama metriği degil, sampler davranisini
da degistirdigini gosteriyor.

## 7. 25-Sample Visual Subset

Visual subset protokolu:

- `25` sample
- `evenly_spaced` secim
- hem `specs/` hem `waveforms/`

### 7.1 v2 subset mean metrics

| Method | spec_corr | LSD | MR-LSD |
| --- | ---: | ---: | ---: |
| Oracle mu | `0.9520` | `1.8183` | `1.6703` |
| DDPM | `0.8625` | `1.9479` | `1.8058` |
| DDIM | `0.8545` | `1.9841` | `1.8383` |

Subset artefacts:

- `ML/autoencoder/experiments/DDPMvsDDIM/results/sampler_comparison_eventwise_v2_adaln_subset25/specs/`
- `ML/autoencoder/experiments/DDPMvsDDIM/results/sampler_comparison_eventwise_v2_adaln_subset25/waveforms/`

### 7.2 v3 subset mean metrics

| Method | spec_corr | LSD | MR-LSD |
| --- | ---: | ---: | ---: |
| Oracle mu | `0.9520` | `1.8183` | `1.6703` |
| DDPM | `0.8623` | `1.9455` | `1.8009` |
| DDIM | `0.8654` | `1.9283` | `1.7777` |

Subset artefacts:

- `ML/autoencoder/experiments/DDPMvsDDIM/results/sampler_comparison_eventwise_v3_adaln_vpred_subset25/specs/`
- `ML/autoencoder/experiments/DDPMvsDDIM/results/sampler_comparison_eventwise_v3_adaln_vpred_subset25/waveforms/`

Yorum:

- Visual subset de full-test sonucuyla uyumlu:
  - `v2`: `DDPM > DDIM`
  - `v3`: `DDIM > DDPM`

## 8. Technical Interpretation

### 8.1 What v2 proved

`v2`, condition-aware ve daha guclu denoiser mimarisinin gercekten fayda
sagladigini gosterdi.

Bu iki seyi kanitladi:

1. `v1`'deki ana darbo gaz sampler degil, ortak denoiser gucuydu
2. condition/time bilgisini her blokta kullanmak kaliteyi artiriyor

### 8.2 What v3 proved

`v3`, `v-prediction`in bu problemde ozellikle `DDIM` icin anlamli fayda
sagladigini gosterdi.

Sonuc:

- `DDPM` hala guclu
- fakat `v3` ile `DDIM`, aggregate ortalamada `DDPM`'i yakalayip cok az geciyor

Bu bulgu onemli, cunku ilk `v1` sonucunda `DDIM` zayif gorunuyordu. Simdi
gorulen sey su:

- `DDIM` dogasi geregi zayif degil
- ortak model yeterince iyi kurulmadiginda zayif gorunuyor
- objective secimi `DDIM` performansini ciddi etkiliyor

## 9. Main Conclusion

Bu fazin sonucunda ana iddia su sekilde guncellenebilir:

1. `DDPM vs DDIM` farkini adil okumak icin once ortak latent diffusion modelini
   yeterince guclu kurmak gerekiyor
2. baseline `v1` bu is icin yeterli degildi
3. `v2` model-side iyilestirme ile kaliteyi belirgin arttirdi
4. `v3` ise hem genel kaliteyi daha da arttirdi hem de `DDIM`'i gercekten
   rekabetci hale getirdi

Bugunku resmi lider varyant:

- `v3 = AdaLNResMLP + v-prediction`

## 10. Recommended Next Step

Bu noktada en mantikli sonraki deney:

- `v4 = AdaLNResMLP + v-prediction + min-SNR weighting`

Gerekce:

- `v2` mimari etkisini gosterdik
- `v3` objective etkisini gosterdik
- simdi loss weighting etkisini izole etmek mantikli

Training gerektirmeyen ikinci sonraki adim:

- kazanan varyant (`v3`) icin `DDIM step ablation`
  - `DDIM-25`
  - `DDIM-50`
  - `DDIM-100`

Bu da kalite-hiz tradeoff'unu acik sekilde verir.
