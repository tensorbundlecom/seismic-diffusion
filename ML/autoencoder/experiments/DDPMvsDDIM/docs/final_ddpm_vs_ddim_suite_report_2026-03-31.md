# Final DDPM vs DDIM Suite Report (2026-03-31)

Bu rapor, `DDPMvsDDIM` kutusunda `v3` diffusion modeli uzerinde yapilan
nihai sampler karsilastirmasini toplar.

Bu raporun amaci herhangi bir sampler'i savunmak degil; ayni model,
ayni split, ayni latent cache ve ayni metriklerle tum sampler seceneklerini
yan yana koymaktir.

## 1. Frozen Setup

- Stage-1: `stage1_eventwise_v1_best.pt`
- diffusion model: `v3 = AdaLNResMLP + v-prediction`
- split: frozen event-wise test split
- samples: `8838`
- metrics:
  - `spec_corr` (yuksek daha iyi)
  - `LSD` (dusuk daha iyi)
  - `MR-LSD` (dusuk daha iyi)

Samplers:

- `DDPM-200`
- `DDIM-25`
- `DDIM-50`
- `DDIM-100`

Oracle referans:

- `Oracle mu`
- bu bir sampler degil; encoder'dan gelen gercek `mu(x)` latentinin
  decoder'a verilmesiyle olusan ust-sinir referansidir

## 1.1 How To Read This Report

Bu raporda sadece tek bir sey degisiyor:

- sampler

Su seyler sabit:

- ayni `v3` diffusion checkpoint
- ayni `stage1_eventwise_v1_best.pt`
- ayni frozen event-wise test split
- ayni `8838` ornek
- ayni metrikler
- ayni seed policy

Rapor uc bagimsiz pairwise kosudan olusuyor:

1. `DDPM-200` vs `DDIM-25`
2. `DDPM-200` vs `DDIM-50`
3. `DDPM-200` vs `DDIM-100`

Bu nedenle:

- `DDPM-200` her satirda yeniden hesaplanmistir
- `DDPM-200` sayilari satirlar arasinda pratikte ayni olmalidir
- son basamaklarda gorulen kucuk farklar, ayni referansin yeniden kosulmasindan
  kaynaklanan numerik tekrar farklaridir; metodolojik fark degildir

Bu rapordaki tablolar su sekilde okunmali:

- kalite tablosu:
  - hangi sampler en iyi son kaliteyi verdi
- runtime tablosu:
  - her pairwise kosunun toplam duvar suresi
  - deployment icin asil anlamli sayi: `ms/sample`
- kaynak tablosu:
  - ilgili pairwise kosu boyunca gozlenen process/GPU kullanimi

## 2. Quality Comparison

| Method | spec_corr | LSD | MR-LSD |
| --- | ---: | ---: | ---: |
| Oracle mu | `0.9544` | `1.6058` | `1.4533` |
| DDPM-200 | `0.8963` | `1.7194` | `1.5743` |
| DDIM-25 | `0.8981` | `1.7138` | `1.5680` |
| DDIM-50 | `0.8968` | `1.7169` | `1.5713` |
| DDIM-100 | `0.8961` | `1.7186` | `1.5730` |

### Okuma

- `DDIM-25`, uc ana kalite metriğinde de en iyi sampler sonucu verdi.
- `DDIM-50`, `DDPM-200` ile cok yakin ama onunde.
- `DDIM-100`, `DDIM-25` ve `DDIM-50`'nin gerisine dustu.
- `DDPM-200` kotu degil; ancak bu setup'ta kalite lideri degil.

Kalite farklari kucuk ama yon tutarli:

- `DDIM-25` vs `DDPM-200`
  - `spec_corr`: `+0.0018`
  - `LSD`: `-0.0056`
  - `MR-LSD`: `-0.0062`

Bu farklar devasa degil; ama uc metrikte de ayni yone gittigi icin
operasyonel olarak anlamli.

## 3. Runtime Comparison

Not:

- `wall_time_min`, tum evaluation kosusunun toplam sureidir
- her kosuda hem `DDPM` hem ilgili `DDIM` sampler'i hesaplandigi icin,
  deployment acisindan asil yararli hiz metrikleri:
  - `avg_ddpm_total_time_ms`
  - `avg_ddim_total_time_ms`

| Pairwise run | Wall time (min) | Samples/s | DDPM total ms/sample | DDIM total ms/sample |
| --- | ---: | ---: | ---: | ---: |
| DDPM-200 vs DDIM-25 | `40.31` | `3.654` | `233.33` | `33.08` |
| DDPM-200 vs DDIM-50 | `44.26` | `3.328` | `229.69` | `63.53` |
| DDPM-200 vs DDIM-100 | `53.61` | `2.747` | `230.89` | `125.96` |

Bu tabloda `wall_time_min` ile `ms/sample` ayni sey degildir:

- `wall_time_min`:
  - tum pairwise evaluation'in toplami
  - yani o kosuda hem `DDPM` hem ilgili `DDIM` calisiyor
- `avg_ddpm_total_time_ms` ve `avg_ddim_total_time_ms`:
  - deployment veya inference acisindan dogrudan kullanilmasi gereken sayilar
  - cunku sampler basina gercek gecikmeyi verir

### Hız oranları

`DDPM-200` referansina gore:

- `DDIM-25`: yaklasik `7.1x` daha hizli
- `DDIM-50`: yaklasik `3.6x` daha hizli
- `DDIM-100`: yaklasik `1.8x` daha hizli

## 4. Resource Comparison

| Pairwise run | Peak RSS MB | GPU util avg | GPU util peak | Torch peak allocated MB | Torch peak reserved MB |
| --- | ---: | ---: | ---: | ---: | ---: |
| DDPM-200 vs DDIM-25 | `1664.63` | `67.50` | `83.0` | `340.34` | `1262.0` |
| DDPM-200 vs DDIM-50 | `1539.62` | `68.70` | `90.0` | `340.34` | `1262.0` |
| DDPM-200 vs DDIM-100 | `1671.25` | `67.77` | `73.0` | `340.34` | `1262.0` |

Ek not:

- GPU bellek kullanimi sampler secenegine gore dramatik degismedi
- farkin ana kaynagi sure ve step sayisi
- CPU yuzdeleri process bazli toplam kullanim olarak kaydedildi;
  cok cekirdekli bir sistemde `100%` uzeri degerler normaldir
- kaynak tablosundaki degerler, ilgili pairwise kosunun toplam profiline aittir;
  yalnizca `DDIM`'in tek basina ayak izi degildir

## 5. Neutral Interpretation

Bu raporun sonucu su sekilde okunmali:

1. `DDPM-200` hala guclu bir referans.
   - Kalite olarak rekabetci
   - Stabil
   - Ancak bu final setup'ta en iyi sampler sonucu degil

2. `DDIM-25`, bu belirli problemde en iyi trade-off'u veriyor.
   - Kalite olarak en iyi
   - Hiz olarak da acik ara onde

3. `DDIM` icin daha fazla step eklemek otomatik kazanc getirmedi.
   - `25 -> 50 -> 100` ilerlerken kalite monoton artmadi
   - Bu, bu latent problem icin erken durmanin yeterli oldugunu gosteriyor

4. Bu sonuc, yalnizca `v3` backbone'u icin gecerlidir.
   - `v1` ve `v2` gibi daha zayif modellerde tablo farkliydi
   - Yani sampler sonucu, ortak diffusion model kalitesinden bagimsiz degil

## 6. Practical Closure

Bu rapordan cikan operasyonel sonuc su:

- herhangi bir sampler lehine on-kabul ile hareket edilmedi
- tum samplerlar ayni kosullarda olculdu
- bu frozen setup icinde en iyi kalite-hiz dengesi `DDIM-25` tarafinda
  gozlendi

`DDPM-200` icin ek yorum:

- zayif bir baseline degil
- guclu ve adil referans olarak olculdu
- ancak bu final setup'ta kalite lideri degil
- hiz tarafinda da `DDIM-25`'in belirgin gerisinde

Bu nedenle, mevcut frozen setup icinde ek `DDPM` denemesi metodolojik olarak
zorunlu gorunmuyor.

Bu raporun iddiasi su kadar:

- `v3` modeli uzerinde
- `DDPM-200`, `DDIM-25`, `DDIM-50`, `DDIM-100`
  dogrudan karsilastirildi
- en iyi sonuc `DDIM-25` ile gozlendi

Bu raporun iddia etmedigi sey:

- `DDIM` her problemde `DDPM`'den ustundur
- farkli backbone veya farkli latent hedeflerinde ayni siralama mutlaka
  korunur

## 7. Official Result Files

- suite summary:
  - `ML/autoencoder/experiments/DDPMvsDDIM/results/final_sampler_suite_v3/suite_summary.json`
  - `ML/autoencoder/experiments/DDPMvsDDIM/results/final_sampler_suite_v3/suite_summary.md`
- per-run summaries:
  - `ML/autoencoder/experiments/DDPMvsDDIM/results/final_sampler_suite_v3/ddim_025/summary.json`
  - `ML/autoencoder/experiments/DDPMvsDDIM/results/final_sampler_suite_v3/ddim_050/summary.json`
  - `ML/autoencoder/experiments/DDPMvsDDIM/results/final_sampler_suite_v3/ddim_100/summary.json`
