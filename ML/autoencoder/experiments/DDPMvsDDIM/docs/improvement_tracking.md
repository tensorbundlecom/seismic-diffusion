# DDPMvsDDIM Improvement Tracking

Bu dosya, frozen `v1` sonucundan sonra sadece diffusion model tarafinda
yapilacak iyilestirmeleri ve ciktilar geldikce eklenen sonuclari takip etmek
icin tutulur.

Temel kural:

- Stage-1 backbone degismeyecek
- event-wise split degismeyecek
- latent target `mu` olarak kalacak
- ana metrikler ayni kalacak:
  - `spec_corr`
  - `LSD`
  - `MR-LSD`

## 1. Frozen v1 Reference

Referans run:

- report:
  - `ML/autoencoder/experiments/DDPMvsDDIM/docs/current_state_2026-03-29.md`
- full evaluation:
  - `ML/autoencoder/experiments/DDPMvsDDIM/results/sampler_comparison_eventwise_v1/summary.json`

Reference mean metrics:

| Method | spec_corr | LSD | MR-LSD |
| --- | ---: | ---: | ---: |
| Oracle mu | `0.9544` | `1.6058` | `1.4533` |
| DDPM | `0.8806` | `1.7623` | `1.6192` |
| DDIM | `0.8679` | `1.8094` | `1.6685` |

## 2. Improvement Goal

Amac:

- ayni condition seti ile
- daha guclu ortak diffusion modeli kurup
- `DDPM` ve `DDIM` farkini daha saglikli okumak

Beklenen iyilesme:

- her iki sampler da `oracle mu`'ya yaklasmali
- sonra `DDPM vs DDIM` farki daha anlamli yorumlanabilmeli

## 3. Planned Model-Side Upgrades

### P1. Bigger Condition-Aware Denoiser

Mevcut:

- `ResMLPDenoiser`
- hidden `512`
- depth `6`
- time/condition injection: giriste tek toplama

Plan:

- daha buyuk hidden dim
- daha derin residual yapi
- her blokta daha guclu condition/time modulation

Current execution target:

- variant spec:
  - `ML/autoencoder/experiments/DDPMvsDDIM/docs/variant_v2_adaln_resmlp.md`
- target run:
  - `diffusion_eventwise_v2_adaln`

Durum:

- `ACTIVE`

### P2. v-prediction

Mevcut:

- `epsilon prediction`

Plan:

- `adaln_resmlp` mimarisi korunup objective `v-prediction` olacak

Variant spec:

- `ML/autoencoder/experiments/DDPMvsDDIM/docs/variant_v3_adaln_vpred.md`

Durum:

- `ACTIVE`

### P3. min-SNR weighting

Mevcut:

- plain timestep-uniform MSE

Plan:

- `v3` setup korunup `min-SNR` loss weighting eklenecek

Variant spec:

- `ML/autoencoder/experiments/DDPMvsDDIM/docs/variant_v4_adaln_vpred_minsnr.md`

Durum:

- `DONE`

### P4. Self-conditioning

Durum:

- `CANDIDATE`

Not:

- ilk uc iyilestirmeden sonra degerlendirilecek

## 4. Result Table Template

Yeni sonuclar geldikce bu tablo doldurulacak.

| Variant | Change | DDPM spec_corr | DDIM spec_corr | DDPM LSD | DDIM LSD | DDPM MR-LSD | DDIM MR-LSD | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| v1 | baseline ResMLP | `0.8806` | `0.8679` | `1.7623` | `1.8094` | `1.6192` | `1.6685` | current reference |
| v2 | AdaLNResMLP + epsilon | `0.8893` | `0.8850` | `1.7344` | `1.7552` | `1.5901` | `1.6115` | DDPM still slightly ahead |
| v3 | AdaLNResMLP + v-pred | `0.8963` | `0.8968` | `1.7194` | `1.7169` | `1.5743` | `1.5713` | current best; DDIM becomes competitive |
| v4 | AdaLNResMLP + v-pred + min-SNR | `0.8862` | `0.8854` | `1.7319` | `1.7314` | `1.5883` | `1.5877` | unsuccessful; better than v1 but below v3 |

## 5. Interpretation Rule

Bir varyant basarili sayilacaksa:

1. en az bir sampler'i `v1` reference'in uzerine tasimali
2. iyilesme tek metric'e sikismamali
3. `DDPM vs DDIM` farki daha tutarli hale gelmeli

Kritik not:

- yalnizca `DDPM > DDIM` demek yeterli degil
- amac, once ortak modelin kalitesini yukari cekmek

## 6. Current Decision

Final sampler study icin secilen varyant:

- `v3 = AdaLNResMLP + v-prediction`

Neden:

- `v3` su ana kadarki en iyi genel kaliteyi verdi
- `v4` val loss iyilestirse de downstream kaliteyi iyilestirmedi
- sonraki adim artik yeni training degil, instrumented final sampler suite

Final suite spec:

- `ML/autoencoder/experiments/DDPMvsDDIM/docs/final_sampler_suite_v3_2026-03-31.md`
