# Variant v2: AdaLN Condition-Aware ResMLP

Bu belge, `DDPMvsDDIM` kutusundaki ilk model-side iyilestirme varyantini
tanimlar.

## 1. Goal

`v1` sonucunda ana sorun sampler secimi degil, ortak diffusion denoiser'in
gucu olarak goruldu.

Bu nedenle `v2`'de:

- Stage-1 backbone sabit
- event-wise split sabit
- cache sabit
- objective sabit (`epsilon prediction`)
- schedule sabit (`cosine`)

yalnizca denoiser mimarisi guclendirilecektir.

## 2. v1 -> v2 Difference

### v1

- model: `ResMLPDenoiser`
- hidden dim: `512`
- depth: `6`
- time ve condition bilgisi giriste tek toplama ile enjekte ediliyor

### v2

- model: `AdaLNResMLPDenoiser`
- hidden dim: `768`
- depth: `10`
- her residual blokta:
  - adaptive layer norm modulation
  - condition+time context ile `shift/scale/gate`
- cikista da condition-aware final modulation var

## 3. Rationale

Mevcut `v1` modeli latent + time + condition bilgisini sadece giriste bir kez
topluyor. Bu, condition'dan latent uretme isinde zayif kaliyor.

`v2`'nin beklenen faydasi:

1. condition bilgisinin tum bloklara ulasmasi
2. timestep bilgisinin tum bloklara ulasmasi
3. latent update'lerin daha kontrollu ve daha guclu olmasi

Bu degisiklik, condition setini degistirmeden modele daha fazla ifade gucu verir.

## 4. Frozen v2 Training Spec

- model type: `adaln_resmlp`
- hidden dim: `768`
- depth: `10`
- dropout: `0.0`
- objective: `epsilon prediction`
- schedule: `cosine`
- timesteps: `200`
- cond mode: `embedding_plus_raw`
- batch size: `2048`
- epochs: `100`
- lr: `1e-4`
- seed: `42`
- run name: `diffusion_eventwise_v2_adaln`

## 5. Acceptance Rule

`v2` ancak su durumda ilerletilir:

1. smoke run temiz gecerse
2. training stabil giderse
3. full evaluation'da en az bir ana metricte `v1`'i gecer

## 6. Output Convention

- run:
  - `ML/autoencoder/experiments/DDPMvsDDIM/runs/diffusion/diffusion_eventwise_v2_adaln/`
- summary:
  - `ML/autoencoder/experiments/DDPMvsDDIM/results/sampler_comparison_eventwise_v2_adaln/summary.json`
- visual subset:
  - `ML/autoencoder/experiments/DDPMvsDDIM/results/sampler_comparison_eventwise_v2_adaln_subset25/`
