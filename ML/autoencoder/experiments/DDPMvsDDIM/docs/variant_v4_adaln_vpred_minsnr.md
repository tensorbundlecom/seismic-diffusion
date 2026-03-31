# Variant v4: AdaLN ResMLP + v-prediction + min-SNR weighting

Bu varyant, `v3` uzerine tek bir yeni eksen ekler:

- `min-SNR` loss weighting

## 1. Goal

`v3`, `AdaLNResMLP + v-prediction` ile su ana kadar en iyi sonucu verdi.
`v4` amaci, ayni mimari ve objective'i koruyup timestep katkisinin daha dengeli
ogrenilmesini saglamaktir.

Bu sayede:

- dusuk-SNR timestep'lerin egitimi domine etmesi azaltilir
- sampler kalitesi, ozellikle `DDIM`, daha da iyilesebilir

## 2. v3 -> v4 Difference

### v3

- model: `adaln_resmlp`
- objective: `v-prediction`
- weighting: `none`

### v4

- model: `adaln_resmlp`
- objective: `v-prediction`
- weighting: `min_snr`
- gamma: `5.0`

## 3. Rationale

Bu deneyde kullandigimiz standart diffusion loss, tum timestep'lere esit
agirlik verir. Oysa pratikte bazi timestep rejimleri orantisiz bicimde baskin
olabilir.

`min-SNR` weighting ile:

- timestep agirliklari `SNR_t` uzerinden normalize edilir
- `v-prediction` icin agirlik:
  - `min(SNR_t, gamma) / (SNR_t + 1)`

Burada:

- `gamma = 5.0`

Bu secim, standart diffusers/practical diffusion egitiminde kullanilan
yaygin bir dengeleme yaklasimina dayanir.

## 4. Frozen v4 Training Spec

- model type: `adaln_resmlp`
- hidden dim: `768`
- depth: `10`
- prediction target: `v`
- loss weighting: `min_snr`
- min-SNR gamma: `5.0`
- schedule: `cosine`
- timesteps: `200`
- cond mode: `embedding_plus_raw`
- batch size: `2048`
- epochs: `100`
- lr: `1e-4`
- seed: `42`
- run name: `diffusion_eventwise_v4_adaln_vpred_minsnr`

## 5. Acceptance Rule

`v4` basarili sayilacaksa:

1. training stabil olmali
2. `v3`'e gore en az bir ana metrikte net iyilesme vermeli
3. iyilesme sadece DDPM'e degil, DDIM tarafina da yansimali

## 6. Output Convention

- run:
  - `ML/autoencoder/experiments/DDPMvsDDIM/runs/diffusion/diffusion_eventwise_v4_adaln_vpred_minsnr/`
- summary:
  - `ML/autoencoder/experiments/DDPMvsDDIM/results/sampler_comparison_eventwise_v4_adaln_vpred_minsnr/summary.json`
- visual subset:
  - `ML/autoencoder/experiments/DDPMvsDDIM/results/sampler_comparison_eventwise_v4_adaln_vpred_minsnr_subset25/`
