# Variant v3: AdaLN ResMLP + v-prediction

Bu varyant, `v2` mimarisinin uzerine tek bir ana degisiklik ekler:

- objective `epsilon` yerine `v-prediction`

## 1. Goal

`v2` mimarisi daha guclu condition/time modulation getiriyor.
`v3`'te mimariyi sabit tutup objective'i degistiriyoruz. Boylece:

- mimari etkisi `v2`
- objective etkisi `v3`

ayri okunabilir.

## 2. v2 -> v3 Difference

### v2

- model: `adaln_resmlp`
- objective: `epsilon prediction`

### v3

- model: `adaln_resmlp`
- objective: `v-prediction`

## 3. Rationale

`v-prediction`, latent diffusion tarafinda daha stabil hedef verebilir.
Ozellikle:

- farkli timestep rejimlerinde hedef olcegini daha dengeli tutabilir
- `DDIM` tarafinda daha iyi davranabilir

Bu nedenle `v3`, sampler farkini okumadan once ortak modelin objective
seciminden kazanc elde edip etmedigini test eder.

## 4. Frozen v3 Training Spec

- model type: `adaln_resmlp`
- hidden dim: `768`
- depth: `10`
- prediction target: `v`
- objective loss: plain MSE on `v`
- schedule: `cosine`
- timesteps: `200`
- cond mode: `embedding_plus_raw`
- batch size: `2048`
- epochs: `100`
- lr: `1e-4`
- seed: `42`
- run name: `diffusion_eventwise_v3_adaln_vpred`

## 5. Acceptance Rule

`v3` basarili sayilacaksa:

1. training stabil olmali
2. `v2`'ye karsi en az bir ana metrikte anlamli iyilesme vermeli
3. ozellikle `DDIM` tarafinda tutarli bir kazanc saglamasi beklenir

## 6. Output Convention

- run:
  - `ML/autoencoder/experiments/DDPMvsDDIM/runs/diffusion/diffusion_eventwise_v3_adaln_vpred/`
- summary:
  - `ML/autoencoder/experiments/DDPMvsDDIM/results/sampler_comparison_eventwise_v3_adaln_vpred/summary.json`
- visual subset:
  - `ML/autoencoder/experiments/DDPMvsDDIM/results/sampler_comparison_eventwise_v3_adaln_vpred_subset25/`
