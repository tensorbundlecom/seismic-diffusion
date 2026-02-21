# LatentShapeVAE Protocol Decisions (V1)

Status: `FROZEN_V1`
Date: `2026-02-18`

## D1 - Problem Tanimi

- Amaç: conditional bilgiyi disarida birakarak latent-space geometri hipotezini yalniz VAE ile test etmek.
- Ana iddia: kaliteyi bozmadan, `q(z)` aggregate seklini `N(0, I)` hedefine yaklastiran model secilebilir.

## D2 - Veri ve Split

- Veri kaynagi: `data/external_dataset/extracted/data/filtered_waveforms/HH`
- Split: event-wise
  - train / val / test / ood-event
  - OOD setindeki event id'ler train/val/testte kesinlikle olmayacak.
- Split artefakti:
  - `protocol/frozen_event_splits_v1.json`
  - `protocol/splits_v1/*.txt`

## D3 - Preprocessing ve Normalizasyon

- Kanal: 3C (`HHE`, `HHN`, `HHZ`)
- Fs: 100 Hz
- Segment: dosyadan gelen sabit pencere (su an 7001 sample)
- Onislem:
  - detrend + demean
  - bandpass: `0.5-20 Hz`
- Normalizasyon:
  - **per-trace normalization yok**
  - train-only global channel-wise mean/std (3 kanal icin ayri)
  - frozen artefakt: `protocol/waveform_stats_v1.json`

## D4 - Model ve Loss

- Model: unconditional diagonal-Gaussian VAE
  - encoder output: `mu(x), logvar(x)`
  - `q(z|x)=N(mu, diag(exp(logvar)))`
- Decoder output: reconstructed 3C waveform
- Loss:
  - `L_time = MSE(x_hat, x)`
  - `L_mrstft = MR-STFT log-mag loss` (multi-resolution)
  - `L = L_time + lambda_mr * L_mrstft + beta * KL_raw`

## D5 - Baselines

- Deterministic AE (KL yok)
- VAE beta=0
- VAE beta>0 (sabit / annealing / free-bits)

## D6 - Collapse Guard

- Active Units (AU) raporlanacak.
- Per-dim KL median raporlanacak.
- Collapse adayi:
  - `AU < 0.1 * latent_dim` veya
  - per-dim KL median ~ 0
- Collapse adayi modeller sampling gate'te ek kontrol ile elenecek.

## D7 - Latent Shape Metrikleri

Tanım:

- `Cov_mu = Cov_x[mu(x)]`
- `Mean_Sigma = E_x[diag(exp(logvar(x)))]`
- `Cov_agg = Cov_mu + Mean_Sigma`

Moment-matched Gaussian:

- `m_hat = E_x[mu(x)]`
- `S_hat = Cov_agg`
- `q_hat(z) = N(m_hat, S_hat)`

Karar metrikleri:

- `||m_hat||_2`
- `diag_mae = mean(|diag(S_hat)-1|)`
- `offdiag_mean_abs(corr(S_hat))` (yalniz ek tanisal)
- `eig_ratio = max_eig(S_hat)/min_eig(S_hat)`
- `KL(q_hat || N(0,I))`
- `W2^2(q_hat, N(0,I))`

Not: KL/W2 burada tam `q(z)` divergence degil; moment-matched Gaussian approximation metrigidir.

## D8 - Kalite ve Sampling Gate

- Kalite kapisi:
  - IID test + OOD-event testte reconstruction metrikleri
  - `time-MSE`, `MR-STFT`, ve waveform tabanli secili metrikler
- Prior sampling gate:
  - `z ~ N(0,I)` sample -> decoder
  - realism metrikleri (frozen):
    - band energy ratios
    - envelope peak/kurtosis/duration
    - PSD slope + spectral centroid

## D9 - Karar Kurali

1. Kalite kapisini gecen adaylar tutulur.
2. Bu adaylar icinde latent-shape composite (KL/W2/diag/eig/mean) ile siralanir.
3. Tie-break: daha kucuk model.

## D10 - Compute Stratejisi

- Stage-1: hizli IID tarama (kisa butce)
- Stage-2: top-K tam butce + IID/OOD + sampling gate

## D11 - Stage-1 Secim Karari (2026-02-19)

- Stage-1 sonunda operasyonel aday olarak `VAE beta=0.1` secildi.
- Gerekce:
  - Latent-shape metriklerinde (`KL_moment`, `W2_moment`, `diag_mae`, `eig_ratio`) en tutarli performans.
  - Prior-sampling realism skorunda en iyi/rekabetci sonuc.
  - `AE` ve `beta0` varyantlari sayisal stabil hale getirilse de latent geometri hedefini karsilamiyor.
- Sonraki adim:
  - Stage-2 multi-seed dogrulama (`seed=42,43,44`) ayni beta rejimi ile.
  - IID + OOD (`ood_event`) latent-shape ve prior-sampling raporu ile final karar.

## D12 - Stage-2 Logvar Stabilizasyonu (2026-02-20)

Problem:
- Stage-2 `beta=0.1` calismasinda `s43/s44` test splitinde nadir fakat asiri buyuk posterior varyans outlierlari
  (`max_var` ~ `2.87e11` ve `8.65e7`) goruldu.
- Bu outlierlar latent-shape metriklerini bozdu; OOD tarafinda ayni siddette tekrarlanmadi.

Uygulanan patch:
- VAE encoder logvar cikisi icin opsiyonel bounded parametrizasyon eklendi:
  - `logvar_mode=bounded_sigmoid`
  - `logvar = logvar_min + (logvar_max-logvar_min)*sigmoid(raw)`
- Rerunlar:
  - `lsv_stage2_vae_base_ld64_b0p1_s43_logvfixv1`
  - `lsv_stage2_vae_base_ld64_b0p1_s44_logvfixv1`
  - ayarlar: `logvar_min=-12`, `logvar_max=8`

Sonuc ozeti:
- Outlier siddeti belirgin dustu:
  - `s43`: `max_var 2.87e11 -> 2.87e3`, `mean_sigma_diag_mean 5.83e5 -> 0.945`
  - `s44`: `max_var 8.65e7 -> 2.98e3`, `mean_sigma_diag_mean 177 -> 1.336`
- `s43` icin test latent-shape ciddi duzeldi (`diag_mae 1476.64 -> 0.0646`).
- `s44` icin iyilesme var ama tam duzelme yok (`diag_mae 176.07 -> 1.4189`, offdiag yuksek kaldi).
- Prior-sampling realism metriklerinde net tek-yonlu iyilesme yok (trade-off devam ediyor).

Karar:
- Stabilizasyon patchinin devam etmesi gerekceli (outlier rejimini kiriyor).
- Ancak patch tek basina tum seedleri ideal latent-shape rejimine getirmiyor.
- Operasyonel referans aday olarak `s42` korunur; patchli rejim bir sonraki grid icin aday olur.

## D13 - 10-Seed Robustness Turu (2026-02-20)

Amaç:
- `beta=0.1 + bounded_logvar` rejiminin seed-robust olup olmadigini confirmatory sekilde olcmek.
- Tek-seed basari/hataya dayali yorum riskini azaltmak.

Frozen recipe:
- model: `vae/base/latent_dim=64`
- `beta=0.1`
- `logvar_mode=bounded_sigmoid`
- `logvar_min=-12`, `logvar_max=8`
- optimizasyon: `lr=2e-4`, `batch=64`, `max_steps=12000`, `grad_clip=0.5`, `amp=0`

Frozen seeds:
- `42, 43, 44, 45, 46, 47, 48, 49, 50, 51`
- run name pattern:
  - `lsv_stage2_vae_base_ld64_b0p1_s<seed>_logvfixv2`

Degerlendirme paketi:
- latent shape: `test + ood_event`
- prior sampling realism: `test + ood_event`
- latent variance outlier audit: `test + ood_event`
- 10-seed aggregate rapor:
  - `results/stage2_beta0p1_logvarfix_10seeds_v2/robustness_summary.md`

Not:
- Bu turda recipe degistirme yok; yalniz seed etkisi olculecek.
- Sonraki karar adimi aggregate dagilimlar (mean/std/p90 + fail-pattern) uzerinden verilecek.

## D14 - Closure Karari (2026-02-20)

10-seed (`42..51`) bounded-logvar turu tamamlandi.

Durum:
- **Numerik stabilite:** PASS
- **Seed-robust latent-shape:** FAIL
- **Ana latent-shape optimizasyon iddiasi:** PARTIAL
- **Operasyonel checkpoint secimi:** PASS

Operasyonel secim:
- run: `lsv_stage2_vae_base_ld64_b0p1_s43_logvfixv2`
- checkpoint:
  - `ML/autoencoder/experiments/LatentShapeVAE/checkpoints/lsv_stage2_vae_base_ld64_b0p1_s43_logvfixv2_best.pt`
- frozen kayit:
  - `protocol/selected_operational_checkpoint_v2.json`

Final rapor:
- `ML/autoencoder/experiments/LatentShapeVAE/docs/final_closure_report_2026-02-20.md`

Kural:
- Bu closure'dan sonra ayni hatti \"daha fazla seed daha fazla tekrar\" mantigiyla devam ettirmeyiz.
- Yeni bir hipotez veya yeni frozen policy olmadan ek train acilmaz.

## D15 - Latent=32 Mini Format Turu (2026-02-20)

Amaç:
- `latent_dim=64` yerine daha kompakt bir bottleneck ile latent-shape/stabilite dengesini test etmek.
- Buyuk grid yerine sinirli bir mini deneyle yon tayin etmek.

Frozen stage3-v1 setup:
- backbone: `base`
- latent_dim: `32`
- seeds: `42,43,44`
- formatlar:
  1. `fmtA_b0p1_lmax8`
  2. `fmtB_b0p1_lmax6`
  3. `fmtC_b0p03_anneal_lmax6`

Scriptler:
- train:
  - `training/run_stage3_ld32_formats_v1.sh`
- eval:
  - `evaluation/run_stage3_ld32_formats_v1.sh`
  - `evaluation/wait_and_run_stage3_ld32_formats_eval_v1.sh`
  - `evaluation/summarize_stage3_ld32_formats_v1.py`

Degerlendirme:
- latent-shape (test + ood_event)
- prior-sampling realism (test + ood_event)
- outlier audit
- format bazli mean±std ozet tablosu.
