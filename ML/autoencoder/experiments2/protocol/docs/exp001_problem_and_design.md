# Experiment 001: Problem, Domain Rationale, and Design Decisions

## 1) Context and objective

Bu deneyin amaci, verilen deprem-ve-istasyon kosullari altinda tek-istasyon sismik kayitlarini uretebilen, izlenebilir ve tekrar uretilebilir bir CVAE hatti kurmaktir.

Bu hat iki gorevi ayni anda yurutur:

- Reconstruction: `x, c -> x_hat`
- Condition-only generation (ana odak): `c -> x_hat`

Bu ayrim bilerek korunur. Cunku reconstruction skoru iyi olan bir model, condition-only uretimde iyi olmayabilir.

## 2) Problem definition (seismology side)

Uretilecek hedef sinyal, istasyon tarafindan gozlenen deprem cevabinin zaman-frekans temsili (complex STFT) olarak tanimlanir.

Modelin condition tarafinda ogrenmesi beklenen fiziksel icerik:

- Kaynak buyuklugu etkisi (`magnitude`)
- Kaynak-istasyon geometri etkisi (`repi_km`, `azimuth`)
- Kaynak derinligi etkisi (`depth_km`)
- P/S varis zamanlama yapisi (1D hiz modelinden turetilen travel-time feature'lari)
- Istasyon-ozel davranis (station embedding)

## 3) Why this setup was chosen

Ana hedef condition-only oldugu icin model ailesi CVAE secildi.

Complex STFT seciminin nedeni:

- Faz bilgisini korumak
- Zamanlama davranisini magnitude-only yaklasimdan daha iyi tasimak
- Gerekirse iSTFT tarafinda daha anlamli geri donus saglamak

Event-wise split seciminin nedeni:

- Ayni eventin train/teste dagilmasini engellemek
- Leakage kaynakli yapay performansi onlemek

Veri kaynagi secimi:

- Bu fazda temel kaynak olarak `data/external_dataset/extracted/data/filtered_waveforms/HH` donduruldu.
- Pencereleme/hizalama ve preprocessing uyumu bu kaynaga gore sabitlenecek.
- Pencereleme yeniden tanimlanmayacak; mevcut dataset penceresi oldugu gibi kullanilacak.
- Units/instrument response bu fazda degistirilmeyecek; HH counts-domain oldugu gibi kullanilacak.

Manifest butunluk politikasi (frozen):

- `fail_on_manifest_drop=true` ve `max_manifest_drop_rate=0.0` kullanilir (sessiz drop kabul edilmez).
- Station listede olmayan kod varsa run fail eder (station-index fallback yok).
- External HH auditinde phase-pick eksigi oldugu dogrulanan 2 cift explicit exclusion ile sabitlenmistir:
  - `20151130230709_BALB`
  - `20151130230709_TVSB`
- Tum drop nedenleri ve ornekleri `*.drop_report.json` dosyasina yazilir.

## 4) Frozen decisions (D001-D015)

| ID | Decision | Frozen choice |
|---|---|---|
| D001 | Gorev tipi | Reconstruction + Condition-only (ana odak: condition-only) |
| D002 | Split politikasi | Event-wise split + ayri OOD event split |
| D003 | Girdi temsili | Complex STFT |
| D004 | Model ailesi | CVAE |
| D005 | Condition seti | Geometri + travel-time turevleri + station embedding |
| D006 | Normalizasyon | Train-only global z-score |
| D007 | Training loss | Complex L1 + LogMag L1 (+ KL regularization) |
| D008 | Eval metrikleri | Secili zamanlama+spektral set |
| D009 | Egitim butcesi | 180 epoch, AdamW, RLROP, early stopping |
| D010 | Checkpoint politikasi | `best_val_loss.pt` + `best_condgen_composite.pt` |
| D011 | STFT sabitleri | `256/256/32`, `hann`, onesided, Nyquist-drop, `2x128x220` |
| D012 | Condition-only latent sampling | `z~N(0,I)`, multi-sample aggregate (`K=8/32`), fixed seed bank |
| D013 | Condgen composite | pre-gate + robust-z family composite (`0.35/0.45/0.20`) |
| D014 | Onset picker parametreleri | max-derivative + confidence gate (`P±4s`, `S±6s`, `conf>=2.5`) |
| D015 | Imbalance guardrail | balanced: `M>=5 +5%`, `All <=8%`, `3<=M<5 <=10%` |

## 5) Input/Output contract

### Input

- Complex STFT hedef tensori: `X` (Z-bileseni, iki kanal: real/imag, sabit sekil `2 x 128 x 220`)
- Condition:
  - `magnitude`
  - `repi_km`
  - `depth_km`
  - `azimuth_sin`
  - `azimuth_cos`
  - `tP_ref_s`
  - `tS_ref_s`
  - `dtPS_ref_s`
  - `station_id` (embedding ile)

### Output

- Model cikisi: `X_hat` (Z-bileseni complex STFT)

### Modes

- Reconstruction mode: `X, c -> X_hat`
- Condition-only mode: `c -> X_hat` (ana rapor modu)

Condition-only latent sampling (frozen):

- `z ~ N(0, I)` ile stochastic uretim.
- Sweep/hizli eval: `K=8` ornek.
- Final rapor: `K=32` ornek.
- Tum modellerde ayni sabit seed bank kullanilir.
- Raporlama `mean ± std` seklindedir; tek-ornek cherry-pick yoktur.

## 6) Why this condition set

`lat/lon` dogrudan verilmedi. Ilk hatta fiziksel olarak daha dogrudan ve yorumlanabilir feature'lar secildi.

Travel-time feature mantigi:

- 1D hiz modeli tum bolge icin sabit olabilir
- Ham hiz profili tek basina ayrismaz
- Geometri ile birlestirilmis travel-time turevleri ornek-bazli fiziksel sinyal tasir
- Aktif profil: `Depths/Vp/Vs` knot modeli (`depth_unit=m`, `velocity_unit=m/s`), derinlikte lineer interpolasyon + uc noktalarda clamp

Bu nedenle `tP_ref_s`, `tS_ref_s`, `dtPS_ref_s` condition setine eklendi.

Referans cercevesi (frozen):

- `tP_ref_s` ve `tS_ref_s` pencere baslangicina gore tanimlanir (window-relative).
- `dtPS_ref_s` fark olarak ayni kalir.

## 7) Split policy and leakage control

Split birimi trace degil eventtir.

- Train/Val/Test event listeleri ayriktir.
- OOD event listesi ayrica ayriktir.
- Ayni eventin kayitlari farkli splitlere dagitilmaz.
- Splitler `frozen_event_splits_*.json` ile sabitlenir; ayni frozen dosya kullanildiginda OOD seti calismalar arasinda degismez.

Bu, performansin gercek genelleme davranisina daha yakin olmasini saglar.

## 7.1) Windowing/Alignment (frozen for this phase)

Bu fazda veri tekrar pencerelenmez.

- External HH setindeki mevcut pencere semantigi korunur.
- Audit (ornekleme) bulgusu:
  - 3 bilesen
  - `100 Hz`
  - ~`70 s` kayit suresi
  - `start-origin` farki degisken (sabit origin-window degil)

Sonuc:

- Model tasarimi bu mevcut pencere semantigine uyumlu olacak.
- STFT zaman-boyutu preprocess asamasinda sabitlenecek (dynamic collate yok).
- Sabitleme kurali: `right-pad / right-crop` (fallback; hedef `T_fix=220` frame).

## 7.1.1) STFT constants (frozen)

- `n_fft = 256`
- `win_length = 256`
- `hop_length = 32` (`noverlap = 224`)
- `window = hann`
- `return_onesided = True`
- `boundary = zeros`, `padded = True`

Eksensel sabitleme:

- Frekans: onesided cikista `129` bin olusur; Nyquist bin drop edilerek `128` bin kullanilir.
- Zaman: external pencere semantigi korunarak hedef `220` frame (`70s @ 100 Hz`), uyumsuzlukta `right-pad / right-crop`.

## 7.2) Filtering policy (frozen)

- External train data ikinci kez filtrelenmeyecek.
- `data/external_dataset/extracted/data/filtered_waveforms/HH` oldugu gibi kullanilacak.
- Exp001 dataset katmaninda ek waveform bandpass adimi olmayacak.

## 7.3) Units and instrument response policy (frozen)

- Signal units: `HH counts (as-is)`.
- Instrument response removal uygulanmayacak.
- Tum train/val/test/OOD degerlendirmeleri ayni unit semantiginde kalacak.

Sinirlama:

- Bu fazda mutlak fiziksel genlik yorumu (`m/s`, `m/s^2`) kapsam disidir.

## 8) Normalization policy

Train split istatistikleri tek kaynak olarak kullanilir.

- Numeric condition feature'lari: feature-wise z-score
- Complex STFT (Z-only): ortak RMS scale ile normalize edilir
  - `s_z = sqrt(mean(R^2 + I^2))` (train-only)
  - `R_norm = R / (s_z + eps)`, `I_norm = I / (s_z + eps)`
- `station_id`: normalize edilmez (embedding)

Per-frequency normalization bu fazda uygulanmaz. (Ileri asama adayi)

## 9) Model objective

CVAE objective, reconstruction + regularization dengesine dayanir.

- Reconstruction cekirdegi:
  - `L_complex`
  - `L_logmag`
  - `L_recon = lambda_c * L_complex + lambda_m * L_logmag`

- Regularization:
  - KL terimi (`q(z|x,c)` ile prior arasinda)
  - Collapse-onleme:
    - `beta` lineer warmup (`30` epoch, `beta_max=1.0`)
    - per-dim free-bits (`0.03` nats)
  - Sayisal stabilite:
    - posterior `logvar` model icinde clamp edilir: `[-8.0, 4.0]`
    - boylece `exp(logvar)` kaynakli KL patlamasi riski azaltilir

Toplam kayip:

`L_total = L_recon + beta_t * KL_fb(...)`

Not:

- Bu fazda MR-STFT training loss aktif degil.
- MR-STFT icin guvenli recipe notu protokolde sakli.

## 10) Evaluation metrics (frozen set)

Secili metrikler:

- `complex_l1`
- `mr_lsd`
- `xcorr_max`
- `xcorr_lag_s`
- `envelope_corr`
- `band_energy_ratio_error`
- `onset_mae_p_s`
- `onset_mae_s_s`
- `onset_mae_dtps_s`

Amac:

- Spektral kaliteyi olcmek
- Global ve lokal zamanlama davranisini ayirmak
- P/S zamanlama dogrulugunu dogrudan izlemek

Band metrik detaylari (frozen):

- `low_band_error`: `0.5-2 Hz`
- `high_band_error`: `8-20 Hz`
- `mid_band_error`: `2-8 Hz` (diagnostic only, secim/gate/composite disi)
- Hesap enerji-pay tabanlidir:
  - `E_B = sum_{f in B,t} |X(f,t)|^2`
  - `p_B = E_B / (E_total + eps)`
  - `err_B = |p_B_pred - p_B_true|`

## 11) Onset Delta-t measurement protocol

Onset olcumu tek tip picker ile hem gercek hem tahmin sinyallerine uygulanir.

Adimlar:

1. Complex STFT'den `|X(f,t)|` cikar.
2. Frame enerji egirisi hesapla: `E(t)=sum_f |X(f,t)|^2`.
3. Log ve yumusat:
   - `E_log(t)=log1p(E(t))`
   - moving-average `3` frame
4. Turev:
   - `dE(t)=E_s(t)-E_s(t-1)`
5. Arama pencereleri:
   - P: `tP_ref_s +/- 4.0 s`
   - S: `tS_ref_s +/- 6.0 s`
   - Ek kural: `S_start >= tP_pick + 1.0 s`
6. Pick:
   - Her pencere icinde `argmax(dE)` pick adayi.
7. Confidence:
   - `conf=(dE_peak - median(dE_win)) / (MAD(dE_win)+1e-8)`
   - Faz `failure` kosulu: `conf < 2.5`
8. Fiziksel tutarlilik:
   - `tS_pick <= tP_pick` ise `failure_s`
9. Metrikler:
   - `onset_mae_p_s`
   - `onset_mae_s_s`
   - `onset_mae_dtps_s`

Bu protokol insan-etiket bagimliligini azaltir ve model-model karsilastirmayi tutarli yapar.

### 11.1) Picker QA and manual approval gate

Onset metrikleri icin picker kalitesi iki katmanla dogrulanir:

- Otomatik QA:
  - reproducibility
  - physics consistency
  - reference agreement
  - failure behavior

- Manuel QA (zorunlu):
  - P/S pick overlay gorselleri uzman-goz ile incelenir.
  - Kullanici onayi `owner_review.md` dosyasina kaydedilir.
  - Onay olmadan onset metrikleri nihai kabul edilmez (provisional kalir).

Failure handling (frozen):

- Onset MAE metrikleri sadece evaluable ornekler uzerinden hesaplanir.
- Failure ornekleri MAE'ye zorla dahil edilmez.
- Ayrica su oranlar zorunlu raporlanir:
  - `onset_failure_rate_p`
  - `onset_failure_rate_s`
  - `onset_evaluable_rate`

Failure tanimi (frozen):

- Failure faz-bazli tanimlanir:
  - `failure_p`: P gate gecis yok
  - `failure_s`: S gate gecis yok
- `onset_mae_dtps_s` sadece her iki faz da evaluable ise hesaplanir.

## 12) Training budget and checkpoint policy

Frozen budget:

- Optimizer: AdamW
- Epoch: 180 (max)
- Batch: 64
- LR: `2e-4`
- Scheduler: ReduceLROnPlateau (`factor=0.5`, `patience=6`)
- Early stop: `min_epochs=40`, `patience=20`, `min_delta=5e-5`

Checkpoint policy:

- `best_val_loss.pt`
- `best_condgen_composite.pt`

Gecici dosyalar run sonunda temizlenir.

### 12.1) `best_condgen_composite` tanimi (frozen)

Bu skor yalnizca checkpoint secimi icindir; training loss'un parcasi degildir.

Pre-gate (zorunlu):

- `onset_evaluable_rate >= 0.70`
- `onset_failure_rate_p <= 0.30`
- `onset_failure_rate_s <= 0.35`
- `abs_xcorr_lag_s <= 2.5` (batch ortalama mutlak lag)

Kalibrasyon:

- Ilk `N_calib=6` condition-only eval sonucundan metric-wise `median` ve `MAD` hesaplanir.
- `scale_i = 1.4826 * MAD_i + 1e-8`

Robust-z yon duzeltmesi:

- Lower-better metriklerde: `z_i = (x_i - med_i) / scale_i`
- Higher-better metriklerde: `z_i = (med_i - x_i) / scale_i`
- `z_i` clip: `[-4, 4]`

Aile skorlamasi:

- `z_spec  = mean(z_complex_l1, z_mr_lsd)`
- `z_time  = mean(z_abs_xcorr_lag, z_onset_p, z_onset_s, z_onset_dtps)`
- `z_shape = mean(z_xcorr_max, z_envelope_corr, z_band_energy_ratio_error)`

Composite:

- `z_comp = 0.35*z_spec + 0.45*z_time + 0.20*z_shape`
- `lower is better`

Ornekleme ve final secim:

- Sweep/hizli eval: `K=8`
- Run sonu final rerank: `K=32`
- Gecici top-3 aday checkpoint tutulur; finalde tek `best_condgen_composite.pt` birakilir.

## 13) Run structure (strict)

Her run su agacta tutulur:

`runs/exp001/run_YYYYMMDD_HHMM_<tag>/`

Zorunlu icerik:

- `config_resolved.yaml`
- `train.log`
- `checkpoints/`
- `metrics/`
- `plots/`
- `tmp/`

## 14) Scope boundaries

Bu fazda bilerek disarida birakilanlar:

- Condition ablation
- `lat/lon` tabanli alternatif setler
- Per-frequency normalization
- MR-STFT training loss

Bu sinirlarin amaci ilk hatti temiz, izlenebilir ve tartismasiz kurmaktir.

## 14.1) Data imbalance policy (frozen)

Iki-asamali politika:

- Asama-1: Dogal dagilimla egitim, magnitude-bin bazli zorunlu raporlama
- Asama-2: Soft weighted sampler (S1) ile ek kosu

Soft sampler formu:

- Binler: `M<3`, `3<=M<5`, `M>=5`
- `w_bin = (N_med / N_bin)^alpha`
- Baslangic hiperparametre: `alpha=0.5`, `w_max=3.0` (clamp)

Model secim kurali:

- Birincil: `M>=5` bin performansi
- Koruma kapisi: `All` ve orta-bin performansi belirli limitten fazla bozulmayacak

Guardrail esikleri (frozen, D015):

- Birincil kapı: `M>=5` composite goreli iyilesme `>= 5%`
- Koruma kapilari:
  - `All` composite bozulmasi `<= 8%`
  - `3<=M<5` composite bozulmasi `<= 10%`
- Tail iyilesse bile koruma kapisi asilirsa model reject edilir.
- Asama-2 uygulamasinda referans Asama-1 by-bin condition-only sonucu `imbalance_guardrails.reference_cond_eval_json` ile verilir.

## 15) Success criteria for this phase

Bu faz basarili sayilacak eger:

- Tanimlanan iki gorev modu ayrik ve dogru raporlaniyorsa
- Condition-only metrikleri istikrali sekilde uretiliyorsa
- Onset Delta-t protokolu sorunsuz calisiyorsa
- Run yapisi ve checkpoint disiplini bozulmuyorsa

Bu dokuman, kodlama oncesi problem tanimini ve tercih gerekcelerini tek noktada sabitler.
