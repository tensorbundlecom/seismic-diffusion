# Decisions Log

## 2026-02-24

- `experiments2/` klasoru olusturuldu.
- Is akisi sifirdan, adim adim ve izlenebilir sekilde ilerleyecek.
- Dosya yapisi olarak Alternatif 1 secildi.
- Kullanilmayan alt klasorler kaldirildi; cekirdek yapi `protocol/configs/src/runs` olarak sadeletirildi.
- `decision_contract.md` eklendi; karar ve terim standardi sabitlendi.
- `experiment_001.md` zorunlu karar kimlikleri (D001-D010) ile guncellendi.
- `exp001_problem_and_design.md` eklendi; problem tanimi, domain gerekceleri ve karar ozeti tek dokumanda toplandi.

## 2026-02-25

- `D011` donduruldu: STFT sabitleri (`256/256/32`, `2x128x220`).
- `D012` donduruldu: condition-only latent sampling (`z~N(0,I)`, `K=8/32`, fixed seed bank).
- `D013` donduruldu: `best_condgen_composite` (pre-gate + robust-z aile skoru).
- `D014` donduruldu: onset picker sayisal parametreleri (`P±4s`, `S±6s`, `conf>=2.5`).
- `D015` donduruldu: imbalance guardrail esikleri (balanced).
- `exp001` core kodu yazildi (`dataset/model/train/evaluate/visualize`).
- Syntax kontrolu: `python3 -m compileall -q ML/autoencoder/experiments2/src` PASS.
- Ortam notu: mevcut shell'de `torch` ve `obspy` yok; bu nedenle runtime smoke/egitim calistirilmadi.

## 2026-02-26

- Dataset split guardrail'lari sertlestirildi:
  - `ood_policy` config'i aktif kullaniliyor (desteklenen: `latest_by_origin_time`).
  - `ood_event_ratio=0.0` durumu ayri ele alindi (OOD bos set).
  - `min_events_per_split`, `max_missing_origin_events`, `max_unassigned_samples` kontrolu eklendi.
- Split/normalization cache dogrulama guclendirildi:
  - `manifest_sha256` ve split/config metadata eslesmesi zorunlu.
- Manifest cache dogrulama guclendirildi:
  - `manifest_sha256`, `waveform_dir`, `event_catalog`, `phase_pick_dir`, `station_list_file`, `max_manifest_files` kontrolu eklendi.
  - `velocity_model_sha256` kontrolu eklendi (hiz modeli degisince manifest stale kalamaz).
- Manifest kalite/guardrail sertlestirmesi:
  - `station_idx` icin fallback kaldirildi; station listede yoksa drop reason olarak yazilir ve strict policy ile run fail eder.
  - Drop reason raporu eklendi: `*.drop_report.json` (reason count + ornekler).
  - `fail_on_manifest_drop=true`, `max_manifest_drop_rate=0.0` ile sessiz veri kaybi engellendi.
  - External datasette tespit edilen 2 adet phase-pick eksigi icin explicit exclusion listesi eklendi:
    - `20151130230709_BALB`
    - `20151130230709_TVSB`
  - Full external rebuild dogrulamasi:
    - `num_files_scanned=88921`
    - `num_rows=88919`
    - `drop_counts_by_reason={"excluded_event_station": 2}`
    - `num_rows_dropped_nonexcluded=0` (strict policy PASS)
- VAE sayisal stabilite guncellemesi:
  - Encoder posterior `logvar` cikisi model icinde clamp edildi: `[-8.0, 4.0]`.
  - Amac: `exp(logvar)` kaynakli KL patlamasi ve non-finite deger riskini minimum mudahale ile azaltmak.
- Split/OOD persistence notu:
  - `frozen_event_splits_*.json` referansi kullanildiginda OOD seti sabittir.
  - `--force-split` veya split/manifest fingerprint degisimi olursa split yeniden olusur.
- Condition-only K-sampling determinism duzeltmesi:
  - `evaluate_condition_only` icinde RNG her batch yerine her-`k` seviyesinde olusturulur.
  - Boylece ayni `k` icinde tum batchler tek deterministik rastgele akisi paylasir (batch-basi tekrar eden z deseni engellenir).
- Evaluation feature-index guvenligi:
  - `evaluate._batch_metrics` icindeki `magnitude/tP_ref_s/tS_ref_s` indeksleri sabit sayi yerine `numeric_feature_order` uzerinden dinamik bulunur.
  - Boylece condition sirasi degisirse metrik hesabi sessizce bozulmaz, acik hata verir.
- Datetime normalizasyonu guclendirildi:
  - timezone-aware zamanlar UTC naive'e normalize ediliyor.
- 1D hiz modeli guncellendi:
  - Placeholder katmanli model yerine kullanici-saglanan `Depths/Vp/Vs` knot profili kullaniliyor.
  - Birimler: `depth_unit=m`, `velocity_unit=m/s`.
  - Derinlikte lineer interpolasyon + uc noktalarda clamp uygulanir.

## 2026-02-27

- `protocol/` klasoru duzenlendi:
  - `docs/` (karar/spec dokumanlari)
  - `frozen/base/` (full artifactler)
  - `frozen/smoke/` (smoke artifactler)
  - `reports/` (audit/sanity ciktilari)
- `exp001_base.json` ve `exp001_smoke.json` artifact yollari yeni frozen dizinlerine tasindi.
- Smoke train/eval ile yeni dizin yapisi dogrulandi (layout check PASS).

## Decision Register

### D001

- `ID`: D001
- `Question`: Gorev tipi ne olacak?
- `Options`: A=`reconstruction-only`, B=`condition-only generation`, C=`ikisi de ayrik rapor`
- `Chosen`: C (ana odak B)
- `Why`: Reconstruction ve condition-only generation ayni degildir. Ikisini ayri skorlamak zorunlu. Ana urun hedefi condition-only generation oldugu icin odak B.
- `Impact`: Eval pipeline iki moda ayrilacak. Tum raporlar `reconstruction` ve `condition_only` olarak ayri tablo verecek.
- `Status`: Frozen
- `Date`: 2026-02-24
- `Note`: Condition-only metriklerinde zaman kaymasi etkisi kritik. Bu konu D008'de ayrica dondurulecek.

### D002

- `ID`: D002
- `Question`: Veri split politikasi ne olacak?
- `Options`: A=`random trace split`, B=`event-wise split + ayri OOD event split`, C=`event-wise + station-heldout OOD`
- `Chosen`: B
- `Why`: Random trace split leakage riski tasir. Event-wise split ile ayni deprem train/test karismaz. Ayrica ayri OOD event seti ile gercek genelleme olculur.
- `Impact`: Split birimi trace degil event olacak. Tum train/val/test ve OOD listeleri event-id tabanli dondurulecek.
- `Status`: Frozen
- `Date`: 2026-02-24

### D003

- `ID`: D003
- `Question`: Girdi temsil formu ne olacak?
- `Options`: A=`waveform`, B=`magnitude STFT`, C=`complex STFT`
- `Chosen`: C
- `Why`: Faz bilgisini korumak ve P/S zamanlama yapisini daha guvenilir tasimak icin complex STFT secildi.
- `Impact`: Veri pipeline complex temsil (real+imag veya esdegeri) uretecek. Loss ve metrikler complex/STFT odakli olacak.
- `Status`: Frozen
- `Date`: 2026-02-24
- `Note`: Exact complex STFT sabitleri `D011` ile donduruldu.

### D011

- `ID`: D011
- `Question`: STFT sabitleri ve frame semantigi nasil dondurulecek?
- `Options`: A=`legacy uyum (256/64, 129x111)`, B=`paper-uyumlu zaman cozunurlugu + external pencere koruma (256/32, 128x220)`, C=`paper-strict truncate 4064 (256/32, 128x128)`
- `Chosen`: B
- `Why`: P/S zamanlama metrikleri icin daha iyi zaman cozunurlugu (`hop=32`) saglarken external datasetteki mevcut pencere semantigini bozmadan ilerler. Paper ile frekans/zaman olcegi uyumlu kalir, ancak 4064 truncate ile bilgi kaybi olusmaz.
- `Impact`:
  - STFT:
    - `n_fft = 256`
    - `win_length = 256`
    - `hop_length = 32` (`noverlap = 224`)
    - `window = hann`
    - `return_onesided = True`
    - `boundary = zeros`, `padded = True`
  - Frekans ekseni:
    - Onesided cikista `129` bin olusur; Nyquist bin drop edilerek `128` bin kullanilir.
  - Zaman ekseni:
    - External pencere semantigi korunur.
    - `T_fix = 220` frame hedeflenir (`70 s @ 100 Hz`, `hop=32`).
    - Uyumsuzlukta `right-pad / right-crop` fallback uygulanir.
  - Model girisi:
    - Z-only complex temsil: `2 x 128 x 220` (`real, imag`).
- `Status`: Frozen
- `Date`: 2026-02-25

### D012

- `ID`: D012
- `Question`: Condition-only generation modunda latent `z` nasil secilecek?
- `Options`: A=`deterministik z=0`, B=`tek stochastic ornek (z~N(0,I))`, C=`coklu stochastic ornek + aggregate`, D=`sicaklik (tau) tuning`
- `Chosen`: C
- `Why`: CVAE condition-only uretim dogasi geregi stochastic'tir. Tek ornek metrikleri yuksek varyansli yapar; coklu ornek + aggregate daha adil ve daha stabil model karsilastirmasi saglar.
- `Impact`:
  - Inference dagilimi: `z ~ N(0, I)` (`tau=1.0`, sabit)
  - Her condition kaydi icin coklu ornekleme uygulanir:
    - Sweep/fast eval: `K=8`
    - Final eval/report: `K=32`
  - Reproducibility:
    - Sabit seed bank kullanilir (run config'e kaydedilir).
    - Tum modeller ayni seed bank ile degerlendirilir.
  - Raporlama:
    - Metrikler `mean ± std` olarak verilir.
    - Tek-iyi-ornek secimi (cherry-pick) yasak.
  - Not:
    - Bu fazda `tau` tuning yapilmaz; `tau=1.0` frozen.
- `Status`: Frozen
- `Date`: 2026-02-25

### D013

- `ID`: D013
- `Question`: `best_condgen_composite` skoru nasil tanimlanacak?
- `Options`: A=`ham weighted-sum`, B=`min-max weighted-sum`, C=`rank-based`, D=`gate + robust z composite`
- `Chosen`: D
- `Why`: Metrikler farkli olcek ve farkli yonlerde. Ham/min-max toplamlarda olcek-dominasyonu riski yuksek. Gate + robust-z yaklasimi daha stabil, outlier'a daha dayanikli ve model karsilastirmasi icin daha adil.
- `Impact`:
  - Kullanim alani:
    - Yalniz `best_condgen_composite.pt` secimi icin kullanilir.
    - Training loss/gradient icinde kullanilmaz.
  - Pre-gate (zorunlu):
    - `onset_evaluable_rate >= 0.70`
    - `onset_failure_rate_p <= 0.30`
    - `onset_failure_rate_s <= 0.35`
    - `abs_xcorr_lag_s <= 2.5` (batch ortalama mutlak lag)
    - Gate gecmeyen epoch `best_condgen` adayi olamaz.
  - Kalibrasyon:
    - Ilk `N_calib=6` condition-only eval sonucundan metric-wise `median` ve `MAD` hesaplanir.
    - `scale_i = 1.4826 * MAD_i + 1e-8`
  - Yonu normalize edilmis robust z:
    - Lower-better metrikler:
      - `complex_l1`, `mr_lsd`, `abs_xcorr_lag_s`, `band_energy_ratio_error`, `onset_mae_p_s`, `onset_mae_s_s`, `onset_mae_dtps_s`
      - `z_i = (x_i - med_i) / scale_i`
    - Higher-better metrikler:
      - `xcorr_max`, `envelope_corr`
      - `z_i = (med_i - x_i) / scale_i`
    - `z_i` clip: `[-4, 4]`
  - Aile skorlamasi:
    - `z_spec  = mean(z_complex_l1, z_mr_lsd)`
    - `z_time  = mean(z_abs_xcorr_lag, z_onset_p, z_onset_s, z_onset_dtps)`
    - `z_shape = mean(z_xcorr_max, z_envelope_corr, z_band_energy_ratio_error)`
  - Composite:
    - `z_comp = 0.35*z_spec + 0.45*z_time + 0.20*z_shape`
    - `lower is better`
  - Degerlendirme ornekleme:
    - Sweep/hizli: `K=8` (D012 ile uyumlu)
    - Run sonu final rerank: `K=32`
    - Top-3 aday checkpoint gecici tutulur; finalde tek `best_condgen_composite.pt` birakilir.
- `Status`: Frozen
- `Date`: 2026-02-25

### D014

- `ID`: D014
- `Question`: Onset picker sayisal parametreleri ve fail kurallari nasil dondurulecek?
- `Options`: A=`threshold+slope`, B=`max-derivative + confidence gate`, C=`STA/LTA`
- `Chosen`: B
- `Why`: Daha az hiperparametre ile daha stabil ve tekrar-uretilebilir onset olcumu verir. STFT-enerji egirisi uzerinde olcek degisimlerine daha dayaniklidir.
- `Impact`:
  - Giris serisi:
    - `E(t) = sum_f |X(f,t)|^2`
    - `E_log(t) = log1p(E(t))`
    - Yumusatma: moving-average `3` frame
    - Turev: `dE(t) = E_s(t) - E_s(t-1)`
  - Arama pencereleri:
    - P: `tP_ref_s +/- 4.0 s`
    - S: `tS_ref_s +/- 6.0 s`
    - Ek kural: `S_start >= tP_pick + 1.0 s`
  - Pick kurali:
    - Pencere icinde `argmax(dE)` noktasi pick adayi.
  - Confidence:
    - `conf = (dE_peak - median(dE_win)) / (MAD(dE_win) + 1e-8)`
    - Faz `failure` kosulu: `conf < 2.5`
  - Fiziksel tutarlilik:
    - `tS_pick <= tP_pick` ise `failure_s`.
  - Metrik etkisi:
    - `onset_mae_p_s`, `onset_mae_s_s`, `onset_mae_dtps_s` D008/P2/P3 kurallariyla ayni sekilde hesaplanir.
    - Failure oranlari zorunlu raporlanir.
- `Status`: Frozen
- `Date`: 2026-02-25

### D015

- `ID`: D015
- `Question`: I3 kapsaminda magnitude-bin seciminde guardrail esikleri ne olacak?
- `Options`: A=`strict`, B=`balanced`, C=`lenient`
- `Chosen`: B
- `Why`: Tail (`M>=5`) iyilesmesini onceliklendirirken global dagilimda asiri bozulmayi engelleyen dengeli bir kural seti saglar.
- `Impact`:
  - Karsilastirma tabani:
    - Asama-2 (soft sampler) modeli, Asama-1 (dogal dagilim) baseline'ina gore degerlendirilir.
    - Pratikte bu baseline, `imbalance_guardrails.reference_cond_eval_json` ile verilir (`condition_only.by_bin_mean` veya esdeger by-bin cikti).
  - Birincil kapı (tail kazanci):
    - `M>=5` kritik metrik composite'inde goreli iyilesme `>= 5%` olmalidir.
  - Guardrail kapisi:
    - `All` composite bozulmasi `<= 8%`
    - `3<=M<5` composite bozulmasi `<= 10%`
  - Composite yonu:
    - Tum karsilastirmalar lower-is-better composite uzerinden yapilir (`D013` ile uyumlu normalize edilmis skorlar).
  - Fail kosulu:
    - Tail iyilesmesi var olsa bile guardrail asilirsa model "reject" edilir.
- `Status`: Frozen
- `Date`: 2026-02-25

### D004

- `ID`: D004
- `Question`: Model ailesi ne olacak?
- `Options`: A=`AE`, B=`VAE`, C=`CVAE`
- `Chosen`: C
- `Why`: Ana odak condition-only generation oldugu icin condition bilgisini dogrudan kullanan aile secildi.
- `Impact`: Encoder/decoder condition alacak; inference yolunda condition->sample akisi zorunlu olacak.
- `Status`: Frozen
- `Date`: 2026-02-24

### D005

- `ID`: D005
- `Question`: Condition degisken seti ne olacak?
- `Options`: A=`temel geometri`, B=`temel geometri + 1D-hiz-modelinden turetilmis travel-time feature'lari`, C=`B + event/station lat-lon`
- `Chosen`: B
- `Why`: Koordinatlari modele dolayli cozdurmek yerine fiziksel olarak dogrudan etkili feature'lar kullanildi. 1D hiz modeli sabit oldugu icin ham profil degil, ornek-bazli travel-time turevleri secildi.
- `Impact`: Condition vektoru asagidaki sabit setten olusacak:
  - `magnitude`
  - `repi_km`
  - `depth_km`
  - `azimuth_sin`
  - `azimuth_cos`
  - `tP_ref_s` (1D hiz modelinden)
  - `tS_ref_s` (1D hiz modelinden)
  - `dtPS_ref_s = tS_ref_s - tP_ref_s`
  - `station_id` (embedding olarak)
- `Status`: Frozen
- `Date`: 2026-02-24
- `Note`: Bu asamada ablation yapilmayacak. Lat-lon ve alternatif condition setleri bu deney kapsaminda degil.

### D006

- `ID`: D006
- `Question`: Normalizasyon politikasi ne olacak?
- `Options`: A=`per-trace`, B=`train-only global z-score`, C=`train-only per-frequency z-score`
- `Chosen`: B
- `Why`: Ilk hat icin en dusuk karmasiklik ve en yuksek izlenebilirlik. Fiziksel enerji dagilimini asiri duzlestirmeden stabil egitim saglar.
- `Impact`:
  - Numeric condition feature'lari train split uzerinden feature-wise z-score
  - Complex STFT normalizasyonu `B1/B2` kararlarina gore ortak RMS scale ile uygulanir
  - `station_id` embedding normalizasyon disi
- `Status`: Frozen
- `Date`: 2026-02-24
- `Note`: `C` (per-frequency z-score) ileri asama adayi olarak not edildi; bu deneyde uygulanmayacak.

### D007

- `ID`: D007
- `Question`: Loss fonksiyonu ne olacak?
- `Options`: A=`complex L1`, B=`complex L1 + log-magnitude L1`, C=`B + multi-resolution STFT (MR-STFT)`
- `Chosen`: B
- `Why`: Ilk hat icin en dusuk karmasiklik ile faz+genlik tutarliligini korur. Ekstra loss dominasyonu ve agir tuning riskini azaltir.
- `Impact`:
  - Ana recon loss: `L_recon = lambda_c * L_complex + lambda_m * L_logmag`
  - Baslangic agirliklari: `lambda_c=1.0`, `lambda_m=1.0`
  - VAE regularizasyonu D004 kapsaminda KL terimi ile eklenecek (detayi D007-not altinda)
- `Status`: Frozen
- `Date`: 2026-02-24
- `Note (Future-C Recipe)`: MR-STFT sonraki asamada denenebilir ama bu deneyde kapali:
  - Form: `L_total = L_B + lambda_mr * L_MR`
  - Guvenli baslangic: `lambda_mr=0.05`
  - Warmup: ilk `%20` epoch `lambda_mr=0`, sonra lineer artir
  - Dominasyon kontrolu: `||grad(L_MR)|| / ||grad(L_B)||` hedef `0.1-0.3`
  - Alarm: `L_MR` duserken `L_B` bozulursa `lambda_mr` azalt
  - Model seciminde MR metrikleri ikincil; ana kapilar B-metrikleri + OOD

### D008

- `ID`: D008
- `Question`: Metrik seti ne olacak?
- `Options`: A=`genel STFT metrikleri`, B=`zaman kaymasina dayanikli secili set`, C=`genis metrik havuzu`
- `Chosen`: B
- `Why`: Ana odak condition-only oldugu icin hem spektral kaliteyi hem zaman kaymasi etkisini ayri yakalayan, az ama ayristirici bir set secildi.
- `Impact`: Raporlarda su metrikler zorunlu:
  - `complex_l1`
  - `mr_lsd`
  - `xcorr_max`
  - `xcorr_lag_s`
  - `envelope_corr`
  - `band_energy_ratio_error`
  - `onset_mae_p_s`
  - `onset_mae_s_s`
  - `onset_mae_dtps_s`
- `Status`: Frozen
- `Date`: 2026-02-24
- `Note (Onset Delta-t Measurement Protocol)`:
  - Giris sinyali: complex STFT'den elde edilen `|X(f,t)|`
  - Zaman serisi: `E(t)=sum_f |X(f,t)|^2` (frame-enerji egirisi)
  - Yumusatma: kisa hareketli ortalama (sabit pencere)
  - Arama pencereleri:
    - P icin: `tP_ref_s +/- wP`
    - S icin: `tS_ref_s +/- wS`
    - `tP_ref_s` ve `tS_ref_s` D005'teki 1D model turevi referanslardir
  - Pick kurali:
    - Ilk tercih: pencere icinde esik gecisi + pozitif egim kosulu
    - Yedek tercih: ayni pencerede maksimum pozitif turev noktasi
  - Metrikler:
    - `onset_mae_p_s = mean(|tP_pred - tP_true|)`
    - `onset_mae_s_s = mean(|tS_pred - tS_true|)`
    - `onset_mae_dtps_s = mean(|(tS_pred-tP_pred) - (tS_true-tP_true)|)`
  - Not: `*_true` ayni picker ile ground-truth sinyalden uretilir; boylece insan-etiket bagimliligi azalir.
  - Exact picker parametreleri `D014` ile dondurulmustur.

### D009

- `ID`: D009
- `Question`: Egitim butcesi ve early stopping politikasi ne olacak?
- `Options`: A=`hizli`, B=`dengeli`, C=`agir`
- `Chosen`: C
- `Why`: Daha derin optimizasyon icin daha uzun egitim penceresi secildi.
- `Impact`:
  - `optimizer = AdamW`
  - `max_epochs = 180`
  - `batch_size = 64`
  - `learning_rate = 2e-4`
  - `lr_scheduler = ReduceLROnPlateau(factor=0.5, patience=6)`
  - `early_stopping.min_epochs = 40`
  - `early_stopping.patience = 20`
  - `early_stopping.min_delta = 5e-5`
- `Status`: Frozen
- `Date`: 2026-02-24

### D010

- `ID`: D010
- `Question`: Checkpoint secim kriteri ne olacak?
- `Options`: A=`best val_total_loss`, B=`best condition-only composite`, C=`ikisi de`
- `Chosen`: C
- `Why`: D001 geregi iki gorev ayri raporlanacak; ana odak condition-only olsa da reconstruction takibi korunmali.
- `Impact`:
  - Her run icin yalniz iki resmi checkpoint yazilacak:
    - `best_val_loss.pt`
    - `best_condgen_composite.pt`
  - Diger epoch checkpointleri varsayilan olarak kapali tutulacak.
  - Run klasoru duzeni sabit:
    - `runs/exp001/run_YYYYMMDD_HHMM_<tag>/config_resolved.yaml`
    - `runs/exp001/run_YYYYMMDD_HHMM_<tag>/train.log`
    - `runs/exp001/run_YYYYMMDD_HHMM_<tag>/checkpoints/`
    - `runs/exp001/run_YYYYMMDD_HHMM_<tag>/metrics/`
    - `runs/exp001/run_YYYYMMDD_HHMM_<tag>/plots/`
    - `runs/exp001/run_YYYYMMDD_HHMM_<tag>/tmp/`
- `Status`: Frozen
- `Date`: 2026-02-24
- `Note`: Orta-katman/gecici dosyalar `run_dir/tmp/` altinda tutulur; top-k disindakiler run icinde temizlenir. `best_condgen_composite` tanimi `D013` ile sabitlenmistir.

### A0

- `ID`: A0
- `Question`: Temporal alignment ve preprocessing uyumu icin temel veri kaynagi ne olacak?
- `Options`: A=`data/filtered_waveforms/HH`, B=`data/filtered_waveforms_broadband/HH`, C=`data/external_dataset/extracted/data/filtered_waveforms/HH`
- `Chosen`: C
- `Why`: Bu fazda external dataset ile ilerleme karari verildi. Tum sonraki pencereleme/hizalama kararlarinda referans bu kaynak olacak.
- `Impact`: Experiment001 train/val/test ve OOD hazirliginda external HH seti esas alinacak. Pipeline farkliliklari bu kaynaga gore dondurulecek.
- `Status`: Frozen
- `Date`: 2026-02-24

### A1

- `ID`: A1
- `Question`: Temporal alignment/windowing nasil belirlenecek?
- `Options`: A=`origin-aligned fixed window`, B=`P-aligned window`, C=`external dataset penceresini oldugu gibi kullan`
- `Chosen`: C
- `Why`: Bu fazda veri uyumlulugu oncelikli. Mevcut external HH kayitlari yeniden pencerelenmeyecek.
- `Impact`:
  - Windowing kurali external datasetten aynen miras alinacak.
  - Train/val/test/OOD icin ayni pencereleme semantigi korunacak.
  - Ek yeniden kesme/hizalama adimi eklenmeyecek.
- `Status`: Frozen
- `Date`: 2026-02-24
- `Evidence (sample audit)`:
  - `n=2000` dosya ornegi
  - 3 bilesen sabit
  - `fs=100 Hz` sabit
  - sure ~`70 s` sabit
  - `start - event_origin` dagilimi degisken (`~ -8.7 s` ile `~ +12.5 s`)

### A2

- `ID`: A2
- `Question`: Travel-time condition feature'lari hangi referansla verilecek?
- `Options`: A=`origin-relative`, B=`window-relative`, C=`dtPS-only`
- `Chosen`: B
- `Why`: External datasette pencere baslangici origin'e sabit degil. Bu nedenle modelin girdisiyle tutarli referans, pencere-ici zamandir.
- `Impact`:
  - D005 icindeki travel-time feature'lari su sekilde yorumlanacak:
    - `tP_ref_s`: pencere baslangicina gore referans P varisi
    - `tS_ref_s`: pencere baslangicina gore referans S varisi
    - `dtPS_ref_s = tS_ref_s - tP_ref_s`
  - Onset metrik pencereleri de ayni referansta kurulacak.
- `Status`: Frozen
- `Date`: 2026-02-24

### B1

- `ID`: B1
- `Question`: Complex STFT normalizasyonunda real/imag nasil olceklenecek?
- `Options`: A=`real/imag ayri mean-std`, B=`real/imag ortak scale`, C=`magnitude normalize + phase untouched`
- `Chosen`: B
- `Why`: Complex geometriyi korurken stabil egitim saglar. Faz yapisini kanal-bazli asimetrik olcekle bozma riskini azaltir.
- `Impact`: Complex tensorde real/imag cifti ayni olcek katsayisi ile normalize edilecek.
- `Status`: Frozen
- `Date`: 2026-02-24

### C0

- `ID`: C0
- `Question`: Bilesen stratejisi ne olacak?
- `Options`: A=`ENZ (3 bilesen)`, B=`Z-only`
- `Chosen`: B
- `Why`: Ilk fazda karmasayi azaltmak, pipeline'i yalinlastirmak ve metrik yorumunu tek-kanalda sabitlemek icin Z-only secildi.
- `Impact`:
  - Giris/ cikis tek bilesen (Z) uzerinden kurulacak.
  - Onset metrikleri tek-kanal uzerinden olculecek.
  - ENZ cok-kanal protokolu bu faz kapsaminda degil.
- `Status`: Frozen
- `Date`: 2026-02-24

### B2

- `ID`: B2
- `Question`: Z-only complex normalizasyonda ortak scale nasil hesaplanacak?
- `Options`: A=`global RMS scale`, B=`global std scale`, C=`P95 magnitude scale`, D=`per-frequency scale`
- `Chosen`: A
- `Why`: Ilk faz icin en yaln, stabil ve complex geometriyi koruyan cozum.
- `Impact`:
  - Train split uzerinden tek katsayi hesaplanir:
    - `s_z = sqrt(mean(R^2 + I^2))`
  - Normalizasyon:
    - `R_norm = R / (s_z + eps)`
    - `I_norm = I / (s_z + eps)`
  - Ayni `s_z` train/val/test/OOD/inference icin kullanilir.
- `Status`: Frozen
- `Date`: 2026-02-24

### C1

- `ID`: C1
- `Question`: STFT time-frame uzunlugu sabitlenecek mi?
- `Options`: A=`dynamic pad (collate-time)`, B=`preprocess asamasinda sabit frame`
- `Chosen`: B
- `Why`: Decoder cikis boyutunu sabitlemek, condition-only uretimde shape karmasasini onlemek ve metrikleri tutarli hesaplamak icin.
- `Impact`: Tum ornekler preprocess asamasinda ayni STFT zaman-boyutuna getirilecek.
- `Status`: Frozen
- `Date`: 2026-02-24

### C2

- `ID`: C2
- `Question`: Sabit STFT frame'e getirme yontemi ne olacak?
- `Options`: A=`right-pad / right-crop`, B=`center-pad / center-crop`, C=`left-pad / left-crop`
- `Chosen`: A
- `Why`: Window-relative zaman referansini (tP/tS) koruyan en tutarli yontem.
- `Impact`:
  - `T < T_fix` ise sona pad eklenir.
  - `T > T_fix` ise sondan crop edilir.
  - Baslangic frame referansi korunur.
- `Status`: Frozen
- `Date`: 2026-02-24
- `Note`: External HH audit'inde standart durumda `T_real = T_fix` (70s@100Hz -> STFT 111 frame). Bu kural guvenlik fallback'i olarak kalir.

### K1

- `ID`: K1
- `Question`: KL agirliklama ve collapse-onleme stratejisi ne olacak?
- `Options`: A=`sabit beta`, B=`linear annealing`, C=`linear annealing + free-bits`
- `Chosen`: C
- `Why`: Condition-only odakli CVAE'de posterior collapse riskini azaltmak icin beta schedule ve per-dim KL tabani birlikte secildi.
- `Impact`:
  - `beta_max = 1.0`
  - `warmup_epoch = 30` (lineer artisim: `beta_t = min(1.0, epoch/30)`)
  - `free_bits_per_dim = 0.03` (nats)
  - KL uygulamasi: per-dim clamp
    - `KL_fb = mean_i(max(KL_i, 0.03))`
  - Toplam kayip:
    - `L_total = L_recon + beta_t * KL_fb`
- `Status`: Frozen
- `Date`: 2026-02-24

### P1

- `ID`: P1
- `Question`: Onset picker guvenilirligi nasil dogrulanacak?
- `Options`: A=`sadece otomatik metrik`, B=`otomatik + manuel goz kontrolu`, C=`sadece manuel`
- `Chosen`: B
- `Why`: Otomatik kalite kontrolleri zorunlu olacak; buna ek olarak uzman-goz onayi ile hatali picker davranisi erken yakalanacak.
- `Impact`:
  - Otomatik picker QA zorunlu:
    - reproducibility kontrolu
    - physics consistency (`tS>tP`, dtPS araligi)
    - reference agreement (1D-ref ile dagilim)
    - failure behavior (dusuk enerji/sahte pick kontrolu)
  - Manuel kontrol zorunlu:
    - Her run icin secilen ornek setinde P/S pick overlay gorselleri uretilir.
    - `owner_review.md` dosyasinda kullanici onayi kaydi olmadan picker "approved" sayilmaz.
    - Onaysiz durumda onset metrikleri "provisional" etiketlenir.
- `Status`: Frozen
- `Date`: 2026-02-24

### P2

- `ID`: P2
- `Question`: Onset metriği hesaplanamayan (failure) ornekler nasil ele alinacak?
- `Options`: A=`MAE'ye dahil et`, B=`MAE'den cikar ve ayri failure raporla`
- `Chosen`: B
- `Why`: Picker failure durumlarini onset hata metriğine karistirmamak ve raporu daha yorumlanabilir yapmak icin.
- `Impact`:
  - `onset_mae_*` metrikleri sadece evaluable orneklerde hesaplanacak.
  - Ayrica zorunlu olarak raporlanacak:
    - `onset_failure_rate_p`
    - `onset_failure_rate_s`
    - `onset_evaluable_rate`
- `Status`: Frozen
- `Date`: 2026-02-24

### P3

- `ID`: P3
- `Question`: Failure tanimi nasil olacak?
- `Options`: A=`tek ortak failure`, B=`P ve S icin faz-bazli failure`
- `Chosen`: B
- `Why`: P ve S'nin sinyal karakteri farkli oldugu icin failure takibi faz-bazli yapilmali.
- `Impact`:
  - `failure_p`: P gate gecilmezse
  - `failure_s`: S gate gecilmezse
  - `onset_mae_p_s` sadece `not failure_p` orneklerinde
  - `onset_mae_s_s` sadece `not failure_s` orneklerinde
  - `onset_mae_dtps_s` sadece `not failure_p and not failure_s` orneklerinde
- `Status`: Frozen
- `Date`: 2026-02-24

### I1

- `ID`: I1
- `Question`: Data imbalance handling stratejisi ne olacak?
- `Options`: A=`dogal dagilim + bin-wise rapor`, B=`yalniz weighted sampler`, C=`iki-asamali (A sonra soft-weighted S1)`
- `Chosen`: C
- `Why`: Once tarafsiz baseline gorulup, gerekli ise kontrollu soft dengeleme uygulanacak. Boylece hem izlenebilirlik hem tail iyilestirme korunur.
- `Impact`:
  - Asama-1: dogal dagilim ile egitim + zorunlu magnitude-bin rapor
  - Asama-2: soft weighted sampler ile ek run (S1)
  - Karsilastirma ayni metrik seti ile yapilacak.
- `Status`: Frozen
- `Date`: 2026-02-24

### I2

- `ID`: I2
- `Question`: Soft weighted sampler formulu ne olacak?
- `Options`: A=`w=1/Nbin`, B=`w=(Nmed/Nbin)^alpha + clamp`, C=`tam balanced batch`
- `Chosen`: B
- `Why`: Ham ters-frekans (A) cok agresif olabilir. B secenegi nadir binleri guclendirirken veri priorini tamamen bozmaz.
- `Impact`:
  - Binler: `M<3`, `3<=M<5`, `M>=5` (kritik bin)
  - Agirlik: `w_bin = (N_med / N_bin)^alpha`
  - Baslangic: `alpha=0.5`, `w_max=3.0` (clamp)
  - Sampler: `WeightedRandomSampler`
- `Status`: Frozen
- `Date`: 2026-02-24

### I3

- `ID`: I3
- `Question`: Model secim kapisi nasil olacak?
- `Options`: A=`yalniz global`, B=`yalniz M>=5`, C=`M>=5 birincil + global koruma`
- `Chosen`: C
- `Why`: Kritik deger M>=5 bininde oldugu icin birincil kapidir; ancak modelin genel dagilimda cok bozulmasi engellenmelidir.
- `Impact`:
  - Birincil secim: `M>=5` bin metrikleri
  - Ikinci kapı: `All` ve `3<=M<5` metriklerinde izin verilen dusus limiti asilmayacak.
  - Rapor tablosu zorunlu: `All`, `M<3`, `3<=M<5`, `M>=5`.
- `Status`: Frozen
- `Date`: 2026-02-24

### F1

- `ID`: F1
- `Question`: External train datasina ek waveform filtresi uygulanacak mi?
- `Options`: A=`evet, ek bandpass`, B=`hayir, external filtered data oldugu gibi`
- `Chosen`: B
- `Why`: Data uyumu ve tekrar-uretilebilirlik icin external train set oldugu gibi kullanilacak; ikinci bir filtre adimi eklenmeyecek.
- `Impact`:
  - `data/external_dataset/extracted/data/filtered_waveforms/HH` dogrudan kullanilacak.
  - Dataset katmaninda ek bandpass uygulanmayacak.
  - Train/val/test/OOD arasinda ayni politika korunacak.
- `Status`: Frozen
- `Date`: 2026-02-24

### F2

- `ID`: F2
- `Question`: Low/High frekans metrik bantlari nasil tanimlanacak?
- `Options`: A=`low(0.5-2), mid(2-8), high(8-20)`, B=`low(0.5-1), high(10-25)`, C=`low(1-3), high(6-15)`
- `Chosen`: A
- `Why`: Sismolojik olarak yorumlanabilir ayrim ve metrik stabilitesi arasinda en dengeli secenek.
- `Impact`:
  - `low_band_error`: `0.5-2 Hz`
  - `high_band_error`: `8-20 Hz`
  - `mid_band_error (2-8 Hz)`: sadece diagnostic, model secim kapisina dahil degil
  - Enerji pay tabanli hesap:
    - `E_B = sum_{f in B,t} |X(f,t)|^2`
    - `p_B = E_B / (E_total + eps)`
    - `err_B = |p_B_pred - p_B_true|`
- `Status`: Frozen
- `Date`: 2026-02-24

### U1

- `ID`: U1
- `Question`: Units/instrument response politikasi ne olacak?
- `Options`: A=`HH counts (as-is)`, B=`response-removed velocity (m/s)`, C=`response-removed acceleration (m/s^2)`
- `Chosen`: A
- `Why`: Exp001 fazinda veri uyumu ve operasyonel sadelik oncelikli. Mevcut external HH veri oldugu gibi kullanilacak.
- `Impact`:
  - Signal units: `counts-domain` (HH as-is)
  - Response removal uygulanmayacak.
  - Tum train/val/test/OOD degerlendirmeleri ayni unit semantiginde kalacak.
- `Status`: Frozen
- `Date`: 2026-02-24
- `Limitation`: Mutlak fiziksel genlik yorumu (m/s, m/s^2) bu fazda kapsanmaz.
