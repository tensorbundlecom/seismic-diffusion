# NonDiagonalRigid - Decision Log

Bu dosya, tartisma sorularina verilen kararlarin resmi kaydidir.  
Her karar sabitlendikten sonra sonraki deneylerde degistirilmeyecek.

---

## Q1 - Bu Calisma Standart Verimlilestirme mi, Yenilik mi?

### Karar

- **Cekirdek iddia bir `problem-specific methodological policy` iddiasidir.**
- **Tek basina "model kucult + kaliteyi kontrol et" kismi standarttir.**
- **Yenilik iddiasi, OffDiagonal/ODI'yi karar mekanizmasina baglayan ve Q61-Q64 ile dogrulanan policy katmanindadir.**

### Kapsam Siniri

1. Bu karar `VAE genel teorisi` iddiasi degildir.
2. Bu karar, `conditional seismic generation (STFT domain)` problem ailesi icin gecerlidir.
3. Universal "offdiag sifir olmali" teoremi kurulmaz; policy faydasi sinanir.

---

## Q2 - Iddia Teorik mi, Pratik Policy mi?

### Karar

- **Iddia: pratik ve test edilebilir bir policy.**
- Teorik genelleme seviyesi: **sinirli**.

### Formulasyon

- "Sabit fairness + sabit kalite toleransi altinda, OffDiagonal/ODI-temelli secim policy'si daha kucuk modeli kaliteyi koruyarak secebilir."

---

## Q3 - Birincil Cikti Hedefi Nedir?

### Karar

- **Birincil hedef: conditional generation kalitesi (`c -> x_hat`)**
- Cikti uzayi: **modelin dogrudan urettigi STFT/log-magnitude uzayi**

### Not

- Waveform tarafi ikincil diagnostic olarak kalir.

---

## Q4 - Basari Tek Metrik mi, Composite mi?

### Karar

- **Basari birincil olarak composite quality score ile olculecek.**
- Composite, Q58'de sabitlenen 6 primary STFT metrigin normalize birlesimidir.
- `ODI` composite kalite skoruna dahil edilmez; **gating/selection** icin ayri kullanilir.

---

## Q5 - Kesif vs Dogrulama Ayrimi Nasil?

### Karar

1. **Kesif (exploratory):**
   - `NonDiagonel` altindaki onceki taramalar, esik sezgileri, ilk bulgular.
2. **Dogrulama (confirmatory):**
   - `NonDiagonalRigid` altinda pre-registered kararlar (Q55-Q64 + Q61-Q64),
   - sabit fairness ve sabit stop policy ile tekrar edilen runlar.

### Raporlama Kurali

- Confirmatory raporda exploratory bulgular "motivasyon" olarak kullanilir, "nihai kanit" olarak kullanilmaz.

---

## Q6 - Ana Degerlendirme Senaryosu Hangisi?

### Karar

- **Ana senaryo: conditional generation (`c -> x_hat`).**
- Ayrintili gerekce Q55 ile ortaktir.

---

## Q7 - Reconstruction Testi Zorunlu mu?

### Karar

- **Evet, zorunlu (secondary/gating).**
- Amac: posterior bagimlilik ve latent diagnostiklerini dogru zeminde olcmek.

---

## Q8 - Conditional Generation Testi Zorunlu mu?

### Karar

- **Evet, zorunlu (primary ranking).**

---

## Q9 - Birincil Karar Reconstruction mi Generation mi?

### Karar

- **Birincil karar generation uzerinden verilir.**
- Reconstruction sonuclari yorumlayici/gating katmanidir.

---

## Q10 - Inference Protocolunde Hangi `z` Kullanilacak?

### Karar

1. **Primary generation protocol (deterministic):**
   - `z = mu_p(c)` (prior mean)
2. **Secondary stochastic protocol:**
   - `z ~ p(z|c)` ile coklu ornekleme
3. Reconstruction tarafinda:
   - `z = mu_q(x)` (deterministic reconstruction skoru)

### Gerekce

- Primary rankingde sampling varyansini azaltir ve model-karsilastirmayi daha stabil yapar.

---

## Q11 - Generation Testinde Kac Ornekleme Alinacak?

### Karar

- **Primary:** `N=1` (deterministic, `z=mu_p(c)`)
- **Secondary:** `N=16` (`z~p(z|c)`), event-bazli ortalama + CI

### Not

- Gerekiyorsa ek duyarllik analizi `N=32` ile raporlanabilir; ancak resmi karar `N=16`dir.

---

## Q12 - Degerlendirme Deterministic mi, Multi-sample mi?

### Karar

- **Iki katmanli raporlama zorunlu:**
  1. Deterministic primary skor (`z=mu_p(c)`)
  2. Multi-sample secondary skor (`N=16`, `z~p(z|c)`)

### Raporlama

- Nihai model secim tablosu primary (deterministic) skora gore siralanir.
- Ek tabloda stochastic expectation ve belirsizlik araligi verilir.

---

## Q13 - Hangi Train/Val/Test/OOD Listeleri Dondurulacak?

### Karar

- **Tum split listeleri tek bir manifest ile dondurulur (V1).**
- Dosya: `protocol/frozen_splits_v1.json`

### Icerik

1. `train_files`
2. `val_files`
3. `test_files`
4. `ood_primary_files` (post-training custom)

### V1 Freeze Artefakti

- `protocol/frozen_splits_v1.json`
- `protocol/splits_v1/*.txt`

---

## Q14 - Liste Hash Kaydi Zorunlu mu?

### Karar

- **Evet, zorunlu.**
- Her split listesi icin `sha256` hash kaydedilir.

### Dosya

- Hash kaydi ayni manifestte tutulur: `protocol/frozen_splits_v1.json`

---

## Q15 - Istasyon Subseti Sabit mi?

### Karar

- **Evet, sabit.**
- Split manifestte tanimli station listesi disina cikilmaz.

### Kural

- Egitim ve degerlendirme scriptleri runtime'da "ek istasyon kesfi" yapmaz.

---

## Q16 - Event-level Leakage Nasil Engellenecek?

### Karar

- **Split birimi event-level olacak.**

### Uygulama

1. Once event listesi split edilir.
2. Sonra event'e bagli tum station-segment kayitlari ayni splitte tutulur.
3. Ayni `event_id` birden fazla splitte gorunemez.

---

## Q17 - Condition Normalization Istatistikleri Nereden Hesaplanacak?

### Karar

- **Sadece train splitten hesaplanir.**
- Val/Test/OOD tarafinda yeniden fit edilmez; sadece apply edilir.

### Dosya

- `protocol/normalization_stats_v1.json` (frozen)

### V1 Freeze Notu

- Stats, `train_files` listesi uzerinden hesaplanmistir (seed=42 event-level split).

---

## Q18 - OOD Seti Tek mi, Iki mi?

### Karar

- **Primary confirmatory: tek OOD seti (`post-training custom`).**
- Ikinci OOD seti, bu turun resmi gating kararina dahil edilmez; sadece extension olarak raporlanabilir.

### Gerekce

- Bu turda over-complication onlenir; operasyonel risk dusurulur.
- Confirmatory eksen tek ve net tutulur.

---

## Q19 - Fair Karsilastirma `iso-step` / `iso-epoch` / `iso-time`?

### Karar

- **Primary fairness: `iso-step`**
- `iso-time` sadece secondary verimlilik raporu olarak tutulur.

---

## Q20 - Early Stopping Tum Modellere Ayni Policy mi?

### Karar

- **Evet, birebir ayni policy zorunlu.**
- Parametreler: `val_check_every_steps`, `patience`, `min_delta`, `max_steps` ortak.

---

## Q21 - LR Schedule Tum Modellere Ayni mi?

### Karar

- **Evet, ayni schedule/kurallar kullanilacak.**
- Aileye/latent'e ozel LR tuning yapilmayacak (confirmatory turda).

---

## Q22 - Parametre Farki Buyuk Modellerde Optimizer Ayni mi?

### Karar

- **Evet, optimizer ayarlari sabit tutulacak.**
- Sadece sayisal stabiliteyi bozan acik hata varsa (NaN/divergence), tum aileye global fix uygulanabilir.

### Not

- Noktasal model-bazli optimizer "iyilestirmesi" confirmatory turda yasak.

---

## Q23 - Seed Sayisi Kac?

### Karar

- **Seed sayisi: 3 (zorunlu minimum).**
- Tum model noktalarinda ayni seed seti kullanilir.

---

## Q24 - Her Model Icin Kac Replicate?

### Karar

- **Replicate = 3** (Q23 ile ayni seed seti)
- Nihai rapor: mean, std, 95% CI

### Not

- Butce uygun olursa extension turda 5 seed'e cikilabilir; ancak V1 resmi karar 3'tur.

---

## Q25 - Backbone Seviyeleri Neler Olacak?

### Karar

- **V1 confirmatory backbone seti: `large`, `small`**
- `xsmall` bu turda resmi matrise alinmaz (opsiyonel extension).

### Gerekce

- Over-complication ve run patlamasini engeller.
- Hipotezi test etmek icin iki net kapasite seviyesi yeterli.

---

## Q26 - Latent Boyut Listesi Ne Olacak?

### Karar

- **Sabit latent grid:** `[16, 32, 48, 64, 96, 128, 160]`

### Not

- Bu karar Q57 ile aynidir; burada kapsam tekrar sabitlenmistir.

---

## Q27 - FullCov ve Baseline Her Noktada Birebir Esit Taranacak mi?

### Karar

- **Evet.**
- Her backbone ve latent noktasi icin her iki ailede ayni run sayisi, ayni seed, ayni fairness policy uygulanir.

---

## Q28 - Karsilastirma Matrisi Once Dondurulup Sonra Degistirilmeyecek mi?

### Karar

- **Evet, dondurulacak.**
- Matris dosyasi V1 olarak kaydedilecek ve confirmatory turda degistirilmeyecek.

### Dosya

- `configs/model_grid_v1.yaml` (olusturulacak)

---

## Q29 - Minimum Egitim Suresi (Warmup) Tanimlanacak mi?

### Karar

- **Evet, zorunlu warmup var.**
- `min_steps_before_early_stop = max(5 * val_check_every_steps, 0.20 * max_steps)`

### Gerekce

- Kucuk modellerin erken sans eseri iyi gorunme riskini azaltir.
- Buyuk modellerin gec acilan performansini adil sekilde yakalar.

---

## Q30 - Birincil Kalite Metrikleri Hangileri?

### Karar

- **Primary (STFT-only) metrik seti:**
  1. `SSIM_spec` (higher better)
  2. `SC_spec` (lower better)
  3. `LSD_spec` (lower better)
  4. `OnsetErr_spec` (lower better)
  5. `EnvCorr_spec` (higher better)
  6. `BandProfileDist_spec` (lower better)

### Not

- Bu set Q58 ile uyumludur.

---

## Q31 - Ikinci Kalite Metrikleri Hangileri?

### Karar

- **Secondary/diagnostic metrikler:**
  1. `S-Corr_spec`
  2. `STA/LTA_peak_amp_err_spec`
  3. Waveform tabanli metrikler (`DTW`, waveform `XCorr`, waveform `MR-LSD`, vb.)

---

## Q32 - Metric Direction (High/Low) Nasil Sabitlenecek?

### Karar

- **Higher is better:** `SSIM_spec`, `EnvCorr_spec`, `S-Corr_spec`, waveform `XCorr`
- **Lower is better:** `SC_spec`, `LSD_spec`, `OnsetErr_spec`, `BandProfileDist_spec`, `STA/LTA_peak_amp_err_spec`, `DTW`, waveform `MR-LSD`

---

## Q33 - Composite Quality Score Formulu Pre-register Edilecek mi?

### Karar

- **Evet, pre-register edilecek.**

### Formul (V1)

1. Her primary metrik event-bazli robust z ile normalize edilir.
2. Lower-better metrikler isaret cevrimi ile higher-better formata getirilir.
3. `Q = mean(z_primary_1 ... z_primary_6)`

### Not

- `Q` sadece primary metriklerden hesaplanir.

---

## Q34 - Agirlikli mi, Esit Agirlikli mi?

### Karar

- **V1: esit agirlikli composite.**
- Agirlikli varyantlar extension olarak ayrica raporlanabilir, fakat resmi secim kuralina girmez.

---

## Q35 - Bootstrap CI Zorunlu mu?

### Karar

- **Evet, zorunlu.**
- Event-level bootstrap ile `%95 CI` verilir (Q ve ODI dahil ana ozetlerde).

---

## Q36 - Coklu Karsilastirma Duzeltmesi Uygulanacak mi?

### Karar

- **Evet.**
- Pairwise model karsilastirmalarinda primary metrikler icin `BH-FDR` uygulanir.

### Not

- Tek metrikli basit ranking tablolarinda p-deger yerine etki buyuklugu + CI esas alinabilir.

---

## Q37 - Offdiag Hangi Seviyede Olculecek?

### Karar

- **Iki seviyede olculecek:**
  1. Primary: aggregated posterior (`Sigma_agg`, `Corr_agg`)
  2. Secondary diagnostic: sample-level `q(z|x)` ozetleri

### Not

- Policy/gating kararlarinda primary seviye esas alinir.

---

## Q38 - Offdiag Ozetleri Hangileri Olacak?

### Karar

- **Zorunlu ozetler:**
  1. `offdiag_mean`
  2. `offdiag_p95`
  3. `offdiag_max`
  4. `offdiag_energy_ratio` (offdiag enerji / toplam enerji)

---

## Q39 - TC Tanimi Ne Olacak?

### Karar

- **Gaussian TC (aggregated) kullanilacak.**
- `TC_agg = 0.5 * (sum(log(diag(Sigma_agg))) - logdet(Sigma_agg))`
- Boyut-normalize rapor: `tc_per_dim = TC_agg / d`

---

## Q40 - MI Nasil Olculecek?

### Karar

- **Pairwise Gaussian MI (aggregated corr)**
- `MI_ij = -0.5 * log(1 - rho_ij^2)`
- Rapor: `pairwise_mi_mean`, opsiyonel `pairwise_mi_p95`

---

## Q41 - Basis-rotation Kontrolu Zorunlu mu?

### Karar

- **Evet, zorunlu.**
- Offdiag yorumunun basis-duyarliligini kontrol etmek icin rotation-stress analizi uygulanir.

---

## Q42 - Rotation Sayisi Kac Olacak?

### Karar

- **V1: 64 rastgele ortogonal rotasyon**
- Rapor: `rot_std`, `rot_p95`

---

## Q43 - Tek Bagimlilik Indeksi (`ODI`) Kullanilacak mi?

### Karar

- **Evet, ODI resmi secondary karar indeksidir.**
- Ham bilesenler (`offdiag_*`, `tc_per_dim`, `pairwise_mi_*`, `rot_*`) her zaman birlikte raporlanir.

---

## Q44 - Policy Resmi Karar Kurali Olarak Sabitlenecek mi?

### Karar

- **Evet.**
- Q60'ta verilen `Q + ODI` stop policy resmi karar mekanizmasidir.

---

## Q45 - Stop Rule Denklemi Nedir?

### Karar

- `Q_{k+1} >= Q_k - eps` ve `ODI_{k+1} <= ODI_k + delta` ise devam.
- Aksi halde dur; bir onceki adim secilir.

---

## Q46 - `eps` ve `delta` Ne Olacak?

### Karar

- `eps = 0.015`
- `delta = 0.15`

### Not

- Bu degerler V1 confirmatory turda sabittir.

---

## Q47 - Nihai Secim Kriteri Nedir?

### Karar

- **Nihai secim: en kucuk kabul edilebilir model.**
- Pareto grafikler raporlanir; ancak karar kurali olarak V1'de ikincildir.

---

## Q48 - Policy Reconstruction ve Generation Icin Ayri mi?

### Karar

- **Tek policy var; ana eksen generation.**
- Reconstruction, latent diagnostik ve yorumlayici gating icin ikinci katman olarak raporlanir.

---

## Q49 - Zorunlu Rapor Tablolari Hangileri?

### Karar

- **Zorunlu tablolar:**
  1. Model grid kalite tablosu (Q, primary metricler, CI)
  2. Model grid bagimlilik tablosu (`ODI` + ham bilesenler, CI)
  3. Policy secim tablosu (`quality-only`, `joint`, secilen model)
  4. Parametre/sure tablosu (`param_count`, `time-to-quality`)
  5. Q61-Q64 evidence gate pass/fail ozeti

---

## Q50 - Zorunlu Grafikler Hangileri?

### Karar

- **Zorunlu grafikler:**
  1. `Q vs latent_dim` (aile ayri)
  2. `ODI vs latent_dim` (aile ayri)
  3. `Q vs ODI` Pareto
  4. `offdiag_heatmap` (secilen modeller)
  5. `stepwise ODI_k vs Drop_{k->k+1}` (predictive utility)
  6. ID vs OOD policy secim tutarlilik grafigi

---

## Q51 - Hangi Failure Case'ler Zorunlu Raporlanacak?

### Karar

- **Zorunlu failure case listesi:**
  1. NaN/divergence
  2. Early-stop before warmup (policy ihlali)
  3. Null test fail (Q61)
  4. Predictive utility fail (Q62)
  5. OOD'de eps disi kalite dususu
  6. Seedler arasi secim tutarsizligi

---

## Q52 - Negative Result Raporlamasi Zorunlu mu?

### Karar

- **Evet, zorunlu.**
- Q61-Q64'ten herhangi biri fail ise acikca yazilir; gizlenmez.

---

## Q53 - Reproducibility Checklist Nelerden Olusacak?

### Karar

- **Asgari checklist:**
  1. Split manifest + hash (`frozen_splits_v1.json`)
  2. Normalization stats (`normalization_stats_v1.json`)
  3. Frozen model grid (`model_grid_v1.yaml`)
  4. Frozen evidence thresholds (`offdiag_minimum_evidence_v1.yaml`)
  5. Seed list + ortam bilgisi + commit id
  6. Tum metrik ciktilari ham CSV/JSON
  7. Q61-Q64 evidence summary

---

## Q54 - Makale Iddia Siniri (Scope Boundary) Nasil Yazilacak?

### Karar

- **Iddia siniri acik yazilacak:**
  - Bu calisma, belirli bir conditional seismic generation probleminde,
    OffDiagonal/ODI-temelli model secim policy'sinin pratik faydasini test eder.
  - "Tum VAE'lerde latentler bagimsiz olmalidir" gibi evrensel bir teorem iddiasi kurulmaz.
  - Sonuclar, bu veri/protokol ailelerinde confirmatory olarak gecerlidir.

## Q55 - Birincil Degerlendirme Reconstruction mi, Generation mi?

### Karar

- **Birincil degerlendirme: `Conditional Generation`**
- **Ikinci/gating degerlendirme: `Reconstruction + posterior bagimlilik analizi (offdiag/TC/MI)`**

### Gerekce (Neden?)

1. Nihai kullanim senaryosu `condition -> waveform/spec` uretimidir.
2. Arastirma amaci, condition altinda deprem benzeri sinyal uretim kalitesini artirmaktir.
3. Reconstruction, modele hedef girdiyi verdigi icin generation zorlugunu tam temsil etmez.
4. Buna ragmen hipotezimiz latent bagimliligi (offdiag) ile ilgili oldugu icin, sadece generation bakmak yetersiz olur.
5. Offdiag etkisi en dogrudan `q(z|x)` tarafinda olculur; bu da reconstruction/pseudo-reconstruction protokolu ile desteklenmelidir.

### Uygulama (Nasil?)

1. **Primary endpoint (ranking):**
   - Conditional generation metrikleri ile model siralamasi yapilir.
2. **Secondary endpoint (interpretation/gating):**
   - Reconstruction metrikleri + `TC/MI/offdiag` raporlanir.
3. **Nihai secim kurali:**
   - En kucuk model secilir, ancak:
     - generation kalitesi en iyi modele gore kabul edilen tolerans icinde olmali (`eps`),
     - bagimlilik indeksi (`ODI`) kotulesmemeli (`delta`).

### Yanlis Yorumlari Engelleme Notu

- `Generation iyi -> latent bagimsizligi iyidir` otomatik olarak dogru degildir.
- `Offdiag buyuk -> model kotudur` otomatik olarak dogru degildir.
- Bu nedenle kararlar tek metrikle degil, primary+secondary birlikte verilecektir.

### Bu Kararin Kapsam Siniri

- Bu karar, NonDiagonalRigid deney hatti icin gecerlidir.
- Daha sonraki makale metninde bu karar, `problem-specific evaluation protocol` olarak yazilacaktir.

---

## Q56 - Fairness Tanimi (`iso-step` mi `iso-time` mi?)

### Karar

- **Primary fairness: `iso-step`**
- **Secondary fairness raporu: `time-to-quality` (yardimci analiz)**
- **Early stopping: acik, ama tum modellerde birebir ayni kural**

### Gerekce (Neden?)

1. `Iso-step`, optimizer guncelleme sayisini esitler ve model-buyuklugu itirazlarina karsi daha savunulabilir bir ana fairness ekseni verir.
2. `Iso-time` tek basina adil degildir; buyuk modeller dogasi geregi daha yavas oldugu icin ceza etkisi yaratir.
3. Buna ragmen pratik verimlilik icin `time-to-quality` mutlaka ikincil raporlanmalidir.
4. Buyuk modelin yeterince egitildigi itirazini kapatmak icin `max_steps` buyuk modele gore belirlenir.

### Uygulama (Nasil?)

1. Referans model: `FullCov large` (en zor/agir aday).
2. Pilot calismada referans modelin val-loss plato noktasi bulunur.
3. `max_steps` bu plato seviyesine gore sabitlenir (tum grid icin tek deger).
4. Tum modeller ayni:
   - `max_steps`
   - `val_check_every_steps`
   - `patience` (eval adedi)
   - `min_delta`
   ile calisir.
5. Her modelde:
   - `best checkpoint`
   - `best_step`
   - `stop_reason` (early-stop / max-steps)
   kaydi zorunludur.
6. Eger referans model seed'lerinden biri `max_steps` sonunda halen plato yapmiyorsa, butce yetersiz sayilir ve tum deney yeniden daha yuksek `max_steps` ile tekrarlanir.

### Raporlama Zorunlulugu

- Ana tablo: iso-step best kalite karsilastirmasi
- Ek tablo: `time-to-quality` (or. quality'nin %98'ine ulasma adim/sure)

### Bu Kararin Kapsam Siniri

- Bu karar, NonDiagonalRigid deney hatti icin tum mimari karsilastirmalarda zorunludur.

---

## Q57 - Latent Tarama Listesi Ne Olacak?

### Karar

- **Detayli latent grid sabitlendi:**  
  - `[16, 32, 48, 64, 96, 128, 160]`

### Gerekce (Neden?)

1. `ld64` ve `ld128` arasinda tek adim degil, ara gecisler de gorulmeli.
2. Erken bozulma/iyilesme noktalarini bulmak icin daha yogun tarama gerekir.
3. Katı hipotez testinde tek tük nokta yerine egri davranisi gereklidir.

### Uygulama (Nasil?)

1. Bu latent listesi, karsilastirilan tum model ailelerine ayni uygulanir.
2. Sonradan latent degeri ekleme/karma yapilmaz (pre-registered grid korunur).
3. Sonuclar latent-boyut egri tablosu olarak raporlanir:
   - kalite egri
   - bagimlilik egri (`ODI`, `TC`, `offdiag`)
   - model boyutu egri

### Operasyonel Not

- Bu karar run sayisini arttirir; ancak dogrulama gucunu belirgin sekilde artirdigi icin kabul edilmistir.

---

## Q58 - Birincil Kalite Metrik Seti Ne Olacak?

### Karar

- **Birincil kalite metrikleri `STFT/model-output` uzayinda olcecek.**
- **Waveform tabanli metrikler birincil rankinge girmeyecek; sadece ikincil/diagnostic olacak.**

### Gerekce (Neden?)

1. Modelin dogrudan cikti uzayi: log-magnitude STFT.
2. Waveform tarafi (GL/ISTFT/sampling) ek rastgelelik ve geri-donusum hatasi tasir.
3. Birincil kararlar, modelin ogrenip urettigi uzayda verilirse daha net ve itiraza kapali olur.
4. Metrik seti birbirinden farkli kalite boyutlarini olcmelidir (tek bir sinif metrik degil).

### Birincil Metrik Seti (STFT-Only, Revize)

1. `SSIM_spec` (↑)  
   - yerel zaman-frekans yapisal benzerlik
2. `SC_spec` (↓)  
   - global spektral yakinlik / relatif norm farki
3. `LSD_spec` (↓)  
   - log-spektral sadakat (zayif enerji farklarini da gorur)
4. `OnsetErr_spec` (↓)  
   - spektral enerji tabanli varis/onset zaman hatasi
5. `EnvCorr_spec` (↑)  
   - spektral-zaman zarf benzerligi
6. `BandProfileDist_spec` (↓)  
   - dusuk/orta/yuksek bant enerji dagilim uyumu

### Ikincil (Diagnostic) Metrikler

1. `S-Corr_spec` (↑)  
   - `SC_spec` ile yuksek tekrar riski oldugu icin primary disina alindi
2. `STA/LTA_peak_amp_err_spec` (↓)  
   - onset zamanindan ayri olarak tepe genlik/oran farkini izler
3. Waveform tabanli metrikler (`DTW`, waveform `XCorr`, waveform `MR-LSD`, vb.)  
   - sadece ek analiz

Not (metrik tekrar kontrolu):

- Onceki auditte `S-Corr` ve `SC` arasinda yuksek tekrar sinyali goruldu.
- Bu nedenle primary sette ikisinden yalnizca `SC_spec` birakildi.

### Uygulama Kurali

1. Tum primary metrikler, model cikti spektrumundan hesaplanir.
2. Hesaplama uzayi:
   - normalize edilmis cikti, sample bazli `mag_min/mag_max` ile log-magnitude uzayina geri cekilir.
3. Primary composite kalite skoru sadece bu 6 primary metrigi kullanir.

---

## Q59 - ODI (OffDiagonal Dependency Index) Tanimi Ne Olacak?

### Karar

- **ODI, latent bagimliligini tek skorda birlestiren secondary karar indeksidir.**
- **Primary rankingi belirlemez; quality-gated model seciminde kullanilir.**

### Kullanım Amaci (Nerede/Nasil?)

1. Mimari kucultme adimlarinda (`ld160 -> ... -> ld16`) `devam/dur` sinyali vermek.
2. Kaliteye yakin modeller arasinda bagimsizlik yonunde tie-break yapmak.
3. Nihai raporda `quality vs dependency` Pareto uzayi cikarmak.

### ODI Bilesenleri

ODI, asagidaki 4 ana bagimlilik olcusunu birlestirir:

1. `offdiag_mean`: `mean(|Corr_agg_offdiag|)`
2. `offdiag_p95`: `p95(|Corr_agg_offdiag|)`
3. `tc_per_dim`: `TC_agg / d`
4. `pairwise_mi_mean`: `mean(-0.5 * log(1-rho_ij^2))`, `rho_ij = Corr_agg(i,j)`

Opsiyonel basis-stability cezasi:

5. `rot_std`: rastgele ortogonal rotasyonlarda `offdiag_mean` standart sapmasi

### Temel Kovaryans Tanimi

Her model icin degerlendirme setinde:

- `Sigma_agg = Cov_x(mu(x)) + E_x[Sigma(x)]`

Burada:

- diagonal posterior modellerde: `Sigma(x) = diag(exp(logvar(x)))`
- full-cov modellerde: `Sigma(x) = L(x)L(x)^T`

`Corr_agg`, `Sigma_agg`'den turetilir ve offdiag/MI olculeri bunun uzerinden hesaplanir.

### TC Tanimi

- Gaussian toplam korelasyon:
  - `TC_agg = 0.5 * (sum(log(diag(Sigma_agg))) - logdet(Sigma_agg))`
- boyut etkisini dengelemek icin:
  - `tc_per_dim = TC_agg / d`

### Normalizasyon (Robust z)

Her bilesen su sekilde normalize edilir:

- `rz(x) = (x - median_ref) / (1.4826 * MAD_ref + eps)`

`ref` istatistikleri:

1. sabit model matrisi + sabit seed seti uzerinden hesaplanir,
2. bir kez dondurulur,
3. sonraki tum karsilastirmalarda aynen kullanilir.

Not:

- Min-max yerine robust z secildi; outlier etkisini azaltir ve daha stabil karar verir.

### Formul

Temel indeks:

- `ODI_base = 0.25*rz(offdiag_mean) + 0.25*rz(offdiag_p95) + 0.25*rz(tc_per_dim) + 0.25*rz(pairwise_mi_mean)`

Rotation-cezali indeks:

- `ODI = ODI_base + lambda_rot * rz(rot_std)`

Varsayilan:

- `lambda_rot = 0.5`

### Neden Bu Tasarim?

1. `offdiag` tek basina karar vermek icin yetersizdir.
2. `TC` tek basina boyut etkisine duyarli olabilir; bu nedenle `per-dim` kullanilir.
3. `MI`, bagimlilik bilgisini tamamlar.
4. `rot_std`, basis hassasiyetini ceza olarak dahil eder.
5. Bilesik indeks, tek metrik oynakligina karsi daha dayaniklidir.

### Yorumlama Kurali

- `ODI` kucukse bagimlilik daha dusuk kabul edilir.
- `ODI` primary kalite metriklerinin yerine gecmez.
- Nihai secimde quality toleransi ile birlikte kullanilir (Q60 kararinda sabitlenecek).

### Raporlama Zorunlulugu

Her model/seed icin su ham degerler zorunlu kaydedilir:

- `offdiag_mean`, `offdiag_p95`, `tc_per_dim`, `pairwise_mi_mean`, `rot_std`, `ODI_base`, `ODI`

Toplu raporda:

- model ortalamasi
- seed varyansi
- 95% CI/bootstrapped aralik

---

## Q60 - Stop Rule Esikleri (`eps`, `delta`) Ne Olacak?

### Karar

- **Stop rule aktif:**  
  - `Q_{k+1} >= Q_k - eps`
  - `ODI_{k+1} <= ODI_k + delta`
- **Esikler (varsayilan, pre-registered):**
  - `eps = 0.015`  (quality skorunda %1.5 tolerans)
  - `delta = 0.15` (ODI robust-z biriminde tolerans)

### Gerekce (Neden?)

1. Kucuk kalite dalgalanmalari (seed/eval noise) nedeniyle asiri erken durmayi engeller.
2. Bagimlilikta cok kucuk oynakliklari “gercek bozulma” sanma riskini azaltir.
3. Esikler sade ve uygulanabilir tutuldu; asiri karmasik kontrol mekanizmasi eklenmedi.
4. Tum modellerde ayni esitlikler kullanildigi icin karar adaleti korunur.

### Uygulama (Nasil?)

1. Latent adimlari buyukten kucuge sirali izlenir.
2. Her adimda (ayni backbone/model ailesi icinde) `Q` ve `ODI` farki hesaplanir.
3. Her iki kosul da saglaniyorsa bir sonraki daha kucuk modele gecilir.
4. Kosullardan biri bozulursa bir onceki adim “en kucuk kabul edilebilir model” olarak secilir.

### Tie-Break ve Kararsizlik Kurali

- Birden fazla aday kabul ediliyorsa en kucuk parametreli aday secilir.
- Eger farklar CI ile kararsiz bolgedeyse (sinira cok yakin), konservatif davranilir ve bir onceki model tutulur.

### Not

- Bu esikler ilk confirmatory tur icin sabitlendi.
- Ilk tur bittikten sonra degisiklik yapilacaksa yeni turda yeni versiyon numarasi ile tekrar pre-register edilir.

---

## Q61 - OffDiagonal Sinyalinin Gercek Oldugunu Gosteren Minimum Kanit

### Karar

- **Zorunlu Kanit-1 (Signal Reality):** OffDiagonal/ODI degerleri, uygun null dagilima gore sistematik olarak sifirdan ayirt edilmelidir.

### Uygulama (Nasil?)

1. Her model noktasinda (backbone x latent x seed) `Sigma_agg` ve `Corr_agg` uzerinden:
   - `offdiag_mean`, `offdiag_p95`, `tc_per_dim` hesaplanir.
2. Iki null uretilir:
   - `null_shuffle`: her latent boyut, sample ekseninde bagimsiz permute edilir (marjinal korunur, bagimlilik bozulur).
   - `null_gauss`: ayni diag varyans ile bagimsiz Gaussian sentetik latent.
3. Her metrik icin `z_null = (real - mean(null)) / std(null)` hesaplanir.

### Gecme Kriteri (Pass/Fail)

- Bir model ailesinde tarama noktalarinin en az `%70`'inde:
  - `z_null(offdiag_mean) >= 2`
  - `z_null(tc_per_dim) >= 2`
- Aksi durumda OffDiagonal sinyali "yetersiz/kararsiz" kabul edilir.

---

## Q62 - ODI'nin Ongoru Degerini Gosteren Minimum Kanit

### Karar

- **Zorunlu Kanit-2 (Predictive Utility):** `ODI_k`, bir sonraki kucultme adimindaki kalite dususunu ongormelidir.

### Uygulama (Nasil?)

1. Sirali latent patikasinda (`ld160 -> ... -> ld16`) her adim icin:
   - `ODI_k`
   - `Drop_{k->k+1} = max(0, Q_k - Q_{k+1})`
2. Her seed/model ailesinde `Spearman(ODI_k, Drop_{k->k+1})` hesaplanir.
3. Bootstrap (event-level) ile CI uretilir.

### Gecme Kriteri (Pass/Fail)

- Aile-bazli medyan korelasyon:
  - `rho_med >= 0.40`
  - `%95 CI alt siniri > 0`
- En az bir ailede bu kosul saglanmali, diger ailede isaret tersine donmemelidir (`rho_med >= 0`).

---

## Q63 - ODI-Policy'nin Pratik Faydasini Gosteren Minimum Kanit

### Karar

- **Zorunlu Kanit-3 (Decision Benefit):** Joint policy (`Q+ODI`) en az bir pratik hedefte kalite-only policy'den ustun olmalidir.

### Karsilastirilacak Policy'ler

1. `Quality-only`: sadece `Q` toleransi ile secim
2. `ODI-only` (diagnostic): sadece bagimlilik odakli secim
3. `Joint (resmi)`: `Q` + `ODI` stop-rule (Q60)

### Gecme Kriteri (Pass/Fail)

- `Joint`, `Quality-only`e gore asagidakilerden en az birini saglamalidir:
  1. `Q` farki tolerans icinde kalirken (`<= eps`), parametre sayisinda en az `%20` azalma
  2. Parametre benzer seviyede (`+-5%`) iken, `ODI`de en az `0.30` robust-z iyilesme
- Ve ayni anda OOD birincil kalite skoru `Quality-only`den `eps`ten fazla kotulesmemelidir.

---

## Q64 - Model Ailesi ve OOD Uzerinde Tekrarlanabilirlik Kaniti

### Karar

- **Zorunlu Kanit-4 (Cross-Setting Consistency):** OffDiagonal-temelli karar davranisi farkli model ailesi ve dagilimlarda tutarli gorulmelidir.

### Uygulama (Nasil?)

1. En az iki model ailesi:
   - `Baseline (diag posterior)`
   - `FullCov`
2. En az iki dagilim:
   - `ID/Test`
   - `OOD (post-training custom)`
3. Her kombinasyonda seed'ler icin:
   - secilen latent adim
   - `Q`, `ODI`, parametre sayisi
   raporlanir.

### Gecme Kriteri (Pass/Fail)

- Aileler arasinda policy yonu tutarli olmali:
  - asiri kucultmede `ODI` artisi + kalite bozulmasi birlikte gorulmeli.
- Seed tutarliligi:
  - secilen latent adimlarin en az `%66`si ayni veya komsu adimda olmali (`+-1` step).
- Eger bu kosullar saglanmazsa "genelleme iddiasi" kurulmaz; sadece setting-spesifik sonuc olarak raporlanir.

---

## OffDiagonal Dayanak Notu (V1)

Bu protokolde OffDiagonal/ODI, **yardimci bir grafik degil**, hipotezin test eksenidir.  
Ancak methodolojik olarak tek basina yeterli kabul edilmez; kalite kapisi (`Q`) ile birlikte yorumlanir.

Nihai iddiayi kurmak icin **Q61-Q64 dort minimum kanitin tamami** zorunludur.

Esiklerin makine-okunur kopyasi: `configs/offdiag_minimum_evidence_v1.yaml`

---

## Q65 - Backbone Kucultmeyi Latentten Once Ayri Fazda mi Test Edecegiz?

### Karar

- **Evet.**
- Faz-1: backbone policy secimi (latent sabit)
- Faz-2: secilen policy ile latent tarama

### Gerekce

- Latent etkisi ile backbone daraltma etkisini karistirmamak.

---

## Q66 - Kucultme Politikalari Hangileri Olacak?

### Karar

- **Policy seti:** `width_only`, `depth_only`, `hybrid`
- Bu policy seti iki model ailesinde de ayni test edilir.

---

## Q67 - Kucultme Seviyeleri Ne Olacak?

### Karar

- **Scale seti:** `1.00`, `0.75`, `0.50`

### Not

- Gerekirse sadece tie-break extension icin tek ara nokta (`0.65` veya `0.85`) eklenebilir; V1 resmi grid degismez.

---

## Q68 - Stage-1 (Policy Secimi) Icin Latent Sabit Deger Ne Olacak?

### Karar

- **Sabit latent:** `ld=128`
- **Sanity-check:** finalist policy'lerde `ld=96` hizli kontrol (diagnostic)

---

## Q69 - Policy Secimi Hangi Karar Kurali ile Yapilacak?

### Karar

- **Ortak policy secimi zorunlu** (iki aile icin tek policy/scale secilir).
- Secim sirasi:
  1. Primary `Q` (STFT primary composite) icinde `eps` toleransi
  2. Guardrail metriklerde anlamli bozulma olmamasi
  3. `ODI` ve model karmasikligi tie-break

### Rapor

- `quality-only` ve `Q+ODI` iki gorunum de raporlanir; resmi secim `Q+ODI` kuralina gore yapilir.

---

## Q70 - Receptive Field/Downsample Yapisini Sabit Tutmak Zorunlu mu?

### Karar

- **Evet, zorunlu.**
- Backbone kucultmede mimumkun oldugunca ayni receptive-field ailesi korunur.

### Not

- RF degisimi gerekiyorsa o aday ayrica "confounded" etiketi ile raporlanir ve resmi secime alinmaz.

---

## Q71 - Stage-1 ve Stage-2 Butcesi Nasil Sabitlenecek?

### Karar

- **Iki asamali butce:**
  1. `Pilot eleme`: kisa iso-step, early-stop kapali
  2. `Final dogrulama`: tam iso-step, resmi early-stop acik

### V1 Butce Profili

- Pilot:
  - `max_steps=12000`
  - `val_check_every_steps=2000`
  - `patience_evals=9999` (pratikte early-stop kapali)
- Final:
  - `max_steps=60000`
  - `val_check_every_steps=1000`
  - `patience_evals=10`

---

## Q72 - Nihai Secimde Backbone Policy Kapisi Zorunlu mu?

### Karar

- **Evet, zorunlu.**
- Latent taramaya gecmeden once `ortak backbone policy + scale` secimi kapanmis olmalidir.

### Akis

1. Faz-1: policy/scale secimi
2. Faz-2: latent tarama
3. Faz-3: Q61-Q64 evidence gate

---

## Backbone Faz Notu (V1)

Q65-Q72 kararlarinin operasyonel kopyalari:

- Metodoloji: `docs/backbone_shrink_methodology_discussion.md`
- Policy grid: `configs/backbone_policy_grid_v1.yaml`
- Egitim butcesi: `configs/training_budget_v1.yaml`
