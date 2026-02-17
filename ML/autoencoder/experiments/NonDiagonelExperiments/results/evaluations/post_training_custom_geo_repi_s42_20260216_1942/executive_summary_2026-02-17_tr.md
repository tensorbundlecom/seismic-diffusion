# Executive Summary - NonDiagonel Programı (2026-02-17)

## 1) Amaç ve Hipotez

Test edilen ana iddia:

- `FullCovariance posterior'da off-diagonal terimler sıfıra yakın değilse mimari optimal değildir.`
- buna bağlı olarak: `kalite korunurken bağımlılığın (non-diagonal etkilerin) çok küçüldüğü bir optimal mimari bulunmalıdır.`

Bu özet, şimdiye kadar tamamlanan tüm eğitim, OOD ve latent-bağımlılık analizlerini birleştirir.

---

## 2) Uygulanan Deneyler

### A. Sabit koşul tasarımı

Model koşulları standartlaştırıldı:

- `magnitude`
- `log1p(epicentral_distance_km)`
- `depth_km`
- `sin(azimuth), cos(azimuth)`
- `station_embedding`

### B. Mimari eksenleri

1. `large + ld128` (referans)
2. `small + ld128`
3. `small + ld64`

Her eksende iki model:

- `BaselineGeo` (diagonal posterior)
- `FullCovGeo` (full covariance posterior)

### C. Değerlendirme protokolü

- OOD seti: post-training custom HH (`52` örnek)
- metrikler: SSIM, S-Corr, SC, STA/LTA Err, LSD, MR-LSD, Arias Err, Env Corr, DTW, XCorr
- latent bağımlılık analizleri:
  - `TC` (Total Correlation)
  - off-diagonal özetleri
  - pairwise Gaussian `MI`
  - basis-rotation stress (ortogonal rotasyon)

---

## 3) Eğitim Sonuçları

| Model | En İyi Val Loss | Epoch |
|:---|---:|---:|
| BaselineGeo_large_ld128 | **173.7884** | 31 |
| BaselineGeo_small_ld128 | 174.1540 | 28 |
| BaselineGeo_small_ld64 | 176.8245 | 29 |
| FullCovGeo_large_ld128 | 176.5579 | 99 |
| FullCovGeo_small_ld128 | 179.3707 | 29 |
| FullCovGeo_small_ld64 | 178.4879 | 29 |

Okuma:

- `small_ld128 baseline`, `large baseline`e çok yakın (+0.3656).
- Aynı ölçeklerde FullCov varyantları baseline’dan daha yüksek val loss üretti.

---

## 4) OOD Sonuçları (52 Örnek)

Kaynak: `model_family_eval_small_phase_20260217_1216/summary.md`

### Metrik liderlik

- genel dengede en güçlü model: **`BaselineGeo_small_ld128`**
  - SSIM, S-Corr, SC, STA/LTA Err, MR-LSD, Env Corr alanlarında en iyi
- FullCov güçlü kaldığı alanlar:
  - `FullCovGeo_small_ld128`: LSD ve DTW’de en iyi
- `BaselineGeo_small_ld64`: XCorr ve Arias Err’de iyi olsa da genel profilde istikrarlı lider değil

Yorum:

- FullCov belirli metriklerde yerel kazanç veriyor.
- Çoklu metrik dengesi açısından `BaselineGeo_small_ld128` en iyi aday.

---

## 5) TC/MI ve Non-Diagonal Bulguları

Kaynak: `latent_dependency_tc_mi_20260217_1228/report.md`

| Model | TC_agg ↓ | Offdiag Mean ↓ | Offdiag p95 ↓ | Pairwise MI Mean ↓ |
|:---|---:|---:|---:|---:|
| BaselineGeo_large_ld128 | 69.0160 | 0.1733 | 0.4412 | 0.0279 |
| FullCovGeo_large_ld128 | 42.5538 | 0.1404 | 0.3814 | 0.0181 |
| BaselineGeo_small_ld128 | 72.4135 | 0.1844 | 0.4776 | 0.0314 |
| FullCovGeo_small_ld128 | 38.3087 | **0.1372** | **0.3550** | **0.0171** |
| BaselineGeo_small_ld64 | 42.7390 | 0.2453 | 0.5730 | 0.0525 |
| FullCovGeo_small_ld64 | **28.5853** | 0.1994 | 0.4958 | 0.0359 |

Kritik nokta:

- Hiçbir model “off-diagonal ~ 0” seviyesine yaklaşmadı.
- Offdiag mean bandı yaklaşık `0.137–0.245`.

---

## 6) Basis-Rotation Sonucu

Aynı kovaryans, sadece koordinat sistemi ortogonal rotasyonla değiştirildiğinde bile off-diagonal özetleri değişiyor.

Anlamı:

- off-diagonal büyüklüğü koordinat/basis duyarlı.
- tek başına offdiag ile “optimal” kararı vermek güvenilir değil.
- quality + TC/MI + offdiag birlikte okunmalı.

---

## 7) Pareto Grafiği Ne Anlatıyor?

Kaynak: `latent_dependency_tc_mi_20260217_1228/pareto_scores.json`

Grafikte:

- X: `Quality Score` (10 OOD metriğinin normalize ortalaması)
- Y: `Independence Score` (`TC_agg`, offdiag mean/p95, pairwise MI mean’in normalize ortalaması)
- Pareto-nondominated: hem kalite hem bağımsızlıkta aynı anda geçen başka model yok

Önemli not:

- Bu skorlar **göreli** (yalnızca karşılaştırılan 6 model içinde).
- mutlak fiziksel kalite ölçeği değildir.

---

## 8) Hipotez Değerlendirmesi

### Katı hipotez durumu

İddia:

- `Kalite korunurken FullCov non-diagonal terimler sıfıra yaklaşmalı.`

Elde edilen veri:

- test edilen hiçbir modelde near-zero non-diagonal yok.
- kalite lideri model (`BaselineGeo_small_ld128`) bile belirgin bağımlılık taşıyor.
- `ld64` daraltması bağımlılığı near-zero seviyeye çekmedi.

Sonuç:

- **Katı hipotez mevcut deneylerle desteklenmiyor.**

### Desteklenen kısım

1. FullCov bazı spektral/zamansal metriklerde faydalı olabilir (LSD/DTW).
2. Genel OOD dengesi için en iyi model: `BaselineGeo_small_ld128`.
3. Bağımlılık yapısı tüm test edilen mimarilerde anlamlı düzeyde mevcut.

---

## 9) Operasyonel Karar Önerisi

Mevcut durumda ana referans model:

- `BaselineGeo_small_ld128`

Hipotez programı devam edecekse:

- eşikleri net ve sabit tut:
  - `offdiag_mean <= 0.03`
  - `offdiag_p95 <= 0.10`
  - kalite düşüşü <= `%1-2`
- bu üç koşulu birlikte sağlayan aday çıkmadan “hipotez doğrulandı” demeyelim.

---

## 10) Dosya İndeksi

- OOD karşılaştırma:
  - `ML/autoencoder/experiments/NonDiagonel/results/evaluations/post_training_custom_geo_repi_s42_20260216_1942/model_family_eval_small_phase_20260217_1216/`
- TC/MI + basis kontrol:
  - `ML/autoencoder/experiments/NonDiagonel/results/evaluations/post_training_custom_geo_repi_s42_20260216_1942/latent_dependency_tc_mi_20260217_1228/`
- kapsamlı teknik rapor:
  - `ML/autoencoder/experiments/NonDiagonel/results/evaluations/post_training_custom_geo_repi_s42_20260216_1942/comprehensive_hypothesis_report_2026-02-17.md`
- bu executive özet (TR):
  - `ML/autoencoder/experiments/NonDiagonel/results/evaluations/post_training_custom_geo_repi_s42_20260216_1942/executive_summary_2026-02-17_tr.md`
