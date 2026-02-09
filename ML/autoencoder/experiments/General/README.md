# General Experiment Hub

Bu dizin, `kalem_seismic` projesindeki tüm oto-kodlayıcı deneyleri için ortak çekirdek mantığı, veri çekme araçlarını ve genel değerlendirme scriptlerini barındıran merkezi yönetim birimidir.

## 📦 Klasör Yapısı

### 1. `core/` (Merkezi Çekirdek)
Tüm deneylerin (FullCovariance, NormalizingFlow vb.) bağımlı olduğu temel dosyalar burada yer alır.
- `stft_dataset.py`: Tüm deneyler için standartlaştırılmış veri yükleyici (100Hz desteği, OOD_K normalizasyonu).
- `model_baseline.py`: Standart Conditional Variational Autoencoder (CVAE) mimarisi.

### 2. `setup/` (Veri Hazırlama)
Sismik verilerin (IRIS/KOERI) indirilmesi, filtrelenmesi ve önişlenmesi için kullanılan konsolide edilmiş araçlar.
- `download_koeri_ood.py`: KOERI üzerinden OOD verisi çekme scripti.
- `preprocess_koeri_ood.py`: Resampling (100Hz) ve bandpass filtreleme araçları.

### 3. `evaluation/` (Değerlendirme)
Modellerin performansını karşılaştırmak için kullanılan genel araçlar.
- `evaluate_diverse_ood.py`: 10 farklı OOD depremi üzerinden modelleri yarıştıran ana script.
- `calculate_seismic_metrics.py`: SSIM, LSD, Arias, DTW gibi sismolojik metrik hesaplamaları.
- `archive/`: Kullanım ömrünü tamamlamış eski görselleştirme ve test scriptleri.

### 4. `checkpoints/`
Eğitilmiş model ağırlıklarının (Best-case) saklandığı dizin.

### 5. `visualizations/`
Deneyler sonucunda üretilen karşılaştırmalı grafikler, dalga formu yığınları ve spektrogram gridleri.

---

## 🌍 OOD (Out-of-Distribution) Veri Seti Detayları

Değerlendirmede kullanılan 10 adet KOERI depreminin teknik detayları:

| Kod | Tarih | Saat | Enlem | Boylam | Derinlik (km) | Büyüklük (ML) | Bölge |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **OOD_K_01** | 2010.10.03 | 17:49:03 | 40.832 | 28.176 | 3.2 | 3.5 | Marmara Denizi |
| **OOD_K_02** | 2010.10.04 | 17:01:47 | 40.177 | 27.678 | 12.0 | 2.5 | Marmara Bölgesi |
| **OOD_K_03** | 2010.11.03 | 02:51:27 | 40.413 | 26.296 | 14.2 | 5.3 | Saros Körfezi |
| **OOD_K_04** | 2010.12.30 | 12:18:39 | 40.453 | 29.202 | 7.0 | 2.0 | Marmara Denizi |
| **OOD_K_05** | 2011.07.25 | 17:57:22 | 40.824 | 27.747 | 12.2 | 5.1 | Marmara Denizi |
| **OOD_K_06** | 2012.02.18 | 02:21:38 | 40.458 | 26.381 | 14.5 | 2.6 | Çanakkale |
| **OOD_K_07** | 2012.05.04 | 05:38:14 | 40.313 | 27.006 | 7.2 | 4.4 | Marmara Bölgesi |
| **OOD_K_08** | 2013.04.09 | 11:42:24 | 40.539 | 28.135 | 5.0 | 3.4 | Marmara Denizi |
| **OOD_K_09** | 2013.08.29 | 06:20:36 | 40.347 | 27.457 | 17.4 | 4.4 | Marmara Bölgesi |
| **OOD_K_10** | 2013.12.05 | 01:09:25 | 40.831 | 27.925 | 8.9 | 2.2 | Marmara Denizi |

---

## 📊 Model Performans Karşılaştırması (10 Diverse KOERI OOD)

Eğitim setinde bulunmayan (2010-2013) ve Marmara bölgesinden seçilen 10 farklı deprem (M2.0 - M5.3) üzerindeki güncel sonuçlar:

| Model | SSIM ↑ | LSD ↓ | Arias Err ↓ | Env Corr ↑ | DTW ↓ | XCorr ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline CVAE** | 0.7046 | 3.493 | 0.4518 | 0.3678 | 10283.06 | 0.2139 |
| **Full Covariance** | 0.6437 | 3.574 | **0.4150** | 0.3439 | 11943.60 | **0.2276** |
| **Normalizing Flow** | **0.7124** | 3.668 | 0.4446 | **0.3661** | **9779.05** | 0.2159 |

**Gözlemler:**
- **Enerji Sadakati (Arias Err)**: Yenilenen rekonstrüksiyon yöntemiyle (Scipy-based GL) enerji hatası 1.0 (tam kayıp) seviyesinden makul seviyelere (~0.45) çekildi. En iyi enerji korunumunu **Full Covariance** modeli sağladı.
- **Normalizing Flow**: Spektrogram yapısal benzerliğinde (SSIM) ve zamansal hizalamada (DTW) liderliğini koruyarak en "doğal" sismogramları üreten model oldu.
- **Full Covariance**: Maksimum çapraz korelasyon (XCorr) ve Arias hatasında en iyi sonuçları vererek sinyal gücünü ve fazını en iyi koruyan modeldir.

*Not: Tüm testler eğitim verisiyle uyumlu olması için **100Hz** örnekleme hızında yapılmıştır.*
