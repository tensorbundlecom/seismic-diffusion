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
