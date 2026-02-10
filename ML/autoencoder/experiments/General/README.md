# General Experiment Hub

Bu dizin, `kalem_seismic` projesindeki tüm oto-kodlayıcı deneyleri için ortak çekirdek mantığı, veri çekme araçlarını ve genel değerlendirme scriptlerini barındıran merkezi yönetim birimidir.

## 📦 Klasör Yapısı
### 📂 Directory Structure

#### `General/core`
Shared logic for all models.
- `stft_dataset.py`: Unified STFT dataset loader (handles both HH and BH)
- `model_baseline.py`: Base CVAE architecture (Encoder/Decoder)

#### `General/setup`
Data preparation scripts.
- `download_post_training_ood.py`: Downloads 10 diverse HH channel events (2022-2024)
- `preprocess_post_training_ood.py`: Preprocessing pipeline (100Hz, 0.5-45Hz)
- `archive/`: Older setup scripts (including initial BH channel attemps)

#### `General/evaluation`
Evaluation tools.
- `evaluate_post_training_ood.py`: **Main evaluation script** (HH OOD)
- `evaluate_diverse_ood.py`: Comparison script (Reference)
- `archive/`: Deprecated debugging and analysis tools

### 4. `checkpoints/`
Eğitilmiş model ağırlıklarının (Best-case) saklandığı dizin.

### 5. `visualizations/`
Deneyler sonucunda üretilen karşılaştırmalı grafikler, dalga formu yığınları ve spektrogram gridleri.

---

## 🌍 OOD (Out-of-Distribution) Veri Seti Detayları

**Post-Training HH Channel Dataset** (2022-2024) - Eğitim sonrası dönemden, enstrüman uyumlu (HH kanalları) 10 deprem:

| Kod | Tarih | Saat | Enlem | Boylam | Derinlik (km) | Büyüklük | Bölge |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **OOD_POST_01** | 2022.07.21 | 15:44:25 | 40.143 | 27.387 | 24.4 | M4.3 | Marmara Bölgesi |
| **OOD_POST_02** | 2022.12.13 | 03:21:17 | 40.352 | 27.026 | 10.0 | M4.2 | Marmara Bölgesi |
| **OOD_POST_03** | 2023.05.04 | 01:50:01 | 40.431 | 26.225 | 10.0 | M4.2 | Saros Körfezi |
| **OOD_POST_04** | 2023.11.07 | 20:05:47 | 40.497 | 27.531 | 11.8 | M4.1 | Marmara Bölgesi |
| **OOD_POST_05** | 2023.12.04 | 07:42:19 | 40.438 | 28.856 | 6.5 | M5.1 | Marmara Denizi |
| **OOD_POST_06** | 2023.12.17 | 20:53:53 | 40.730 | 29.059 | 12.4 | M4.2 | Marmara Denizi |
| **OOD_POST_07** | 2024.01.27 | 03:17:35 | 40.516 | 28.812 | 10.0 | M3.0 | Marmara Denizi |
| **OOD_POST_08** | 2024.02.27 | 13:09:54 | 40.297 | 26.977 | 12.0 | M4.2 | Marmara Bölgesi |
| **OOD_POST_09** | 2024.05.26 | 21:38:19 | 40.818 | 28.308 | 15.1 | M3.3 | Marmara Denizi |
| **OOD_POST_10** | 2024.12.12 | 11:34:52 | 40.459 | 26.168 | 10.0 | M4.2 | Saros Körfezi |

---

## 📊 Model Performans Karşılaştırması (Post-Training HH OOD)

Eğitim sonrası dönemden (2022-2024) seçilen 10 deprem (M3.0-M5.1) üzerinde **HH kanalları** ile yapılan değerlendirme (56 waveform):

| Model | SSIM ↑ | S-Corr ↑ | SC ↓ | STA/LTA Err ↓ | LSD ↓ | MR-LSD ↓ | Arias Err ↓ | Env Corr ↑ | DTW ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline CVAE** | **0.6153** | 0.9399 | 0.262 | 0.069 | 2.02 | **1.65** | 0.47 | 0.3759 | **6615.67** |
| **Full Covariance** | 0.5939 | 0.9333 | 0.279 | 0.076 | **1.88** | 1.77 | **0.43** | 0.3551 | 7110.65 |
| **Normalizing Flow** | 0.5980 | **0.9458** | **0.247** | **0.054** | 2.11 | 2.11 | **0.43** | **0.3815** | 6742.01 |

**Gözlemler:**
- **Enstrüman Uyumu**: HH kanalları kullanılarak yapılan bu değerlendirme, eğitim dataseti ile tam uyumlu olduğu için geçerli bir OOD testidir.
- **Normalizing Flow**: Spektrogram yapısal benzerliğinde (SSIM) ve zarf korelasyonunda (Env Corr) en iyi performansı göstererek en "doğal" sismogramları üreten model olmuştur.
- **Full Covariance**: Spektral mesafe (LSD) ve enerji korunumunda (Arias Err) liderliğini sürdürerek fiziksel doğruluğu en iyi koruyan modeldir.
- **DTW Skorları**: Tüm modeller zamansal hizalamada BH testlerine göre çok daha iyi performans gösterdi, bu da enstrüman uyumunun önemini doğruluyor.

*Not: Tüm testler eğitim verisiyle uyumlu olması için **100Hz** örnekleme hızında yapılmıştır.*
