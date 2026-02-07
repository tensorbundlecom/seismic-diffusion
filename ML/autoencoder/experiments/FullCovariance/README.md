# Full Covariance CVAE Experiment

Bu klasör, sismik veriler üzerinde "Full Covariance" (Tam Kovaryans) matrisi kullanan deneysel Koşullu Varyasyonel Oto-Kodlayıcı (CVAE) çalışmasını içerir. 

## 📂 Klasör Yapısı

- `core/`: Model mimarisi (`FullCovCVAE`) ve özel kayıp fonksiyonları (`loss_utils`).
- `training/`: Eğitim scriptleri ve hiperparametre ayarları.
- `evaluation/`: Eğitilmiş modelin sismik metriklerle (SSIM, LSD, DTW vb.) ve OOD (Out-of-Distribution) verileriyle testi.
- `simulation/`: Hiç yaşanmamış sismik senaryoların (Örn: Adalar M5.0) üretilmesi.
- `setup/`: Tarihsel OOD verilerinin IRIS/KOERI üzerinden indirilmesi ve ön işleme süreçleri.
- `results/`: Üretilen spektrogramlar, interaktif Plotly raporları ve CSV çıktıları.
- `utils_debug/`: Geliştirme sürecindeki analiz ve hata ayıklama kodları.

## 🚀 Temel Özellikler

1. **Gelişmiş Latent Space**: Sadece ortalama (mu) ve varyans (sigma) değil, gizli değişkenler arasındaki tam korelasyonu (Full Covariance) öğrenir.
2. **Fiziksel Koşullandırma**: Üretim süreci Deprem Büyüklüğü (Magnitude), Konum (Latitude, Longitude, Depth) ve İstasyon bilgisiyle şartlandırılmıştır.
3. **OOD Dayanıklılığı**: Model, eğitim setinde bulunmayan (2010-2015 arası) büyük tarihi depremler üzerinde test edilmiştir.

## 📈 Özet Bulgular

- **Zaman Uyumu**: Full Covariance modeli, Baseline modele göre **DTW (Dynamic Time Warping)** skorunda daha başarılıdır; yani sismik yapısal benzerliği zaman boyutunda daha iyi yakalar.
- **Spektral Doğruluk**: Baseline model, spektrogram netliği (SSIM) ve enerji korunumu (Arias Intensity) açısından şuan için daha kararlıdır.
- **Üretim**: Model, koordinat ve magnitude girildiğinde istenilen istasyon için gerçekçi sentetik sismogramlar üretebilmektedir.

## 🛠️ Kullanım

Scriptler, proje kök dizininden (root) veya kendi klasörlerinden `sys.path` düzeltmeleri sayesinde çalıştırılabilir. 

Örnek simülasyon çalıştırma:
```bash
python ML/autoencoder/experiments/FullCovariance/simulation/simulate_synthetic_event.py
```

---
*Hazırlayan: Antigravity AI Assistant*
