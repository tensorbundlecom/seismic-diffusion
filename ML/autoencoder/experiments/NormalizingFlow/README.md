# Normalizing Flow CVAE Experiment

Bu klasör, sismik spektrogramların latent uzayını daha esnek bir şekilde modellemek için **RealNVP** tabanlı Normalizing Flow katmanlarını kullanan deneysel CVAE çalışmasını içerir.

## 📂 Klasör Yapısı

- `core/`: Deneye özel model mimarisi (`FlowCVAE`) ve akış tabanlı kayıp fonksiyonları (`loss_utils.py`). **Önemli**: Temel model ve veri yükleyici artık `General/core` üzerinden çekilmektedir.
- `training/`: Normalizing Flow modeline özel eğitim scriptleri.
- `evaluation/`: Model performansı ve OOD testleri.
- `results/`: Üretilen spektrogramlar ve model çıktıları.

## 🚀 Temel Özellikler

1. **Esnek Latent Dağılım**: Standart Gaussian yerine, Normalizing Flow katmanları ile daha karmaşık ve veriyle uyumlu bir latent dağılım öğrenir.
2. **RealNVP Katmanları**: Conditional Affine Coupling katmanları ile sismik sinyalin özelliklerine göre latent uzayı dönüştürür.
3. **Yüksek Sadakat**: SSIM metriklerinde baseline ve Full Covariance modellerine göre daha yüksek spektral benzerlik sağlar.

## 📈 Özet Bulgular

- **Spektral Detay**: Normalizing Flow, özellikle yüksek frekanslı sismik bileşenleri temsil etmede diğer modellerden daha başarılıdır.
- **Kayıp Değeri**: External veri seti (29GB) üzerinde en düşük final loss değerine bu model ulaşmıştır.
