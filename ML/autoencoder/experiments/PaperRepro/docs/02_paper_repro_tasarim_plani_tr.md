# PaperRepro Tasarim Plani

## 1. Ana hedef

Amac su soruya temiz cevap vermek:

> Makaledeki ana tasarim ilkeleri, bizim veri ve degerlendirme rejimimiz altinda da bizim mevcut modellerimizden daha iyi calisiyor mu?

Bunu cevaplamak icin yeni hat su sekilde kurulacak:
- Stage-1: paper-faithful compressor VAE
- Stage-2: latent EDM
- Eval: paper + kalem metriklerinin birlikte raporlanmasi

## 2. Bu kutu neyi yapmayacak?

Asagidakiler bu ilk fazin disinda tutulacak:
- full covariance posterior denemeleri
- normalizing flow posterior denemeleri
- station embedding merkezli condition denemeleri
- true W-space denemeleri
- DDPM/DDIM'i tekrar ana eksen yapmak

Bu kutunun amaci paper-faithful temel hatti kurmaktir; ekstra arastirma eksenleri sonra gelir.

## 3. Sabit ilk kararlar

### 3.1 Yeni kutu izole olacak

Yeni klasor su konumda tutulacak:
- `ML/autoencoder/experiments/PaperRepro/`

Bu kutu:
- kendi `core/`, `training/`, `evaluation/`, `configs/`, `docs/` alanlarini tasiyacak
- tarihsel deney kutularina runtime bagimliligi olmayacak
- sadece veri kaynagindan yararlanacak

### 3.2 Split disiplini paper'dan degil bizden alinacak

Paper'daki random sample split alinmayacak.
Bizim taraftaki daha guvenilir pratik korunacak:
- frozen event-wise split
- ayri OOD kataloglari

### 3.3 Condition set fiziksel scalar predictor merkezli olacak

Ilk hedef condition set su olacak:
- magnitude
- hypocentral distance
- hypocenter depth
- azimuth / azimuthal-gap turevi bilgi
- VS30 ilk fazda kullanilmayacak

Not:
- VS30 bizde yoksa bu acik bir sapma olarak dokumante edilecek
- station embedding ilk paper-faithful hatta kullanilmayacak

### 3.4 Stage-1 VAE tiny-KL rejiminde kurulacak

Stage-1'in amaci su olacak:
- olabildigince yuksek sadakatli spectrogram sikistirma
- diffusion icin yerel yapiyi koruyan latent map uretme

Bunun icin:
- 2D convolutional encoder / decoder
- 2D latent tensor
- tiny KL agirligi

### 3.5 Stage-2 diffusion EDM olacak

Stage-2'de:
- latent uzerinde calisan 2D U-Net denoiser
- EDM loss / sigma schedule / sampling mantigi
- condition, low-dimensional scalar vector olarak enjekte edilecek

## 4. Kod mimarisi

Ilk hedef klasor yapisi:

- `ML/autoencoder/experiments/PaperRepro/README.md`
- `ML/autoencoder/experiments/PaperRepro/docs/`
- `ML/autoencoder/experiments/PaperRepro/configs/`
- `ML/autoencoder/experiments/PaperRepro/setup/`
- `ML/autoencoder/experiments/PaperRepro/core/`
- `ML/autoencoder/experiments/PaperRepro/training/`
- `ML/autoencoder/experiments/PaperRepro/evaluation/`
- `ML/autoencoder/experiments/PaperRepro/results/`
- `ML/autoencoder/experiments/PaperRepro/logs/`

Beklenen sorumluluklar:

### `setup/`
- bizim veri kaynagimizi paper-faithful formata ceviren builder scriptleri
- frozen split olusturma scriptleri
- global normalization istatistikleri
- metadata audit scriptleri

### `core/`
- spectrogram representation sinifi
- inverse representation yardimcilari
- dataset / datamodule
- Stage-1 VAE modeli
- Stage-2 EDM modeli
- U-Net bloklari
- condition embedding bloklari

### `training/`
- `train_autoencoder.py`
- `extract_latent_cache.py`
- `train_latent_edm.py`
- ortak launch / monitor yardimcilari

### `evaluation/`
- reconstruction eval
- prior / generation eval
- paper-style distribution metrics
- kalem-style spectral metrics
- runtime/resource raporlama
- OOD evaluation

## 5. Veri hazirligi icin zorunlu kontroller

Koda gecmeden once su maddeler netlesmeli ve script ile dogrulanmali:

### 5.1 Veri kontrati
- waveform sampling rate gercekten sabit mi?
- component set ne olacak? ilk hedef paper'a yakin olmak icin 3-component olmali
- pencere uzunlugu sabitlenebiliyor mu?
- tum kayitlar ayni fiziksel birimde mi?

### 5.2 Metadata kontrati
- magnitude her ornek icin var mi?
- event lat / lon / depth var mi?
- station lat / lon var mi?
- hypocentral distance dogrudan var mi, yoksa hesap mi gerekecek?
- azimuth veya azimuth-gap icin gerekli geometri hesaplanabiliyor mu?
- VS30 var mi?

### 5.3 Representation kontrati
- fixed/global normalization icin gerekli global istatistikler cikartilabiliyor mu?
- STFT boyutlari paper'a yakin tutulabiliyor mu?
- Griffin-Lim geri donus kalitesi kabul edilebilir mi?

## 6. Ilk implementasyon asamalari

### Asama 0 - Dokuman ve veri emniyeti
Cikti:
- mevcut checkout'a dokunmadan yeni clone ve branch
- karsilastirma notu
- bu tasarim plani

### Asama 1 - Dataset audit
Cikti:
- metadata var/yok raporu
- hangi condition alanlarinin dogrudan kullanilabilecegi
- hangi alanlarin hesaplanmasi gerektigi
- sabit pencere ve sampling karari

### Asama 2 - Frozen split ve global stats
Cikti:
- `frozen_event_splits_*.json`
- `global_representation_stats_*.json`
- train/val/test/OOD sayim raporu

### Asama 3 - Representation ve dataset builder
Cikti:
- waveform -> log-spectrogram donusumu
- fixed/global normalization
- inverse transform smoke testi
- training icin okunabilir dataset formati

### Asama 4 - Stage-1 VAE
Cikti:
- reconstruction train scripti
- checkpointler
- latent shape / recon kalite raporu
- ornek recon gorselleri

### Asama 5 - Latent cache cikarma
Cikti:
- Stage-1 encoder ciktilarinin cache'i
- train/val/test ayrimi korunmus latent artifactlari

### Asama 6 - Stage-2 EDM
Cikti:
- latent diffusion train scripti
- sampler scriptleri
- runtime / GPU / RAM loglari

### Asama 7 - Evaluation paketi
Cikti:
- paper-style metrikler
- kalem-style metrikler
- OOD raporu
- karsilastirma tablolari

## 7. Ilk fazda raporlanacak metrikler

### 7.1 Bizim metriklerimiz
- `spec_corr`
- `LSD`
- `MR-LSD`
- runtime
- GPU memory
- ornek basina inference suresi
- event-wise OOD sonucu

### 7.2 Paper'a yakin metrikler
- Fourier amplitude distribution karsilastirmasi
- classifier accuracy
- classifier embedding distance / Frechet benzeri metrik

Not:
- classifier tabanli metrikler ilk fazin zorunlu minimum paper-style paketine dahildir

## 8. En kritik riskler

### 8.1 VS30 eksigi
Eger veri tarafinda VS30 yoksa, paper ile birebir aynilik bozulur.
Bu durumda:
- ya acik deviation olarak raporlanir
- ya da site proxy icin ayri bir kaynak aranir

### 8.2 Azimuthal gap hesaplama riski
Paper predictor setindeki azimuthal gap, tek istasyonlu satir bazli veride dogrudan hazir olmayabilir.
Bunu dogru tanimlamadan feature eklemek yanlis olur.

### 8.3 Representation uyumsuzlugu
Bizim tarihsel per-sample normalization aliskanligimiz bu hatta tasinmamali.
Aksi halde paper-faithful olma iddiasi zayiflar.

### 8.4 Griffin-Lim tavan etkisi
Waveform kalitesinin ust sinirini Griffin-Lim belirleyebilir.
Bu nedenle reconstruction ve generation yorumlarinda representation ceiling'i ayri not edilmelidir.

## 9. Dogrudan sonraki isler

Koda gecmeden once siradaki zorunlu adimlar:
1. veri kontratini script ile cikarmak
2. condition alanlarinin hangilerinin gercekten mevcut oldugunu dogrulamak
3. pencere / sampling / component kararlarini dondurmek
4. PaperRepro kutusunun bos kod iskeletini kurmak

Bu dort adim tamamlanmadan model koduna gecilmeyecek.

## 10. Karar dosyasi

Bu plandaki acik noktalar ayri bir karar defterinde tutulur:
- `ML/autoencoder/experiments/PaperRepro/docs/03_acik_sorular_ve_karar_kaydi_tr.md`

Bu dosya, kodlamaya gecmeden once tartisilarak doldurulacaktir.
