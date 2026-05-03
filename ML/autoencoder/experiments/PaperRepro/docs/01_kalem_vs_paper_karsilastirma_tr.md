# Kalem vs Paper Karsilastirmasi

## 1. Kapsam

Bu not, uc seyi karsilastirir:
- bizim tarihsel `ML/autoencoder/experiments/*` deney cizgimiz
- `2410.19343v2` makalesindeki ana model fikri
- makaleye ait `tqdne` kod tabani uzerinden gordugumuz mimari kararlar

Amac, sadece "benzeyen yerler" listesini cikarmak degil; neden yeni bir dal acmamiz gerektigini netlestirmektir.

## 2. Ortak nokta nerede?

Iki tarafta da ana is akisi su aileye ait:
- waveform
- STFT / log-spectrogram
- autoencoder veya VAE ile sikistirma
- condition verilmis generative model
- tekrar waveform'a donus

Yani problem ailesi ortak.
Fark, bu aile icinde hangi noktaya modeling kapasitesi ayrildigi.

## 3. Bizim tarihsel cizgimiz ne yapti?

Bizim repo tarihsel olarak daha cok su sorular etrafinda ilerledi:
- CVAE dogrudan hedef STFT'yi ne kadar iyi uretebilir?
- posterior'u full covariance yapinca ne oluyor?
- latent'e flow ekleyince ne oluyor?
- deterministic condition embedding ekleyince ne oluyor?
- latent diffusion eklendiginde DDPM mi DDIM mi daha iyi?

Bu cizginin paper'a en yakin kutulari sunlar oldu:
- `ML/autoencoder/experiments/LegacyCondDiffusion/`
- `ML/autoencoder/experiments/DDPMvsDDIM/`

Ama bunlar bile paper ile tam ayni degil.

## 4. Kritik farklar

### 4.1 Stage-1'in rolu farkli

Bizde tarihsel olarak Stage-1 cogu zaman dogrudan generator gibi kullanildi.
Yani CVAE'nin kendisi condition alip hedef STFT'yi uretsin beklentisi baskindi.

Paper tarafinda Stage-1'in asil rolu farkli:
- once iyi bir sikistirma yapmak
- latent'i diffusion icin daha kolay modellenebilir hale getirmek

Bu fark kucuk degil.
Bu, butun sistemin nasil kurulacagini degistiriyor.

### 4.2 Latent geometrisi farkli

Bizim paper-adjacent hatlarimizda latent genelde su sekildeydi:
- flat vector latent
- ornek: `128-D` gibi tek vektor

Paper tarafinda latent su mantikta:
- 2D latent map
- spectro-temporal yerellik korunuyor
- diffusion denoiser bu yapiyi U-Net ile kullanabiliyor

Bu farkin pratik sonucu:
- bizim `vector latent + MLP denoiser` hatti, yerel yapilari tasimada dogal olarak daha zayif
- paper tarafi 2D latent uzerinde daha dogal bir diffusion problemi cozuyor

### 4.3 Diffusion omurgasi farkli

Bizim `DDPMvsDDIM/` kutusunda:
- latent vector
- `ResMLP` / `AdaLN-ResMLP`
- sampler ekseni: `DDPM` vs `DDIM`

Paper tarafinda:
- latent 2D tensor
- 2D `U-Net`
- `EDM` formulasyonu
- Heun-tipi sampling

Bu nedenle bizim onceki sonucumuz netti:
- `DDIM` bu frozen setup'ta `DDPM`'den daha iyi trade-off verdi
- ama asil darbogaz sampler degil, model tasarimi

### 4.4 Condition set farkli

Bizim tarihsel hatlarda condition genelde sunlari icerdi:
- magnitude
- event location `(lat, lon, depth)`
- station embedding

Paper tarafinda condition daha fiziksel ve daha GMPE/GMM benzeri:
- hypocentral distance
- magnitude
- VS30
- hypocenter depth
- azimuthal gap

Bu fark cok onemli.
Cunku station embedding agirlikli bir tasarim ile fiziksel scalar predictor agirlikli tasarim ayni sey degil.

### 4.5 Normalization farkli

Bizim ana STFT dataset hattimizda tarihsel olarak per-sample min-max normalizasyon kullandik.
Bu, ornekler arasi amplitude semantiklerini zayiflatabiliyor.

Paper tarafinda:
- log-magnitude temsil
- fixed/global normalization
- veri seti boyunca ayni olcek mantigi

Paper-faithful yeni hat icin bu farki korumak gerekir.

### 4.6 Split disiplini farkli

Paper kodunda random sample split goruluyor.
Bizim son donem en iyi pratiklerimizde ise:
- event-wise split
- ayri OOD setleri
- custom evaluation kataloglari

Burada paper'i taklit etmek dogru olmaz.
Makaledeki mimariyi alip, bizim daha siki split disiplinimizle test etmek daha dogru olur.

### 4.7 Evaluation felsefesi farkli

Bizim guclu oldugumuz yerler:
- `spec_corr`
- `LSD`
- `MR-LSD`
- runtime / kaynak kullanimi
- event-wise OOD
- waveform / spectrogram gorselleri

Paper tarafinin guclu oldugu yerler:
- classifier-based evaluation
- Frechet-style distribution metrikleri
- Fourier amplitude distribution karsilastirmalari

Yeni hat icin dogru yol, iki tarafin da iyi kisimlarini birlestirmektir.

## 5. Neden patch degil yeni dal?

Mevcut `experiments/*` kutularinin hicbiri su dordunu birlikte tasimiyor:
- 2D latent compressor VAE
- 2D latent U-Net EDM
- paper benzeri condition set
- paper-faithful normalization

Bu nedenle eski kutular uzerine parca parca patch atmak yerine, yeni bir deney kutusu kurmak daha dogrudur.
Aksi halde hibrit ve yorumu zor bir sistem cikar.

## 6. Bizden korunacak seyler

Yeni hatta sunlar korunmali:
- event-wise split disiplini
- ayri OOD degerlendirmesi
- detayli log ve artifact kaydi
- runtime / GPU / RAM / sure raporlama
- net markdown dokumantasyonu
- terim disiplinine dikkat

## 7. Paper'dan alinacak seyler

Yeni hatta sunlar alinmali:
- Stage-1'i generator degil compressor gibi kurmak
- tiny-KL rejimi
- 2D latent map
- latent diffusion icin 2D U-Net
- EDM egitimi ve sampling mantigi
- fiziksel scalar predictor condition set
- classifier / distribution tabanli ek metrikler

## 8. Dogrudan sonuc

Bu karsilastirmanin sonucu su:
- Eski hatlarimizi biraz daha ince ayarlamak ana cevabi vermeyecek.
- En mantikli sonraki adim, paper-faithful ama bizim split/eval disipliniyle uyumlu yeni bir model hatti kurmaktir.
- Bu nedenle yeni branch ve yeni izole klasor acilmasi dogru karardir.
