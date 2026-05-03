# Paper'dan Izinli Sapmalar

## Amac

Bu not, `PaperRepro` kutusunun paper'a hangi noktalarda sadik kaldigini ve hangi noktalarda bilincli olarak saptigini acikca kayda gecirmek icin tutulur.

Buradaki sapmalar "rastgele fark" degil, veri ve problem gercegimiz nedeniyle alinmis kontrollu kararlardir.

## Izinli sapmalar

### 1. Split rejimi

- Paper tarafindaki random sample split yerine:
  - event-wise `train/val/test`
  - ayri `event-heldout OOD`

Bu sapma bilincli ve metodolojik olarak daha siki bir secimdir.

### 2. VS30 kullanilmamasi

- Paper predictor setinde `VS30` vardir.
- Bizim mevcut veri ve metadata envanterimizde hazir `VS30` yoktur.
- Harici esleme ilk faza alinmamistir.

### 3. Station elevation kullanilmamasi

- Paper veya benzeri fiziksel predictor setlerine eklenebilecek bir site feature olmasina ragmen,
- mevcut veri hattinda hazir station elevation metadata yoktur.

### 4. Station embedding'in ana modelde kullanilmamasi

- Station kimligi metadata olarak tutulur,
- ancak ana modelde condition olarak kullanilmaz.

Bu karar, fiziksel predictor seti ile istasyon ezberleme etkisini ayirmak icin alinmistir.

### 5. Azimuthal gap turetimi

- Paper tarafinda bu feature hazir predictor olarak kullanilir.
- Bizde ise `azimuthal_gap_deg`, phase bulletinlerden event-level scalar olarak turetilecektir.

Yani feature anlami korunur, fakat veri kaynagi/uretim yolu farklidir.

### 6. Veri kaynagi ve kanal kontrati

- Paper veri kaynagi yerine bizim `external_dataset` kullanilir.
- Ilk faz yalnizca `HH` aileli, `100 Hz`, `3-component` sample'lar ile sinirlidir.

### 7. Condition setinin mevcut/turetilebilir kisimlarla sinirli kalmasi

Ilk faz condition seti:

- `ML`
- `depth_km`
- `hypocentral_distance_km`
- `azimuth_sin`
- `azimuth_cos`
- `azimuthal_gap_deg`
- `tP_ref_s`
- `dtPS_ref_s`

Paper'daki butun predictor'lar birebir kopyalanmamis, yalnizca mevcut veya guvenli bicimde turetilebilenler alinmistir.

## Korunan temel hat

Asagidaki ana omurga korunur:

- `3-component`
- `100 Hz`
- `4064 sample`
- `magnitude-only log-spectrogram`
- `128 x 128` temsil
- `8 x 32 x 32` latent harita
- `Stage-1 VAE`
- `Stage-2 latent EDM`
- `Griffin-Lim` inversion

## Sonuc

Bu kutu birebir paper reproduksiyonu degildir.

Bu kutu, paper'daki temel model mantigini alip bizim veri rejimimize kontrollu sekilde uyarlayan bir ana hat kurmayi hedefler. Bu nedenle yukaridaki sapmalar izinli, sinirli ve acik sekilde kayda gecirilmis kabul edilir.
