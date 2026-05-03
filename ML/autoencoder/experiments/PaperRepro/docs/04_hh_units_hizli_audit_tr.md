# HH Units Hizli Audit

## Amac

Bu notun amaci, `PaperRepro` ilk fazi icin kullanilacak `HH filtered` verinin fiziksel birim ve temel waveform kontrati acisindan hizli bir audit ozetini kayda gecirmektir.

Bu auditin amaci:
- verinin ilk faz icin kullanilabilir olup olmadigini hizlica dogrulamak
- `counts as-is` kararini teknik olarak temellendirmek
- tam instrument-response standardizasyonunu ilk fazdan bilincli olarak ayirmak

Bu auditin amaci olmayan sey:
- tum istasyonlar icin kesin `m/s` veya `m/s^2` fiziksel birim dogrulamasi yapmak
- response removal pipeline'i kurmak

## Kaynaklar

- `data/external_dataset/extracted/data/filtered_waveforms/HH`
- `data/external_dataset/extracted/data/waveform_summary.csv`
- `ML/autoencoder/experiments2/protocol/frozen/base/manifest_exp001.jsonl`
- `ML/autoencoder/experiments2/protocol/docs/exp001_problem_and_design.md`
- `ML/autoencoder/experiments2/protocol/docs/decisions.md`

## Hizli Bulgular

### 1. Onceki protocol karari

`experiments2` tarafinda daha once su karar acikca alinmis:

- `HH counts (as-is)`
- `Instrument response removal uygulanmayacak`

Bu iz su dosyalarda acik:

- `ML/autoencoder/experiments2/protocol/docs/exp001_problem_and_design.md:178`
- `ML/autoencoder/experiments2/protocol/docs/exp001_problem_and_design.md:179`
- `ML/autoencoder/experiments2/protocol/docs/decisions.md:694`

Yani bu veri zaten onceki sistemde fiziksel birime cevrilmis bir waveform olarak degil, counts-domain HH veri olarak kullanilmis.

### 2. Response metadata taramasi

`external_dataset` altinda su tip dosyalar icin hizli bir tarama yapildi:

- `*.xml`
- `*.stationxml`
- `*.resp`
- `SACPZ*`
- `*.pz`

Sonuc:

- hizli taramada `external_dataset` altinda response removal icin hazir bir StationXML/RESP envanteri bulunmadi

Bu bulgu tek basina "imkansiz" demek degil, ama ilk fazda response-removed fiziksel birim standardizasyonunun hazir ve dogrudan elde olmadigini gosteriyor.

### 3. HH waveform kontrati

`manifest_exp001.jsonl` uzerinden kullanilan `HH` setindeki tum dosyalar hizli script ile tarandi.

Tarama kapsami:

- toplam dosya: `88919`

Sonuc:

- okuma hatasi: `0`
- trace sayisi: tum dosyalarda `3`
- component seti: tum dosyalarda `E/N/Z`
- sampling rate: tum dosyalarda `100 Hz`
- npts: tum dosyalarda `7001`
- sure: tum dosyalarda `70.0 s`

Bu, ilk faz veri kontrati acisindan cok temiz bir durumdur.

### 4. Ham ornek veri tipi

Ilk birkac `HH` dosyasinin trace verileri okunup kontrol edildi.

Gozlem:

- Obspy okuduktan sonra array dtype `float64`
- genlikler counts-benzeri buyukluklerde
- fakat bu tek basina response-corrected fiziksel birim oldugunu gostermiyor

Bu nedenle bu bulgu sadece "okunabilir ve sayisal olarak tutarli" anlamina gelir; fiziksel birim kaniti olarak kullanilmaz.

## Karar

`PaperRepro` ilk fazinda:

- `HH filtered` veri kullanilacak
- veri `counts-domain as-is` kabul edilecek
- instrument response removal uygulanmayacak
- bu durum raporda acik limitation olarak yazilacak

## Sinir

Bu audit su soruyu cevaplamaz:

- "Tum istasyonlar kesin olarak ayni fiziksel birimde mi?"

Bu sorunun kesin cevabi icin ayri bir response metadata / inventory calismasi gerekir. Bu ise ilk fazi gereksiz yere bloklayan daha buyuk bir preprocessing isi olur.

## Sonuc

Ilk faz icin veri tarafinda blocker gorulmedi.

Counts-domain HH filtered veri:

- yapisal olarak tutarli
- onceki protocol ile uyumlu
- hizli audit ile dogrulanmis

Bu nedenle ilk fazin veri birim karari guvenli sekilde dondurulabilir.
