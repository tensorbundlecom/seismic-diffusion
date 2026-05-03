# Acik Sorular ve Karar Kaydi

## 1. Amac

Bu dosya, `PaperRepro` kutusunu kodlamaya baslamadan once cevaplanmasi gereken tum temel sorulari toplar.
Amac:
- kararlari daginik tutmamak
- once neyin dondurulecegini netlestirmek
- model koduna gecmeden once acik noktayi minimuma indirmek

Bu dosya bir "checklist" degil, bir "karar defteri" olarak kullanilacak.
Her soru icin tartisma sonunda su alanlar doldurulacak:
- `Karar`
- `Gerekce`
- `Dosya / artifact`
- `Durum`

Durum alanlari:
- `Acik`
- `Karar verildi`
- `Script ile dogrulandi`
- `Sapma olarak kaydedildi`

## 2. Tartisma sirasi

Sorular asagidaki sirayla kapatilacak:
1. kapsam ve basari tanimi
2. veri kontrati
3. metadata ve condition kontrati
4. representation kontrati
5. Stage-1 VAE kontrati
6. Stage-2 EDM kontrati
7. evaluation kontrati
8. operasyon ve artifact kontrati
9. paper'dan sapmalar

Bu siralama bilerek secildi.
Veri ve temsil kararlari dondurulmadan model mimarisi dondurulmayacak.

## 3. Kapsam ve basari tanimi

### Q01. Bu kutunun birincil hedefi tam olarak nedir?
- Neden kritik: "paper-faithful yeniden yazim" ile "bizim veri uzerinde daha iyi model kurma" ayni hedef degil.
- Tartisilacak secenekler:
  - makaleye olabildigince sadik reproduksiyon
  - paper ilkelerini alip bizim veri rejimine gore uyarlanmis yeni ana model
  - bu ikisinin iki asamali yapisi
- Karar: `Paper'daki temel model mantigini referans alip, bizim veri yapimiza ve hedef problemimize gore uyarlanmis yeni ana model gelistirmek.`
- Gerekce: `Amac birebir paper reproduksiyonu degil; ayni gorev icin daha kaliteli ve bizim veri rejimimize gercekten uyan bir model kurmak. Bu nedenle paper'daki yararli tasarim ilkeleri alinacak, fakat veri ve metadata gercegimize uymayan kisimlar bilincli sekilde degistirilecek.`
- Dosya / artifact: `ML/autoencoder/experiments/PaperRepro/docs/02_paper_repro_tasarim_plani_tr.md`
- Durum: `Karar verildi`

### Q02. Bu kutuda ana karsilastirma kime karsi yapilacak?
- Neden kritik: "bizim mevcut modeller" ifadesi belirsiz kalirsa raporlar savunulamaz olur.
- Tartisilacak secenekler:
  - sadece `DDPMvsDDIM`
  - `LegacyCondDiffusion` + `DDPMvsDDIM`
  - Stage-1 icin ayri, Stage-2 icin ayri baseline seti
- Karar: `Ilk faz ana karsilastirma sadece ML/autoencoder/experiments/DDPMvsDDIM ile yapilacak.`
- Gerekce: `Bu kisim gereksiz yere uzatilmayacak. Yeni hattin en yakin ve en anlamli mevcut referansi DDPMvsDDIM kutusu. Ilk fazda odakli ve okunur bir karsilastirma tercih edildi.`
- Dosya / artifact: `ML/autoencoder/experiments/DDPMvsDDIM/README.md`
- Durum: `Karar verildi`

### Q03. "Basarili" sayilmak icin hangi minimum sonuc gerekiyor?
- Neden kritik: basari kriteri basta yazilmazsa sonradan hedef kayar.
- Tartisilacak eksenler:
  - spectral fidelity
  - waveform realism
  - OOD dayanikliligi
  - runtime / kaynak kullanimi
  - paper-style distribution metrics
- Karar: `Minimum basari kriteri, kalite + OOD odakli olacak. Ana kalite metrikleri olarak bizim mevcut spectral metriklerimiz kullanilacak; buna ek olarak makaledeki ise yarar distribution / realism metrikleri de rapora eklenecek. Runtime ve compute maliyeti ilk faz basari kapisina dahil edilmeyecek.`
- Gerekce: `Ilk hedef, mevcut DDPMvsDDIM referansina gore gercek kalite iyilesmesi gormek. Bu nedenle sadece IID test degil OOD davranisi da zorunlu tutulacak. Ayni anda hem bizim kullanisli kalite metriklerimizi hem de makaledeki anlamli metrikleri tasimak daha saglam bir degerlendirme verir. Ancak ilk fazda compute maliyetini basari tanimina sokmak erken ve gereksiz kisitlayici olur.`
- Dosya / artifact: `ML/autoencoder/experiments/PaperRepro/docs/02_paper_repro_tasarim_plani_tr.md`
- Durum: `Karar verildi`

### Q04. Ilk fazda hangi seyleri bilincli olarak disarida birakiyoruz?
- Neden kritik: scope creep olursa yeni kutu yine karisir.
- Aday konular:
  - full covariance posterior
  - normalizing flow posterior
  - station embedding
  - W-space varyantlari
  - DDPM/DDIM tekrar benchmark
  - classifier-free guidance
- Karar: `Ilk fazda full covariance, normalizing flow, W-space, DDPM/DDIM ek benchmark ve classifier-free guidance disarida birakilacak. Station embedding ve SNR embedding de ilk ana hatta alinmayacak; bunlar ancak sonraki fazlarda ayrica denenebilecek. Buna karsilik derived physics feature tarafinda travel-time ailesi ilk faz condition adaylari icinde tutulacak.`
- Gerekce: `Ilk hedef temiz bir ana hat kurmak. Ezberleme riski tasiyan veya problemi gereksiz buyuten embedding ve posterior varyantlari ilk fazi bulandirir. Travel-time turevleri ise inference aninda hesaplanabilir, fiziksel olarak yorumlanabilir ve onset/geometri bilgisini dogrudan tasiyabilir.`
- Dosya / artifact: `ML/autoencoder/experiments/PaperRepro/docs/02_paper_repro_tasarim_plani_tr.md`
- Durum: `Karar verildi`

## 4. Veri kontrati

### Q05. Ilk fazda hangi veri kaynagi kullanilacak?
- Neden kritik: farkli veri kaynaklari farkli metadata ve farkli filtreleme mantigi getirir.
- Tartisilacak secenekler:
  - sadece `external_dataset`
  - restricted / filtered tarihsel set
  - ikili protokol
- Karar: `Ilk fazin resmi veri kaynagi external_dataset olacak. Kucuk subsetler sadece smoke/debug amacli kullanilabilecek, ancak raporlanan asil sonuclar external_dataset tabanli olacak.`
- Gerekce: `Yeni ana model hatti en genis ve en ciddi veri tabani uzerinde kurulacak. Bu veri kaynagi, dalga dosyalari, event katalogu, phase-pick arsivi ve onceki manifest turevleri ile birlikte kullanilabilir durumda. Ayrica son bircok tarihsel deney de ayni veri kaynagiyla yapildigi icin mevcut referanslarla karsilastirma acisindan da en dogru taban bu.`
- Dosya / artifact: `ML/autoencoder/experiments2/protocol/frozen/base/manifest_exp001.jsonl.meta.json`
- Durum: `Karar verildi`

### Q06. Ornek birimi tam olarak nedir?
- Neden kritik: "bir sample = event-station-window" mi, yoksa daha farkli bir sey mi oldugu net olmali.
- Sorulacak alt noktalar:
  - tek kayit bir event-station eslesmesi mi?
  - 3 component tek ornek icinde mi tutulacak?
  - ayni eventten birden fazla pencere olacak mi?
- Karar: `Bir sample, tek bir event-station eslesmesine ait sabit uzunluklu tek bir zaman penceresi olacak. Bu sample icinde 3 component birlikte tutulacak. Ilk fazda ayni event-station ciftinden birden fazla pencere uretilmeyecek.`
- Gerekce: `Hedef problem, condition verip tek bir istasyondaki deprem kaydini uretmek. Bu hedefin en dogru veri birimi event-station-window tanimidir. 3-component yapinin korunmasi, polarizasyon ve componentler arasi iliskiyi tasimak icin gerekli. Coklu pencere veya multi-station sample yapisi ise ilk fazda gereksiz karmasiklik yaratir.`
- Dosya / artifact: `ML/autoencoder/experiments/PaperRepro/docs/02_paper_repro_tasarim_plani_tr.md`
- Durum: `Karar verildi`

### Q07. Hedef sampling rate sabit mi, degilse nasil standardize edilecek?
- Neden kritik: Stage-1 representation ve Stage-2 latent boyutu buna bagli.
- Karar: `Ilk fazda sadece HH kanal ailesi kullanilacak ve sampling rate tek, sabit bir deger olmak zorunda olacak. Veri audit asamasinda HH ailesi icinde sampling rate sapmasi olup olmadigi script ile dogrulanacak; sapma varsa veri standardize edilmeden model asamasina gecilmeyecek.`
- Gerekce: `Farkli sampling rate'leri veya farkli kanal ailelerini ilk fazda karistirmak representation kontratini ve frekans ekseni yorumunu bulandirir. Temiz bir ana hat icin tek kanal ailesi ve tek sampling rate zorunlu. HH hem veri hacmi hem de onceki deneylerle tutarlilik acisindan en dogru baslangic noktasi.`
- Dosya / artifact: `ML/autoencoder/experiments/PaperRepro/docs/02_paper_repro_tasarim_plani_tr.md`
- Durum: `Karar verildi`

### Q08. Hedef component set ne olacak?
- Neden kritik: paper 3-component calisiyor; bizim onceki pek cok hatti component bazinda farkli ilerledi.
- Tartisilacak secenekler:
  - ENZ birlikte
  - sadece Z
  - iki asamali: once Z, sonra 3-component
- Karar: `Ilk fazda kati 3-component zorunlulugu olacak. Her sample, ayni event-station-window'a ait 3 componenti eksiksiz icerecek. Eksik component iceren kayitlar modele alinmayacak. Component order sabit olacak; isimlendirme veri audit ile dogrulanip dondurulecek.`
- Gerekce: `Hedef problem 3-component deprem kaydi uretmek. Bu nedenle tensor anlami her sample icin ayni kalmali. Eksik component'e tolerans tanimak, ayni sekilli ama farkli fiziksel icerikli girdiler olusturur ve ilk fazi bulandirir. Temiz ana hat icin 3-component kontrati kati tutulacak.`
- Dosya / artifact: `ML/autoencoder/experiments/PaperRepro/docs/02_paper_repro_tasarim_plani_tr.md`
- Durum: `Karar verildi`

### Q09. Pencereleme mantigi ne olacak?
- Neden kritik: origin-time aligned, P-aligned veya sabit pre/post pencere secimi condition anlami ve onset yorumunu degistirir.
- Tartisilacak alt noktalar:
  - pencere referansi
  - pencere uzunlugu
  - pre-event gurultu payi
  - uzun sinyalin kesilmesi / kisa sinyalin pad edilmesi
- Karar: `Ilk fazda pencereleme, event origin-time referansli sabit bir kuralla yeniden tanimlanacak. Pencere, P-arrival'a gore hizalanmayacak ve mevcut external pencere semantigi oldugu gibi miras alinmayacak. Pre-event marj ve toplam pencere suresi ayrica netlestirilecek.`
- Gerekce: `Travel-time ve benzeri fiziksel condition'lari gercekten anlamli kullanmak icin zaman referansinin fiziksel olarak temiz olmasi gerekir. P-alignment yapilirsa travel-time bilgisinin bir kismi veri tarafinda ezilir. Mevcut hazir external pencere ise hizli ama bulanık bir kontrat verir. Bu nedenle origin-time referansli sabit pencere en dogru ana yon.`
- Dosya / artifact: `ML/autoencoder/experiments/PaperRepro/docs/02_paper_repro_tasarim_plani_tr.md`
- Durum: `Karar verildi`

### Q10. Ters donusum icin tum ornekler sabit uzunlukta mi olacak?
- Neden kritik: 2D spectrogram boyutu ve batchleme mantigi buna bagli.
- Tartisilacak secenekler:
  - kati sabit pencere
  - sabit pencere + mask
  - degisken uzunluk yasak
- Karar: `Tum ornekler sabit uzunlukta olacak. Origin-time referansli pencere, pre-origin marj olmadan baslayacak ve toplam uzunluk 4064 sample (yaklasik 40.64 s @ 100 Hz) olacak. Ilk fazda maskli degisken uzunluk yaklasimi kullanilmayacak. Ham kayit hedef pencereyi tam kapsamadiginda zero-pad ile sabit tensor kontrati korunacak; bu durum artifact olarak ayrica kaydedilecek.`
- Gerekce: `2D spectrogram + latent map + U-Net akisi icin sabit tensor kontrati en temiz cozum. Paper tarafinda da pratikte 4064 sample kullaniliyor. Bizim veride teorik tP ve tS dagilimlari bu uzunlugun P ve S fazlarini buyuk oranda kapsayacak kadar yeterli oldugunu gosteriyor. Ayrica mevcut external pencereler origin'e gore temiz olmadigi icin yeni ve deterministik bir origin-time kontrati kurmak daha dogru.`
- Dosya / artifact: `ML/autoencoder/experiments/PaperRepro/docs/02_paper_repro_tasarim_plani_tr.md`
- Durum: `Karar verildi`

### Q11. Fiziksel birim tutarliligi var mi?
- Neden kritik: counts, velocity, acceleration karismasi olursa paper-style condition/genlik yorumu bozulur.
- Tartisilacak alt noktalar:
  - instrument correction var mi
  - tum istasyonlar ayni birimde mi
  - birim net degilse ne yapilacak
- Karar: `Ilk fazda HH filtered veri counts-domain as-is kullanilacak. Instrument response removal uygulanmayacak. Fiziksel birim konusu hizli audit ile dokumante edilecek, fakat m/s veya m/s^2 seviyesinde tam fiziksel birim standardizasyonu ilk fazi bloke etmeyecek.`
- Gerekce: `Tam response audit ve fiziksel birim donusumu buyuk bir on-is olur. Buna karsin elimizde hem onceki protocol kararlarinda hem de mevcut HH envanterinde tutarli bir counts-domain kullanim izi var. 88919 HH kaydinin hizli auditinde tum sample'lar 3 trace, ENZ, 100 Hz ve 7001 sample olarak tutarli cikti; external_dataset altinda StationXML/RESP benzeri response dosyasi da bulunmadi. Bu nedenle ilk faz icin en temiz ve hizli karar HH filtered counts-domain semantigini koruyup bunu acik limitation olarak yazmaktir.`
- Dosya / artifact: `ML/autoencoder/experiments/PaperRepro/docs/04_hh_units_hizli_audit_tr.md`
- Durum: `Script ile dogrulandi`

### Q12. Veri dengesizligi nasil ele alinacak?
- Neden kritik: magnitude ve distance dagilimlari egitimi ciddi seklide kaydirabilir.
- Tartisilacak secenekler:
  - tamamen dogal sampling
  - weighted sampler
  - sadece evaluation'da bin-wise rapor
  - train'de yumusak agirliklandirma + eval'da bin-wise rapor
- Karar: `Ilk fazda train tarafinda soft magnitude-weighted sampler kullanilacak. Evaluation tarafinda ise global metriklere ek olarak magnitude-bin ve distance-bin bazli raporlama yapilacak. Hard-balanced sampler uygulanmayacak.`
- Gerekce: `Manifest auditinde 88919 sample icinde [0,2) ve [2,3) magnitude araliklari baskin, [5,10) araliginda ise sadece 22 sample ve 3 event bulunuyor. Bu nedenle tamamen dogal sampling buyuk magnitudeleri neredeyse yok sayar, hard balance ise asiri seyrek ust kuyruğu ezberletme riski tasir. En dengeli yol, train'de yumusak magnitude agirliklandirmasi yapip performansi hem magnitude hem distance binleri uzerinden ayri raporlamaktir.`
- Dosya / artifact: `ML/autoencoder/experiments/PaperRepro/docs/03_acik_sorular_ve_karar_kaydi_tr.md`
- Durum: `Script ile dogrulandi`

## 5. Metadata ve condition kontrati

### Q13. Ilk faz condition listesi tam olarak ne olacak?
- Neden kritik: paper'a yakinlik ve bizim veri gercegimiz burada belirleniyor.
- Adaylar:
  - magnitude
  - hypocentral distance
  - depth
  - azimuth
  - azimuthal gap
  - VS30
  - station elevation
  - travel-time turevleri
- Karar: `Ilk faz condition seti su alanlardan olusacak: magnitude, depth_km, hypocentral_distance_km, azimuth_sin, azimuth_cos, azimuthal_gap_deg, tP_ref_s, dtPS_ref_s. VS30 ve station elevation ilk fazda kullanilmayacak.`
- Gerekce: `Bu liste paper'daki fiziksel predictor mantigina yeterince yakindir, ama yalnizca mevcut veya mevcut veriden turetilebilir alanlari kullanir. repi_km yerine hypocentral distance daha dogru geometri verir. Azimuth derece yerine sin/cos olarak verilecektir. tP_ref_s ve dtPS_ref_s birlikte hem ilk gelis zamanini hem de P-S ayrimini tasir. VS30 ve station elevation ise su an elde hazir olmadigi icin ilk fazi gereksiz yere bloke etmeden disarida birakilmistir.`
- Dosya / artifact: `ML/autoencoder/experiments/PaperRepro/docs/03_acik_sorular_ve_karar_kaydi_tr.md`
- Durum: `Karar verildi`

### Q14. Magnitude tipi ne olacak?
- Neden kritik: ML, Mw, Md gibi tipler karisiyorsa condition semantigi bozulur.
- Karar: `Ilk fazda magnitude condition'i olarak ML kullanilacak. ML eksik olan cok az sayidaki kayit veri hazirlama asamasinda disarida birakilacak.`
- Gerekce: `Event katalog auditinde ML kolonu neredeyse tamamen dolu, Mw ise cok seyrek, diger magnitude kolonlari ise ilk faz icin kullanisli degil. xM teorik olarak daha dolu olsa da semantigi daha karisik olabilir. Tek tip ve temiz bir condition semantigi icin ML en dogru secimdir.`
- Dosya / artifact: `ML/autoencoder/experiments/PaperRepro/docs/03_acik_sorular_ve_karar_kaydi_tr.md`
- Durum: `Script ile dogrulandi`

### Q15. Hypocentral distance dogrudan var mi, yoksa hesaplanacak mi?
- Neden kritik: paper condition setinin ana kolonlarindan biri bu.
- Tartisilacak alt noktalar:
  - yatay distance mi
  - hypocentral mi
  - depth hangi tanimla eklenecek
- Karar: `Ilk fazda distance condition'i dogrudan repi_km olarak alinmayacak; her sample icin hypocentral_distance_km = sqrt(repi_km^2 + depth_km^2) seklinde hesaplanacak ve tek distance kolonu olarak bu kullanilacak. depth_km ayri condition olarak kalacak.`
- Gerekce: `Manifestte repi_km ve depth_km tam dolu oldugu icin hypocentral distance turetimi maliyetsizdir. Hypocentral distance, yalnizca yatay uzaklik kullanmaktan daha dogru bir fiziksel geometri verir ve paper'daki predictor mantigina daha yakindir. repi_km ile depth_km zaten varken bunlara ek olarak repi ve hypocentral'i birlikte vermek gereksiz redundancy yaratir.`
- Dosya / artifact: `ML/autoencoder/experiments/PaperRepro/docs/03_acik_sorular_ve_karar_kaydi_tr.md`
- Durum: `Script ile dogrulandi`

### Q16. Azimuth ve azimuthal gap bu veride ne anlama gelecek?
- Neden kritik: tek istasyonlu satir bazli veri ile network-level azimuthal gap ayni sey olmayabilir.
- Tartisilacak secenekler:
  - event-to-station azimuth
  - network-level azimuthal gap
  - bu feature'in ilk fazda cikarilmasi
- Karar: `Ilk fazda azimuth ve azimuthal gap birlikte kullanilacak. Azimuth, event-to-station yon bilgisini tasiyan azimuth_sin ve azimuth_cos olarak kalacak. Azimuthal gap ise phase bulletinlerden event-level bir scalar ozellik olarak turetilip azimuthal_gap_deg adiyla eklenecek.`
- Gerekce: `Azimuth ile azimuthal gap ayni fiziksel bilgiyi tasimaz. Azimuth sample-bazli yol geometrisini, azimuthal gap ise event-level network coverage olcusunu verir. Bu nedenle ikisini birlikte kullanmak mantiklidir. Azimuth'i derece olarak degil sin/cos olarak vermek acisal sureksizlik sorununu onler. Gap tum istasyonlar icin event bazinda ayni olacak, fakat bu event-level bir condition olarak kabul edilebilir.`
- Dosya / artifact: `ML/autoencoder/experiments/PaperRepro/docs/03_acik_sorular_ve_karar_kaydi_tr.md`
- Durum: `Script ile dogrulandi`

### Q17. VS30 mevcut mu? Degilse ne yapilacak?
- Neden kritik: paper predictor setinin ana bilesenlerinden biri.
- Tartisilacak secenekler:
  - dogrudan kullan
  - harici kaynakla esle
  - proxy kullan
  - ilk fazda cikar ve deviation olarak kaydet
- Karar: `VS30 ilk fazda kullanilmayacak. Harici VS30 esleme veya proxy feature uretilmeyecek. Bu eksik bilgi acik bir sapma olarak dokumante edilecek. Station embedding, VS30 yerine gecen bir feature olarak kabul edilmeyecek.`
- Gerekce: `Mevcut veri ve repo taramasinda hazir VS30 metadata bulunmadi. Harici esleme ilk fazi gereksiz yere uzatir, proxy kullanimi ise fiziksel semantigi bulandirir. Station embedding site bilgisinin bir kismini dolayli olarak tasiyabilir, ancak fiziksel, yorumlanabilir ve yeni istasyona genellenebilir bir VS30 yerine gecmez. Bu nedenle en temiz karar, VS30'yi ilk fazdan cikarip bunu acik sapma olarak yazmaktir.`
- Dosya / artifact: `ML/autoencoder/experiments/PaperRepro/docs/03_acik_sorular_ve_karar_kaydi_tr.md`
- Durum: `Script ile dogrulandi`

### Q18. Station identity tamamen disari mi alinacak?
- Neden kritik: paper-faithful olmak ile veriyi ezberleme riski burada catisiyor.
- Tartisilacak secenekler:
  - hic station-id yok
  - sadece audit icin var
  - ana modelde yok, ablation'da var
- Karar: `Ilk faz ana modelinde station identity condition olarak kullanilmayacak. station_code ve station_idx yalnizca audit, split kontrolu ve per-station analiz icin tutulacak. Station embedding ancak daha sonraki ayri bir ablation fazinda test edilebilecek.`
- Gerekce: `Station identity teknik olarak kolayca modele eklenebilir, ancak bu durumda fiziksel predictor'lar ile istasyon ezberleme etkisi birbirine karisir. Ilk fazin amaci fiziksel condition setiyle temiz bir ana model kurmak oldugu icin station-id ana modelden disarida tutulmalidir. Buna karsin metadata olarak elde tutulmasi, split dogrulamasi ve station-bazli analiz icin gereklidir.`
- Dosya / artifact: `ML/autoencoder/experiments/PaperRepro/docs/03_acik_sorular_ve_karar_kaydi_tr.md`
- Durum: `Karar verildi`

### Q19. Derived physics feature eklenecek mi?
- Neden kritik: travel-time, P-S farki, 1D velocity-derived feature'lar fiziksel bilgi tasiyabilir ama paper'dan uzaklastirir.
- Tartisilacak secenekler:
  - hic eklenmesin
  - sonraki faza not dusulsun
  - ilk fazda cok sinirli bir derived feature olsun
- Karar: `Ilk fazda derived physics feature kullanilacak, ancak yalnizca daha once condition setine dahil edilen sinirli liste ile sinirli kalinacak: hypocentral_distance_km, azimuthal_gap_deg, tP_ref_s ve dtPS_ref_s. Bunun disinda ekstra 1D velocity-derived feature paketi acilmayacak.`
- Gerekce: `Bu liste ilk faz icin yeterli fiziksel bilgi tasir ve condition setine zaten kontrollu sekilde dahil edilmistir. tP_ref_s ile dtPS_ref_s birlikte ilk gelis zamanini ve P-S ayrimini kompakt sekilde verir. Buna ek olarak tS_ref_s gibi ekstra turevler ayni bilgiyi buyuk olcude tekrar eder. Daha genis bir derived feature paketi ilk fazi sisirir ve hangi kazancin hangi feature'dan geldigi sorusunu bulandirir.`
- Dosya / artifact: `ML/autoencoder/experiments/PaperRepro/docs/03_acik_sorular_ve_karar_kaydi_tr.md`
- Durum: `Script ile dogrulandi`

### Q20. Condition normalizasyonu nasil yapilacak?
- Neden kritik: scalar feature'larin olcekleri cok farkli olabilir.
- Tartisilacak secenekler:
  - train split istatistikleriyle z-score
  - min-max
  - paper'a benzer hazir normalized features uretimi
- Karar: `Condition normalizasyonu karma rejimle yapilacak. magnitude, depth_km, hypocentral_distance_km, azimuthal_gap_deg, tP_ref_s ve dtPS_ref_s train split istatistikleri ile z-score normalize edilecek. azimuth_sin ve azimuth_cos ise dogal olarak [-1, 1] araliginda olduklari icin oldugu gibi birakilacak.`
- Gerekce: `Scalar feature'larin olcekleri ciddi sekilde farkli oldugu icin normalize etmeden egitim yapmak dogru olmaz. Min-max yaklasimi aykiri degerlere daha hassastir. Train split z-score standard ve stabil bir secimdir. Buna karsin azimuth_sin ve azimuth_cos zaten dogal olarak normalize feature'lardir; bunlari tekrar z-score etmek gereksizdir.`
- Dosya / artifact: `ML/autoencoder/experiments/PaperRepro/docs/03_acik_sorular_ve_karar_kaydi_tr.md`
- Durum: `Script ile dogrulandi`

## 6. Representation kontrati

### Q21. Giris/ cikis temsili magnitude-only log-spectrogram mi olacak?
- Neden kritik: paper'a yakinlik icin ana karar bu.
- Tartisilacak secenekler:
  - magnitude-only log-spectrogram
  - complex STFT
  - waveform + auxiliary spectral loss
- Karar: `Ilk fazda giris ve cikis temsili magnitude-only log-spectrogram olacak. Complex STFT ve waveform-domain ana temsil ilk fazda kullanilmayacak.`
- Gerekce: `Bu branch'in temel amaci paper'daki ana model mantigini alip bizim verimize uyarlamak oldugu icin, representation seviyesinde paper ile ayni ailede kalmak en dogru secimdir. magnitude-only log-spectrogram, 2D autoencoder + 2D latent diffusion akisi icin daha sade ve dogrudan bir baslangic sunar. Faz ve waveform ceiling sorunlari yok sayilmayacak; bunlar inverse transform ve ceiling raporlarinda ayri ele alinacaktir.`
- Dosya / artifact: `ML/autoencoder/experiments/PaperRepro/docs/03_acik_sorular_ve_karar_kaydi_tr.md`
- Durum: `Karar verildi`

### Q22. Representation kac boyutlu olacak?
- Neden kritik: latent map boyutu ve U-Net olcegi buna bagli.
- Tartisilacak alt noktalar:
  - zaman boyu
  - frekans boyu
  - component'lerin channel olarak ele alinmasi
- Karar: `Representation boyutlari ilk fazda paper ile ayni ailede sabitlenecek. Giris spectrogram tensori 3 x 128 x 128 olacak. Autoencoder latent haritasi ise 8 x 32 x 32 olacak. 3 component, spectrogram channel'i olarak birlikte tutulacak.`
- Gerekce: `Bu branch'te ilk hedef temiz bir ana hat kurmak oldugu icin, temsil boyutlarinda da paper'a yakin ve kodla tutarli kalmak en dogru yaklasimdir. tqdne tarafinda 4064 sample, 256 STFT channel ve 32 hop ile 128 x 128 log-spectrogram hedefleniyor. Spatial latent boyutu, channel_mult = (1, 2, 4) nedeniyle 32 x 32'ye iniyor. Latent kanal sayisinda README ve eski run isimlerinde 4 izi bulunsa da aktif config ve mimari kodu latent_channels = 8 kullaniyor. Bu nedenle ilk fazda boyut seviyesinde yeni bir optimizasyon aramiyor, ancak stale isimlendirme yerine kodun gercek omurgasini esas aliyoruz.`
- Dosya / artifact: `ML/autoencoder/experiments/PaperRepro/docs/03_acik_sorular_ve_karar_kaydi_tr.md`
- Durum: `Script ile dogrulandi`

### Q23. STFT parametreleri ne olacak?
- Neden kritik: paper'a yakinlik ve ters donus kalitesi buna bagli.
- Tartisilacak alt noktalar:
  - `n_fft`
  - `hop_length`
  - pencere tipi
  - zero padding kurali
- Karar: `Ilk faz STFT parametreleri paper ile ayni ailede sabitlenecek: n_fft = 256, hop_length = 32, pencere = hann. STFT temsili 4064 sample giris uzerinde zero-padding / center davranisiyla 129 x 128 sekilde hesaplanacak, ardindan Nyquist bandi atilarak 128 x 128 magnitude-only log-spectrogram elde edilecek. Inverse tarafta Griffin-Lim n_iter = 128 kullanilacak.`
- Gerekce: `tqdne tarafinda stft_channels = 256 ve hop_size = 32 secimi kullaniliyor; 4064 sample ile bu kombinasyon tam 128 x 128 spectrogram ailesini veriyor. Nyquist bandinin atilmasi da representation kodunda acik. Bu nedenle ilk fazda STFT seviyesinde yeni optimizasyon aramak yerine paper ile uyumlu ve sekil olarak dogrulanmis parametreleri dondurmak en dogru yoldur.`
- Dosya / artifact: `ML/autoencoder/experiments/PaperRepro/docs/03_acik_sorular_ve_karar_kaydi_tr.md`
- Durum: `Script ile dogrulandi`

### Q24. Representation normalizasyonu nasil yapilacak?
- Neden kritik: bizim onceki per-sample min-max aliskanligimiz burada tekrar edilmemeli.
- Tartisilacak secenekler:
  - fixed clipping + global linear scale
  - global z-score
  - train global min/max
- Karar: `Representation normalizasyonu paper ile ayni fixed scaling rejimiyle yapilacak. Magnitude spectrogram once clip(1e-8) ile alt esikten kirpilacak, sonra log alinacak, ardindan log_clip = log(1e-8) ve log_max = 3 kullanilarak lineer sekilde [-1, 1] araligina map edilecek. Global z-score veya train global min/max ilk fazda kullanilmayacak.`
- Gerekce: `Bu branch'te once paper mantigini bizim veride temiz sekilde kurmak istiyoruz. Fixed clipping + sabit log scaling, fiziksel enerji hiyerarsisini global bir eksende korur ve paper ile karsilastirmayi temiz tutar. Global z-score bazen optimizasyonu kolaylastirabilir, ancak train dagilimina daha bagimli hale gelir ve elde edilecek farkin modelden mi normalization'dan mi geldigi sorusunu bulandirir. Bu nedenle ilk fazda daha savunulabilir secim fixed scaling'dir.`
- Dosya / artifact: `ML/autoencoder/experiments/PaperRepro/docs/03_acik_sorular_ve_karar_kaydi_tr.md`
- Durum: `Script ile dogrulandi`

### Q25. Inverse transform yolu ne olacak?
- Neden kritik: waveform yorumlarinda ceiling etkisi var.
- Tartisilacak secenekler:
  - sadece Griffin-Lim
  - Griffin-Lim + ceiling notu
  - temsil alaninda ana skor, waveform sadece yardimci skor
- Karar: `Inverse transform ilk fazda Griffin-Lim ile yapilacak. Ancak waveform uzayindaki ciktilar ana basari kapisi olarak kullanilmayacak; waveform metrikleri, gorseller ve sanity kontrolleri yardimci analiz olarak raporlanacak. Griffin-Lim ceiling etkisi her waveform yorumunda acikca not edilecek.`
- Gerekce: `Paper'in ana hattinda magnitude-only log-spectrogram inversion'u Griffin-Lim ile yapiliyor. Bu nedenle inverse tarafta ayni ailede kalmak dogrudur. Buna karsin Griffin-Lim hatasi ile model hatasini birbirine karistirmamak gerekir. En savunulabilir yol, ana skorlari representation alaninda tutup waveform uzayini yardimci analiz olarak kullanmaktir.`
- Dosya / artifact: `ML/autoencoder/experiments/PaperRepro/docs/03_acik_sorular_ve_karar_kaydi_tr.md`
- Durum: `Karar verildi`

### Q26. Representation ceiling nasil raporlanacak?
- Neden kritik: model hatasi ile inverse transform hatasini ayirmak gerekir.
- Tartisilacak alt noktalar:
  - oracle reconstruction deneyi
  - ideal representation -> waveform ceiling testi
  - paper-style ve kalem-style skorlarin ceiling'e gore yorumlanmasi
- Karar: `Representation ceiling ucu seviyede raporlanacak: (1) Stage-1 oracle spectrogram referansi, (2) generated spectrogram ciktilari, (3) oracle spectrogram'dan Griffin-Lim ile elde edilen waveform ceiling referansi. Modelin ana degerlendirmesi once spectrogram alaninda yapilacak; waveform skor ve gorselleri ise bu ceiling baglaminda yorumlanacak.`
- Gerekce: `Magnitude-only log-spectrogram hattinda model hatasi ile inverse transform hatasini ayirmak zorunludur. Stage-1 oracle referansi autoencoder ceiling'ini, gercek spectrogram -> Griffin-Lim hattı ise waveform ceiling'ini verir. Bu iki referans olmadan diffusion, autoencoder ve inverse transform etkileri birbirine karisir. Daha onceki diffusion kutularinda kullanilan oracle mantigi da bu ayrimi faydali sekilde destekliyor.`
- Dosya / artifact: `ML/autoencoder/experiments/PaperRepro/docs/03_acik_sorular_ve_karar_kaydi_tr.md`
- Durum: `Karar verildi`

## 7. Split ve evaluation veri rejimi

### Q27. Train/val/test split tam olarak nasil dondurulecek?
- Neden kritik: tum karsilastirmalar ayni splitte yapilmali.
- Tartisilacak alt noktalar:
  - event-wise tanim
  - seed
  - oranlar
  - min event sayisi
- Karar: `Ana split event-wise olarak dondurulacak. Oranlar train/val/test = 80/10/10 olacak ve split seed = 42 secilecek. Ayni event_id'ye ait tum sample'lar ayni splitte kalacak. Ek minimum station sayisi filtresi uygulanmayacak; ancak preprocessing sonrasinda en az bir gecerli sample'i kalan eventler split havuzuna girecek.`
- Gerekce: `Manifestte 12721 unique event var; bu hacim event-wise split icin fazlasiyla yeterli. 80/10/10 secimi train tarafini gereksiz yere daraltmadan val ve test icin de yeterli event hacmi birakir. Ayrica sonraki soruda ayri bir OOD rejimi tanimlanacagi icin ana splitte train kapasitesini korumak daha dogrudur.`
- Dosya / artifact: `ML/autoencoder/experiments/PaperRepro/docs/03_acik_sorular_ve_karar_kaydi_tr.md`
- Durum: `Script ile dogrulandi`

### Q28. OOD seti nasil tanimlanacak?
- Neden kritik: paper'a yakin ama bizim disiplinimizle uyumlu olmak icin ayri OOD rejimi lazim.
- Tartisilacak secenekler:
  - zaman bazli OOD
  - event-heldout OOD
  - station-heldout OOD
  - magnitude / distance extremum OOD
- Karar: `Ilk fazda ana OOD rejimi ayri bir event-heldout OOD havuzu olarak tanimlanacak. Bu OOD event'leri train/val/test splitlerinden tamamen ayri tutulacak. Ilk operasyonel freeze icin OOD oranı event bazinda 0.10 olarak alinacak. Time-based OOD, station-heldout OOD ve magnitude/distance extremum OOD ilk fazda kullanilmayacak.`
- Gerekce: `Event-heldout OOD, hem bizim event-wise disiplinimizle en uyumlu hem de en temiz genelleme testidir. Time-based OOD veri toplama donemi kaynakli ek driftleri isin icine katar. Station-heldout OOD ise ilk fazda fiziksel predictor setini test etmekten farkli bir soruya kayar. Magnitude extremum OOD da mevcut ust kuyruk seyrek oldugu icin zayif bir ilk-faz tercihi olur.`
- Dosya / artifact: `ML/autoencoder/experiments/PaperRepro/docs/03_acik_sorular_ve_karar_kaydi_tr.md`
- Durum: `Karar verildi`

### Q29. Stage-1 ve Stage-2 ayni split altyapisini mi kullanacak?
- Neden kritik: latent cache ile original waveform splitleri birebir hizali olmali.
- Karar: `Evet. Stage-1 ve Stage-2 birebir ayni split altyapisini kullanacak. Train/val/test/OOD event havuzlari, sample listeleri ve indeks hizasi iki asama boyunca korunacak. Stage-2 latent cache'leri yalnizca Stage-1 ile ayni splitteki sample'lardan uretilecek.`
- Gerekce: `Latent diffusion asamasinda condition, latent cache ve referans sample'larin birbirine tam hizali olmasi gerekir. Split veya sample filtresi Stage-1 ile Stage-2 arasinda degisirse karsilastirma kirlenir ve cache semantigi bozulur. Paper tarafinda da iki asama ayni config ve ayni dataloader mantigi uzerinden ilerliyor.`
- Dosya / artifact: `ML/autoencoder/experiments/PaperRepro/docs/03_acik_sorular_ve_karar_kaydi_tr.md`
- Durum: `Karar verildi`

### Q30. Evaluation'da test ve OOD ayri mi raporlanacak?
- Neden kritik: tek tabloya yigma sonuc yorumu bulandirir.
- Tartisilacak alt noktalar:
  - IID test
  - OOD event
  - belki magnitute / distance binleri
- Karar: `Evet. Evaluation'da IID test ve event-heldout OOD ayri ana bloklar olarak raporlanacak. Her iki blokta da global metrikler ile magnitude-bin ve distance-bin bazli alt tablolar verilecek. Tek bir karisik toplam tablo ana rapor formati olarak kullanilmayacak.`
- Gerekce: `Bu branch'in hedefi yalnizca reconstruction kalitesi degil, genelleme davranisini da net gostermektir. Test ve OOD sonuclarini tek tabloya yigmak yorumu bulandirir. Q12'de dondurulen bin-wise raporlama mantigi da ayri test ve OOD bloklariyla birlikte daha anlamli hale gelir.`
- Dosya / artifact: `ML/autoencoder/experiments/PaperRepro/docs/03_acik_sorular_ve_karar_kaydi_tr.md`
- Durum: `Karar verildi`

## 8. Stage-1 VAE kontrati

### Q31. Stage-1 encoder/decoder tam olarak nasil bir aileden olacak?
- Neden kritik: paper-faithful olmanin cekirdek noktasi.
- Tartisilacak secenekler:
  - tam conv AE/VAE
  - residual conv VAE
  - attention eklenmis encoder/decoder
- Karar: `Stage-1 encoder/decoder ailesi 2D residual convolutional VAE olacak. Self-attention kullanilmayacak. Released tqdne kodundaki autoencoder omurgasi esas alinacak: dims = 2, conv_kernel_size = 3, model_channels = 64, channel_mult = (1, 2, 4), dropout = 0.1, num_res_blocks = 2. Encoder downsample ederek, decoder upsample ederek 128 x 128 log-spectrogram'u 8 x 32 x 32 latent haritaya ve tekrar geri tasiyacak.`
- Gerekce: `tqdne tarafindaki calisan autoencoder omurgasi residual conv bloklari, downsample/upsample adimlari ve attention'siz 2D yapidan olusuyor. Bu aile, paper'in Stage-1 icin verdigi genel tarifle uyumlu ve bizim gercekten yeniden kuracagimiz released implementation hattini temsil ediyor. PDF metninde autoencoder icin 3 residual block ifadesi gecse de, kodda num_res_blocks = 2 kullaniliyor; bu noktada calisan released kod daha guvenilir kaynak kabul edildi.`
- Dosya / artifact: `ML/autoencoder/experiments/PaperRepro/docs/03_acik_sorular_ve_karar_kaydi_tr.md`, `/home/gms/kalem_seismic/tqdne/tqdne/architectures.py`, `/home/gms/kalem_seismic/tqdne/tqdne/blocks.py`, `/home/gms/kalem_seismic_paper_repro/Seismic-wave-generation-Diffusion.pdf`
- Durum: `Script ile dogrulandi`

### Q32. Latent spatial shape nasil secilecek?
- Neden kritik: Stage-2 diffusion kapasitesi ve temsil kalitesi dogrudan bundan etkilenir.
- Tartisilacak alt noktalar:
  - latent channels
  - downsample katsayisi
  - hedef latent HxW
- Karar: `Stage-1 latent haritasi 8 x 32 x 32 olarak sabitlenecek. Downsample katsayisi 4 olacak; yani 128 x 128 log-spectrogram girisi encoder sonunda 32 x 32 spatial latent haritaya indirgenecek. Stage-2 latent diffusion da bu sekli birebir kullanacak.`
- Gerekce: `Paper metni latent kanal sayisini acikca dondurmuyor; bu nedenle belirleyici kaynak tqdne kodunun aktif config ve mimari hattidir. experiments/config.py ve generate_waveforms.py latent_channels = 8 tanimliyor. architectures.py icindeki 2D autoencoder konfigurasyonu ise channel_mult = (1, 2, 4) ile 128 x 128 girisi 32 x 32 spatial latent'e indiriyor. README ve bazi run isimlerinde gecen 32 x 32 x 4 ifadesi stale isimlendirme olarak degerlendirildi; mimarinin gercek calisan omurgasi 8 x 32 x 32'dir.`
- Dosya / artifact: `ML/autoencoder/experiments/PaperRepro/docs/03_acik_sorular_ve_karar_kaydi_tr.md`, `/home/gms/kalem_seismic/tqdne/experiments/config.py`, `/home/gms/kalem_seismic/tqdne/tqdne/architectures.py`, `/home/gms/kalem_seismic/tqdne/tqdne/generate_waveforms.py`
- Durum: `Script ile dogrulandi`

### Q33. KL agirligi ve schedule ne olacak?
- Neden kritik: Stage-1'in generator degil compressor gibi davranmasi icin burasi cok hassas.
- Tartisilacak secenekler:
  - sabit tiny KL
  - warmup ile tiny KL
  - annealing
  - free-bits kullanmama
- Karar: `Stage-1 autoencoder icin KL agirligi sabit 1e-6 olacak. KL annealing, warmup, cyclic beta veya free-bits ilk fazda kullanilmayacak.`
- Gerekce: `Makale hyperparameter tablosu spectrogram latent diffusion autoencoder icin KL Weight = 1e-6 veriyor. tqdne kodunda da experiments/config.py icinde kl_weight = 1e-6 tanimli ve LightningAutoencoder kaybi dogrudan recon_mse + kl_weight * kl_div olarak kurulmus. Ayrica kodda KL annealing ya da free-bits mekanizmasi yok. Bu kadar kucuk sabit KL, Stage-1'i prior'a kuvvetle zorlayan bir generative VAE yerine yuksek rekonstruksiyon kalitesi hedefleyen bir compressor gibi tutar; bizim ilk faz amacimiz da budur.`
- Dosya / artifact: `ML/autoencoder/experiments/PaperRepro/docs/03_acik_sorular_ve_karar_kaydi_tr.md`, `/home/gms/kalem_seismic/tqdne/experiments/config.py`, `/home/gms/kalem_seismic/tqdne/tqdne/autoencoder.py`, `/home/gms/kalem_seismic_paper_repro/Seismic-wave-generation-Diffusion.pdf`
- Durum: `Script ile dogrulandi`

### Q34. Stage-1 loss tam olarak hangi terimlerden olusacak?
- Neden kritik: sadece pixel/spectrogram L1 ile mi, yoksa ek spectral yardimcilarla mi egitecegimiz netlesmeli.
- Tartisilacak secenekler:
  - L1 reconstruction + KL
  - L2 reconstruction + KL
  - MR spectral loss + KL
  - component-weighted recon + KL
- Karar: `Stage-1 loss ilk fazda yalnizca reconstruction MSE + 1e-6 * KL terimlerinden olusacak. L1, multi-resolution spectral loss, adversarial loss veya component-weighted ek kayiplar ilk fazda kullanilmayacak.`
- Gerekce: `Makale autoencoder'i generative model olarak degil, veri compression araci olarak kullandigini acikca soyluyor. tqdne kodunda da LightningAutoencoder kaybi dogrudan recon_mse + kl_weight * kl_div olarak tanimli. Bu branch'te once paper ile ayni ana hatti kurmak istiyoruz; ek kayiplar simdiden eklenirse iyilesmenin model omurgasindan mi yoksa loss engineering'den mi geldigi belirsizlesir. Bu nedenle ilk fazda en temiz ve savunulabilir secim yalnizca MSE + tiny KL'dir.`
- Dosya / artifact: `ML/autoencoder/experiments/PaperRepro/docs/03_acik_sorular_ve_karar_kaydi_tr.md`, `/home/gms/kalem_seismic/tqdne/tqdne/autoencoder.py`, `/home/gms/kalem_seismic_paper_repro/Seismic-wave-generation-Diffusion.pdf`
- Durum: `Script ile dogrulandi`

### Q35. Stage-1 condition'i encoder, decoder veya her ikisine birden mi girecek?
- Neden kritik: paper-faithful conditional compression davranisi icin tanim gerekli.
- Karar: `Stage-1 autoencoder ilk fazda tamamen unconditional olacak. Scalar condition seti ne encoder'a ne decoder'a verilecek. Condition bilgisi yalnizca Stage-2 latent EDM tarafina girecek.`
- Gerekce: `tqdne tarafinda spectrogram autoencoder egitimi get_train_and_val_loader(..., cond=False) ile kuruluyor; yani Stage-1'e scalar condition tasinmiyor. autoencoder.py icinde cond_signal destegi olsa da bu genel API'nin bir parcasi; paper'daki spectrogram latent diffusion hattinda aktif mekanizma degil. Bu karar, Stage-1'i saf bir spectrogram compressor olarak tutar ve Stage-2'nin scalar predictor'lardan latent ureten generative rolunu netlestirir.`
- Dosya / artifact: `ML/autoencoder/experiments/PaperRepro/docs/03_acik_sorular_ve_karar_kaydi_tr.md`, `/home/gms/kalem_seismic/tqdne/experiments/train_autoencoder.py`, `/home/gms/kalem_seismic/tqdne/tqdne/dataloader.py`, `/home/gms/kalem_seismic/tqdne/tqdne/autoencoder.py`
- Durum: `Script ile dogrulandi`

### Q36. Latent cache icin encoder'dan ne alinacak?
- Neden kritik: `mu`, sample, veya baska deterministic temsil Stage-2 sorununu degistirir.
- Tartisilacak secenekler:
  - `mu`
  - posterior sample
  - deterministic bottleneck
- Karar: `Stage-2 icin latent cache ve online encode yolu paper kodundaki gibi posterior sample kullanacak. Yani Stage-1 encoder'dan dogrudan sample edilen latent alinacak; mu-temelli deterministic cache ilk fazda kullanilmayacak.`
- Gerekce: `tqdne tarafinda LightningAutoencoder.encode(x) fonksiyonu posterior sample donuyor; mean'i ayri expose eden bir public API yok. LightningEDM.step ve sample fonksiyonlari da Stage-1 latentini autoencoder.encode(...) uzerinden aliyor. Dolayisiyla paper-faithful ilk fazda Stage-2'nin gordugu latent, deterministic mu degil posterior sample'dir. Bu secim tiny KL ile birlikte Stage-1'in calisan omurgasini aynen korur; mu-cache gibi bir deterministiklestirme sonraki ablation konusu olarak birakilir.`
- Dosya / artifact: `ML/autoencoder/experiments/PaperRepro/docs/03_acik_sorular_ve_karar_kaydi_tr.md`, `/home/gms/kalem_seismic/tqdne/tqdne/autoencoder.py`, `/home/gms/kalem_seismic/tqdne/tqdne/edm.py`
- Durum: `Script ile dogrulandi`

### Q37. Stage-1 erken durdurma ve model secim kriteri ne olacak?
- Neden kritik: sadece val loss ile secim yanlis olabilir.
- Tartisilacak secenekler:
  - val recon
  - composite recon
  - representation ceiling'e yakinlik
- Karar: `Stage-1 erken durdurma ve best checkpoint secimi icin ana monitor validation/reconstruction_loss olacak. validation/loss ve validation/kl_divergence loglanacak, ancak model secim kriteri olmayacak.`
- Gerekce: `Ilk fazda Stage-1'in gorevi generative prior kurmak degil, yuksek reconstruction kalitesiyle spectrogram compression yapmaktir. Zaten KL agirligi 1e-6 gibi tiny bir deger oldugu icin toplam validation/loss pratikte reconstruction terimine cok yakindir; ancak checkpoint secimini dogrudan validation/reconstruction_loss uzerinden yapmak amaci daha net temsil eder. Composite ya da downstream-ceiling tabanli secim ise ilk faz icin gereksiz ek karmaşıklik getirir. Implementasyon sirasinda released tqdne trainer'inin validation/loss monitorunu bu karara gore acikca override etmek zorunludur; aksi halde dokuman ve kod davranisi ayrisir.`
- Dosya / artifact: `ML/autoencoder/experiments/PaperRepro/docs/03_acik_sorular_ve_karar_kaydi_tr.md`, `/home/gms/kalem_seismic/tqdne/tqdne/autoencoder.py`, `/home/gms/kalem_seismic/tqdne/experiments/train_autoencoder.py`
- Durum: `Karar verildi`

## 9. Stage-2 EDM kontrati

### Q38. Denoiser tam olarak nasil condition alacak?
- Neden kritik: scalar predictor'larin U-Net icine nasil girecegi basta dondurulmali.
- Tartisilacak secenekler:
  - time embedding uzerinden MLP
  - AdaGN / AdaLN benzeri injection
  - concat + projection
- Karar: `Stage-2 denoiser scalar condition vektorunu MLP ile embedding uzayina tasiyacak ve bu embedding noise/time embedding'e eklenecek. cond_emb_scale kullanilmayacak, use_scale_shift_norm kapali kalacak, condition spatial kanallara concat edilmeyecek.`
- Gerekce: `tqdne tarafinda UNetModel condition'i cond_mlp ile embed_dim boyutuna cikariyor ve bunu time embedding'e ekliyor. train_latent_edm.py ile kurulan gercek spectrogram latent EDM hattinda FiLM benzeri scale-shift conditioning aktif degil; cond_emb_scale da kullanilmiyor. Ilk fazda paper ile ayni conditioning mekanizmasini korumak, yeni bir injection ekseni acmaktan daha dogru.`
- Dosya / artifact: `ML/autoencoder/experiments/PaperRepro/docs/03_acik_sorular_ve_karar_kaydi_tr.md`, `/home/gms/kalem_seismic/tqdne/tqdne/unet.py`, `/home/gms/kalem_seismic/tqdne/tqdne/architectures.py`, `/home/gms/kalem_seismic/tqdne/experiments/train_latent_edm.py`
- Durum: `Script ile dogrulandi`

### Q39. U-Net omurgasinin minimum spesifikasyonu ne olacak?
- Neden kritik: "2D U-Net" tek basina yeterince belirli degil.
- Tartisilacak alt noktalar:
  - kac seviye
  - residual block sayisi
  - attention hangi resolution'da
  - channel multiplier'lar
- Karar: `Stage-2 latent EDM icin 2D U-Net omurgasi paper/tqdne ile ayni minimum spesifikasyonda sabitlenecek: dims = 2, conv_kernel_size = 3, model_channels = 128, channel_mult = (1, 2, 4, 4), num_res_blocks = 2, attention_resolutions = (8,), num_heads = 4, dropout = 0.1, flash_attention = False, use_causal_mask = False.`
- Gerekce: `tqdne tarafinda spectrogram latent EDM'in gercek calisan omurgasi get_2d_unet_config ile bu parametreleri kullaniyor. Makale hyperparameter tablosu da 3x3 kernel, tek attention seviyesi ve 0.1 dropout ailesiyle uyumlu. Ilk fazda yeni kapasite arayisi yerine dogrudan paper kodunun calisan 2D U-Net omurgasini almak daha dogru.`
- Dosya / artifact: `ML/autoencoder/experiments/PaperRepro/docs/03_acik_sorular_ve_karar_kaydi_tr.md`, `/home/gms/kalem_seismic/tqdne/tqdne/architectures.py`, `/home/gms/kalem_seismic/tqdne/tqdne/unet.py`, `/home/gms/kalem_seismic_paper_repro/Seismic-wave-generation-Diffusion.pdf`
- Durum: `Script ile dogrulandi`

### Q40. EDM loss ve sigma schedule hangi parametrelerle kurulacak?
- Neden kritik: paper'a yakinlik burada teknik olarak belirlenir.
- Tartisilacak alt noktalar:
  - sigma min / max
  - rho
  - sigma data
  - deterministic vs stochastic sampling
- Karar: `Stage-2 latent diffusion icin EDM objective ve sigma parametreleri paper/tqdne ile ayni tutulacak. EDM parametreleri: sigma_min = 0.002, sigma_max = 80.0, rho = 7.0, sigma_data = 0.5, P_mean = -1.2, P_std = 1.2, S_churn = 40, S_min = 0.05, S_max = 50, S_noise = 1.003. Ilk raporlanacak varsayilan sampling modu deterministic olacak ve num_sampling_steps = 25 kalacak.`
- Gerekce: `Makale metni Karras et al. (2022) preconditioning ve weighting formulasyonunu kullandigini ve ikinci mertebe Heun orneklemesi icin 25 step sectigini acikca yaziyor. tqdne kodunda da EDM sinifinin aktif defaultlari bu degerlerle tanimli ve train_latent_edm.py bunlari override etmiyor. Ilk fazda bu seviyede yeni tuning ekseni acmak yerine paper kodunun calisan EDM omurgasini aynen almak daha dogru.`
- Dosya / artifact: `ML/autoencoder/experiments/PaperRepro/docs/03_acik_sorular_ve_karar_kaydi_tr.md`, `/home/gms/kalem_seismic/tqdne/tqdne/edm.py`, `/home/gms/kalem_seismic/tqdne/experiments/train_latent_edm.py`, `/home/gms/kalem_seismic_paper_repro/Seismic-wave-generation-Diffusion.pdf`
- Durum: `Script ile dogrulandi`

### Q41. Stage-2 egitim hedefi ne olacak?
- Neden kritik: latent map diffusion'da hangi target'in denoise edildigi acik olmali.
- Tartisilacak secenekler:
  - clean latent target
  - V-pred benzeri formulasyon
  - EDM standard objective
- Karar: `Stage-2 egitim hedefi clean latent target olacak. Yani model, perturbe edilmis latent z_t ve condition verildiginde temiz latent z_0'i tahmin edecek. Objective olarak paper/tqdne ile ayni EDM standard weighted denoising objective kullanilacak; V-pred veya epsilon-target formulasyonlari ilk fazda kullanilmayacak.`
- Gerekce: `Makale denoiser D_theta(z_t, t, c)'nin orijinal latent z_0'i tahmin ettigini acikca yaziyor. tqdne kodunda da LightningEDM.step fonksiyonunda loss, (pred - sample)^2 * loss_weight(sigma) olarak kurulmus; burada sample temiz latent temsilidir. Bu nedenle paper-faithful ilk fazda hedefin clean latent olmasi teknik olarak en dogru secimdir.`
- Dosya / artifact: `ML/autoencoder/experiments/PaperRepro/docs/03_acik_sorular_ve_karar_kaydi_tr.md`, `/home/gms/kalem_seismic/tqdne/tqdne/edm.py`, `/home/gms/kalem_seismic_paper_repro/Seismic-wave-generation-Diffusion.pdf`
- Durum: `Script ile dogrulandi`

### Q42. Sampling tarafinda ilk raporlanacak konfigurasyonlar hangileri olacak?
- Neden kritik: step sayisi ve sampler secimi runtime-qualtiy dengesini belirler.
- Tartisilacak alt noktalar:
  - deterministic Heun
  - stochastic sampling
  - kisa / orta / uzun step seti
- Karar: `Ilk fazda ana sampling konfigurasyonu paper ile ayni olacak: deterministic ikinci mertebe Heun sampler ve 25 step. Stochastic sampling veya daha uzun step setleri ilk ana rapora girmeyecek; bunlar sonraki sampling ablation fazina birakilacak.`
- Gerekce: `Makale sampling icin acikca second-order Heun method using 25 steps corresponding to a total of 50 model evaluations diyor. tqdne kodunda da LightningEDM varsayilani deterministic_sampling = True ve num_sampling_steps = 25. Bu nedenle ilk raporlanacak ana konfigurasyonun paper ile ayni olmasi en temiz secimdir.`
- Dosya / artifact: `ML/autoencoder/experiments/PaperRepro/docs/03_acik_sorular_ve_karar_kaydi_tr.md`, `/home/gms/kalem_seismic/tqdne/tqdne/edm.py`, `/home/gms/kalem_seismic_paper_repro/Seismic-wave-generation-Diffusion.pdf`
- Durum: `Script ile dogrulandi`

### Q43. Classifier-free guidance benzeri ekler ilk fazda olacak mi?
- Neden kritik: paper-faithful ana hat ile ekstra diffusion teknikleri karismamali.
- Karar: `Hayir. Ilk fazda classifier-free guidance veya benzeri ekstra diffusion teknikleri kullanilmayacak. Ilk faz yalnizca temel latent EDM hattini kuracak. Guidance, sampler trick'leri ve benzeri ekler ancak sonraki ayri arastirma fazlarinda acilabilecek.`
- Gerekce: `Ilk fazin amaci paper'a yakin ve teknik olarak temiz bir temel hat kurmaktir. Guidance benzeri ekler kaliteyi etkileyebilir, ancak bu durumda elde edilen farkin temel modelden mi yoksa ek diffusion hilesinden mi geldigi anlasilmaz. Q04'te cizilen scope siniri ile de uyumludur.`
- Dosya / artifact: `ML/autoencoder/experiments/PaperRepro/docs/03_acik_sorular_ve_karar_kaydi_tr.md`
- Durum: `Karar verildi`

## 10. Evaluation kontrati

### Q44. Stage-1 reconstruction icin ana metrikler neler olacak?
- Neden kritik: Stage-1'i sadece val loss ile degil, anlamli reconstruction metrikleriyle okumak gerekir.
- Tartisilacak adaylar:
  - L1 / L2
  - `spec_corr`
  - `LSD`
  - `MR-LSD`
  - oracle inversion ceiling
- Karar: `Stage-1 reconstruction icin ana metrikler representation uzayinda tutulacak: validation/reconstruction_loss, spec_corr ve LSD. MR-LSD, waveform MSE, waveform ASD ve oracle inversion ceiling ise yardimci metrikler olarak raporlanacak.`
- Gerekce: `Stage-1'in asli gorevi magnitude-only log-spectrogram compression oldugu icin ana karar metriğinin representation uzayinda olmasi gerekir. tqdne tarafinda waveform MSE ve ASD loglansa da bunlar Griffin-Lim ceiling ve waveform inverse etkisiyle karisabilir. Bu nedenle primary metrikleri spectrogram alaninda tutup, paper-style waveform ASD ve waveform MSE'yi secondary sanity paketi olarak okumak daha dogru dengedir.`
- Dosya / artifact: `ML/autoencoder/experiments/PaperRepro/docs/03_acik_sorular_ve_karar_kaydi_tr.md`, `/home/gms/kalem_seismic/tqdne/experiments/train_autoencoder.py`, `/home/gms/kalem_seismic/tqdne/tqdne/metric.py`
- Durum: `Karar verildi`

### Q45. Stage-2 generation icin ana metrikler neler olacak?
- Neden kritik: paper-style ve bizim mevcut pratiklerimiz birlikte ama karismadan raporlanmali.
- Tartisilacak adaylar:
  - `spec_corr`
  - `LSD`
  - `MR-LSD`
  - waveform envelope / peak istatistikleri
  - distribution metrics
- Karar: `Stage-2 generation icin ana metrik seti hibrit olacak. Primary metrikler: spec_corr, LSD, MR-LSD, envelope benzerligi ve Fourier amplitude distribution benzerligi. Secondary metrikler: waveform MSE, oracle/inversion ceiling ve classifier-tabanli paper metrikleri. PGA, PGV, SA ve cAI ilk fazda ana basari kapisina alinmayacak.`
- Gerekce: `Stage-2 gercek generation problemi oldugu icin sadece spectrogram kalite metrikleri yetmez; time-domain envelope ve Fourier amplitude dagilimi de izlenmelidir. Buna karsin HH counts as-is setup'inda fiziksel birim hassasiyeti isteyen peak ground motion metric'lerini ilk fazda ana karar kriteri yapmak saglam degildir. Bu nedenle hem bizim spectrogram kalite metriklerimizi hem de paper'dan gelen distribution bakisini koruyan, ancak birim-duyarli engineering metric'leri ikinci planda tutan hibrit set en dogru dengedir.`
- Dosya / artifact: `ML/autoencoder/experiments/PaperRepro/docs/03_acik_sorular_ve_karar_kaydi_tr.md`, `/home/gms/kalem_seismic_paper_repro/Seismic-wave-generation-Diffusion.pdf`
- Durum: `Karar verildi`

### Q46. Paper-style metric setinin minimum paketi ne olacak?
- Neden kritik: classifier / Frechet benzeri isler ek maliyet getirir; neyin zorunlu oldugu basta yazilmali.
- Tartisilacak secenekler:
  - Fourier amplitude distribution only
  - Fourier + simple embedding distance
  - Fourier + classifier accuracy + embedding distance
- Karar: `Ilk faz icin minimum paper-style metric paketi tam olarak su uc metrigi icerecek: (1) Fourier amplitude spectra Frechet Distance, (2) classifier accuracy, (3) classifier embedding Frechet Distance.`
- Gerekce: `Makalenin 4.5 bolumundeki temel paper-style degerlendirme uclusu budur. Sadece Fourier FD kullanmak spektral dagilimi gorur ama sinif-ayiricilik ve embedding uzayi kalitesini kacirir. Sadece embedding tarafi da yetmez. Paper'a mumkun oldugunca yakin ilerleme karariyla uyumlu minimum paket, bu uc metrigi birlikte raporlamaktir.`
- Dosya / artifact: `ML/autoencoder/experiments/PaperRepro/docs/03_acik_sorular_ve_karar_kaydi_tr.md`, `/home/gms/kalem_seismic_paper_repro/Seismic-wave-generation-Diffusion.pdf`, `/home/gms/kalem_seismic/tqdne/tqdne/metric.py`, `/home/gms/kalem_seismic/tqdne/experiments/train_classifier.py`
- Durum: `Script ile dogrulandi`

### Q47. OOD evaluation'da hangi metrikler zorunlu olacak?
- Neden kritik: paper-faithful mimarinin bizim daha zor protokolde de guclu kalip kalmadigini gormek istiyoruz.
- Karar: `OOD evaluation'da zorunlu metrik seti testteki ana set ile ayni olacak: spec_corr, LSD, MR-LSD, envelope benzerligi, Fourier amplitude spectra Frechet Distance, classifier accuracy ve classifier embedding Frechet Distance. Bu metrikler global olarak ve ayri ayri magnitude-bin ile distance-bin tablolarinda raporlanacak.`
- Gerekce: `Bu branch'in ana farki sadece paper-faithful mimari kurmak degil, bunu daha siki bir event-heldout OOD rejiminde de sinamaktir. Bu nedenle OOD tarafini testten daha hafif bir metrik paketiyle gecmek metodolojik olarak dogru olmaz. Q30, Q45 ve Q46 ile tutarli sekilde, OOD'de de ayni zorunlu ana kalite ve paper-style metric setini korumak gerekir.`
- Dosya / artifact: `ML/autoencoder/experiments/PaperRepro/docs/03_acik_sorular_ve_karar_kaydi_tr.md`
- Durum: `Karar verildi`

### Q48. Gorsel artifact paketi ne olacak?
- Neden kritik: sayilar tek basina yeterli degil; gorsel kontrol standart olmali.
- Tartisilacak artifactlar:
  - spectrogram grid
  - waveform grid
  - latent sample grid
  - distance / magnitude bin bazli ornekler
- Karar: `Ilk faz icin zorunlu gorsel artifact paketi standart ve iki seviyeli olacak. Stage-1 icin target vs oracle spectrogram grid ve target vs oracle waveform grid uretilecek. Stage-2 icin target vs generated spectrogram grid ve target vs generated waveform grid uretilecek. Bunlara ek olarak envelope distribution plot'lari, Fourier amplitude distribution plot'lari ve test/OOD ayrimiyla magnitude-bin ile distance-bin kapsayan temsilci ornek grid'leri zorunlu olacak. Latent grid veya traversal benzeri daha ileri gorseller ilk fazda zorunlu degil.`
- Gerekce: `Ilk fazda gorsel paket hem bizim oracle/ceiling mantigimizi hem de paper'daki distribution bakisini birlikte tasimalidir. Rastgele birkac ornek gostermek yeterli olmaz; test ve OOD ayrimini, mag/dist kapsamasini ve waveform-ceiling baglamini standard hale getirmek gerekir. Buna karsin latent traversal gibi ileri analiz gorselleri temel hat oturmadan zorunlu pakete alinmamistir.`
- Dosya / artifact: `ML/autoencoder/experiments/PaperRepro/docs/03_acik_sorular_ve_karar_kaydi_tr.md`
- Durum: `Karar verildi`

### Q49. Insan-gozu ile kontrol zorunlu mu?
- Neden kritik: onset, coda, enerji yapisi gibi seylere salt metrik yetmeyebilir.
- Tartisilacak secenekler:
  - evet, zorunlu qualitative check
  - hayir, sadece destekleyici
- Karar: `Evet. Insan-gozu ile qualitative kontrol zorunlu olacak. Sayisal metriklerin yaninda standard bir spectrogram grid, waveform grid ve secilmis zor/ornekleyici sample kontrolleri her ana evaluation paketinde yer alacak.`
- Gerekce: `Bu problemde onset, coda, enerji dagilimi ve gorsel bozulma gibi davranislar salt metrikle her zaman yakalanmaz. Onceki deneylerde de gorsel kontrol gerekliydi ve kullanici onayi istenen bir adim olarak tanimlandi. Bu nedenle qualitative kontrol yardimci degil, zorunlu bir degerlendirme parcasi olacak.`
- Dosya / artifact: `ML/autoencoder/experiments/PaperRepro/docs/03_acik_sorular_ve_karar_kaydi_tr.md`
- Durum: `Karar verildi`

## 11. Operasyon ve artifact kontrati

### Q50. Klasor yapisinda ilk gunden hangi alt klasorler acilacak?
- Neden kritik: sonradan log/artifact dagilmasin.
- Beklenen adaylar:
  - `configs/`
  - `setup/`
  - `core/`
  - `training/`
  - `evaluation/`
  - `results/`
  - `logs/`
- Karar: `Ilk gunden su alt klasorler acilacak: configs/, setup/, core/, training/, evaluation/, results/, logs/, docs/. Gerekli tum kod, freeze dosyalari, raporlar ve run artifactlari bu yapi icinde tutulacak; PaperRepro disina kod bagimliligi kurulmayacak.`
- Gerekce: `Bu branch bagimsiz ve temiz bir ana hat kurmak icin acildi. Sonradan log, config ve artifact dagilmasini onlemek icin klasor kontratinin ilk gunden sabitlenmesi gerekir. docs/ klasoru zaten aktif kullaniliyor; diger alt klasorler de veri hazirlama, cekirdek model, egitim, degerlendirme ve run ciktilarini ayirmak icin zorunludur.`
- Dosya / artifact: `ML/autoencoder/experiments/PaperRepro/README.md`
- Durum: `Karar verildi`

### Q51. Run naming ve checkpoint naming standardi ne olacak?
- Neden kritik: Stage-1 ve Stage-2 run'lari karisabilir.
- Karar: `Run naming yapisal ama okunur bir standartla sabitlenecek. Stage-1 ve Stage-2 run klasorleri sirasiyla run_YYYYMMDD_HHMM_s1_ae_<tag> ve run_YYYYMMDD_HHMM_s2_ledm_<tag> formatinda adlandirilacak. Tag, en az su bilgileri icerecek: hh100, ori4064, logspec128, lat8x32x32, evt801010, s42 ve versiyon etiketi. Checkpoint adlari last.ckpt ile birlikte Stage-1 icin best_recon.ckpt, Stage-2 icin best_val_loss.ckpt seklinde acik amac tasiyacak.`
- Gerekce: `Bu branch'te hedeflenen duzen, isimden stage, temsil kontrati, latent sekli, split ve seed bilgisinin hemen anlasilmasidir. Paper/tqdne'deki kisa isimler bu repo disiplini icin fazla belirsiz, asiri detayli isimler ise gereksiz karmasik olur. Yapisal ama okunur standart, eski x4/x8 adlandirma karmasasinin tekrar etmesini de engeller.`
- Dosya / artifact: `ML/autoencoder/experiments/PaperRepro/docs/03_acik_sorular_ve_karar_kaydi_tr.md`
- Durum: `Karar verildi`

### Q52. Runtime ve kaynak kullanimi nasil kaydedilecek?
- Neden kritik: paper ile ve bizim eski modellerle adil karsilastirma icin sure/GPU kullanimi lazim.
- Tartisilacak alt noktalar:
  - wall-clock sure
  - GPU memory peak
  - CPU / RAM
  - sample-per-second
- Karar: `Runtime ve kaynak kullanimi tam operasyonel paket olarak kaydedilecek. Her run icin training_wall_clock_sec, evaluation_wall_clock_sec, inference_wall_clock_sec, gpu_name, gpu_peak_memory_mb, cpu_ram_snapshot_mb ve samples_per_sec veya waveforms_per_sec zorunlu olacak. Bu bilgiler her run altinda resource_summary.json ve kisa bir resource_summary.md artifact'i olarak tutulacak.`
- Gerekce: `Bu alan basari kapisinin parcasi olmasa da paper ile ve bizim eski hatlarla adil karsilastirma icin duzenli kaynak kaydi gerekir. Yalnizca wall-clock tutmak eksik, sadece GPU memory tutmak da yetersiz olur. Bir kez standart paket kurulursa sonraki butun run'larda dusuk maliyetle tekrar kullanilir ve yorum gucu belirgin sekilde artar.`
- Dosya / artifact: `ML/autoencoder/experiments/PaperRepro/docs/03_acik_sorular_ve_karar_kaydi_tr.md`
- Durum: `Karar verildi`

### Q53. Uzun surecek isler nasil calistirilacak?
- Neden kritik: terminal kopmasi ve log kaybi olmamali.
- Tartisilacak secenekler:
  - `screen`
  - `nohup`
  - launcher script + log file standardi
- Karar: `Uzun surecek isler detached sekilde calistirilacak. Tercihli yontem launcher script + screen oturumu + standart log dosyasi olacak. Gerekirse nohup yedek yontem olarak kullanilabilir, ancak her run icin tekil log dosyasi ve acik run ismi zorunlu olacak.`
- Gerekce: `Bu repoda daha once uzun egitim ve evaluation sureclerinde terminal kopmasina dayaniklilik ve detayli log zorunlu hale geldi. Tek komutla dogrudan arka plana atmak yerine launcher + standart log disiplini kullanmak, tekrar etme ve hata ayiklama acisindan daha temizdir.`
- Dosya / artifact: `ML/autoencoder/experiments/PaperRepro/docs/03_acik_sorular_ve_karar_kaydi_tr.md`
- Durum: `Karar verildi`

### Q54. Reproducibility icin hangi tohumlar ve hangi config artifactlari zorunlu olacak?
- Neden kritik: tekrar uretilebilirlik icin minimum kayit seti tanimli olmali.
- Karar: `Ilk faz icin tam reproducibility paketi zorunlu olacak. Her run icin split_seed = 42, train_seed ve gerekiyorsa eval_seed kaydedilecek. Ayrica config_snapshot, split_manifest veya event listeleri, condition_norm_stats, representation_config, metric_config ve Stage-2 icin upstream_checkpoint_ref artifact'lari zorunlu olacak. Stage-2 latent cache veya encode-cikti artifact'i uretiliyorsa buna ait cache_seed, cache_manifest ve cache_build_config de zorunlu tutulacak. Dosya dagilimini onlemek icin bu artifact'lar run klasorunun altinda sabit alt dizinlerde tutulacak; ayni bilgi birden fazla yere kopyalanmayacak.`
- Gerekce: `Bu branch'te yarim reproducibility paketi yeterli degildir; ayni split, ayni normalization ve ayni upstream checkpoint olmadan sonuclar guvenli sekilde tekrar uretilemez. Ozellikle Stage-2 tarafinda posterior sample temelli latent kullaniliyorsa cache olusturma adimi da stochastic olabilir; bu nedenle latent cache varsa onun seed ve manifest bilgisi de acik artifact olarak kaydedilmelidir. Ote yandan artifact'lari rastgele dagitmak ileride VCS ve klasor takibini zorlastirir. Bu nedenle hem tam kayit seti hem de tekil, standart run-alti yerlesim zorunlu tutulur.`
- Dosya / artifact: `ML/autoencoder/experiments/PaperRepro/docs/03_acik_sorular_ve_karar_kaydi_tr.md`
- Durum: `Karar verildi`

## 12. Paper sapmalari ve karar sinirlari

### Q55. Paper'dan izinli sapmalarin listesi ne olacak?
- Neden kritik: paper-faithful iddiasi ancak sapmalar acik yazilirsa savunulabilir.
- Olasi sapmalar:
  - event-wise split
  - VS30 eksigi
  - azimuthal gap tanim farki
  - farkli pencere uzunlugu
  - farkli STFT boyutu
- Karar: `Ilk faz icin izinli sapmalar su listeyle sinirlanacak: (1) random sample split yerine event-wise train/val/test + ayri event-heldout OOD, (2) VS30 kullanilmamasi, (3) station elevation kullanilmamasi, (4) station embedding'in ana modelde kullanilmamasi, (5) azimuthal gap'in paper'daki hazir feature yerine phase bulletinlerden event-level turetilmesi, (6) veri kaynagi olarak bizim external_dataset + HH-only kontratinin kullanilmasi, (7) paper condition setinin yalnizca mevcut/turetilebilir kisimlariyla sinirli kalinmasi.`
- Gerekce: `Bu sapmalarin tamami daha once veri, split ve condition kontratlarinda tartisildi ve veri gercegimizden kaynaklaniyor. Paper-faithful iddiasini savunulabilir tutmanin tek yolu, bu sapmalari bastan acikca yazmaktir. Buna karsin temsil ailesi, latent diffusion yapisi ve temel STFT/log-spectrogram hatti korunuyor.`
- Dosya / artifact: `ML/autoencoder/experiments/PaperRepro/docs/05_paper_sapmalari_tr.md`
- Durum: `Karar verildi`

### Q56. Hangi noktada "paper-faithful temel hat kuruldu" diyecegiz?
- Neden kritik: bitis kriteri olmadan surekli yeni sey eklenir.
- Tartisilacak minimumlar:
  - Stage-1 calisiyor
  - Stage-2 latent EDM calisiyor
  - event-wise test sonucu var
  - OOD sonucu var
  - paper-style en az bir distribution metric var
- Karar: `Paper-faithful temel hat kuruldu denebilmesi icin su kosullar birlikte saglanmis olacak: Stage-1 egitimi tamamlanmis olacak, Stage-2 latent EDM egitimi tamamlanmis olacak, Stage-2 gecerli sample uretebilecek, IID test raporu olacak, ayri OOD event raporu olacak, primary generation metric seti hesaplanmis olacak, minimum paper-style metric paketi hesaplanmis olacak, zorunlu gorsel artifact paketi uretilmis olacak, resource summary mevcut olacak ve reproducibility artifact'lari eksiksiz kaydedilmis olacak.`
- Gerekce: `Temel hat kuruldu demek icin sadece egitimin bitmis olmasi yeterli degildir; teknik olarak calisan, raporlanmis ve tekrar uretilebilir bir iki-asamali sistem gerekir. Buna karsin eski referanslarla tam benchmark veya ileri ablation'lar bu esigin parcasi olmak zorunda degildir. Bu nedenle en dogru bitis kriteri, calisan Stage-1 ve Stage-2'nin test/OOD, ana kalite metrikleri, paper-style minimum paket, gorsel paket ve operasyonel/reproducibility artifact'lari ile birlikte tamamlanmis olmasidir.`
- Dosya / artifact: `ML/autoencoder/experiments/PaperRepro/docs/03_acik_sorular_ve_karar_kaydi_tr.md`
- Durum: `Karar verildi`

### Q57. Hangi noktada ek arastirma eksenlerine gecis serbest olacak?
- Neden kritik: temel hat oturmadan ablation / extension baslatmak bu kutuyu da kirletir.
- Karar: `Ek arastirma eksenlerine ancak temel iki-asamali hat kurulduktan sonra gecilecek. Bu esik icin en az su kosullar saglanmis olacak: Stage-1 egitimi tamamlanmis ve calisir durumda, Stage-2 latent EDM egitimi tamamlanmis ve sample uretebiliyor, event-wise test raporu var, ayri OOD raporu var, en az bir paper-style distribution metric uretilmis, oracle ve inverse ceiling raporu alinmis. Bu noktadan once station embedding, normalization ablation, ek derived feature, FiLM/AdaGN benzeri alternative conditioning injection'lari, CFG ve benzeri extension'lar acilmayacak.`
- Gerekce: `Bu branch'in temel riski, eski kutular gibi erken ablation ve extension acarak tekrar karmasiklasmasidir. Temel hat bitmeden yeni eksen acmak yorum kabiliyetini dusurur. Bu nedenle extension kapisi, teknik olarak calisan ve raporlanmis bir temel hat uzerinden acilmalidir. FiLM/AdaGN gibi conditioning ablation'lari da bu ikinci faza bilincli olarak ertelenmistir.`
- Dosya / artifact: `ML/autoencoder/experiments/PaperRepro/docs/03_acik_sorular_ve_karar_kaydi_tr.md`
- Durum: `Karar verildi`

## 13. Kapanis notu

Bu dosya tamamlansin diye degil, tartisma sirasini zorlamak icin yazildi.
Kodlamaya gecis kriteri su olacak:
- veri kontrati kapandi
- metadata kontrati kapandi
- representation kontrati kapandi
- Stage-1 ve Stage-2 minimum kontratlari kapandi
- sapmalar kayda gecti

Bu seviyeye gelmeden model implementasyonu baslatilmayacak.
