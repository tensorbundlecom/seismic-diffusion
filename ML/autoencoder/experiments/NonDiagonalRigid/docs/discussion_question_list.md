# Rigid Deney Tasarimi - Detayli Soru Listesi

Bu listeyi sirayla cevaplayip her cevabi `protocol/decisions.md` dosyasina karar olarak yazacagiz.

## A) Arastirma Kapsami ve Ana Iddia

1. Birincil iddiamiz tam olarak nedir? (tek cumle)
2. Iddia `VAE genel teorisi` mi, yoksa `bu problem ailesi` icin pratik bir policy mi?
3. Birincil cikti hedefi nedir?
4. Basariyi belirleyen tek ana metrik olacak mi, yoksa composite skor mu?
5. Sonucumuzun hangi kisimlari `kesif`, hangi kisimlari `dogrulama` (confirmatory) olacak?

## B) Degerlendirme Senaryosu (En Kritik)

6. Ana degerlendirme senaryosu hangisi olacak?
7. `Reconstruction` testi zorunlu mu?
8. `Conditional generation` testi zorunlu mu?
9. Eger iki senaryo da olacaksa, birincil karar reconstruction uzerinden mi generation uzerinden mi verilecek?
10. Inference protocolunde `z=mu` mu, `z~q(z|x)` mi, `z~p(z|c)` mi kullanilacak?
11. Generation testinde kac ornekleme (sample count) alinacak?
12. Degerlendirme tek-run deterministic mi, yoksa multi-sample expectation mi?

## C) Veri ve Split Katiligi

13. Hangi train/val/test/OOD dosya listeleri dondurulacak?
14. Tum listeler hash ile kaydedilecek mi?
15. Istasyon subseti sabit mi?
16. Event-level leakage’i nasil engelleyecegiz?
17. Condition normalization istatistikleri sadece train splitten mi hesaplanacak?
18. OOD seti tek mi olacak yoksa iki OOD seti mi (yakın ve uzak dagilim)?

## D) Model Karsilastirma Adaleti (Fairness)

19. Karsilastirma `iso-step`, `iso-epoch`, `iso-time` hangisine gore adil sayilacak?
20. Early stopping tek policy ile tum modellere ayni mi uygulanacak?
21. LR schedule tum modellerde birebir ayni mi olacak?
22. Parametre farki cok buyuk modellerde optimizer ayarlari ayni kalacak mi?
23. Seed sayisi kac olacak? (minimum 3 onerilir)
24. Her model icin kac tekrar (replicate) yapilacak?

## E) Mimari Tarama Matrisinin Sabitlenmesi

25. Backbone seviyeleri neler olacak? (orn: large/small/xsmall)
26. Latent boyut listesi ne olacak? (orn: 16/32/64/96/128/160)
27. FullCov ve Baseline her noktada birebir esit taranacak mi?
28. Karsilastirma matrisi once dondurulup sonra hic degistirilmeyecek mi?
29. Erken durdurma olmadan minimum egitim suresi (warmup epoch/step) tanimlanacak mi?

## F) Metrik Seti ve Karar Kurali

30. Birincil kalite metrikleri hangileri olacak?
31. Ikinci kalite metrikleri hangileri olacak?
32. Hangi metrikler `higher is better`, hangileri `lower is better` olarak sabitlenecek?
33. Composite Quality Score formulunu onceden sabitleyecek miyiz?
34. Agirlikli skor mu esit agirlik mi kullanacagiz?
35. Metriklerin bootstrap CI’si zorunlu olacak mi?
36. Coklu karsilastirma duzeltmesi (FDR vb.) uygulanacak mi?

## G) Latent Bagimlilik Analizi (Offdiag + TC/MI)

37. Offdiag hangi seviyede olculcek? (`q(z|x)`, aggregated posterior, her ikisi)
38. Offdiag ozetleri hangileri olacak? (mean, p95, max, energy ratio)
39. `TC` tanimi tam olarak ne olacak? (Gaussian TC / estimator secimi)
40. `MI` nasil olculecek? (pairwise Gaussian MI / estimator)
41. Basis-rotation kontrolu zorunlu mu?
42. Rotation sayisi kac olacak?
43. Bagimlilik icin tek indeks (`ODI`) tanimlanacak mi?

## H) Policy ve Stop Rule

44. Asagidaki policy’yi resmi karar kurali yapiyor muyuz?
45. `Q_{k+1} >= Q_k - eps` ve `ODI_{k+1} <= ODI_k + delta` ise devam, aksi halde dur.
46. `eps` ve `delta` degerleri ne olacak?
47. Nihai secim: `en kucuk kabul edilebilir model` mi, yoksa `Pareto nondominated` mi?
48. Policy reconstruction ve generation senaryolari icin ayri mi olacak?

## I) Raporlama ve Itirazlara Kapanis

49. Tum runlar icin zorunlu rapor tablolari hangileri olacak?
50. Tum grafiklerin zorunlu listesi nedir?
51. Hangi failure case’leri raporda zorunlu yazacagiz?
52. `Negative result` raporlamasi zorunlu mu?
53. Reproducibility checklist maddeleri neler olacak?
54. Makale iddia siniri (scope boundary) nasil yazilacak?

## J) Ilk Oturumda Cevaplanmasi Gereken Cekirdek Sorular

55. Birincil degerlendirme reconstruction mi generation mi?
56. Fairness tanimi (`iso-step`/`iso-time`) hangisi?
57. Latent tarama listesi ne?
58. Birincil kalite metrik seti ne?
59. `ODI` tanimi ne?
60. Stop rule esikleri (`eps`, `delta`) ne?

## K) OffDiagonal Temelli Minimum Kanit Seti (Rigid)

61. OffDiagonal sinyalinin gercek oldugunu (noise degil) gosteren minimum kanit ne?
62. OffDiagonal/ODI'nin model seciminde ongoru degeri oldugunu gosteren minimum kanit ne?
63. ODI-policy'nin kalite-only policy'ye gore pratik fayda sagladigini gosteren minimum kanit ne?
64. Bulgularin model ailesi ve OOD setlerinde tekrarlandigini gosteren minimum kanit ne?

## L) Backbone Kucultme Metodolojisi (Yeni Tartisma)

65. Backbone kucultmeyi latent taramasindan once ayri bir fazda mi test edecegiz?
66. Kucultme politikalari hangileri olacak? (`width-only`, `depth-only`, `hybrid`)
67. Kucultme seviyeleri ne olacak? (orn. `1.0`, `0.75`, `0.5`)
68. Stage-1 (policy secimi) icin latent sabit deger ne olacak? (oneri: `ld=128`)
69. Policy secimi hangi karar kurali ile yapilacak? (`Q` + guardrail + `ODI`)
70. Receptive field/downsample yapisini sabit tutmayi zorunlu kiliyor muyuz?
71. Stage-1 ve Stage-2 toplam run butcesi nasil sabitlenecek?
72. Nihai secimde "en kucuk model" oncesinde "en iyi kucultme politikasi" kapisi olacak mi?
