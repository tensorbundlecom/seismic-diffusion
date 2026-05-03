# Stage-2 Metric Gap Audit

Tarih: `2026-04-27`

Bu not, Stage-2 train ve ilk evaluation tamamlandiktan sonra karar kaydina gore zorunlu olup henuz eksik veya hatali olan metric/artifact'lari kapatmak icin acildi.

## Tespit Edilen Bosluklar

1. `classifier_accuracy` eksikti.
2. `classifier_embedding_fd` eksikti.
3. `magnitude-bin` ve `distance-bin` tablolari eksikti.
4. `envelope_similarity` implementasyonu hataliydi.
   - eski hesap, `(C, T)` waveformda `axis=1` ile kanal-basina tek skalar cikariyor ve zaman-serisi envelope korelasyonu olcmuyordu.
5. Stage-2 gorsel paketinde `magnitude-bin` ve `distance-bin` kapsayan acik temsilci grid artifact'i eksikti.

## Alinan Aksiyonlar

1. PaperRepro classifier stage eklendi:
   - `ML/autoencoder/experiments/PaperRepro/training/train_paper_metrics_classifier.py`
2. Classifier dataset/model eklendi:
   - `ML/autoencoder/experiments/PaperRepro/core/classifier_datasets.py`
   - `ML/autoencoder/experiments/PaperRepro/core/paper_metrics_classifier.py`
   - `ML/autoencoder/experiments/PaperRepro/core/binning.py`
3. Stage-2 evaluation script genislestirildi:
   - classifier checkpoint zorunlu hale getirildi
   - `bin_metrics_<split>.json`
   - `bin_metrics_<split>.md`
   - `test/ood magnitude_bin_grid`
   - `test/ood distance_bin_grid`
   artifactlari eklendi
4. `envelope_similarity` hesaplamasi zaman ekseni uzerinden duzeltildi.

## Acik Durum

- Classifier full training baslatildi.
- Classifier checkpoint hazir olur olmaz Stage-2 evaluation yeniden kosulacak.
- Bu rerun bitmeden Stage-2 icin “tum zorunlu metric paketi tamamlandi” denmeyecek.
