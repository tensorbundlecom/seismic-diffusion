# Setup

Bu klasor, kodlamaya gecmeden once gerekli veri-hazirlama ve audit adimlarini tutar.

Ilk scriptler:
- `paths.py`: sabit yol kontrati
- `audit_external_dataset.py`: HH veri kontratini hizli dogrulama
- `metadata.py`: event katalogu, phase pick ve 1D hiz modeli yardimcilari
- `build_condition_table.py`: canonical sample-level condition manifesti uretir
- `build_eventwise_split.py`: event-wise split, sample manifest ve condition norm istatistiklerini uretir
- `windowing.py`: origin-time sabit pencereyi waveform'dan cikartir
- `build_representation_smoke.py`: log-spectrogram tensor kontratini kucuk ornekle dogrular
- `audit_stage1_dataset.py`: Stage-1 dataset nesnesinin splitler boyunca tensor kontratini dogrular
- `build_stage2_latent_cache.py`: egitilmis Stage-1 checkpoint'ten Stage-2 latent cache ve latent norm istatistiklerini uretir

Kurallar:
- baska deney klasorlerinden kod import edilmez
- veri okunabilir, ama kod bagimliligi `PaperRepro/` disina kurulmaz
- audit ciktilari `results/setup/` altina yazilir
- Stage-2 cache artifactlari `results/stage2_cache/<stage1_run_name>/` altina yazilir
