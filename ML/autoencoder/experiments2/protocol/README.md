# protocol

Bu klasor, `experiments2/exp001` icin hem dokumantasyon hem de frozen artifactleri tutar.

## Alt Klasorler

- `docs/`
  - Yorumlanabilir, insan-okur dokumanlar.
  - Ornek: karar kayitlari, deney tanimi, problem ve tasarim notlari.
  - Giris noktasi: `docs/README.md`

- `frozen/base/`
  - Full exp001 icin sabitlenen artifact dosyalari.
  - Ornek: `manifest_exp001.jsonl`, `frozen_event_splits_exp001.json`, `normalization_stats_exp001.json`.

- `frozen/smoke/`
  - Smoke konfigu icin ayni artifact setinin kucuk versiyonu.

- `reports/`
  - Tekrarlanabilir audit/sanity ciktilari.
  - Ornek: manifest tutarlilik auditleri, travel-time sanity ozetleri, D015 Stage-1 referans dosyalari.

## Isletim Kurali

- Kod tarafinda artifact konumlari dogrudan config'ten okunur (`configs/exp001_base.json`, `configs/exp001_smoke.json`).
- Ayni frozen dosyalar kullanildiginda split/OOD sabit kalir.
- `--force-manifest` veya `--force-split` verilirse frozen artifactler yeniden uretilir.

## Path Notlari

- Base artifact pathleri:
  - `protocol/frozen/base/frozen_event_splits_exp001.json`
  - `protocol/frozen/base/manifest_exp001.jsonl`
  - `protocol/frozen/base/normalization_stats_exp001.json`
- Smoke artifact pathleri:
  - `protocol/frozen/smoke/frozen_event_splits_exp001_smoke.json`
  - `protocol/frozen/smoke/manifest_exp001_smoke.jsonl`
  - `protocol/frozen/smoke/normalization_stats_exp001_smoke.json`
