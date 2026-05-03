# Setup Results

Bu klasorde ilk veri-hazirlama asamasinin uretilmis artifact'lari tutulur.

Ana dosyalar:
- `audit_external_dataset_hh.json`
- `audit_external_dataset_hh.md`
- `condition_manifest_paper_repro_v1.meta.json`
- `condition_manifest_paper_repro_v1.md`
- `event_splits_paper_repro_v1.json`
- `event_splits_paper_repro_v1.md`
- `sample_manifest_paper_repro_v1.meta.json`
- `condition_norm_stats_paper_repro_v1.json`
- `representation_smoke_paper_repro_v1.json`
- `representation_smoke_paper_repro_v1.md`
- `stage1_dataset_audit_paper_repro_v1.json`
- `stage1_dataset_audit_paper_repro_v1.md`

Notlar:
- `*.jsonl` manifest dosyalari buyuk oldugu icin `.gitignore` ile disarida tutulur.
- `window_has_full_coverage` ve `requires_left_pad` gibi alanlar, origin-time pencere kontratinin ne kadar veri kaybi/padding gerektirdigini izlemek icin tutulur.
- `output_range: [-1, 1]` normalization nominal araliktir; `log_max=3` ustundeki enerji cepleri hard-clamp edilmedigi icin representation degerleri 1.0 ustune cikabilir. Bu, released `tqdne` davranisiyla uyumludur.
