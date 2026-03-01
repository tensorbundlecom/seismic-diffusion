# D013 Gate Audit Protocol

Bu dokuman, condition-only iki-asamali gate esiklerinin (D013) dogrulanmasi ve kalibrasyonu icindir.

## Amaç

- Stage-1 onset-health gate'in kotu epochlari gecirip gecirmedigini (`false accept`) olcmek
- Stage-2 quality gate'in secim davranisini olcmek
- Gate'in iyi epochlari gereksiz eleyip elemedigini (`false reject`) olcmek
- Eger gerekiyorsa esikleri sistematik sweep ile guncellemek

## Girdi

- Bir veya daha fazla run klasoru (`runs/exp001/run_...`)
- Her run altinda `metrics/cond_eval_epoch_*.json`
- Opsiyonel manuel etiket CSV (`run,epoch,label`)

## Etiket Kaynagi

1. **Manual (onerilen)**:
   - Etiketler `good/bad` olarak uzman inceleme ile verilir.
2. **Proxy (gecici)**:
   - Script, ortogonal kalite metriklerinden otomatik label uretir:
     - `xcorr_max`
     - `envelope_corr`
     - `mr_lsd`
     - `onset_mae_dtps_s`

Not: Proxy ile elde edilen threshold onerisi **provisional** kabul edilir.

## Komut

```bash
python3 -m ML.autoencoder.experiments2.src.audit_gate \
  --run-dirs \
    ML/autoencoder/experiments2/runs/exp001/run_20260301_0258_exp001_e1_kl_tune_s42 \
    ML/autoencoder/experiments2/runs/exp001/run_20260301_0616_exp001_e2_condpath_weaken_s42 \
    ML/autoencoder/experiments2/runs/exp001/run_20260301_0934_exp001_e3_bandloss_s42 \
  --quality-enabled \
  --out-json ML/autoencoder/experiments2/protocol/reports/gate_audit_exp001_proxy.json \
  --out-md ML/autoencoder/experiments2/protocol/reports/gate_audit_exp001_proxy.md
```

Manual labels ile:

```bash
python3 -m ML.autoencoder.experiments2.src.audit_gate \
  --run-dirs ML/autoencoder/experiments2/runs/exp001/run_20260301_0258_exp001_e1_kl_tune_s42 \
  --quality-enabled \
  --labels-csv ML/autoencoder/experiments2/protocol/reports/gate_labels_manual.csv \
  --out-json ML/autoencoder/experiments2/protocol/reports/gate_audit_exp001_manual.json \
  --out-md ML/autoencoder/experiments2/protocol/reports/gate_audit_exp001_manual.md
```

Opsiyonel: mevcut frozen threshold'lari explicit vermek istersen:

```bash
  --current-min-onset-evaluable-rate 0.60 \
  --current-max-onset-failure-p 0.05 \
  --current-max-onset-failure-s 0.35 \
  --current-max-abs-xcorr-lag-s 3.0 \
  --current-min-xcorr-max 0.74 \
  --current-min-envelope-corr 0.69 \
  --current-max-mr-lsd 0.0195 \
  --current-max-onset-mae-dtps-s 2.0
```

## Cikti

- JSON:
  - `current`: mevcut gate performansi
  - `recommended`: sweep top-1 threshold
  - `sweep_topk`: en iyi kombinasyonlar
  - `disagreements`: gate-etiket uyusmazlik listesi
- Markdown:
  - hizli yonetsel ozet

## Karar Kurali

Esik guncellemesi icin minimum:

- `bad_accept_rate` mevcuttan daha iyi olmali
- `good_reject_rate` kabul edilebilir seviyede kalmali
- Owner manuel inceleme ile `disagreements` listesi onaylanmali

## Etiket CSV Formati

`protocol/reports/gate_labels_template.csv` dosyasini baz alin:

- `run`: run klasor adi veya tam run yolu
- `epoch`: epoch numarasi
- `label`: `good` veya `bad`
- `note`: opsiyonel aciklama
