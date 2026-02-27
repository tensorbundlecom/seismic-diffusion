# configs

Bu klasor deney konfig dosyalari icindir.
Her deney icin tek bir ana config tutulacak.

- `exp001_base.json`: exp001 frozen-protocol ana config.
- `exp001_smoke.json`: hizli smoke testi icin kucuk konfig.
- `exp001_stage2_weighted.json`: Stage-2 soft-weighted sampler kosusu icin hazir config.

Imbalance guardrail notu:

- Stage-2 (soft weighted sampler) kosularinda `train.use_weighted_sampler=true` ise
  `imbalance_guardrails.reference_cond_eval_json` zorunludur.
- Bu dosya Stage-1 (dogal dagilim) referansindan `by_bin_mean` metriklerini tasir
  (`condition_only.by_bin_mean` veya esdeger yol).
- D015 geregi, weighted kosuda hicbir aday guardrail gecemezse run reject edilir.

Ornek referans path:

- `ML/autoencoder/experiments2/runs/exp001/run_YYYYMMDD_HHMM_exp001_base/metrics/cond_eval_epoch_XXX.json`

Not:

- `exp001_stage2_weighted.json` su an `protocol/reports/stage1_reference_cond_eval_smoke_epoch002.json`
  referansini kullanir (smoke-provisional).
- Full Stage-1 base run tamamlandiginda bu path'in full-stage1 referansi ile guncellenmesi onerilir.
