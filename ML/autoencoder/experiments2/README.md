# experiments2

Bu klasor temiz bir ikinci deney hatti icindir.
Hedef: az ama net dosya yapisi ile adim adim ilerlemek.

## Ilkeler

- Eski `experiments/` klasorune dokunma.
- Her karar `protocol/` altinda kayitli olsun.
- Her kod dosyasi satir satir aciklanacak.
- Run ciktilari tek yerde (`runs/`) toplansin.

## Klasor Yapisi (Alternatif 1)

- `protocol/docs/`: kararlar, deney tanimi, domain-gerekce dokumanlari
- `protocol/frozen/base/`: full exp001 frozen artifactleri (manifest/split/stats)
- `protocol/frozen/smoke/`: smoke frozen artifactleri
- `protocol/reports/`: audit ve sanity raporlari
- `configs/`: sabitlenmis yaml/json config dosyalari
- `src/`: dataset, model, train, eval, visualize kodu
- `runs/`: run bazli ciktilar (log, checkpoint, metric, plot)

## Exp001 Kod Durumu

- `src/dataset.py`:
  - External HH waveform manifest hazirlama
  - Strict manifest integrity (`fail_on_manifest_drop`, `max_manifest_drop_rate`, drop-reason report)
  - Event-wise + ayri OOD split freeze
  - Train-only condition normalizasyonu
  - Train-only global RMS complex STFT olcegi
  - Z-only complex STFT (`2x128x220`)
- `src/model.py`:
  - CVAE (`q(z|x,c)`, condition encoder + station embedding)
  - Posterior `logvar` icin sayisal stabilite clamp'i (`[-8, 4]`)
  - Condition-only sample (`z ~ N(0,I)`)
- `src/train.py`:
  - KL warmup + free-bits
  - `best_val_loss.pt` + `best_condgen_composite.pt`
  - D013 pre-gate + robust-z composite
  - D015 imbalance guardrail (stage-2 weighted run icin)
  - Top-3 condgen aday ve final rerank
- `src/evaluate.py`:
  - D008 metrikleri
  - D014 onset picker (max-derivative + confidence)
  - Condition-only K-sampling (`mean ± std`)
  - Bin-wise rapor (`all`, `M<3`, `3<=M<5`, `M>=5`)
- `src/visualize.py`:
  - Target/Reconstruction/Condition-only STFT + waveform gorselleri

## Calistirma

### 1) Smoke (hizli kontrol)

```bash
python3 -m ML.autoencoder.experiments2.src.train \
  --config ML/autoencoder/experiments2/configs/exp001_smoke.json \
  --run-tag smoke
```

### 2) Full Exp001

```bash
python3 -m ML.autoencoder.experiments2.src.train \
  --config ML/autoencoder/experiments2/configs/exp001_base.json \
  --run-tag exp001_base
```

Not (D015):

- Stage-2 weighted run acarken `train.use_weighted_sampler=true` ise
  `imbalance_guardrails.reference_cond_eval_json` alanina Stage-1 referans dosyasi verilmelidir.
- Aksi halde train guardrail setup asamasinda hata verir.
- Referans verilse bile hicbir aday D015 guardrail'i gecemezse run `reject` edilir ve train hata ile sonlanir.

### 2.1) Stage-1 -> Stage-2 (onerilen akis)

1. Stage-1 (dogal dagilim) kos:

```bash
python3 -m ML.autoencoder.experiments2.src.train \
  --config ML/autoencoder/experiments2/configs/exp001_base.json \
  --run-tag exp001_base
```

2. Stage-1 run icinden referans dosyayi sec:

- Ornek dosya: `runs/exp001/run_..._exp001_base/metrics/cond_eval_epoch_XXX.json`
- Dosya icinde `cond_eval.by_bin_mean` alaninin oldugunu dogrula.

3. Stage-2 configte referansi ayarla:

- `ML/autoencoder/experiments2/configs/exp001_stage2_weighted.json`
- `imbalance_guardrails.reference_cond_eval_json = "<stage1_cond_eval_json_path>"`
- Mevcut configte smoke-provisional referans tanimlidir:
  - `ML/autoencoder/experiments2/protocol/reports/stage1_reference_cond_eval_smoke_epoch002.json`
- Full Stage-1 base run sonrasi bu path'i base referansla guncelle.

4. Stage-2 (weighted) kos:

```bash
python3 -m ML.autoencoder.experiments2.src.train \
  --config ML/autoencoder/experiments2/configs/exp001_stage2_weighted.json \
  --run-tag exp001_stage2_weighted
```

### 3) Checkpoint degerlendirme

```bash
python3 -m ML.autoencoder.experiments2.src.evaluate \
  --config ML/autoencoder/experiments2/configs/exp001_base.json \
  --checkpoint ML/autoencoder/experiments2/runs/exp001/run_YYYYMMDD_HHMM_exp001_base/checkpoints/best_condgen_composite.pt \
  --split all
```

## Kritik Pathler

- Stage-1 base config: `ML/autoencoder/experiments2/configs/exp001_base.json`
- Stage-1 smoke config: `ML/autoencoder/experiments2/configs/exp001_smoke.json`
- Stage-2 weighted config: `ML/autoencoder/experiments2/configs/exp001_stage2_weighted.json`
- Frozen artifacts (base): `ML/autoencoder/experiments2/protocol/frozen/base/`
- Frozen artifacts (smoke): `ML/autoencoder/experiments2/protocol/frozen/smoke/`
- Protocol docs index: `ML/autoencoder/experiments2/protocol/docs/README.md`

### 4) Gorsellestirme

```bash
python3 -m ML.autoencoder.experiments2.src.visualize \
  --config ML/autoencoder/experiments2/configs/exp001_base.json \
  --checkpoint ML/autoencoder/experiments2/runs/exp001/run_YYYYMMDD_HHMM_exp001_base/checkpoints/best_condgen_composite.pt \
  --split test \
  --num-samples 12
```

## Bagimliliklar

Bu deneyin calismasi icin en az su Python paketleri gerekir:

- `torch`
- `numpy`
- `scipy`
- `matplotlib`
- `obspy`

## Standart Run Agaci

Her run su formatta acilacak:

`runs/exp001/run_YYYYMMDD_HHMM_<tag>/`

Bu klasorun icinde:

- `config_resolved.yaml`
- `train.log`
- `checkpoints/`
- `metrics/`
- `plots/`
- `tmp/`

## Split/OOD Sabitleme Kurali

- OOD event seti ve train/val/test event setleri `protocol/frozen/{base|smoke}/frozen_event_splits_*.json` dosyalarindan okunur.
- Ayni config + ayni frozen split dosyasi kullanildiginda OOD **degismez**.
- Asagidaki durumlarda split/OOD yeniden olusabilir:
  - `--force-split`
  - `--force-manifest` (manifest degisirse split cache invalid olur)
  - split config degisiklikleri (`seed`, `ood_event_ratio`, oranlar, policy)
- Not: `exp001_base.json` ve `exp001_smoke.json` farkli artifact dosyalari kullandigi icin OOD setleri farklidir.
