# experiments2

Bu klasor temiz bir ikinci deney hatti icindir.
Hedef: az ama net dosya yapisi ile adim adim ilerlemek.

## Ilkeler

- Eski `experiments/` klasorune dokunma.
- Her karar `protocol/` altinda kayitli olsun.
- Her kod dosyasi satir satir aciklanacak.
- Run ciktilari tek yerde (`runs/`) toplansin.

## Klasor Yapisi (Alternatif 1)

- `protocol/`: kararlar ve deney tanimlari
- `configs/`: sabitlenmis yaml/json config dosyalari
- `src/`: dataset, model, train, eval, visualize kodu
- `runs/`: run bazli ciktilar (log, checkpoint, metric, plot)

## Exp001 Kod Durumu

- `src/dataset.py`:
  - External HH waveform manifest hazirlama
  - Event-wise + ayri OOD split freeze
  - Train-only condition normalizasyonu
  - Train-only global RMS complex STFT olcegi
  - Z-only complex STFT (`2x128x220`)
- `src/model.py`:
  - CVAE (`q(z|x,c)`, condition encoder + station embedding)
  - Condition-only sample (`z ~ N(0,I)`)
- `src/train.py`:
  - KL warmup + free-bits
  - `best_val_loss.pt` + `best_condgen_composite.pt`
  - D013 pre-gate + robust-z composite
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

### 3) Checkpoint degerlendirme

```bash
python3 -m ML.autoencoder.experiments2.src.evaluate \
  --config ML/autoencoder/experiments2/configs/exp001_base.json \
  --checkpoint ML/autoencoder/experiments2/runs/exp001/run_YYYYMMDD_HHMM_exp001_base/checkpoints/best_condgen_composite.pt \
  --split all
```

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

`runs/exp001/run_YYYYMMDD_HHMM/`

Bu klasorun icinde:

- `config_resolved.yaml`
- `train.log`
- `checkpoints/`
- `metrics/`
- `plots/`
