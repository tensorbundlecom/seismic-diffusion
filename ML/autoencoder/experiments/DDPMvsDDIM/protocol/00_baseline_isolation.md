# Baseline Isolation Note

## Amac
`LegacyCondBaseline` mimarisini `DDPMvsDDIM` kutusuna tasimak ve ilk asamada sadece lokal kod ile calistigini dogrulamak.

## Frozen Kurallar
- Baska deney kutularindan Python importu yok.
- Ortak dataset kodu da lokal kopya.
- Veri dosyalari repo-level `data/` altindan okunabilir.
- Davranis Stage-0 icin bilincli olarak `LegacyCondBaseline` ile ayni tutulur.

## Yapilan Lokalizasyon
- `LegacyCondBaseline/core/model_wbaseline.py`
  -> `DDPMvsDDIM/core/model_legacy_cond_baseline.py`
- `LegacyCondBaseline/core/loss_utils.py`
  -> `DDPMvsDDIM/core/loss_utils.py`
- `General/core/stft_dataset.py`
  -> `DDPMvsDDIM/core/stft_dataset.py`
- `LegacyCondBaseline/training/train_wbaseline_external.py`
  -> `DDPMvsDDIM/training/train_legacy_cond_baseline_external.py`

## Eklenen Dogrulamalar
- import/forward/sample smoke testi
- limitli batch ile 1-epoch smoke training
- grep tabanli dis import audit

## Bilincli Olarak Henuz Yapilmayanlar
- DDPM implementation
- DDIM sampler
- latent cache pipeline
- diffusion-specific evaluation
