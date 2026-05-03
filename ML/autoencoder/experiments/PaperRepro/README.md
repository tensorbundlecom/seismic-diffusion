# PaperRepro

## Amac

Bu klasor, `2410.19343v2` makalesindeki ana hatla uyumlu yeni bir model cizgisi kurmak icin acildi.
Amac, mevcut `experiments/*` dallarini yamayarak degil, sifirdan ama kontrollu bir sekilde paper-faithful bir hat kurmaktir.

Bu kutu:
- mevcut `ML/autoencoder/experiments/*` deneylerini degistirmez
- tarihsel modelleri referans olarak tutar
- event-wise split ve OOD disiplinini korur
- paper tarafindaki 2-stage mantigi benimser: `Stage-1 compressor VAE + Stage-2 latent EDM`

## Bu branch neden ayri?

Mevcut calisma dizininde `experiments2/` altinda ayri ve kirli bir WIP durumu var.
Bu nedenle veri kaybi riskini sifira indirmek icin yeni bir local clone ve yeni branch acildi:

- clone: `/home/gms/kalem_seismic_paper_repro`
- branch: `paper-repro-edm`

Bu sayede:
- mevcut checkout oldugu gibi korunur
- yeni model hatti bagimsiz gelisir
- ileride dogrudan karsilastirma daha temiz olur

## Ilk dokumanlar

- `ML/autoencoder/experiments/PaperRepro/docs/01_kalem_vs_paper_karsilastirma_tr.md`
- `ML/autoencoder/experiments/PaperRepro/docs/02_paper_repro_tasarim_plani_tr.md`
- `ML/autoencoder/experiments/PaperRepro/docs/03_acik_sorular_ve_karar_kaydi_tr.md`
- `ML/autoencoder/experiments/PaperRepro/docs/04_hh_units_hizli_audit_tr.md`
- `ML/autoencoder/experiments/PaperRepro/docs/05_paper_sapmalari_tr.md`

## Ilk sabit kararlar

- Yeni hat, mevcut `DDPMvsDDIM/` klasorunun uzerine patch atilarak kurulmayacak.
- Paper'a yakin olmak icin `vector latent + MLP denoiser` yerine `2D latent + 2D U-Net + EDM` yonune gidilecek.
- Paper'daki random sample split taklit edilmeyecek; bizim event-wise split ve OOD disiplini korunacak.
- Condition set, istasyon kimligi merkezli degil fiziksel scalar predictor merkezli olacak.
- Paper ile tam ayni olmayan sapmalar acikca dokumante edilecek.
