# Training

Bu klasor Stage-1 ve Stage-2 egitim giris noktalarini tutacak.

Mevcut:
- `train_stage1_autoencoder.py`
  - `best_recon.ckpt` monitoru: `validation/reconstruction_loss`
  - `last.ckpt` ve `resource_summary.json` uretir
  - `--dry-run` ile tek-batch smoke calistirabilir
- `train_stage2_edm.py`
  - `best_val_loss.ckpt` monitoru: `validation/loss`
  - `last.ckpt` ve `resource_summary.json` uretir
  - `--dry-run` ile latent cache uzerinde tek-batch smoke calistirabilir
- `train_paper_metrics_classifier.py`
  - paper-style minimum paket icin gereken classifier'i egitir
  - `best_val_accuracy.pt`, `last.pt`, `real_data_eval.json` ve `resource_summary.json` uretir
  - `--dry-run` ile tek-batch classifier smoke calistirabilir

Plan:
- gerekirse detached calisma icin launcher scriptleri
