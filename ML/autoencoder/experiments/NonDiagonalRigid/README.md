# NonDiagonalRigid

Bu klasor, `NonDiagonel` altindaki kesif (exploratory) bulgulari daha katı ve itiraza kapali bir deney protokolune donusturmek icin acildi.

Ana hedef:

- Hipotezleri onceden sabitlemek
- Egitim/degerlendirme adaletini (fairness) sabitlemek
- Karar kurallarini acik ve tekrarlanabilir yapmak
- OffDiagonal temelli iddiayi 4 zorunlu kanit ile test etmek

Alt klasorler:

- `protocol/`: hipotezler, karar esikleri, fairness kurallari
- `configs/`: run matrisi ve sabit konfig dosyalari
- `core/`: policy-aware model bilesenleri
- `training/`: egitim scriptleri
- `evaluation/`: metrik ve analiz scriptleri
- `results/`: tum ciktilar
- `docs/`: tartisma notlari ve karar kayitlari
- `logs/`: calisma loglari
- `checkpoints/`: model agirliklari

Karar kaydi:

- `protocol/decisions.md`: sirayla cevaplanan sorularin resmi karar logu
- `protocol/minimum_evidence_framework.md`: Q61-Q64 icin 4 minimum kanit ve pass/fail cercevesi
- `configs/offdiag_minimum_evidence_v1.yaml`: minimum kanit esikleri (makine okunur)
- `configs/model_grid_v1.yaml`: confirmatory model tarama matrisi (frozen)
  - not: Faz-2 latent tarama matrisi; Faz-1 policy kapisindan sonra kullanilir
- `configs/backbone_policy_grid_v1.yaml`: backbone kucultme policy matrisi (Faz-1, frozen)
- `configs/training_budget_v1.yaml`: iki-asamali egitim butcesi (pilot/final, frozen)
- `protocol/freeze_splits_and_stats_v1.py`: split + train-only normalization freeze uretici
- `evaluation/run_evidence_gates.py`: Q61-Q64 icin cikti semasi olusturan iskelet runner
- `protocol/frozen_splits_v1.json`: train/val/test/OOD dosya listesi ve hash manifesti
- `protocol/normalization_stats_v1.json`: train-only condition normalization parametreleri
- `docs/backbone_shrink_methodology_discussion.md`: backbone kucultme metodolojisi (fazli tasarim)
- `protocol/backbone_phase_workflow_v1.md`: Q65-Q72 operasyonel faz akisi
- `training/train_rigid_single.py`: frozen split/stats ile tek-run trainer
- `training/launch_rigid_grid_v1.py`: pilot/final budget profili destekli grid launcher
- `core/model_policy_geo.py`: Phase-1 icin width/depth/hybrid policy-aware mimari
- `training/train_rigid_policy_single.py`: Phase-1 policy-aware tek-run trainer
- `training/launch_rigid_policy_grid_v1.py`: Phase-1 policy grid launcher (pilot/final)
- `training/README.md`: phase komutlari ve log takip komutlari

Durum:

- Q1-Q64 karar seti `protocol/decisions.md` icinde V1 olarak dolduruldu.
- V1 split/normalization freeze dosyalari olusturuldu (`status=FROZEN`).
- Q65-Q72 backbone metodoloji kararlari eklendi (ortak policy + iki-asamali butce).
