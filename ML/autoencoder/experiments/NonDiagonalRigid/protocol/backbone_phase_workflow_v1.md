# Backbone-Then-Latent Workflow (V1)

Bu dokuman Q65-Q72 kararlarinin operasyonel akisidir.

## Faz-1: Backbone Policy Secimi

Girdi:
- `configs/backbone_policy_grid_v1.yaml`
- `configs/training_budget_v1.yaml` (`pilot`, `final`)
- `training/train_rigid_policy_single.py`
- `training/launch_rigid_policy_grid_v1.py`

Adimlar:
1. Pilot asamasi:
   - policy x scale adaylari icin kisa run
   - amac: acikca zayif adaylari elemek
2. Final asamasi:
   - shortlist adaylari tam butce ile kos
3. Ortak policy sec:
   - iki aile icin tek policy+scale
   - secim kurali: `Q` + guardrail + `ODI`

Uygulama notu:
- Bu faz icin policy-aware backbone kuruculari (width/depth/hybrid + scale) gerekir.
- `training/train_rigid_single.py` su an latent/grid fazi icin hazirdir; policy-aware trainer ayri eklenecektir.

## Faz-2: Latent Tarama

Girdi:
- Faz-1'de secilen ortak policy+scale
- `configs/model_grid_v1.yaml` (latent listesi)
- `configs/training_budget_v1.yaml` (`pilot`, `final`)
- `training/train_rigid_single.py`
- `training/launch_rigid_grid_v1.py`

Adimlar:
1. Pilot latent tarama
2. Final latent tarama
3. Stop-rule (`eps`, `delta`) ile en kucuk kabul edilebilir modeli sec

## Faz-3: Evidence Gates

- Q61-Q64 hesaplari
- PASS/FAIL raporu
- quality-only vs joint policy karsilastirmasi
