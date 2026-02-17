# NonDiagonalRigid Training

Bu klasor, frozen protokole bagli egitim scriptlerini icerir.

## Phase-1 (Backbone Policy)

Tek run:

```bash
python3 ML/autoencoder/experiments/NonDiagonalRigid/training/train_rigid_policy_single.py \
  --model_family baseline_diag \
  --policy width_only \
  --scale 0.75 \
  --latent_dim 128 \
  --seed 42
```

Grid komutlari (pilot):

```bash
python3 ML/autoencoder/experiments/NonDiagonalRigid/training/launch_rigid_policy_grid_v1.py \
  --mode script \
  --phase pilot \
  --python-bin ~/miniconda3/envs/psp_env/bin/python
```

Detayli/kalici calisma (detached master):

```bash
python3 ML/autoencoder/experiments/NonDiagonalRigid/training/launch_rigid_policy_grid_v1.py \
  --mode launch \
  --phase pilot \
  --python-bin ~/miniconda3/bin/python
```

Grid komutlari (final):

```bash
python3 ML/autoencoder/experiments/NonDiagonalRigid/training/launch_rigid_policy_grid_v1.py \
  --mode script \
  --phase final \
  --python-bin ~/miniconda3/envs/psp_env/bin/python
```

## Phase-2 (Latent Sweep)

Tek run:

```bash
python3 ML/autoencoder/experiments/NonDiagonalRigid/training/train_rigid_single.py \
  --model_family baseline_diag \
  --backbone large \
  --latent_dim 128 \
  --seed 42
```

Grid:

```bash
python3 ML/autoencoder/experiments/NonDiagonalRigid/training/launch_rigid_grid_v1.py \
  --mode script \
  --phase final \
  --python-bin ~/miniconda3/envs/psp_env/bin/python
```

Detayli/kalici calisma (detached master):

```bash
python3 ML/autoencoder/experiments/NonDiagonalRigid/training/launch_rigid_grid_v1.py \
  --mode launch \
  --phase final \
  --python-bin ~/miniconda3/bin/python
```

## Log Takibi

```bash
ls -1t ML/autoencoder/experiments/NonDiagonalRigid/logs/*.log | head
tail -f ML/autoencoder/experiments/NonDiagonalRigid/logs/<run_name>.log
tail -f ML/autoencoder/experiments/NonDiagonalRigid/logs/run_rigid_policy_grid_v1.master.log
```

Policy fazi durum ozeti:

```bash
python3 ML/autoencoder/experiments/NonDiagonalRigid/evaluation/monitor_phase1_policy_runs.py \
  --run-prefix rigid_policy_pilot_
```
