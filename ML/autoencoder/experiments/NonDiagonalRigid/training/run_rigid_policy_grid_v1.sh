#!/usr/bin/env bash
set -euo pipefail

# generated_at=2026-02-17T15:47:28.472647
# total_jobs=54 phase=pilot latent_dim=128
# budget: max_steps=12000 val_check=2000 patience=9999

ROOT_LOG_DIR="ML/autoencoder/experiments/NonDiagonalRigid/logs"
mkdir -p "$ROOT_LOG_DIR"

echo "[1/54] running rigid_policy_pilot_baseline_diag_width_only_sc1p0_ld128_s42"
/home/gms/miniconda3/bin/python ML/autoencoder/experiments/NonDiagonalRigid/training/train_rigid_policy_single.py --model_family baseline_diag --policy width_only --scale 1.0 --latent_dim 128 --seed 42 --base_channels 32,64,128,256 --batch_size 32 --num_workers 8 --max_steps 12000 --val_check_every_steps 2000 --patience_evals 9999 --min_delta 0.0 --min_steps_before_early_stop 12001 --run_name rigid_policy_pilot_baseline_diag_width_only_sc1p0_ld128_s42 >> $ROOT_LOG_DIR/rigid_policy_pilot_baseline_diag_width_only_sc1p0_ld128_s42.launch.log 2>&1

echo "[2/54] running rigid_policy_pilot_baseline_diag_width_only_sc1p0_ld128_s43"
/home/gms/miniconda3/bin/python ML/autoencoder/experiments/NonDiagonalRigid/training/train_rigid_policy_single.py --model_family baseline_diag --policy width_only --scale 1.0 --latent_dim 128 --seed 43 --base_channels 32,64,128,256 --batch_size 32 --num_workers 8 --max_steps 12000 --val_check_every_steps 2000 --patience_evals 9999 --min_delta 0.0 --min_steps_before_early_stop 12001 --run_name rigid_policy_pilot_baseline_diag_width_only_sc1p0_ld128_s43 >> $ROOT_LOG_DIR/rigid_policy_pilot_baseline_diag_width_only_sc1p0_ld128_s43.launch.log 2>&1

echo "[3/54] running rigid_policy_pilot_baseline_diag_width_only_sc1p0_ld128_s44"
/home/gms/miniconda3/bin/python ML/autoencoder/experiments/NonDiagonalRigid/training/train_rigid_policy_single.py --model_family baseline_diag --policy width_only --scale 1.0 --latent_dim 128 --seed 44 --base_channels 32,64,128,256 --batch_size 32 --num_workers 8 --max_steps 12000 --val_check_every_steps 2000 --patience_evals 9999 --min_delta 0.0 --min_steps_before_early_stop 12001 --run_name rigid_policy_pilot_baseline_diag_width_only_sc1p0_ld128_s44 >> $ROOT_LOG_DIR/rigid_policy_pilot_baseline_diag_width_only_sc1p0_ld128_s44.launch.log 2>&1

echo "[4/54] running rigid_policy_pilot_baseline_diag_width_only_sc0p75_ld128_s42"
/home/gms/miniconda3/bin/python ML/autoencoder/experiments/NonDiagonalRigid/training/train_rigid_policy_single.py --model_family baseline_diag --policy width_only --scale 0.75 --latent_dim 128 --seed 42 --base_channels 32,64,128,256 --batch_size 32 --num_workers 8 --max_steps 12000 --val_check_every_steps 2000 --patience_evals 9999 --min_delta 0.0 --min_steps_before_early_stop 12001 --run_name rigid_policy_pilot_baseline_diag_width_only_sc0p75_ld128_s42 >> $ROOT_LOG_DIR/rigid_policy_pilot_baseline_diag_width_only_sc0p75_ld128_s42.launch.log 2>&1

echo "[5/54] running rigid_policy_pilot_baseline_diag_width_only_sc0p75_ld128_s43"
/home/gms/miniconda3/bin/python ML/autoencoder/experiments/NonDiagonalRigid/training/train_rigid_policy_single.py --model_family baseline_diag --policy width_only --scale 0.75 --latent_dim 128 --seed 43 --base_channels 32,64,128,256 --batch_size 32 --num_workers 8 --max_steps 12000 --val_check_every_steps 2000 --patience_evals 9999 --min_delta 0.0 --min_steps_before_early_stop 12001 --run_name rigid_policy_pilot_baseline_diag_width_only_sc0p75_ld128_s43 >> $ROOT_LOG_DIR/rigid_policy_pilot_baseline_diag_width_only_sc0p75_ld128_s43.launch.log 2>&1

echo "[6/54] running rigid_policy_pilot_baseline_diag_width_only_sc0p75_ld128_s44"
/home/gms/miniconda3/bin/python ML/autoencoder/experiments/NonDiagonalRigid/training/train_rigid_policy_single.py --model_family baseline_diag --policy width_only --scale 0.75 --latent_dim 128 --seed 44 --base_channels 32,64,128,256 --batch_size 32 --num_workers 8 --max_steps 12000 --val_check_every_steps 2000 --patience_evals 9999 --min_delta 0.0 --min_steps_before_early_stop 12001 --run_name rigid_policy_pilot_baseline_diag_width_only_sc0p75_ld128_s44 >> $ROOT_LOG_DIR/rigid_policy_pilot_baseline_diag_width_only_sc0p75_ld128_s44.launch.log 2>&1

echo "[7/54] running rigid_policy_pilot_baseline_diag_width_only_sc0p5_ld128_s42"
/home/gms/miniconda3/bin/python ML/autoencoder/experiments/NonDiagonalRigid/training/train_rigid_policy_single.py --model_family baseline_diag --policy width_only --scale 0.5 --latent_dim 128 --seed 42 --base_channels 32,64,128,256 --batch_size 32 --num_workers 8 --max_steps 12000 --val_check_every_steps 2000 --patience_evals 9999 --min_delta 0.0 --min_steps_before_early_stop 12001 --run_name rigid_policy_pilot_baseline_diag_width_only_sc0p5_ld128_s42 >> $ROOT_LOG_DIR/rigid_policy_pilot_baseline_diag_width_only_sc0p5_ld128_s42.launch.log 2>&1

echo "[8/54] running rigid_policy_pilot_baseline_diag_width_only_sc0p5_ld128_s43"
/home/gms/miniconda3/bin/python ML/autoencoder/experiments/NonDiagonalRigid/training/train_rigid_policy_single.py --model_family baseline_diag --policy width_only --scale 0.5 --latent_dim 128 --seed 43 --base_channels 32,64,128,256 --batch_size 32 --num_workers 8 --max_steps 12000 --val_check_every_steps 2000 --patience_evals 9999 --min_delta 0.0 --min_steps_before_early_stop 12001 --run_name rigid_policy_pilot_baseline_diag_width_only_sc0p5_ld128_s43 >> $ROOT_LOG_DIR/rigid_policy_pilot_baseline_diag_width_only_sc0p5_ld128_s43.launch.log 2>&1

echo "[9/54] running rigid_policy_pilot_baseline_diag_width_only_sc0p5_ld128_s44"
/home/gms/miniconda3/bin/python ML/autoencoder/experiments/NonDiagonalRigid/training/train_rigid_policy_single.py --model_family baseline_diag --policy width_only --scale 0.5 --latent_dim 128 --seed 44 --base_channels 32,64,128,256 --batch_size 32 --num_workers 8 --max_steps 12000 --val_check_every_steps 2000 --patience_evals 9999 --min_delta 0.0 --min_steps_before_early_stop 12001 --run_name rigid_policy_pilot_baseline_diag_width_only_sc0p5_ld128_s44 >> $ROOT_LOG_DIR/rigid_policy_pilot_baseline_diag_width_only_sc0p5_ld128_s44.launch.log 2>&1

echo "[10/54] running rigid_policy_pilot_baseline_diag_depth_only_sc1p0_ld128_s42"
/home/gms/miniconda3/bin/python ML/autoencoder/experiments/NonDiagonalRigid/training/train_rigid_policy_single.py --model_family baseline_diag --policy depth_only --scale 1.0 --latent_dim 128 --seed 42 --base_channels 32,64,128,256 --batch_size 32 --num_workers 8 --max_steps 12000 --val_check_every_steps 2000 --patience_evals 9999 --min_delta 0.0 --min_steps_before_early_stop 12001 --run_name rigid_policy_pilot_baseline_diag_depth_only_sc1p0_ld128_s42 >> $ROOT_LOG_DIR/rigid_policy_pilot_baseline_diag_depth_only_sc1p0_ld128_s42.launch.log 2>&1

echo "[11/54] running rigid_policy_pilot_baseline_diag_depth_only_sc1p0_ld128_s43"
/home/gms/miniconda3/bin/python ML/autoencoder/experiments/NonDiagonalRigid/training/train_rigid_policy_single.py --model_family baseline_diag --policy depth_only --scale 1.0 --latent_dim 128 --seed 43 --base_channels 32,64,128,256 --batch_size 32 --num_workers 8 --max_steps 12000 --val_check_every_steps 2000 --patience_evals 9999 --min_delta 0.0 --min_steps_before_early_stop 12001 --run_name rigid_policy_pilot_baseline_diag_depth_only_sc1p0_ld128_s43 >> $ROOT_LOG_DIR/rigid_policy_pilot_baseline_diag_depth_only_sc1p0_ld128_s43.launch.log 2>&1

echo "[12/54] running rigid_policy_pilot_baseline_diag_depth_only_sc1p0_ld128_s44"
/home/gms/miniconda3/bin/python ML/autoencoder/experiments/NonDiagonalRigid/training/train_rigid_policy_single.py --model_family baseline_diag --policy depth_only --scale 1.0 --latent_dim 128 --seed 44 --base_channels 32,64,128,256 --batch_size 32 --num_workers 8 --max_steps 12000 --val_check_every_steps 2000 --patience_evals 9999 --min_delta 0.0 --min_steps_before_early_stop 12001 --run_name rigid_policy_pilot_baseline_diag_depth_only_sc1p0_ld128_s44 >> $ROOT_LOG_DIR/rigid_policy_pilot_baseline_diag_depth_only_sc1p0_ld128_s44.launch.log 2>&1

echo "[13/54] running rigid_policy_pilot_baseline_diag_depth_only_sc0p75_ld128_s42"
/home/gms/miniconda3/bin/python ML/autoencoder/experiments/NonDiagonalRigid/training/train_rigid_policy_single.py --model_family baseline_diag --policy depth_only --scale 0.75 --latent_dim 128 --seed 42 --base_channels 32,64,128,256 --batch_size 32 --num_workers 8 --max_steps 12000 --val_check_every_steps 2000 --patience_evals 9999 --min_delta 0.0 --min_steps_before_early_stop 12001 --run_name rigid_policy_pilot_baseline_diag_depth_only_sc0p75_ld128_s42 >> $ROOT_LOG_DIR/rigid_policy_pilot_baseline_diag_depth_only_sc0p75_ld128_s42.launch.log 2>&1

echo "[14/54] running rigid_policy_pilot_baseline_diag_depth_only_sc0p75_ld128_s43"
/home/gms/miniconda3/bin/python ML/autoencoder/experiments/NonDiagonalRigid/training/train_rigid_policy_single.py --model_family baseline_diag --policy depth_only --scale 0.75 --latent_dim 128 --seed 43 --base_channels 32,64,128,256 --batch_size 32 --num_workers 8 --max_steps 12000 --val_check_every_steps 2000 --patience_evals 9999 --min_delta 0.0 --min_steps_before_early_stop 12001 --run_name rigid_policy_pilot_baseline_diag_depth_only_sc0p75_ld128_s43 >> $ROOT_LOG_DIR/rigid_policy_pilot_baseline_diag_depth_only_sc0p75_ld128_s43.launch.log 2>&1

echo "[15/54] running rigid_policy_pilot_baseline_diag_depth_only_sc0p75_ld128_s44"
/home/gms/miniconda3/bin/python ML/autoencoder/experiments/NonDiagonalRigid/training/train_rigid_policy_single.py --model_family baseline_diag --policy depth_only --scale 0.75 --latent_dim 128 --seed 44 --base_channels 32,64,128,256 --batch_size 32 --num_workers 8 --max_steps 12000 --val_check_every_steps 2000 --patience_evals 9999 --min_delta 0.0 --min_steps_before_early_stop 12001 --run_name rigid_policy_pilot_baseline_diag_depth_only_sc0p75_ld128_s44 >> $ROOT_LOG_DIR/rigid_policy_pilot_baseline_diag_depth_only_sc0p75_ld128_s44.launch.log 2>&1

echo "[16/54] running rigid_policy_pilot_baseline_diag_depth_only_sc0p5_ld128_s42"
/home/gms/miniconda3/bin/python ML/autoencoder/experiments/NonDiagonalRigid/training/train_rigid_policy_single.py --model_family baseline_diag --policy depth_only --scale 0.5 --latent_dim 128 --seed 42 --base_channels 32,64,128,256 --batch_size 32 --num_workers 8 --max_steps 12000 --val_check_every_steps 2000 --patience_evals 9999 --min_delta 0.0 --min_steps_before_early_stop 12001 --run_name rigid_policy_pilot_baseline_diag_depth_only_sc0p5_ld128_s42 >> $ROOT_LOG_DIR/rigid_policy_pilot_baseline_diag_depth_only_sc0p5_ld128_s42.launch.log 2>&1

echo "[17/54] running rigid_policy_pilot_baseline_diag_depth_only_sc0p5_ld128_s43"
/home/gms/miniconda3/bin/python ML/autoencoder/experiments/NonDiagonalRigid/training/train_rigid_policy_single.py --model_family baseline_diag --policy depth_only --scale 0.5 --latent_dim 128 --seed 43 --base_channels 32,64,128,256 --batch_size 32 --num_workers 8 --max_steps 12000 --val_check_every_steps 2000 --patience_evals 9999 --min_delta 0.0 --min_steps_before_early_stop 12001 --run_name rigid_policy_pilot_baseline_diag_depth_only_sc0p5_ld128_s43 >> $ROOT_LOG_DIR/rigid_policy_pilot_baseline_diag_depth_only_sc0p5_ld128_s43.launch.log 2>&1

echo "[18/54] running rigid_policy_pilot_baseline_diag_depth_only_sc0p5_ld128_s44"
/home/gms/miniconda3/bin/python ML/autoencoder/experiments/NonDiagonalRigid/training/train_rigid_policy_single.py --model_family baseline_diag --policy depth_only --scale 0.5 --latent_dim 128 --seed 44 --base_channels 32,64,128,256 --batch_size 32 --num_workers 8 --max_steps 12000 --val_check_every_steps 2000 --patience_evals 9999 --min_delta 0.0 --min_steps_before_early_stop 12001 --run_name rigid_policy_pilot_baseline_diag_depth_only_sc0p5_ld128_s44 >> $ROOT_LOG_DIR/rigid_policy_pilot_baseline_diag_depth_only_sc0p5_ld128_s44.launch.log 2>&1

echo "[19/54] running rigid_policy_pilot_baseline_diag_hybrid_sc1p0_ld128_s42"
/home/gms/miniconda3/bin/python ML/autoencoder/experiments/NonDiagonalRigid/training/train_rigid_policy_single.py --model_family baseline_diag --policy hybrid --scale 1.0 --latent_dim 128 --seed 42 --base_channels 32,64,128,256 --batch_size 32 --num_workers 8 --max_steps 12000 --val_check_every_steps 2000 --patience_evals 9999 --min_delta 0.0 --min_steps_before_early_stop 12001 --run_name rigid_policy_pilot_baseline_diag_hybrid_sc1p0_ld128_s42 >> $ROOT_LOG_DIR/rigid_policy_pilot_baseline_diag_hybrid_sc1p0_ld128_s42.launch.log 2>&1

echo "[20/54] running rigid_policy_pilot_baseline_diag_hybrid_sc1p0_ld128_s43"
/home/gms/miniconda3/bin/python ML/autoencoder/experiments/NonDiagonalRigid/training/train_rigid_policy_single.py --model_family baseline_diag --policy hybrid --scale 1.0 --latent_dim 128 --seed 43 --base_channels 32,64,128,256 --batch_size 32 --num_workers 8 --max_steps 12000 --val_check_every_steps 2000 --patience_evals 9999 --min_delta 0.0 --min_steps_before_early_stop 12001 --run_name rigid_policy_pilot_baseline_diag_hybrid_sc1p0_ld128_s43 >> $ROOT_LOG_DIR/rigid_policy_pilot_baseline_diag_hybrid_sc1p0_ld128_s43.launch.log 2>&1

echo "[21/54] running rigid_policy_pilot_baseline_diag_hybrid_sc1p0_ld128_s44"
/home/gms/miniconda3/bin/python ML/autoencoder/experiments/NonDiagonalRigid/training/train_rigid_policy_single.py --model_family baseline_diag --policy hybrid --scale 1.0 --latent_dim 128 --seed 44 --base_channels 32,64,128,256 --batch_size 32 --num_workers 8 --max_steps 12000 --val_check_every_steps 2000 --patience_evals 9999 --min_delta 0.0 --min_steps_before_early_stop 12001 --run_name rigid_policy_pilot_baseline_diag_hybrid_sc1p0_ld128_s44 >> $ROOT_LOG_DIR/rigid_policy_pilot_baseline_diag_hybrid_sc1p0_ld128_s44.launch.log 2>&1

echo "[22/54] running rigid_policy_pilot_baseline_diag_hybrid_sc0p75_ld128_s42"
/home/gms/miniconda3/bin/python ML/autoencoder/experiments/NonDiagonalRigid/training/train_rigid_policy_single.py --model_family baseline_diag --policy hybrid --scale 0.75 --latent_dim 128 --seed 42 --base_channels 32,64,128,256 --batch_size 32 --num_workers 8 --max_steps 12000 --val_check_every_steps 2000 --patience_evals 9999 --min_delta 0.0 --min_steps_before_early_stop 12001 --run_name rigid_policy_pilot_baseline_diag_hybrid_sc0p75_ld128_s42 >> $ROOT_LOG_DIR/rigid_policy_pilot_baseline_diag_hybrid_sc0p75_ld128_s42.launch.log 2>&1

echo "[23/54] running rigid_policy_pilot_baseline_diag_hybrid_sc0p75_ld128_s43"
/home/gms/miniconda3/bin/python ML/autoencoder/experiments/NonDiagonalRigid/training/train_rigid_policy_single.py --model_family baseline_diag --policy hybrid --scale 0.75 --latent_dim 128 --seed 43 --base_channels 32,64,128,256 --batch_size 32 --num_workers 8 --max_steps 12000 --val_check_every_steps 2000 --patience_evals 9999 --min_delta 0.0 --min_steps_before_early_stop 12001 --run_name rigid_policy_pilot_baseline_diag_hybrid_sc0p75_ld128_s43 >> $ROOT_LOG_DIR/rigid_policy_pilot_baseline_diag_hybrid_sc0p75_ld128_s43.launch.log 2>&1

echo "[24/54] running rigid_policy_pilot_baseline_diag_hybrid_sc0p75_ld128_s44"
/home/gms/miniconda3/bin/python ML/autoencoder/experiments/NonDiagonalRigid/training/train_rigid_policy_single.py --model_family baseline_diag --policy hybrid --scale 0.75 --latent_dim 128 --seed 44 --base_channels 32,64,128,256 --batch_size 32 --num_workers 8 --max_steps 12000 --val_check_every_steps 2000 --patience_evals 9999 --min_delta 0.0 --min_steps_before_early_stop 12001 --run_name rigid_policy_pilot_baseline_diag_hybrid_sc0p75_ld128_s44 >> $ROOT_LOG_DIR/rigid_policy_pilot_baseline_diag_hybrid_sc0p75_ld128_s44.launch.log 2>&1

echo "[25/54] running rigid_policy_pilot_baseline_diag_hybrid_sc0p5_ld128_s42"
/home/gms/miniconda3/bin/python ML/autoencoder/experiments/NonDiagonalRigid/training/train_rigid_policy_single.py --model_family baseline_diag --policy hybrid --scale 0.5 --latent_dim 128 --seed 42 --base_channels 32,64,128,256 --batch_size 32 --num_workers 8 --max_steps 12000 --val_check_every_steps 2000 --patience_evals 9999 --min_delta 0.0 --min_steps_before_early_stop 12001 --run_name rigid_policy_pilot_baseline_diag_hybrid_sc0p5_ld128_s42 >> $ROOT_LOG_DIR/rigid_policy_pilot_baseline_diag_hybrid_sc0p5_ld128_s42.launch.log 2>&1

echo "[26/54] running rigid_policy_pilot_baseline_diag_hybrid_sc0p5_ld128_s43"
/home/gms/miniconda3/bin/python ML/autoencoder/experiments/NonDiagonalRigid/training/train_rigid_policy_single.py --model_family baseline_diag --policy hybrid --scale 0.5 --latent_dim 128 --seed 43 --base_channels 32,64,128,256 --batch_size 32 --num_workers 8 --max_steps 12000 --val_check_every_steps 2000 --patience_evals 9999 --min_delta 0.0 --min_steps_before_early_stop 12001 --run_name rigid_policy_pilot_baseline_diag_hybrid_sc0p5_ld128_s43 >> $ROOT_LOG_DIR/rigid_policy_pilot_baseline_diag_hybrid_sc0p5_ld128_s43.launch.log 2>&1

echo "[27/54] running rigid_policy_pilot_baseline_diag_hybrid_sc0p5_ld128_s44"
/home/gms/miniconda3/bin/python ML/autoencoder/experiments/NonDiagonalRigid/training/train_rigid_policy_single.py --model_family baseline_diag --policy hybrid --scale 0.5 --latent_dim 128 --seed 44 --base_channels 32,64,128,256 --batch_size 32 --num_workers 8 --max_steps 12000 --val_check_every_steps 2000 --patience_evals 9999 --min_delta 0.0 --min_steps_before_early_stop 12001 --run_name rigid_policy_pilot_baseline_diag_hybrid_sc0p5_ld128_s44 >> $ROOT_LOG_DIR/rigid_policy_pilot_baseline_diag_hybrid_sc0p5_ld128_s44.launch.log 2>&1

echo "[28/54] running rigid_policy_pilot_fullcov_width_only_sc1p0_ld128_s42"
/home/gms/miniconda3/bin/python ML/autoencoder/experiments/NonDiagonalRigid/training/train_rigid_policy_single.py --model_family fullcov --policy width_only --scale 1.0 --latent_dim 128 --seed 42 --base_channels 32,64,128,256 --batch_size 32 --num_workers 8 --max_steps 12000 --val_check_every_steps 2000 --patience_evals 9999 --min_delta 0.0 --min_steps_before_early_stop 12001 --run_name rigid_policy_pilot_fullcov_width_only_sc1p0_ld128_s42 >> $ROOT_LOG_DIR/rigid_policy_pilot_fullcov_width_only_sc1p0_ld128_s42.launch.log 2>&1

echo "[29/54] running rigid_policy_pilot_fullcov_width_only_sc1p0_ld128_s43"
/home/gms/miniconda3/bin/python ML/autoencoder/experiments/NonDiagonalRigid/training/train_rigid_policy_single.py --model_family fullcov --policy width_only --scale 1.0 --latent_dim 128 --seed 43 --base_channels 32,64,128,256 --batch_size 32 --num_workers 8 --max_steps 12000 --val_check_every_steps 2000 --patience_evals 9999 --min_delta 0.0 --min_steps_before_early_stop 12001 --run_name rigid_policy_pilot_fullcov_width_only_sc1p0_ld128_s43 >> $ROOT_LOG_DIR/rigid_policy_pilot_fullcov_width_only_sc1p0_ld128_s43.launch.log 2>&1

echo "[30/54] running rigid_policy_pilot_fullcov_width_only_sc1p0_ld128_s44"
/home/gms/miniconda3/bin/python ML/autoencoder/experiments/NonDiagonalRigid/training/train_rigid_policy_single.py --model_family fullcov --policy width_only --scale 1.0 --latent_dim 128 --seed 44 --base_channels 32,64,128,256 --batch_size 32 --num_workers 8 --max_steps 12000 --val_check_every_steps 2000 --patience_evals 9999 --min_delta 0.0 --min_steps_before_early_stop 12001 --run_name rigid_policy_pilot_fullcov_width_only_sc1p0_ld128_s44 >> $ROOT_LOG_DIR/rigid_policy_pilot_fullcov_width_only_sc1p0_ld128_s44.launch.log 2>&1

echo "[31/54] running rigid_policy_pilot_fullcov_width_only_sc0p75_ld128_s42"
/home/gms/miniconda3/bin/python ML/autoencoder/experiments/NonDiagonalRigid/training/train_rigid_policy_single.py --model_family fullcov --policy width_only --scale 0.75 --latent_dim 128 --seed 42 --base_channels 32,64,128,256 --batch_size 32 --num_workers 8 --max_steps 12000 --val_check_every_steps 2000 --patience_evals 9999 --min_delta 0.0 --min_steps_before_early_stop 12001 --run_name rigid_policy_pilot_fullcov_width_only_sc0p75_ld128_s42 >> $ROOT_LOG_DIR/rigid_policy_pilot_fullcov_width_only_sc0p75_ld128_s42.launch.log 2>&1

echo "[32/54] running rigid_policy_pilot_fullcov_width_only_sc0p75_ld128_s43"
/home/gms/miniconda3/bin/python ML/autoencoder/experiments/NonDiagonalRigid/training/train_rigid_policy_single.py --model_family fullcov --policy width_only --scale 0.75 --latent_dim 128 --seed 43 --base_channels 32,64,128,256 --batch_size 32 --num_workers 8 --max_steps 12000 --val_check_every_steps 2000 --patience_evals 9999 --min_delta 0.0 --min_steps_before_early_stop 12001 --run_name rigid_policy_pilot_fullcov_width_only_sc0p75_ld128_s43 >> $ROOT_LOG_DIR/rigid_policy_pilot_fullcov_width_only_sc0p75_ld128_s43.launch.log 2>&1

echo "[33/54] running rigid_policy_pilot_fullcov_width_only_sc0p75_ld128_s44"
/home/gms/miniconda3/bin/python ML/autoencoder/experiments/NonDiagonalRigid/training/train_rigid_policy_single.py --model_family fullcov --policy width_only --scale 0.75 --latent_dim 128 --seed 44 --base_channels 32,64,128,256 --batch_size 32 --num_workers 8 --max_steps 12000 --val_check_every_steps 2000 --patience_evals 9999 --min_delta 0.0 --min_steps_before_early_stop 12001 --run_name rigid_policy_pilot_fullcov_width_only_sc0p75_ld128_s44 >> $ROOT_LOG_DIR/rigid_policy_pilot_fullcov_width_only_sc0p75_ld128_s44.launch.log 2>&1

echo "[34/54] running rigid_policy_pilot_fullcov_width_only_sc0p5_ld128_s42"
/home/gms/miniconda3/bin/python ML/autoencoder/experiments/NonDiagonalRigid/training/train_rigid_policy_single.py --model_family fullcov --policy width_only --scale 0.5 --latent_dim 128 --seed 42 --base_channels 32,64,128,256 --batch_size 32 --num_workers 8 --max_steps 12000 --val_check_every_steps 2000 --patience_evals 9999 --min_delta 0.0 --min_steps_before_early_stop 12001 --run_name rigid_policy_pilot_fullcov_width_only_sc0p5_ld128_s42 >> $ROOT_LOG_DIR/rigid_policy_pilot_fullcov_width_only_sc0p5_ld128_s42.launch.log 2>&1

echo "[35/54] running rigid_policy_pilot_fullcov_width_only_sc0p5_ld128_s43"
/home/gms/miniconda3/bin/python ML/autoencoder/experiments/NonDiagonalRigid/training/train_rigid_policy_single.py --model_family fullcov --policy width_only --scale 0.5 --latent_dim 128 --seed 43 --base_channels 32,64,128,256 --batch_size 32 --num_workers 8 --max_steps 12000 --val_check_every_steps 2000 --patience_evals 9999 --min_delta 0.0 --min_steps_before_early_stop 12001 --run_name rigid_policy_pilot_fullcov_width_only_sc0p5_ld128_s43 >> $ROOT_LOG_DIR/rigid_policy_pilot_fullcov_width_only_sc0p5_ld128_s43.launch.log 2>&1

echo "[36/54] running rigid_policy_pilot_fullcov_width_only_sc0p5_ld128_s44"
/home/gms/miniconda3/bin/python ML/autoencoder/experiments/NonDiagonalRigid/training/train_rigid_policy_single.py --model_family fullcov --policy width_only --scale 0.5 --latent_dim 128 --seed 44 --base_channels 32,64,128,256 --batch_size 32 --num_workers 8 --max_steps 12000 --val_check_every_steps 2000 --patience_evals 9999 --min_delta 0.0 --min_steps_before_early_stop 12001 --run_name rigid_policy_pilot_fullcov_width_only_sc0p5_ld128_s44 >> $ROOT_LOG_DIR/rigid_policy_pilot_fullcov_width_only_sc0p5_ld128_s44.launch.log 2>&1

echo "[37/54] running rigid_policy_pilot_fullcov_depth_only_sc1p0_ld128_s42"
/home/gms/miniconda3/bin/python ML/autoencoder/experiments/NonDiagonalRigid/training/train_rigid_policy_single.py --model_family fullcov --policy depth_only --scale 1.0 --latent_dim 128 --seed 42 --base_channels 32,64,128,256 --batch_size 32 --num_workers 8 --max_steps 12000 --val_check_every_steps 2000 --patience_evals 9999 --min_delta 0.0 --min_steps_before_early_stop 12001 --run_name rigid_policy_pilot_fullcov_depth_only_sc1p0_ld128_s42 >> $ROOT_LOG_DIR/rigid_policy_pilot_fullcov_depth_only_sc1p0_ld128_s42.launch.log 2>&1

echo "[38/54] running rigid_policy_pilot_fullcov_depth_only_sc1p0_ld128_s43"
/home/gms/miniconda3/bin/python ML/autoencoder/experiments/NonDiagonalRigid/training/train_rigid_policy_single.py --model_family fullcov --policy depth_only --scale 1.0 --latent_dim 128 --seed 43 --base_channels 32,64,128,256 --batch_size 32 --num_workers 8 --max_steps 12000 --val_check_every_steps 2000 --patience_evals 9999 --min_delta 0.0 --min_steps_before_early_stop 12001 --run_name rigid_policy_pilot_fullcov_depth_only_sc1p0_ld128_s43 >> $ROOT_LOG_DIR/rigid_policy_pilot_fullcov_depth_only_sc1p0_ld128_s43.launch.log 2>&1

echo "[39/54] running rigid_policy_pilot_fullcov_depth_only_sc1p0_ld128_s44"
/home/gms/miniconda3/bin/python ML/autoencoder/experiments/NonDiagonalRigid/training/train_rigid_policy_single.py --model_family fullcov --policy depth_only --scale 1.0 --latent_dim 128 --seed 44 --base_channels 32,64,128,256 --batch_size 32 --num_workers 8 --max_steps 12000 --val_check_every_steps 2000 --patience_evals 9999 --min_delta 0.0 --min_steps_before_early_stop 12001 --run_name rigid_policy_pilot_fullcov_depth_only_sc1p0_ld128_s44 >> $ROOT_LOG_DIR/rigid_policy_pilot_fullcov_depth_only_sc1p0_ld128_s44.launch.log 2>&1

echo "[40/54] running rigid_policy_pilot_fullcov_depth_only_sc0p75_ld128_s42"
/home/gms/miniconda3/bin/python ML/autoencoder/experiments/NonDiagonalRigid/training/train_rigid_policy_single.py --model_family fullcov --policy depth_only --scale 0.75 --latent_dim 128 --seed 42 --base_channels 32,64,128,256 --batch_size 32 --num_workers 8 --max_steps 12000 --val_check_every_steps 2000 --patience_evals 9999 --min_delta 0.0 --min_steps_before_early_stop 12001 --run_name rigid_policy_pilot_fullcov_depth_only_sc0p75_ld128_s42 >> $ROOT_LOG_DIR/rigid_policy_pilot_fullcov_depth_only_sc0p75_ld128_s42.launch.log 2>&1

echo "[41/54] running rigid_policy_pilot_fullcov_depth_only_sc0p75_ld128_s43"
/home/gms/miniconda3/bin/python ML/autoencoder/experiments/NonDiagonalRigid/training/train_rigid_policy_single.py --model_family fullcov --policy depth_only --scale 0.75 --latent_dim 128 --seed 43 --base_channels 32,64,128,256 --batch_size 32 --num_workers 8 --max_steps 12000 --val_check_every_steps 2000 --patience_evals 9999 --min_delta 0.0 --min_steps_before_early_stop 12001 --run_name rigid_policy_pilot_fullcov_depth_only_sc0p75_ld128_s43 >> $ROOT_LOG_DIR/rigid_policy_pilot_fullcov_depth_only_sc0p75_ld128_s43.launch.log 2>&1

echo "[42/54] running rigid_policy_pilot_fullcov_depth_only_sc0p75_ld128_s44"
/home/gms/miniconda3/bin/python ML/autoencoder/experiments/NonDiagonalRigid/training/train_rigid_policy_single.py --model_family fullcov --policy depth_only --scale 0.75 --latent_dim 128 --seed 44 --base_channels 32,64,128,256 --batch_size 32 --num_workers 8 --max_steps 12000 --val_check_every_steps 2000 --patience_evals 9999 --min_delta 0.0 --min_steps_before_early_stop 12001 --run_name rigid_policy_pilot_fullcov_depth_only_sc0p75_ld128_s44 >> $ROOT_LOG_DIR/rigid_policy_pilot_fullcov_depth_only_sc0p75_ld128_s44.launch.log 2>&1

echo "[43/54] running rigid_policy_pilot_fullcov_depth_only_sc0p5_ld128_s42"
/home/gms/miniconda3/bin/python ML/autoencoder/experiments/NonDiagonalRigid/training/train_rigid_policy_single.py --model_family fullcov --policy depth_only --scale 0.5 --latent_dim 128 --seed 42 --base_channels 32,64,128,256 --batch_size 32 --num_workers 8 --max_steps 12000 --val_check_every_steps 2000 --patience_evals 9999 --min_delta 0.0 --min_steps_before_early_stop 12001 --run_name rigid_policy_pilot_fullcov_depth_only_sc0p5_ld128_s42 >> $ROOT_LOG_DIR/rigid_policy_pilot_fullcov_depth_only_sc0p5_ld128_s42.launch.log 2>&1

echo "[44/54] running rigid_policy_pilot_fullcov_depth_only_sc0p5_ld128_s43"
/home/gms/miniconda3/bin/python ML/autoencoder/experiments/NonDiagonalRigid/training/train_rigid_policy_single.py --model_family fullcov --policy depth_only --scale 0.5 --latent_dim 128 --seed 43 --base_channels 32,64,128,256 --batch_size 32 --num_workers 8 --max_steps 12000 --val_check_every_steps 2000 --patience_evals 9999 --min_delta 0.0 --min_steps_before_early_stop 12001 --run_name rigid_policy_pilot_fullcov_depth_only_sc0p5_ld128_s43 >> $ROOT_LOG_DIR/rigid_policy_pilot_fullcov_depth_only_sc0p5_ld128_s43.launch.log 2>&1

echo "[45/54] running rigid_policy_pilot_fullcov_depth_only_sc0p5_ld128_s44"
/home/gms/miniconda3/bin/python ML/autoencoder/experiments/NonDiagonalRigid/training/train_rigid_policy_single.py --model_family fullcov --policy depth_only --scale 0.5 --latent_dim 128 --seed 44 --base_channels 32,64,128,256 --batch_size 32 --num_workers 8 --max_steps 12000 --val_check_every_steps 2000 --patience_evals 9999 --min_delta 0.0 --min_steps_before_early_stop 12001 --run_name rigid_policy_pilot_fullcov_depth_only_sc0p5_ld128_s44 >> $ROOT_LOG_DIR/rigid_policy_pilot_fullcov_depth_only_sc0p5_ld128_s44.launch.log 2>&1

echo "[46/54] running rigid_policy_pilot_fullcov_hybrid_sc1p0_ld128_s42"
/home/gms/miniconda3/bin/python ML/autoencoder/experiments/NonDiagonalRigid/training/train_rigid_policy_single.py --model_family fullcov --policy hybrid --scale 1.0 --latent_dim 128 --seed 42 --base_channels 32,64,128,256 --batch_size 32 --num_workers 8 --max_steps 12000 --val_check_every_steps 2000 --patience_evals 9999 --min_delta 0.0 --min_steps_before_early_stop 12001 --run_name rigid_policy_pilot_fullcov_hybrid_sc1p0_ld128_s42 >> $ROOT_LOG_DIR/rigid_policy_pilot_fullcov_hybrid_sc1p0_ld128_s42.launch.log 2>&1

echo "[47/54] running rigid_policy_pilot_fullcov_hybrid_sc1p0_ld128_s43"
/home/gms/miniconda3/bin/python ML/autoencoder/experiments/NonDiagonalRigid/training/train_rigid_policy_single.py --model_family fullcov --policy hybrid --scale 1.0 --latent_dim 128 --seed 43 --base_channels 32,64,128,256 --batch_size 32 --num_workers 8 --max_steps 12000 --val_check_every_steps 2000 --patience_evals 9999 --min_delta 0.0 --min_steps_before_early_stop 12001 --run_name rigid_policy_pilot_fullcov_hybrid_sc1p0_ld128_s43 >> $ROOT_LOG_DIR/rigid_policy_pilot_fullcov_hybrid_sc1p0_ld128_s43.launch.log 2>&1

echo "[48/54] running rigid_policy_pilot_fullcov_hybrid_sc1p0_ld128_s44"
/home/gms/miniconda3/bin/python ML/autoencoder/experiments/NonDiagonalRigid/training/train_rigid_policy_single.py --model_family fullcov --policy hybrid --scale 1.0 --latent_dim 128 --seed 44 --base_channels 32,64,128,256 --batch_size 32 --num_workers 8 --max_steps 12000 --val_check_every_steps 2000 --patience_evals 9999 --min_delta 0.0 --min_steps_before_early_stop 12001 --run_name rigid_policy_pilot_fullcov_hybrid_sc1p0_ld128_s44 >> $ROOT_LOG_DIR/rigid_policy_pilot_fullcov_hybrid_sc1p0_ld128_s44.launch.log 2>&1

echo "[49/54] running rigid_policy_pilot_fullcov_hybrid_sc0p75_ld128_s42"
/home/gms/miniconda3/bin/python ML/autoencoder/experiments/NonDiagonalRigid/training/train_rigid_policy_single.py --model_family fullcov --policy hybrid --scale 0.75 --latent_dim 128 --seed 42 --base_channels 32,64,128,256 --batch_size 32 --num_workers 8 --max_steps 12000 --val_check_every_steps 2000 --patience_evals 9999 --min_delta 0.0 --min_steps_before_early_stop 12001 --run_name rigid_policy_pilot_fullcov_hybrid_sc0p75_ld128_s42 >> $ROOT_LOG_DIR/rigid_policy_pilot_fullcov_hybrid_sc0p75_ld128_s42.launch.log 2>&1

echo "[50/54] running rigid_policy_pilot_fullcov_hybrid_sc0p75_ld128_s43"
/home/gms/miniconda3/bin/python ML/autoencoder/experiments/NonDiagonalRigid/training/train_rigid_policy_single.py --model_family fullcov --policy hybrid --scale 0.75 --latent_dim 128 --seed 43 --base_channels 32,64,128,256 --batch_size 32 --num_workers 8 --max_steps 12000 --val_check_every_steps 2000 --patience_evals 9999 --min_delta 0.0 --min_steps_before_early_stop 12001 --run_name rigid_policy_pilot_fullcov_hybrid_sc0p75_ld128_s43 >> $ROOT_LOG_DIR/rigid_policy_pilot_fullcov_hybrid_sc0p75_ld128_s43.launch.log 2>&1

echo "[51/54] running rigid_policy_pilot_fullcov_hybrid_sc0p75_ld128_s44"
/home/gms/miniconda3/bin/python ML/autoencoder/experiments/NonDiagonalRigid/training/train_rigid_policy_single.py --model_family fullcov --policy hybrid --scale 0.75 --latent_dim 128 --seed 44 --base_channels 32,64,128,256 --batch_size 32 --num_workers 8 --max_steps 12000 --val_check_every_steps 2000 --patience_evals 9999 --min_delta 0.0 --min_steps_before_early_stop 12001 --run_name rigid_policy_pilot_fullcov_hybrid_sc0p75_ld128_s44 >> $ROOT_LOG_DIR/rigid_policy_pilot_fullcov_hybrid_sc0p75_ld128_s44.launch.log 2>&1

echo "[52/54] running rigid_policy_pilot_fullcov_hybrid_sc0p5_ld128_s42"
/home/gms/miniconda3/bin/python ML/autoencoder/experiments/NonDiagonalRigid/training/train_rigid_policy_single.py --model_family fullcov --policy hybrid --scale 0.5 --latent_dim 128 --seed 42 --base_channels 32,64,128,256 --batch_size 32 --num_workers 8 --max_steps 12000 --val_check_every_steps 2000 --patience_evals 9999 --min_delta 0.0 --min_steps_before_early_stop 12001 --run_name rigid_policy_pilot_fullcov_hybrid_sc0p5_ld128_s42 >> $ROOT_LOG_DIR/rigid_policy_pilot_fullcov_hybrid_sc0p5_ld128_s42.launch.log 2>&1

echo "[53/54] running rigid_policy_pilot_fullcov_hybrid_sc0p5_ld128_s43"
/home/gms/miniconda3/bin/python ML/autoencoder/experiments/NonDiagonalRigid/training/train_rigid_policy_single.py --model_family fullcov --policy hybrid --scale 0.5 --latent_dim 128 --seed 43 --base_channels 32,64,128,256 --batch_size 32 --num_workers 8 --max_steps 12000 --val_check_every_steps 2000 --patience_evals 9999 --min_delta 0.0 --min_steps_before_early_stop 12001 --run_name rigid_policy_pilot_fullcov_hybrid_sc0p5_ld128_s43 >> $ROOT_LOG_DIR/rigid_policy_pilot_fullcov_hybrid_sc0p5_ld128_s43.launch.log 2>&1

echo "[54/54] running rigid_policy_pilot_fullcov_hybrid_sc0p5_ld128_s44"
/home/gms/miniconda3/bin/python ML/autoencoder/experiments/NonDiagonalRigid/training/train_rigid_policy_single.py --model_family fullcov --policy hybrid --scale 0.5 --latent_dim 128 --seed 44 --base_channels 32,64,128,256 --batch_size 32 --num_workers 8 --max_steps 12000 --val_check_every_steps 2000 --patience_evals 9999 --min_delta 0.0 --min_steps_before_early_stop 12001 --run_name rigid_policy_pilot_fullcov_hybrid_sc0p5_ld128_s44 >> $ROOT_LOG_DIR/rigid_policy_pilot_fullcov_hybrid_sc0p5_ld128_s44.launch.log 2>&1

