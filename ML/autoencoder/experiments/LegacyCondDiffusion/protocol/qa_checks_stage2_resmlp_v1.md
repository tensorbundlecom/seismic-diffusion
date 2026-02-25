# QA Checks - Stage2 ResMLP (`diff_resmlp_w_plus_c_v1`)

Date: 2026-02-24

## 1) Run Integrity

- Stage-1 status: completed (100/100).
- Stage-2 diffusion status: completed (100/100).
- Checkpoints exist:
  - `checkpoints/stage1_wbaseline_best.pt`
  - `checkpoints/diff_resmlp_w_plus_c_v1_best.pt`
  - `checkpoints/diff_resmlp_w_plus_c_v1_latest.pt`
- Training history saved:
  - `results/diff_resmlp_w_plus_c_v1_train_history.json`

## 2) Numeric Stability

- No NaN/Inf in train/val loss history.
- Diffusion best validation loss:
  - `0.0129918931` at epoch `92`.

## 3) OOD Eval Smoke + Full

- Smoke (`max_samples=10`):
  - `results/post_training_custom_ood_metrics_diff_resmlp_w_plus_c_v1_smoke10.json`
- Full (`num_samples=52`):
  - `results/post_training_custom_ood_metrics_diff_resmlp_w_plus_c_v1.json`

## 4) Oracle vs Generated Diagnostic

- Full comparison file:
  - `results/oracle_vs_generated_metrics_full.json`
- Observation:
  - Oracle and generated metrics are nearly identical across all reported metrics.

## 5) Latent Usage Diagnostic (Critical)

- Train cache latent magnitude:
  - `mean_abs(mu) ~= 1.64e-4`
  - `mean(||mu||) ~= 2.35e-3`
  - `per-dim std mean ~= 1.36e-4`
- Decoder sensitivity to latent:
  - `mean |decoder(mu)-decoder(rand_z)| ~= 1.5e-3` (very small)
- Generated diversity for fixed condition:
  - Diffusion sample-to-sample pixel std mean `~4.26e-7` (near-deterministic).

Conclusion:
- Stage-1 posterior collapsed to near-zero latent.
- Decoder behavior is condition-dominant and mostly latent-insensitive.
- Current stage-2 result is operationally valid (pipeline runs) but not suitable as a stochastic generative model.

