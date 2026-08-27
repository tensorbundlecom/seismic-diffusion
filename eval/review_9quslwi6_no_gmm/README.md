# Evaluation review: W&B run `9quslwi6`

Model: `sid-v1-bounded-sweep-lr=0.00018602056009368563-base_channels=64-layers_per_block=1-weight_decay=0.007670944346035483`

Checkpoint: `ML/diffusion/checkpoints/ddpm/sid-v1-bounded-sweep/wandb_runs/9quslwi6/best_val`

Evaluation settings:

- deterministic 10% validation sample (`seed=0`) where dataset sampling applies;
- 1,000 diffusion steps and 200 Griffin–Lim iterations;
- corrected SciPy-spectrum-to-librosa inverse-STFT amplitude contract;
- no GMM curves, populations, panels, predictions, or probability ratios;
- outputs were not logged to W&B.

## Figures

- [First-order characteristics](first_order/first_order_9quslwi6_best_val_steps1000_chE_modelc8ba8394b23f_acc_n998_sel3b09952107.png)
- [Peak-amplitude bias](peak_amplitudes/fig_9quslwi6_best_val_steps1000_chE_modelc8ba8394b23f_acc_n998_sel3b09952107_none_val.png)
- [Peak-amplitude residual distributions](residual_distributions/fig_residuals_9quslwi6_best_val_steps1000_chE_modelc8ba8394b23f_acc_n998_sel3b09952107_none_e.png)
- [Realization distribution](distributions/fig_9quslwi6_best_val_steps1000_chE_modelc8ba8394b23f_acc_n998_sel3b09952107_idx87911_modelc8ba8394b23f_n100_pga.png)
- [Shake duration](shake_duration/fig_duration_9quslwi6_best_val_steps1000_chE_modelc8ba8394b23f_acc_n998_sel3b09952107.png)
- [Attenuation](attenuation/fig_attenuation_M2.5_EDC_none.png)
- [Magnitude and Vs30 scaling](scaling/fig_scaling_EDC_T0.3_R60.png)
- [GWM probability](model_probabilities/fig_probabilities_EDC_T0.3_none.png)
