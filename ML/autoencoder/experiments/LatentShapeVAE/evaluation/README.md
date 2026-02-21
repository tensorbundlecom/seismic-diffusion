# Evaluation Guide

## Latent shape analizi

```bash
~/miniconda3/bin/python ML/autoencoder/experiments/LatentShapeVAE/evaluation/analyze_latent_shape.py \
  --checkpoints ML/autoencoder/experiments/LatentShapeVAE/checkpoints/lsv_vae_base_ld64_s42_best.pt \
  --split test
```

Ciktilar:
- `results/latent_shape_<split>_<timestamp>/latent_shape_summary.csv`
- Her run icin:
  - `cov_mu.npy`, `mean_sigma.npy`, `cov_agg.npy`, `corr_agg.npy`
  - `corr_agg_heatmap.png`, `eigvals_cov_agg.png`, `pca_mu.png`

## Prior sampling realism analizi

```bash
~/miniconda3/bin/python ML/autoencoder/experiments/LatentShapeVAE/evaluation/evaluate_prior_sampling.py \
  --checkpoints ML/autoencoder/experiments/LatentShapeVAE/checkpoints/lsv_vae_base_ld64_s42_best.pt \
  --split ood_event \
  --num_real_samples 2000 \
  --num_generated_samples 2000
```

Ozet:
- `results/prior_sampling_<split>_<timestamp>/prior_sampling_realism_summary.csv`
- `realism_composite` ne kadar dusukse o kadar iyi.

## ELBO terms (proxy) analizi

```bash
~/miniconda3/bin/python ML/autoencoder/experiments/LatentShapeVAE/evaluation/analyze_elbo_terms.py \
  --checkpoint ML/autoencoder/experiments/LatentShapeVAE/checkpoints/lsv_vae_base_ld64_s42_best.pt \
  --split test
```

Not:
- `E_log_p_x_given_z_proxy = -recon_total` (time+MR-STFT reconstruction proxy)
- `E_KL_q_to_p = kl_raw`

## Latent logvar/variance outlier audit

```bash
~/miniconda3/bin/python ML/autoencoder/experiments/LatentShapeVAE/evaluation/audit_latent_var_outliers.py \
  --checkpoints \
    ML/autoencoder/experiments/LatentShapeVAE/checkpoints/lsv_stage2_vae_base_ld64_b0p1_s42_best.pt \
    ML/autoencoder/experiments/LatentShapeVAE/checkpoints/lsv_stage2_vae_base_ld64_b0p1_s43_best.pt \
    ML/autoencoder/experiments/LatentShapeVAE/checkpoints/lsv_stage2_vae_base_ld64_b0p1_s44_best.pt \
  --splits test ood_event \
  --var_thresholds 10,1000,100000 \
  --output_dir ML/autoencoder/experiments/LatentShapeVAE/results/latent_var_outlier_audit_stage2_beta0p1_v1
```

Ciktilar:
- `results/latent_var_outlier_audit_*/audit_summary.csv`
- Split/run bazinda:
  - `sample_metrics.csv`
  - `topk_outliers.csv`
  - `event_summary.csv`
  - `station_summary.csv`
  - `summary.json`

## Stage-2 logvar-fix karsilastirma batch'i

Eski (`s42/s43/s44`) ve logvar-fix rerun (`s43_logvfixv1`, `s44_logvfixv1`) checkpointlerini
tek seferde latent-shape + prior-sampling + outlier-audit ile degerlendirir:

```bash
setsid bash -lc 'ML/autoencoder/experiments/LatentShapeVAE/evaluation/run_stage2_beta0p1_logvarfix_compare_v1.sh' \
  > ML/autoencoder/experiments/LatentShapeVAE/logs/stage2_beta0p1_logvarfix_compare_v1.log 2>&1 < /dev/null &
```

## Stage-2 logvar-fix 10-seed robustness eval (v2)

Bu batch su analizleri tek seferde calistirir:
- latent shape (test + ood_event)
- prior sampling realism (test + ood_event)
- latent variance outlier audit
- 10-seed aggregate robustness ozeti

```bash
setsid bash -lc 'ML/autoencoder/experiments/LatentShapeVAE/evaluation/run_stage2_beta0p1_logvarfix_10seeds_v2.sh' \
  > ML/autoencoder/experiments/LatentShapeVAE/logs/stage2_beta0p1_logvarfix_10seeds_v2.eval.log 2>&1 < /dev/null &
```

Ana ozet:
- `ML/autoencoder/experiments/LatentShapeVAE/results/stage2_beta0p1_logvarfix_10seeds_v2/robustness_summary.md`

## N(0,1) benzerlik skoru (tek metrik)

Tum `latent_shape_summary.csv` ciktilarini tarayip run/split bazinda tek skor uretir:

```bash
~/miniconda3/bin/python ML/autoencoder/experiments/LatentShapeVAE/evaluation/compute_n01_similarity.py \
  --output_dir ML/autoencoder/experiments/LatentShapeVAE/results/n01_similarity_v1
```

Ciktilar:
- `results/n01_similarity_v1/n01_similarity_per_run_split.csv`
- `results/n01_similarity_v1/n01_similarity_ranking_test.md`
- `results/n01_similarity_v1/n01_similarity_ranking_ood_event.md`

## Stage-3 latent=32 format eval (v1)

Train tamamlaninca otomatik eval icin:

```bash
setsid bash -lc 'ML/autoencoder/experiments/LatentShapeVAE/evaluation/wait_and_run_stage3_ld32_formats_eval_v1.sh' \
  > ML/autoencoder/experiments/LatentShapeVAE/logs/stage3_ld32_formats_v1.pipeline.log 2>&1 < /dev/null &
```

Elle eval baslatmak icin:

```bash
setsid bash -lc 'ML/autoencoder/experiments/LatentShapeVAE/evaluation/run_stage3_ld32_formats_v1.sh' \
  > ML/autoencoder/experiments/LatentShapeVAE/logs/stage3_ld32_formats_v1.eval.log 2>&1 < /dev/null &
```

Ana format ozeti:
- `ML/autoencoder/experiments/LatentShapeVAE/results/stage3_ld32_formats_v1/stage3_ld32_formats_summary.md`
