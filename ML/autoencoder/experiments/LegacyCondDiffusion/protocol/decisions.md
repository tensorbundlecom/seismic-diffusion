# LegacyCondDiffusion - Frozen Decisions (v1)

Date: 2026-02-24

## Accepted Decisions

1. Stage-1 backbone policy
- Use current legacy-condition baseline checkpoint only for quick smoke checks.
- Official experiment track will train Stage-1 inside this folder.

2. Latent target for diffusion training
- Primary: `z_target = mu(x, c)`.
- Secondary ablation later: sampled `z ~ q(z|x,c)`.

3. Latent normalization
- Compute per-dimension `z_mean`, `z_std` on train split only.
- Apply same stats to val/test/ood/inference.

4. Conditioning strategy (hybrid)
- Diffusion receives both `w_cond` and raw physical condition vector.
- `w_cond` remains main condition path.
- Decoder remains `w_cond`-conditioned.
- Ablation set: `w_cond-only`, `w_cond+c`, `c-only`.

5. Diffusion denoiser architecture
- Main model: residual MLP denoiser (vector latent).
- Ablation: 1D U-Net denoiser.
- 2D U-Net is out-of-scope for v1 (current latent is 1D vector, not spatial latent grid).

7. Split policy
- Event-wise split policy is accepted.

9. Evaluation policy
- Multi-metric evaluation; no single-metric winner.
- Metrics are interpreted separately and jointly.

10. Artifact policy
- Keep both STFT and waveform outputs for analysis.
- Heavy artifacts remain ignored via `.gitignore`.

## Paper Alignment Note (2410.19343v2)

- Aligned:
  - Two-stage idea (autoencoder/latent model then diffusion generation).
  - Conditional latent diffusion objective.
  - Multi-metric waveform evaluation.
- Different by design:
  - This experiment uses legacy condition-embedding (`w_cond`) + station embedding logic.
  - Diffusion denoiser is vector-latent-focused (MLP/1D), not image-latent 2D U-Net.
  - Event-wise split rigor is stricter than random split style.
