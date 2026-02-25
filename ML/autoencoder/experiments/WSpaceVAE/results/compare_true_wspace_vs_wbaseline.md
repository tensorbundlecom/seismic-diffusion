# True W-Space VAE vs LegacyCondBaseline

| Metric | TrueWSpaceVAE | LegacyCondBaseline |
| :--- | ---: | ---: |
| ssim | 0.3401 | 0.6237 |
| lsd | 2.0147 | 1.8123 |
| sc | 0.5311 | 0.1822 |
| s_corr | 0.7421 | 0.9731 |
| sta_lta_err | 0.3631 | NA |
| mr_lsd | 1.4524 | NA |
| arias_err | 0.7835 | 0.3609 |
| env_corr | 0.1909 | 0.5951 |
| dtw | 13447.6511 | 11764.3249 |
| xcorr | 0.1722 | 0.2118 |

Notes:
- LegacyCondBaseline uses deterministic condition embedding (legacy `w_cond`).
- TrueWSpaceVAE uses stochastic latent `u` mapped to `w = M(u)`.
