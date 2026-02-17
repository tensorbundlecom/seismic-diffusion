# Phase-2 Penalty Comparison (2026-02-17)

## Scope

Compare FullCov control runs with increasing off-diagonal penalty strength.

Reference checkpoints:

- `fullcov_geo_repi_external_s42_20260215_best.pt` (FullCov reference, 100 epochs)
- `fullcov_geo_control_diag_only_external_s42_e30_20260216_best.pt` (diag-only control)
- `fullcov_geo_control_offdiag_l{1e-4,5e-4,20,100,400}_external_s42_e30_20260216_best.pt`

## Validation Best Comparison

| Run | Best Epoch | Best Val Loss | Delta vs Ref | Best Val Recon | Best Val KL | Best Pen | Best Pen Scaled |
|:---|---:|---:|---:|---:|---:|---:|---:|
| FullCov ref (100e) | 99 | 176.5579 | +0.0000 | 168.6560 | 7.9018 | 0.000000 | 0.000000 |
| Diag-only ctrl (30e) | 29 | 177.1075 | +0.5496 | 167.6980 | 9.4094 | 0.000000 | 0.000000 |
| Offdiag `1e-4` (30e) | 18 | 176.4756 | -0.0823 | 168.5722 | 7.9034 | 0.001529 | ~0.000000 |
| Offdiag `5e-4` (30e) | 18 | 176.4811 | -0.0768 | 168.5726 | 7.9085 | 0.001529 | ~0.000001 |
| Offdiag `20` (30e) | 18 | 176.5075 | -0.0503 | 168.5733 | 7.9045 | 0.001486 | 0.029714 |
| Offdiag `100` (30e) | 18 | 176.5898 | +0.0319 | 168.5897 | 7.8670 | 0.001331 | 0.133107 |
| Offdiag `400` (30e) | 18 | 176.8018 | +0.2440 | 168.5611 | 7.8571 | 0.000959 | 0.383581 |

Notes:

- For `1e-4` and `5e-4`, effective penalty contribution is negligible.
- For `20/100/400`, penalty contribution is clearly active in total loss.

## OOD Off-Diagonal Correlation Summary (52 Samples)

| Run | mean_abs_offdiag | p95_abs_offdiag | max_abs_offdiag | offdiag_energy_ratio |
|:---|---:|---:|---:|---:|
| FullCov ref | 0.055524 | 0.144686 | 0.769812 | 0.661713 |
| Diag-only ctrl | 0.069322 | 0.174650 | 0.983904 | 0.722747 |
| Offdiag `1e-4` | 0.054433 | 0.141747 | 0.802712 | 0.656305 |
| Offdiag `5e-4` | 0.054480 | 0.142061 | 0.800176 | 0.656601 |
| Offdiag `20` | 0.054232 | 0.141444 | 0.800198 | 0.655165 |
| Offdiag `100` | 0.053722 | 0.139353 | 0.805640 | 0.651939 |
| Offdiag `400` | 0.051875 | 0.133256 | 0.808285 | 0.640485 |

## Interpretation

- `diag-only` is clearly worse in validation and yields the highest raw off-diagonal statistics.
- Mild off-diagonal suppression (`20`) keeps validation near reference while reducing off-diagonal modestly.
- Strong suppression (`100`, `400`) reduces off-diagonal more, but degrades validation fit.
- This indicates a trade-off: off-diagonal can be reduced, but aggressive suppression hurts optimization quality.
- Practical sweet spot in this phase is around `lambda ~ 20` (or weaker), not `100+`.
