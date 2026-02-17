# WAblation Experiment

## Goal

Isolate the effect of `W` conditioning and station embedding on reconstruction quality.

This experiment tests whether gains come from:
- mapping network (`W-space`)
- station identity
- or their interaction

## Why This Matters

Current all-model OOD results show mixed behavior:
- W-based variants improve some spectral metrics (`LSD`, `MR-LSD`) in parts of the matrix.
- But they do not consistently improve alignment metrics (`DTW`, `XCorr`).

Without ablation, we cannot tell if improvements come from `W`, station embedding, or architecture side effects.

## Hypotheses

- `H1`: `W` mapping improves spectral fidelity (`LSD`, `MR-LSD`) versus raw conditioning.
- `H2`: station embedding improves envelope correlation (`Env Corr`) and spectral correlation (`S-Corr`).
- `H3`: `W + station` gives best spectral metrics but may hurt timing alignment (`DTW`).
- `H4`: raw physical-only conditioning is a stronger baseline for timing metrics.
- `H5`: if `H1` and `H2` both hold, interaction term (`W + station`) should beat each one alone on at least 2 core metrics.

## Controlled Setup

Keep these fixed across variants:
- dataset: external HH training set
- OOD evaluation: post-training custom OOD (same 52 HH waveforms)
- train/val split policy
- optimizer, beta, epochs, batch size
- seed and preprocessing

Only vary:
- `use_mapping_network` (`False/True`)
- `use_station_embedding` (`False/True`)

## Variant Matrix (Phase-1)

- `A0_phys_raw_no_station`: mapping off, station off
- `A1_phys_raw_station`: mapping off, station on
- `A2_phys_w_no_station`: mapping on, station off
- `A3_phys_w_station`: mapping on, station on

## Success Criteria

- If `A2 > A0` on `LSD/MR-LSD`, then `W` helps spectral fidelity.
- If `A1 > A0` on `Env Corr/S-Corr`, then station embedding helps contextual conditioning.
- If `A3` dominates both `A1` and `A2`, interaction is beneficial.
- If no consistent gain, keep simpler condition path in production.

## File Layout

- `core/model_w_ablation.py`: configurable CVAE model (`W` on/off, station on/off)
- `core/loss_utils.py`: beta-CVAE loss
- `training/train_w_ablation_external.py`: single-variant trainer
- `training/run_ablation_matrix.py`: matrix runner/command generator
- `checkpoints/`: variant checkpoints
- `logs/`: run logs
- `results/`: configs + histories + final comparison tables

## Current Status

- [x] Experimental design and hypotheses documented
- [x] Configurable ablation model scaffolded
- [x] Training entrypoints prepared
- [x] Pilot matrix started (`epochs=20`, `max_items=30000`, sequential A0->A3)
- [x] Pilot matrix runs completed
- [x] Pilot summary logged in README
- [ ] OOD comparison table generated

## Pilot Findings (2026-02-15)

Pilot setup:
- `epochs=20`
- `max_items=30000`
- external HH dataset

Best validation loss per variant:

| Variant | Best Val Loss |
| :--- | ---: |
| `A0_phys_raw_no_station` | `11869.45` |
| `A1_phys_raw_station` | `11767.91` |
| `A2_phys_w_no_station` | `11836.71` |
| `A3_phys_w_station` | `11722.84` |

Relative to `A0`:
- `A1`: `-0.86%` (station embedding gain)
- `A2`: `-0.28%` (W mapping gain)
- `A3`: `-1.24%` (best; combined gain)

Interpretation:
- Station embedding has a clear positive effect in this pilot.
- W mapping alone gives smaller but positive gain.
- The combination (`W + station`) gives the best result.
- For current direction, keeping both in conditioning path is justified.
