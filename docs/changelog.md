# Documentation changelog

**Last updated:** 2026-08-26

## TL;DR

This file records documentation-facing changes to pipeline contracts. It is not a substitute for version control history or model-training records.

## 2026-08-26 — Diffusion sweep parameter contract

- Documented validated `train.py` controls for optimizer (`--lr`, `--weight_decay`), loaders (`--batch_size`), and configurable Diffusers U-Net architecture (`--base_channels`, `--layers_per_block`), including defaults and the base-channel GroupNorm constraint.
- Documented W&B sweep readiness, W&B/checkpoint `training_config` persistence, initial bounded architecture candidates and approximate parameter counts, and the default-on W&B final evaluation behavior.

## 2026-08-25 — Domain-aware AE PhaseNet perceptual contract

- Documented that omitted AE `--amplitude_epsilon` now resolves by waveform domain: `1.0` for `instrument_counts`, preserving legacy `log1p`, and `1e-12` for `physical_acceleration`; an explicit value overrides either default.
- Documented PhaseNet's two exact input paths: count-domain training always retains legacy v1 `expm1(normalized.clamp_min(0))`, including under global normalization, while physical acceleration requires global bounds and uses the exact physical inverse before PhaseNet.
- Documented the persisted `phasenet_input_mode` configuration field and the fail-fast requirement for `--global_normalization` when physical-acceleration PhaseNet loss is enabled.

## 2026-08-25 — Demo AE selector uses checkpoint provenance

- Documented that the latent-mode demo lists only autoencoders explicitly bound to an available diffusion checkpoint by saved provenance. Selecting an AE loads the newest compatible diffusion checkpoint, preferring the active training type when possible.
- Documented that equal latent tensor shapes do not make independently trained AEs and diffusion models interchangeable; the demo prevents unsupported arbitrary pairings.
- Documented checkpoint-bound loading of reference metadata, station data, optional Vs30, reconstruction fallbacks, and waveform paths. Random EQ/reference plots now switch domains with the active pipeline, and a failed artifact or waveform-domain validation rolls back the whole switch.

## 2026-08-24 — AE-scoped embedding exports

- Documented that embedding exports are stored under `embeddings/<AE run name>/`, preserving independently generated artifacts for multiple autoencoders and requiring `--overwrite` only to stage and install a complete replacement of the same AE's core export artifacts.
- Documented `train.py --embeddings_dir` as the explicit selection of one complete embedding export, whose `source.json` identifies the AE checkpoint and waveform domain; documented the legacy flat-directory fallback and matching station-location/Vs30 helper commands.

## 2026-08-24 — Waveform-domain-aware evaluation

- Added explicit `instrument_counts` versus `physical_acceleration` provenance across AE training, embedding export, diffusion checkpoints, and evaluation caches.
- Evaluation now removes instrument response only for counts-domain waveforms. Physical acceleration remains in `m/s^2`; velocity metrics use frequency-domain integration rather than an add/remove-response round trip.

## 2026-08-24 — Best-validation diffusion checkpoints

- Documented strict finite `Loss/val` minimization, earliest-epoch tie retention, checkpoint metadata, and TensorBoard/W&B metrics and summary fields for the selected checkpoint.
- Documented W&B run-ID-isolated `best_val` and final `unet2d` paths, staged best-checkpoint replacement with rollback protection, and final evaluation's use of the current run's best checkpoint with a final-model fallback.

## 2026-08-24 — Configurable W&B sweep display names

- Added W&B-only `--wandb_run_name` and `--wandb_name_params` options for deterministic, parameter-suffixed sweep run names without changing experiment, checkpoint, or TensorBoard paths.

## 2026-08-24 — Run-isolated final evaluation outputs

- Documented per-W&B-run final-evaluation roots at `eval/runs/<training_type>/<sanitized-experiment-or-W&B-name>/<wandb-run-id>/`, with evaluator-specific subdirectories for PNGs, caches, and temporary artifacts. The run ID prevents concurrent runs, including repeated sweep configurations, from sharing files.
- Documented the `SEISMIC_EVAL_OUTPUT_ROOT` subprocess handoff, W&B output-path records, and preservation of the standalone `eval/<evaluator>/` output default.

## 2026-08-24 — End-of-training W&B diffusion evaluation

- Documented the default final evaluation suite for W&B-tracked diffusion training: its selected checkpoint is evaluated by all eight existing evaluators, and their statuses and generated figures are logged to the original W&B run before it finishes.
- Documented the automatic suite's deterministic evenly spaced 10% selection and fixed synthetic seed `0`, while preserving standalone evaluators' full-data `--fraction 1.0` default.
- Documented explicit checkpoint/cache threading and reuse of the first-order record selection by dependent evaluators; selection-derived cache tags prevent result mixing or clobbering. Also documented live child progress and the eight-task suite progress bar, failure/skip reporting, and `--no-run_final_evaluation` for avoiding the potentially long post-training evaluation.

## 2026-08-22 — Global-normalized diffusion migration

- Documented final-train-split-only fitting for latent and continuous-conditioning statistics.
- Documented schema-v2 diffusion checkpoint provenance, checkpoint-local embedding snapshots, split records, and immutable mappings.
- Documented exact global log-magnitude inversion with `expm1`, plus the bypass of amplitude restoration for global AEs.
- Documented legacy per-event compatibility and checkpoint-aware evaluation/demo cache identity.

## 2026-08-22 — Physical-unit waveform preprocessing

- Documented the strict, epoch-aware response-inventory gate for raw waveform correction.
- Documented the separate, resumable physical-acceleration archive, its processing defaults, provenance artifacts, QC thresholds, and deterministic review PNGs.
- Documented the explicit `KO` fallback for blank raw MiniSEED network headers, including raw/effective NSLC reporting, the 51-trace/17-file coverage rationale, and opt-out behavior.
- Added an invertible epsilon-aware physical log-magnitude transform, checkpoint/export/reconstruction provenance, legacy `log1p` compatibility, and fail-fast non-finite VAE diagnostics.

## Related docs

- [Global-normalized latent diffusion](features/global-normalized-diffusion.md)
- [Physical-unit waveform preprocessing](features/physical-unit-waveform-preprocessing.md)
- [Glossary](glossary.md)
- [Documentation overview](README.md)
