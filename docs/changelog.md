# Documentation changelog

**Last updated:** 2026-08-22

## TL;DR

This file records documentation-facing changes to pipeline contracts. It is not a substitute for version control history or model-training records.

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
