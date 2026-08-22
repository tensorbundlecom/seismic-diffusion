# Documentation changelog

**Last updated:** 2026-08-22

## TL;DR

This file records documentation-facing changes to pipeline contracts. It is not a substitute for version control history or model-training records.

## 2026-08-22 — Global-normalized diffusion migration

- Documented final-train-split-only fitting for latent and continuous-conditioning statistics.
- Documented schema-v2 diffusion checkpoint provenance, checkpoint-local embedding snapshots, split records, and immutable mappings.
- Documented exact global log-magnitude inversion with `expm1`, plus the bypass of amplitude restoration for global AEs.
- Documented legacy per-event compatibility and checkpoint-aware evaluation/demo cache identity.

## Related docs

- [Global-normalized latent diffusion](features/global-normalized-diffusion.md)
- [Glossary](glossary.md)
- [Documentation overview](README.md)
