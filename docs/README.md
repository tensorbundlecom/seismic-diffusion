# Project documentation

**Last updated:** 2026-08-22

## TL;DR

This documentation records reproducible pipeline contracts whose configuration and provenance affect model results. It covers both the physical-unit waveform archive created before autoencoder training and the global-normalized embedding, diffusion, reconstruction, and evaluation pipeline.

## Contents

- [Global-normalized diffusion](features/global-normalized-diffusion.md) — training statistics, schema-v2 checkpoint provenance, reconstruction behavior, and compatibility.
- [Physical-unit waveform preprocessing](features/physical-unit-waveform-preprocessing.md) — response inventory coverage, acceleration-archive creation, QC, and review plots.
- [Glossary](glossary.md) — project-specific normalization and provenance terminology.
- [Changelog](changelog.md) — human-readable documentation changes.

## Documentation scope

The top-level [project README](../README.md) describes the dataset and model families. The [autoencoder README](../ML/autoencoder/README.md) documents AE training and the embedding-export input contract. This folder documents cross-component behavior that must remain reproducible from waveform preprocessing through evaluation. Generated paths below `data/` are normally ignored by Git; their manifests and reports are authoritative for each preprocessing run.

## Related docs

- [Autoencoder training and embedding export](../ML/autoencoder/README.md)
- [Global-normalized diffusion](features/global-normalized-diffusion.md)
- [Physical-unit waveform preprocessing](features/physical-unit-waveform-preprocessing.md)
- [Glossary](glossary.md)
