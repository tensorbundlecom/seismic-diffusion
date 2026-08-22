# Project documentation

**Last updated:** 2026-08-22

## TL;DR

This documentation records the operational contracts that connect autoencoder embedding export, latent diffusion training, reconstruction, and evaluation. The global-normalization migration is documented here because its saved statistics and provenance are required to reproduce a trained model's outputs.

## Contents

- [Global-normalized diffusion](features/global-normalized-diffusion.md) — training statistics, schema-v2 checkpoint provenance, reconstruction behavior, and compatibility.
- [Glossary](glossary.md) — project-specific normalization and provenance terminology.
- [Changelog](changelog.md) — human-readable documentation changes.

## Documentation scope

The top-level [project README](../README.md) describes the dataset and model families. The [autoencoder README](../ML/autoencoder/README.md) documents AE training and the embedding-export input contract. This folder documents cross-component behavior that must remain consistent after a diffusion model has been trained.

## Related docs

- [Autoencoder training and embedding export](../ML/autoencoder/README.md)
- [Global-normalized diffusion](features/global-normalized-diffusion.md)
- [Glossary](glossary.md)
