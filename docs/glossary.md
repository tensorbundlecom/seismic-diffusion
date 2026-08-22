# Glossary

**Last updated:** 2026-08-22

## TL;DR

These terms describe the normalization and provenance contracts shared by the autoencoder and diffusion pipeline. They are intentionally precise because substituting a current embedding file for a checkpoint-recorded contract can change generated waveforms.

## Terms

### Global normalization

One scalar `global_min`/`global_max` pair fitted on training-split `log1p` STFT magnitudes. It preserves relative amplitude information between events; validation and test values are allowed outside `[0, 1]`.

### Per-event normalization

The legacy AE behavior that independently normalizes each event/component. Its inverse reconstruction requires event-level amplitude restoration, so the diffusion path may use `AmpMLP`, a predicted gain, and post–Griffin–Lim waveform rescaling.

### Latent normalization

The diffusion input transform `(embedding - mean) / std`. The mean and standard deviation are fitted from final training embeddings only.

### Continuous conditioning normalization

The standardization of numeric conditioning features (such as event/site properties) using final training rows only. Categorical station IDs are handled separately by a learned embedding table when enabled.

### Embedding provenance

The checkpoint-recorded snapshot of the embedding export contract: source identity, AE checkpoint, AE normalization, STFT configuration, channels, count, and latent shape.

### Schema-v2 checkpoint

A diffusion checkpoint whose training configuration embeds normalization statistics, split/mapping information, and embedding provenance, and whose directory includes `embedding_source.json` and `embedding_scale.json` snapshots.

## Related docs

- [Global-normalized latent diffusion](features/global-normalized-diffusion.md)
- [Documentation overview](README.md)
- [Autoencoder training and embedding export](../ML/autoencoder/README.md)
