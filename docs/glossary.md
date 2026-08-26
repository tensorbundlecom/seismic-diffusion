# Glossary

**Last updated:** 2026-08-26

## TL;DR

These terms describe the physical-unit preprocessing, normalization, and provenance contracts shared by the waveform, autoencoder, and diffusion pipeline. They are intentionally precise because mixing incompatible units or substituting a current embedding file for a checkpoint-recorded contract can change generated waveforms.

## Terms

### Global normalization

One scalar `global_min`/`global_max` pair fitted on training-split log STFT magnitudes after applying the saved amplitude epsilon. It preserves relative amplitude information between events; validation and test values are allowed outside `[0, 1]`.

### Amplitude epsilon

The positive physical-unit floor in `log(magnitude + amplitude_epsilon)`. New physical-acceleration AEs default to `1e-12`; the value is saved with global bounds and inverted after decoding. Legacy checkpoints without this field imply `1.0`, exactly matching `log1p`.

### Per-event normalization

The legacy AE behavior that independently normalizes each event/component. Its inverse reconstruction requires event-level amplitude restoration, so the diffusion path may use `AmpMLP`, a predicted gain, and post–Griffin–Lim waveform rescaling.

### Latent normalization

The diffusion input transform `(embedding - mean) / std`. The mean and standard deviation are fitted from final training embeddings only.

### Continuous conditioning normalization

The standardization of numeric conditioning features (such as event/site properties) using final training rows only. Categorical station IDs are handled separately by a learned embedding table when enabled.

### Diffusion architecture metadata

The saved U-Net width and depth configuration, including `base_channels` and `layers_per_block`, stored with a diffusion checkpoint. It identifies the architecture needed to load that checkpoint; `base_channels` must be a positive multiple of 32 because the Diffusers U-Net uses GroupNorm.

### Embedding provenance

The checkpoint-recorded snapshot of the embedding export contract: source identity, AE checkpoint, AE normalization, STFT configuration, channels, count, and latent shape.

### Provenance-compatible AE/diffusion pair

An autoencoder and latent diffusion checkpoint explicitly connected by the diffusion checkpoint's saved embedding provenance. Matching latent tensor shapes alone do not create this compatibility: the diffusion model was trained in the selected AE's specific latent space.

### Schema-v2 checkpoint

A diffusion checkpoint whose training configuration embeds normalization statistics, split/mapping information, and embedding provenance, and whose directory includes `embedding_source.json` and `embedding_scale.json` snapshots.

### Counts

Raw digitizer output recorded in MiniSEED. Counts depend on the recording instrument and must not be mixed with response-corrected physical units in a globally normalized training corpus.

### FDSN

The Federation of Digital Seismograph Networks data-service ecosystem. The response inventory tool queries FDSN-compatible clients to acquire missing StationXML metadata.

### NSLC

The full SEED identifier `network.station.location.channel`. A response lookup requires all four fields and a valid time epoch; station-only or channel-family-only matches are insufficient.

### Effective NSLC

The NSLC used for response lookup after applying an explicitly configured blank-network fallback. The raw/header NSLC remains unchanged in reports and manifests. The default fallback is `KO`; the preprocessing guide explains how to disable it.

### Response epoch

The interval over which a StationXML channel response applies. A trace can be corrected only when an epoch covers its entire recording interval.

### StationXML

XML metadata describing stations, channels, and instrument responses. The merged project inventory defaults to `eval/station_responses.xml`.

### ACC

ObsPy's response-removal output code for acceleration. In this workflow `ACC` corresponds to SI acceleration in `m/s^2`.

### Water level

The optional deconvolution stabilization limit passed to response removal, expressed in dB. This workflow defaults to no water level and relies on its explicit pre-filter, avoiding unwanted suppression when converting sensors from a non-native quantity to acceleration.

### Pre-filter

The four-frequency taper applied during response removal to suppress unstable deconvolution outside the target range. The default is `0.5, 1.0, 20.0, 22.5` Hz.

## Related docs

- [Global-normalized latent diffusion](features/global-normalized-diffusion.md)
- [Physical-unit waveform preprocessing](features/physical-unit-waveform-preprocessing.md)
- [Documentation overview](README.md)
- [Autoencoder training and embedding export](../ML/autoencoder/README.md)
