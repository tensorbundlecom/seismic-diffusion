# Global-normalized latent diffusion

**Last updated:** 2026-08-22

## TL;DR

Latent diffusion now fits its latent and continuous-conditioning normalization statistics on the final training rows only, after validation and held-out-station rows are selected. Schema-v2 diffusion checkpoints preserve the full embedding/AE contract needed to reconstruct global-normalized spectrograms. For a global-normalized AE, reconstruction applies the exact saved log-domain inverse followed by `expm1`; it does not use the legacy amplitude MLP, predicted gain, or post–Griffin–Lim waveform rescale.

## Why this exists

The AE's `--global_normalization` option preserves absolute log-magnitude differences across waveform events. Diffusion must therefore retain the associated global bounds and must not apply the per-event amplitude restoration designed for legacy embeddings. A checkpoint must also remain interpretable after the mutable `embeddings/` directory has changed.

## Training contract

`ML/diffusion/train.py` reads the exported embedding contract from `ML/diffusion/embeddings/source.json` and validates its count, latent shape, normalization fields, and STFT metadata before training.

The split is resolved before normalization:

1. Any selected held-out stations are removed from the train/validation candidate rows.
2. The validation split is selected from the remaining rows.
3. Latent mean and standard deviation are fitted over the final training embeddings only.
4. The continuous conditioning mean and standard deviation are fitted over those same final training rows only.

Validation and held-out stations therefore cannot influence either normalization fit. Station-ID embeddings, when enabled, are categorical model inputs and are not included in continuous-conditioning normalization. Held-out station IDs are persisted with the split, including when station-ID conditioning is enabled; their embedding-table rows exist in the model but receive no training gradient, so this setting is not a pure unseen-station generalization test.

Run diffusion training from `ML/diffusion`, for example:

```bash
python train.py \
  --prediction_target x0 \
  --data_mode latent \
  --use_vs30 true
```

Use the same AE checkpoint for embedding export that the diffusion run is meant to consume. For a named global-normalized AE, export explicitly:

```bash
python create_embeddings.py \
  --data_dir ../../data/filtered_waveforms_snr2_2-15hz \
  --waveform_summary ../../data/waveform_summary.csv \
  --target_freq_bins 33 --target_time_bins 701 \
  --ae_checkpoint ../autoencoder/checkpoints/vae-global-v1/best_model.pt
```

Do not overwrite an embedding export and assume an existing diffusion checkpoint adopts it: the checkpoint's own provenance is authoritative at inference time.

## Schema-v2 checkpoint provenance

Each checkpoint directory contains `training_config.json` plus checkpoint-local copies of `embedding_source.json` and `embedding_scale.json`. The config contains:

- `embedding_provenance`: source file identity and hash, a source snapshot, AE checkpoint path and SHA-256 hash, AE normalization mode/bounds, STFT settings, channels, embedding count, and latent shape.
- `data_normalization`: training-only latent mean and standard deviation.
- `conditioning_normalization`: training-only continuous-conditioning mean and standard deviation.
- `split`: training, validation, and held-out embedding indices plus held-out station IDs.
- `mappings`: immutable station-index-to-name and channel-index-to-type mappings.

These fields are included in the serialized model checkpoint as well as the checkpoint-local JSON artifacts. They allow evaluation and the demo to use the model's recorded normalization/STFT/source contract rather than current global files under `embeddings/`.

## Reconstruction behavior

For a global-normalized AE, decoded values are inverted exactly using the bounds recorded with the AE embedding source:

```text
log_magnitude = decoded * (global_max - global_min) + global_min
magnitude = expm1(log_magnitude)
```

The inverse intentionally does not clamp a decoded value to `[0, 1]`; generated values outside that interval remain part of the learned output. The magnitude is then passed to Griffin–Lim as configured by the recorded STFT settings.

The global path skips all legacy amplitude restoration:

- no `AmpMLP` load or inference;
- no predicted log-amplitude/gain application;
- no waveform rescaling after Griffin–Lim.

Legacy per-event-normalized AEs remain supported. Their reconstruction path continues to use the saved legacy inverse gain and amplitude model/postprocessing where applicable.

## Evaluation, demo, and caches

`demo/app.py` and the scripts under `eval/` resolve reconstruction details from the selected diffusion checkpoint first. Older checkpoints without schema-v2 provenance may fall back to the existing embedding source and legacy behavior.

Generated-evaluation cache identities include the diffusion checkpoint/model configuration together with normalization and embedding-source identity. This prevents reuse of samples that were produced with a different AE normalization contract or diffusion checkpoint.

No training or evaluation command is run as part of this migration. Train a new global-normalized AE, export its embeddings, then train a new diffusion model before relying on the new global reconstruction path.

## Related docs

- [Documentation overview](../README.md)
- [Autoencoder global-normalization contract](../../ML/autoencoder/README.md)
- [Glossary](../glossary.md)
- [Changelog](../changelog.md)
