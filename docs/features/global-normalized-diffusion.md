# Global-normalized latent diffusion

**Last updated:** 2026-08-26

## TL;DR

Latent diffusion fits latent and continuous-conditioning statistics on final training rows only. Schema-v2 checkpoints preserve the complete AE contract needed to reconstruct global-normalized spectrograms, including the physical-amplitude epsilon. Reconstruction applies `exp(log_magnitude) - amplitude_epsilon`; it does not use the legacy amplitude MLP, predicted gain, or post–Griffin–Lim waveform rescale.

## Why this exists

The AE's `--global_normalization` option preserves absolute log-magnitude differences across waveform events. Diffusion must therefore retain the associated global bounds and must not apply the per-event amplitude restoration designed for legacy embeddings. A checkpoint must also remain interpretable after the mutable `embeddings/` directory has changed.

## Training contract

`ML/diffusion/create_embeddings.py` writes one export per AE run at `ML/diffusion/embeddings/<AE run name>/`. An export contains `embeddings.pt`, `metadata.json`, and `source.json`; the latter identifies the AE name, checkpoint, and waveform domain as well as the reconstruction contract. Existing core artifacts for the same AE are protected unless export is invoked with `--overwrite`, while exports from different AEs coexist. An intentional replacement is fully staged before installation and preserves export-local station artifacts.

`ML/diffusion/train.py` selects one complete export with `--embeddings_dir` and reads its `source.json`. It validates the count, latent shape, normalization fields, STFT metadata, recorded AE name, and checkpoint identity before training. The selected source is therefore the authority for which AE and waveform domain the diffusion run consumes, rather than whichever files happen to be in an `embeddings/` root.

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
  --use_vs30 true \
  --embeddings_dir embeddings/vae-global-physical-v2
```

Use the same AE checkpoint for embedding export that the diffusion run is meant to consume. For a named global-normalized AE, export explicitly:

```bash
python create_embeddings.py \
  --data_dir ../../data/physical_waveforms_snr2_2-15hz \
  --waveform_summary ../../data/waveform_summary.csv \
  --target_freq_bins 33 --target_time_bins 701 \
  --ae_checkpoint ../autoencoder/checkpoints/vae-global-physical-v2/best_model.pt
```

That command writes `embeddings/vae-global-physical-v2/`. Prepare station locations for this export before diffusion training, and compute Vs30 there when that conditioning feature is enabled:

```bash
python fetch_station_locations.py --embeddings_dir embeddings/vae-global-physical-v2
python compute_station_vs30.py --embeddings_dir embeddings/vae-global-physical-v2
```

The default `--embeddings_dir embeddings` preserves compatibility with legacy flat exports, including their station lookup files. New training should supply the AE-specific directory explicitly.

Do not overwrite an embedding export and assume an existing diffusion checkpoint adopts it: the checkpoint's own provenance is authoritative at inference time.

### Sweep-ready optimizer, loader, and U-Net settings

`ML/diffusion/train.py` exposes the following CLI settings for reproducible parameter sweeps. Their parsed values configure the AdamW optimizer, data loaders, or diffusion U-Net as indicated, are logged in the W&B run configuration, and are saved in the checkpoint `training_config` so a selected run can be reconstructed without relying on sweep-controller state.

| CLI option | Default | Validation and effect |
| --- | ---: | --- |
| `--lr` | `1e-4` | Positive finite learning rate for the optimizer. |
| `--weight_decay` | `1e-2` | Finite weight decay of zero or greater for the optimizer. |
| `--batch_size` | `32` | Positive integer batch size for the training and validation loaders. |
| `--base_channels` | `64` | Positive width that is a multiple of 32; sets the U-Net's base channel count. The multiple-of-32 requirement comes from Diffusers GroupNorm. |
| `--layers_per_block` | `2` | Positive integer number of U-Net layers per block. |

For bounded initial architecture sweeps, use `--base_channels` values `32` or `64` with `--layers_per_block` values `1` or `2`. The resulting models are approximately 4.2M, 6.1M, 16.4M, and 24.1M parameters, respectively (in that width/layer combination order). These are estimates for comparing sweep candidates, not a checkpoint compatibility guarantee; architecture metadata stored with a checkpoint remains authoritative when loading it.

For example, a compact sweep candidate can be started with:

```bash
python train.py \
  --lr 1e-4 --weight_decay 1e-2 --batch_size 32 \
  --base_channels 32 --layers_per_block 1 \
  --wandb_run_name diffusion-sweep \
  --wandb_name_params lr,weight_decay,batch_size,base_channels,layers_per_block
```

This CLI contract is ready for W&B sweeps. Final evaluation remains enabled by default for W&B-tracked runs; add `--no-run_final_evaluation` only when a sweep run should skip the end-of-training evaluation suite.

### PhaseNet during physical AE training

The optional AE PhaseNet perceptual loss also depends on the waveform domain. For a `physical_acceleration` AE, it requires the exact global-normalization bounds and inverts a normalized spectrogram to physical magnitude with `max(exp(normalized * (global_max - global_min) + global_min) - amplitude_epsilon, 0)` before PhaseNet. Therefore physical PhaseNet training requires `--global_normalization`. Count-domain AEs deliberately retain the legacy normalized `expm1` PhaseNet input even when global normalization is enabled; they do not denormalize through these bounds.

The AE configuration records the selected `phasenet_input_mode`, so the perceptual-loss contract is inspectable alongside `waveform_domain`, `amplitude_epsilon`, and the global bounds.

## Schema-v2 checkpoint provenance

Each checkpoint directory contains `training_config.json` plus checkpoint-local copies of `embedding_source.json` and `embedding_scale.json`. The config contains:

- `embedding_provenance`: source file identity and hash, a source snapshot, AE checkpoint path and SHA-256 hash, AE normalization mode/bounds/amplitude epsilon, STFT settings, channels, embedding count, and latent shape.
- `data_normalization`: training-only latent mean and standard deviation.
- `conditioning_normalization`: training-only continuous-conditioning mean and standard deviation.
- `split`: training, validation, and held-out embedding indices plus held-out station IDs.
- `mappings`: immutable station-index-to-name and channel-index-to-type mappings.

These fields are included in the serialized model checkpoint as well as the checkpoint-local JSON artifacts. They allow evaluation and the demo to use the model's recorded normalization/STFT/source contract rather than current global files under `embeddings/`.

## Best-validation checkpoint selection

Whenever validation runs, diffusion training retains a `best_val` checkpoint for the lowest finite `Loss/val` observed in the current process. A candidate must be a *strict* improvement; equal losses deliberately leave the existing checkpoint intact, so the earliest epoch that attains a tied value remains selected. Non-finite validation losses never create or replace this checkpoint.

For ordinary training, the checkpoint is saved at:

```text
checkpoints/<training_type>/<experiment_name>/best_val
```

When W&B is active, its run ID isolates both the selected and final checkpoints so concurrent or repeated runs using the same experiment name cannot overwrite one another:

```text
checkpoints/<training_type>/<experiment_name>/wandb_runs/<wandb-run-id>/best_val
checkpoints/<training_type>/<experiment_name>/wandb_runs/<wandb-run-id>/unet2d
```

Each best-checkpoint update is written to a complete staged directory before replacing the previous selection. If the replacement fails, the previous checkpoint is restored. The checkpoint's `training_config.json` records its selection metric (`Loss/val`), minimize goal, selected loss and epoch, path, and earliest-tie policy. On each improvement, TensorBoard records `Loss/val_best`; W&B records `Loss/val_best`, `Checkpoint/best_val_epoch`, and `Checkpoint/best_val_path`, and adds `best_val_loss`, `best_val_epoch`, and `best_val_checkpoint` to the run summary.

## Reconstruction behavior

For a global-normalized AE, decoded values are inverted exactly using the bounds recorded with the AE embedding source:

```text
log_magnitude = decoded * (global_max - global_min) + global_min
magnitude = max(exp(log_magnitude) - amplitude_epsilon, 0)
```

Legacy global checkpoints lacking `amplitude_epsilon` use `1.0`, which reduces exactly to the historical `expm1` inverse.

The inverse intentionally does not clamp a decoded value to `[0, 1]`; generated values outside that interval remain part of the learned output. The magnitude is then passed to Griffin–Lim as configured by the recorded STFT settings.

The global path skips all legacy amplitude restoration:

- no `AmpMLP` load or inference;
- no predicted log-amplitude/gain application;
- no waveform rescaling after Griffin–Lim.

Legacy per-event-normalized AEs remain supported. Their reconstruction path continues to use the saved legacy inverse gain and amplitude model/postprocessing where applicable.

## Evaluation, demo, and caches

`demo/app.py` and the scripts under `eval/` resolve reconstruction details from the selected diffusion checkpoint first. Older checkpoints without schema-v2 provenance may fall back to the existing embedding source and legacy behavior.

### Demo model selectors

For latent-mode checkpoints, the demo's **Autoencoder** selector lists only AEs that at least one available diffusion checkpoint explicitly identifies in its saved provenance. Selecting an AE automatically loads the newest compatible diffusion checkpoint, preferring a checkpoint with the currently selected training type (`ddpm` or `flow_matching`) when one is available. The demo then reloads the AE recorded by that checkpoint, along with its reconstruction settings.

Do not treat matching latent tensor shapes as proof that an arbitrary AE and diffusion checkpoint can be combined. A diffusion model learns the particular AE latent space used for its embedding export; combining same-shape models without an explicit provenance binding can yield invalid decoding. The selector intentionally prevents that unsupported pairing.

A checkpoint switch also changes the demo's complete reference-data bundle. The selected checkpoint's `embedding_provenance.embeddings_dir` supplies its `metadata.json`, `station_locations.json`, optional `station_vs30.json`, `source.json`, `scale.json`, and source-recorded STFT fallback; recorded metadata paths are used to load the reference waveforms. Consequently, **Random EQ** and the reference STFT/waveform plots always use records from the active pipeline's embedding export rather than mutable files from another export.

The demo checks that the export's `waveform_domain` agrees with the checkpoint provenance before publishing the new bundle. A failed model, AE, artifact, or domain validation leaves the previously loaded model and reference data intact, so a failed switch cannot pair generated results with reference waveforms from another domain.

Generated-evaluation cache identities include the diffusion checkpoint/model configuration together with normalization and embedding-source identity. This prevents reuse of samples that were produced with a different AE normalization contract or diffusion checkpoint.

Waveform domain is part of the same immutable provenance chain. New AE and embedding exports record either `instrument_counts` or `physical_acceleration`; diffusion checkpoints snapshot that value and evaluation cache identities include it. Counts-domain evaluation removes the matching station response only when physical ground motion is requested. Physical-acceleration evaluation keeps the generated and real waveforms in `m/s^2` and bypasses response removal; velocity metrics use zero-DC frequency-domain integration. Legacy checkpoints without this field remain counts-domain by default and can be evaluated explicitly with `--waveform_domain physical_acceleration` when their true domain is known.

### End-of-training W&B evaluation

For a W&B-tracked diffusion run, `ML/diffusion/train.py` saves the final `unet2d` checkpoint before starting the evaluation suite. It evaluates the selected finite `best_val` checkpoint when one was saved during this process; otherwise it falls back to `unet2d` (for example, when validation is disabled or produces no finite loss). It then runs the eight existing evaluators—first-order waveforms, peak amplitudes, residual distributions, distributions, shake duration, attenuation, scaling, and model probabilities—and logs their per-evaluator status and newly reported output figures to the *same* W&B run. The run is finished only after this logging completes.

The automatic suite uses a deterministic, evenly spaced 10% subset (`--fraction 0.1`) of the first-order validation records and of each evaluator's selected observed records. Synthetic generation uses seed `0`. This keeps runs comparable while avoiding full-dataset post-training evaluation. Standalone evaluator commands retain their full-data default (`--fraction 1.0`), so this reduced fraction applies only to the training-triggered suite.

The orchestration passes that exact selected-checkpoint path to evaluators that require a checkpoint. It obtains the first-order waveform cache from that evaluator's output and passes that exact cache to the dependent peak-amplitude, distribution, and duration evaluators; it also derives and passes the corresponding peak cache to residual-distribution evaluation. Dependent evaluators therefore reuse the first-order selection rather than sampling again. Cache filenames include a stable tag derived from the ordered selected records (as well as the existing model/domain identity), preventing results for different selections from being mixed or overwritten.

Automatic-suite artifacts are isolated per tracked run. Their root is:

```text
eval/runs/<training_type>/<sanitized-experiment-or-W&B-name>/<wandb-run-id>/
```

Each evaluator writes beneath that root, for example `first_order/`, `peak_amplitudes/`, and `attenuation/`. PNGs, generated-waveform caches, derived peak caches, and evaluator temporary files are all scoped there, so two agents may evaluate the same experiment name concurrently without sharing output paths. The run ID is the collision-prevention boundary; the experiment or W&B display name is sanitized only to make the directory readable. Training passes this root to evaluator subprocesses through `SEISMIC_EVAL_OUTPUT_ROOT`. The chosen root is recoverable from the W&B run configuration (training type and experiment name), the `evaluation_output_root` summary field, and the `Evaluation/output_root` status log.

Standalone evaluator invocations do not set this environment variable and retain their historical paths under `eval/<evaluator>/`.

Evaluation child-process output is streamed live to the training terminal, including each evaluator's own progress bars. The parent also displays an eight-task `tqdm` suite bar, advancing for completed and skipped tasks.

Final evaluation is enabled by default only while W&B is active. It can take a long time, so opt out for a tracked run with:

```bash
python train.py --no-run_final_evaluation
```

An individual evaluator failure is recorded in W&B and does not stop independent evaluators from running. Dependents whose required cache was not produced are logged as skipped, and the run receives an overall evaluation status summary.

### W&B sweep display names

`--wandb_run_name` sets a W&B-only display-name base and does not affect `--experiment_name`, checkpoint directories, or TensorBoard paths. For sweeps, use `--wandb_name_params` with a comma-separated list of parsed CLI names to append the selected values. For example:

```bash
python train.py --wandb_run_name prediction-target-sweep --wandb_name_params prediction_target
```

creates `prediction-target-sweep-prediction_target=epsilon` for an epsilon run. Multiple parameters are appended in the supplied order. Names are validated against the parsed CLI arguments; empty, duplicate, or unknown names fail before W&B initialization. Booleans and null values are rendered as lowercase `true`/`false` and `none`.

No training or evaluation command is run as part of this migration. Train a new global-normalized AE, export its embeddings, then train a new diffusion model before relying on the new global reconstruction path.

## Related docs

- [Documentation overview](../README.md)
- [Autoencoder global-normalization contract](../../ML/autoencoder/README.md)
- [Glossary](../glossary.md)
- [Changelog](../changelog.md)
