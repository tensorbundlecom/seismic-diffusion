# Seismic Autoencoder Training

**Last updated:** 2026-08-22

## TL;DR

`train.py` (VAE) and `train_cvae.py` (CVAE) can give a run a stable `--name` and can normalize STFT magnitudes globally with `--global_normalization`. Global normalization fits one log-domain min/max on the training split only, so it preserves amplitude differences between events and stations. `ML/diffusion/create_embeddings.py` reads the selected checkpoint's normalization and preprocessing contract, allowing it to export inputs that match either globally normalized or legacy per-event-normalized AEs.

This directory contains code for training a convolutional autoencoder on seismic waveform data converted to STFT spectrograms.

## Files

- `model.py`: Convolutional autoencoder architecture
- `stft_dataset.py`: PyTorch dataset for loading seismic waveforms and converting to STFT
- `normalization_cache.py`: Content-aware cache for fitted global-normalization bounds
- `train.py`: Training script with tensorboard logging and checkpointing
- `inference.py`: Inference script for testing trained models
- `test_pipeline.py`: Quick test to verify the training pipeline works

## Dataset

The dataset loads seismic waveforms from `data/filtered_waveforms/` directory. Each mseed file contains 3 components (E, N, Z) which are converted to STFT spectrograms and stacked as a 3-channel image.

### Supported Channels
- `HH`: High gain broadband (91,429 files)
- `HN`: High gain short period (35,136 files)
- `EH`: Extremely high gain short period (4,453 files)
- `BH`: Broadband (1,151 files)

## Quick Start

### 1. Basic Training

Train on HH channel data with default settings:

```bash
python train.py
```

### 2. Train on Multiple Channels

```bash
python train.py --channels HH HN EH --batch_size 32 --num_epochs 100
```

### 3. Adjust STFT Parameters

```bash
python train.py --nperseg 512 --noverlap 384 --nfft 512
```

With PhaseNet perceptual loss:

```bash
python train.py --nperseg 512 --noverlap 384 --nfft 512 --use_phasenet_perceptual --phasenet_weight 0.05 --phasenet_pretrained stead
```

### 4. Resume Training

```bash
python train.py --resume checkpoints/20231216_120000/best_model.pt
```

### 5. Named global-normalized VAE run

```bash
python train.py \
  --name vae-global-v1 \
  --global_normalization \
  --channels HH HN EH BH
```

This writes checkpoints to `checkpoints/vae-global-v1/` and TensorBoard logs to `logs/vae-global-v1/`.

### 6. Named global-normalized CVAE run

```bash
python train_cvae.py \
  --name cvae-global-v1 \
  --global_normalization \
  --channels HH HN EH BH
```

The CVAE uses its own default roots: `checkpoints_cvae/cvae-global-v1/` and `logs_cvae/cvae-global-v1/`.

## Command Line Arguments

### Data Arguments
- `--data_dir`: Path to filtered waveforms directory (default: `../../data/filtered_waveforms`)
- `--channels`: Channel types to include (default: `HH`)

### STFT Arguments
- `--nperseg`: Length of each segment for STFT (default: 256)
- `--noverlap`: Number of points to overlap between segments (default: 192)
- `--nfft`: Length of the FFT used (default: 256)
- `--global_normalization`: Use one shared min/max for all post-resize, `log1p` STFT-magnitude values. It is fitted on the training split only. Without this flag, the legacy per-event, per-component min/max normalization remains the default.

### Global Normalization Contract

With `--global_normalization`, the dataset first computes magnitude STFTs, applies `log1p`, and performs any configured resize. It then fits one scalar `global_min` and `global_max` across every finite value in every component of the training indices only. The same fixed pair is reused for training, validation, and test samples:

```text
normalized = (log1p_magnitude - global_min) / (global_max - global_min)
```

Values are not clipped. Therefore validation or test values may lie outside `[0, 1]`, which is intentional and avoids leakage from those splits into the fitted range. A zero fitted range maps all values to zero.

The run configuration and checkpoints persist `normalization_mode`, `global_min`, `global_max`, and `run_name`. Keep the checkpoint and these statistics together whenever the model is used downstream.

#### Progress and normalization cache

Before global normalization is fitted, training fingerprints the ordered training split and displays a `tqdm` progress bar. On a cache miss, a second progress bar tracks the one-time STFT scan that fits the bounds. The resulting JSON cache entry is written atomically, so an interrupted write is not treated as a valid cache entry.

VAE cache entries live in `checkpoints/.normalization_cache/`; CVAE entries live in `checkpoints_cvae/.normalization_cache/`. A matching entry skips the expensive STFT scan (the lightweight file-fingerprinting pass still runs to validate it). The cache key covers the exact ordered training indices, each selected file's resolved path, size, and modification time, plus all STFT/resampling/resize preprocessing settings. Any change to those inputs creates a new entry automatically.

### Model Arguments
- `--latent_dim`: Dimension of latent space (default: 128)

### Training Arguments
- `--batch_size`: Batch size for training (default: 16)
- `--num_epochs`: Number of training epochs (default: 50)
- `--lr`: Learning rate (default: 0.001)
- `--weight_decay`: Weight decay for optimizer (default: 1e-5)
- `--val_split`: Validation split ratio (default: 0.1)
- `--num_workers`: Number of data loader workers (default: 4)

### Checkpoint Arguments
- `--checkpoint_dir`: Directory to save checkpoints (default: `checkpoints`)
- `--log_dir`: Directory for tensorboard logs (default: `logs`)
- `--name`: Optional stable run name. It must begin with an alphanumeric character and contain only letters, numbers, `.`, `_`, or `-`.
- `--save_interval`: Save checkpoint every N epochs (default: 5)
- `--resume`: Path to checkpoint to resume from

When `--name` is supplied, that exact name is used as the subdirectory under both checkpoint and log roots. If either named directory is nonempty, training refuses to start unless `--resume` points to a checkpoint inside that named checkpoint directory. Without `--name`, the existing timestamp-based directory naming is retained.

### Device Arguments
- `--device`: Device to train on (`cuda` or `cpu`, default: `cuda`)
- `--seed`: Random seed (default: 42)

## Monitoring Training

### TensorBoard

Launch tensorboard to monitor training progress:

```bash
tensorboard --logdir logs
```

Then open http://localhost:6006 in your browser.

TensorBoard logs include:
- Training and validation loss curves
- Sample input spectrograms
- Sample reconstructed spectrograms

### Checkpoints

Checkpoints are saved in `checkpoints/<run-name-or-timestamp>/`:
- `checkpoint_epoch_N.pt`: Regular checkpoints (keeps last 3)
- `best_model.pt`: Best model based on validation loss
- `config.json`: Training configuration

For a named run, the matching TensorBoard location is `logs/<run-name>/`; an unnamed run continues to use `logs/<timestamp>/`.

### Preparing diffusion embeddings

`ML/diffusion/create_embeddings.py` defaults to the latest **timestamped** AE checkpoint and does not search named run directories. For any named run, pass its checkpoint explicitly. For example, from `ML/diffusion`:

```bash
python create_embeddings.py \
  --data_dir ../../data/filtered_waveforms_snr2_2-15hz \
  --waveform_summary ../../data/waveform_summary.csv \
  --ae_checkpoint ../autoencoder/checkpoints/vae-global-v1/best_model.pt
```

For a globally normalized AE, embedding export reads `normalization_mode`, `global_min`, and `global_max` from the selected checkpoint and applies that exact fixed pair after `log1p` and any STFT resizing. It never fits bounds over the export dataset. A global-normalized checkpoint without usable saved bounds fails rather than silently refitting or falling back to per-event normalization. Legacy checkpoints continue to use per-event normalization.

Unless explicitly supplied on the command line, export uses the checkpoint's saved channel groups, resize geometry (`target_freq_bins` and `target_time_bins`), resampling rate, and duration. The STFT window settings (`nperseg`, `noverlap`, and `nfft`) always come from the checkpoint configuration. Legacy checkpoints fall back to `HH`, 100 Hz, and 70 seconds where those saved values do not exist. CLI values for `--channels`, `--target_freq_bins`, `--target_time_bins`, `--resample_hz`, and `--target_seconds` override the corresponding saved settings; use overrides only when they intentionally match the AE's training input contract.

The output `embeddings/source.json` records the selected checkpoint, resolved STFT settings, resolved channels, normalization mode, and global bounds (or `null` bounds for per-event normalization), alongside the embedding count and shape.

## Example Training Sessions

### Fast Prototyping (Small Dataset)
```bash
python train.py --channels HH --batch_size 32 --num_epochs 10 --num_workers 2
```

### Full Training (All Channels)
```bash
python train.py --channels HH HN EH BH --batch_size 64 --num_epochs 100 --lr 1e-3 --num_workers 8
```

### High-Resolution STFT
```bash
python train.py --nperseg 512 --noverlap 384 --nfft 512 --batch_size 8
```

### High-Resolution STFT + PhaseNet
```bash
python train.py --nperseg 512 --noverlap 384 --nfft 512 --batch_size 8 --use_phasenet_perceptual --phasenet_weight 0.05 --phasenet_pretrained stead
```

## Model Architecture

The autoencoder consists of:

### Encoder
- Input: (3, H, W) - 3-channel STFT spectrogram
- Conv2d layers with stride 2 for downsampling
- Features: 32 → 64 → 128 → 256
- Output: (256, H/16, W/16) latent representation

### Decoder
- Input: (256, H/16, W/16) latent representation
- ConvTranspose2d layers with stride 2 for upsampling
- Features: 256 → 128 → 64 → 32 → 3
- Output: (3, H, W) reconstructed spectrogram

### Loss Function
Mean Squared Error (MSE) between input and reconstructed spectrograms.

## Expected Output

```
Using device: cuda
Loading dataset from ../../data/filtered_waveforms...
Found 91429 mseed files
Train size: 82286
Val size: 9143
Creating model...
Model parameters: 2,345,123
Starting training for 50 epochs...

Epoch 1/50 [Train]: 100%|██████████| 5143/5143 [10:23<00:00, 8.25it/s, loss=0.0234]
Epoch 1/50 [Val]: 100%|██████████| 572/572 [01:15<00:00, 7.61it/s, loss=0.0198]

Epoch 1/50
  Train Loss: 0.023456
  Val Loss:   0.019876
  New best validation loss!
Saved checkpoint to checkpoints/20231216_120000/checkpoint_epoch_0.pt
Saved best model to checkpoints/20231216_120000/best_model.pt
```

## Notes

- The dataset automatically pads spectrograms to the same size within each batch
- STFT parameters significantly affect the size of the spectrograms and memory usage
- Larger `nfft` values will increase frequency resolution but also memory usage
- Training on all channels will take longer but may produce a more robust model
- Adjust `batch_size` based on your GPU memory (larger is generally better)
- Use `num_workers > 0` for faster data loading (but start with 0 for debugging)

## Testing Trained Models

After training, you can test the model and visualize reconstructions:

```bash
python inference.py checkpoints/20231216_120000/best_model.pt --num_samples 10
```

This will:
- Load the trained model
- Test on random samples from the dataset
- Compute reconstruction metrics (MSE, MAE)
- Save visualization images to `inference_results/`

### Inference Options
- `--num_samples`: Number of samples to test (default: 10)
- `--output_dir`: Directory to save visualizations (default: `inference_results`)
- `--device`: Device to use (`cuda` or `cpu`)

## Troubleshooting

### Out of Memory
- Reduce `--batch_size`
- Reduce `--nfft` (e.g., 128 or 256 instead of 512)
- Use `--device cpu` (slower but no memory limit)

### Slow Training
- Increase `--num_workers` (typically 2-8)
- Use `--device cuda` if available
- Reduce number of channels or dataset size for prototyping

### No CUDA Available
- The script will automatically fall back to CPU
- Consider using Google Colab or a cloud GPU instance for faster training

### Testing the Pipeline
Before running full training, test that everything works:
```bash
python test_pipeline.py
```
This will verify the data loading, model creation, and training on a small subset.

## Related docs

- [VAE training implementation](train.py)
- [CVAE training implementation](train_cvae.py)
- [STFT dataset normalization implementation](stft_dataset.py)
- [Diffusion embedding export](../diffusion/create_embeddings.py)
