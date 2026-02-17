# Seismic Autoencoder Training

This directory contains code for training a convolutional autoencoder on seismic waveform data converted to STFT spectrograms.

## Status Note (2026-02-17)

This document describes the legacy root autoencoder scripts.
Current isolated experiment workflows are maintained under:

- `ML/autoencoder/experiments/General/README.md`
- `ML/autoencoder/experiments/NonDiagonel/README.md`
- `ML/autoencoder/experiments/NonDiagonalRigid/README.md`

Use experiment-level READMEs for the latest training/evaluation commands and protocol decisions.

## Files

- `model.py`: Convolutional autoencoder architecture
- `stft_dataset.py`: PyTorch dataset for loading seismic waveforms and converting to STFT
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

### 4. Resume Training

```bash
python train.py --resume checkpoints/20231216_120000/best_model.pt
```

## Command Line Arguments

### Data Arguments
- `--data_dir`: Path to filtered waveforms directory (default: `../../data/filtered_waveforms`)
- `--channels`: Channel types to include (default: `HH`)

### STFT Arguments
- `--nperseg`: Length of each segment for STFT (default: 256)
- `--noverlap`: Number of points to overlap between segments (default: 192)
- `--nfft`: Length of the FFT used (default: 256)

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
- `--save_interval`: Save checkpoint every N epochs (default: 5)
- `--resume`: Path to checkpoint to resume from

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

Checkpoints are saved in `checkpoints/<timestamp>/`:
- `checkpoint_epoch_N.pt`: Regular checkpoints (keeps last 3)
- `best_model.pt`: Best model based on validation loss
- `config.json`: Training configuration

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
