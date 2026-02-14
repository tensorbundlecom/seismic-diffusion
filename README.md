# Seismic Diffusion Project

An exploration of applying diffusion models to seismic waveform data from the Marmara region of Turkey.

## Data

This project works with seismic waveform data from earthquake events in the Marmara Sea region of Turkey, spanning from 2014 to present. The data was sourced from Kandilli Observatory seismic network.

### Raw Data Collection

We collected three main types of raw data:

- **Earthquake Catalog**: Event metadata including timestamps, locations (latitude/longitude), depths, and magnitudes (ranging from ML 0.8 to 4.9). The catalog contains approximately 14,882 earthquake events from the region.

- **Phase Picks**: Seismological phase arrival times (P-wave and S-wave picks) recorded at various seismic stations. These phase picks are stored in GSE2.0 format and include information about:
  - Station distance and azimuth
  - Arrival times with time residuals
  - Signal-to-noise ratios (SNR)
  - Local magnitude estimates at each station

- **Seismic Waveforms**: Three-component seismic recordings stored in miniSEED format, downloaded from Kandilli Observatory's network. Each waveform file contains:
  - Three channels of ground motion data (typically East-West, North-South, Vertical)
  - Sampling rates varying by instrument (HH, BH, EH, HN channel codes indicating different sample rates)

### Processed Dataset

From the raw data, we created a curated dataset of high-quality waveforms:

- **Initial Collection**: ~164,000 waveforms with computed quality metrics
- **Quality Filtered Dataset**: ~129,000 waveforms meeting quality criteria

Each waveform entry includes:
- Event ID and magnitude
- Signal-to-noise ratios for all three components (snr1, snr2, snr3)
- Data gap metrics for each component (gap1, gap2, gap3)
- File path to the miniSEED waveform data

The quality filtering focuses on selecting waveforms with high SNR (good signal quality) and low gap values (minimal missing data), ensuring clean input data for machine learning experiments.

### Waveform Filtering and Preprocessing

After selecting high-quality waveforms, we applied signal processing steps to prepare the data for machine learning:

**Quality Thresholds:**
- **SNR Threshold**: Maximum SNR across the three components must exceed 1.5
- **Gap Threshold**: All three components must have zero data gaps (gap_max = 0)

These criteria ensure that only recordings with strong, continuous signals are included in the training dataset.

**Bandpass Filtering:**

The waveforms were filtered using magnitude-dependent and instrument-specific bandpass filters. The filtering frequencies were chosen to capture the relevant frequency content for different earthquake magnitudes while accounting for the varying sampling rates of different instrument types:

| Instrument | Magnitude | Low Freq (Hz) | High Freq (Hz) |
|------------|-----------|---------------|----------------|
| **HH** (High-rate, 100 Hz) | 0-1 | 2.0 | 25 |
| | 1-2 | 1.0 | 20 |
| | 2-3 | 0.5 | 15 |
| | 3-4 | 0.3 | 10 |
| | 4+ | 0.1 | 8 |
| **BH** (Broadband, 20-80 Hz) | 0-1 | 0.5 | 10 |
| | 1-2 | 0.5 | 8 |
| | 2-3 | 0.5 | 5 |
| | 3-4 | 0.3 | 5 |
| | 4+ | 0.1 | 3 |
| **EH** (Extremely high-rate, 100+ Hz) | 0-1 | 3.0 | 40 |
| | 1-2 | 2.0 | 35 |
| | 2-3 | 1.0 | 25 |
| | 3-4 | 0.5 | 20 |
| | 4+ | 0.3 | 15 |

**Preprocessing Pipeline:**

Before applying the bandpass filter, each waveform underwent a series of standard preprocessing steps to remove artifacts and prepare the signal:

1. **Demeaning (DC Offset Removal)**: Subtracts the mean value from the entire trace. Seismic sensors can drift over time, introducing a constant offset that doesn't represent actual ground motion. Removing this offset ensures the waveform is centered around zero.

2. **Linear Detrending**: Removes long-period linear trends in the data. Sensor drift or temperature changes can cause gradual increases or decreases in the baseline that aren't related to seismic activity. Detrending eliminates these instrumental artifacts.

3. **Tapering (5% Cosine Taper)**: Applies a smooth cosine taper to the first and last 5% of the signal. This gradual fade-in/fade-out at the edges prevents spectral leakage when applying frequency-domain operations. Without tapering, abrupt signal edges can introduce artificial high-frequency components during filtering.

4. **Bandpass Filtering**: Applies the magnitude-appropriate frequency filter (see table above) to isolate the frequency band containing the earthquake signal. Lower magnitude earthquakes generate higher frequency signals, while larger events produce more low-frequency energy. The filter removes both low-frequency noise (microseisms, atmospheric pressure changes) and high-frequency noise (electronic interference, cultural noise).

The filtered waveforms were saved in miniSEED format, preserving the original timing and metadata. This preprocessing ensures that the machine learning models train on clean signals that represent actual ground motion from earthquakes rather than instrumental or environmental noise.

## Machine Learning Approaches

We explored several deep learning architectures for learning compressed representations of seismic waveforms, with the eventual goal of applying diffusion models for waveform generation.

### Data Representation: STFT Spectrograms

Rather than working directly with time-series waveform data, we converted the three-component seismic signals into Short-Time Fourier Transform (STFT) spectrograms. This time-frequency representation offers several advantages:

- **Frequency Content Visualization**: Spectrograms reveal which frequencies are present at different times during the earthquake signal
- **2D Image Format**: Enables use of standard computer vision architectures (CNNs) designed for image data
- **Natural Multi-channel Structure**: The three components (East-West, North-South, Vertical) map naturally to RGB-like channels

With default STFT parameters (nperseg=256, noverlap=192, nfft=256), each waveform is transformed into a 3-channel spectrogram with shape (3, 129, ~110) representing frequency bins × time bins across the three components.

### Autoencoder (AE)

**Architecture:**

The standard autoencoder uses a symmetric convolutional architecture to compress and reconstruct spectrogram images:

- **Encoder**: Four convolutional layers progressively downsample the input (32 → 64 → 128 → 256 feature channels) with 2x2 striding, reducing spatial dimensions by 16×. Batch normalization and ReLU activations provide training stability and nonlinearity.

- **Latent Space**: A spatial feature map of shape (256, H/16, W/16) serves as the compressed representation, containing ~65k features for a 256×256 input.

- **Decoder**: Four transposed convolutional layers mirror the encoder, progressively upsampling back to the original dimensions (256 → 128 → 64 → 32 → 3 channels).

**Training Objective:**

Minimizes mean squared error (MSE) between input and reconstructed spectrograms, learning to preserve the most salient features needed for accurate reconstruction.

**Purpose:**

Establishes a baseline for deterministic compression and reconstruction, testing whether convolutional architectures can capture the structure of seismic spectrograms.

### Variational Autoencoder (VAE)

**Architecture:**

The VAE extends the autoencoder with probabilistic latent representations:

- **Encoder**: Outputs two vectors instead of a single latent code: μ (mean) and log σ² (log-variance), each of dimension 128. These parameterize a diagonal Gaussian distribution in latent space.

- **Reparameterization**: Samples latent vectors using z = μ + σ ⊙ ε where ε ~ N(0, I), enabling backpropagation through the stochastic sampling operation.

- **Latent Space**: A 1D latent vector (128-dimensional) instead of spatial features. This compact representation forces the model to learn a smooth, structured latent space.

- **Decoder**: Projects the latent vector back to spatial dimensions before applying transposed convolutions to reconstruct the spectrogram.

**Training Objective:**

VAE loss = Reconstruction Loss + β × KL Divergence

- **Reconstruction Loss**: MSE between input and output (same as AE)
- **KL Divergence**: Regularizes the latent space to match a standard normal distribution N(0, I), ensuring smooth interpolation and meaningful sampling
- **β parameter**: Balances reconstruction fidelity vs. latent space regularity (β-VAE framework)

**Purpose:**

Learning a probabilistic latent space enables two key capabilities:
1. **Generation**: Sample new waveforms by drawing z ~ N(0, I) and decoding
2. **Interpolation**: Smoothly interpolate between different earthquake signals in latent space

The 1D latent representation significantly reduces the graininess seen in spatial latent VAEs, producing cleaner reconstructions.

### Conditional Variational Autoencoder (CVAE)

**Architecture:**

The CVAE extends the VAE to condition both encoding and decoding on earthquake metadata:

- **Conditioning Variables**:
  - Event magnitude (1 feature)
  - Event location: latitude, longitude, depth (3 features, normalized)
  - Recording station (learned embedding of dimension 16)

- **Condition Processing**: A small neural network processes the concatenated condition features (magnitude + location + station embedding) into a 64-dimensional conditioning vector.

- **Encoder**: Concatenates image features with conditioning features before projecting to μ and log σ², making the latent distribution depend on the metadata.

- **Decoder**: Receives both the sampled latent vector z and the conditioning features, enabling controlled generation based on desired earthquake characteristics.

**Training Objective:**

Same as VAE (reconstruction + KL divergence), but now the distributions are conditioned on metadata: Q(z|x, c) and P(x|z, c) where c represents the conditioning information.

**Purpose:**

Enables controlled generation by specifying earthquake properties:
- Generate waveforms for specific magnitude ranges
- Create location-specific or station-specific synthetic waveforms
- Learn how waveform characteristics vary with magnitude, distance, and recording station
- Potentially generate training data for underrepresented scenarios (e.g., large magnitude events in specific locations)

The CVAE is particularly valuable for seismology applications where physical parameters (magnitude, location) have known relationships with waveform characteristics, and we want the model to explicitly learn these relationships rather than treating them as latent factors.

### Summary of Architectures

| Model | Latent Space | Stochastic | Conditional | Generation | Use Case |
|-------|--------------|------------|-------------|------------|----------|
| **AE** | Spatial (256, H/16, W/16) | No | No | No | Baseline compression/reconstruction |
| **VAE** | Vector (128-dim) | Yes | No | Yes | Unconditional generation, smooth latent space |
| **CVAE** | Vector (128-dim) | Yes | Yes | Yes | Controlled generation with metadata |

All models were implemented in PyTorch and trained on NVIDIA GPUs using the Adam optimizer with MSE reconstruction loss (and KL divergence for variational models).
