# Seismic Diffusion Project

A deep learning framework for generating synthetic seismic waveforms using latent diffusion models. This project combines variational autoencoders (VAEs) with conditional diffusion models to learn the distribution of seismic waveforms and generate realistic synthetic data conditioned on physical parameters such as earthquake magnitude, hypocenter location, and recording station.

## Table of Contents

- [Overview](#overview)
- [Methodology](#methodology)
- [Model Architecture](#model-architecture)
- [Data Representation](#data-representation)
- [Project Structure](#project-structure)
- [Technical Pipeline](#technical-pipeline)
- [Key Innovations](#key-innovations)
- [Quick Start](#quick-start)

## Overview

This project implements a two-stage generative model for seismic waveforms:

**Stage 1: Variational Autoencoder (VAE)**
- Learns a compressed latent representation of seismic waveform spectrograms
- Reduces high-dimensional time-frequency data to a compact latent space
- Enables reconstruction of realistic waveforms from latent codes

**Stage 2: Conditional Diffusion Model**
- Operates in the learned latent space (not raw data space)
- Learns to generate latent embeddings conditioned on seismic metadata
- Uses iterative denoising to sample from the learned distribution

The combination enables **controllable generation** of synthetic seismic data with specified characteristics (magnitude, location, etc.), which has applications in data augmentation, hazard assessment, and understanding seismic wave propagation.

## Methodology

### Latent Diffusion Approach

Rather than applying diffusion directly to high-dimensional spectrograms, this project uses a **latent diffusion** approach:

1. **Compression**: A VAE encoder compresses 3-channel spectrograms into low-dimensional latent vectors (typically 128-256 dimensions)
2. **Diffusion in Latent Space**: The diffusion model learns to generate these latent vectors, which is much more efficient than working in pixel space
3. **Reconstruction**: The VAE decoder converts generated latent vectors back to spectrograms

This approach provides several advantages:
- **Computational efficiency**: Diffusion operates on ~128D vectors instead of ~100,000D images
- **Better training stability**: The regularized latent space is easier to model
- **Semantic latent space**: The VAE forces meaningful structure in the latent representation

### Conditional Generation

The diffusion model is conditioned on physical parameters:

- **Event Magnitude**: Controls the energy/amplitude characteristics
- **Hypocenter Location**: Latitude, longitude, and depth (3D)
- **Station Information**: Which seismometer recorded the signal

During generation, users can specify these parameters to produce waveforms with desired characteristics. The model learns the complex relationships between these parameters and waveform features (frequency content, duration, amplitude patterns).

### Data Processing Strategy

**Frequency Domain Representation**: Raw seismic time series are transformed into spectrograms using Short-Time Fourier Transform (STFT). This provides several advantages:
- Captures both temporal and spectral information
- More compact than raw time series
- Aligns with how seismologists analyze data

**Magnitude-Dependent Filtering**: Different earthquake magnitudes generate energy in different frequency bands. The preprocessing applies magnitude-specific bandpass filters:
- Larger events (M ≥ 4): Lower frequency bands (longer periods)
- Smaller events (M < 4): Higher frequency bands (shorter periods)

This ensures the model sees appropriately filtered data matching natural physical processes.

**Quality Control**: Only high-quality waveforms are used for training:
- High Signal-to-Noise Ratio (SNR > threshold)
- Low gap ratio (continuous data)
- Complete three-component recordings (E, N, Z)

## Model Architecture

### Stage 1: Variational Autoencoder

The VAE learns a probabilistic mapping between spectrograms and a latent space.

**Encoder Architecture:**
```
Input: 3-channel STFT spectrogram (E, N, Z components)
  → Conv2D(3→32, kernel=3, stride=2) + BatchNorm + ReLU
  → Conv2D(32→64, kernel=3, stride=2) + BatchNorm + ReLU  
  → Conv2D(64→128, kernel=3, stride=2) + BatchNorm + ReLU
  → Conv2D(128→256, kernel=3, stride=2) + BatchNorm + ReLU
  → Flatten
  → FC → μ (mean) and log(σ²) (log-variance)
  → Reparameterization: z = μ + σ * ε, where ε ~ N(0, 1)
Output: Latent vector z ∈ ℝ^d (typically d=128 or 256)
```

**Decoder Architecture:**
```
Input: Latent vector z ∈ ℝ^d
  → FC → Reshape to (256, H/16, W/16)
  → ConvTranspose2D(256→128, kernel=3, stride=2) + BatchNorm + ReLU
  → ConvTranspose2D(128→64, kernel=3, stride=2) + BatchNorm + ReLU
  → ConvTranspose2D(64→32, kernel=3, stride=2) + BatchNorm + ReLU
  → ConvTranspose2D(32→3, kernel=3, stride=2) + Tanh
Output: Reconstructed 3-channel spectrogram
```

**Training Objective:**

The VAE is trained with a combined loss:

```
L = L_reconstruction + β × L_KL

where:
  L_reconstruction = MSE(x, x̂)  # Mean squared error
  L_KL = -0.5 × Σ(1 + log(σ²) - μ² - σ²)  # KL divergence from N(0,1)
  β controls the weight of the KL term (β-VAE)
```

The β parameter is crucial:
- β = 1: Standard VAE
- β > 1: Encourages disentangled representations
- β < 1: Prioritizes reconstruction quality

**Conditional VAE (CVAE) Variant:**

The CVAE extends the standard VAE by conditioning on metadata:

```
Conditioning vector c = [magnitude, latitude, longitude, depth, station_embedding]

Encoder: q(z | x, c) - conditions on both input and metadata
Decoder: p(x | z, c) - conditions on both latent and metadata
```

The conditioning vectors are concatenated with activations at multiple layers, allowing the model to modulate its behavior based on physical parameters.

### Stage 2: Conditional Diffusion Model

The diffusion model learns the distribution of latent embeddings p(z) conditioned on seismic parameters.

**Forward Process (Adding Noise):**

A Markov chain that gradually adds Gaussian noise to the latent embedding:

```
q(z_t | z_{t-1}) = N(z_t; √(1-β_t) × z_{t-1}, β_t × I)

where:
  t ∈ [1, T] is the diffusion timestep (typically T=1000)
  β_t is a noise schedule (gradually increases from β_1 ≈ 0.0001 to β_T ≈ 0.02)
  z_0 is the original latent embedding
  z_T ≈ N(0, I) is pure noise
```

**Reverse Process (Denoising):**

A learned neural network predicts and removes noise iteratively:

```
p_θ(z_{t-1} | z_t, c) = N(z_{t-1}; μ_θ(z_t, t, c), Σ_θ(z_t, t, c))

where:
  θ are the neural network parameters
  c is the conditioning information (magnitude, location, station)
```

**Denoising Network Architecture:**

```
Input: [z_t, t, c]
  → Time embedding: t → sin/cos positional encoding → MLP → time_emb
  → Condition embedding: c → MLP → cond_emb
  → Combined: [z_t, time_emb, cond_emb] → Concatenate
  → MLP layers with skip connections:
      Hidden1(512) → ReLU → Hidden2(512) → ReLU → ... → Output(latent_dim)
  → Residual connection: output + z_t
Output: Predicted noise ε_θ(z_t, t, c)
```

**Training Objective:**

The model is trained to predict the noise that was added:

```
L = E_t,z_0,ε [||ε - ε_θ(z_t, t, c)||²]

where:
  ε ~ N(0, I) is the random noise that was added
  z_t = √(ᾱ_t) × z_0 + √(1-ᾱ_t) × ε
  ᾱ_t = Π_{i=1}^t (1-β_i)
```

**Generation Process:**

To generate a new waveform:

1. Sample z_T ~ N(0, I) (random noise)
2. For t = T down to 1:
   - Predict noise: ε̂ = ε_θ(z_t, t, c)
   - Compute z_{t-1} using the reverse diffusion equation
3. Decode z_0 through VAE decoder to get spectrogram
4. (Optional) Convert spectrogram back to time series via inverse STFT

## Data Representation

### Three-Component Seismic Data

Seismometers record ground motion in three orthogonal directions:
- **E (East)**: Horizontal motion in the east-west direction
- **N (North)**: Horizontal motion in the north-south direction  
- **Z (Vertical)**: Vertical motion (up-down)

These three components together capture the complete 3D ground motion vector.

### Short-Time Fourier Transform (STFT)

Raw time series are converted to spectrograms using STFT:

```
S(n, k) = Σ_{m=0}^{N-1} x[m] × w[n-m] × e^{-j2πkm/N}

Parameters:
  - nperseg: Window length (default: 256 samples)
  - noverlap: Overlap between windows (default: 192 samples, 75% overlap)
  - nfft: FFT length (default: 256)
```

This produces a 2D representation where:
- **Horizontal axis**: Time (sequence of windows)
- **Vertical axis**: Frequency (0 to Nyquist frequency)
- **Color/Intensity**: Magnitude of that frequency at that time

The model processes three such spectrograms (one per component) as a 3-channel image.

### Channel Types

Different seismometer types are used:
- **HH**: High-gain, high-frequency (broadband), 100 Hz sampling
- **BH**: Broadband, high-gain, 20-80 Hz sampling
- **EH**: Extremely high-frequency short-period, 100+ Hz sampling
- **HN**: High-gain accelerometer, 100-200 Hz sampling

Each channel type requires appropriate frequency filtering based on its sensitivity and sampling characteristics.

## Key Innovations

### 1. Latent Space Diffusion

Unlike pixel-space diffusion models (like those used for images), this project performs diffusion in a learned latent space. This approach:

- **Reduces computational cost**: Diffusion on 128D vectors vs. 100,000D images (~800x fewer dimensions)
- **Improves sample quality**: The VAE's latent space has learned structure that's easier to model
- **Enables faster sampling**: Fewer dimensions means faster iterative denoising
- **Regularization**: The VAE's KL term encourages a smooth, well-behaved latent space

### 2. Conditional Generation with Physical Constraints

The model learns meaningful relationships between seismic parameters and waveform characteristics:

**Magnitude Conditioning**: 
- Larger magnitudes → higher amplitudes, longer durations, more low-frequency energy
- The model learns these scaling relationships from data

**Location Conditioning**:
- Distance affects amplitude decay (geometrical spreading)
- Path effects: waves traveling through different geological structures
- The model implicitly learns distance-dependent attenuation

**Station Conditioning**:
- Each station has unique site effects (local geology)
- The embedding layer captures station-specific characteristics

### 3. Multi-Scale Architecture

The convolutional architecture captures seismic features at multiple scales:
- **Early layers**: Fine-grained features (high-frequency content, sharp arrivals)
- **Deep layers**: Coarse features (overall envelope, duration)
- **Skip connections**: Preserve details during reconstruction

### 4. Magnitude-Adaptive Filtering

Preprocessing applies physics-informed filtering:
- Small earthquakes (M < 2): 2-10 Hz (higher frequencies dominate)
- Medium earthquakes (M 2-4): 1-5 Hz  
- Large earthquakes (M ≥ 4): 0.5-2 Hz (lower frequencies, longer periods)

This matches the corner frequency scaling: f_c ∝ 1/M₀^(1/3), where M₀ is seismic moment.

## Project Structure

```
.
├── preprocessing/          # Data acquisition and preprocessing
│   ├── download_waveforms.py          # Downloads from FDSN networks
│   ├── combine_picks.py               # Organizes phase pick catalogs
│   ├── preprocess_waveforms.py        # Applies filtering pipeline
│   ├── create_data_summary.py         # Quality control metrics (SNR, gaps)
│   ├── convert_waveforms_to_png.py    # Visualization
│   └── *.ipynb                        # Interactive data exploration
│
├── ML/autoencoder/        # Stage 1: Representation learning
│   ├── model.py                       # VAE/CVAE architectures
│   ├── train.py                       # Standard VAE training
│   ├── train_cvae.py                  # Conditional VAE training
│   ├── inference.py                   # Model evaluation
│   ├── hyperparameter_search.py       # Automated HPO with Optuna
│   ├── stft_dataset.py                # Dataset loader (on-the-fly STFT)
│   └── stft_dataset_with_metadata.py  # Dataset with conditioning info
│
├── ML/diffusion/          # Stage 2: Generative modeling
│   ├── diffusion_model.py             # Conditional diffusion implementation
│   ├── embed_waveforms.py             # Create latent embeddings from VAE
│   └── generate_samples.py            # Sample generation pipeline
│
└── data/                  # Data storage
    ├── phase_picks/                   # Seismic catalogs (P-wave arrival times)
    ├── waveforms/                     # Raw miniSEED files
    ├── filtered_waveforms/            # Preprocessed data
    └── events/                        # Event metadata
```

## Technical Pipeline

### End-to-End Workflow

```
1. DATA ACQUISITION
   Phase Pick Catalogs (GSE2 format)
   → Contains: event time, location, magnitude, P-wave arrival times
   ↓
   [download_waveforms.py]
   → Downloads 3-component waveforms from FDSN networks
   → Time window: [-60s, +240s] relative to P-wave arrival
   ↓
   Raw Waveforms (miniSEED format)

2. PREPROCESSING
   [preprocess_waveforms.py]
   → Detrend & demean (remove DC offset and linear trends)
   → Taper (cosine taper at edges)
   → Bandpass filter (magnitude and channel-dependent)
   ↓
   [create_data_summary.py]
   → Calculate SNR (signal/noise ratio)
   → Calculate gap ratio (data continuity)
   → Compute spectral characteristics
   ↓
   Filtered, Quality-Controlled Waveforms

3. REPRESENTATION LEARNING (Stage 1)
   [train.py / train_cvae.py]
   → On-the-fly STFT computation
   → Train VAE/CVAE on spectrograms
   → Learn encoder q(z|x) and decoder p(x|z)
   ↓
   Trained VAE Model
   ↓
   [embed_waveforms.py]
   → Encode all waveforms to latent space
   → Save embeddings + metadata
   ↓
   Latent Embeddings Dataset

4. GENERATIVE MODELING (Stage 2)
   [diffusion_model.py]
   → Train conditional diffusion on embeddings
   → Learn p(z|c) where c = (magnitude, location, station)
   ↓
   Trained Diffusion Model

5. SYNTHESIS
   [generate_samples.py]
   → Specify desired parameters (M, lat, lon, depth, station)
   → Diffusion model generates latent z
   → VAE decoder reconstructs spectrogram
   → (Optional) Inverse STFT → time series
   ↓
   Synthetic Seismic Waveforms
```

### Key Processing Details

**STFT Computation:**
- Computed on-the-fly during training (not pre-computed)
- Parameters chosen to balance time-frequency resolution
- 75% overlap ensures smooth reconstruction
- Magnitude values normalized to log scale for better dynamic range

**Data Augmentation:**
- Random time shifts (±5 seconds)
- Amplitude scaling (simulate different distances)
- Component rotation (random horizontal orientations)

**Quality Control Thresholds:**
- SNR > 3.0 (signal at least 3x stronger than noise)
- Gap ratio < 0.1 (less than 10% missing data)
- All three components present and aligned

## Quick Start

### Training the Complete Pipeline

```bash
# 1. Download and preprocess data
cd preprocessing
python download_waveforms.py
python preprocess_waveforms.py
python create_data_summary.py

# 2. Train VAE
cd ../ML/autoencoder
python train.py --model_type vae --latent_dim 128 --beta 1.0 --epochs 100

# 3. Generate embeddings
cd ../diffusion
python embed_waveforms.py --vae_checkpoint ../autoencoder/checkpoints/best_vae.pt

# 4. Train diffusion model
python diffusion_model.py --epochs 200

# 5. Generate synthetic samples
python generate_samples.py \
    --magnitude 4.5 \
    --latitude 39.0 \
    --longitude -120.0 \
    --depth 10.0 \
    --num_samples 100
```

### Requirements

```bash
pip install torch obspy pandas numpy scipy matplotlib tqdm tensorboard optuna
```


## Applications and Use Cases

### 1. Data Augmentation for Machine Learning
- Generate additional training samples for earthquake early warning systems
- Balance datasets across magnitude ranges
- Create samples for underrepresented scenarios

### 2. Seismic Hazard Assessment
- Synthesize ground motions for scenario earthquakes
- Explore variability in waveforms for fixed source parameters
- Supplement sparse observations in certain regions

### 3. Network Planning
- Simulate expected waveforms for proposed station locations
- Test detection algorithms on synthetic data
- Optimize sensor placement

### 4. Scientific Understanding
- Explore learned representations of seismic features
- Understand relationships between source parameters and waveform characteristics
- Identify what features the model considers important (via latent space analysis)

## Implementation Details

### Computational Requirements

**VAE Training:**
- GPU: NVIDIA GPU with 8+ GB VRAM (e.g., RTX 2080, V100)
- Training time: ~4-8 hours for 100 epochs on ~10,000 waveforms
- Batch size: 32-64 depending on GPU memory

**Diffusion Training:**
- GPU: 8+ GB VRAM
- Training time: ~2-4 hours for 200 epochs on latent embeddings
- Much faster than pixel-space diffusion due to low dimensionality

**Inference:**
- Generation: ~1-5 seconds per sample (depends on diffusion steps)
- Can run on CPU if necessary (slower)

### Hyperparameter Considerations

**VAE β Parameter:**
- β = 1.0: Standard VAE, balanced reconstruction/regularization
- β = 0.5-0.8: Better reconstruction quality, less structured latent space
- β = 1.5-4.0: More disentangled latent space, potential quality loss

**Latent Dimension:**
- 64: Very compressed, faster training/inference, may lose details
- 128: Good balance (recommended)
- 256-512: High quality, more expressive, slower

**Diffusion Timesteps:**
- T = 1000: Standard choice, good quality
- T = 500: Faster sampling, slight quality decrease
- Fewer steps during inference: Use DDIM sampling for ~50-100 steps

### Data Characteristics

The project handles real seismic data with inherent challenges:

**Non-Stationary Signals**: Earthquake waveforms are transient events with time-varying frequency content (P-wave, S-wave, surface waves arrive at different times)

**High Dynamic Range**: Amplitudes can vary by orders of magnitude depending on magnitude and distance

**Missing Data**: Real networks have gaps, timing issues, and instrument problems

**Site Effects**: Each station has unique local geological conditions affecting recordings

The preprocessing and model design address these challenges through appropriate filtering, normalization, and quality control.

## License

[Specify your license here]

## Citation

If you use this code in your research, please cite:

```
[Add citation information if applicable]
```

## Contact

[Add contact information]
