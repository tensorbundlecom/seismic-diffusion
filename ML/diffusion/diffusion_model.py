import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import argparse
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import torchaudio.transforms as T

# Use non-interactive backend for matplotlib
matplotlib.use('Agg')

# Add parent directory to path to import VAE model
sys.path.append(str(Path(__file__).parent.parent / "autoencoder"))
from model import VariationalAutoencoder


class EmbeddingDataset(Dataset):
    """Dataset for loading embeddings and conditioning information."""
    
    def __init__(self, embeddings_path, conditioning_path):
        """
        Args:
            embeddings_path: Path to .npy file with embeddings
            conditioning_path: Path to .npz file with conditioning data
        """
        self.embeddings = np.load(embeddings_path)
        conditioning = np.load(conditioning_path)
        
        self.magnitudes = conditioning['magnitudes']
        self.locations = conditioning['locations']
        self.station_indices = conditioning['station_indices']
        
        print(f"Loaded {len(self.embeddings)} samples")
        print(f"  Embeddings shape: {self.embeddings.shape}")
        print(f"  Magnitudes shape: {self.magnitudes.shape}")
        print(f"  Locations shape: {self.locations.shape}")
        print(f"  Station indices shape: {self.station_indices.shape}")
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return {
            'embedding': torch.FloatTensor(self.embeddings[idx]),
            'magnitude': torch.FloatTensor([self.magnitudes[idx]]),
            'location': torch.FloatTensor(self.locations[idx]),
            'station_idx': torch.LongTensor([self.station_indices[idx]])
        }


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for timesteps."""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ConditionalMLP(nn.Module):
    """MLP-based diffusion model with conditioning."""
    
    def __init__(
        self,
        embedding_dim=256,
        time_dim=256,
        hidden_dims=[512, 512, 512],
        num_stations=46,
        station_embed_dim=32,
        dropout=0.1
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.time_dim = time_dim
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Station embedding
        self.station_embed = nn.Embedding(num_stations, station_embed_dim)
        
        # Conditioning dimension: magnitude (1) + location (3) + station_embed (32)
        cond_dim = 1 + 3 + station_embed_dim
        
        # Input dimension: embedding + time + conditioning
        input_dim = embedding_dim + time_dim + cond_dim
        
        # Build MLP
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, embedding_dim))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x, t, magnitude, location, station_idx):
        """
        Args:
            x: (B, embedding_dim) - noisy embeddings
            t: (B,) - timesteps
            magnitude: (B, 1) - earthquake magnitude
            location: (B, 3) - normalized location (lat, lon, depth)
            station_idx: (B, 1) - station index
        
        Returns:
            (B, embedding_dim) - predicted noise
        """
        # Get time embeddings
        t_emb = self.time_mlp(t)  # (B, time_dim)
        
        # Get station embeddings
        station_emb = self.station_embed(station_idx.squeeze(-1))  # (B, station_embed_dim)
        
        # Concatenate all conditioning
        cond = torch.cat([magnitude, location, station_emb], dim=-1)  # (B, cond_dim)
        
        # Concatenate input, time, and conditioning
        h = torch.cat([x, t_emb, cond], dim=-1)  # (B, input_dim)
        
        # Pass through MLP
        noise_pred = self.mlp(h)  # (B, embedding_dim)
        
        return noise_pred


def griffin_lim(magnitude, n_iter=32, nperseg=256, noverlap=248, nfft=256, 
                sampling_rate=100.0, window='hann'):
    """
    Griffin-Lim algorithm to reconstruct phase from magnitude spectrogram using torchaudio.
    
    Args:
        magnitude: Magnitude spectrogram (freq_bins, time_bins) - numpy array or torch tensor
        n_iter: Number of iterations
        nperseg: STFT window length
        noverlap: STFT overlap
        nfft: FFT size
        sampling_rate: Sampling rate
        window: Window type
        
    Returns:
        Reconstructed waveform as numpy array
    """
    # Calculate hop length from nperseg and noverlap
    magnitude = np.power(magnitude, 2)  # Ensure magnitude is in linear scale
    hop_length = nperseg - noverlap
    
    # Convert to torch tensor if numpy array
    if isinstance(magnitude, np.ndarray):
        magnitude_tensor = torch.from_numpy(magnitude).float()
    else:
        magnitude_tensor = magnitude.float()
    
    # For STFT, we need only positive frequencies (one-sided)
    # If magnitude has full spectrum, take only first half
    if magnitude_tensor.shape[0] > nfft // 2 + 1:
        magnitude_tensor = magnitude_tensor[:nfft // 2 + 1, :]
    
    # Create Griffin-Lim transform
    griffin_lim_transform = T.GriffinLim(
        n_fft=nfft,
        n_iter=n_iter,
        win_length=nperseg,
        hop_length=hop_length,
        window_fn=torch.hann_window,
        power=1.0,  # magnitude spectrogram (not power)
        momentum=0.99,
        length=None,
        rand_init=True
    )
    
    # Apply Griffin-Lim algorithm
    waveform_tensor = griffin_lim_transform(magnitude_tensor)
    
    # Convert back to numpy
    waveform = waveform_tensor.numpy()
    
    return waveform


def spectrogram_to_waveform(spectrogram, sampling_rate=100.0, nperseg=256, 
                           noverlap=248, nfft=256, normalize=True,
                           n_iter=32):
    """
    Convert spectrogram back to 1D waveform using Griffin-Lim algorithm (via torchaudio).
    
    Args:
        spectrogram: numpy array of shape (freq_bins, time_bins) - magnitude spectrogram
        sampling_rate: Sampling rate in Hz
        nperseg: Length of each segment for STFT
        noverlap: Number of points to overlap between segments
        nfft: Length of the FFT used
        normalize: Whether the spectrogram is normalized
        n_iter: Number of Griffin-Lim iterations (default: 32)
        
    Returns:
        waveform: 1D numpy array containing the reconstructed signal
    """
    # Reverse normalization if applied
    if normalize:
        # Note: We can't perfectly reverse normalization without the original min/max
        # but we can scale to a reasonable range
        spectrogram = spectrogram.copy()
    
    # Use librosa's Griffin-Lim algorithm to reconstruct waveform from magnitude spectrogram
    waveform = griffin_lim(
        spectrogram,
        n_iter=n_iter,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        sampling_rate=sampling_rate,
        window='hann'
    )
    
    return waveform


def load_vae_decoder(vae_checkpoint_path, device='cuda'):
    """
    Load VAE decoder from checkpoint.
    
    Args:
        vae_checkpoint_path: Path to the VAE checkpoint file
        device: Device to load the model on
        
    Returns:
        vae_model: Loaded VAE model (in eval mode)
    """
    if not os.path.exists(vae_checkpoint_path):
        print(f"Warning: VAE checkpoint not found at {vae_checkpoint_path}")
        return None
    
    print(f"Loading VAE checkpoint from {vae_checkpoint_path}")
    checkpoint = torch.load(vae_checkpoint_path, map_location=device, weights_only=False)
    
    # Get config
    config = checkpoint.get('config', {})
    state_dict = checkpoint['model_state_dict']
    
    # Create VAE model
    vae_model = VariationalAutoencoder(
        in_channels=config.get('in_channels', 3),
        latent_dim=config.get('latent_dim', 256)
    ).to(device)
    
    # Handle dynamic FC layer initialization in VAE encoder
    if 'encoder.fc_mu.weight' in state_dict:
        flattened_dim = state_dict['encoder.fc_mu.weight'].shape[1]
        vae_model.encoder._initialize_fc_layers(flattened_dim)
    
    vae_model.load_state_dict(state_dict)
    vae_model.eval()
    
    print(f"Loaded VAE model (latent_dim={config.get('latent_dim', 256)})")
    
    return vae_model


def generate_and_log_samples(
    model,
    diffusion,
    vae_decoder,
    val_loader,
    writer,
    epoch,
    device,
    num_samples=4,
    station_map=None,
    stft_params=None
):
    """
    Generate sample waveforms and log them to tensorboard.
    
    Args:
        model: Diffusion model
        diffusion: GaussianDiffusion object
        vae_decoder: VAE decoder model (or None to skip generation)
        val_loader: Validation data loader
        writer: TensorBoard writer
        epoch: Current epoch number
        device: Device to generate on
        num_samples: Number of samples to generate
        station_map: Dictionary mapping station indices to names
        stft_params: Dictionary with STFT parameters (sampling_rate, nperseg, noverlap, nfft)
    """
    if vae_decoder is None:
        print("Skipping sample generation (no VAE decoder available)")
        return
    
    # Default STFT parameters
    if stft_params is None:
        stft_params = {
            'sampling_rate': 100.0,
            'nperseg': 256,
            'noverlap': 248,
            'nfft': 256
        }
    
    model.eval()
    
    # Fixed location: (40.5088, 28.8380, 10.0 km depth) at ARMT station
    # Normalized coordinates based on training data bounds:
    # Lat: [39.7500, 41.5263], Lon: [25.8002, 29.9973], Depth: [0.0, 29.7]
    target_lat_norm = 0.427180
    target_lon_norm = 0.723785
    target_depth_norm = 0.336700
    target_station_idx = 1  # ARMT
    
    # Create fixed location and station for all samples
    location = torch.FloatTensor([[target_lat_norm, target_lon_norm, target_depth_norm]] * num_samples).to(device)
    station_idx = torch.LongTensor([[target_station_idx]] * num_samples).to(device)

    # Use range of magnitudes (1.0 to 4.5) for better visualization
    magnitude = torch.FloatTensor([1.0 + 3.5 * (i / (num_samples - 1)) for i in range(num_samples)]).unsqueeze(-1).to(device)
    
    # Get station name for display
    station_name = station_map.get(target_station_idx, f'Station_{target_station_idx}') if station_map else f'Station_{target_station_idx}'
    
    print(f"  Generating samples for:")
    print(f"    Location: (40.5088°, 28.8380°, 10.0 km depth)")
    print(f"    Station: {station_name} (index {target_station_idx})")
    print(f"    Magnitudes: {[f'{m.item():.1f}' for m in magnitude.squeeze()]}")
    
    # Generate embeddings using diffusion model
    with torch.no_grad():
        print(f"Generating {num_samples} samples for epoch {epoch}...")
        embeddings = diffusion.sample(
            model=model,
            shape=(num_samples, model.embedding_dim),
            magnitude=magnitude,
            location=location,
            station_idx=station_idx
        )
        
        # Decode embeddings to waveforms using VAE decoder
        waveforms = vae_decoder.decode(embeddings)
    
    # Create figure with spectrograms
    fig_spec, axes_spec = plt.subplots(num_samples, 3, figsize=(15, 4 * num_samples))
    
    if num_samples == 1:
        axes_spec = axes_spec.reshape(1, -1)
    
    channel_names = ['Z', 'N', 'E']  # Vertical, North, East components
    
    # Create figure with 1D waveforms
    fig_wave, axes_wave = plt.subplots(num_samples, 3, figsize=(15, 3 * num_samples))
    
    if num_samples == 1:
        axes_wave = axes_wave.reshape(1, -1)
    
    for i in range(num_samples):
        waveform_spec = waveforms[i].cpu().numpy()  # (C, H, W) - spectrogram
        
        # Get metadata for title
        mag = magnitude[i].item()
        loc = location[i].cpu().numpy()
        sta_idx = station_idx[i].item()
        station_name = station_map.get(sta_idx, f'Station_{sta_idx}') if station_map else f'Station_{sta_idx}'
        
        # Plot spectrograms
        for j, channel_name in enumerate(channel_names):
            ax = axes_spec[i, j]
            im = ax.imshow(waveform_spec[j], aspect='auto', cmap='seismic', 
                          interpolation='nearest', origin='lower')
            ax.set_title(f'Sample {i+1} - Channel {channel_name}')
            ax.set_ylabel('Frequency')
            ax.set_xlabel('Time')
            fig_spec.colorbar(im, ax=ax)
        
        # Add metadata as row title with actual coordinates
        title = f"Mag: {mag:.1f} | {station_name} | (40.51°, 28.84°, 10km)"
        axes_spec[i, 0].set_ylabel(f'{title}\nFrequency', fontsize=9)
        
        # Convert spectrograms to 1D waveforms and plot
        for j, channel_name in enumerate(channel_names):
            # Convert spectrogram to 1D waveform using Griffin-Lim
            if i == 0 and j == 0:
                spec_shape = waveform_spec[j].shape
                hop_length = stft_params['nperseg'] - stft_params['noverlap']
                expected_samples = (spec_shape[1] - 1) * hop_length + stft_params['nperseg']
                expected_duration = expected_samples / stft_params['sampling_rate']
                print(f"  Spectrogram shape: {spec_shape}")
                print(f"  Expected waveform: {expected_samples} samples ({expected_duration:.1f} seconds)")
                print(f"  Hop length: {hop_length}, nperseg: {stft_params['nperseg']}")
            
            waveform_1d = spectrogram_to_waveform(
                waveform_spec[j],
                sampling_rate=stft_params['sampling_rate'],
                nperseg=stft_params['nperseg'],
                noverlap=stft_params['noverlap'],
                nfft=stft_params['nfft'],
                normalize=True
            )
            
            if i == 0 and j == 0:
                duration = len(waveform_1d) / stft_params['sampling_rate']
                print(f"  Actual waveform: {len(waveform_1d)} samples ({duration:.1f} seconds)")
            
            # Plot 1D waveform
            ax = axes_wave[i, j]
            time_axis = np.arange(len(waveform_1d)) / stft_params['sampling_rate']
            ax.plot(time_axis, waveform_1d, 'b-', linewidth=0.5)
            ax.set_title(f'Sample {i+1} - Channel {channel_name} ({len(waveform_1d)/stft_params["sampling_rate"]:.0f}s)')
            ax.set_ylabel('Amplitude')
            ax.set_xlabel('Time (s)')
            ax.grid(True, alpha=0.3)
        
        # Add metadata as row title for waveforms
        axes_wave[i, 0].set_ylabel(f'{title}\nAmplitude', fontsize=9)
    
    plt.tight_layout()
    
    # Log spectrograms to tensorboard
    writer.add_figure('generated_samples/spectrograms', fig_spec, epoch)
    plt.close(fig_spec)
    
    # Log 1D waveforms to tensorboard
    fig_wave.tight_layout()
    writer.add_figure('generated_samples/waveforms_1d', fig_wave, epoch)
    plt.close(fig_wave)
    
    print(f"Logged {num_samples} generated samples (spectrograms + 1D waveforms) to tensorboard")


class GaussianDiffusion:
    """Gaussian Diffusion process (DDPM)."""
    
    def __init__(
        self,
        timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        device='cuda'
    ):
        self.timesteps = timesteps
        self.device = device
        
        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        
    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion process: q(x_t | x_0)"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None]
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, model, x_start, t, magnitude, location, station_idx, noise=None):
        """Compute loss for training."""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Forward process
        x_noisy = self.q_sample(x_start, t, noise)
        
        # Predict noise
        predicted_noise = model(x_noisy, t, magnitude, location, station_idx)
        
        # MSE loss
        loss = F.mse_loss(predicted_noise, noise)
        
        return loss
    
    @torch.no_grad()
    def p_sample(self, model, x, t, magnitude, location, station_idx):
        """Single reverse diffusion step: p(x_{t-1} | x_t)"""
        betas_t = self.betas[t][:, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None]
        sqrt_recip_alphas_t = torch.sqrt(1.0 / self.alphas[t])[:, None]
        
        # Predict noise
        predicted_noise = model(x, t, magnitude, location, station_idx)
        
        # Mean of p(x_{t-1} | x_t)
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t[0] == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t][:, None]
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def sample(self, model, shape, magnitude, location, station_idx):
        """Generate samples from random noise."""
        batch_size = shape[0]
        device = self.device
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        # Reverse diffusion
        for i in tqdm(reversed(range(self.timesteps)), desc='Sampling', total=self.timesteps):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t, magnitude, location, station_idx)
        
        return x


def train(
    model,
    diffusion,
    train_loader,
    val_loader,
    num_epochs,
    lr,
    device,
    log_dir,
    checkpoint_dir,
    save_interval=10,
    vae_decoder=None,
    station_map=None,
    sample_interval=5,
    stft_params=None
):
    """Training loop."""
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    
    writer = SummaryWriter(log_dir)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_losses = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch in pbar:
            x_start = batch['embedding'].to(device)
            magnitude = batch['magnitude'].to(device)
            location = batch['location'].to(device)
            station_idx = batch['station_idx'].to(device)
            
            # Sample random timesteps
            batch_size = x_start.shape[0]
            t = torch.randint(0, diffusion.timesteps, (batch_size,), device=device).long()
            
            # Compute loss
            loss = diffusion.p_losses(model, x_start, t, magnitude, location, station_idx)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            pbar.set_postfix({'loss': loss.item()})
            
            writer.add_scalar('train/loss', loss.item(), global_step)
            global_step += 1
        
        avg_train_loss = np.mean(train_losses)
        
        # Validation
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                x_start = batch['embedding'].to(device)
                magnitude = batch['magnitude'].to(device)
                location = batch['location'].to(device)
                station_idx = batch['station_idx'].to(device)
                
                batch_size = x_start.shape[0]
                t = torch.randint(0, diffusion.timesteps, (batch_size,), device=device).long()
                
                loss = diffusion.p_losses(model, x_start, t, magnitude, location, station_idx)
                val_losses.append(loss.item())
        
        avg_val_loss = np.mean(val_losses)
        
        # Logging
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.6f}')
        print(f'  Val Loss: {avg_val_loss:.6f}')
        print(f'  LR: {scheduler.get_last_lr()[0]:.6f}')
        
        writer.add_scalar('epoch/train_loss', avg_train_loss, epoch)
        writer.add_scalar('epoch/val_loss', avg_val_loss, epoch)
        writer.add_scalar('epoch/lr', scheduler.get_last_lr()[0], epoch)
        
        scheduler.step()
        
        # Generate and log sample images
        if (epoch + 1) % sample_interval == 0:
            generate_and_log_samples(
                model=model,
                diffusion=diffusion,
                vae_decoder=vae_decoder,
                val_loader=val_loader,
                writer=writer,
                epoch=epoch,
                device=device,
                num_samples=4,
                station_map=station_map,
                stft_params=stft_params
            )
        
        # Save checkpoint
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_path)
            print(f'Saved checkpoint: {checkpoint_path}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, best_model_path)
            print(f'Saved best model: {best_model_path}')
    
    writer.close()
    print('Training completed!')


def main():
    parser = argparse.ArgumentParser(description='Train diffusion model on seismic embeddings')
    parser.add_argument('--embeddings_dir', type=str, default='embeddings',
                        help='Directory containing embeddings and conditioning')
    parser.add_argument('--embedding_dim', type=int, default=256,
                        help='Dimension of embeddings')
    parser.add_argument('--time_dim', type=int, default=256,
                        help='Dimension of time embeddings')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[512, 512, 512],
                        help='Hidden dimensions for MLP')
    parser.add_argument('--num_stations', type=int, default=46,
                        help='Number of stations')
    parser.add_argument('--station_embed_dim', type=int, default=32,
                        help='Dimension of station embeddings')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate')
    parser.add_argument('--timesteps', type=int, default=100,
                        help='Number of diffusion timesteps')
    parser.add_argument('--beta_start', type=float, default=1e-2,
                        help='Starting beta value')
    parser.add_argument('--beta_end', type=float, default=0.1,
                        help='Ending beta value')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--sample_interval', type=int, default=5,
                        help='Generate and log samples every N epochs')
    parser.add_argument('--vae_checkpoint', type=str, default=None,
                        help='Path to VAE checkpoint for sample generation')
    parser.add_argument('--metadata_path', type=str, 
                        default='embeddings/train_metadata.json',
                        help='Path to metadata JSON for station mapping')
    parser.add_argument('--stft_config', type=str,
                        default='embeddings/train_config.json',
                        help='Path to config JSON with STFT parameters')
    parser.add_argument('--sampling_rate', type=float, default=100.0,
                        help='Sampling rate for inverse STFT (Hz)')
    parser.add_argument('--nperseg', type=int, default=16,
                        help='STFT segment length')
    parser.add_argument('--noverlap', type=int, default=12,
                        help='STFT overlap')
    parser.add_argument('--nfft', type=int, default=32,
                        help='STFT FFT length')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create datasets
    train_dataset = EmbeddingDataset(
        os.path.join(args.embeddings_dir, 'train_embeddings.npy'),
        os.path.join(args.embeddings_dir, 'train_conditioning.npz')
    )
    
    val_dataset = EmbeddingDataset(
        os.path.join(args.embeddings_dir, 'val_embeddings.npy'),
        os.path.join(args.embeddings_dir, 'val_conditioning.npz')
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    model = ConditionalMLP(
        embedding_dim=args.embedding_dim,
        time_dim=args.time_dim,
        hidden_dims=args.hidden_dims,
        num_stations=args.num_stations,
        station_embed_dim=args.station_embed_dim,
        dropout=args.dropout
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model parameters: {num_params:,}')
    
    # Create diffusion
    diffusion = GaussianDiffusion(
        timesteps=args.timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        device=device
    )
    
    # Load VAE decoder for sample generation (optional)
    vae_decoder = None
    if args.vae_checkpoint:
        try:
            vae_decoder = load_vae_decoder(args.vae_checkpoint, device)
        except Exception as e:
            print(f"Warning: Failed to load VAE decoder: {e}")
            print("Sample generation will be skipped during training")
    else:
        print("No VAE checkpoint provided. Sample generation will be skipped during training")
    
    # Load station mapping for better sample visualization (optional)
    station_map = None
    if os.path.exists(args.metadata_path):
        try:
            with open(args.metadata_path, 'r') as f:
                metadata = json.load(f)
            station_map = {}
            for sample in metadata:
                station_map[sample['station_idx']] = sample['station_name']
            print(f"Loaded {len(station_map)} station mappings")
        except Exception as e:
            print(f"Warning: Failed to load station mapping: {e}")
    else:
        print(f"Metadata file not found at {args.metadata_path}. Station indices will be used in samples.")
    
    # Load STFT parameters for inverse STFT (optional)
    stft_params = None
    if os.path.exists(args.stft_config):
        try:
            with open(args.stft_config, 'r') as f:
                config = json.load(f)
            vae_config = config.get('vae_config', {})
            stft_params = {
                'sampling_rate': vae_config.get('sampling_rate', args.sampling_rate),
                'nperseg': vae_config.get('nperseg', args.nperseg),
                'noverlap': vae_config.get('noverlap', args.noverlap),
                'nfft': vae_config.get('nfft', args.nfft)
            }
            print(f"Loaded STFT parameters from config:")
            print(f"  Sampling rate: {stft_params['sampling_rate']} Hz")
            print(f"  nperseg: {stft_params['nperseg']}")
            print(f"  noverlap: {stft_params['noverlap']}")
            print(f"  nfft: {stft_params['nfft']}")
        except Exception as e:
            print(f"Warning: Failed to load STFT config: {e}")
            print("Using default/command-line STFT parameters")
    
    if stft_params is None:
        stft_params = {
            'sampling_rate': args.sampling_rate,
            'nperseg': args.nperseg,
            'noverlap': args.noverlap,
            'nfft': args.nfft
        }
        print("Using STFT parameters from command-line arguments")
    
    # Create output directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join('logs', timestamp)
    checkpoint_dir = os.path.join('checkpoints', timestamp)
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save config
    config = vars(args)
    config['num_params'] = num_params
    config['timestamp'] = timestamp
    
    with open(os.path.join(checkpoint_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f'Log directory: {log_dir}')
    print(f'Checkpoint directory: {checkpoint_dir}')
    
    # Train
    train(
        model=model,
        diffusion=diffusion,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        lr=args.lr,
        device=device,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        save_interval=args.save_interval,
        vae_decoder=vae_decoder,
        station_map=station_map,
        sample_interval=args.sample_interval,
        stft_params=stft_params
    )


if __name__ == '__main__':
    main()
