import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.signal as signal
import os
import sys

# Add paths for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../')))
from ..core.stft_dataset_fixed import SeismicSTFTDatasetWithMetadata
from ..core.model_baseline_fixed import ConditionalVariationalAutoencoder
from ..core.model_full_cov import FullCovCVAE

def griffin_lim(mag_spectrogram, n_fft=256, n_perseg=256, noverlap=192, iterations=50):
    mag_spectrogram = mag_spectrogram.astype(np.complex64)
    rng = np.random.default_rng()
    phase = np.exp(2j * np.pi * rng.random(mag_spectrogram.shape))
    print(f"Debug: mag_spectrogram shape: {mag_spectrogram.shape}, phase shape: {phase.shape}")
    for _ in range(iterations):
        stft = mag_spectrogram * phase
        waveform = signal.istft(stft, nfft=n_fft, nperseg=n_perseg, noverlap=noverlap)[1]
        _, _, next_stft = signal.stft(waveform, nfft=n_fft, nperseg=n_perseg, noverlap=noverlap)
        print(f"Debug: next_stft shape: {next_stft.shape}")
        if next_stft.shape != mag_spectrogram.shape:
             # Resize to match mag_spectrogram exactly
             h, w = mag_spectrogram.shape
             next_stft = next_stft[:h, :w]
             if next_stft.shape[1] < w:
                 pad_w = w - next_stft.shape[1]
                 next_stft = np.pad(next_stft, ((0,0), (0, pad_w)))
        phase = np.exp(1j * np.angle(next_stft))
    return signal.istft(mag_spectrogram * phase, nfft=n_fft, nperseg=n_perseg, noverlap=noverlap)[1]

def normalize_location(lat, lon, depth):
    # Based on previous catalog analysis
    lat_min, lat_max = 40.0067, 41.3648
    lon_min, lon_max = 26.017, 29.5157
    depth_min, depth_max = 0.0, 24.351
    
    n_lat = (lat - lat_min) / (lat_max - lat_min)
    n_lon = (lon - lon_min) / (lon_max - lon_min)
    n_depth = (depth - depth_min) / (depth_max - depth_min)
    
    return torch.tensor([n_lat, n_lon, n_depth], dtype=torch.float32)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = os.path.join(os.path.dirname(__file__), "..", "results", "synthetic_scenarios")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Dataset for Station Mapping
    ds = SeismicSTFTDatasetWithMetadata(
        data_dir="data/filtered_waveforms",
        event_file="data/events/koeri_catalog.txt",
        channels=['HH']
    )
    
    # 2. Load Models
    print("Loading Models for Simulation...")
    expected_shape = (129, 111)
    # Weights (relative to root)
    baseline_chk = "ML/autoencoder/checkpoints/checkpoints_cvae/20260207_163500/best_model.pt"
    full_cov_chk = "ML/autoencoder/experiments/FullCovariance/checkpoints/checkpoints_full_cov/koeri_catalog/full_cov_cvae_best.pt"
    
    baseline_checkpoint = torch.load(baseline_chk, map_location=device)
    baseline_model = ConditionalVariationalAutoencoder(in_channels=3, latent_dim=128, num_stations=125).to(device)
    baseline_model.load_state_dict(baseline_checkpoint['model_state_dict'] if 'model_state_dict' in baseline_checkpoint else baseline_checkpoint)
    baseline_model.eval()
    
    experimental_checkpoint = torch.load(full_cov_chk, map_location=device)
    full_cov_model = FullCovCVAE(in_channels=3, latent_dim=128, num_stations=125).to(device)
    full_cov_model.load_state_dict(experimental_checkpoint['model_state_dict'], strict=False)
    full_cov_model.eval()
    
    # 3. Define Scenario: Adalar M5.0
    # Lat: 40.85, Lon: 29.00, Depth: 10.0
    mag = torch.tensor([5.0], dtype=torch.float32).to(device)
    loc = normalize_location(40.85, 29.00, 10.0).unsqueeze(0).to(device)
    
    target_stations = ["ADVT", "EDC", "YLV"] # Princes' Islands, Erdek, Yalova
    
    for station_name in target_stations:
        if station_name not in ds.station_to_idx:
            print(f"Station {station_name} not found in training mapping, skipping.")
            continue
            
        print(f"Generating synthetic data for station: {station_name}")
        stat_idx = torch.tensor([ds.station_to_idx[station_name]], dtype=torch.long).to(device)
        
        # Sample latent variable z ~ N(0, I)
        z = torch.randn(1, 128).to(device)
        
        with torch.no_grad():
            # Baseline Decode
            s_base = baseline_model.decode(z, mag, loc, stat_idx)
            # Full Cov Decode
            s_full = full_cov_model.decoder(z, mag, loc, stat_idx)
            
            # Interpolate to expected training shape for griffin_lim compatibility
            if s_base.shape[2:] != expected_shape:
                s_base = torch.nn.functional.interpolate(s_base, size=expected_shape, mode='bilinear', align_corners=False)
            if s_full.shape[2:] != expected_shape:
                s_full = torch.nn.functional.interpolate(s_full, size=expected_shape, mode='bilinear', align_corners=False)
            
        # Extract Z-channel (index 2)
        s_base_z = s_base[0, 2].cpu().numpy()
        s_full_z = s_full[0, 2].cpu().numpy()
        
        # Inverse STFT (Waveform)
        w_base = griffin_lim(s_base_z)
        w_full = griffin_lim(s_full_z)
        time = np.linspace(0, 70, len(w_base))
        
        # --- Visualization ---
        # 1. Static Plot (STFT + Wave)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Spectrograms
        axes[0, 0].imshow(s_base_z, aspect='auto', origin='lower', cmap='viridis')
        axes[0, 0].set_title(f"Baseline STFT - {station_name}")
        
        axes[0, 1].imshow(s_full_z, aspect='auto', origin='lower', cmap='viridis')
        axes[0, 1].set_title(f"Full Cov STFT - {station_name}")
        
        # Waveforms
        axes[1, 0].plot(time, w_base, color='blue', alpha=0.8)
        axes[1, 0].set_title(f"Baseline Waveform - {station_name}")
        axes[1, 0].set_ylim(-15, 15)
        
        axes[1, 1].plot(time, w_full, color='red', alpha=0.8)
        axes[1, 1].set_title(f"Full Cov Waveform - {station_name}")
        axes[1, 1].set_ylim(-15, 15)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"sim_M50_{station_name}.png"))
        plt.close()
        
        # 2. Interactive Plotly
        fig_inter = make_subplots(rows=1, cols=1)
        fig_inter.add_trace(go.Scatter(x=time, y=w_base, name="Baseline CVAE", line=dict(color='blue')))
        fig_inter.add_trace(go.Scatter(x=time, y=w_full, name="Full Covariance", line=dict(color='red')))
        fig_inter.update_layout(title=f"Synthetic M5.0 Princes' Islands - Station: {station_name}")
        fig_inter.write_html(os.path.join(output_dir, f"interactive_M50_{station_name}.html"))
        
    print(f"Simulation completed. Results in {output_dir}")

if __name__ == "__main__":
    main()
