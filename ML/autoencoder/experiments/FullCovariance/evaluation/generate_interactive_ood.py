import torch
import numpy as np
import pandas as pd
import obspy
from pathlib import Path
import scipy.signal as signal
import os
import sys
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add paths for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../')))
from ..core.stft_dataset_fixed import SeismicSTFTDatasetWithMetadata
from ..core.model_baseline_fixed import ConditionalVariationalAutoencoder, CVAEDecoder
from ..core.model_full_cov import FullCovCVAE

def griffin_lim(mag_spectrogram, n_fft=256, hop_length=64, iterations=30):
    mag_spectrogram = mag_spectrogram.astype(np.complex64)
    rng = np.random.default_rng()
    phase = np.exp(2j * np.pi * rng.random(mag_spectrogram.shape))
    
    for _ in range(iterations):
        stft = mag_spectrogram * phase
        waveform = signal.istft(stft, nfft=n_fft)[1]
        _, _, next_stft = signal.stft(waveform, nfft=n_fft)
        
        # Match shapes if necessary
        if next_stft.shape != mag_spectrogram.shape:
             # Resize next_stft to match mag_spectrogram
             h, w = mag_spectrogram.shape
             next_stft = next_stft[:h, :w]
             
        phase = np.exp(1j * np.angle(next_stft))
    
    stft = mag_spectrogram * phase
    return signal.istft(stft, nfft=n_fft)[1]

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = os.path.join(os.path.dirname(__file__), "..", "results", "interactive_ood")
    os.makedirs(output_dir, exist_ok=True)
    
    # Weights (relative to root)
    baseline_chk = "ML/autoencoder/checkpoints/checkpoints_cvae/20260207_163500/best_model.pt"
    full_cov_chk = "ML/autoencoder/experiments/FullCovariance/checkpoints/checkpoints_full_cov/koeri_catalog/full_cov_cvae_best.pt"
    
    print("Loading Models...")
    # Baseline
    baseline_checkpoint = torch.load(baseline_chk, map_location=device)
    baseline_model = ConditionalVariationalAutoencoder(in_channels=3, latent_dim=128, num_stations=125).to(device)
    # Check if it's wrapped
    if 'model_state_dict' in baseline_checkpoint:
        baseline_model.load_state_dict(baseline_checkpoint['model_state_dict'])
    else:
        baseline_model.load_state_dict(baseline_checkpoint)
    baseline_model.eval()
    
    # Full Cov
    experimental_checkpoint = torch.load(full_cov_chk, map_location=device)
    full_cov_model = FullCovCVAE(in_channels=3, latent_dim=128, num_stations=125).to(device)
    full_cov_model.load_state_dict(experimental_checkpoint['model_state_dict'], strict=False)
    full_cov_model.eval()
    
    # Dataset
    test_dataset = SeismicSTFTDatasetWithMetadata(
        data_dir="data/ood_waveforms/filtered",
        event_file="data/events/ood_catalog.txt",
        magnitude_col="ML"
    )
    
    ood_events = ["OOD_1", "OOD_2", "OOD_3", "OOD_4", "OOD_5"]
    expected_shape = (129, 111)
    
    for event_id in ood_events:
        print(f"Processing {event_id}...")
        event_indices = [i for i in range(len(test_dataset)) if test_dataset[i][4]['event_id'] == event_id]
        if not event_indices:
            continue
            
        fig = make_subplots(
            rows=len(event_indices), cols=1,
            subplot_titles=[f"Station: {test_dataset[idx][4]['station_name']}" for idx in event_indices],
            vertical_spacing=0.05
        )
        
        for i, idx in enumerate(event_indices):
            spec, mag, loc, station_idx, meta = test_dataset[idx]
            station_name = meta['station_name']
            
            # Prepare inputs
            spec_in = spec.unsqueeze(0)
            if spec_in.shape[2:] != expected_shape:
                spec_in = torch.nn.functional.interpolate(spec_in, size=expected_shape, mode='bilinear', align_corners=False)
            
            spec_in = spec_in.to(device)
            mag_in = mag.unsqueeze(0).to(device)
            loc_in = loc.unsqueeze(0).to(device)
            stat_in = torch.tensor([station_idx]).to(device)
            
            with torch.no_grad():
                recon_base, _, _ = baseline_model(spec_in, mag_in, loc_in, stat_in)
                recon_full, _, _ = full_cov_model(spec_in, mag_in, loc_in, stat_in)
            
            # Reconstruct Z-channel (index 2)
            z_orig = griffin_lim(spec_in[0, 2].cpu().numpy())
            z_base = griffin_lim(recon_base[0, 2].cpu().numpy())
            z_full = griffin_lim(recon_full[0, 2].cpu().numpy())
            
            time = np.linspace(0, 70, len(z_orig))
            
            # Add traces (linked via legendgroup)
            fig.add_trace(go.Scatter(
                x=time, y=z_orig, 
                name="Original", 
                legendgroup="original", 
                showlegend=(i == 0), 
                line=dict(color='black', width=1)
            ), row=i+1, col=1)
            
            fig.add_trace(go.Scatter(
                x=time, y=z_base, 
                name="Baseline CVAE", 
                legendgroup="baseline", 
                showlegend=(i == 0), 
                line=dict(color='blue', width=1)
            ), row=i+1, col=1)
            
            fig.add_trace(go.Scatter(
                x=time, y=z_full, 
                name="Full Covariance", 
                legendgroup="full_cov", 
                showlegend=(i == 0), 
                line=dict(color='red', width=1)
            ), row=i+1, col=1)
            
            fig.update_yaxes(title_text="Amplitude", row=i+1, col=1, range=[-15, 15])
            
        fig.update_layout(
            height=350 * len(event_indices),
            title_text=f"OOD Event Reconstruction: {event_id} (Interactive & Linked Legend)",
            legend=dict(groupclick="toggleitem"), # Toggle all in group
            showlegend=True
        )
        
        html_path = os.path.join(output_dir, f"interactive_{event_id}.html")
        fig.write_html(html_path)
        print(f"Saved: {html_path}")

if __name__ == "__main__":
    main()
