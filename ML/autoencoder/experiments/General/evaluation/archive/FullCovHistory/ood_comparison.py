import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
from pathlib import Path
from obspy import read
import scipy.signal

# Add paths for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../')))
from ..core.stft_dataset_fixed import SeismicSTFTDatasetWithMetadata
from ..core.model_baseline_fixed import ConditionalVariationalAutoencoder
from ..core.model_full_cov import FullCovCVAE

def griffin_lim(mag_stft, n_iter=64, n_fft=256, hop_length=64, log_scale=True):
    """Simple Griffin-Lim for phase recovery with inverse transforms."""
    spec = mag_stft.copy()
    
    # Undo log scaling to restore dynamic range
    if log_scale:
        spec = np.expm1(spec)
        
    noverlap = n_fft - hop_length
    phase = np.random.randn(*spec.shape)
    stft = spec * np.exp(1j * phase)
    
    for _ in range(n_iter):
        waveform = scipy.signal.istft(stft, nfft=n_fft, noverlap=noverlap)[1]
        _, _, new_stft = scipy.signal.stft(waveform, nfft=n_fft, noverlap=noverlap)
        # Handle shape mismatch if any
        new_stft = new_stft[:, :spec.shape[1]]
        phase = np.angle(new_stft)
        stft = spec * np.exp(1j * phase)
        
    return scipy.signal.istft(stft, nfft=n_fft, noverlap=noverlap)[1]

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Weights (relative to root)
    baseline_chk = "ML/autoencoder/checkpoints/checkpoints_cvae/20260207_163500/best_model.pt"
    full_cov_chk = "ML/autoencoder/experiments/FullCovariance/checkpoints/checkpoints_full_cov/koeri_catalog/full_cov_cvae_best.pt"
    
    print("Loading Baseline Model...")
    baseline_checkpoint = torch.load(baseline_chk, map_location=device)
    baseline_model = ConditionalVariationalAutoencoder(in_channels=3, latent_dim=128, num_stations=125).to(device)
    baseline_model.load_state_dict(baseline_checkpoint['model_state_dict'], strict=False)
    baseline_model.eval()

    print("Loading Full Covariance Model...")
    experimental_checkpoint = torch.load(full_cov_chk, map_location=device)
    experimental_model = FullCovCVAE(in_channels=3, latent_dim=128, num_stations=125).to(device)
    experimental_model.load_state_dict(experimental_checkpoint['model_state_dict'], strict=False)
    experimental_model.eval()

    # Create OOD dataset dummy to get the same station mapping
    # Note: We use the SAME station mapping as training by passing any dataset that was used for training
    # or manually ensuring the mapping is consistent. 
    # For simplicity, we'll use the OOD dataset but we MUST ensure station IDs match.
    # In reality, the station_idx is fixed during training.
    
    test_dataset = SeismicSTFTDatasetWithMetadata(
        data_dir="data/ood_waveforms/filtered",
        event_file="data/events/ood_catalog.txt",
        magnitude_col="ML"
    )
    
    # Pick one sample per OOD event
    ood_events = ["OOD_1", "OOD_2", "OOD_3", "OOD_4", "OOD_5"]
    expected_shape = (129, 111)
    output_dir = os.path.join(os.path.dirname(__file__), "..", "results", "ood_results")
    os.makedirs(output_dir, exist_ok=True)
    
    selected_indices = []
    for event_id in ood_events:
        event_samples = [i for i in range(len(test_dataset)) if test_dataset[i][4]['event_id'] == event_id]
        if event_samples:
            hh_samples = [i for i in event_samples if "HH" in test_dataset[i][4].get('channel', '') or "HH" in test_dataset.file_paths[i].name]
            if hh_samples:
                selected_indices.append(hh_samples[0])
            else:
                selected_indices.append(event_samples[0])

    print(f"Generating individual plots for {len(selected_indices)} unique events.")
    
    for i, idx in enumerate(selected_indices):
        spec, mag, loc, station_idx, meta = test_dataset[idx]
        event_label = meta['event_id']
        
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
            recon_full, _, _ = experimental_model(spec_in, mag_in, loc_in, stat_in)

        # Reconstruct Z-channel (Vertical) - index 2
        spec_orig = spec_in[0, 2].cpu().numpy()
        spec_base = recon_base[0, 2].cpu().numpy()
        spec_full = recon_full[0, 2].cpu().numpy()

        wav_orig = griffin_lim(spec_orig)
        wav_base = griffin_lim(spec_base)
        wav_full = griffin_lim(spec_full)

        time = np.linspace(0, 70, len(wav_orig))
        
        # Create a single figure for this event
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Original
        axes[0].plot(time, wav_orig, color='black', alpha=0.8, lw=1)
        axes[0].set_title(f"Original: {event_label} (ML:{meta['magnitude']})\nStation: {meta['station_name']}")
        axes[0].set_ylabel("Amplitude")
        
        # Baseline
        axes[1].plot(time, wav_base, color='blue', alpha=0.8, lw=1)
        axes[1].set_title("Baseline CVAE Recon")
        
        # Full Cov
        axes[2].plot(time, wav_full, color='red', alpha=0.8, lw=1)
        axes[2].set_title("Full Covariance Recon")

        for ax in axes:
            ax.grid(alpha=0.3)
            ax.set_ylim(-15, 15)
            ax.set_xlabel("Time (s)")

        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"comparison_{event_label}.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved: {plot_path}")

    print("Individual OOD plots saved in outputs/ood_results/")

if __name__ == "__main__":
    main()
