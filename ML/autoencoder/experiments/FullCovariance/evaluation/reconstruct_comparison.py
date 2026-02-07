import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
from scipy.signal import istft
import librosa

# Add necessary paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'ML')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'ML/autoencoder')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'ML/autoencoder/experiments/full_cov_cvae')))

from autoencoder.stft_dataset_with_metadata import SeismicSTFTDatasetWithMetadata
from autoencoder.model import ConditionalVariationalAutoencoder
from ML.autoencoder.experiments.full_cov_cvae.model_full_cov import FullCovCVAE

def reconstruct_signal(magnitude_spec, nperseg=256, noverlap=192, n_iter=32):
    """
    Reconstruct signal from magnitude spectrogram using Griffin-Lim.
    magnitude_spec: (freq_bins, time_bins)
    """
    # Griffin-Lim implementation via librosa
    # Use the same STFT parameters as in dataset
    y_inv = librosa.griffinlim(
        magnitude_spec, 
        n_iter=n_iter, 
        hop_length=nperseg - noverlap, 
        win_length=nperseg,
        window='hann',
        center=True
    )
    return y_inv

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Paths (adjust if necessary)
    baseline_chk = "checkpoints_cvae/20260207_145312/best_model.pt"
    # The full cov save path will be something like:
    full_cov_chk = "checkpoints_full_cov/koeri_catalog/full_cov_cvae_best.pt"
    
    if not os.path.exists(full_cov_chk):
        print(f"Waiting for {full_cov_chk} to be created...")
        return

    # 1. Load Data
    data_dir = "data/filtered_waveforms"
    event_file = "data/events/koeri_catalog.txt"
    dataset = SeismicSTFTDatasetWithMetadata(data_dir=data_dir, event_file=event_file, channels=['HH'], magnitude_col='ML', normalize=True, log_scale=True)
    num_stations = len(dataset.station_names)
    
    # 2. Load Models
    print("Loading Baseline Model...")
    baseline_model = ConditionalVariationalAutoencoder(in_channels=3, latent_dim=128, num_stations=num_stations).to(device)
    # Trigger dynamic layer initialization
    with torch.no_grad():
        dummy_in = torch.randn(1, 3, 129, 111).to(device)
        dummy_mag = torch.zeros(1).to(device)
        dummy_loc = torch.zeros(1, 3).to(device)
        dummy_stat = torch.zeros(1, dtype=torch.long).to(device)
        baseline_model(dummy_in, dummy_mag, dummy_loc, dummy_stat)
    
    baseline_st = torch.load(baseline_chk, map_location=device)
    baseline_model.load_state_dict(baseline_st['model_state_dict'])
    baseline_model.eval()
    
    print("Loading Full Covariance Model...")
    full_cov_model = FullCovCVAE(in_channels=3, latent_dim=128, num_stations=num_stations).to(device)
    # Trigger dynamic layer initialization
    with torch.no_grad():
        full_cov_model(dummy_in, dummy_mag, dummy_loc, dummy_stat)
        
    full_cov_st = torch.load(full_cov_chk, map_location=device)
    full_cov_model.load_state_dict(full_cov_st['model_state_dict'])
    full_cov_model.eval()
    
    # 3. Select Samples (Indices for different magnitudes)
    # Mag 5.1: 11774951, Mag 4.2: 11912391, Mag 3.0: 11796360
    selected_indices = [5, 0, 3] # Approximations based on sorted catalog if applicable, or search
    
    # Better: Search for specific IDs in dataset
    target_ids = ['11774951', '11912391', '11796360']
    test_indices = []
    for i, file_path in enumerate(dataset.file_paths):
        if any(tid in file_path.name for tid in target_ids):
            test_indices.append(i)
            if len(test_indices) >= 3: break

    # 4. Run Reconstruction
    fig, axes = plt.subplots(len(test_indices), 3, figsize=(18, 5 * len(test_indices)))
    
    with torch.no_grad():
        for i, idx in enumerate(test_indices):
            spec, mag, loc, station, meta = dataset[idx]
            spec_in = spec.unsqueeze(0).to(device)
            mag_in = mag.unsqueeze(0).to(device)
            loc_in = loc.unsqueeze(0).to(device)
            station_in = station.unsqueeze(0).to(device)
            
            # Baseline Recon
            recon_base, _, _ = baseline_model(spec_in, mag_in, loc_in, station_in)
            # Full Cov Recon
            recon_full, _, _ = full_cov_model(spec_in, mag_in, loc_in, station_in)
            
            # Pick Z channel (index 2) for time domain plot
            orig_z = spec[2].cpu().numpy()
            base_z = recon_base[0, 2].cpu().numpy()
            full_z = recon_full[0, 2].cpu().numpy()
            
            # Griffin-Lim Reconstruction
            wav_orig = reconstruct_signal(orig_z)
            wav_base = reconstruct_signal(base_z)
            wav_full = reconstruct_signal(full_z)
            
            # Plot
            time = np.linspace(0, 70, len(wav_orig)) # 70s as per preprocessing
            axes[i, 0].plot(time, wav_orig, color='black', alpha=0.7)
            axes[i, 0].set_title(f"Original (ML: {meta['magnitude']}) - {meta['station_name']}")
            
            axes[i, 1].plot(time, wav_base, color='blue', alpha=0.7)
            axes[i, 1].set_title("Baseline CVAE Recon")
            
            axes[i, 2].plot(time, wav_full, color='red', alpha=0.7)
            axes[i, 2].set_title("Full Covariance Recon")
            
            for j in range(3):
                axes[i, j].set_xlabel("Time (s)")
                axes[i, j].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("reconstruction_comparison.png")
    print("Comparison plot saved as reconstruction_comparison.png")

if __name__ == "__main__":
    main()
