import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
import librosa

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from ML.autoencoder.experiments.General.core.stft_dataset import SeismicSTFTDatasetWithMetadata
from ML.autoencoder.experiments.General.core.model_baseline import ConditionalVariationalAutoencoder
from ML.autoencoder.experiments.FullCovariance.core.model_full_cov import FullCovCVAE
from ML.autoencoder.experiments.NormalizingFlow.core.model_flow import FlowCVAE

def reconstruct_signal(magnitude_spec, mag_min=0.0, mag_max=1.0, nperseg=256, noverlap=192, n_iter=64, log_scale=True):
    spec = magnitude_spec.copy()
    
    # 1. Denormalize from [0, 1] back to log-magnitude range
    if mag_max > mag_min:
        spec = spec * (mag_max - mag_min) + mag_min
        
    # 2. Inverse Log Scale
    if log_scale:
        spec = np.expm1(spec)
        
    return librosa.griffinlim(spec, n_iter=n_iter, hop_length=nperseg - noverlap, win_length=nperseg, center=True)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Checkpoints
    baseline_chk = "ML/autoencoder/checkpoints/checkpoints_cvae/20260207_163500/best_model.pt"
    full_cov_chk = "checkpoints_full_cov/koeri_catalog/full_cov_cvae_best.pt"
    flow_chk = "ML/autoencoder/experiments/NormalizingFlow/checkpoints/flow_cvae_best.pt"
    
    # Load Data (using the same filtered data)
    # The dataset code was just modified to return mag_min/max in metadata!
    dataset = SeismicSTFTDatasetWithMetadata(
        data_dir="data/filtered_waveforms",
        event_file="data/events/koeri_catalog.txt",
        channels=['HH'],
        magnitude_col='ML'
    )
    # The baseline was trained with 125 stations. Even if this subset has fewer, 
    # the model architecture expects 125 to match the checkpoint weights.
    num_stations = 125 
    
    # Initialize and Load Models
    print("Loading Baseline (125 stations)...")
    base_model = ConditionalVariationalAutoencoder(in_channels=3, latent_dim=128, num_stations=125).to(device)
    base_st = torch.load(baseline_chk, map_location=device)
    # Trigger dummy init for baseline if needed
    with torch.no_grad():
        d_in = torch.randn(1, 3, 129, 111).to(device)
        d_m = torch.zeros(1).to(device)
        d_l = torch.zeros(1, 3).to(device)
        d_s = torch.zeros(1, dtype=torch.long).to(device)
        base_model(d_in, d_m, d_l, d_s)
    base_model.load_state_dict(base_st['model_state_dict'], strict=False)
    base_model.eval()

    print(f"Loading Full Cov (41 stations)...")
    fc_model = FullCovCVAE(in_channels=3, latent_dim=128, num_stations=41).to(device)
    fc_st = torch.load(full_cov_chk, map_location=device)
    # Trigger dummy init
    with torch.no_grad(): fc_model(d_in, d_m, d_l, d_s)
    fc_model.load_state_dict(fc_st['model_state_dict'])
    fc_model.eval()

    print(f"Loading Normalizing Flow (41 stations, 8 layers)...")
    flow_model = FlowCVAE(in_channels=3, latent_dim=128, num_stations=41, flow_layers=8).to(device)
    flow_st = torch.load(flow_chk, map_location=device)
    flow_model.load_state_dict(flow_st['model_state_dict'])
    flow_model.eval()

    # Pick samples
    test_indices = [5, 12, 18] # Diverse samples
    
    fig, axes = plt.subplots(len(test_indices), 4, figsize=(24, 4 * len(test_indices)))
    
    with torch.no_grad():
        for i, idx in enumerate(test_indices):
            spec, mag, loc, station, meta = dataset[idx] # This likely won't get the new metadata unless we reload module logic or restart?
            # Actually, standard python import caching might be an issue if running in persistent process,
            # but here each `run_command` starts a fresh python process, so the modified user code will be loaded fresh.
            
            spec_in = spec.unsqueeze(0).to(device)
            mag_in = mag.unsqueeze(0).to(device)
            loc_in = loc.unsqueeze(0).to(device)
            station_in = station.unsqueeze(0).to(device)
            
            # Recons
            r_base, _, _ = base_model(spec_in, mag_in, loc_in, station_in)
            r_fc, _, _ = fc_model(spec_in, mag_in, loc_in, station_in)
            r_flow, _, _, _, _ = flow_model(spec_in, mag_in, loc_in, station_in)
            
            # Get Min/Max for correct inversion
            # Note: Dataset returns tensors, but metadata is a dict of values
            mag_min = meta.get('mag_min', 0.0)
            mag_max = meta.get('mag_max', 1.0)
            
            # Time domain recons for Z channel (index 2)
            orig_wav = reconstruct_signal(spec[2].cpu().numpy(), mag_min=mag_min, mag_max=mag_max)
            base_wav = reconstruct_signal(r_base[0, 2].cpu().numpy(), mag_min=mag_min, mag_max=mag_max)
            fc_wav = reconstruct_signal(r_fc[0, 2].cpu().numpy(), mag_min=mag_min, mag_max=mag_max)
            flow_wav = reconstruct_signal(r_flow[0, 2].cpu().numpy(), mag_min=mag_min, mag_max=mag_max)
            
            time = np.linspace(0, 70, len(orig_wav))
            
            # Determine max amplitude for the row to share y-axis
            all_wavs = [orig_wav, base_wav, fc_wav, flow_wav]
            max_amp = max([np.max(np.abs(w)) for w in all_wavs]) * 1.1
            
            # Plot
            axes[i, 0].plot(time, orig_wav, color='black', alpha=0.8)
            axes[i, 0].set_title(f"Original (ML: {meta['magnitude']})")
            
            axes[i, 1].plot(time, base_wav, color='blue', alpha=0.8)
            axes[i, 1].set_title("Baseline CVAE")
            
            axes[i, 2].plot(time, fc_wav, color='red', alpha=0.8)
            axes[i, 2].set_title("Full Covariance")
            
            axes[i, 3].plot(time, flow_wav, color='purple', alpha=0.8)
            axes[i, 3].set_title("Normalizing Flow")
            
            for j in range(4):
                axes[i, j].set_xlabel("Time (s)")
                axes[i, j].grid(True, alpha=0.3)
                axes[i, j].set_ylim(-max_amp, max_amp)

    plt.tight_layout()
    plt.savefig("final_3model_comparison.png")
    print("Final comparison saved to final_3model_comparison.png")

if __name__ == "__main__":
    main()
