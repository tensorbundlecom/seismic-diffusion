import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys

# Add paths for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../')))
from ..core.stft_dataset_fixed import SeismicSTFTDatasetWithMetadata
from ..core.model_baseline_fixed import ConditionalVariationalAutoencoder
from ..core.model_full_cov import FullCovCVAE

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = os.path.join(os.path.dirname(__file__), "..", "results", "ood_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Weights (relative to root)
    baseline_chk = "ML/autoencoder/checkpoints/checkpoints_cvae/20260207_163500/best_model.pt"
    full_cov_chk = "ML/autoencoder/experiments/FullCovariance/checkpoints/checkpoints_full_cov/koeri_catalog/full_cov_cvae_best.pt"
    
    print("Loading Models...")
    # Baseline
    baseline_checkpoint = torch.load(baseline_chk, map_location=device)
    baseline_model = ConditionalVariationalAutoencoder(in_channels=3, latent_dim=128, num_stations=125).to(device)
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
        print(f"Processing Spectrograms for {event_id}...")
        event_indices = [i for i in range(len(test_dataset)) if test_dataset[i][4]['event_id'] == event_id]
        
        for idx in event_indices:
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
            
            # Get Z-channel (index 2) Spectrograms
            # These are already log-scaled and normalized from dataset if configured
            s_orig = spec_in[0, 2].cpu().numpy()
            s_base = recon_base[0, 2].cpu().numpy()
            s_full = recon_full[0, 2].cpu().numpy()
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            im0 = axes[0].imshow(s_orig, aspect='auto', origin='lower', cmap='viridis')
            axes[0].set_title(f"Original STFT: {event_id} - {station_name}")
            plt.colorbar(im0, ax=axes[0])
            
            im1 = axes[1].imshow(s_base, aspect='auto', origin='lower', cmap='viridis')
            axes[1].set_title("Baseline CVAE STFT")
            plt.colorbar(im1, ax=axes[1])
            
            im2 = axes[2].imshow(s_full, aspect='auto', origin='lower', cmap='viridis')
            axes[2].set_title("Full Covariance STFT")
            plt.colorbar(im2, ax=axes[2])
            
            for ax in axes:
                ax.set_xlabel("Time Bins")
                ax.set_ylabel("Freq Bins")
            
            plt.tight_layout()
            save_path = os.path.join(output_dir, f"stft_{event_id}_{station_name}.png")
            plt.savefig(save_path, dpi=150)
            plt.close()
            
    print(f"STFT Spectrograms saved in {output_dir}")

if __name__ == "__main__":
    main()
