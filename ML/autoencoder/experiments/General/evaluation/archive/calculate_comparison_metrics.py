import torch
import numpy as np
import os
import sys
from pathlib import Path

# Add project root to path (assuming script is 5 levels deep)
# ML/autoencoder/experiments/General/evaluation/calculate_comparison_metrics.py
# Root is at (.....)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../')))

from ML.autoencoder.experiments.FullCovariance.core.stft_dataset_fixed import SeismicSTFTDatasetWithMetadata
from ML.autoencoder.experiments.FullCovariance.core.model_baseline_fixed import ConditionalVariationalAutoencoder
from ML.autoencoder.experiments.FullCovariance.core.model_full_cov import FullCovCVAE
from ML.autoencoder.experiments.NormalizingFlow.core.model_flow import FlowCVAE

def calculate_correlation(a, b):
    a_flat = a.flatten()
    b_flat = b.flatten()
    return np.corrcoef(a_flat, b_flat)[0, 1]

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Checkpoints
    baseline_chk = "ML/autoencoder/checkpoints/checkpoints_cvae/20260207_163500/best_model.pt"
    full_cov_chk = "checkpoints_full_cov/koeri_catalog/full_cov_cvae_best.pt"
    flow_chk = "ML/autoencoder/experiments/NormalizingFlow/checkpoints/flow_cvae_best.pt"
    
    # Load Dataset (Filtered OOD)
    station_list_file = "data/station_list_125.json"
    import json
    with open(station_list_file, 'r') as f:
        station_list = json.load(f)
        
    dataset = SeismicSTFTDatasetWithMetadata(
        data_dir="data/ood_waveforms/filtered",
        event_file="data/events/ood_catalog.txt",
        channels=['HH', 'BH'],
        magnitude_col='ML',
        station_list=station_list
    )
    
    # Initialize Models
    base_model = ConditionalVariationalAutoencoder(in_channels=3, latent_dim=128, num_stations=125).to(device)
    base_st = torch.load(baseline_chk, map_location=device)
    base_model.load_state_dict(base_st['model_state_dict'], strict=False)
    base_model.eval()

    fc_model = FullCovCVAE(in_channels=3, latent_dim=128, num_stations=125).to(device)
    fc_st = torch.load(full_cov_chk, map_location=device)
    fc_model.load_state_dict(fc_st['model_state_dict'])
    fc_model.eval()

    flow_model = FlowCVAE(in_channels=3, latent_dim=128, num_stations=125, flow_layers=8).to(device)
    flow_st = torch.load(flow_chk, map_location=device)
    flow_model.load_state_dict(flow_st['model_state_dict'])
    flow_model.eval()
    
    metrics = {
        'Baseline': {'mse': [], 'corr': []},
        'FullCov': {'mse': [], 'corr': []},
        'Flow': {'mse': [], 'corr': []}
    }
    
    with torch.no_grad():
        for i in range(len(dataset)):
            try:
                spec, mag, loc, station_idx, meta = dataset[i]
                
                spec_in = spec.unsqueeze(0).to(device)
                # Resize
                target_size = (129, 111)
                if spec_in.shape[2:] != target_size:
                    spec_in = torch.nn.functional.interpolate(spec_in, size=target_size, mode='bilinear', align_corners=False)
                
                mag_in = mag.unsqueeze(0).to(device)
                loc_in = loc.unsqueeze(0).to(device)
                station_in = station_idx.unsqueeze(0).to(device)
                
                # Run Models (Note: comparing spectrograms directly)
                r_base, _, _ = base_model(spec_in, mag_in, loc_in, station_in)
                r_fc, _, _ = fc_model(spec_in, mag_in, loc_in, station_in)
                r_flow, _, _, _, _ = flow_model(spec_in, mag_in, loc_in, station_in)
                
                # Metrics (on Z channel [2])
                orig_z = spec_in[0, 2].cpu().numpy()
                
                # Baseline
                b_z = r_base[0, 2].cpu().numpy()
                metrics['Baseline']['mse'].append(np.mean((orig_z - b_z)**2))
                metrics['Baseline']['corr'].append(calculate_correlation(orig_z, b_z))
                
                # FullCov
                fc_z = r_fc[0, 2].cpu().numpy()
                metrics['FullCov']['mse'].append(np.mean((orig_z - fc_z)**2))
                metrics['FullCov']['corr'].append(calculate_correlation(orig_z, fc_z))
                
                # Flow
                fl_z = r_flow[0, 2].cpu().numpy()
                metrics['Flow']['mse'].append(np.mean((orig_z - fl_z)**2))
                metrics['Flow']['corr'].append(calculate_correlation(orig_z, fl_z))
                
            except Exception as e:
                print(f"Error on sample {i}: {e}")

    # Results
    print("\nModel Evaluation on OOD Samples (Spectrogram Domain)")
    print("-" * 50)
    for model_name, data in metrics.items():
        avg_mse = np.mean(data['mse'])
        avg_corr = np.mean(data['corr'])
        print(f"{model_name:>10} | Avg MSE: {avg_mse:.6f} | Avg Corr: {avg_corr:.4f}")

if __name__ == "__main__":
    main()
