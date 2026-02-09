import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import obspy
import librosa
from pathlib import Path
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../')))

from ML.autoencoder.experiments.General.core.stft_dataset import SeismicSTFTDatasetWithMetadata
from ML.autoencoder.experiments.FullCovariance.core.model_baseline_fixed import ConditionalVariationalAutoencoder
from ML.autoencoder.experiments.FullCovariance.core.model_full_cov import FullCovCVAE
from ML.autoencoder.experiments.NormalizingFlow.core.model_flow import FlowCVAE

def reconstruct_signal(magnitude_spec, mag_min=0.0, mag_max=1.0, nperseg=256, noverlap=192, n_iter=64, log_scale=True):
    spec = magnitude_spec.copy()
    if mag_max > mag_min:
        spec = spec * (mag_max - mag_min) + mag_min
    if log_scale:
        spec = np.expm1(spec)
    return librosa.griffinlim(spec, n_iter=n_iter, hop_length=nperseg - noverlap, win_length=nperseg, center=True)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = "ML/autoencoder/experiments/General/visualizations/final_external_comparison"
    os.makedirs(output_dir, exist_ok=True)
    
    # Checkpoints
    baseline_chk = "ML/autoencoder/checkpoints/checkpoints_cvae/20260207_163500/best_model.pt"
    # New external checkpoints
    full_cov_chk = "checkpoints_full_cov/20140101_20251101_0.0_9.0_9_339/full_cov_cvae_best.pt"
    flow_chk = "ML/autoencoder/experiments/NormalizingFlow/checkpoints/flow_cvae_best.pt"
    
    # Station Lists
    station_file_125 = "data/station_list_125.json"
    station_file_ext = "data/station_list_external_full.json"
    
    with open(station_file_125, 'r') as f:
        station_list_125 = json.load(f)
    with open(station_file_ext, 'r') as f:
        station_list_ext = json.load(f)
        
    print(f"Loaded 125 OOD stations and {len(station_list_ext)} External Training stations.")

    # Dataset (OOD uses 125 list)
    dataset = SeismicSTFTDatasetWithMetadata(
        data_dir="data/ood_waveforms/filtered",
        event_file="data/events/ood_catalog.txt",
        channels=['HH', 'BH'],
        magnitude_col='ML',
        station_list=station_list_125
    )
    
    # Initialize Models
    print("Loading Baseline (125 stations)...")
    base_model = ConditionalVariationalAutoencoder(in_channels=3, latent_dim=128, num_stations=125).to(device)
    base_st = torch.load(baseline_chk, map_location=device)
    base_model.load_state_dict(base_st['model_state_dict'], strict=False)
    base_model.eval()

    print("Loading Full Cov (46 stations)...")
    fc_model = FullCovCVAE(in_channels=3, latent_dim=128, num_stations=46).to(device)
    fc_st = torch.load(full_cov_chk, map_location=device)
    fc_model.load_state_dict(fc_st['model_state_dict'])
    fc_model.eval()

    print("Loading Normalizing Flow (46 stations)...")
    flow_model = FlowCVAE(in_channels=3, latent_dim=128, num_stations=46, flow_layers=8).to(device)
    flow_st = torch.load(flow_chk, map_location=device)
    flow_model.load_state_dict(flow_st['model_state_dict'])
    flow_model.eval()
    
    print(f"Processing {len(dataset)} OOD samples...")
    
    with torch.no_grad():
        for i in range(len(dataset)):
            try:
                spec, mag, loc, station_idx, meta = dataset[i]
                
                # Identify Station
                station_name = station_list_125[station_idx.item()]
                
                # Check if station exists in external training set
                if station_name not in station_list_ext:
                    print(f"Skipping {meta['file_name']}: Station {station_name} not in external training set.")
                    continue
                
                # Get index for external models
                ext_station_idx = station_list_ext.index(station_name)
                ext_station_in = torch.tensor([ext_station_idx], dtype=torch.long).to(device)
                
                # Prepare Inputs
                spec_in = spec.unsqueeze(0)
                target_size = (129, 111)
                if spec_in.shape[2:] != target_size:
                    spec_in = torch.nn.functional.interpolate(spec_in, size=target_size, mode='bilinear', align_corners=False)
                
                spec_in = spec_in.to(device)
                mag_in = mag.unsqueeze(0).to(device)
                loc_in = loc.unsqueeze(0).to(device)
                base_station_in = station_idx.unsqueeze(0).to(device) # For Baseline
                
                # Run Models
                r_base, _, _ = base_model(spec_in, mag_in, loc_in, base_station_in)
                r_fc, _, _ = fc_model(spec_in, mag_in, loc_in, ext_station_in)
                r_flow, _, _, _, _ = flow_model(spec_in, mag_in, loc_in, ext_station_in)

                # Define event_name early for debug
                event_name = f"{meta['event_id']}_{meta['station_name']}"

                # DEBUG PRINTS
                print(f"--- Event: {event_name} ---")
                print(f"Spec IN: Min={spec_in.min().item():.4f}, Max={spec_in.max().item():.4f}, Mean={spec_in.mean().item():.4f}")
                print(f"Base OUT: Min={r_base.min().item():.4f}, Max={r_base.max().item():.4f}, Mean={r_base.mean().item():.4f}")
                print(f"FC OUT: Min={r_fc.min().item():.4f}, Max={r_fc.max().item():.4f}, Mean={r_fc.mean().item():.4f}")
                print(f"Flow OUT: Min={r_flow.min().item():.4f}, Max={r_flow.max().item():.4f}, Mean={r_flow.mean().item():.4f}")
                
                # Post-process (Load raw/filter for plotting)
                filtered_path = meta['file_path']
                st_filt = obspy.read(filtered_path)
                tr_filt = st_filt.select(component="Z")[0] if st_filt.select(component="Z") else st_filt[0]
                filt_data = tr_filt.data
                rec_time = np.linspace(0, 70, len(filt_data))
                
                mag_min = meta.get('mag_min', 0.0)
                mag_max = meta.get('mag_max', 1.0)
                
                base_wav = reconstruct_signal(r_base[0, 2].cpu().numpy(), mag_min, mag_max)
                fc_wav = reconstruct_signal(r_fc[0, 2].cpu().numpy(), mag_min, mag_max)
                flow_wav = reconstruct_signal(r_flow[0, 2].cpu().numpy(), mag_min, mag_max)

                print(f"Mag Min: {mag_min}, Mag Max: {mag_max}")
                print(f"Base Wav: Min={base_wav.min():.4e}, Max={base_wav.max():.4e}")
                print(f"FC Wav: Min={fc_wav.min():.4e}, Max={fc_wav.max():.4e}")
                print(f"Flow Wav: Min={flow_wav.min():.4e}, Max={flow_wav.max():.4e}")
                
                # Resize to common length if needed (simple fix for plotting)
                min_len = min(len(filt_data), len(base_wav))
                filt_data = filt_data[:min_len]
                base_wav = base_wav[:min_len]
                fc_wav = fc_wav[:min_len]
                flow_wav = flow_wav[:min_len]
                rec_time = rec_time[:min_len]
                
                # Plot
                fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
                event_name = f"{meta['event_id']}_{meta['station_name']}"
                fig.suptitle(f"External Model Comparison: {event_name}", fontsize=16)
                
                max_amp = max(np.max(np.abs(filt_data)), np.max(np.abs(base_wav)), np.max(np.abs(fc_wav)), np.max(np.abs(flow_wav))) * 1.1

                axes[0].plot(rec_time, filt_data, 'black', alpha=0.8)
                axes[0].set_title("Original Filtered")
                axes[0].set_ylim(-max_amp, max_amp)
                
                axes[1].plot(rec_time, base_wav, 'blue', alpha=0.8)
                axes[1].set_title("Baseline (125st)")
                axes[1].set_ylim(-max_amp, max_amp)

                axes[2].plot(rec_time, fc_wav, 'red', alpha=0.8)
                axes[2].set_title("Full Covariance (External 46st)")
                axes[2].set_ylim(-max_amp, max_amp)

                axes[3].plot(rec_time, flow_wav, 'purple', alpha=0.8)
                axes[3].set_title("Normalizing Flow (External 46st)")
                axes[3].set_ylim(-max_amp, max_amp)
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"final_{event_name}.png"))
                plt.close()
                print(f"Examples saved for {event_name}")

            except Exception as e:
                print(f"Error {e}")

if __name__ == "__main__":
    main()
