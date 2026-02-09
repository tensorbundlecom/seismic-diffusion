import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import obspy
import librosa
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../')))

from ML.autoencoder.experiments.FullCovariance.core.stft_dataset_fixed import SeismicSTFTDatasetWithMetadata
from ML.autoencoder.experiments.FullCovariance.core.model_baseline_fixed import ConditionalVariationalAutoencoder
from ML.autoencoder.experiments.FullCovariance.core.model_full_cov import FullCovCVAE
from ML.autoencoder.experiments.NormalizingFlow.core.model_flow import FlowCVAE

def reconstruct_signal(magnitude_spec, mag_min=0.0, mag_max=1.0, nperseg=256, noverlap=192, n_iter=64, log_scale=True):
    spec = magnitude_spec.copy()
    
    # 1. Denormalize
    if mag_max > mag_min:
        spec = spec * (mag_max - mag_min) + mag_min
        
    # 2. Inverse Log Scale
    if log_scale:
        spec = np.expm1(spec)
        
    return librosa.griffinlim(spec, n_iter=n_iter, hop_length=nperseg - noverlap, win_length=nperseg, center=True)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = "ML/autoencoder/experiments/General/visualizations/ood_comparison_results"
    os.makedirs(output_dir, exist_ok=True)
    
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
    print("Loading Baseline (125 stations)...")
    base_model = ConditionalVariationalAutoencoder(in_channels=3, latent_dim=128, num_stations=125).to(device)
    base_st = torch.load(baseline_chk, map_location=device)
    with torch.no_grad():
        d_in = torch.randn(1, 3, 129, 111).to(device)
        d_m = torch.zeros(1).to(device)
        d_l = torch.zeros(1, 3).to(device)
        d_s = torch.zeros(1, dtype=torch.long).to(device)
        base_model(d_in, d_m, d_l, d_s)
    base_model.load_state_dict(base_st['model_state_dict'], strict=False)
    base_model.eval()

    print("Loading Full Cov (125 stations)...")
    fc_model = FullCovCVAE(in_channels=3, latent_dim=128, num_stations=125).to(device)
    fc_st = torch.load(full_cov_chk, map_location=device)
    with torch.no_grad(): fc_model(d_in, d_m, d_l, d_s)
    fc_model.load_state_dict(fc_st['model_state_dict'])
    fc_model.eval()

    print("Loading Normalizing Flow (125 stations, 8 layers)...")
    flow_model = FlowCVAE(in_channels=3, latent_dim=128, num_stations=125, flow_layers=8).to(device)
    flow_st = torch.load(flow_chk, map_location=device)
    flow_model.load_state_dict(flow_st['model_state_dict'])
    flow_model.eval()
    
    print(f"Processing {len(dataset)} OOD samples...")
    
    # Iterate through all samples
    with torch.no_grad():
        for i in range(len(dataset)):
            try:
                spec, mag, loc, station_idx, meta = dataset[i]
                
                # Paths
                filtered_path = meta['file_path']
                raw_path = filtered_path.replace("filtered", "raw")
                
                if not os.path.exists(raw_path):
                    print(f"Skipping {meta['file_name']}, raw file not found at {raw_path}")
                    continue
                    
                # Load Raw Waveform (Z channel)
                st_raw = obspy.read(raw_path)
                tr_raw = st_raw.select(component="Z")[0] if st_raw.select(component="Z") else st_raw[0]
                raw_data = tr_raw.data
                raw_time = np.linspace(0, len(raw_data)/tr_raw.stats.sampling_rate, len(raw_data))

                # Load Filtered Waveform (Z channel)
                st_filt = obspy.read(filtered_path)
                tr_filt = st_filt.select(component="Z")[0] if st_filt.select(component="Z") else st_filt[0]
                filt_data = tr_filt.data
                filt_time = np.linspace(0, len(filt_data)/tr_filt.stats.sampling_rate, len(filt_data))
                
                # Prepare Model Inputs
                spec_in = spec.unsqueeze(0)
                target_size = (129, 111)
                if spec_in.shape[2:] != target_size:
                    spec_in = torch.nn.functional.interpolate(spec_in, size=target_size, mode='bilinear', align_corners=False)
                
                spec_in = spec_in.to(device)
                mag_in = mag.unsqueeze(0).to(device)
                loc_in = loc.unsqueeze(0).to(device)
                station_in = station_idx.unsqueeze(0).to(device)
                
                # Run Models
                r_base, _, _ = base_model(spec_in, mag_in, loc_in, station_in)
                r_fc, _, _ = fc_model(spec_in, mag_in, loc_in, station_in)
                r_flow, _, _, _, _ = flow_model(spec_in, mag_in, loc_in, station_in)
                
                # Get Metadata for Scaling
                mag_min = meta.get('mag_min', 0.0)
                mag_max = meta.get('mag_max', 1.0)
                
                # Reconstruct
                base_wav = reconstruct_signal(r_base[0, 2].cpu().numpy(), mag_min, mag_max)
                fc_wav = reconstruct_signal(r_fc[0, 2].cpu().numpy(), mag_min, mag_max)
                flow_wav = reconstruct_signal(r_flow[0, 2].cpu().numpy(), mag_min, mag_max)
                
                # Common Time Vector
                rec_time = np.linspace(0, 70, len(base_wav))
                
                # Dynamic Scaling Logic
                target_peak = np.max(np.abs(filt_data))
                
                if target_peak > 0:
                    base_peak = np.max(np.abs(base_wav))
                    if base_peak > 0: base_wav = base_wav * (target_peak / base_peak)
                    
                    fc_peak = np.max(np.abs(fc_wav))
                    if fc_peak > 0: fc_wav = fc_wav * (target_peak / fc_peak)
                    
                    flow_peak = np.max(np.abs(flow_wav))
                    if flow_peak > 0: flow_wav = flow_wav * (target_peak / flow_peak)
                
                # Shared plotting scale (excluding Raw)
                filtered_and_recon = [filt_data, base_wav, fc_wav, flow_wav]
                max_amp = max([np.max(np.abs(w)) for w in filtered_and_recon]) * 1.1
                
                # Plot
                fig, axes = plt.subplots(5, 1, figsize=(12, 15), sharex=False)
                event_name = f"{meta['event_id']}_{meta['station_name']}"
                fig.suptitle(f"OOD Event: {event_name}", fontsize=16)
                
                # 1. Original (Raw)
                axes[0].plot(raw_time, raw_data, color='black', alpha=0.9, lw=0.8)
                axes[0].set_title(f"1. Original (Raw)")
                axes[0].set_ylabel("Amplitude")
                axes[0].grid(True, alpha=0.3)
                
                # 2. Original Filtered
                axes[1].plot(filt_time, filt_data, color='blue', alpha=0.9, lw=0.8)
                axes[1].set_title(f"2. Original Filtered")
                axes[1].set_ylabel("Amplitude")
                axes[1].grid(True, alpha=0.3)
                axes[1].set_ylim(-max_amp, max_amp)
                
                # 3. CVAE
                axes[2].plot(rec_time, base_wav, color='green', alpha=0.9, lw=0.8)
                axes[2].set_title(f"3. CVAE (Baseline)")
                axes[2].set_ylabel("Amplitude")
                axes[2].grid(True, alpha=0.3)
                axes[2].set_ylim(-max_amp, max_amp)
                
                # 4. FullCovariance
                axes[3].plot(rec_time, fc_wav, color='red', alpha=0.9, lw=0.8)
                axes[3].set_title(f"4. FullCovariance CVAE")
                axes[3].set_ylabel("Amplitude")
                axes[3].grid(True, alpha=0.3)
                axes[3].set_ylim(-max_amp, max_amp)
                
                # 5. Normalizing Flows
                axes[4].plot(rec_time, flow_wav, color='purple', alpha=0.9, lw=0.8)
                axes[4].set_title(f"5. Normalizing Flows")
                axes[4].set_ylabel("Amplitude")
                axes[4].set_xlabel("Time (s)")
                axes[4].grid(True, alpha=0.3)
                axes[4].set_ylim(-max_amp, max_amp)
                
                plt.tight_layout(rect=[0, 0.03, 1, 0.97])
                save_path = os.path.join(output_dir, f"ood_{event_name}.png")
                plt.savefig(save_path, dpi=150)
                plt.close()
                print(f"Saved {save_path}")
                
            except Exception as e:
                print(f"Error processing {i}: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    main()
