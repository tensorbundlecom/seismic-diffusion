import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import obspy
import librosa
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../')))

from ML.autoencoder.experiments.FullCovariance.core.stft_dataset_fixed import SeismicSTFTDatasetWithMetadata
from ML.autoencoder.experiments.FullCovariance.core.model_baseline_fixed import ConditionalVariationalAutoencoder
from ML.autoencoder.experiments.FullCovariance.core.model_full_cov import FullCovCVAE
from ML.autoencoder.experiments.NormalizingFlow.core.model_flow import FlowCVAE

def reconstruct_signal(magnitude_spec, mag_min=0.0, mag_max=1.0, nperseg=256, noverlap=192, n_iter=64):
    spec = magnitude_spec.copy()
    if mag_max > mag_min:
        spec = spec * (mag_max - mag_min) + mag_min
    spec = np.expm1(spec)
    return librosa.griffinlim(spec, n_iter=n_iter, hop_length=nperseg - noverlap, win_length=nperseg, center=True)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = "ML/autoencoder/experiments/General/visualizations/final_all_external_comparison"
    os.makedirs(output_dir, exist_ok=True)
    
    # Checkpoints (All trained on external 29GB dataset - Organized)
    baseline_chk = "ML/autoencoder/experiments/General/checkpoints/baseline_external_best.pt"
    full_cov_chk = "ML/autoencoder/experiments/FullCovariance/checkpoints/full_cov_external_best.pt"
    flow_chk = "ML/autoencoder/experiments/NormalizingFlow/checkpoints/flow_external_best.pt"
    
    with open("data/station_list_125.json", 'r') as f:
        station_list_125 = json.load(f)
    with open("data/station_list_external_full.json", 'r') as f:
        station_list_ext = json.load(f)
        
    print(f"Loaded 125 OOD stations and {len(station_list_ext)} External Training stations.")

    dataset = SeismicSTFTDatasetWithMetadata(
        data_dir="data/ood_waveforms/filtered",
        event_file="data/events/ood_catalog.txt",
        channels=['HH', 'BH'],
        magnitude_col='ML',
        station_list=station_list_125
    )
    
    print("Loading Baseline (46 stations - External)...")
    base_model = ConditionalVariationalAutoencoder(in_channels=3, latent_dim=128, num_stations=46).to(device)
    base_st = torch.load(baseline_chk, map_location=device)
    base_model.load_state_dict(base_st['model_state_dict'])
    base_model.eval()

    print("Loading Full Cov (46 stations - External)...")
    fc_model = FullCovCVAE(in_channels=3, latent_dim=128, num_stations=46).to(device)
    fc_st = torch.load(full_cov_chk, map_location=device)
    fc_model.load_state_dict(fc_st['model_state_dict'])
    fc_model.eval()

    print("Loading Normalizing Flow (46 stations - External)...")
    flow_model = FlowCVAE(in_channels=3, latent_dim=128, num_stations=46, flow_layers=8).to(device)
    flow_st = torch.load(flow_chk, map_location=device)
    flow_model.load_state_dict(flow_st['model_state_dict'])
    flow_model.eval()
    
    print(f"Processing {len(dataset)} OOD samples...")
    
    with torch.no_grad():
        for i in range(len(dataset)):
            try:
                spec, mag, loc, station_idx, meta = dataset[i]
                station_name = station_list_125[station_idx.item()]
                event_name = f"{meta['event_id']}_{station_name}"
                
                if station_name not in station_list_ext:
                    print(f"Skipping {event_name}: Station not in external set.")
                    continue
                
                ext_station_idx = station_list_ext.index(station_name)
                ext_station_in = torch.tensor([ext_station_idx], dtype=torch.long).to(device)
                
                spec_in = spec.unsqueeze(0)
                if spec_in.shape[2:] != (129, 111):
                    spec_in = torch.nn.functional.interpolate(spec_in, size=(129, 111), mode='bilinear', align_corners=False)
                
                spec_in = spec_in.to(device)
                mag_in = mag.unsqueeze(0).to(device)
                loc_in = loc.unsqueeze(0).to(device)
                
                # All models now use ext_station_in
                r_base, _, _ = base_model(spec_in, mag_in, loc_in, ext_station_in)
                r_fc, _, _ = fc_model(spec_in, mag_in, loc_in, ext_station_in)
                r_flow, _, _, _, _ = flow_model(spec_in, mag_in, loc_in, ext_station_in)
                
                # Load filtered waveform
                filtered_path = meta['file_path']
                st_filt = obspy.read(filtered_path)
                tr_filt = st_filt.select(component="Z")[0] if st_filt.select(component="Z") else st_filt[0]
                filt_data = tr_filt.data
                sr = tr_filt.stats.sampling_rate
                filt_time = np.arange(len(filt_data)) / sr
                
                mag_min = meta.get('mag_min', 0.0)
                mag_max = meta.get('mag_max', 1.0)
                
                base_wav = reconstruct_signal(r_base[0, 2].cpu().numpy(), mag_min, mag_max)
                fc_wav = reconstruct_signal(r_fc[0, 2].cpu().numpy(), mag_min, mag_max)
                flow_wav = reconstruct_signal(r_flow[0, 2].cpu().numpy(), mag_min, mag_max)
                
                # Time vectors
                recon_sr = 100.0
                base_time = np.arange(len(base_wav)) / recon_sr
                fc_time = np.arange(len(fc_wav)) / recon_sr
                flow_time = np.arange(len(flow_wav)) / recon_sr
                
                # Plot - Separate Y-axis for each to see characteristics clearly
                fig, axes = plt.subplots(4, 1, figsize=(14, 12))
                fig.suptitle(f"Unified External Model Comparison: {event_name}", fontsize=16)
                
                axes[0].plot(filt_time, filt_data, 'black', lw=0.5)
                axes[0].set_title(f"Original Filtered (Peak: {np.max(np.abs(filt_data)):.1f})")
                axes[0].set_ylabel("Amplitude")
                
                axes[1].plot(base_time, base_wav, 'blue', lw=0.5)
                axes[1].set_title(f"Baseline (External) (Peak: {np.max(np.abs(base_wav)):.1f})")
                axes[1].set_ylabel("Amplitude")

                axes[2].plot(fc_time, fc_wav, 'red', lw=0.5)
                axes[2].set_title(f"FullCov (External) (Peak: {np.max(np.abs(fc_wav)):.1f})")
                axes[2].set_ylabel("Amplitude")

                axes[3].plot(flow_time, flow_wav, 'purple', lw=0.5)
                axes[3].set_title(f"Flow (External) (Peak: {np.max(np.abs(flow_wav)):.1f})")
                axes[3].set_ylabel("Amplitude")
                axes[3].set_xlabel("Time (s)")
                
                for ax in axes:
                    ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"all_ext_{event_name}.png"), dpi=150)
                plt.close()
                print(f"Saved all_ext_{event_name}.png")

            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    main()
