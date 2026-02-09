import torch
import numpy as np
import os
import sys
import obspy
import librosa
import json
from scipy.stats import pearsonr

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

def calculate_metrics(target, prediction):
    # Ensure they are the same length
    min_len = min(len(target), len(prediction))
    t = target[:min_len]
    p = prediction[:min_len]
    
    # 1. Normal Correlation
    corr, _ = pearsonr(t, p)
    
    # 2. Normalized MSE (using 0-1 range for fair comparison)
    t_min, t_max = np.min(t), np.max(t)
    t_norm = (t - t_min) / (t_max - t_min + 1e-8)
    p_norm = (p - np.min(p)) / (np.max(p) - np.min(p) + 1e-8)
    mse = np.mean((t_norm - p_norm)**2)
    
    # 3. MAE (Mean Absolute Error) 
    mae = np.mean(np.abs(t_norm - p_norm))
    
    # 4. R2 Score
    ss_res = np.sum((t_norm - p_norm)**2)
    ss_tot = np.sum((t_norm - np.mean(t_norm))**2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    # 5. SNR (Simple version)
    noise = t - p
    snr = 10 * np.log10(np.var(t) / (np.var(noise) + 1e-8))
    
    return {
        'mse': mse,
        'corr': corr,
        'mae': mae,
        'r2': r2,
        'snr': snr
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Checkpoints
    baseline_chk = "checkpoints_baseline_external/baseline_cvae_best.pt"
    full_cov_chk = "checkpoints_full_cov/20140101_20251101_0.0_9.0_9_339/full_cov_cvae_best.pt"
    flow_chk = "ML/autoencoder/experiments/NormalizingFlow/checkpoints/flow_cvae_best.pt"
    
    with open("data/station_list_125.json", 'r') as f:
        station_list_125 = json.load(f)
    with open("data/station_list_external_full.json", 'r') as f:
        station_list_ext = json.load(f)

    dataset = SeismicSTFTDatasetWithMetadata(
        data_dir="data/ood_waveforms/filtered",
        event_file="data/events/ood_catalog.txt",
        channels=['HH', 'BH'],
        magnitude_col='ML',
        station_list=station_list_125
    )
    
    # Load Models
    base_model = ConditionalVariationalAutoencoder(in_channels=3, latent_dim=128, num_stations=46).to(device)
    base_model.load_state_dict(torch.load(baseline_chk, map_location=device)['model_state_dict'])
    base_model.eval()

    fc_model = FullCovCVAE(in_channels=3, latent_dim=128, num_stations=46).to(device)
    fc_model.load_state_dict(torch.load(full_cov_chk, map_location=device)['model_state_dict'])
    fc_model.eval()

    flow_model = FlowCVAE(in_channels=3, latent_dim=128, num_stations=46, flow_layers=8).to(device)
    flow_model.load_state_dict(torch.load(flow_chk, map_location=device)['model_state_dict'])
    flow_model.eval()
    
    print(f"Calculating detailed metrics for {len(dataset)} samples...")
    
    metric_keys = ['corr', 'mse', 'mae', 'r2', 'snr']
    results = {
        'Baseline': {k: [] for k in metric_keys},
        'FullCov': {k: [] for k in metric_keys},
        'Flow': {k: [] for k in metric_keys}
    }
    
    with torch.no_grad():
        for i in range(len(dataset)):
            try:
                spec, mag, loc, station_idx, meta = dataset[i]
                station_name = station_list_125[station_idx.item()]
                
                if station_name not in station_list_ext:
                    continue
                
                ext_station_idx = station_list_ext.index(station_name)
                ext_station_in = torch.tensor([ext_station_idx], dtype=torch.long).to(device)
                
                spec_in = spec.unsqueeze(0)
                target_size = (129, 111)
                if spec_in.shape[2:] != target_size:
                    spec_in = torch.nn.functional.interpolate(spec_in, size=target_size, mode='bilinear', align_corners=False)
                
                spec_in = spec_in.to(device)
                mag_in = mag.unsqueeze(0).to(device)
                loc_in = loc.unsqueeze(0).to(device)
                
                # Forward passes
                r_base, _, _ = base_model(spec_in, mag_in, loc_in, ext_station_in)
                r_fc, _, _ = fc_model(spec_in, mag_in, loc_in, ext_station_in)
                r_flow, _, _, _, _ = flow_model(spec_in, mag_in, loc_in, ext_station_in)
                
                # Ground truth
                st_filt = obspy.read(meta['file_path'])
                tr_filt = st_filt.select(component="Z")[0] if st_filt.select(component="Z") else st_filt[0]
                filt_data = tr_filt.data
                
                mag_min, mag_max = meta.get('mag_min', 0.0), meta.get('mag_max', 1.0)
                
                # Reconstruct
                base_wav = reconstruct_signal(r_base[0, 2].cpu().numpy(), mag_min, mag_max)
                fc_wav = reconstruct_signal(r_fc[0, 2].cpu().numpy(), mag_min, mag_max)
                flow_wav = reconstruct_signal(r_flow[0, 2].cpu().numpy(), mag_min, mag_max)
                
                # Calculate metrics
                m_base = calculate_metrics(filt_data, base_wav)
                m_fc = calculate_metrics(filt_data, fc_wav)
                m_flow = calculate_metrics(filt_data, flow_wav)
                
                for k in metric_keys:
                    results['Baseline'][k].append(m_base[k])
                    results['FullCov'][k].append(m_fc[k])
                    results['Flow'][k].append(m_flow[k])
                
            except Exception as e:
                print(f"Error sample {i}: {e}")

    # Aggregated table
    print("\n--- DETAILED OOD METRICS SUMMARY (Averaged) ---")
    header = "| Model | Corr ↑ | MSE ↓ | MAE ↓ | R2 ↑ | SNR (dB) ↑ |"
    sep = "| :--- | :---: | :---: | :---: | :---: | :---: |"
    print(header)
    print(sep)
    for model in ['Baseline', 'FullCov', 'Flow']:
        res = [f"| {model}"]
        for k in metric_keys:
            val = np.nanmean(results[model][k])
            if k == 'mse':
                res.append(f"{val:.5f}")
            elif k == 'snr':
                res.append(f"{val:.2f}")
            else:
                res.append(f"{val:.4f}")
        print(" | ".join(res) + " |")

if __name__ == "__main__":
    main()
