import torch
import numpy as np
import os
import sys
import obspy
import librosa
import json
from scipy.signal import hilbert
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from skimage.metrics import structural_similarity as ssim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../')))

from ML.autoencoder.experiments.General.core.stft_dataset import SeismicSTFTDatasetWithMetadata
from ML.autoencoder.experiments.General.core.model_baseline import ConditionalVariationalAutoencoder
from ML.autoencoder.experiments.FullCovariance.core.model_full_cov import FullCovCVAE
from ML.autoencoder.experiments.NormalizingFlow.core.model_flow import FlowCVAE

def reconstruct_signal(magnitude_spec, mag_min=0.0, mag_max=1.0, nperseg=256, noverlap=192, n_iter=64):
    spec = magnitude_spec.copy()
    if mag_max > mag_min:
        spec = spec * (mag_max - mag_min) + mag_min
    spec = np.expm1(spec)
    return librosa.griffinlim(spec, n_iter=n_iter, hop_length=nperseg - noverlap, win_length=nperseg, center=True)

def calculate_ssim_spectrogram(spec1, spec2):
    """SSIM on spectrogram (2D image similarity)"""
    # Normalize to 0-1 range
    s1 = (spec1 - np.min(spec1)) / (np.max(spec1) - np.min(spec1) + 1e-8)
    s2 = (spec2 - np.min(spec2)) / (np.max(spec2) - np.min(spec2) + 1e-8)
    return ssim(s1, s2, data_range=1.0)

def calculate_lsd(spec1, spec2):
    """Log-Spectral Distance"""
    # Add small epsilon to avoid log(0)
    s1 = np.log(spec1 + 1e-8)
    s2 = np.log(spec2 + 1e-8)
    return np.sqrt(np.mean((s1 - s2)**2))

def calculate_arias_intensity(signal, dt=0.01):
    """Arias Intensity (cumulative energy measure)"""
    # Arias Intensity = (pi / 2g) * integral(a^2 dt)
    g = 9.81  # gravity
    arias = (np.pi / (2 * g)) * np.trapz(signal**2, dx=dt)
    return arias

def calculate_envelope_correlation(sig1, sig2):
    """Correlation between signal envelopes"""
    env1 = np.abs(hilbert(sig1))
    env2 = np.abs(hilbert(sig2))
    
    # Ensure same length
    min_len = min(len(env1), len(env2))
    env1 = env1[:min_len]
    env2 = env2[:min_len]
    
    return np.corrcoef(env1, env2)[0, 1]

def calculate_dtw_distance(sig1, sig2):
    """Dynamic Time Warping distance (normalized)"""
    # Downsample for efficiency
    factor = max(1, len(sig1) // 1000)
    s1 = sig1[::factor].reshape(-1, 1)
    s2 = sig2[::factor].reshape(-1, 1)
    
    distance, _ = fastdtw(s1, s2, dist=euclidean)
    # Normalize by length
    return distance / len(s1)

def calculate_cross_correlation(sig1, sig2):
    """Maximum cross-correlation coefficient"""
    # Ensure same length
    min_len = min(len(sig1), len(sig2))
    s1 = sig1[:min_len]
    s2 = sig2[:min_len]
    
    # Normalize
    s1 = (s1 - np.mean(s1)) / (np.std(s1) + 1e-8)
    s2 = (s2 - np.mean(s2)) / (np.std(s2) + 1e-8)
    
    # Cross-correlation
    xcorr = np.correlate(s1, s2, mode='full')
    return np.max(np.abs(xcorr)) / len(s1)

def calculate_seismic_metrics(target_wav, pred_wav, target_spec, pred_spec):
    """Calculate all seismic metrics"""
    metrics = {}
    
    # 1. SSIM (Spectrogram)
    metrics['ssim'] = calculate_ssim_spectrogram(target_spec, pred_spec)
    
    # 2. LSD (Spectral Distance)
    metrics['lsd'] = calculate_lsd(target_spec, pred_spec)
    
    # 3. Arias Intensity Error
    arias_target = calculate_arias_intensity(target_wav)
    arias_pred = calculate_arias_intensity(pred_wav)
    metrics['arias_err'] = np.abs(arias_target - arias_pred) / (np.abs(arias_target) + 1e-8)
    
    # 4. Envelope Correlation
    metrics['env_corr'] = calculate_envelope_correlation(target_wav, pred_wav)
    
    # 5. DTW (Time Distance)
    metrics['dtw'] = calculate_dtw_distance(target_wav, pred_wav)
    
    # 6. Cross-Correlation
    metrics['xcorr'] = calculate_cross_correlation(target_wav, pred_wav)
    
    return metrics

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Checkpoints (External Dataset - Organized)
    baseline_chk = "ML/autoencoder/experiments/General/checkpoints/baseline_external_best.pt"
    full_cov_chk = "ML/autoencoder/experiments/FullCovariance/checkpoints/full_cov_external_best.pt"
    flow_chk = "ML/autoencoder/experiments/NormalizingFlow/checkpoints/flow_external_best.pt"
    
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
    print("Loading models...")
    base_model = ConditionalVariationalAutoencoder(in_channels=3, latent_dim=128, num_stations=46).to(device)
    base_model.load_state_dict(torch.load(baseline_chk, map_location=device)['model_state_dict'])
    base_model.eval()

    fc_model = FullCovCVAE(in_channels=3, latent_dim=128, num_stations=46).to(device)
    fc_model.load_state_dict(torch.load(full_cov_chk, map_location=device)['model_state_dict'])
    fc_model.eval()

    flow_model = FlowCVAE(in_channels=3, latent_dim=128, num_stations=46, flow_layers=8).to(device)
    flow_model.load_state_dict(torch.load(flow_chk, map_location=device)['model_state_dict'])
    flow_model.eval()
    
    print(f"Calculating seismic metrics for {len(dataset)} OOD samples...")
    
    metric_keys = ['ssim', 'lsd', 'arias_err', 'env_corr', 'dtw', 'xcorr']
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
                
                # Ground truth waveform
                st_filt = obspy.read(meta['file_path'])
                tr_filt = st_filt.select(component="Z")[0] if st_filt.select(component="Z") else st_filt[0]
                filt_data = tr_filt.data
                
                mag_min, mag_max = meta.get('mag_min', 0.0), meta.get('mag_max', 1.0)
                
                # Reconstruct waveforms
                base_wav = reconstruct_signal(r_base[0, 2].cpu().numpy(), mag_min, mag_max)
                fc_wav = reconstruct_signal(r_fc[0, 2].cpu().numpy(), mag_min, mag_max)
                flow_wav = reconstruct_signal(r_flow[0, 2].cpu().numpy(), mag_min, mag_max)
                
                # Get spectrograms (Z channel)
                target_spec = spec_in[0, 2].cpu().numpy()
                base_spec = r_base[0, 2].cpu().numpy()
                fc_spec = r_fc[0, 2].cpu().numpy()
                flow_spec = r_flow[0, 2].cpu().numpy()
                
                # Calculate metrics
                m_base = calculate_seismic_metrics(filt_data, base_wav, target_spec, base_spec)
                m_fc = calculate_seismic_metrics(filt_data, fc_wav, target_spec, fc_spec)
                m_flow = calculate_seismic_metrics(filt_data, flow_wav, target_spec, flow_spec)
                
                for k in metric_keys:
                    results['Baseline'][k].append(m_base[k])
                    results['FullCov'][k].append(m_fc[k])
                    results['Flow'][k].append(m_flow[k])
                
                print(f"Processed sample {i+1}/{len(dataset)}")
                
            except Exception as e:
                print(f"Error sample {i}: {e}")

    # Aggregated table
    print("\n" + "="*80)
    print("SEISMIC METRICS SUMMARY (OOD Dataset - 18 Samples)")
    print("="*80)
    header = "| Model | SSIM ↑ | LSD ↓ | Arias Err ↓ | Env Corr ↑ | DTW ↓ | XCorr ↑ |"
    sep = "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
    print(header)
    print(sep)
    
    for model in ['Baseline', 'FullCov', 'Flow']:
        row = [f"| **{model}**"]
        for k in metric_keys:
            val = np.nanmean(results[model][k])
            if k in ['ssim', 'env_corr', 'xcorr']:
                row.append(f"{val:.4f}")
            elif k == 'lsd':
                row.append(f"{val:.3f}")
            elif k == 'arias_err':
                row.append(f"{val:.4f}")
            else:  # dtw
                row.append(f"{val:.2f}")
        print(" | ".join(row) + " |")
    
    print("\n" + "="*80)
    print("Legend:")
    print("  ↑ = Higher is better  |  ↓ = Lower is better")
    print("  SSIM: Structural Similarity (spectrogram)")
    print("  LSD: Log-Spectral Distance")
    print("  Arias Err: Relative error in Arias Intensity (energy)")
    print("  Env Corr: Envelope correlation (amplitude modulation)")
    print("  DTW: Dynamic Time Warping distance (temporal alignment)")
    print("  XCorr: Maximum cross-correlation (phase similarity)")
    print("="*80)

if __name__ == "__main__":
    main()
