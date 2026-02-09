import torch
import numpy as np
import os
import sys
import obspy
import librosa
import json
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import hilbert
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr

# Add project root to path
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

def calculate_seismic_metrics(target_wav, pred_wav, target_spec, pred_spec):
    """Calculate all sismolojik metrikleri"""
    metrics = {}
    
    # 1. SSIM
    s1 = (target_spec - np.min(target_spec)) / (np.max(target_spec) - np.min(target_spec) + 1e-8)
    s2 = (pred_spec - np.min(pred_spec)) / (np.max(pred_spec) - np.min(pred_spec) + 1e-8)
    metrics['ssim'] = ssim(s1, s2, data_range=1.0)
    
    # 2. LSD
    metrics['lsd'] = np.sqrt(np.mean((np.log(target_spec + 1e-8) - np.log(pred_spec + 1e-8))**2))
    
    # 3. Arias Intensity Error
    def get_arias(sig):
        return (np.pi / (2 * 9.81)) * np.trapz(sig**2, dx=0.01)
    a_target = get_arias(target_wav)
    a_pred = get_arias(pred_wav)
    metrics['arias_err'] = np.abs(a_target - a_pred) / (np.abs(a_target) + 1e-8)
    
    # 4. Env Correlation
    e1 = np.abs(hilbert(target_wav))
    e2 = np.abs(hilbert(pred_wav))
    min_len = min(len(e1), len(e2))
    metrics['env_corr'] = np.corrcoef(e1[:min_len], e2[:min_len])[0, 1]
    
    # 5. DTW
    factor = max(1, len(target_wav) // 500)
    s1 = target_wav[::factor].reshape(-1, 1)
    s2 = pred_wav[::factor].reshape(-1, 1)
    dist, _ = fastdtw(s1, s2, dist=euclidean)
    metrics['dtw'] = dist / len(s1)
    
    # 6. XCorr
    s1 = (target_wav - np.mean(target_wav)) / (np.std(target_wav) + 1e-8)
    s2 = (pred_wav - np.mean(pred_wav)) / (np.std(pred_wav) + 1e-8)
    min_len = min(len(s1), len(s2))
    xcorr = np.correlate(s1[:min_len], s2[:min_len], mode='full')
    metrics['xcorr'] = np.max(np.abs(xcorr)) / len(s1[:min_len])
    
    return metrics

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = "ML/autoencoder/experiments/General/visualizations/diverse_ood_evaluation"
    os.makedirs(output_dir, exist_ok=True)
    
    # Paths
    baseline_chk = "ML/autoencoder/experiments/General/checkpoints/baseline_external_best.pt"
    full_cov_chk = "ML/autoencoder/experiments/FullCovariance/checkpoints/full_cov_external_best.pt"
    flow_chk = "ML/autoencoder/experiments/NormalizingFlow/checkpoints/flow_external_best.pt"
    station_list_file = "data/station_list_external_full.json"
    ood_catalog = "data/events/ood_catalog_koeri.txt"
    ood_data_dir = "data/ood_waveforms/koeri/filtered"

    with open(station_list_file, 'r') as f:
        station_list = json.load(f)

    # Models
    print("Loading models...")
    num_stations = 46 # Verified from station list file
    base_model = ConditionalVariationalAutoencoder(in_channels=3, latent_dim=128, num_stations=num_stations).to(device)
    base_model.load_state_dict(torch.load(baseline_chk, map_location=device)['model_state_dict'])
    base_model.eval()

    fc_model = FullCovCVAE(in_channels=3, latent_dim=128, num_stations=num_stations).to(device)
    fc_model.load_state_dict(torch.load(full_cov_chk, map_location=device)['model_state_dict'])
    fc_model.eval()

    flow_model = FlowCVAE(in_channels=3, latent_dim=128, num_stations=num_stations, flow_layers=8).to(device)
    flow_model.load_state_dict(torch.load(flow_chk, map_location=device)['model_state_dict'])
    flow_model.eval()

    # Create subdirs
    os.makedirs(f"{output_dir}/specs", exist_ok=True)
    os.makedirs(f"{output_dir}/waveforms", exist_ok=True)

    # Dataset
    dataset = SeismicSTFTDatasetWithMetadata(
        data_dir=ood_data_dir,
        event_file=ood_catalog,
        channels=['HH', 'BH'],
        magnitude_col='ML',
        station_list=station_list
    )
    
    print(f"Loaded {len(dataset)} OOD waveform files.")
    
    metric_keys = ['ssim', 'lsd', 'arias_err', 'env_corr', 'dtw', 'xcorr']
    results = {m: {k: [] for k in metric_keys} for m in ['Baseline', 'FullCov', 'Flow']}
    
    TARGET_FS = 100.0 # Standard for KOERI training set
    
    with torch.no_grad():
        for i in range(len(dataset)):
            try:
                spec, mag, loc, station_idx, meta = dataset[i]
                event_id = meta['event_id']
                station_name = station_list[station_idx.item()]
                print(f"Processing {event_id} at {station_name}...")
                
                # Target waveform (using filtered mseed)
                st_gt = obspy.read(meta['file_path'])
                st_gt.resample(TARGET_FS)
                tr_gt = st_gt.select(component="Z")[0] if st_gt.select(component="Z") else st_gt[0]
                gt_wav = tr_gt.data.astype(np.float32)
                
                # Length for ~111 time bins: (110 * (256-192)) + 256 = 7040 + 256 = 7296 samples
                target_len = 7300 
                if len(gt_wav) > target_len:
                    gt_wav = gt_wav[:target_len]
                elif len(gt_wav) < target_len:
                    gt_wav = np.pad(gt_wav, (0, target_len - len(gt_wav)))
                
                # Recalculate spectrogram for the 100Hz resampled signal
                f, t, Zxx = signal.stft(
                    gt_wav,
                    fs=TARGET_FS,
                    nperseg=256,
                    noverlap=192,
                    nfft=256,
                    boundary='zeros'
                )
                mag_spec = np.abs(Zxx)
                mag_spec = np.log1p(mag_spec)
                mag_min, mag_max = mag_spec.min(), mag_spec.max()
                if mag_max > mag_min:
                    mag_spec = (mag_spec - mag_min) / (mag_max - mag_min + 1e-8)
                
                # Model expects (batch, 3, 129, 111)
                spec_in = torch.from_numpy(mag_spec).unsqueeze(0).unsqueeze(0).to(device).float() # (1, 1, 129, T)
                spec_in = spec_in.repeat(1, 3, 1, 1) # (1, 3, 129, T)
                
                # Ensure exact spatial size
                if spec_in.shape[2:] != (129, 111):
                    spec_in = torch.nn.functional.interpolate(spec_in, size=(129, 111), mode='bilinear', align_corners=False)
                
                mag_in = mag.unsqueeze(0).to(device)
                loc_in = loc.unsqueeze(0).to(device)
                sta_in = station_idx.unsqueeze(0).to(device)
                
                # Inference
                r_base, _, _ = base_model(spec_in, mag_in, loc_in, sta_in)
                r_fc, _, _ = fc_model(spec_in, mag_in, loc_in, sta_in)
                r_flow, _, _, _, _ = flow_model(spec_in, mag_in, loc_in, sta_in)
                
                # Reconstruct
                orig_spec = spec_in[0, 2].cpu().numpy()
                base_spec = r_base[0, 2].cpu().numpy()
                fc_spec = r_fc[0, 2].cpu().numpy()
                flow_spec = r_flow[0, 2].cpu().numpy()
                
                # mag_min/max should ideally come from the specific sample
                wav_base = reconstruct_signal(base_spec, mag_min, mag_max)
                wav_fc = reconstruct_signal(fc_spec, mag_min, mag_max)
                wav_flow = reconstruct_signal(flow_spec, mag_min, mag_max)
                
                # Calculate metrics (ensure lengths match)
                min_len = min(len(gt_wav), len(wav_base), len(wav_fc), len(wav_flow))
                m_base = calculate_seismic_metrics(gt_wav[:min_len], wav_base[:min_len], orig_spec, base_spec)
                m_fc = calculate_seismic_metrics(gt_wav[:min_len], wav_fc[:min_len], orig_spec, fc_spec)
                m_flow = calculate_seismic_metrics(gt_wav[:min_len], wav_flow[:min_len], orig_spec, flow_spec)
                
                for k in metric_keys:
                    results['Baseline'][k].append(m_base[k])
                    results['FullCov'][k].append(m_fc[k])
                    results['Flow'][k].append(m_flow[k])
                
                # 1. Visualization - SPECTROGRAMS (2x2 Grid)
                plt.figure(figsize=(12, 10))
                titles = ["Original", "Baseline", "Full Covariance", "Normalizing Flow"]
                specs = [orig_spec, base_spec, fc_spec, flow_spec]
                
                for idx, (title, s) in enumerate(zip(titles, specs)):
                    plt.subplot(2, 2, idx+1)
                    plt.imshow(s, aspect='auto', origin='lower', cmap='magma')
                    plt.title(f"{title} Spectrogram")
                    plt.colorbar()
                
                plt.suptitle(f"OOD Spectrogram Comparison: {event_id} at {station_name}", fontsize=15)
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.savefig(f"{output_dir}/specs/{event_id}_{station_name}_specs.png")
                plt.close()

                # 2. Visualization - WAVEFORMS (4x1 Vertical Stack)
                plt.figure(figsize=(12, 15))
                wavs = [gt_wav[:min_len], wav_base[:min_len], wav_fc[:min_len], wav_flow[:min_len]]
                colors = ['black', 'tab:blue', 'tab:green', 'tab:red']
                
                times = np.arange(min_len) / TARGET_FS
                
                for idx, (title, w, color) in enumerate(zip(titles, wavs, colors)):
                    plt.subplot(4, 1, idx+1)
                    plt.plot(times, w / (np.max(np.abs(w)) + 1e-8), color=color, linewidth=1)
                    plt.title(f"{title} Waveform")
                    plt.ylim(-1.1, 1.1)
                    plt.grid(True, alpha=0.3)
                    if idx < 3: plt.xlabel("") # Hide intermediate x-labels
                
                plt.xlabel("Time (s)")
                plt.suptitle(f"OOD Waveform Comparison: {event_id} at {station_name}", fontsize=15)
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.savefig(f"{output_dir}/waveforms/{event_id}_{station_name}_waveforms.png")
                plt.close()
                
            except Exception as e:
                print(f"Error on {i}: {e}")

    # Aggregated Summary
    print("\n" + "="*80)
    print("DIVERSE OOD SEISMIC METRICS SUMMARY")
    print("="*80)
    header = "| Model | SSIM ↑ | LSD ↓ | Arias Err ↓ | Env Corr ↑ | DTW ↓ | XCorr ↑ |"
    print(header)
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for model in ['Baseline', 'FullCov', 'Flow']:
        row = [f"| **{model}**"]
        for k in metric_keys:
            val = np.nanmean(results[model][k])
            if k in ['ssim', 'env_corr', 'xcorr']: row.append(f"{val:.4f}")
            elif k == 'lsd': row.append(f"{val:.3f}")
            elif k == 'arias_err': row.append(f"{val:.4f}")
            else: row.append(f"{val:.2f}")
        print(" | ".join(row) + " |")
    print("="*80)

if __name__ == "__main__":
    main()
