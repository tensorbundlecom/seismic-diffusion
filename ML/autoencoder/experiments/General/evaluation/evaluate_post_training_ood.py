import torch
import numpy as np
import os
import sys
import obspy
from scipy import signal
from scipy.signal import hilbert
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from skimage.metrics import structural_similarity as ssim
import json
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../')))

from ML.autoencoder.experiments.General.core.stft_dataset import SeismicSTFTDatasetWithMetadata
from ML.autoencoder.experiments.General.core.model_baseline import ConditionalVariationalAutoencoder
from ML.autoencoder.experiments.FullCovariance.core.model_full_cov import FullCovCVAE
from ML.autoencoder.experiments.NormalizingFlow.core.model_flow import FlowCVAE

def reconstruct_signal(magnitude_spec, mag_min=0.0, mag_max=1.0, nperseg=256, noverlap=192, nfft=256, fs=100.0, n_iter=64):
    """Griffin-Lim implementation using scipy.signal.stft/istft to ensure scaling consistency."""
    spec = magnitude_spec.copy()
    if mag_max > mag_min:
        spec = spec * (mag_max - mag_min) + mag_min
    spec = np.expm1(spec)
    
    # Initialize with random phase
    phase = np.exp(2j * np.pi * np.random.rand(*spec.shape))
    
    for i in range(n_iter):
        stft_complex = spec * phase
        _, waveform = signal.istft(stft_complex, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft, boundary='zeros')
        _, _, new_Zxx = signal.stft(waveform, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft, boundary='zeros')
        
        if new_Zxx.shape != spec.shape:
            min_f = min(new_Zxx.shape[0], spec.shape[0])
            min_t = min(new_Zxx.shape[1], spec.shape[1])
            phase_angle = np.angle(new_Zxx[:min_f, :min_t])
            new_phase = np.zeros_like(spec, dtype=complex)
            new_phase[:min_f, :min_t] = np.exp(1j * phase_angle)
            phase = new_phase
        else:
            phase = np.exp(1j * np.angle(new_Zxx))
            
    stft_complex = spec * phase
    _, waveform = signal.istft(stft_complex, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft, boundary='zeros')
    return waveform

def calculate_seismic_metrics(target_wav, pred_wav, target_spec, pred_spec, fs=100.0):
    """Calculate all seismic and spectral metrics."""
    metrics = {}
    
    # --- 1. STFT-Level Metrics (Deterministic) ---
    
    # 1.1 SSIM (Structural Similarity)
    s1 = (target_spec - np.min(target_spec)) / (np.max(target_spec) - np.min(target_spec) + 1e-8)
    s2 = (pred_spec - np.min(pred_spec)) / (np.max(pred_spec) - np.min(pred_spec) + 1e-8)
    metrics['ssim'] = ssim(s1, s2, data_range=1.0)
    
    # 1.2 LSD (Log-Spectral Distance)
    metrics['lsd'] = np.sqrt(np.mean((np.log(target_spec + 1e-8) - np.log(pred_spec + 1e-8))**2))
    
    # 1.3 Spectral Convergence (SC)
    # Norm of diff / Norm of target
    metrics['sc'] = np.linalg.norm(target_spec - pred_spec) / (np.linalg.norm(target_spec) + 1e-8)
    
    # 1.4 Spectral Correlation (S-Corr)
    metrics['s_corr'] = np.corrcoef(target_spec.flatten(), pred_spec.flatten())[0, 1]
    
    # 1.5 Spectral STA/LTA Error
    # STA/LTA on spectral power (sum across frequencies)
    spec_power_target = np.sum(target_spec, axis=0)
    spec_power_pred = np.sum(pred_spec, axis=0)
    
    def get_spectral_sta_lta(power, sta_len=5, lta_len=40):
        sta = np.convolve(power, np.ones(sta_len)/sta_len, mode='same')
        lta = np.convolve(power, np.ones(lta_len)/lta_len, mode='same')
        return sta / (lta + 1e-8)
    
    sl_target = get_spectral_sta_lta(spec_power_target)
    sl_pred = get_spectral_sta_lta(spec_power_pred)
    metrics['sta_lta_err'] = np.abs(np.max(sl_target) - np.max(sl_pred)) / (np.max(sl_target) + 1e-8)

    # 1.6 Multi-Resolution STFT (STFT-MR)
    # Average LSD across different window sizes
    mr_lsd = []
    for n_fft in [64, 128, 512]:
        hop = n_fft // 4
        _, _, Z1 = signal.stft(target_wav, fs=fs, nperseg=n_fft, noverlap=n_fft-hop)
        _, _, Z2 = signal.stft(pred_wav, fs=fs, nperseg=n_fft, noverlap=n_fft-hop)
        T_spec = np.abs(Z1)
        P_spec = np.abs(Z2)
        min_f = min(T_spec.shape[0], P_spec.shape[0])
        min_t = min(T_spec.shape[1], P_spec.shape[1])
        mr_lsd.append(np.sqrt(np.mean((np.log(T_spec[:min_f, :min_t] + 1e-8) - np.log(P_spec[:min_f, :min_t] + 1e-8))**2)))
    metrics['mr_lsd'] = np.mean(mr_lsd)

    # --- 2. Waveform-Level Metrics (Stochastic due to Griffin-Lim) ---
    
    # 2.1 Arias Intensity Error
    def get_arias(sig):
        # Using trapezoid (scipy 1.10+) or trapz (older)
        try:
            from scipy.integrate import trapezoid
            return (np.pi / (2 * 9.81)) * trapezoid(sig**2, dx=1/fs)
        except ImportError:
            return (np.pi / (2 * 9.81)) * np.trapz(sig**2, dx=1/fs)
            
    a_target = get_arias(target_wav)
    a_pred = get_arias(pred_wav)
    metrics['arias_err'] = np.abs(a_target - a_pred) / (np.abs(a_target) + 1e-8)
    
    # 2.2 Env Correlation
    e1 = np.abs(hilbert(target_wav))
    e2 = np.abs(hilbert(pred_wav))
    min_len = min(len(e1), len(e2))
    metrics['env_corr'] = np.corrcoef(e1[:min_len], e2[:min_len])[0, 1]
    
    # 2.3 DTW
    factor = max(1, len(target_wav) // 500)
    s1 = target_wav[::factor].reshape(-1, 1)
    s2 = pred_wav[::factor].reshape(-1, 1)
    dist, _ = fastdtw(s1, s2, dist=euclidean)
    metrics['dtw'] = dist / len(s1)
    
    # 2.4 XCorr
    s1 = (target_wav - np.mean(target_wav)) / (np.std(target_wav) + 1e-8)
    s2 = (pred_wav - np.mean(pred_wav)) / (np.std(pred_wav) + 1e-8)
    min_len = min(len(s1), len(s2))
    xcorr = np.correlate(s1[:min_len], s2[:min_len], mode='full')
    metrics['xcorr'] = np.max(np.abs(xcorr)) / len(s1[:min_len])
    
    return metrics

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = "ML/autoencoder/experiments/General/visualizations/post_training_ood_evaluation"
    os.makedirs(output_dir, exist_ok=True)
    
    # Paths
    baseline_chk = "ML/autoencoder/experiments/General/checkpoints/baseline_external_best.pt"
    full_cov_chk = "ML/autoencoder/experiments/FullCovariance/checkpoints/full_cov_external_best.pt"
    flow_chk = "ML/autoencoder/experiments/NormalizingFlow/checkpoints/flow_external_best.pt"
    station_list_file = "data/station_list_external_full.json"
    ood_catalog = "data/events/ood_catalog_post_training.txt"
    ood_data_dir = "data/ood_waveforms/post_training/filtered"

    with open(station_list_file, 'r') as f:
        station_list = json.load(f)

    # Models
    print("Loading models...")
    num_stations = 46
    base_model = ConditionalVariationalAutoencoder(in_channels=3, latent_dim=128, num_stations=num_stations).to(device)
    base_model.load_state_dict(torch.load(baseline_chk, map_location=device)['model_state_dict'])
    base_model.eval()

    fc_model = FullCovCVAE(in_channels=3, latent_dim=128, num_stations=num_stations).to(device)
    fc_model.load_state_dict(torch.load(full_cov_chk, map_location=device)['model_state_dict'])
    fc_model.eval()

    flow_model = FlowCVAE(in_channels=3, latent_dim=128, num_stations=num_stations, flow_layers=8).to(device)
    flow_model.load_state_dict(torch.load(flow_chk, map_location=device)['model_state_dict'])
    flow_model.eval()

    # Dataset
    dataset = SeismicSTFTDatasetWithMetadata(
        data_dir=ood_data_dir,
        event_file=ood_catalog,
        channels=['HH'],  # HH channels only
        magnitude_col='xM',
        station_list=station_list
    )
    
    print(f"Loaded {len(dataset)} HH OOD waveform files (post-training 2022-2024).")
    
    metric_keys = ['ssim', 'lsd', 'sc', 's_corr', 'sta_lta_err', 'mr_lsd', 'arias_err', 'env_corr', 'dtw', 'xcorr']
    results = {m: {k: [] for k in metric_keys} for m in ['Baseline', 'FullCov', 'Flow']}
    
    TARGET_FS = 100.0
    
    with torch.no_grad():
        for i in range(len(dataset)):
            try:
                spec, mag, loc, station_idx, meta = dataset[i]
                # Target waveform
                file_path = meta['file_path']
                file_name = os.path.basename(file_path)
                # Example: OOD_POST_01_ADVT_HH.mseed
                parts = file_name.split('_')
                if len(parts) >= 4:
                    event_id = "_".join(parts[:3])  # OOD_POST_01
                    station_name = parts[3]         # ADVT
                else:
                    event_id = meta['event_id']
                    station_name = station_list[station_idx.item()]
                
                print(f"Processing {event_id} at {station_name}...")
                
                st_gt = obspy.read(file_path)
                st_gt.resample(TARGET_FS)
                tr_gt = st_gt.select(component="Z")[0] if st_gt.select(component="Z") else st_gt[0]
                gt_wav = tr_gt.data.astype(np.float32)
                
                target_len = 7300
                if len(gt_wav) > target_len:
                    gt_wav = gt_wav[:target_len]
                elif len(gt_wav) < target_len:
                    gt_wav = np.pad(gt_wav, (0, target_len - len(gt_wav)))
                
                # Recalculate spectrogram
                f, t, Zxx = signal.stft(gt_wav, fs=TARGET_FS, nperseg=256, noverlap=192, nfft=256, boundary='zeros')
                mag_spec = np.abs(Zxx)
                mag_spec = np.log1p(mag_spec)
                mag_min, mag_max = mag_spec.min(), mag_spec.max()
                if mag_max > mag_min:
                    mag_spec = (mag_spec - mag_min) / (mag_max - mag_min + 1e-8)
                
                spec_in = torch.from_numpy(mag_spec).unsqueeze(0).unsqueeze(0).to(device).float().repeat(1, 3, 1, 1)
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
                
                wav_base = reconstruct_signal(base_spec, mag_min, mag_max, nperseg=256, noverlap=192, fs=TARGET_FS)
                wav_fc = reconstruct_signal(fc_spec, mag_min, mag_max, nperseg=256, noverlap=192, fs=TARGET_FS)
                wav_flow = reconstruct_signal(flow_spec, mag_min, mag_max, nperseg=256, noverlap=192, fs=TARGET_FS)
                
                # Calculate metrics
                min_len = min(len(gt_wav), len(wav_base), len(wav_fc), len(wav_flow))
                m_base = calculate_seismic_metrics(gt_wav[:min_len], wav_base[:min_len], orig_spec, base_spec, fs=TARGET_FS)
                m_fc = calculate_seismic_metrics(gt_wav[:min_len], wav_fc[:min_len], orig_spec, fc_spec, fs=TARGET_FS)
                m_flow = calculate_seismic_metrics(gt_wav[:min_len], wav_flow[:min_len], orig_spec, flow_spec, fs=TARGET_FS)
                
                for k in metric_keys:
                    results['Baseline'][k].append(m_base[k])
                    results['FullCov'][k].append(m_fc[k])
                    results['Flow'][k].append(m_flow[k])
                
                # Save visualizations
                os.makedirs(f"{output_dir}/specs", exist_ok=True)
                os.makedirs(f"{output_dir}/waveforms", exist_ok=True)
                
                # Spectrogram comparison
                fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                axes[0, 0].imshow(orig_spec, aspect='auto', origin='lower', cmap='viridis')
                axes[0, 0].set_title(f'Original - {event_id} @ {station_name}')
                axes[0, 1].imshow(base_spec, aspect='auto', origin='lower', cmap='viridis')
                axes[0, 1].set_title(f'Baseline (SSIM: {m_base["ssim"]:.3f})')
                axes[1, 0].imshow(fc_spec, aspect='auto', origin='lower', cmap='viridis')
                axes[1, 0].set_title(f'FullCov (SSIM: {m_fc["ssim"]:.3f})')
                axes[1, 1].imshow(flow_spec, aspect='auto', origin='lower', cmap='viridis')
                axes[1, 1].set_title(f'Flow (SSIM: {m_flow["ssim"]:.3f})')
                plt.tight_layout()
                plt.savefig(f"{output_dir}/specs/{event_id}_{station_name}_specs.png", dpi=150, bbox_inches='tight')
                plt.close()
                
                # Waveform comparison
                time = np.arange(min_len) / TARGET_FS
                fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
                axes[0].plot(time, gt_wav[:min_len], 'k-', linewidth=0.8, label='Ground Truth')
                axes[0].set_ylabel('Amplitude')
                axes[0].set_title(f'{event_id} @ {station_name} - HH Channel')
                axes[0].legend(loc='upper right')
                axes[0].grid(True, alpha=0.3)
                
                axes[1].plot(time, wav_base[:min_len], 'b-', linewidth=0.8, label=f'Baseline (XCorr: {m_base["xcorr"]:.3f})')
                axes[1].set_ylabel('Amplitude')
                axes[1].legend(loc='upper right')
                axes[1].grid(True, alpha=0.3)
                
                axes[2].plot(time, wav_fc[:min_len], 'g-', linewidth=0.8, label=f'FullCov (XCorr: {m_fc["xcorr"]:.3f})')
                axes[2].set_ylabel('Amplitude')
                axes[2].legend(loc='upper right')
                axes[2].grid(True, alpha=0.3)
                
                axes[3].plot(time, wav_flow[:min_len], 'r-', linewidth=0.8, label=f'Flow (XCorr: {m_flow["xcorr"]:.3f})')
                axes[3].set_ylabel('Amplitude')
                axes[3].set_xlabel('Time (s)')
                axes[3].legend(loc='upper right')
                axes[3].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(f"{output_dir}/waveforms/{event_id}_{station_name}_waveforms.png", dpi=150, bbox_inches='tight')
                plt.close()
                    
            except Exception as e:
                print(f"  Error processing {event_id}: {e}")
    
    # Print results
    print("\n" + "="*80)
    print("POST-TRAINING HH OOD SEISMIC METRICS SUMMARY (2022-2024)")
    print("="*80)
    print("| Model | SSIM ↑ | LSD ↓ | SC ↓ | S-Corr ↑ | STA/LTA Err ↓ | MR-LSD ↓ | Arias Err ↓ | Env Corr ↑ | DTW ↓ | XCorr ↑ |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    
    for model_name in ['Baseline', 'FullCov', 'Flow']:
        row = f"| **{model_name}** |"
        for k in metric_keys:
            vals = results[model_name][k]
            if len(vals) > 0:
                avg = np.mean(vals)
                if k in ['ssim', 's_corr', 'env_corr', 'xcorr']:
                    row += f" {avg:.4f} |"
                elif k in ['sc', 'sta_lta_err']:
                    row += f" {avg:.3f} |"
                else:
                    row += f" {avg:.2f} |"
            else:
                row += " N/A |"
        print(row)
    print("="*80)

if __name__ == "__main__":
    main()
