import torch
import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from skimage.metrics import structural_similarity as ssim
import os
import sys
from pathlib import Path

# Add paths for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../')))
from ..core.stft_dataset_fixed import SeismicSTFTDatasetWithMetadata
from ..core.model_baseline_fixed import ConditionalVariationalAutoencoder
from ..core.model_full_cov import FullCovCVAE

def get_arias_intensity(v, fs=100):
    # Arias Intensity formula: (pi / 2g) * integral(a(t)^2 dt)
    # Here we treat sismogram as proxy for acceleration if unit is unknown
    dt = 1.0 / fs
    return (np.pi / (2 * 9.81)) * np.trapz(v**2, dx=dt)

def get_envelope_correlation(v1, v2):
    env1 = np.abs(signal.hilbert(v1))
    env2 = np.abs(signal.hilbert(v2))
    return np.corrcoef(env1, env2)[0, 1]

def griffin_lim(mag_spectrogram, n_fft=256, iterations=30):
    mag_spectrogram = mag_spectrogram.astype(np.complex64)
    rng = np.random.default_rng()
    phase = np.exp(2j * np.pi * rng.random(mag_spectrogram.shape))
    for _ in range(iterations):
        stft = mag_spectrogram * phase
        waveform = signal.istft(stft, nfft=n_fft)[1]
        _, _, next_stft = signal.stft(waveform, nfft=n_fft)
        if next_stft.shape != mag_spectrogram.shape:
             h, w = mag_spectrogram.shape
             next_stft = next_stft[:h, :w]
        phase = np.exp(1j * np.angle(next_stft))
    stft = mag_spectrogram * phase
    return signal.istft(stft, nfft=n_fft)[1]

def calculate_lsd(spec1, spec2):
    # Log-Spectral Distance
    # spec1, spec2 are magnitude spectrograms
    lsd = np.sqrt(np.mean((20 * np.log10(spec1 + 1e-8) - 20 * np.log10(spec2 + 1e-8))**2))
    return lsd

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Weights (relative to root)
    baseline_chk = "ML/autoencoder/checkpoints/checkpoints_cvae/20260207_163500/best_model.pt"
    full_cov_chk = "ML/autoencoder/experiments/FullCovariance/checkpoints/checkpoints_full_cov/koeri_catalog/full_cov_cvae_best.pt"
    
    print("Loading Models for Evaluation...")
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
    
    expected_shape = (129, 111)
    results = []
    
    # We'll evaluate all 18 OOD samples
    print(f"Starting evaluation on {len(test_dataset)} samples...")
    
    for i in range(len(test_dataset)):
        spec, mag, loc, station_idx, meta = test_dataset[i]
        event_id = meta['event_id']
        station_name = meta['station_name']
        
        # Prepare inputs with interpolation
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
        
        # Focus on Z channel (index 2)
        s_orig = spec_in[0, 2].cpu().numpy()
        s_base = recon_base[0, 2].cpu().numpy()
        s_full = recon_full[0, 2].cpu().numpy()
        
        w_orig = griffin_lim(s_orig)
        w_base = griffin_lim(s_base)
        w_full = griffin_lim(s_full)
        
        # 1. Cross-Correlation
        cc_base = np.corrcoef(w_orig, w_base)[0, 1]
        cc_full = np.corrcoef(w_orig, w_full)[0, 1]
        
        # 2. DTW (on a downsampled version to keep it fast)
        ds_orig = w_orig[::10]
        ds_base = w_base[::10]
        ds_full = w_full[::10]
        # Use abs difference for scalar sequences
        scalar_dist = lambda x, y: np.abs(x - y)
        dtw_base, _ = fastdtw(ds_orig, ds_base, dist=scalar_dist)
        dtw_full, _ = fastdtw(ds_orig, ds_full, dist=scalar_dist)
        
        # 3. LSD
        lsd_base = calculate_lsd(s_orig, s_base)
        lsd_full = calculate_lsd(s_orig, s_full)
        
        # 4. SSIM (on spectrograms)
        ssim_base = ssim(s_orig, s_base, data_range=s_orig.max() - s_orig.min())
        ssim_full = ssim(s_orig, s_full, data_range=s_orig.max() - s_orig.min())
        
        # 5. Arias Intensity Error
        ai_orig = get_arias_intensity(w_orig)
        ai_base = get_arias_intensity(w_base)
        ai_full = get_arias_intensity(w_full)
        ai_err_base = abs(ai_orig - ai_base) / (ai_orig + 1e-8)
        ai_err_full = abs(ai_orig - ai_full) / (ai_orig + 1e-8)
        
        # 6. Envelope Correlation
        env_cc_base = get_envelope_correlation(w_orig, w_base)
        env_cc_full = get_envelope_correlation(w_orig, w_full)

        results.append({
            'event_id': event_id,
            'station': station_name,
            'cc_baseline': cc_base,
            'cc_fullcov': cc_full,
            'dtw_baseline': dtw_base,
            'dtw_fullcov': dtw_full,
            'lsd_baseline': lsd_base,
            'lsd_fullcov': lsd_full,
            'ssim_baseline': ssim_base,
            'ssim_fullcov': ssim_full,
            'ai_err_baseline': ai_err_base,
            'ai_err_fullcov': ai_err_full,
            'env_cc_baseline': env_cc_base,
            'env_cc_fullcov': env_cc_full
        })
        print(f"Evaluated {event_id} - {station_name}")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(os.path.dirname(__file__), "..", "results", "ood_metrics_comparison.csv"), index=False)
    
    # Summary Table
    summary = df.mean(numeric_only=True)
    print("\n" + "="*50)
    print("AVERAGE METRICS ACROSS ALL OOD SAMPLES")
    print("="*50)
    print(f"{'Metric':<25} | {'Baseline':<12} | {'Full Cov':<12}")
    print("-" * 55)
    metrics_list = ['cc', 'dtw', 'lsd', 'ssim', 'ai_err', 'env_cc']
    for m in metrics_list:
        b_val = summary[f"{m}_baseline"]
        f_val = summary[f"{m}_fullcov"]
        print(f"{m:<25} | {b_val:<12.4f} | {f_val:<12.4f}")
    
    # Winner
    print("\n" + "="*50)
    print("WINNER PER METRIC")
    print("="*50)
    for m in metrics_list:
        b_val = summary[f"{m}_baseline"]
        f_val = summary[f"{m}_fullcov"]
        if m in ['dtw', 'lsd', 'ai_err']: # Lower is better
            winner = "Baseline" if b_val < f_val else "Full Cov"
        else: # Higher is better
            winner = "Baseline" if b_val > f_val else "Full Cov"
        print(f"{m:<25}: {winner}")

if __name__ == "__main__":
    main()
