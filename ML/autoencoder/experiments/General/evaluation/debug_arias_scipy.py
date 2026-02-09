import torch
import numpy as np
import os
import sys
import obspy
import librosa
from scipy import signal
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../')))

from ML.autoencoder.experiments.General.core.stft_dataset import SeismicSTFTDatasetWithMetadata
from ML.autoencoder.experiments.General.core.model_baseline import ConditionalVariationalAutoencoder

def scipy_griffin_lim(magnitude_spec, n_iter=64, fs=100.0, nperseg=256, noverlap=192, nfft=256):
    """Griffin-Lim implementation using scipy.signal.stft/istft to ensure scaling consistency."""
    spec = magnitude_spec.copy()
    
    # Initialize with random phase
    phase = np.exp(2j * np.pi * np.random.rand(*spec.shape))
    
    for i in range(n_iter):
        stft_complex = spec * phase
        _, waveform = signal.istft(stft_complex, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft, boundary='zeros')
        
        # Take STFT of the reconstructed waveform to get new phase
        _, _, new_Zxx = signal.stft(waveform, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft, boundary='zeros')
        
        # Ensure shapes match (scipy.stft might return slightly different number of bins/frames)
        if new_Zxx.shape != spec.shape:
             # This should not happen if params are identical, but let's be safe
             min_f = min(new_Zxx.shape[0], spec.shape[0])
             min_t = min(new_Zxx.shape[1], spec.shape[1])
             phase = np.exp(1j * np.angle(new_Zxx[:min_f, :min_t]))
             # Pad or truncate spec if needed (though it shouldn't be)
        else:
            phase = np.exp(1j * np.angle(new_Zxx))
            
    # Final reconstruction
    stft_complex = spec * phase
    _, waveform = signal.istft(stft_complex, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft, boundary='zeros')
    return waveform

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    baseline_chk = "ML/autoencoder/experiments/General/checkpoints/baseline_external_best.pt"
    station_list_file = "data/station_list_external_full.json"
    ood_catalog = "data/events/ood_catalog_koeri.txt"
    ood_data_dir = "data/ood_waveforms/koeri/filtered"

    with open(station_list_file, 'r') as f:
        station_list = json.load(f)

    base_model = ConditionalVariationalAutoencoder(in_channels=3, latent_dim=128, num_stations=46).to(device)
    base_model.load_state_dict(torch.load(baseline_chk, map_location=device)['model_state_dict'])
    base_model.eval()

    dataset = SeismicSTFTDatasetWithMetadata(
        data_dir=ood_data_dir,
        event_file=ood_catalog,
        channels=['HH', 'BH'],
        magnitude_col='ML',
        station_list=station_list
    )
    
    TARGET_FS = 100.0
    
    with torch.no_grad():
        for i in [0]:
            spec, mag, loc, station_idx, meta = dataset[i]
            st_gt = obspy.read(meta['file_path'])
            st_gt.resample(TARGET_FS)
            tr_gt = st_gt.select(component="Z")[0] if st_gt.select(component="Z") else st_gt[0]
            gt_wav = tr_gt.data.astype(np.float32)
            
            target_len = 7300
            if len(gt_wav) > target_len: gt_wav = gt_wav[:target_len]
            elif len(gt_wav) < target_len: gt_wav = np.pad(gt_wav, (0, target_len - len(gt_wav)))
            
            f, t, Zxx = signal.stft(gt_wav, fs=TARGET_FS, nperseg=256, noverlap=192, nfft=256, boundary='zeros')
            mag_spec = np.abs(Zxx)
            
            # STFT Baseline: reconstruct from real phase to check energy
            _, wav_perfect = signal.istft(Zxx, fs=TARGET_FS, nperseg=256, noverlap=192, nfft=256, boundary='zeros')
            
            # Model Inference
            mag_log = np.log1p(mag_spec)
            mag_min, mag_max = mag_log.min(), mag_log.max()
            mag_norm = (mag_log - mag_min) / (mag_max - mag_min + 1e-8)
            
            spec_in = torch.from_numpy(mag_norm).unsqueeze(0).unsqueeze(0).to(device).float().repeat(1, 3, 1, 1)
            if spec_in.shape[2:] != (129, 111):
                spec_in = torch.nn.functional.interpolate(spec_in, size=(129, 111), mode='bilinear', align_corners=False)
            
            r_base, _, _ = base_model(spec_in, mag.unsqueeze(0).to(device), loc.unsqueeze(0).to(device), station_idx.unsqueeze(0).to(device))
            base_spec_norm = r_base[0, 2].cpu().numpy()
            
            # Denormalize
            base_spec_log = base_spec_norm * (mag_max - mag_min) + mag_min
            base_spec_lin = np.expm1(base_spec_log)
            
            # Reconstruct with Scipy Griffin-Lim
            wav_scipy = scipy_griffin_lim(base_spec_lin, n_iter=64)
            
            def get_arias(sig):
                return (np.pi / (2 * 9.81)) * np.trapz(sig**2, dx=0.01)
            
            a_target = get_arias(gt_wav)
            a_perfect = get_arias(wav_perfect)
            a_pred = get_arias(wav_scipy)
            
            print(f"Sample {i}:")
            print(f"  GT Max: {np.max(np.abs(gt_wav)):.4f}")
            print(f"  Perfect Recon Max: {np.max(np.abs(wav_perfect)):.4f}")
            print(f"  Scipy Recon Max: {np.max(np.abs(wav_scipy)):.4f}")
            print(f"  Arias Target: {a_target:.4f}")
            print(f"  Arias Perfect: {a_perfect:.4f}")
            print(f"  Arias Pred: {a_pred:.4f}")
            print(f"  Arias Error (Scipy): {np.abs(a_target - a_pred) / (a_target + 1e-8):.4f}")

if __name__ == "__main__":
    main()
