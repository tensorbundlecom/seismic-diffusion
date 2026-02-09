import torch
import numpy as np
import os
import sys
import obspy
import librosa
import json
from scipy import signal

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../')))

from ML.autoencoder.experiments.General.core.stft_dataset import SeismicSTFTDatasetWithMetadata
from ML.autoencoder.experiments.General.core.model_baseline import ConditionalVariationalAutoencoder

def reconstruct_signal(magnitude_spec, mag_min=0.0, mag_max=1.0, nperseg=256, noverlap=192, n_iter=64):
    spec = magnitude_spec.copy()
    if mag_max > mag_min:
        spec = spec * (mag_max - mag_min) + mag_min
    spec = np.expm1(spec)
    return librosa.griffinlim(spec, n_iter=n_iter, hop_length=nperseg - noverlap, win_length=nperseg, center=True)

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
        for i in [0]: # Check first sample
            spec, mag, loc, station_idx, meta = dataset[i]
            st_gt = obspy.read(meta['file_path'])
            st_gt.resample(TARGET_FS)
            tr_gt = st_gt.select(component="Z")[0] if st_gt.select(component="Z") else st_gt[0]
            gt_wav = tr_gt.data.astype(np.float32)
            
            target_len = 7300
            if len(gt_wav) > target_len: gt_wav = gt_wav[:target_len]
            elif len(gt_wav) < target_len: gt_wav = np.pad(gt_wav, (0, target_len - len(gt_wav)))
            
            f, t, Zxx = signal.stft(gt_wav, fs=TARGET_FS, nperseg=256, noverlap=192, nfft=256, boundary='zeros')
            mag_spec = np.log1p(np.abs(Zxx))
            mag_min, mag_max = mag_spec.min(), mag_spec.max()
            
            # Normalize for model
            if mag_max > mag_min:
                mag_spec_norm = (mag_spec - mag_min) / (mag_max - mag_min + 1e-8)
            else:
                mag_spec_norm = mag_spec
            
            spec_in = torch.from_numpy(mag_spec_norm).unsqueeze(0).unsqueeze(0).to(device).float()
            spec_in = spec_in.repeat(1, 3, 1, 1)
            if spec_in.shape[2:] != (129, 111):
                spec_in = torch.nn.functional.interpolate(spec_in, size=(129, 111), mode='bilinear', align_corners=False)
            
            r_base, _, _ = base_model(spec_in, mag.unsqueeze(0).to(device), loc.unsqueeze(0).to(device), station_idx.unsqueeze(0).to(device))
            base_spec = r_base[0, 2].cpu().numpy()
            
            wav_base = reconstruct_signal(base_spec, mag_min, mag_max)
            
            def get_arias(sig):
                return (np.pi / (2 * 9.81)) * np.trapz(sig**2, dx=0.01)
            
            a_target = get_arias(gt_wav)
            a_pred = get_arias(wav_base)
            
            print(f"Sample {i}:")
            print(f"  GT Max: {np.max(np.abs(gt_wav)):.4f}")
            print(f"  Pred Max: {np.max(np.abs(wav_base)):.4f}")
            print(f"  Arias Target: {a_target:.8f}")
            print(f"  Arias Pred: {a_pred:.8f}")
            print(f"  Arias Error: {np.abs(a_target - a_pred) / (a_target + 1e-8):.4f}")

if __name__ == "__main__":
    main()
