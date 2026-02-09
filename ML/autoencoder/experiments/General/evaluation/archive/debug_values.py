import torch
import numpy as np
import os
import sys
import obspy

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from ML.autoencoder.experiments.General.core.stft_dataset import SeismicSTFTDatasetWithMetadata
from ML.autoencoder.experiments.General.core.model_baseline import ConditionalVariationalAutoencoder

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Dataset (Filtered OOD)
    dataset = SeismicSTFTDatasetWithMetadata(
        data_dir="data/ood_waveforms/filtered",
        event_file="data/events/ood_catalog.txt",
        channels=['HH', 'BH'],
        magnitude_col='ML'
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Check index 0 (should be OOD_1_EDC_BH.mseed or similar)
    idx = 0
    spec, mag, loc, station_idx, meta = dataset[idx]
    
    print(f"\n--- Sample {idx} Metadata ---")
    print(f"File: {meta['file_name']}")
    print(f"Mag Min: {meta.get('mag_min')}")
    print(f"Mag Max: {meta.get('mag_max')}")
    print(f"Station: {meta['station_name']} (idx: {station_idx.item()})")
    
    print(f"\n--- Input Spectrogram Stats ---")
    print(f"Shape: {spec.shape}")
    print(f"Min: {spec.min().item()}")
    print(f"Max: {spec.max().item()}")
    print(f"Mean: {spec.mean().item()}")
    
    # Load Baseline Model
    baseline_chk = "ML/autoencoder/checkpoints/checkpoints_cvae/20260207_163500/best_model.pt"
    base_model = ConditionalVariationalAutoencoder(in_channels=3, latent_dim=128, num_stations=125).to(device)
    base_st = torch.load(baseline_chk, map_location=device)
    base_model.load_state_dict(base_st['model_state_dict'], strict=False)
    base_model.eval()
    
    # Prepare Input
    spec_in = spec.unsqueeze(0)
    # Resize if needed
    target_size = (129, 111)
    if spec_in.shape[2:] != target_size:
        print(f"Interpolating from {spec_in.shape[2:]} to {target_size}")
        spec_in = torch.nn.functional.interpolate(spec_in, size=target_size, mode='bilinear', align_corners=False)
    
    spec_in = spec_in.to(device)
    mag_in = mag.unsqueeze(0).to(device)
    loc_in = loc.unsqueeze(0).to(device)
    station_in = station_idx.unsqueeze(0).to(device)
    
    # Run Model
    with torch.no_grad():
        recon, _, _ = base_model(spec_in, mag_in, loc_in, station_in)
        
    print(f"\n--- Model Output Stats ---")
    print(f"Shape: {recon.shape}")
    print(f"Min: {recon.min().item()}")
    print(f"Max: {recon.max().item()}")
    print(f"Mean: {recon.mean().item()}")
    
    # Denormalize manually
    mag_min = meta.get('mag_min', 0.0)
    mag_max = meta.get('mag_max', 1.0)
    
    recon_np = recon[0, 2].cpu().numpy()
    
    # Step 1: Denorm
    denorm = recon_np * (mag_max - mag_min) + mag_min
    print(f"\n--- Denormalized Stats ---")
    print(f"Min: {denorm.min()}")
    print(f"Max: {denorm.max()}")
    
    # Step 2: Inverse Log
    inv_log = np.expm1(denorm)
    print(f"\n--- Inverse Log Stats ---")
    print(f"Min: {inv_log.min()}")
    print(f"Max: {inv_log.max()}")
    
    # Load Filtered Waveform (Z channel)
    st_filt = obspy.read(meta['file_path'])
    tr_filt = st_filt.select(component="Z")[0] if st_filt.select(component="Z") else st_filt[0]
    filt_data = tr_filt.data
    
    print(f"\n--- Filtered Waveform Stats ---")
    print(f"Min: {filt_data.min()}")
    print(f"Max: {filt_data.max()}")
    print(f"Abs Max: {np.max(np.abs(filt_data))}")
    
    # Load Raw Waveform if possible
    raw_path = meta['file_path'].replace("filtered", "raw")
    if os.path.exists(raw_path):
        st_raw = obspy.read(raw_path)
        tr_raw = st_raw.select(component="Z")[0] if st_raw.select(component="Z") else st_raw[0]
        raw_data = tr_raw.data
        print(f"\n--- Raw Waveform Stats ---")
        print(f"Min: {raw_data.min()}")
        print(f"Max: {raw_data.max()}")
        print(f"Abs Max: {np.max(np.abs(raw_data))}")
    else:
        print(f"\nRaw path not found: {raw_path}")

    import librosa
    
    # Run Griffin-Lim
    # Note: dataset uses nperseg=256, noverlap=192 -> hop=64
    nperseg=256
    hop_length=64
    win_length=256
    
    # Librosa expects specific shape/args
    # inv_log is already 2D (F, T) because we selected [0, 2] earlier
    spec_z = inv_log 
    
    reconstructed_waveform = librosa.griffinlim(
        spec_z, 
        n_iter=32, 
        hop_length=hop_length, 
        win_length=win_length,
         center=True
    )
    
    print(f"\n--- Reconstructed Waveform Stats ---")
    print(f"Min: {reconstructed_waveform.min()}")
    print(f"Max: {reconstructed_waveform.max()}")
    print(f"Abs Max: {np.max(np.abs(reconstructed_waveform))}")
    
    ratio = np.max(np.abs(filt_data)) / (np.max(np.abs(reconstructed_waveform)) + 1e-9)
    print(f"\nAmplitude Ratio (Filtered / Recon): {ratio:.4f}")

if __name__ == "__main__":
    main()
