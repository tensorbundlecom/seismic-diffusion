import obspy
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

files = [
    "data/ood_waveforms/koeri/filtered/OOD_K_02_CTKS.mseed",
    "data/ood_waveforms/koeri/filtered/OOD_K_06_ADVT.mseed",
    "data/ood_waveforms/koeri/filtered/OOD_K_06_EDC.mseed",
    "data/ood_waveforms/koeri/filtered/OOD_K_08_ADVT.mseed"
]

for f_path in files:
    try:
        st = obspy.read(f_path)
        print(f"\nFile: {f_path}")
        for tr in st:
            data = tr.data
            max_amp = np.max(np.abs(data))
            mean_amp = np.mean(np.abs(data))
            snr = max_amp / (mean_amp + 1e-8)
            print(f"  Channel: {tr.stats.channel}")
            print(f"    Length: {len(data)} samples")
            print(f"    Sampling Rate: {tr.stats.sampling_rate} Hz")
            print(f"    Max Amp: {max_amp:.2f}")
            print(f"    Mean Abs Amp: {mean_amp:.2f}")
            print(f"    SNR-ish (Max/Mean): {snr:.2f}")
            
            # Check for zeros or weird patterns
            zero_count = np.sum(data == 0)
            if zero_count > 0:
                print(f"    Zeros: {zero_count} samples")
            
    except Exception as e:
        print(f"Error reading {f_path}: {e}")
