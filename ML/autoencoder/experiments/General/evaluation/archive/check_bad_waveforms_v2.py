import obspy
import numpy as np
from pathlib import Path

files = [
    "data/ood_waveforms/koeri/filtered/BH/OOD_K_02_CTKS_BH.mseed",
    "data/ood_waveforms/koeri/filtered/BH/OOD_K_06_ADVT_BH.mseed",
    "data/ood_waveforms/koeri/filtered/BH/OOD_K_06_EDC_BH.mseed"
]

for f_path in files:
    try:
        st = obspy.read(f_path)
        print(f"\nFile: {f_path}")
        # Merge if multiple traces for same component
        st.merge(fill_value=0)
        
        for tr in st:
            data = tr.data.astype(np.float32)
            max_amp = np.max(np.abs(data))
            std_amp = np.std(data)
            snr = max_amp / (std_amp + 1e-8)
            
            print(f"  Channel: {tr.stats.channel}")
            print(f"    Length: {len(data)} samples ({len(data)/tr.stats.sampling_rate:.2f} s)")
            print(f"    Max Amp: {max_amp:.2f}")
            print(f"    Std Dev: {std_amp:.2f}")
            print(f"    Simple SNR: {snr:.2f}")
            
            # Check for clipped data or large spikes
            if max_amp > 1e6:
                print("    WARNING: Possible clipping or instrument artifact (Very high amplitude)")
            
            # Check if signal is flat
            if std_amp < 1e-3:
                print("    WARNING: Very flat signal (Dead channel?)")

    except Exception as e:
        print(f"Error reading {f_path}: {e}")
