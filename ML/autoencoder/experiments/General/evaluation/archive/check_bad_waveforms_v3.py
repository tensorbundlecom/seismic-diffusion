import obspy
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

f_path = "data/ood_waveforms/koeri/filtered/BH/OOD_K_02_CTKS_BH.mseed"
st = obspy.read(f_path)
st.merge(fill_value=0)
tr = st.select(component="Z")[0]
data = tr.data.astype(np.float32)

print(f"Stats for {f_path}:")
print(f"  First 10 samples: {data[:10]}")
print(f"  Last 10 samples: {data[-10:]}")
print(f"  Mean: {np.mean(data):.6f}")
print(f"  Std: {np.std(data):.6f}")
print(f"  Min: {np.min(data):.6f}")
print(f"  Max: {np.max(data):.6f}")

# Check for large jumps (derivative)
diff = np.diff(data)
print(f"  Max Diff: {np.max(np.abs(diff)):.6f}")
print(f"  Mean Abs Diff: {np.mean(np.abs(diff)):.6f}")

# Check for repetitive patterns or zeros
unique, counts = np.unique(data, return_counts=True)
print(f"  Unique values count: {len(unique)} / {len(data)}")
if len(unique) < len(data) / 2:
    print("  WARNING: High number of duplicate values detected.")
