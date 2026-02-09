import obspy
import matplotlib.pyplot as plt
import numpy as np
import os

def compare_filters(raw_mseed_path, output_path="ML/autoencoder/experiments/General/visualizations/filter_comparison_0.1_5hz.png"):
    st = obspy.read(raw_mseed_path)
    # Pick Z channel
    z_tr = None
    for tr in st:
        if tr.stats.channel.endswith('Z'):
            z_tr = tr
            break
    if z_tr is None: z_tr = st[0]
    
    # Create two copies for different filters
    st_broad = st.copy()
    st_narrow = st.copy()
    
    # Cleanup (standard steps)
    for s in [st_broad, st_narrow]:
        s.detrend("demean")
        s.detrend("linear")
        s.taper(max_percentage=0.05, type="cosine")
    
    # Apply filters
    # 1. Our "Broad" filter: 0.5 - 40 Hz
    st_broad.filter("bandpass", freqmin=0.5, freqmax=40.0)
    
    # 2. User suggested "Narrow" filter: 0.1 - 5 Hz
    st_narrow.filter("bandpass", freqmin=0.1, freqmax=5.0)
    
    # Plotting
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    
    tr_b = st_broad.select(component="Z")[0] if st_broad.select(component="Z") else st_broad[0]
    tr_n = st_narrow.select(component="Z")[0] if st_narrow.select(component="Z") else st_narrow[0]
    
    time = np.linspace(0, tr_b.stats.npts / tr_b.stats.sampling_rate, tr_b.stats.npts)
    
    axes[0].plot(time, tr_b.data, color='black', lw=0.7)
    axes[0].set_title(f"Broadband Filter (0.5 - 40.0 Hz) - Details preserved\n{tr_b.id}")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(alpha=0.3)
    
    axes[1].plot(time, tr_n.data, color='red', lw=1.0)
    axes[1].set_title(f"Narrow Filter (0.1 - 5.0 Hz) - Smooth, low-frequency focus")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Comparison saved to {output_path}")

if __name__ == "__main__":
    # Use the raw file
    raw_file = "data/waveforms/HH/query?eventid=10166472_ADVT_HH.mseed"
    compare_filters(raw_file)
