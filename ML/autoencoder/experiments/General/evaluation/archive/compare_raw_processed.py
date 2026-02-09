import obspy
import matplotlib.pyplot as plt
import numpy as np
import os

def compare_raw_vs_processed(indices=[5, 12, 18], output_path="ML/autoencoder/experiments/General/visualizations/raw_vs_processed_debug.png"):
    # 1. Get filtered filenames
    filtered_dir = "data/filtered_waveforms/HH"
    filtered_files = sorted([f for f in os.listdir(filtered_dir) if f.endswith(".mseed")])
    
    # 2. Get corresponding raw filenames
    raw_dir = "data/waveforms/HH"
    # Note: Filenames should match generally, but let's be careful
    
    selected_files = []
    for idx in indices:
        if idx < len(filtered_files):
            fname = filtered_files[idx]
            filt_path = os.path.join(filtered_dir, fname)
            
            # Construct raw path
            # Filtered: "query?eventid=10257150_YLV_HH.mseed"
            # Raw might be same or slightly different in structure depending on download script
            # Let's assume name match for now
            raw_path = os.path.join(raw_dir, fname)
            
            if not os.path.exists(raw_path):
                print(f"Raw file not found: {raw_path}")
                # Try finding by event ID match if exact name fails
                event_id = fname.split('eventid=')[1].split('_')[0]
                candidates = [f for f in os.listdir(raw_dir) if event_id in f]
                if candidates:
                    raw_path = os.path.join(raw_dir, candidates[0])
                else:
                    print(f"  No raw candidate found for {event_id}")
                    continue
            
            selected_files.append((raw_path, filt_path))

    if not selected_files:
        print("No files to compare.")
        return

    fig, axes = plt.subplots(len(selected_files), 2, figsize=(20, 4 * len(selected_files)))
    if len(selected_files) == 1: axes = [axes] # Handle single row case

    for i, (raw_p, filt_p) in enumerate(selected_files):
        # Read Raw
        st_raw = obspy.read(raw_p)
        tr_raw = st_raw.select(component="Z")[0] if st_raw.select(component="Z") else st_raw[0]
        
        # Read Filtered
        st_filt = obspy.read(filt_p)
        tr_filt = st_filt.select(component="Z")[0] if st_filt.select(component="Z") else st_filt[0]
        
        # Raw Plot
        ax_r = axes[i, 0]
        ax_r.plot(tr_raw.times(), tr_raw.data, color='gray', lw=0.8)
        ax_r.set_title(f"RAW SOURCE: {os.path.basename(raw_p)}\nRange: [{tr_raw.data.min():.1f}, {tr_raw.data.max():.1f}]")
        ax_r.grid(True, alpha=0.3)
        
        # Filtered Plot
        ax_f = axes[i, 1]
        ax_f.plot(tr_filt.times(), tr_filt.data, color='blue', lw=0.8)
        ax_f.set_title(f"PROCESSED (Filtered): {os.path.basename(filt_p)}\nRange: [{tr_filt.data.min():.1f}, {tr_filt.data.max():.1f}]")
        ax_f.grid(True, alpha=0.3)
        
        # Check for zeros/flatlines
        if np.all(tr_raw.data == 0):
            ax_r.text(0.5, 0.5, "ALL ZEROS", transform=ax_r.transAxes, color='red', fontsize=12, ha='center')
        
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Comparison saved to {output_path}")

if __name__ == "__main__":
    compare_raw_vs_processed()
