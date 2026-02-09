import os
import obspy
from pathlib import Path

def preprocess_post_training_ood():
    """Preprocess HH channel OOD waveforms to match training data pipeline."""
    raw_dir = Path("data/ood_waveforms/post_training/raw/HH")
    filtered_dir = Path("data/ood_waveforms/post_training/filtered/HH")
    filtered_dir.mkdir(parents=True, exist_ok=True)
    
    # Target frequency matching training data
    TARGET_FS = 100.0
    
    print(f"Preprocessing HH waveforms from {raw_dir}...")
    
    mseed_files = list(raw_dir.glob("*.mseed"))
    print(f"Found {len(mseed_files)} HH mseed files.")
    
    processed_count = 0
    for f in mseed_files:
        try:
            st = obspy.read(str(f))
            
            # Verify HH channels
            channel_types = set([tr.stats.channel[:2] for tr in st])
            if "HH" not in channel_types:
                print(f"  Skipping {f.name}: Not HH channel ({channel_types})")
                continue
            
            # Apply same preprocessing as training data
            st.detrend("linear")
            st.taper(max_percentage=0.05, type="cosine")
            
            # Resample to 100Hz (Training set standard)
            st.resample(TARGET_FS)
            
            # Broadband filter matching training data [0.5 - 45 Hz]
            st.filter("bandpass", freqmin=0.5, freqmax=45.0)
            
            # Save to filtered directory
            out_path = filtered_dir / f.name
            st.write(str(out_path), format="MSEED")
            processed_count += 1
            if processed_count % 10 == 0:
                print(f"  Processed {processed_count} files...")
        except Exception as e:
            print(f"  Error processing {f.name}: {e}")

    print(f"\nPreprocessing finished. Successfully processed {processed_count}/{len(mseed_files)} HH files.")

if __name__ == "__main__":
    preprocess_post_training_ood()
