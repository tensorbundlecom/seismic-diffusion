import os
import obspy
from pathlib import Path

def preprocess_koeri_ood():
    raw_dir = Path("data/ood_waveforms/koeri/raw")
    filtered_dir = Path("data/ood_waveforms/koeri/filtered")
    filtered_dir.mkdir(parents=True, exist_ok=True)
    
    # Target frequency matching training data
    TARGET_FS = 100.0
    
    print(f"Preprocessing waveforms from {raw_dir}...")
    
    mseed_files = list(raw_dir.glob("*.mseed"))
    print(f"Found {len(mseed_files)} mseed files.")
    
    processed_count = 0
    for f in mseed_files:
        try:
            st = obspy.read(str(f))
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

    print(f"\nPreprocessing finished. Successfully processed {processed_count} files.")

if __name__ == "__main__":
    preprocess_koeri_ood()
