import os
import obspy
from pathlib import Path

def preprocess_ood_waveforms():
    raw_dir = Path("data/ood_waveforms/raw")
    filtered_dir = Path("data/ood_waveforms/filtered")
    filtered_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Preprocessing waveforms from {raw_dir}...")
    
    mseed_files = list(raw_dir.glob("*.mseed"))
    print(f"Found {len(mseed_files)} mseed files.")
    
    for f in mseed_files:
        try:
            st = obspy.read(str(f))
            st.detrend("linear")
            st.taper(max_percentage=0.05, type="cosine")
            # Broadband filter matching training data
            st.filter("bandpass", freqmin=0.5, freqmax=45.0)
            
            # Save to filtered directory
            out_path = filtered_dir / f.name
            st.write(str(out_path), format="MSEED")
            print(f"  Processed {f.name}")
        except Exception as e:
            print(f"  Error processing {f.name}: {e}")

if __name__ == "__main__":
    preprocess_ood_waveforms()
