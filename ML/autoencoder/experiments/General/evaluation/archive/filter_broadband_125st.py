import obspy
from pathlib import Path
import os
from tqdm import tqdm

def filter_data():
    input_dir = Path("data/waveforms/HH")
    output_dir = Path("data/filtered_waveforms_broadband/HH")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    files = list(input_dir.glob("*.mseed"))
    print(f"Found {len(files)} files to filter.")
    
    for f_path in tqdm(files):
        try:
            st = obspy.read(str(f_path))
            
            # 0.5 - 40 Hz Bandpass
            st.detrend('linear')
            st.taper(max_percentage=0.05, type='cosine')
            st.filter('bandpass', freqmin=0.5, freqmax=40.0, corners=4, zerophase=True)
            
            # Save
            out_path = output_dir / f_path.name
            st.write(str(out_path), format="MSEED")
            
        except Exception as e:
            print(f"Error filtering {f_path.name}: {e}")

if __name__ == "__main__":
    filter_data()
