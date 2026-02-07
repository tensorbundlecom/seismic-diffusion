import os
import json
import pandas as pd
import obspy
from pathlib import Path
from tqdm import tqdm
import sys

# Add project root for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../')))
from ML.autoencoder.stft_dataset_with_metadata import SeismicSTFTDatasetWithMetadata

def bandpass(stream, magnitude, filter_frequencies):
    channel_type = stream[0].stats.channel[:2]
    mag_key = "4+" if magnitude >= 4 else f"{int(magnitude)}-{int(magnitude)+1}"
    
    if channel_type in filter_frequencies:
        freqs = filter_frequencies[channel_type].get(mag_key, filter_frequencies[channel_type].get("4+", [0.1, 5.0]))
    else:
        freqs = [0.1, 20.0]
    
    low_freq, high_freq = freqs
    stream.detrend("demean")
    stream.detrend("linear")
    stream.taper(max_percentage=0.05, type="cosine")
    try:
        stream.filter("bandpass", freqmin=low_freq, freqmax=high_freq)
    except:
        pass
    return stream

def main():
    raw_dir = Path("data/ood_waveforms/raw")
    output_dir = Path("data/ood_waveforms/filtered")
    
    with open("data/filter_frequencies.json", "r") as f:
        filter_frequencies = json.load(f)

    # Load OOD catalog
    catalog_path = "data/events/ood_catalog.txt"
    catalog_df = pd.read_csv(catalog_path, sep='\t')
    mag_map = pd.Series(catalog_df.ML.values, index=catalog_df['Deprem Kodu']).to_dict()

    mseed_files = list(raw_dir.rglob("*.mseed"))
    print(f"Preprocessing {len(mseed_files)} OOD files...")

    for file_path in tqdm(mseed_files):
        filename = file_path.name
        event_id = filename.split('_')[0]
        
        rel_path = file_path.relative_to(raw_dir)
        out_path = output_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            st = obspy.read(str(file_path))
            magnitude = mag_map.get(event_id, 3.0)
            filtered_st = bandpass(st, magnitude, filter_frequencies)
            filtered_st.write(str(out_path), format="MSEED")
        except Exception as e:
            continue
    print("OOD Preprocessing complete.")

if __name__ == "__main__":
    main()
