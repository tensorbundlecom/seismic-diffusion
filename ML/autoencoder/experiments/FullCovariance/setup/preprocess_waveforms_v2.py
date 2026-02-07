import obspy
import pandas as pd
import json
import os
from tqdm import tqdm
from pathlib import Path

def bandpass(stream, magnitude, filter_frequencies):
    channel_type = stream[0].stats.channel[:2]
    if magnitude >= 4:
        mag_key = "4+"
    else:
        mag_key = f"{int(magnitude)}-{int(magnitude)+1}"
    
    if channel_type in filter_frequencies:
        if mag_key in filter_frequencies[channel_type]:
            freqs = filter_frequencies[channel_type][mag_key]
        else:
            # Fallback to nearest range if available
            freqs = filter_frequencies[channel_type].get("4+", [0.1, 5.0])
    else:
        # Generic fallback
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
    with open("data/filter_frequencies.json", "r") as f:
        filter_frequencies = json.load(f)

    # Load catalog to get magnitudes
    catalog_path = "data/events/koeri_catalog.txt"
    catalog_df = pd.read_csv(catalog_path, sep='\t')
    mag_map = pd.Series(catalog_df.xM.values, index=catalog_df['Deprem Kodu']).to_dict()

    waveforms_dir = Path("data/waveforms")
    out_dir = Path("data/filtered_waveforms")
    
    mseed_files = list(waveforms_dir.rglob("*.mseed"))
    print(f"Found {len(mseed_files)} files to process.")

    for file_path in tqdm(mseed_files):
        # Filename usually: eventid_station_type.mseed
        filename = file_path.name
        event_id = filename.split('_')[0]
        
        rel_path = file_path.relative_to(waveforms_dir)
        out_path = out_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists():
            continue

        try:
            st = obspy.read(str(file_path))
            magnitude = mag_map.get(event_id, 3.0) # Default to 3.0 if not found
            
            filtered_st = bandpass(st, magnitude, filter_frequencies)
            filtered_st.write(str(out_path), format="MSEED")
        except Exception as e:
            # print(f"Error processing {filename}: {e}")
            continue

if __name__ == "__main__":
    main()
