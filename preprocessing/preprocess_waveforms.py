import obspy
import pandas as pd
import json
import os
from tqdm import tqdm

def bandpass(stream, magnitude, filter_frequencies):
    """
    Apply a bandpass filter to each trace in the stream.
    
    Parameters:
    -----------
    stream : obspy.Stream
        Stream object containing traces
    filter_frequencies : tuple
        Tuple containing (low_freq, high_freq)
    
    Returns:
    --------
    obspy.Stream : Filtered stream
    """
    channel_type = stream[0].stats.channel[:2]
    # print(channel_type)
    if magnitude >= 4:
        magnitude = "4+"
    else:
        magnitude = f"{int(magnitude)}-{int(magnitude)+1}"
    if channel_type in ["HN", "BH", "EH", "HH"]:
        freqs = filter_frequencies[channel_type][magnitude]
    else:
        raise ValueError(f"Unknown channel type: {channel_type}")
    
    low_freq, high_freq = freqs

    # demean, detrend, taper and filter
    stream.detrend("demean")
    stream.detrend("linear")
    stream.taper(max_percentage=0.05, type="cosine")
    stream.filter("bandpass", freqmin=low_freq, freqmax=high_freq)

    # stream.plot()
    return stream

def main():
    # Load filter frequencies from JSON
    with open("data/filter_frequencies.json", "r") as f:
        filter_frequencies = json.load(f)

    waveforms_df = pd.read_csv("data/high_snr_low_gap_waveforms.csv")


    for index, row in tqdm(waveforms_df.iterrows(), total=waveforms_df.shape[0]):
        file_name = "/".join(row["waveform_file"].split("/")[-2:])
        file_path = os.path.join("data/waveforms", file_name)
        out_path = os.path.join("data/filtered_waveforms", file_name)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)


        if os.path.exists(out_path):
            print(f"Filtered file already exists: {out_path}, skipping.")
            continue

        st = obspy.read(file_path)
        magnitude = row["magnitude"]
        filtered_st = bandpass(st, magnitude, filter_frequencies)
        filtered_st.write(out_path, format="MSEED")

if __name__ == "__main__":
    main()

