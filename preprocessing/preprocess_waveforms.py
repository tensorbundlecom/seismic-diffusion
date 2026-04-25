import argparse
import obspy
import pandas as pd
import json
import os
from tqdm import tqdm

def bandpass(stream, low_freq, high_freq):
    stream.detrend("demean")
    stream.detrend("linear")
    stream.taper(max_percentage=0.05, type="cosine")
    stream.filter("bandpass", freqmin=low_freq, freqmax=high_freq)
    return stream

def bandpass_from_json(stream, magnitude, filter_frequencies):
    channel_type = stream[0].stats.channel[:2]
    mag_key = "4+" if magnitude >= 4 else f"{int(magnitude)}-{int(magnitude)+1}"
    if channel_type not in ["HN", "BH", "EH", "HH"]:
        raise ValueError(f"Unknown channel type: {channel_type}")
    low_freq, high_freq = filter_frequencies[channel_type][mag_key]
    return bandpass(stream, low_freq, high_freq)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snr", type=float, default=None,
                        help="SNR threshold; reads from waveform_summary.csv if set")
    parser.add_argument("--freqmin", type=float, default=None)
    parser.add_argument("--freqmax", type=float, default=None)
    parser.add_argument("--output-dir", default="data/filtered_waveforms")
    args = parser.parse_args()

    fixed_freq = args.freqmin is not None and args.freqmax is not None

    if args.snr is not None:
        df = pd.read_csv("data/waveform_summary.csv")
        df["snr_max"] = df[["snr1", "snr2", "snr3"]].max(axis=1)
        df["gap_max"] = df[["gap1", "gap2", "gap3"]].max(axis=1)
        waveforms_df = df[(df["snr_max"] > args.snr) & (df["gap_max"] == 0)]
        print(f"Selected {len(waveforms_df):,} waveforms (SNR > {args.snr}, gap = 0)")
    else:
        waveforms_df = pd.read_csv("data/high_snr_low_gap_waveforms.csv")

    if not fixed_freq:
        with open("data/filter_frequencies.json", "r") as f:
            filter_frequencies = json.load(f)

    for _, row in tqdm(waveforms_df.iterrows(), total=waveforms_df.shape[0]):
        file_name = "/".join(row["waveform_file"].split("/")[-2:])
        file_path = os.path.join("data/waveforms", file_name)
        out_path = os.path.join(args.output_dir, file_name)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        if os.path.exists(out_path):
            continue

        st = obspy.read(file_path)
        if fixed_freq:
            filtered_st = bandpass(st, args.freqmin, args.freqmax)
        else:
            filtered_st = bandpass_from_json(st, row["magnitude"], filter_frequencies)
        filtered_st.write(out_path, format="MSEED")

if __name__ == "__main__":
    main()

