import os
import obspy
from glob import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.fft import rfft, rfftfreq
from obspy.signal.konnoohmachismoothing import konno_ohmachi_smoothing

def get_all_files(input_dir, extension="mseed"):
    """
    Get all files with the specified extension from the input directory.
    
    Parameters:
    -----------
    input_dir : str or Path
        Directory containing files
    extension : str
        File extension to look for (default: "mseed")
    
    Returns:
    --------
    list : List of file paths
    """
    file_pattern = os.path.join(input_dir, f"*/*.{extension}")
    print(file_pattern)
    files = glob(file_pattern)
    
    if not files:
        print(f"No .{extension} files found in {input_dir}")
    
    return files


def get_gap_ratio(stream):
    """
    Calculate the gap ratio for a given stream.
    
    Parameters:
    -----------
    stream : obspy.Stream
        Stream object containing traces
    
    Returns:
    --------
    float : Gap ratio (total gap duration / total duration)
    """
    gaps = []
    for tr in stream:
        data = tr.data
        # print(data.shape, 70*tr.stats.sampling_rate, np.sum(data == 0))
        gap_size = np.sum(data == 0) + (70*tr.stats.sampling_rate - data.shape[0] + 1)
        # total_size = data.shape[0]
        total_size = 70*tr.stats.sampling_rate  # 70 seconds of data
        gaps.append(gap_size / total_size)

    return gaps


def konno_ohmachi_weight_matrix(freqs, bandwidth=20.0):
    """
    Precompute Konno-Ohmachi smoothing weight matrix.
    freqs: 1D array of frequencies (same as FFT frequency vector)
    returns: W such that smoothed_fft = W @ fft_amplitudes
    """
    f = freqs
    N = len(f)

    # Avoid divide-by-zero at f = 0 by replacing 0 with small value
    f_safe = f.copy()
    f_safe[f_safe == 0] = 1e-20

    ratio = f_safe[:, None] / f_safe[None, :]
    x = (np.log10(ratio)) * bandwidth

    # Konno–Ohmachi kernel
    W = (np.sin(x) / x) ** 4
    W[np.isnan(W)] = 1.0  # diagonal where x = 0

    # Normalize rows
    W /= W.sum(axis=1, keepdims=True)

    return W

def compute_snr_fast(stream, fmin=2.0, fmax=15.0, bexp=20):
    snr_results = []

    tr0 = stream[0]
    sr = tr0.stats.sampling_rate
    nwin = int(sr)  # 1 second windows

    # Frequencies for 1-second FFT
    freqs = np.fft.rfftfreq(nwin, d=1/sr)

    # Precompute Konno-Ohmachi weights (HUGE speed boost)
    W = konno_ohmachi_weight_matrix(freqs, bandwidth=bexp)

    mask = (freqs >= fmin) & (freqs <= fmax)

    for tr in stream:
        d = tr.data

        noise = d[int(9*sr):int(10*sr)]
        signal = d[int(10*sr):int(11*sr)]

        if len(noise) < nwin or len(signal) < nwin:
            snr_results.append((False, 0))
            continue

        fft_noise = np.abs(np.fft.rfft(noise))
        fft_signal = np.abs(np.fft.rfft(signal))

        smooth_noise  = W @ fft_noise
        smooth_signal = W @ fft_signal

        snr = smooth_signal / (smooth_noise + 1e-20)

        valid = np.all(snr[mask] > 10)
        snr_results.append((valid, float(np.mean(snr[mask]))))

    return snr_results



def event_magnitude(events, event_id):
    return events[events["Deprem Kodu"] == event_id]["xM"].values[0]


def load_events(event_file):
    """
    Load events from a given event file.
    
    Parameters:
    -----------
    event_file : str or Path
        Path to the event file
    
    Returns:
    --------
    obspy.Catalog : Catalog object containing events
    """
        # read the events into a DataFrame
    events = pd.read_csv(event_file, delimiter="	", encoding="latin-1")
    # deprem kodu is a string
    events["Deprem Kodu"] = events["Deprem Kodu"].astype(str)
    # events.head()
    return events

def create_summary(events, waveform_files, output_file="data/waveform_summary.csv", checkpoint_interval=10_000):
    """
    Create a summary DataFrame for the given events and waveform files.
    Writes to output file every checkpoint_interval iterations.
    
    Parameters:
    -----------
    events : obspy.Catalog
        Catalog object containing events
    waveform_files : list
        List of waveform file paths
    output_file : str
        Path to the output CSV file (default: "data/waveform_summary.csv")
    checkpoint_interval : int
        Number of iterations between checkpoints (default: 100)
    
    Returns:
    --------
    pd.DataFrame : Summary DataFrame with snr and magnitude information
    """
    summary_data = []
    processed_files = set()
    
    # Load existing data if the output file exists
    if os.path.exists(output_file):
        try:
            existing_df = pd.read_csv(output_file)
            processed_files = set(existing_df['waveform_file'].unique())
            print(f"Resuming from checkpoint: {len(processed_files)} files already processed")
        except Exception as e:
            print(f"Could not load existing checkpoint: {e}")
            existing_df = None
    else:
        existing_df = None
    
    # Filter out already processed files
    remaining_files = [f for f in waveform_files if f not in processed_files]
    print(f"Processing {len(remaining_files)} remaining files out of {len(waveform_files)} total")

    for idx, wf_file in enumerate(tqdm(remaining_files), start=1):
        try:
            st = obspy.read(wf_file)
            snr_values = compute_snr_fast(st)
            
            # Extract event ID from filename
            event_id = os.path.basename(wf_file).split("_")[0]
            magnitude = event_magnitude(events, event_id)
            gaps = get_gap_ratio(st)
            
            # for snr in snr_values:
            summary_data.append({
                "event_id": event_id,
                "magnitude": magnitude,
                "snr1": snr_values[0][1] if len(snr_values) > 0 else np.nan,
                "snr2": snr_values[1][1] if len(snr_values) > 1 else np.nan,
                "snr3": snr_values[2][1] if len(snr_values) > 2 else np.nan,
                "waveform_file": wf_file,
                "gap1": gaps[0] if len(gaps) > 0 else np.nan,
                "gap2": gaps[1] if len(gaps) > 1 else np.nan,
                "gap3": gaps[2] if len(gaps) > 2 else np.nan
            })
        except Exception as e:
            print(f"Error processing {wf_file}: {e}")
        
        # Checkpoint: write to file every checkpoint_interval iterations
        if idx % checkpoint_interval == 0 and summary_data:
            checkpoint_df = pd.DataFrame(summary_data)
            
            # Append to existing file or create new one
            if existing_df is not None or os.path.exists(output_file):
                checkpoint_df.to_csv(output_file, mode='a', header=False, index=False)
            else:
                checkpoint_df.to_csv(output_file, mode='w', header=True, index=False)
                existing_df = checkpoint_df  # Mark that file now exists with header
            
            print(f"\nCheckpoint: Saved {len(summary_data)} records to {output_file}")
            summary_data = []  # Clear the buffer
    
    # Write any remaining data
    if summary_data:
        final_df = pd.DataFrame(summary_data)
        
        if existing_df is not None or os.path.exists(output_file):
            final_df.to_csv(output_file, mode='a', header=False, index=False)
        else:
            final_df.to_csv(output_file, mode='w', header=True, index=False)
        
        print(f"\nFinal write: Saved {len(summary_data)} records to {output_file}")
    
    # Return the complete DataFrame
    return pd.read_csv(output_file) if os.path.exists(output_file) else pd.DataFrame()

if __name__ == "__main__":
    event_file = "data/events/20140101_20251101_0.0_9.0_9_339.txt"
    events = load_events(event_file)
    waveform_files = get_all_files("./data/waveforms", extension="mseed")
    
    # Create summary with checkpointing every 100 files
    summary_df = create_summary(
        events, 
        waveform_files, 
        output_file="data/waveform_summary.csv",
        checkpoint_interval=10_000,
        )
    
    print(f"\nSummary complete! Total records: {len(summary_df)}")
    print(f"\nFirst few rows:")
    print(summary_df.head())
    print(f"\nBasic statistics:")
    print(summary_df.describe())
