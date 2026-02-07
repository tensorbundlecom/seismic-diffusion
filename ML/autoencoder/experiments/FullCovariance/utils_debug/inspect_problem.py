import obspy
import sys

def inspect_mseed(file_path):
    try:
        st = obspy.read(file_path)
        print(f"File: {file_path}")
        print(f"Number of traces: {len(st)}")
        for i, tr in enumerate(st):
            print(f"Trace {i}: {tr.stats.network}.{tr.stats.station}.{tr.stats.location}.{tr.stats.channel} | Start: {tr.stats.starttime} | End: {tr.stats.endtime} | Samples: {tr.stats.npts} | SR: {tr.stats.sampling_rate}")
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    inspect_mseed("data/filtered_waveforms/HH/query?eventid=11431638_EDC_HH.mseed")
