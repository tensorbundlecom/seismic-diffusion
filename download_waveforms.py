import os
import pickle
import json
import pandas as pd
from obspy.clients.fdsn import Client
from obspy import Stream
from tqdm import tqdm
import signal
import sys

def save_checkpoint(download_report, output_dir):
    """
    Save the download report as a checkpoint.
    
    Parameters:
    -----------
    download_report : dict
        Dictionary containing download status for each pick
    output_dir : str or Path
        Directory where checkpoint will be saved
    """
    checkpoint_path = os.path.join(output_dir, "download_checkpoint.json")
    try:
        with open(checkpoint_path, "w") as f:
            json.dump(download_report, f, indent=2)
        # print(f"\nCheckpoint saved to: {checkpoint_path}")
    except Exception as e:
        print(f"\nWarning: Failed to save checkpoint: {e}")

def load_checkpoint(output_dir):
    """
    Load existing checkpoint if available.
    
    Parameters:
    -----------
    output_dir : str or Path
        Directory where checkpoint is saved
    
    Returns:
    --------
    dict : Existing download report or empty dict if no checkpoint exists
    """
    checkpoint_path = os.path.join(output_dir, "download_checkpoint.json")
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r") as f:
                report = json.load(f)
            print(f"Loaded checkpoint with {len(report)} existing entries")
            return report
        except Exception as e:
            print(f"Warning: Failed to load checkpoint: {e}")
            return {}
    return {}

def download_waveforms(catalog_dir, pick_dir, output_dir):
    """
    Download waveforms based on phase picks in the catalog files.
    
    Parameters:
    -----------
    catalog_dir : str or Path
        Directory containing year-based combined catalog files
    pick_dir : str or Path
        Directory containing phase pick files
    output_dir : str or Path
        Directory where downloaded waveforms will be saved
    """
    client = Client("KOERI")
    
    # Load existing checkpoint if available
    download_report = load_checkpoint(output_dir)
    
    # Setup signal handler for graceful interruption
    def signal_handler(sig, frame):
        # print("\n\nInterrupt received! Saving checkpoint...")
        save_checkpoint(download_report, output_dir)
        # print("Checkpoint saved. Exiting...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Counter for periodic checkpointing
    operations_since_checkpoint = 0
    checkpoint_interval = 100  # Save every 100 operations

    # Counter for periodic checkpointing
    operations_since_checkpoint = 0
    checkpoint_interval = 100  # Save every 100 operations

    # for each year file in catalog_dir
    year_files = os.listdir(pick_dir)
    for year_file in year_files:
        year_path = os.path.join(pick_dir, year_file)
        with open(year_path, "rb") as f:
            print(f"Processing pick file: {year_path}")
            catalog = pickle.load(f)
        
        catalog_dict = {}
        for event in catalog:
            event_id = str(event.resource_id).split("/")[-1]
            catalog_dict[event_id] = event
        

        all_events = pd.read_csv(catalog_dir, delimiter="\t", encoding="latin-1")
        for index, row in tqdm(all_events.iterrows(), total=all_events.shape[0]):
            event_id = str(row["Deprem Kodu"])
            # print(f"Checking Event ID: {event_id}")
            catalog_event = catalog_dict.get(event_id)
            if catalog_event:
                # print(f"Event ID: {event_id}, Catalog Event: {catalog_event}")
                for pick in catalog_event.picks:
                    if pick.phase_hint == "Pg":
                        pick_key = f"{event_id}_{pick.waveform_id.station_code}_{pick.waveform_id.channel_code}"
                        
                        # Skip if already processed (from checkpoint)
                        if pick_key in download_report:
                            continue
                        
                        try:
                            # download the waveforms for this pick
                            waveforms = client.get_waveforms(
                                network="KO",
                                station=pick.waveform_id.station_code,
                                location=pick.waveform_id.location_code,
                                channel=pick.waveform_id.channel_code,
                                starttime=pick.time - 10,
                                endtime=pick.time + 60,
                                
                            )
                            waveforms.merge(fill_value=0)
                            
                            # Separate channels by their first two letters (HH, HN, etc.)
                            streams_by_type = {}
                            for trace in waveforms:
                                channel_type = trace.stats.channel[:2]  # Get first 2 letters (HH, HN, etc.)
                                if channel_type not in streams_by_type:
                                    streams_by_type[channel_type] = []
                                streams_by_type[channel_type].append(trace)
                            
                            # Plot each stream type separately with adaptive ylims
                            for channel_type, traces in streams_by_type.items():
                                stream = Stream(traces)
                                output_path = os.path.join(
                                    output_dir,
                                    channel_type,
                                    f"{event_id}_{pick.waveform_id.station_code}_{channel_type}.mseed"
                                )
                                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                                stream.write(output_path, format="MSEED")
                                # print(f"Saved waveform: {output_path}")
                            
                            download_report[pick_key] = "success"
                            operations_since_checkpoint += 1
                        
                        except Exception as e:
                            error_msg = f"{type(e).__name__}: {str(e)}"
                            download_report[pick_key] = error_msg
                            operations_since_checkpoint += 1
                            # print(f"Failed to download {pick_key}: {error_msg}")
                        
                        # Periodic checkpoint
                        if operations_since_checkpoint >= checkpoint_interval:
                            save_checkpoint(download_report, output_dir)
                            operations_since_checkpoint = 0
            else:
                # Event not found in catalog
                pass
    
    # Save the final download report
    save_checkpoint(download_report, output_dir)
    report_path = os.path.join(output_dir, "download_report.json")
    with open(report_path, "w") as f:
        json.dump(download_report, f, indent=2)
    print(f"\nFinal download report saved to: {report_path}")
    
    return download_report

if __name__ == "__main__":
    catalog_directory = "data/events/20140101_20251101_0.0_9.0_9_339.txt"
    pick_directory = "data/phase_picks/years"
    output_directory = "data/waveforms"
    
    download_waveforms(catalog_directory, pick_directory, output_directory)