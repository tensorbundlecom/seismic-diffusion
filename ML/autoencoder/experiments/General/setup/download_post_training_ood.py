import os
import json
import pandas as pd
from obspy.clients.fdsn import Client
from obspy import UTCDateTime, Stream
from tqdm import tqdm
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

def download_single_station(client, network, sta, starttime, endtime, event_id, output_dir, channel_filter="HH?"):
    """Download HH channels specifically for post-training OOD events."""
    try:
        st = client.get_waveforms(
            network=network,
            station=sta,
            location="*",
            channel=channel_filter,  # HH channels only
            starttime=starttime,
            endtime=endtime
        )
        
        if len(st) < 3: 
            return False, "Insufficient channels"
        
        st.merge(fill_value=0)
        
        # Verify it's HH channel
        channel_type = st[0].stats.channel[:2]
        if channel_type != "HH":
            return False, f"Not HH channel: {channel_type}"
        
        save_dir = os.path.join(output_dir, "raw", "HH")
        os.makedirs(save_dir, exist_ok=True)
        
        filename = f"{event_id}_{sta}_HH.mseed"
        st.write(os.path.join(save_dir, filename), format="MSEED")
        return True, f"Success: {sta}"
    except Exception as e:
        return False, str(e)

def download_post_training_ood(catalog_path, output_dir, network="KO", duration_sec=60, max_workers=15):
    """Download HH channel waveforms for post-training OOD events (2022-2024)."""
    
    # Try KOERI first, fallback to IRIS
    clients = [
        ("KOERI", Client("KOERI", timeout=60)),
        ("IRIS", Client("IRIS", timeout=60))
    ]
    
    df = pd.read_csv(catalog_path, sep='\t')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Downloading HH channel waveforms for {len(df)} post-training OOD events...")
    print(f"Date range: 2022-2024 (after training cutoff: 2022-04-02)")
    
    # Target stations from training set
    target_stations = ["ADVT", "CTKS", "EDC", "GEMT", "YLV", "BALB", "GAZ", "ELL", "SIL", "KCT", "MRM"]
    
    total_success = 0
    
    for _, row in df.iterrows():
        event_id = str(row['Deprem Kodu'])
        origin_time = UTCDateTime(f"{row['Olus tarihi'].replace('.','-')}T{row['Olus zamani']}")
        starttime = origin_time - 15
        endtime = origin_time + duration_sec
        
        print(f"\n  Event {event_id} (M{row['xM']}) at {origin_time}:")
        event_success = 0
        
        for client_name, client in clients:
            if event_success >= 3:  # At least 3 stations per event
                break
                
            print(f"    Trying {client_name}...")
            
            for sta in target_stations:
                try:
                    success, msg = download_single_station(
                        client, network, sta, starttime, endtime, 
                        event_id, output_dir, channel_filter="HH?"
                    )
                    if success:
                        print(f"      ✓ {msg}")
                        event_success += 1
                        total_success += 1
                except:
                    continue
        
        print(f"    Event total: {event_success} HH waveforms")
    
    print(f"\n{'='*80}")
    print(f"Download finished. Total HH waveforms: {total_success}")
    print(f"{'='*80}")

if __name__ == "__main__":
    catalog = "data/events/ood_catalog_post_training.txt"
    out_dir = "data/ood_waveforms/post_training"
    
    download_post_training_ood(catalog, out_dir)
