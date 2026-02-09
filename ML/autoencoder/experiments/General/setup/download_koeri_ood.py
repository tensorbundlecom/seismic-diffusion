import os
import json
import pandas as pd
from obspy.clients.fdsn import Client
from obspy import UTCDateTime, Stream
from tqdm import tqdm
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

def download_single_station(client, network, sta, starttime, endtime, event_id, output_dir):
    try:
        # Request all possible earthquake channels
        st = client.get_waveforms(
            network=network,
            station=sta,
            location="*",
            channel="HH?,EH?,BH?,HN?",
            starttime=starttime,
            endtime=endtime
        )
        
        if len(st) < 3: return False
        st.merge(fill_value=0)
        
        # Pick a channel type prefix (HH, BH, etc)
        channel_type = st[0].stats.channel[:2]
        save_dir = os.path.join(output_dir, "raw")
        os.makedirs(save_dir, exist_ok=True)
        
        filename = f"{event_id}_{sta}_{channel_type}.mseed"
        st.write(os.path.join(save_dir, filename), format="MSEED")
        return True
    except Exception:
        return False

def download_koeri_ood(catalog_path, output_dir, network="KO", duration_sec=60, max_workers=15):
    client = Client("KOERI", timeout=60)
    
    # Read our custom catalog (tab separated)
    df = pd.read_csv(catalog_path, sep='\t')
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Downloading waveforms for {len(df)} events from KOERI...")

    tasks = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for _, row in df.iterrows():
            event_id = str(row['Deprem Kodu'])
            origin_time = UTCDateTime(f"{row['Olus tarihi'].replace('.','-')}T{row['Olus zamani']}")
            starttime = origin_time - 15
            endtime = origin_time + duration_sec
            
            # Find ALL stations available at this time in the Marmara region
            try:
                inv = client.get_stations(starttime=starttime, endtime=endtime,
                                         minlatitude=40.0, maxlatitude=41.5,
                                         minlongitude=26.0, maxlongitude=29.5,
                                         level="station")
                target_stations = [sta.code for net in inv for sta in net]
                print(f"  Event {event_id}: Found {len(target_stations)} possible stations.")
                
                for sta in target_stations:
                    tasks.append(executor.submit(download_single_station, client, network, sta, starttime, endtime, event_id, output_dir))
            except Exception as e:
                print(f"  Failed to get stations for {event_id}: {e}")
    
    print(f"Submitted {len(tasks)} download tasks.")
    
    results = []
    for future in tqdm(as_completed(tasks), total=len(tasks), desc="Downloading"):
        results.append(future.result())
    
    print(f"\nDownload finished. Success: {sum(results)} / {len(tasks)}")

if __name__ == "__main__":
    catalog = "data/events/ood_catalog_koeri.txt"
    out_dir = "data/ood_waveforms/koeri"
    
    download_koeri_ood(catalog, out_dir)
