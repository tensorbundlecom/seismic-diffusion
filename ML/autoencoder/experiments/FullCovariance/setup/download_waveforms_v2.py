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
        
        channel_type = st[0].stats.channel[:2]
        save_dir = os.path.join(output_dir, channel_type)
        os.makedirs(save_dir, exist_ok=True)
        
        filename = f"{event_id}_{sta}_{channel_type}.mseed"
        st.write(os.path.join(save_dir, filename), format="MSEED")
        return True
    except:
        return False

def download_waveforms_from_catalog(catalog_path, output_dir, network="KO", duration_sec=60, max_workers=8):
    # Using a fresh client per thread or a shared one with timeout
    client = Client("KOERI", timeout=60)
    df = pd.read_csv(catalog_path, sep='\t')
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Fetching station list from KOERI...")
    inventory = client.get_stations(network=network, level="station")
    stations = [sta.code for net in inventory for sta in net]
    
    preferred_stations = ["EDC", "SIL", "KCT", "MRM", "YLV", "ADVT", "GEMT"]
    target_stations = [s for s in stations if s in preferred_stations]
    if not target_stations: target_stations = stations[:10]
    
    print(f"Targeting {len(target_stations)} stations for download using {max_workers} workers.")

    tasks = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for _, row in df.iterrows():
            event_id = str(row['Deprem Kodu'])
            origin_time = UTCDateTime(f"{row['Olus tarihi'].replace('.','-')}T{row['Olus zamani']}")
            starttime = origin_time - 10
            endtime = origin_time + duration_sec
            
            for sta in target_stations:
                # We submit each station download as a separate task
                tasks.append(executor.submit(download_single_station, client, network, sta, starttime, endtime, event_id, output_dir))
        
        print(f"Submitted {len(tasks)} download tasks.")
        
        # Track progress with tqdm
        results = []
        for future in tqdm(as_completed(tasks), total=len(tasks), desc="Downloading Waveforms"):
            results.append(future.result())
    
    success_count = sum(results)
    print(f"\nDownload complete. Successfully downloaded {success_count} waveform files.")

if __name__ == "__main__":
    catalog_file = "data/events/koeri_catalog.txt"
    output_directory = "data/waveforms"
    
    if os.path.exists(catalog_file):
        download_waveforms_from_catalog(catalog_file, output_directory)
    else:
        print(f"Catalog file {catalog_file} not found.")
