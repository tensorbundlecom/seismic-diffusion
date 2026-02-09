from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import pandas as pd
import os
from tqdm import tqdm

def download_missing_ko_from_iris():
    client = Client("IRIS")
    catalog = "data/events/ood_catalog_koeri.txt"
    df = pd.read_csv(catalog, sep='\t')
    
    output_dir = "data/ood_waveforms/koeri/raw"
    os.makedirs(output_dir, exist_ok=True)
    
    missing_ids = ["OOD_K_08", "OOD_K_09", "OOD_K_10"]
    stations = ["EDC", "SIL", "KCT", "MRM", "YLV", "ADVT", "GEMT", "CTKS", "ALTN", "MFT", "BALB", "GAZ", "ELL", "DKL"]
    
    print(f"Downloading missing {len(missing_ids)} events from IRIS (KO network)...")
    
    success_count = 0
    for _, row in df.iterrows():
        if row['Deprem Kodu'] in missing_ids:
            event_id = row['Deprem Kodu']
            origin_time = UTCDateTime(f"{row['Olus tarihi'].replace('.','-')}T{row['Olus zamani']}")
            starttime = origin_time - 15
            endtime = origin_time + 60
            
            print(f"  Event {event_id} ({origin_time})...")
            
            for sta in stations:
                try:
                    st = client.get_waveforms(network="KO", station=sta, location="*", 
                                             channel="HH?,EH?,BH?,HN?", 
                                             starttime=starttime, endtime=endtime)
                    if len(st) >= 3:
                        st.merge(fill_value=0)
                        channel_type = st[0].stats.channel[:2]
                        filename = f"{event_id}_{sta}_{channel_type}.mseed"
                        st.write(os.path.join(output_dir, filename), format="MSEED")
                        print(f"    - Success: {sta}")
                        success_count += 1
                except:
                    continue
    
    print(f"\nDownload finished. Successfully downloaded {success_count} files for missing events.")

if __name__ == "__main__":
    download_missing_ko_from_iris()
