from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import pandas as pd
import os

def fetch_koeri_catalog(starttime, endtime, minlat, maxlat, minlon, maxlon, minmag, output_file):
    """
    Fetches earthquake catalog from KOERI FDSN service and saves it in a format
    compatible with the SeismicSTFTDatasetWithMetadata.
    """
    print(f"Connecting to IRIS FDSN...")
    client = Client("IRIS")
    
    print(f"Fetching events between {starttime} and {endtime}...")
    try:
        catalog = client.get_events(
            starttime=UTCDateTime(starttime),
            endtime=UTCDateTime(endtime),
            minlatitude=minlat,
            maxlatitude=maxlat,
            minlongitude=minlon,
            maxlongitude=maxlon,
            minmagnitude=minmag
        )
        print(f"Found {len(catalog)} events.")
        
        events_data = []
        for event in catalog:
            origin = event.origins[0]
            mag = event.magnitudes[0] if event.magnitudes else None
            
            # Map to expected AFAD-like columns
            event_dict = {
                'No': '', # Not strictly needed
                'Deprem Kodu': str(event.resource_id).split('/')[-1],
                'Olus tarihi': origin.time.strftime('%Y.%m.%d'),
                'Olus zamani': origin.time.strftime('%H:%M:%S'),
                'Enlem': origin.latitude,
                'Boylam': origin.longitude,
                'Der(km)': origin.depth / 1000.0 if origin.depth else 0.0,
                'xM': mag.mag if mag else 0.0,
                'MD': 0.0,
                'ML': mag.mag if mag and mag.magnitude_type == 'ML' else 0.0,
                'Mw': mag.mag if mag and mag.magnitude_type == 'Mw' else 0.0,
                'Ms': 0.0,
                'Mb': 0.0,
                'Tip': 'RE', # Real
                'Yer': str(event.event_descriptions[0].text) if event.event_descriptions else 'UNKNOWN'
            }
            events_data.append(event_dict)
            
        df = pd.DataFrame(events_data)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save as tab-separated to match original expectations
        df.to_csv(output_file, sep='\t', index=False, encoding='utf-8')
        print(f"Catalog saved to {output_file}")
        return output_file

    except Exception as e:
        print(f"Error fetching catalog: {e}")
        return None

if __name__ == "__main__":
    # Example for Marmara region
    fetch_koeri_catalog(
        starttime="2014-01-01",
        endtime="2024-12-31", 
        minlat=40.0,
        maxlat=41.5,
        minlon=26.0,
        maxlon=30.0,
        minmag=3.0,
        output_file="data/events/koeri_catalog.txt"
    )
