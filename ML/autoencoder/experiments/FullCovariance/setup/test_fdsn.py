from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import pandas as pd
import os

def test_fdsn_services():
    clients = ["KOERI", "IRIS", "GFZ", "ETH"]
    for c_name in clients:
        try:
            print(f"Testing {c_name}...")
            client = Client(c_name)
            # Try a very small query
            cat = client.get_events(starttime=UTCDateTime("2024-01-01"), endtime=UTCDateTime("2024-01-02"), minmagnitude=4.0)
            print(f"  {c_name} success: {len(cat)} events")
            return c_name
        except Exception as e:
            print(f"  {c_name} failed: {e}")
    return None

if __name__ == "__main__":
    best_client = test_fdsn_services()
    print(f"Best client found: {best_client}")
