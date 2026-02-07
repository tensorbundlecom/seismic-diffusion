from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import sys
import os

# Add project root for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../')))

client = Client("IRIS")

# Marmara region bounds from previous analysis
min_lat, max_lat = 40.0, 41.5
min_lon, max_lon = 26.0, 29.5

# Search for older events (2010 - 2015) with Mw > 4.2
starttime = UTCDateTime("2010-01-01")
endtime = UTCDateTime("2015-12-31")

print(f"Searching for events from {starttime} to {endtime}...")
catalog = client.get_events(starttime=starttime, endtime=endtime,
                           minlatitude=min_lat, maxlatitude=max_lat,
                           minlongitude=min_lon, maxlongitude=max_lon,
                           minmagnitude=4.2)

print(f"Found {len(catalog)} events.")
for event in catalog:
    origin = event.origins[0]
    mag = event.magnitudes[0].mag
    print(f"Time: {origin.time}, Lat: {origin.latitude:.4f}, Lon: {origin.longitude:.4f}, Depth: {origin.depth/1000:.1f}km, Mag: {mag}")
