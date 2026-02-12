import json
import os
from pathlib import Path

import pandas as pd
from obspy.clients.fdsn import Client
from obspy.geodetics import gps2dist_azimuth


def load_events(catalog_path):
    df = pd.read_csv(catalog_path, sep='\t')
    df.columns = df.columns.str.strip()
    df['event_id'] = df['Deprem Kodu'].astype(str).str.strip()
    df['latitude'] = pd.to_numeric(df['Enlem'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['Boylam'], errors='coerce')
    df['depth_km'] = pd.to_numeric(df['Der(km)'], errors='coerce')
    df['xM'] = pd.to_numeric(df['xM'], errors='coerce')
    return df


def fetch_station_coords(stations):
    clients = []
    try:
        clients.append(Client('KOERI', timeout=60))
    except Exception as e:
        print(f'[WARN] KOERI client init failed: {e}')
    try:
        clients.append(Client('IRIS', timeout=60))
    except Exception as e:
        print(f'[WARN] IRIS client init failed: {e}')
    if not clients:
        raise RuntimeError('No FDSN clients could be initialized.')

    coords = {}
    for sta in stations:
        for client in clients:
            try:
                inv = client.get_stations(network='KO', station=sta, level='station')
                net = inv[0]
                st = net[0]
                coords[sta] = {
                    'latitude': st.latitude,
                    'longitude': st.longitude,
                    'elevation_m': st.elevation,
                }
                break
            except Exception:
                continue
        if sta not in coords:
            coords[sta] = None
            print(f'[WARN] Station coordinates not found for {sta}')
    return coords


def build_docs(catalog_path, station_list_file, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    events = load_events(catalog_path)
    with open(station_list_file, 'r') as f:
        stations = json.load(f)

    coords = fetch_station_coords(stations)
    coords_path = output_dir / 'post_training_custom_ood_station_coords.json'
    with open(coords_path, 'w') as f:
        json.dump(coords, f, indent=2)

    rows = []
    for _, row in events.iterrows():
        for sta in stations:
            c = coords.get(sta)
            if not c:
                continue
            dist_m, az, _ = gps2dist_azimuth(
                row['latitude'], row['longitude'], c['latitude'], c['longitude']
            )
            rows.append({
                'event_id': row['event_id'],
                'date': row['Olus tarihi'],
                'time': row['Olus zamani'],
                'lat': row['latitude'],
                'lon': row['longitude'],
                'depth_km': row['depth_km'],
                'xM': row['xM'],
                'station': sta,
                'station_lat': c['latitude'],
                'station_lon': c['longitude'],
                'distance_km': dist_m / 1000.0,
                'azimuth_deg': az,
            })

    dist_df = pd.DataFrame(rows)
    dist_csv = output_dir / 'post_training_custom_ood_station_distances.csv'
    dist_df.to_csv(dist_csv, index=False)

    md_path = output_dir / 'post_training_custom_ood.md'
    with open(md_path, 'w') as f:
        f.write('# Post-Training Custom OOD (HH-only)\\n\\n')
        f.write('Stations: ADVT, ARMT, KCTX, YLV, GEML, GELI\\n\\n')
        f.write('## Events\\n\\n')
        f.write('| Event | Date | Time | Lat | Lon | Depth(km) | xM |\\n')
        f.write('| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\\n')
        for _, row in events.iterrows():
            f.write(
                f"| {row['event_id']} | {row['Olus tarihi']} | {row['Olus zamani']} | "
                f"{row['latitude']:.4f} | {row['longitude']:.4f} | {row['depth_km']:.2f} | {row['xM']:.1f} |\\n"
            )

        f.write('\\n## Station Coordinates\\n\\n')
        f.write('| Station | Lat | Lon | Elev (m) |\\n')
        f.write('| :--- | :---: | :---: | :---: |\\n')
        for sta in stations:
            c = coords.get(sta)
            if c:
                f.write(
                    f"| {sta} | {c['latitude']:.4f} | {c['longitude']:.4f} | {c['elevation_m']:.1f} |\\n"
                )
            else:
                f.write(f"| {sta} | N/A | N/A | N/A |\\n")

        f.write('\\n## Event–Station Distances\\n\\n')
        f.write('Saved to: `post_training_custom_ood_station_distances.csv`\\n')

    print(f'[INFO] Docs written: {md_path}')
    print(f'[INFO] Distances CSV: {dist_csv}')
    print(f'[INFO] Station coords JSON: {coords_path}')


if __name__ == '__main__':
    build_docs(
        catalog_path='data/events/ood_catalog_post_training.txt',
        station_list_file='data/station_list_post_custom.json',
        output_dir='ML/autoencoder/experiments/General/setup/docs',
    )
