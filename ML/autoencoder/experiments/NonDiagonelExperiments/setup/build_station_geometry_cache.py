import argparse
import json
from pathlib import Path
from typing import Dict, List

from obspy.clients.fdsn import Client


def fetch_station(station: str, clients: List[Client]) -> Dict:
    for client in clients:
        try:
            inv = client.get_stations(network="KO", station=station, level="station")
            net = inv[0]
            st = net[0]
            return {
                "latitude": float(st.latitude),
                "longitude": float(st.longitude),
                "elevation_m": float(st.elevation),
                "network": "KO",
                "source_client": client.base_url,
            }
        except Exception:
            continue
    return None


def main():
    parser = argparse.ArgumentParser(description="Build station coordinate cache for geometry-conditioned models.")
    parser.add_argument(
        "--station_list",
        default="data/station_list_external_full.json",
        help="JSON file with station code list.",
    )
    parser.add_argument(
        "--output",
        default="ML/autoencoder/experiments/NonDiagonel/results/station_coords_external.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()

    with open(args.station_list, "r") as f:
        stations = sorted(json.load(f))

    clients = []
    for provider in ["KOERI", "IRIS"]:
        try:
            clients.append(Client(provider, timeout=60))
            print(f"[INFO] FDSN client ready: {provider}")
        except Exception as e:
            print(f"[WARN] FDSN client init failed ({provider}): {e}")

    if not clients:
        raise RuntimeError("No FDSN client available.")

    payload = {}
    missing = []
    for sta in stations:
        row = fetch_station(sta, clients)
        if row is None:
            missing.append(sta)
            payload[sta] = None
            print(f"[WARN] Station not resolved: {sta}")
        else:
            payload[sta] = row

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"[INFO] Saved station coordinate cache: {out}")
    print(f"[INFO] Resolved: {len(stations) - len(missing)} / {len(stations)}")
    if missing:
        print(f"[WARN] Missing stations: {missing}")


if __name__ == "__main__":
    main()

