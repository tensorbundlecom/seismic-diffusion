#!/usr/bin/env python
"""
Fetch station coordinates with ObsPy/FDSN directly from metadata station codes.

Output schema:
{
  "EDC": {"latitude": 40.123, "longitude": 27.456, "elevation_m": 123.0, "network": "KO"},
  ...
}
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

from obspy import UTCDateTime
from obspy.clients.fdsn import Client, RoutingClient


def parse_args():
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Fetch station locations via ObsPy/FDSN")
    parser.add_argument(
        "--metadata",
        type=str,
        default=str(base_dir / "embeddings" / "metadata.json"),
        help="Path to embeddings metadata.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(base_dir / "embeddings" / "station_locations.json"),
        help="Output JSON path for station locations",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any station cannot be resolved",
    )
    parser.add_argument(
        "--search_range_deg",
        type=float,
        default=5.0,
        help="Geographic margin (degrees) added around event bounds for station search",
    )
    parser.add_argument(
        "--time_range_days",
        type=float,
        default=1.0,
        help="Half-window (days) around event time for time-constrained station queries",
    )
    return parser.parse_args()


def _parse_event_time(event_id: str) -> Optional[UTCDateTime]:
    if not event_id:
        return None
    try:
        # event_id format in this project: YYYYMMDDhhmmss
        return UTCDateTime.strptime(event_id, "%Y%m%d%H%M%S")
    except Exception:
        return None


def _extract_station_queries(rows, search_range_deg: float):
    station_info: Dict[str, Dict] = {}
    lats = []
    lons = []

    for row in rows:
        station = row.get("station_name")
        if not station:
            continue

        info = station_info.setdefault(station, {"channels": set(), "times": []})

        ch = row.get("channel_type")
        if ch:
            info["channels"].add(ch)

        t = _parse_event_time(str(row.get("event_id", "")))
        if t is not None and not info["times"]:
            # One representative time is enough.
            info["times"].append(t)

        lat = row.get("latitude")
        lon = row.get("longitude")
        if lat is not None and lon is not None:
            try:
                lats.append(float(lat))
                lons.append(float(lon))
            except Exception:
                pass

    geo_bounds = None
    if lats and lons:
        margin_deg = float(search_range_deg)
        geo_bounds = {
            "minlatitude": min(lats) - margin_deg,
            "maxlatitude": max(lats) + margin_deg,
            "minlongitude": min(lons) - margin_deg,
            "maxlongitude": max(lons) + margin_deg,
        }

    return station_info, geo_bounds


def _extract_station_coords(inv, station_code: str) -> Optional[Tuple[float, float, float, str]]:
    for net in inv:
        for sta in net:
            if sta.code == station_code:
                return float(sta.latitude), float(sta.longitude), float(sta.elevation), net.code
    for net in inv:
        for sta in net:
            return float(sta.latitude), float(sta.longitude), float(sta.elevation), net.code
    return None


def _query_station(
    station: str,
    info: Dict,
    geo_bounds: Optional[Dict],
    time_range_days: float,
) -> Optional[Tuple[float, float, float, str]]:
    clients = [
        ("routing", RoutingClient("eida-routing")),
        ("iris", Client("IRIS")),
        ("geofon", Client("GFZ")),
    ]

    channels = sorted(info["channels"]) or [None]
    time = info["times"][0] if info["times"] else None
    dt_seconds = float(time_range_days) * 86400.0

    query_candidates = []
    for network in ["*", "KO"]:
        for use_geo_bounds in [True, False]:
            if time is not None:
                for ch in channels:
                    q = {
                        "network": network,
                        "station": station,
                        "level": "station",
                        "starttime": time - dt_seconds,
                        "endtime": time + dt_seconds,
                    }
                    if ch:
                        q["channel"] = f"{ch}*"
                    if use_geo_bounds and geo_bounds:
                        q.update(geo_bounds)
                    query_candidates.append(q)

            q = {"network": network, "station": station, "level": "station"}
            if use_geo_bounds and geo_bounds:
                q.update(geo_bounds)
            query_candidates.append(q)

    for _, client in clients:
        for kwargs in query_candidates:
            try:
                inv = client.get_stations(**kwargs)
                coords = _extract_station_coords(inv, station)
                if coords is not None:
                    return coords
            except Exception:
                pass
    return None


def main():
    args = parse_args()
    metadata_path = Path(args.metadata).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    rows = json.load(open(metadata_path, "r"))
    station_info, geo_bounds = _extract_station_queries(
        rows,
        search_range_deg=args.search_range_deg,
    )
    if not station_info:
        raise RuntimeError("No station names found in metadata.")

    station_locations = {}
    unresolved = []
    for station_name, info in sorted(station_info.items()):
        coords = _query_station(
            station_name,
            info,
            geo_bounds,
            time_range_days=args.time_range_days,
        )
        if coords is None:
            unresolved.append(station_name)
            continue
        lat, lon, elev, network = coords
        station_locations[station_name] = {
            "latitude": lat,
            "longitude": lon,
            "elevation_m": elev,
            "network": network,
        }
        print(
            f"Resolved {station_name}: lat={lat:.5f}, lon={lon:.5f}, "
            f"elev={elev:.1f}m, network={network}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(station_locations, f, indent=2, sort_keys=True)

    print(f"\nSaved station locations: {output_path}")
    print(f"Resolved: {len(station_locations)} / {len(station_info)}")
    if unresolved:
        print("Unresolved stations:", ", ".join(sorted(unresolved)))
        if args.strict:
            raise RuntimeError("Some stations could not be resolved in strict mode.")


if __name__ == "__main__":
    main()
