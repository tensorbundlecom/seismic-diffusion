#!/usr/bin/env python
"""
Fetch response-level station inventories (FDSN level="response") for every
station/channel-type in the training metadata and save them as one combined
StationXML file. The evaluation scripts use it to deconvolve the instrument
response and express waveforms as physical ground acceleration (m/s^2), like
the GWM paper, instead of raw digitizer counts.

Run from the project root:
    python eval/fetch_station_responses.py

Output: eval/station_responses.xml
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA = ROOT / "ML" / "diffusion" / "embeddings" / "metadata.json"
DEFAULT_OUTPUT = ROOT / "eval" / "station_responses.xml"


def parse_args():
    parser = argparse.ArgumentParser(description="Fetch station response inventories")
    parser.add_argument("--metadata", type=str, default=str(DEFAULT_METADATA))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--network", type=str, default="KO")
    return parser.parse_args()


def build_clients():
    from obspy.clients.fdsn import Client, RoutingClient

    clients = []
    # KOERI operates the KO network, so it goes first.
    for name in ["KOERI", "IRIS", "GFZ"]:
        try:
            clients.append((name, Client(name)))
        except Exception as exc:
            print(f"[responses] Client {name} unavailable: {exc}")
    try:
        clients.append(("eida-routing", RoutingClient("eida-routing")))
    except Exception as exc:
        print(f"[responses] eida-routing unavailable: {exc}")
    return clients


def has_response(inv, station: str) -> bool:
    for net in inv:
        for sta in net:
            if sta.code != station:
                continue
            for cha in sta:
                if cha.response is not None and cha.response.instrument_sensitivity:
                    return True
    return False


def main():
    args = parse_args()
    rows = json.load(open(args.metadata))

    station_channels = {}
    for row in rows:
        name = row.get("station_name")
        ch = row.get("channel_type", "HH")
        if name:
            station_channels.setdefault(name, set()).add(ch)
    print(f"[responses] {len(station_channels)} stations to resolve")

    clients = build_clients()
    combined = None
    resolved, unresolved = [], []

    for station, channels in sorted(station_channels.items()):
        channel_query = ",".join(f"{ch}?" for ch in sorted(channels))
        inv = None
        for network in [args.network, "*"]:
            for client_name, client in clients:
                try:
                    candidate = client.get_stations(
                        network=network,
                        station=station,
                        channel=channel_query,
                        level="response",
                    )
                except Exception:
                    continue
                if candidate is not None and has_response(candidate, station):
                    inv = candidate
                    print(f"[responses] {station}: ok via {client_name} ({channel_query})")
                    break
            if inv is not None:
                break

        if inv is None:
            unresolved.append(station)
            print(f"[responses] {station}: NO response found")
            continue
        resolved.append(station)
        combined = inv if combined is None else combined + inv

    if combined is None:
        raise RuntimeError("No station responses could be fetched.")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.write(str(out), format="STATIONXML")
    print(f"\n[responses] saved {out}  ({len(resolved)}/{len(station_channels)} stations)")
    if unresolved:
        print("[responses] unresolved:", ", ".join(unresolved))


if __name__ == "__main__":
    main()
