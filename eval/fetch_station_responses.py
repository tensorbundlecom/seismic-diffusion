#!/usr/bin/env python
"""Build and validate an epoch-aware response inventory for waveform correction.

This reads the actual raw MiniSEED headers selected for the current AE corpus
(the same ``max(snr1..3) > 2`` and ``max(gap1..3) == 0`` selection used for
``filtered_waveforms_snr2_2-15hz``). It requires an exact
``network.station.location.channel`` response epoch covering every component's
recording interval; finding a response for a different channel, location, or
epoch never counts as coverage.

Examples (from the project root)::

    python eval/fetch_station_responses.py
    python eval/fetch_station_responses.py --no-fetch --strict
    python eval/fetch_station_responses.py --waveform-dir data/filtered_waveforms

Outputs next to ``--output``: exact trace manifest, CSV/JSON coverage report,
resumable FDSN request cache, and merged StationXML.  Never run physical-unit
preprocessing when ``--strict`` fails: mixing corrected and count-domain traces
would invalidate global amplitude normalization.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = ROOT / "data" / "waveforms"
DEFAULT_SUMMARY = ROOT / "data" / "waveform_summary.csv"
DEFAULT_FILTERED_DIR = ROOT / "data" / "filtered_waveforms_snr2_2-15hz"
DEFAULT_OUTPUT = ROOT / "eval" / "station_responses.xml"
_INVENTORY_INDEX: dict[int, tuple[Any, dict[tuple[str, str, str, str], list[tuple[Any, Any, Any]]]]] = {}


@dataclass(frozen=True)
class TraceRequest:
    """Exact response identity and interval required by a MiniSEED trace."""

    source_file: str
    raw_file: str
    # ``network`` is the value as stored in the raw MiniSEED header.  Keep it
    # intact for provenance: some historical KOERI files omitted it even
    # though the source network is known from their acquisition provenance.
    network: str
    effective_network: str
    network_inferred: bool
    station: str
    location: str
    channel: str
    starttime: str
    endtime: str
    sampling_rate: float
    npts: int

    @property
    def nslc(self) -> str:
        """Raw/header NSLC, including a possibly blank network code."""
        return ".".join((self.network, self.station, self.location, self.channel))

    @property
    def effective_nslc(self) -> str:
        """NSLC used to fetch and match StationXML response epochs."""
        return ".".join((self.effective_network, self.station, self.location, self.channel))

    @property
    def request_key(self) -> str:
        return "|".join((self.effective_network, self.station, self.location, self.channel))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch and validate exact channel/epoch StationXML coverage."
    )
    selection = parser.add_argument_group("waveform selection")
    selection.add_argument(
        "--waveform-summary", type=Path, default=DEFAULT_SUMMARY,
        help="Summary CSV used to select raw files (default: data/waveform_summary.csv).",
    )
    selection.add_argument(
        "--snr-threshold", type=float, default=2.0,
        help="Keep summary rows with max SNR > value and max gap == 0 (default: 2).",
    )
    selection.add_argument(
        "--waveform-dir", type=Path, default=None,
        help="Instead scan this selected directory recursively; bypasses summary selection.",
    )
    selection.add_argument(
        "--raw-waveforms-dir", type=Path, default=DEFAULT_RAW_DIR,
        help="Raw counts-domain waveform root used for exact headers.",
    )
    selection.add_argument(
        "--limit-files", type=int, default=None,
        help="Process only the first N selected files (for a small dry run).",
    )
    selection.add_argument(
        "--default-network", default="KO",
        help=(
            "Use this network only for traces whose raw MiniSEED network header is blank "
            "(default: KO). The header value is retained in every manifest/report; pass "
            "an empty string to disable this fallback."
        ),
    )

    output = parser.add_argument_group("output")
    output.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    output.add_argument("--manifest", type=Path, default=None)
    output.add_argument("--coverage-csv", type=Path, default=None)
    output.add_argument("--coverage-json", type=Path, default=None)
    output.add_argument("--fetch-cache", type=Path, default=None)

    fetch = parser.add_argument_group("FDSN fetching")
    fetch.add_argument("--no-fetch", action="store_true", help="Only validate existing --output.")
    fetch.add_argument(
        "--clients", nargs="+", default=["KOERI", "IRIS", "GFZ", "eida-routing"],
        help="FDSN clients tried in order; eida-routing uses ObsPy RoutingClient.",
    )
    fetch.add_argument(
        "--force-refetch", action="store_true",
        help=(
            "Also query already-covered NSLCs for diagnostics. Existing epochs are "
            "kept to avoid duplicate StationXML records."
        ),
    )
    fetch.add_argument(
        "--strict", action="store_true",
        help="Exit non-zero unless every selected three-component waveform is covered.",
    )
    return parser.parse_args()


def _utc(t: str):
    from obspy import UTCDateTime
    return UTCDateTime(t)


def _sidecar(output: Path, name: str) -> Path:
    return output.with_name(name)


def _resolve_raw_path(selected: Path, waveform_dir: Path, raw_dir: Path) -> Path:
    try:
        candidate = raw_dir / selected.resolve().relative_to(waveform_dir.resolve())
    except ValueError:
        candidate = raw_dir / selected.parent.name / selected.name
    return candidate if candidate.exists() else selected


def _summary_paths(summary_path: Path, raw_dir: Path, snr_threshold: float | None) -> list[Path]:
    import pandas as pd

    df = pd.read_csv(summary_path)
    if "waveform_file" not in df:
        raise ValueError(f"{summary_path} has no waveform_file column")
    if snr_threshold is not None:
        needed = {"snr1", "snr2", "snr3", "gap1", "gap2", "gap3"}
        missing = sorted(needed - set(df.columns))
        if missing:
            raise ValueError(f"--snr-threshold requires columns: {', '.join(missing)}")
        snr = df[["snr1", "snr2", "snr3"]].max(axis=1)
        gaps = df[["gap1", "gap2", "gap3"]].max(axis=1)
        df = df[(snr > snr_threshold) & (gaps == 0)]
    paths: list[Path] = []
    for value in df["waveform_file"].dropna().astype(str).unique():
        path = Path(value)
        if path.exists():
            paths.append(path)
            continue
        candidates = [ROOT / path]
        if "waveforms" in path.parts:
            candidates.append(raw_dir.joinpath(*path.parts[path.parts.index("waveforms") + 1:]))
        existing = next((candidate for candidate in candidates if candidate.exists()), None)
        paths.append(existing if existing is not None else path)
    return paths


def selected_files(args: argparse.Namespace) -> list[tuple[Path, Path]]:
    """Return sorted (selected-file, raw-header-file) pairs."""
    raw_dir = args.raw_waveforms_dir.expanduser().resolve()
    if args.waveform_dir:
        waveform_dir = args.waveform_dir.expanduser().resolve()
        if not waveform_dir.exists():
            raise FileNotFoundError(f"Waveform directory does not exist: {waveform_dir}")
        pairs = [(p, _resolve_raw_path(p, waveform_dir, raw_dir)) for p in waveform_dir.rglob("*.mseed")]
    else:
        summary = args.waveform_summary.expanduser().resolve()
        if not summary.exists():
            raise FileNotFoundError(f"Summary does not exist: {summary}")
        pairs = [(p, p) for p in _summary_paths(summary, raw_dir, args.snr_threshold)]
    pairs = sorted(dict.fromkeys(pairs), key=lambda item: str(item[0]))
    if args.limit_files is not None:
        if args.limit_files <= 0:
            raise ValueError("--limit-files must be positive")
        pairs = pairs[:args.limit_files]
    if not pairs:
        raise RuntimeError("No selected MiniSEED files found")
    return pairs


def build_manifest(pairs: Iterable[tuple[Path, Path]], default_network: str = "KO") -> tuple[list[TraceRequest], list[dict[str, Any]]]:
    """Read headers only and retain malformed files in the coverage denominator."""
    from obspy import read

    requests: list[TraceRequest] = []
    files: list[dict[str, Any]] = []
    iterator = tqdm(pairs, desc="Reading MiniSEED headers", unit="file")
    for selected, raw in iterator:
        record: dict[str, Any] = {
            "source_file": str(selected), "raw_file": str(raw), "status": "ok",
            "trace_count": 0, "trace_ids": [], "effective_trace_ids": [],
            "inferred_network_trace_count": 0, "error": None,
        }
        try:
            stream = read(str(raw), headonly=True)
            record["trace_count"] = len(stream)
            for trace in stream:
                s = trace.stats
                header_network = str(s.network or "")
                inferred = bool(not header_network and default_network)
                request = TraceRequest(
                    source_file=str(selected), raw_file=str(raw), network=header_network,
                    effective_network=default_network if inferred else header_network,
                    network_inferred=inferred,
                    station=str(s.station or ""), location=str(s.location or ""),
                    channel=str(s.channel or ""), starttime=str(s.starttime), endtime=str(s.endtime),
                    sampling_rate=float(s.sampling_rate), npts=int(s.npts),
                )
                requests.append(request)
                record["trace_ids"].append(request.nslc)
                record["effective_trace_ids"].append(request.effective_nslc)
                record["inferred_network_trace_count"] += int(inferred)
            if len(stream) != 3:
                record["status"] = "invalid_component_count"
        except Exception as exc:
            record["status"] = "unreadable"
            record["error"] = f"{type(exc).__name__}: {exc}"
        files.append(record)
    return requests, files


def _valid_response(channel: Any) -> bool:
    response = getattr(channel, "response", None)
    return response is not None and getattr(response, "instrument_sensitivity", None) is not None


def _index_inventory(inventory: Any) -> dict[tuple[str, str, str, str], list[tuple[Any, Any, Any]]]:
    """Index StationXML once; coverage validation may inspect 325k traces."""
    cache_key = id(inventory)
    cached = _INVENTORY_INDEX.get(cache_key)
    if cached is not None and cached[0] is inventory:
        return cached[1]
    index: dict[tuple[str, str, str, str], list[tuple[Any, Any, Any]]] = defaultdict(list)
    for network in inventory:
        for station in network:
            for channel in station:
                index[(network.code, station.code, channel.location_code or "", channel.code)].append(
                    (network, station, channel)
                )
    _INVENTORY_INDEX[cache_key] = (inventory, index)
    return index


def find_covering_epoch(inventory: Any, request: TraceRequest) -> tuple[Any, Any, Any] | None:
    """Find only an exact NSLC epoch that covers the complete trace interval."""
    start, end = _utc(request.starttime), _utc(request.endtime)
    exact = (request.effective_network, request.station, request.location, request.channel)
    for network, station, channel in _index_inventory(inventory).get(exact, []):
        channel_start, channel_end = channel.start_date, channel.end_date
        covered = ((channel_start is None or channel_start <= start) and
                   (channel_end is None or channel_end >= end))
        if covered and _valid_response(channel):
            return network, station, channel
    return None


def load_inventory(path: Path) -> Any | None:
    if not path.exists():
        return None
    from obspy import read_inventory
    return read_inventory(str(path))


def build_clients(names: Iterable[str]) -> list[tuple[str, Any]]:
    from obspy.clients.fdsn import Client, RoutingClient
    clients: list[tuple[str, Any]] = []
    for name in names:
        try:
            clients.append((name, RoutingClient(name) if name == "eida-routing" else Client(name)))
        except Exception as exc:
            print(f"[responses] client {name} unavailable: {type(exc).__name__}: {exc}")
    if not clients:
        raise RuntimeError("No FDSN clients could be initialized")
    return clients


def _groups(requests: Iterable[TraceRequest]) -> dict[str, list[TraceRequest]]:
    groups: dict[str, list[TraceRequest]] = defaultdict(list)
    for request in requests:
        groups[request.request_key].append(request)
    return groups


def _load_json(path: Path, fallback: Any) -> Any:
    try:
        with path.open() as handle:
            return json.load(handle)
    except FileNotFoundError:
        return fallback
    except json.JSONDecodeError as exc:
        print(f"[responses] ignoring unreadable cache {path}: {exc}")
        return fallback


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def _atomic_inventory(path: Path, inventory: Any) -> None:
    """Durably checkpoint StationXML so a killed fetch can be resumed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    inventory.write(str(temporary), format="STATIONXML")
    os.replace(temporary, path)


def _channel_epoch_key(network: Any, station: Any, channel: Any) -> tuple[str, str, str, str, str, str]:
    """Identity used to retain distinct response changes but drop exact repeats."""
    return (
        network.code,
        station.code,
        channel.location_code or "",
        channel.code,
        str(channel.start_date or ""),
        str(channel.end_date or ""),
    )


def _merge_inventory(existing: Any | None, incoming: Any) -> Any:
    """Merge StationXML without duplicating response epochs from partial runs."""
    if existing is None:
        return incoming
    merged = copy.deepcopy(existing)
    existing_epochs = {
        _channel_epoch_key(network, station, channel)
        for network in merged
        for station in network
        for channel in station
    }
    # Some FDSN services emit one Network object per station even when the
    # network code is identical. Index stations across all such objects rather
    # than assuming ``network.code`` is unique in a StationXML file.
    networks: dict[str, Any] = {}
    stations: dict[tuple[str, str], tuple[Any, Any]] = {}
    for network in merged:
        networks.setdefault(network.code, network)
        for station in network:
            stations.setdefault((network.code, station.code), (network, station))
    for new_network in incoming:
        target_network = networks.get(new_network.code)
        if target_network is None:
            target_network = copy.deepcopy(new_network)
            merged.networks.append(target_network)
            networks[new_network.code] = target_network
            for station in target_network:
                stations[(target_network.code, station.code)] = (target_network, station)
                for channel in station:
                    existing_epochs.add(_channel_epoch_key(target_network, station, channel))
            continue
        for new_station in new_network:
            target = stations.get((new_network.code, new_station.code))
            if target is None:
                target_station = copy.deepcopy(new_station)
                target_network.stations.append(target_station)
                stations[(new_network.code, new_station.code)] = (target_network, target_station)
                for channel in target_station:
                    existing_epochs.add(_channel_epoch_key(target_network, target_station, channel))
                continue
            _, target_station = target
            for new_channel in new_station:
                key = _channel_epoch_key(new_network, new_station, new_channel)
                if key not in existing_epochs:
                    target_station.channels.append(copy.deepcopy(new_channel))
                    existing_epochs.add(key)
    return merged


def _fetch_group(group: list[TraceRequest], clients: Iterable[tuple[str, Any]]) -> tuple[Any | None, dict[str, Any]]:
    """Fetch every response epoch spanning this exact NSLC's selected times."""
    item = group[0]
    kwargs = {
        "network": item.effective_network, "station": item.station,
        # ``--`` is the FDSN representation of the empty SEED location code.
        "location": item.location if item.location else "--", "channel": item.channel,
        # Select every epoch overlapping the selected time range. Reversing
        # these bounds would accidentally require one response epoch to span
        # the entire multi-year corpus, thereby losing legitimate gain/response
        # changes that must be retained for exact per-trace validation.
        "startbefore": max(_utc(req.endtime) for req in group),
        "endafter": min(_utc(req.starttime) for req in group), "level": "response",
    }
    errors: list[str] = []
    for client_name, client in clients:
        try:
            candidate = client.get_stations(**kwargs)
        except Exception as exc:
            errors.append(f"{client_name}: {type(exc).__name__}: {exc}")
            continue
        if candidate is not None and any(find_covering_epoch(candidate, req) for req in group):
            return candidate, {"status": "fetched", "client": client_name,
                               "query": {key: str(value) for key, value in kwargs.items()}, "errors": errors}
        errors.append(f"{client_name}: returned no matching response epoch")
    return None, {"status": "unresolved", "query": {key: str(value) for key, value in kwargs.items()}, "errors": errors}


def fetch_missing(inventory: Any | None, requests: list[TraceRequest], clients: list[tuple[str, Any]],
                  cache_path: Path, inventory_path: Path, force_refetch: bool) -> tuple[Any | None, dict[str, Any]]:
    """Fetch uncovered NSLCs, checkpointing after every request for resumability."""
    cache = _load_json(cache_path, {"schema_version": 1, "requests": {}})
    cache.setdefault("requests", {})
    groups = _groups(requests)
    for index, (key, group) in enumerate(sorted(groups.items()), start=1):
        already_covered = inventory is not None and all(find_covering_epoch(inventory, req) for req in group)
        if already_covered and not force_refetch:
            cache["requests"][key] = {"status": "covered_by_existing_inventory", "trace_count": len(group)}
            continue
        fetched, outcome = _fetch_group(group, clients)
        outcome["trace_count"] = len(group)
        if fetched is not None:
            if already_covered:
                # Do not append an identical response epoch merely because a
                # diagnostic --force-refetch was requested. Duplicated channel
                # entries make subsequent epoch selection ambiguous.
                outcome["status"] = "verified_existing_inventory"
                print(f"[responses] {index}/{len(groups)} {key}: existing response verified via {outcome['client']}")
            else:
                inventory = _merge_inventory(inventory, fetched)
                # The response object itself must survive an interruption; the
                # JSON cache only records outcomes and cannot rebuild it.
                _atomic_inventory(inventory_path, inventory)
                print(f"[responses] {index}/{len(groups)} {key}: fetched via {outcome['client']} (checkpointed)")
        else:
            print(f"[responses] {index}/{len(groups)} {key}: unresolved")
        cache["requests"][key] = outcome
        _atomic_json(cache_path, cache)
    _atomic_json(cache_path, cache)
    return inventory, cache


def coverage_rows(inventory: Any | None, requests: list[TraceRequest]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for request in requests:
        match = find_covering_epoch(inventory, request) if inventory is not None else None
        row = asdict(request)
        row["nslc"] = request.nslc  # Backwards-compatible raw/header identifier.
        row["header_nslc"] = request.nslc
        row["effective_nslc"] = request.effective_nslc
        row["covered"] = match is not None
        if match:
            network, station, channel = match
            row.update({"matched_network": network.code, "matched_station": station.code,
                        "matched_location": channel.location_code or "", "matched_channel": channel.code,
                        "epoch_start": str(channel.start_date or ""), "epoch_end": str(channel.end_date or "")})
        else:
            row.update({key: "" for key in ("matched_network", "matched_station", "matched_location",
                                               "matched_channel", "epoch_start", "epoch_end")})
        rows.append(row)
    return rows


def _fingerprint(requests: list[TraceRequest]) -> str:
    digest = hashlib.sha256()
    for request in requests:
        digest.update(json.dumps(asdict(request), sort_keys=True).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def write_reports(manifest_path: Path, csv_path: Path, json_path: Path, requests: list[TraceRequest],
                  files: list[dict[str, Any]], rows: list[dict[str, Any]],
                  default_network: str) -> dict[str, Any]:
    fingerprint = _fingerprint(requests)
    network_resolution = {
        "blank_header_network_fallback": default_network or None,
        "description": (
            "For a blank raw MiniSEED network header, effective_network is the configured "
            "fallback used for StationXML fetch/match. Raw network/header NSLC remain unchanged."
            if default_network else
            "Blank raw MiniSEED network headers are not resolved to a fallback network."
        ),
    }
    _atomic_json(manifest_path, {"schema_version": 2, "created_utc": datetime.now(timezone.utc).isoformat(),
                                 "selection_fingerprint_sha256": fingerprint,
                                 "network_resolution": network_resolution,
                                 "trace_requests": [asdict(item) for item in requests], "files": files})
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else list(TraceRequest.__dataclass_fields__) + ["nslc", "covered"]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_file[row["source_file"]].append(row)
    coverage: dict[str, dict[str, Any]] = {}
    for source, file_rows in by_file.items():
        coverage[source] = {"trace_count": len(file_rows), "covered_trace_count": sum(bool(row["covered"]) for row in file_rows),
                            "required_three_components": len(file_rows) == 3,
                            "covered": len(file_rows) == 3 and all(row["covered"] for row in file_rows),
                            "inferred_network_trace_count": sum(bool(row["network_inferred"]) for row in file_rows),
                            "network_inference_applied": any(row["network_inferred"] for row in file_rows)}
    for item in files:
        if item["status"] != "ok":
            coverage[item["source_file"]] = {"trace_count": item["trace_count"], "covered_trace_count": 0,
                                                "required_three_components": False, "covered": False,
                                                "status": item["status"], "error": item["error"]}
    covered = sum(bool(row["covered"]) for row in rows)
    inferred_trace_count = sum(bool(row["network_inferred"]) for row in rows)
    inferred_file_count = len({row["source_file"] for row in rows if row["network_inferred"]})
    report = {"schema_version": 2, "created_utc": datetime.now(timezone.utc).isoformat(),
              "selection_fingerprint_sha256": fingerprint,
              "network_resolution": network_resolution,
              "summary": {"selected_files": len(files), "valid_three_component_files": sum(item["status"] == "ok" for item in files),
                          "invalid_or_unreadable_files": sum(item["status"] != "ok" for item in files),
                          "trace_count": len(rows), "covered_trace_count": covered, "uncovered_trace_count": len(rows) - covered,
                          "inferred_network_trace_count": inferred_trace_count,
                          "inferred_network_file_count": inferred_file_count,
                          "fully_covered_files": sum(item["covered"] for item in coverage.values()),
                          "incomplete_files": sum(not item["covered"] for item in coverage.values()),
                          "complete": bool(rows) and covered == len(rows) and all(item["covered"] for item in coverage.values())},
              "uncovered_nslc_counts": dict(sorted(Counter(row["nslc"] for row in rows if not row["covered"]).items())),
              "files": coverage}
    _atomic_json(json_path, report)
    return report


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    manifest_path = (args.manifest or _sidecar(output, "station_response_manifest.json")).expanduser().resolve()
    csv_path = (args.coverage_csv or _sidecar(output, "station_response_coverage.csv")).expanduser().resolve()
    json_path = (args.coverage_json or _sidecar(output, "station_response_coverage.json")).expanduser().resolve()
    cache_path = (args.fetch_cache or _sidecar(output, "station_response_fetch_cache.json")).expanduser().resolve()

    pairs = selected_files(args)
    print(f"[responses] reading exact MiniSEED headers for {len(pairs):,} selected files")
    default_network = args.default_network.strip().upper()
    requests, files = build_manifest(pairs, default_network=default_network)
    print(f"[responses] found {len(requests):,} traces across {len(_groups(requests)):,} unique NSLCs")
    malformed = sum(item["status"] != "ok" for item in files)
    if malformed:
        print(f"[responses] WARNING: {malformed} files are unreadable or not exactly three-component")

    inventory = load_inventory(output)
    if args.no_fetch:
        if inventory is None:
            print(f"[responses] no existing inventory at {output}; all traces are uncovered")
    else:
        inventory, _ = fetch_missing(
            inventory, requests, build_clients(args.clients), cache_path, output, args.force_refetch
        )
        if inventory is not None:
            _atomic_inventory(output, inventory)
            print(f"[responses] wrote merged StationXML: {output}")
        else:
            print("[responses] no response inventory was fetched")

    report = write_reports(
        manifest_path, csv_path, json_path, requests, files, coverage_rows(inventory, requests), default_network
    )
    summary = report["summary"]
    print(f"[responses] coverage: {summary['covered_trace_count']:,}/{summary['trace_count']:,} traces, "
          f"{summary['fully_covered_files']:,}/{summary['selected_files']:,} full waveforms")
    if summary["inferred_network_trace_count"]:
        print(
            "[responses] blank-network fallback: "
            f"{summary['inferred_network_trace_count']:,} traces in "
            f"{summary['inferred_network_file_count']:,} files used {default_network!r}; "
            "raw MiniSEED headers were not modified"
        )
    print(f"[responses] manifest: {manifest_path}\n[responses] report:   {json_path}")
    if not summary["complete"]:
        print("[responses] INCOMPLETE: do not mix corrected and uncorrected waveforms.")
        if args.strict:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
