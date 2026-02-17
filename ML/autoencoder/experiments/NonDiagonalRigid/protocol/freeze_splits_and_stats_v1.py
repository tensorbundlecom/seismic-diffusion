#!/usr/bin/env python3
"""
Freeze NonDiagonalRigid V1 split manifests and train-only normalization stats.

Outputs:
  - protocol/splits_v1/{train,val,test,ood_primary}_files.txt
  - protocol/frozen_splits_v1.json
  - protocol/normalization_stats_v1.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

NORM_FEATURE_NAMES = ["magnitude", "log_repi_km", "depth_km"]


def _safe_std(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    std = arr.std(axis=0)
    std = np.where(std < eps, 1.0, std)
    return std


def _fit_normalization_stats(rows: List[Tuple[float, float, float]]) -> Dict:
    arr = np.asarray(rows, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("rows must have shape [N, 3]")
    mean = arr.mean(axis=0)
    std = _safe_std(arr)
    return {
        "feature_names": NORM_FEATURE_NAMES,
        "mean": {k: float(v) for k, v in zip(NORM_FEATURE_NAMES, mean)},
        "std": {k: float(v) for k, v in zip(NORM_FEATURE_NAMES, std)},
    }


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(1e-15, 1.0 - a)))
    return r * c


def _azimuth_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    # Forward azimuth from point-1 (event) to point-2 (station), degrees in [0, 360)
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    x = math.sin(dlambda) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    th = math.degrees(math.atan2(x, y))
    return (th + 360.0) % 360.0


def _build_geometry_condition(
    magnitude: float,
    event_lat: float,
    event_lon: float,
    event_depth_km: float,
    station_lat: float,
    station_lon: float,
) -> Dict[str, float]:
    repi_km = _haversine_km(event_lat, event_lon, station_lat, station_lon)
    az = _azimuth_deg(event_lat, event_lon, station_lat, station_lon)
    az_rad = math.radians(az)
    return {
        "magnitude": float(magnitude),
        "log_repi_km": float(np.log1p(repi_km)),
        "depth_km": float(event_depth_km),
        "sin_az": float(math.sin(az_rad)),
        "cos_az": float(math.cos(az_rad)),
        "repi_km": float(repi_km),
        "azimuth_deg": float(az),
    }


def _read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _sha256_lines(lines: Iterable[str]) -> str:
    h = hashlib.sha256()
    for line in lines:
        h.update(line.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _to_float(v: Optional[str]) -> float:
    if v is None:
        return float("nan")
    s = str(v).strip()
    if not s:
        return float("nan")
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _is_nan(x: float) -> bool:
    return x != x


def _load_event_lookup(event_file: Path) -> Dict[str, Dict]:
    for encoding in ["latin1", "windows-1254", "iso-8859-9", "utf-8"]:
        try:
            with event_file.open("r", encoding=encoding, newline="") as f:
                reader = csv.DictReader(f, delimiter="\t")
                rows = list(reader)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"Could not decode {event_file}")

    out: Dict[str, Dict] = {}
    for raw in rows:
        row = {str(k).strip(): v for k, v in raw.items()}
        event_id = str(row.get("Deprem Kodu", "")).strip()
        if not event_id:
            continue

        mags = {}
        for col in ["xM", "MD", "ML", "Mw", "Ms", "Mb"]:
            val = _to_float(row.get(col))
            mags[col] = float("nan") if val == 0.0 else val

        out[event_id] = {
            "event_id": event_id,
            "mags": mags,
            "latitude": _to_float(row.get("Enlem")),
            "longitude": _to_float(row.get("Boylam")),
            "depth_km": _to_float(row.get("Der(km)")),
        }
    return out


def _extract_station_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) < 3:
        return "UNKNOWN"
    return parts[-2]


def _extract_event_id_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) >= 3 and parts[0] == "OOD" and (parts[1] == "K" or parts[1] == "POST"):
        return f"{parts[0]}_{parts[1]}_{parts[2]}"
    if len(parts) >= 2 and parts[0] == "OOD":
        try:
            return f"OOD_{int(parts[1]):02d}"
        except ValueError:
            return f"{parts[0]}_{parts[1]}"
    return parts[0]


def _pick_magnitude(row: Dict, primary_col: str = "ML") -> float:
    mags = row["mags"]
    mag = mags.get(primary_col, float("nan"))
    if _is_nan(mag):
        for col in ["Mw", "ML", "Ms", "Mb", "MD", "xM"]:
            val = mags.get(col, float("nan"))
            if not _is_nan(val):
                mag = val
                break
    if _is_nan(mag):
        mag = 0.0
    return float(mag)


def _write_list(path: Path, items: List[str]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(f"{item}\n")
    return _sha256_lines(items)


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze V1 splits + normalization stats for NonDiagonalRigid.")
    parser.add_argument("--id-data-dir", default="data/external_dataset/extracted/data/filtered_waveforms/HH")
    parser.add_argument("--ood-data-dir", default="data/ood_waveforms/post_training_custom/filtered/HH")
    parser.add_argument(
        "--event-file",
        default="data/external_dataset/extracted/data/events/20140101_20251101_0.0_9.0_9_339.txt",
    )
    parser.add_argument("--station-list-file", default="data/station_list_external_full.json")
    parser.add_argument("--station-coords-file", default="ML/autoencoder/experiments/NonDiagonel/results/station_coords_external.json")
    parser.add_argument("--output-dir", default="ML/autoencoder/experiments/NonDiagonalRigid/protocol")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    args = parser.parse_args()

    id_data_dir = Path(args.id_data_dir)
    ood_data_dir = Path(args.ood_data_dir)
    event_file = Path(args.event_file)
    station_list_file = Path(args.station_list_file)
    station_coords_file = Path(args.station_coords_file)
    out_dir = Path(args.output_dir)
    splits_dir = out_dir / "splits_v1"

    if args.train_ratio <= 0 or args.val_ratio <= 0 or args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("train_ratio and val_ratio must be >0 and sum to <1.0")

    stations: List[str] = sorted(_read_json(station_list_file))
    station_set = set(stations)
    station_coords = _read_json(station_coords_file)
    missing_coords = sorted([s for s in stations if s not in station_coords])
    if missing_coords:
        raise ValueError(f"Missing station coordinates for: {missing_coords[:10]}")

    event_lookup = _load_event_lookup(event_file)

    id_files_all = sorted(id_data_dir.glob("*.mseed"))
    grouped: Dict[str, List[str]] = defaultdict(list)
    skipped_no_event = 0
    skipped_no_station = 0
    skipped_unknown_station = 0
    for fp in id_files_all:
        station = _extract_station_from_filename(fp.name)
        if station == "UNKNOWN":
            skipped_no_station += 1
            continue
        if station not in station_set:
            skipped_unknown_station += 1
            continue
        event_id = _extract_event_id_from_filename(fp.name)
        if event_id not in event_lookup:
            skipped_no_event += 1
            continue
        evt = event_lookup[event_id]
        if _is_nan(evt["latitude"]) or _is_nan(evt["longitude"]) or _is_nan(evt["depth_km"]):
            skipped_no_event += 1
            continue
        grouped[event_id].append(fp.as_posix())

    event_ids = sorted(grouped.keys())
    if not event_ids:
        raise RuntimeError("No valid ID files after filtering.")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(event_ids)

    n_events = len(event_ids)
    n_train_events = int(args.train_ratio * n_events)
    n_val_events = int(args.val_ratio * n_events)
    train_events = sorted(perm[:n_train_events].tolist())
    val_events = sorted(perm[n_train_events : n_train_events + n_val_events].tolist())
    test_events = sorted(perm[n_train_events + n_val_events :].tolist())

    train_event_set = set(train_events)
    val_event_set = set(val_events)
    test_event_set = set(test_events)

    train_files = sorted([p for e in train_events for p in grouped[e]])
    val_files = sorted([p for e in val_events for p in grouped[e]])
    test_files = sorted([p for e in test_events for p in grouped[e]])

    inter = (train_event_set & val_event_set) | (train_event_set & test_event_set) | (val_event_set & test_event_set)
    if inter:
        raise RuntimeError("Event leakage detected between train/val/test.")

    ood_files_all = sorted(ood_data_dir.glob("*.mseed"))
    ood_primary_files = []
    for fp in ood_files_all:
        station = _extract_station_from_filename(fp.name)
        if station in station_set:
            ood_primary_files.append(fp.as_posix())
    ood_primary_files = sorted(ood_primary_files)

    # Write split files
    train_txt = splits_dir / "train_files.txt"
    val_txt = splits_dir / "val_files.txt"
    test_txt = splits_dir / "test_files.txt"
    ood_txt = splits_dir / "ood_primary_files.txt"
    train_events_txt = splits_dir / "train_event_ids.txt"
    val_events_txt = splits_dir / "val_event_ids.txt"
    test_events_txt = splits_dir / "test_event_ids.txt"

    train_hash = _write_list(train_txt, train_files)
    val_hash = _write_list(val_txt, val_files)
    test_hash = _write_list(test_txt, test_files)
    ood_hash = _write_list(ood_txt, ood_primary_files)
    train_events_hash = _write_list(train_events_txt, train_events)
    val_events_hash = _write_list(val_events_txt, val_events)
    test_events_hash = _write_list(test_events_txt, test_events)
    station_hash = _sha256_lines(stations)

    # Compute train-only normalization stats
    norm_rows: List[Tuple[float, float, float]] = []
    for fpath in train_files:
        name = Path(fpath).name
        event_id = _extract_event_id_from_filename(name)
        station = _extract_station_from_filename(name)
        row = event_lookup[event_id]

        magnitude = _pick_magnitude(row, primary_col="ML")
        event_lat = float(row["latitude"])
        event_lon = float(row["longitude"])
        event_depth = float(row["depth_km"])

        sc = station_coords[station]
        cond = _build_geometry_condition(
            magnitude=magnitude,
            event_lat=event_lat,
            event_lon=event_lon,
            event_depth_km=event_depth,
            station_lat=float(sc["latitude"]),
            station_lon=float(sc["longitude"]),
        )
        norm_rows.append((cond["magnitude"], cond["log_repi_km"], cond["depth_km"]))

    norm_stats = _fit_normalization_stats(norm_rows)
    norm_payload = {
        "version": "v1",
        "status": "FROZEN",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "split_manifest": "ML/autoencoder/experiments/NonDiagonalRigid/protocol/frozen_splits_v1.json",
            "train_file_count": len(train_files),
            "features_full_order": ["magnitude", "log_repi_km", "depth_km", "sin_az", "cos_az"],
            "normalized_features": ["magnitude", "log_repi_km", "depth_km"],
        },
        # Compatibility fields for existing dataset loaders (expect top-level mean/std)
        "feature_names": norm_stats["feature_names"],
        "mean": norm_stats["mean"],
        "std": norm_stats["std"],
        "features": norm_stats,
    }

    _write_json(out_dir / "normalization_stats_v1.json", norm_payload)

    split_payload = {
        "version": "v1",
        "status": "FROZEN",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "split_unit": "event_level",
            "seed": args.seed,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": round(1.0 - args.train_ratio - args.val_ratio, 6),
            "id_channel_dir": id_data_dir.as_posix(),
            "ood_channel_dir": ood_data_dir.as_posix(),
            "event_file": event_file.as_posix(),
        },
        "stations": stations,
        "hashes_sha256": {
            "stations": station_hash,
            "train_files": train_hash,
            "val_files": val_hash,
            "test_files": test_hash,
            "ood_primary_files": ood_hash,
            "train_event_ids": train_events_hash,
            "val_event_ids": val_events_hash,
            "test_event_ids": test_events_hash,
        },
        "splits": {
            "train_files": {
                "file": train_txt.as_posix(),
                "count": len(train_files),
                "event_count": len(train_events),
            },
            "val_files": {
                "file": val_txt.as_posix(),
                "count": len(val_files),
                "event_count": len(val_events),
            },
            "test_files": {
                "file": test_txt.as_posix(),
                "count": len(test_files),
                "event_count": len(test_events),
            },
            "ood_primary_files": {
                "file": ood_txt.as_posix(),
                "count": len(ood_primary_files),
                "event_count": len({_extract_event_id_from_filename(Path(p).name) for p in ood_primary_files}),
            },
            "train_event_ids": {"file": train_events_txt.as_posix(), "count": len(train_events)},
            "val_event_ids": {"file": val_events_txt.as_posix(), "count": len(val_events)},
            "test_event_ids": {"file": test_events_txt.as_posix(), "count": len(test_events)},
        },
        "filter_report": {
            "id_files_total_seen": len(id_files_all),
            "id_files_kept": len(train_files) + len(val_files) + len(test_files),
            "skipped_no_event_in_catalog": skipped_no_event,
            "skipped_unknown_station": skipped_unknown_station,
            "skipped_bad_filename_station": skipped_no_station,
            "ood_files_total_seen": len(ood_files_all),
            "ood_files_kept": len(ood_primary_files),
        },
    }

    _write_json(out_dir / "frozen_splits_v1.json", split_payload)

    print("[OK] Frozen split manifest:", (out_dir / "frozen_splits_v1.json").as_posix())
    print("[OK] Frozen normalization stats:", (out_dir / "normalization_stats_v1.json").as_posix())
    print(
        "[OK] Counts:",
        {
            "train_files": len(train_files),
            "val_files": len(val_files),
            "test_files": len(test_files),
            "ood_primary_files": len(ood_primary_files),
            "train_events": len(train_events),
            "val_events": len(val_events),
            "test_events": len(test_events),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
