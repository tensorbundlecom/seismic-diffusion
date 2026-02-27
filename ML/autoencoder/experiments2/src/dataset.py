"""Dataset and data-preparation utilities for experiments2/exp001."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy import signal
from torch.utils.data import Dataset
from collections import Counter, defaultdict

from .utils import ensure_dir, load_json, load_jsonl, save_json, save_jsonl

try:
    from obspy import read  # type: ignore
except Exception as exc:  # pragma: no cover - import-guard for environments without obspy
    read = None
    _OBSPY_IMPORT_ERROR = exc
else:
    _OBSPY_IMPORT_ERROR = None


EARTH_RADIUS_KM = 6371.0
EARTH_KM_PER_DEG = math.pi * EARTH_RADIUS_KM / 180.0
_PHASE_PREFIX_P = ("P",)
_PHASE_PREFIX_S = ("S",)


def _to_utc_naive(dt: datetime) -> datetime:
    """
    Normalize datetime for robust ordering/subtraction.

    - naive datetime -> keep as-is (assumed same reference frame)
    - timezone-aware datetime -> convert to UTC and drop tzinfo
    """
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def _file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _object_sha256(obj: Any) -> str:
    data = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _require_obspy() -> None:
    if read is None:
        raise ImportError(
            "ObsPy is required for mseed reading but is not available in this environment."
        ) from _OBSPY_IMPORT_ERROR


def _to_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None:
            return default
        return float(str(value).strip())
    except Exception:
        return default


def _first_valid_magnitude(row: Mapping[str, Any]) -> float:
    for key in ("xM", "ML", "Mw", "Ms", "Mb", "MD"):
        v = _to_float(row.get(key))
        if math.isfinite(v) and v > 0:
            return v
    return 0.0


def _parse_event_origin(row: Mapping[str, Any]) -> Optional[datetime]:
    date_s = str(row.get("Olus tarihi", "")).strip()
    time_s = str(row.get("Olus zamani", "")).strip()
    if not date_s or not time_s:
        return None
    date_s = date_s.replace("/", ".")
    if "." not in time_s:
        time_s = f"{time_s}.0"
    try:
        return _to_utc_naive(datetime.strptime(f"{date_s} {time_s}", "%Y.%m.%d %H:%M:%S.%f"))
    except ValueError:
        # Fallback if milliseconds are absent.
        try:
            return _to_utc_naive(datetime.strptime(f"{date_s} {time_s.split('.')[0]}", "%Y.%m.%d %H:%M:%S"))
        except ValueError:
            return None


def load_event_catalog(event_catalog_path: str | Path) -> Dict[str, Dict[str, Any]]:
    """
    Parse KOERI event catalog and return event_id -> metadata map.
    """
    event_catalog_path = Path(event_catalog_path)
    with event_catalog_path.open("r", encoding="latin1", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        out: Dict[str, Dict[str, Any]] = {}
        for raw_row in reader:
            row = {str(k).strip(): v for k, v in raw_row.items()}
            event_id = str(row.get("Deprem Kodu", "")).strip()
            if not event_id:
                continue
            depth_km = _to_float(row.get("Der(km)"), default=0.0)
            origin_dt = _parse_event_origin(row)
            out[event_id] = {
                "event_id": event_id,
                "magnitude": _first_valid_magnitude(row),
                "depth_km": depth_km if math.isfinite(depth_km) else 0.0,
                "origin_time": origin_dt.isoformat() if origin_dt is not None else None,
                "origin_date": str(row.get("Olus tarihi", "")).strip(),
                "origin_clock": str(row.get("Olus zamani", "")).strip(),
                "latitude": _to_float(row.get("Enlem"), default=float("nan")),
                "longitude": _to_float(row.get("Boylam"), default=float("nan")),
                "place": str(row.get("Yer", "")).strip(),
            }
    return out


def _extract_event_station(file_path: str | Path) -> tuple[str, str]:
    stem = Path(file_path).stem
    parts = stem.split("_")
    if len(parts) < 2:
        return stem, "UNKNOWN"
    return parts[0], parts[1]


def _parse_phase_picks_targeted(
    phase_pick_dir: str | Path,
    needed_pairs: set[tuple[str, str]],
) -> Dict[tuple[str, str], Dict[str, float]]:
    """
    Parse GSE pick files and extract (dist_deg, azimuth_deg) for requested (event, station) pairs.
    """
    phase_pick_dir = Path(phase_pick_dir)
    out: Dict[tuple[str, str], Dict[str, float]] = {}
    current_event: Optional[str] = None

    files = sorted(phase_pick_dir.glob("*.txt"))
    for fp in files:
        with fp.open("r", encoding="latin1", errors="ignore") as f:
            for line in f:
                if line.startswith("EVENT "):
                    parts = line.strip().split()
                    current_event = parts[1] if len(parts) >= 2 else None
                    continue
                if current_event is None:
                    continue
                if not line.strip():
                    continue
                if line.startswith(("Sta", "Date", "BEGIN", "MSG", "DATA", "EVENT")):
                    continue

                station = line[:5].strip()
                if not station:
                    continue
                key = (current_event, station)
                if key not in needed_pairs:
                    continue

                toks = line.split()
                if len(toks) < 4:
                    continue
                dist_deg = _to_float(toks[1])
                evaz = _to_float(toks[2])
                if not math.isfinite(dist_deg) or not math.isfinite(evaz):
                    continue

                phase_tok = None
                for tok in toks[3:8]:
                    upper = tok.upper()
                    if upper.startswith(_PHASE_PREFIX_P) or upper.startswith(_PHASE_PREFIX_S):
                        phase_tok = upper
                        break
                if phase_tok is None:
                    continue

                rec = out.setdefault(
                    key,
                    {
                        "dist_deg": float(dist_deg),
                        "azimuth_deg": float(evaz),
                        "has_p": False,
                        "has_s": False,
                    },
                )
                rec["dist_deg"] = float(dist_deg)
                rec["azimuth_deg"] = float(evaz)
                if phase_tok.startswith(_PHASE_PREFIX_P):
                    rec["has_p"] = True
                if phase_tok.startswith(_PHASE_PREFIX_S):
                    rec["has_s"] = True
    return out


def _velocity_from_1d_model(depth_km: float, model_cfg: Any) -> tuple[float, float]:
    """
    Returns (vp_km_s, vs_km_s) at a given hypocentral depth.

    Supported config formats:
    1) Layered list (legacy):
       [{"z_top_km":.., "z_bot_km":.., "vp_km_s":.., "vs_km_s":..}, ...]
    2) Knot profile (new):
       {
         "Depths": [...],   # signed depth knots
         "Vp": [...],       # velocity knots
         "Vs": [...],
         "depth_unit": "m"|"km"   (optional, inferred if absent),
         "velocity_unit": "m/s"|"km/s" (optional, inferred if absent)
       }
    """
    z = float(depth_km)

    # Format-2: knot profile
    if isinstance(model_cfg, Mapping) and {"Depths", "Vp", "Vs"}.issubset(model_cfg.keys()):
        depths = [float(v) for v in model_cfg["Depths"]]
        vp_vals = [float(v) for v in model_cfg["Vp"]]
        vs_vals = [float(v) for v in model_cfg["Vs"]]
        if not (len(depths) == len(vp_vals) == len(vs_vals)) or len(depths) == 0:
            raise ValueError("velocity_model_1d knot profile has invalid lengths for Depths/Vp/Vs.")

        depth_unit = str(model_cfg.get("depth_unit", "")).lower().strip()
        vel_unit = str(model_cfg.get("velocity_unit", "")).lower().strip()
        if not depth_unit:
            depth_unit = "m" if max(abs(d) for d in depths) > 1000.0 else "km"
        if not vel_unit:
            vel_unit = "m/s" if max(max(vp_vals), max(vs_vals)) > 100.0 else "km/s"

        if depth_unit in {"m", "meter", "meters"}:
            depth_nodes_km = [abs(d) / 1000.0 for d in depths]
        elif depth_unit in {"km", "kilometer", "kilometers"}:
            depth_nodes_km = [abs(d) for d in depths]
        else:
            raise ValueError(f"Unsupported depth_unit in velocity_model_1d: {depth_unit}")

        if vel_unit in {"m/s", "mps"}:
            vp_nodes = [v / 1000.0 for v in vp_vals]
            vs_nodes = [v / 1000.0 for v in vs_vals]
        elif vel_unit in {"km/s", "kmps"}:
            vp_nodes = list(vp_vals)
            vs_nodes = list(vs_vals)
        else:
            raise ValueError(f"Unsupported velocity_unit in velocity_model_1d: {vel_unit}")

        # Sort by increasing depth and interpolate with endpoint clamping.
        order = np.argsort(np.asarray(depth_nodes_km, dtype=np.float64))
        d = np.asarray([depth_nodes_km[i] for i in order], dtype=np.float64)
        vp = np.asarray([vp_nodes[i] for i in order], dtype=np.float64)
        vs = np.asarray([vs_nodes[i] for i in order], dtype=np.float64)
        z_clamped = float(np.clip(z, d[0], d[-1]))
        vp_km_s = float(np.interp(z_clamped, d, vp))
        vs_km_s = float(np.interp(z_clamped, d, vs))
        return vp_km_s, vs_km_s

    # Format-1: layered profile
    layers = list(model_cfg) if model_cfg is not None else []
    if not layers:
        return 6.0, 3.5
    for layer in layers:
        z_top = float(layer["z_top_km"])
        z_bot = float(layer["z_bot_km"])
        if z >= z_top and z < z_bot:
            return float(layer["vp_km_s"]), float(layer["vs_km_s"])
    if z < float(layers[0]["z_top_km"]):
        return float(layers[0]["vp_km_s"]), float(layers[0]["vs_km_s"])
    last = layers[-1]
    return float(last["vp_km_s"]), float(last["vs_km_s"])


def _preferred_z_trace_from_stream(stream: Any) -> Any:
    candidates = [tr for tr in stream if str(tr.stats.channel).upper().endswith("Z")]
    if not candidates:
        raise ValueError("No Z component trace found in stream.")
    priority = {"HH": 0, "HN": 1, "BH": 2, "EH": 3}
    candidates.sort(key=lambda tr: priority.get(str(tr.stats.channel)[:2], 99))
    return candidates[0]


def _read_starttime_seconds_from_origin(file_path: str | Path, event_origin_iso: str) -> float:
    _require_obspy()
    stream = read(str(file_path), headonly=True)
    z_trace = _preferred_z_trace_from_stream(stream)
    start_dt = _to_utc_naive(z_trace.stats.starttime.datetime)
    origin_dt = _to_utc_naive(datetime.fromisoformat(event_origin_iso))
    return float((start_dt - origin_dt).total_seconds())


def _read_z_waveform(file_path: str | Path, target_fs_hz: float) -> np.ndarray:
    _require_obspy()
    stream = read(str(file_path))
    stream.merge(fill_value=0)
    z_trace = _preferred_z_trace_from_stream(stream)
    fs = float(z_trace.stats.sampling_rate)
    x = np.asarray(z_trace.data, dtype=np.float32)
    if fs <= 0:
        raise ValueError(f"Invalid sampling rate {fs} in {file_path}")
    if abs(fs - target_fs_hz) > 1e-3:
        # Rare fallback for inconsistent files.
        up = int(round(target_fs_hz))
        down = int(round(fs))
        x = signal.resample_poly(x, up, down).astype(np.float32)
    return x


def _compute_complex_stft(
    x: np.ndarray,
    fs_hz: float,
    n_fft: int,
    win_length: int,
    hop_length: int,
    drop_nyquist: bool,
    target_freq_bins: int,
    target_time_frames: int,
) -> np.ndarray:
    noverlap = int(win_length - hop_length)
    _, _, zxx = signal.stft(
        x,
        fs=fs_hz,
        nperseg=win_length,
        noverlap=noverlap,
        nfft=n_fft,
        return_onesided=True,
        boundary="zeros",
        padded=True,
        window="hann",
    )

    if drop_nyquist and zxx.shape[0] > 0:
        zxx = zxx[:-1, :]

    # Enforce fixed frequency axis.
    if zxx.shape[0] < target_freq_bins:
        pad_f = target_freq_bins - zxx.shape[0]
        zxx = np.pad(zxx, ((0, pad_f), (0, 0)), mode="constant")
    elif zxx.shape[0] > target_freq_bins:
        zxx = zxx[:target_freq_bins, :]

    # Right-pad / right-crop in time (C2).
    if zxx.shape[1] < target_time_frames:
        pad_t = target_time_frames - zxx.shape[1]
        zxx = np.pad(zxx, ((0, 0), (0, pad_t)), mode="constant")
    elif zxx.shape[1] > target_time_frames:
        zxx = zxx[:, :target_time_frames]

    return zxx.astype(np.complex64, copy=False)


def _event_origin_dt(event_row: Mapping[str, Any]) -> Optional[datetime]:
    iso = event_row.get("origin_time")
    if iso is None:
        return None
    try:
        return _to_utc_naive(datetime.fromisoformat(str(iso)))
    except Exception:
        return None


def _sample_record_from_file(
    file_path: str | Path,
    event_row: Mapping[str, Any],
    phase_row: Mapping[str, Any],
    velocity_model: Any,
    start_offset_s: float,
) -> Dict[str, Any]:
    event_id, station = _extract_event_station(file_path)
    origin_iso = event_row.get("origin_time")
    if origin_iso is None:
        raise ValueError("missing_origin_time")

    magnitude = float(event_row["magnitude"])
    depth_km = float(event_row["depth_km"])
    dist_deg = float(phase_row["dist_deg"])
    azimuth_deg = float(phase_row["azimuth_deg"])

    repi_km = dist_deg * EARTH_KM_PER_DEG
    vp, vs = _velocity_from_1d_model(depth_km=depth_km, model_cfg=velocity_model)
    rhyp_km = math.sqrt(max(repi_km, 0.0) ** 2 + max(depth_km, 0.0) ** 2)

    tP_origin_s = rhyp_km / max(vp, 1e-6)
    tS_origin_s = rhyp_km / max(vs, 1e-6)
    tP_ref_s = tP_origin_s - start_offset_s
    tS_ref_s = tS_origin_s - start_offset_s

    return {
        "file_path": str(file_path),
        "event_id": event_id,
        "station_code": station,
        "magnitude": magnitude,
        "depth_km": depth_km,
        "repi_km": repi_km,
        "azimuth_deg": azimuth_deg,
        "azimuth_sin": math.sin(math.radians(azimuth_deg)),
        "azimuth_cos": math.cos(math.radians(azimuth_deg)),
        "tP_ref_s": tP_ref_s,
        "tS_ref_s": tS_ref_s,
        "dtPS_ref_s": tS_ref_s - tP_ref_s,
        "origin_time": str(origin_iso),
        "origin_date": str(event_row.get("origin_date", "")),
        "origin_clock": str(event_row.get("origin_clock", "")),
        "latitude": float(event_row.get("latitude", float("nan"))),
        "longitude": float(event_row.get("longitude", float("nan"))),
        "phase_has_p": bool(phase_row.get("has_p", False)),
        "phase_has_s": bool(phase_row.get("has_s", False)),
    }


def build_or_load_manifest(cfg: Mapping[str, Any], force_rebuild: bool = False) -> List[Dict[str, Any]]:
    manifest_path = Path(cfg["artifacts"]["manifest_file"])
    meta_path = manifest_path.with_suffix(manifest_path.suffix + ".meta.json")
    drop_report_path = manifest_path.with_suffix(manifest_path.suffix + ".drop_report.json")
    cfg_max_manifest = cfg["data"].get("max_manifest_files")
    velocity_model_cfg = cfg["conditions"]["velocity_model_1d"]
    velocity_model_sha = _object_sha256(velocity_model_cfg)
    excluded_pairs_cfg = cfg["data"].get("manifest_exclude_event_station_pairs", [])
    excluded_pairs = {str(x) for x in excluded_pairs_cfg}
    excluded_pairs_sha = _object_sha256(sorted(excluded_pairs))
    fail_on_manifest_drop = bool(cfg["data"].get("fail_on_manifest_drop", False))
    max_manifest_drop_rate = float(cfg["data"].get("max_manifest_drop_rate", 1.0))
    max_drop_examples_per_reason = int(cfg["data"].get("max_drop_examples_per_reason", 20))
    if manifest_path.exists() and not force_rebuild:
        if meta_path.exists():
            meta = load_json(meta_path)
            same_limit = meta.get("max_manifest_files") == cfg_max_manifest
            same_waveform_dir = str(meta.get("waveform_dir", "")) == str(cfg["data"]["waveform_dir"])
            same_event_catalog = str(meta.get("event_catalog", "")) == str(cfg["data"]["event_catalog"])
            same_phase_pick_dir = str(meta.get("phase_pick_dir", "")) == str(cfg["data"]["phase_pick_dir"])
            same_station_list = str(meta.get("station_list_file", "")) == str(cfg["data"]["station_list_file"])
            same_velocity_model = str(meta.get("velocity_model_sha256", "")) == velocity_model_sha
            same_exclusions = str(meta.get("excluded_pairs_sha256", "")) == excluded_pairs_sha
            same_fail_policy = bool(meta.get("fail_on_manifest_drop", False)) == fail_on_manifest_drop
            same_max_drop_rate = float(meta.get("max_manifest_drop_rate", -1.0)) == max_manifest_drop_rate
            if (
                same_limit
                and same_waveform_dir
                and same_event_catalog
                and same_phase_pick_dir
                and same_station_list
                and same_velocity_model
                and same_exclusions
                and same_fail_policy
                and same_max_drop_rate
            ):
                current_hash = _file_sha256(manifest_path)
                if str(meta.get("manifest_sha256", "")) == current_hash:
                    return load_jsonl(manifest_path)

    waveform_dir = Path(cfg["data"]["waveform_dir"])
    event_catalog = load_event_catalog(cfg["data"]["event_catalog"])
    velocity_model = velocity_model_cfg

    station_list = load_json(cfg["data"]["station_list_file"])
    station_to_idx = {sta: idx for idx, sta in enumerate(station_list)}

    files = sorted(waveform_dir.glob("*.mseed"))
    max_manifest_files = cfg["data"].get("max_manifest_files")
    if max_manifest_files is not None:
        files = files[: int(max_manifest_files)]
    pairs = {_extract_event_station(fp): True for fp in files}
    phase_map = _parse_phase_picks_targeted(cfg["data"]["phase_pick_dir"], set(pairs.keys()))

    rows: List[Dict[str, Any]] = []
    drop_counts: Counter[str] = Counter()
    drop_examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def add_drop(reason: str, fp: str | Path, event_id: str, station: str, detail: str = "") -> None:
        drop_counts[reason] += 1
        ex = drop_examples[reason]
        if len(ex) < max_drop_examples_per_reason:
            ex.append(
                {
                    "file_path": str(fp),
                    "event_id": str(event_id),
                    "station_code": str(station),
                    "detail": str(detail)[:240],
                }
            )

    def has_nonfinite(sample: Mapping[str, Any]) -> bool:
        keys = ("magnitude", "depth_km", "repi_km", "azimuth_sin", "azimuth_cos", "tP_ref_s", "tS_ref_s", "dtPS_ref_s")
        for k in keys:
            v = float(sample[k])
            if not math.isfinite(v):
                return True
        return False

    for fp in files:
        event_id, station = _extract_event_station(fp)
        pair_key = f"{event_id}_{station}"
        if pair_key in excluded_pairs:
            add_drop("excluded_event_station", fp, event_id, station, detail="excluded by config")
            continue

        event_row = event_catalog.get(event_id)
        phase_row = phase_map.get((event_id, station))
        if event_row is None:
            add_drop("missing_event_catalog", fp, event_id, station)
            continue
        if phase_row is None:
            add_drop("missing_phase_pick", fp, event_id, station)
            continue
        if event_row.get("origin_time") is None:
            add_drop("missing_origin_time", fp, event_id, station)
            continue
        if station not in station_to_idx:
            add_drop("unknown_station", fp, event_id, station)
            continue

        try:
            start_offset_s = _read_starttime_seconds_from_origin(fp, str(event_row["origin_time"]))
        except Exception as exc:
            add_drop("starttime_read_error", fp, event_id, station, detail=str(exc))
            continue

        try:
            sample = _sample_record_from_file(
                file_path=fp,
                event_row=event_row,
                phase_row=phase_row,
                velocity_model=velocity_model,
                start_offset_s=float(start_offset_s),
            )
        except Exception as exc:
            msg = str(exc)
            reason = "velocity_model_error" if "velocity_model_1d" in msg else "sample_build_failed"
            add_drop(reason, fp, event_id, station, detail=msg)
            continue

        if has_nonfinite(sample):
            add_drop("nonfinite_feature", fp, event_id, station)
            continue

        sample["station_idx"] = int(station_to_idx[station])
        rows.append(sample)

    num_scanned = len(files)
    num_kept = len(rows)
    num_dropped_total = num_scanned - num_kept
    num_excluded = int(drop_counts.get("excluded_event_station", 0))
    num_dropped_nonexcluded = max(0, num_dropped_total - num_excluded)
    drop_rate_total = (num_dropped_total / num_scanned) if num_scanned > 0 else 0.0
    drop_rate_nonexcluded = (num_dropped_nonexcluded / num_scanned) if num_scanned > 0 else 0.0

    drop_payload = {
        "num_files_scanned": num_scanned,
        "num_rows_kept": num_kept,
        "num_rows_dropped_total": num_dropped_total,
        "num_rows_dropped_nonexcluded": num_dropped_nonexcluded,
        "drop_rate_total": float(drop_rate_total),
        "drop_rate_nonexcluded": float(drop_rate_nonexcluded),
        "drop_counts_by_reason": dict(drop_counts),
        "drop_examples_by_reason": dict(drop_examples),
        "excluded_pairs_count": len(excluded_pairs),
        "excluded_row_count": num_excluded,
        "excluded_pairs_sha256": excluded_pairs_sha,
    }
    save_json(drop_report_path, drop_payload)

    if fail_on_manifest_drop and num_dropped_nonexcluded > 0:
        raise ValueError(
            "Manifest build failed: dropped rows detected while fail_on_manifest_drop=true. "
            f"dropped_nonexcluded={num_dropped_nonexcluded}, drop_rate_nonexcluded={drop_rate_nonexcluded:.6f}, "
            f"reasons={dict(drop_counts)}; see {drop_report_path}"
        )
    if drop_rate_nonexcluded > max_manifest_drop_rate:
        raise ValueError(
            "Manifest build failed: drop rate exceeded max_manifest_drop_rate. "
            f"drop_rate_nonexcluded={drop_rate_nonexcluded:.6f} > max={max_manifest_drop_rate:.6f}; "
            f"see {drop_report_path}"
        )

    save_jsonl(manifest_path, rows)
    manifest_sha = _file_sha256(manifest_path)
    save_json(
        meta_path,
        {
            "waveform_dir": str(cfg["data"]["waveform_dir"]),
            "event_catalog": str(cfg["data"]["event_catalog"]),
            "phase_pick_dir": str(cfg["data"]["phase_pick_dir"]),
            "station_list_file": str(cfg["data"]["station_list_file"]),
            "max_manifest_files": cfg_max_manifest,
            "num_rows": len(rows),
            "velocity_model_sha256": velocity_model_sha,
            "excluded_pairs_sha256": excluded_pairs_sha,
            "excluded_pairs_count": len(excluded_pairs),
            "fail_on_manifest_drop": fail_on_manifest_drop,
            "max_manifest_drop_rate": max_manifest_drop_rate,
            "num_files_scanned": num_scanned,
            "num_rows_dropped_total": num_dropped_total,
            "num_rows_dropped_nonexcluded": num_dropped_nonexcluded,
            "excluded_row_count": num_excluded,
            "drop_rate_total": float(drop_rate_total),
            "drop_rate_nonexcluded": float(drop_rate_nonexcluded),
            "drop_counts_by_reason": dict(drop_counts),
            "manifest_sha256": manifest_sha,
        },
    )
    return rows


def _split_events(
    manifest_rows: Sequence[Mapping[str, Any]],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    ood_event_ratio: float,
    ood_policy: str,
    min_events_per_split: int,
    max_missing_origin_events: int,
    max_unassigned_samples: int,
) -> Dict[str, Any]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio must be 1.0")
    if ood_policy != "latest_by_origin_time":
        raise ValueError(f"Unsupported ood_policy: {ood_policy}")
    if ood_event_ratio < 0.0 or ood_event_ratio > 1.0:
        raise ValueError("ood_event_ratio must be in [0, 1]")

    events_with_origin: Dict[str, datetime] = {}
    all_events = sorted({str(row["event_id"]) for row in manifest_rows})
    for row in manifest_rows:
        evt = str(row["event_id"])
        if evt in events_with_origin:
            continue
        origin = _event_origin_dt(row)
        if origin is not None:
            events_with_origin[evt] = origin

    missing_origin_events = sorted(set(all_events) - set(events_with_origin.keys()))
    if len(missing_origin_events) > int(max_missing_origin_events):
        raise ValueError(
            "Too many events without origin time for event-wise split: "
            f"{len(missing_origin_events)} > {int(max_missing_origin_events)}"
        )

    sorted_events = sorted(events_with_origin.items(), key=lambda kv: kv[1])
    event_ids_sorted = [eid for eid, _ in sorted_events]
    n_total = len(event_ids_sorted)
    if n_total <= 0:
        raise ValueError("No events with valid origin_time found for split.")

    if float(ood_event_ratio) == 0.0:
        n_ood = 0
    else:
        n_ood = max(1, int(round(n_total * float(ood_event_ratio))))
        if n_total > 1:
            n_ood = min(n_ood, n_total - 1)
    if n_ood == 0:
        ood_events = set()
        in_domain_events = list(event_ids_sorted)
    else:
        ood_events = set(event_ids_sorted[-n_ood:])
        in_domain_events = event_ids_sorted[:-n_ood]
    n_in = len(in_domain_events)

    min_events_per_split = int(min_events_per_split)
    if n_in < 3 * min_events_per_split:
        raise ValueError(
            "Not enough in-domain events after OOD split to satisfy non-empty train/val/test: "
            f"n_in={n_in}, required>={3 * min_events_per_split}"
        )

    rng = np.random.default_rng(int(seed))
    shuffled = list(in_domain_events)
    rng.shuffle(shuffled)

    n_train = int(n_in * train_ratio)
    n_val = int(n_in * val_ratio)
    n_test = n_in - n_train - n_val

    counts = [n_train, n_val, n_test]
    mins = [min_events_per_split, min_events_per_split, min_events_per_split]
    for i in range(3):
        while counts[i] < mins[i]:
            donor = max(range(3), key=lambda j: counts[j] - mins[j])
            if counts[donor] <= mins[donor]:
                raise ValueError(
                    "Could not satisfy min_events_per_split for train/val/test with current ratios and OOD split."
                )
            counts[donor] -= 1
            counts[i] += 1
    n_train, n_val, n_test = counts

    train_events = set(shuffled[:n_train])
    val_events = set(shuffled[n_train : n_train + n_val])
    test_events = set(shuffled[n_train + n_val : n_train + n_val + n_test])

    split = {
        "train": {"indices": [], "events": sorted(train_events)},
        "val": {"indices": [], "events": sorted(val_events)},
        "test": {"indices": [], "events": sorted(test_events)},
        "ood": {"indices": [], "events": sorted(ood_events)},
    }

    unassigned_indices: List[int] = []
    for i, row in enumerate(manifest_rows):
        evt = str(row["event_id"])
        if evt in train_events:
            split["train"]["indices"].append(i)
        elif evt in val_events:
            split["val"]["indices"].append(i)
        elif evt in test_events:
            split["test"]["indices"].append(i)
        elif evt in ood_events:
            split["ood"]["indices"].append(i)
        else:
            unassigned_indices.append(i)

    if len(unassigned_indices) > int(max_unassigned_samples):
        raise ValueError(
            "Too many unassigned samples after split (likely due missing origin_time events): "
            f"{len(unassigned_indices)} > {int(max_unassigned_samples)}"
        )

    for key in split:
        split[key]["num_samples"] = len(split[key]["indices"])
        split[key]["num_events"] = len(split[key]["events"])
    split["_diagnostics"] = {
        "all_event_count": len(all_events),
        "events_with_origin_count": len(events_with_origin),
        "missing_origin_event_count": len(missing_origin_events),
        "missing_origin_events_preview": missing_origin_events[:20],
        "unassigned_sample_count": len(unassigned_indices),
        "unassigned_sample_indices_preview": unassigned_indices[:20],
        "ood_policy": ood_policy,
        "ood_event_ratio": float(ood_event_ratio),
        "min_events_per_split": int(min_events_per_split),
    }
    return split


def build_or_load_frozen_splits(
    cfg: Mapping[str, Any],
    manifest_rows: Sequence[Mapping[str, Any]],
    manifest_sha256: str,
    force_rebuild: bool = False,
) -> Dict[str, Any]:
    split_file = Path(cfg["artifacts"]["frozen_split_file"])
    split_cfg = cfg["split"]
    ood_policy = str(split_cfg.get("ood_policy", "latest_by_origin_time"))
    min_events_per_split = int(split_cfg.get("min_events_per_split", 1))
    max_missing_origin_events = int(split_cfg.get("max_missing_origin_events", 0))
    max_unassigned_samples = int(split_cfg.get("max_unassigned_samples", 0))

    if split_file.exists() and not force_rebuild:
        payload = load_json(split_file)
        meta = payload.get("_meta", {})
        if (
            int(meta.get("manifest_num_rows", -1)) == len(manifest_rows)
            and str(meta.get("manifest_sha256", "")) == str(manifest_sha256)
            and int(meta.get("seed", -1)) == int(cfg["experiment"]["seed"])
            and float(meta.get("train_ratio", -1.0)) == float(split_cfg["train_ratio"])
            and float(meta.get("val_ratio", -1.0)) == float(split_cfg["val_ratio"])
            and float(meta.get("test_ratio", -1.0)) == float(split_cfg["test_ratio"])
            and float(meta.get("ood_event_ratio", -1.0)) == float(split_cfg["ood_event_ratio"])
            and str(meta.get("ood_policy", "")) == ood_policy
            and int(meta.get("min_events_per_split", -1)) == min_events_per_split
            and int(meta.get("max_missing_origin_events", -1)) == max_missing_origin_events
            and int(meta.get("max_unassigned_samples", -1)) == max_unassigned_samples
        ):
            return payload

    split = _split_events(
        manifest_rows=manifest_rows,
        seed=int(cfg["experiment"]["seed"]),
        train_ratio=float(cfg["split"]["train_ratio"]),
        val_ratio=float(cfg["split"]["val_ratio"]),
        test_ratio=float(cfg["split"]["test_ratio"]),
        ood_event_ratio=float(cfg["split"]["ood_event_ratio"]),
        ood_policy=ood_policy,
        min_events_per_split=min_events_per_split,
        max_missing_origin_events=max_missing_origin_events,
        max_unassigned_samples=max_unassigned_samples,
    )
    split["_meta"] = {
        "manifest_num_rows": len(manifest_rows),
        "manifest_sha256": manifest_sha256,
        "seed": int(cfg["experiment"]["seed"]),
        "train_ratio": float(split_cfg["train_ratio"]),
        "val_ratio": float(split_cfg["val_ratio"]),
        "test_ratio": float(split_cfg["test_ratio"]),
        "ood_event_ratio": float(split_cfg["ood_event_ratio"]),
        "ood_policy": ood_policy,
        "min_events_per_split": min_events_per_split,
        "max_missing_origin_events": max_missing_origin_events,
        "max_unassigned_samples": max_unassigned_samples,
    }
    save_json(split_file, split)
    return split


def _iter_stft_energy_terms(
    cfg: Mapping[str, Any],
    manifest_rows: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    max_samples: Optional[int],
) -> Iterable[tuple[float, int]]:
    limit = len(indices) if max_samples is None else min(len(indices), int(max_samples))
    stft_cfg = cfg["stft"]
    fs = float(cfg["data"]["sampling_rate_hz"])
    for idx in indices[:limit]:
        row = manifest_rows[idx]
        x = _read_z_waveform(row["file_path"], fs)
        zxx = _compute_complex_stft(
            x=x,
            fs_hz=fs,
            n_fft=int(stft_cfg["n_fft"]),
            win_length=int(stft_cfg["win_length"]),
            hop_length=int(stft_cfg["hop_length"]),
            drop_nyquist=bool(stft_cfg["drop_nyquist"]),
            target_freq_bins=int(stft_cfg["target_freq_bins"]),
            target_time_frames=int(stft_cfg["target_time_frames"]),
        )
        energy_sum = float(np.sum(np.square(np.real(zxx)) + np.square(np.imag(zxx))))
        yield energy_sum, int(zxx.size)


def build_or_load_normalization_stats(
    cfg: Mapping[str, Any],
    manifest_rows: Sequence[Mapping[str, Any]],
    split_payload: Mapping[str, Any],
    manifest_sha256: str,
    force_rebuild: bool = False,
) -> Dict[str, Any]:
    stats_file = Path(cfg["artifacts"]["normalization_stats_file"])
    stft_cfg = cfg["stft"]
    stft_signature = {
        "n_fft": int(stft_cfg["n_fft"]),
        "win_length": int(stft_cfg["win_length"]),
        "hop_length": int(stft_cfg["hop_length"]),
        "drop_nyquist": bool(stft_cfg["drop_nyquist"]),
        "target_freq_bins": int(stft_cfg["target_freq_bins"]),
        "target_time_frames": int(stft_cfg["target_time_frames"]),
    }
    feature_order_cfg = list(cfg["conditions"]["numeric_feature_order"])
    max_samples_cfg = cfg["normalization"].get("max_samples_for_stft_rms")

    if stats_file.exists() and not force_rebuild:
        payload = load_json(stats_file)
        meta = payload.get("_meta", {})
        if (
            int(meta.get("manifest_num_rows", -1)) == len(manifest_rows)
            and str(meta.get("manifest_sha256", "")) == str(manifest_sha256)
            and payload.get("feature_order", []) == feature_order_cfg
            and meta.get("stft_signature", {}) == stft_signature
            and meta.get("max_samples_for_stft_rms", None) == max_samples_cfg
        ):
            return payload

    train_indices = list(split_payload["train"]["indices"])
    feat_order = feature_order_cfg
    feats = np.asarray(
        [[float(manifest_rows[i][k]) for k in feat_order] for i in train_indices],
        dtype=np.float64,
    )
    means = feats.mean(axis=0)
    stds = feats.std(axis=0)
    stds = np.where(stds < 1e-8, 1.0, stds)

    max_samples = cfg["normalization"].get("max_samples_for_stft_rms")
    total_energy = 0.0
    total_count = 0
    for e_sum, n in _iter_stft_energy_terms(
        cfg=cfg,
        manifest_rows=manifest_rows,
        indices=train_indices,
        max_samples=max_samples,
    ):
        total_energy += e_sum
        total_count += n
    stft_rms = math.sqrt(total_energy / max(total_count, 1))

    payload = {
        "feature_order": feat_order,
        "feature_mean": means.tolist(),
        "feature_std": stds.tolist(),
        "stft_global_rms": float(stft_rms),
        "num_train_samples": len(train_indices),
        "num_stft_rms_samples": len(train_indices) if max_samples is None else min(len(train_indices), int(max_samples)),
        "_meta": {
            "manifest_num_rows": len(manifest_rows),
            "manifest_sha256": manifest_sha256,
            "stft_signature": stft_signature,
            "max_samples_for_stft_rms": max_samples_cfg,
        },
    }
    save_json(stats_file, payload)
    return payload


def prepare_exp001_artifacts(
    cfg: Mapping[str, Any],
    force_rebuild_manifest: bool = False,
    force_rebuild_split: bool = False,
    force_rebuild_stats: bool = False,
) -> tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    ensure_dir(Path(cfg["artifacts"]["protocol_dir"]))
    manifest = build_or_load_manifest(cfg, force_rebuild=force_rebuild_manifest)
    manifest_sha = _file_sha256(Path(cfg["artifacts"]["manifest_file"]))
    split = build_or_load_frozen_splits(
        cfg,
        manifest,
        manifest_sha256=manifest_sha,
        force_rebuild=force_rebuild_split,
    )
    stats = build_or_load_normalization_stats(
        cfg,
        manifest,
        split,
        manifest_sha256=manifest_sha,
        force_rebuild=force_rebuild_stats,
    )
    return manifest, split, stats


class ExternalHHComplexSTFTDataset(Dataset):
    """Z-only complex STFT dataset with frozen feature normalization."""

    def __init__(
        self,
        cfg: Mapping[str, Any],
        manifest_rows: Sequence[Mapping[str, Any]],
        indices: Sequence[int],
        norm_stats: Mapping[str, Any],
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.rows = [dict(manifest_rows[i]) for i in indices]
        self.indices = list(indices)
        self.fs_hz = float(cfg["data"]["sampling_rate_hz"])
        self.stft_cfg = cfg["stft"]

        self.feature_order = list(norm_stats["feature_order"])
        self.feature_mean = np.asarray(norm_stats["feature_mean"], dtype=np.float32)
        self.feature_std = np.asarray(norm_stats["feature_std"], dtype=np.float32)
        self.stft_rms = float(norm_stats["stft_global_rms"])
        self.eps = float(cfg["normalization"]["eps"])

    def __len__(self) -> int:
        return len(self.rows)

    def _normalize_cond(self, raw: np.ndarray) -> np.ndarray:
        return (raw - self.feature_mean) / (self.feature_std + self.eps)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        x = _read_z_waveform(row["file_path"], self.fs_hz)
        zxx = _compute_complex_stft(
            x=x,
            fs_hz=self.fs_hz,
            n_fft=int(self.stft_cfg["n_fft"]),
            win_length=int(self.stft_cfg["win_length"]),
            hop_length=int(self.stft_cfg["hop_length"]),
            drop_nyquist=bool(self.stft_cfg["drop_nyquist"]),
            target_freq_bins=int(self.stft_cfg["target_freq_bins"]),
            target_time_frames=int(self.stft_cfg["target_time_frames"]),
        )

        r = np.real(zxx) / (self.stft_rms + self.eps)
        i = np.imag(zxx) / (self.stft_rms + self.eps)
        x_tensor = torch.from_numpy(np.stack([r, i], axis=0).astype(np.float32))

        cond_raw = np.asarray([float(row[k]) for k in self.feature_order], dtype=np.float32)
        cond_norm = self._normalize_cond(cond_raw)

        return {
            "x": x_tensor,
            "cond": torch.from_numpy(cond_norm.astype(np.float32)),
            "cond_raw": torch.from_numpy(cond_raw.astype(np.float32)),
            "station_idx": torch.tensor(int(row["station_idx"]), dtype=torch.long),
            "meta": {
                "event_id": row["event_id"],
                "station_code": row["station_code"],
                "file_path": row["file_path"],
                "magnitude": float(row["magnitude"]),
            },
        }


def collate_exp001(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "x": torch.stack([b["x"] for b in batch], dim=0),
        "cond": torch.stack([b["cond"] for b in batch], dim=0),
        "cond_raw": torch.stack([b["cond_raw"] for b in batch], dim=0),
        "station_idx": torch.stack([b["station_idx"] for b in batch], dim=0),
        "meta": [b["meta"] for b in batch],
    }


def make_magnitude_bin_indices(rows: Sequence[Mapping[str, Any]]) -> Dict[str, List[int]]:
    out = {"lt3": [], "m3to5": [], "ge5": []}
    for idx, row in enumerate(rows):
        mag = float(row["magnitude"])
        if mag < 3.0:
            out["lt3"].append(idx)
        elif mag < 5.0:
            out["m3to5"].append(idx)
        else:
            out["ge5"].append(idx)
    return out


def make_weighted_sampler_weights(
    rows: Sequence[Mapping[str, Any]],
    alpha: float,
    w_max: float,
) -> np.ndarray:
    bins = make_magnitude_bin_indices(rows)
    n_lt3 = max(1, len(bins["lt3"]))
    n_mid = max(1, len(bins["m3to5"]))
    n_ge5 = max(1, len(bins["ge5"]))
    n_med = float(np.median([n_lt3, n_mid, n_ge5]))

    w_lt3 = min((n_med / n_lt3) ** float(alpha), float(w_max))
    w_mid = min((n_med / n_mid) ** float(alpha), float(w_max))
    w_ge5 = min((n_med / n_ge5) ** float(alpha), float(w_max))

    weights = np.zeros(len(rows), dtype=np.float32)
    for i in bins["lt3"]:
        weights[i] = w_lt3
    for i in bins["m3to5"]:
        weights[i] = w_mid
    for i in bins["ge5"]:
        weights[i] = w_ge5
    return weights
