import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
from obspy.geodetics import gps2dist_azimuth


NUMERIC_FEATURE_NAMES = ["magnitude", "log_repi_km", "depth_km", "sin_az", "cos_az"]
NORM_FEATURE_NAMES = ["magnitude", "log_repi_km", "depth_km"]


def load_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def save_json(path: str, payload: Dict) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)


def _safe_std(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    std = x.std(axis=0)
    std = np.where(std < eps, 1.0, std)
    return std


def fit_normalization_stats(rows: Sequence[Sequence[float]]) -> Dict:
    arr = np.asarray(rows, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != len(NORM_FEATURE_NAMES):
        raise ValueError("rows must have shape [N, 3] for normalization features")

    mean = arr.mean(axis=0)
    std = _safe_std(arr)

    return {
        "feature_names": NORM_FEATURE_NAMES,
        "mean": {k: float(v) for k, v in zip(NORM_FEATURE_NAMES, mean)},
        "std": {k: float(v) for k, v in zip(NORM_FEATURE_NAMES, std)},
    }


def normalize_condition_vector(raw_cond: np.ndarray, stats: Dict) -> np.ndarray:
    """
    raw_cond order:
      [magnitude, log_repi_km, depth_km, sin_az, cos_az]
    """
    out = raw_cond.astype(np.float32).copy()
    if stats is None:
        return out

    for i, name in enumerate(NORM_FEATURE_NAMES):
        m = float(stats["mean"][name])
        s = float(stats["std"][name])
        if abs(s) < 1e-8:
            s = 1.0
        out[i] = (out[i] - m) / s
    return out


def build_geometry_condition(
    magnitude: float,
    event_lat: float,
    event_lon: float,
    event_depth_km: float,
    station_lat: float,
    station_lon: float,
) -> Dict[str, float]:
    dist_m, azimuth_deg, _ = gps2dist_azimuth(event_lat, event_lon, station_lat, station_lon)
    repi_km = float(dist_m) / 1000.0

    az_rad = math.radians(float(azimuth_deg))
    sin_az = math.sin(az_rad)
    cos_az = math.cos(az_rad)

    return {
        "magnitude": float(magnitude),
        "log_repi_km": float(np.log1p(repi_km)),
        "depth_km": float(event_depth_km),
        "sin_az": float(sin_az),
        "cos_az": float(cos_az),
        "repi_km": float(repi_km),
        "azimuth_deg": float(azimuth_deg),
    }


def condition_dict_to_vector(cond: Dict[str, float]) -> np.ndarray:
    return np.asarray([cond[k] for k in NUMERIC_FEATURE_NAMES], dtype=np.float32)
