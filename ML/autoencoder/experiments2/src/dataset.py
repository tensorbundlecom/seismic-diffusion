"""Dataset and data-preparation utilities for experiments2/exp001."""

from __future__ import annotations

import csv
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy import signal
from torch.utils.data import Dataset

from .utils import ensure_dir, load_json, load_jsonl, save_json, save_jsonl

try:
    from obspy import read  # type: ignore
except Exception as exc:  # pragma: no cover - import-guard for environments without obspy
    read = None
    _OBSPY_IMPORT_ERROR = exc
else:
    _OBSPY_IMPORT_ERROR = None


EARTH_KM_PER_DEG = 111.19
_PHASE_PREFIX_P = ("P",)
_PHASE_PREFIX_S = ("S",)


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
        return datetime.strptime(f"{date_s} {time_s}", "%Y.%m.%d %H:%M:%S.%f")
    except ValueError:
        # Fallback if milliseconds are absent.
        try:
            return datetime.strptime(f"{date_s} {time_s.split('.')[0]}", "%Y.%m.%d %H:%M:%S")
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


def _velocity_from_1d_model(depth_km: float, layers: Sequence[Mapping[str, Any]]) -> tuple[float, float]:
    z = float(depth_km)
    if not layers:
        return 6.0, 3.5
    for layer in layers:
        z_top = float(layer["z_top_km"])
        z_bot = float(layer["z_bot_km"])
        if z >= z_top and z < z_bot:
            return float(layer["vp_km_s"]), float(layer["vs_km_s"])
    # Clamp to nearest boundary.
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
    start_dt = z_trace.stats.starttime.datetime
    origin_dt = datetime.fromisoformat(event_origin_iso)
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
        return datetime.fromisoformat(str(iso))
    except Exception:
        return None


def _sample_record_from_file(
    file_path: str | Path,
    event_row: Mapping[str, Any],
    phase_row: Mapping[str, Any],
    velocity_layers: Sequence[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    event_id, station = _extract_event_station(file_path)
    origin_iso = event_row.get("origin_time")
    if origin_iso is None:
        return None

    try:
        start_offset_s = _read_starttime_seconds_from_origin(file_path, str(origin_iso))
    except Exception:
        return None

    magnitude = float(event_row["magnitude"])
    depth_km = float(event_row["depth_km"])
    dist_deg = float(phase_row["dist_deg"])
    azimuth_deg = float(phase_row["azimuth_deg"])

    repi_km = dist_deg * EARTH_KM_PER_DEG
    vp, vs = _velocity_from_1d_model(depth_km=depth_km, layers=velocity_layers)
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
    cfg_max_manifest = cfg["data"].get("max_manifest_files")
    if manifest_path.exists() and not force_rebuild:
        if meta_path.exists():
            meta = load_json(meta_path)
            same_limit = meta.get("max_manifest_files") == cfg_max_manifest
            same_waveform_dir = str(meta.get("waveform_dir", "")) == str(cfg["data"]["waveform_dir"])
            if same_limit and same_waveform_dir:
                return load_jsonl(manifest_path)
        else:
            # Backward-compat: if meta missing, trust existing manifest.
            return load_jsonl(manifest_path)

    waveform_dir = Path(cfg["data"]["waveform_dir"])
    event_catalog = load_event_catalog(cfg["data"]["event_catalog"])
    velocity_layers = cfg["conditions"]["velocity_model_1d"]

    station_list = load_json(cfg["data"]["station_list_file"])
    station_to_idx = {sta: idx for idx, sta in enumerate(station_list)}

    files = sorted(waveform_dir.glob("*.mseed"))
    max_manifest_files = cfg["data"].get("max_manifest_files")
    if max_manifest_files is not None:
        files = files[: int(max_manifest_files)]
    pairs = {_extract_event_station(fp): True for fp in files}
    phase_map = _parse_phase_picks_targeted(cfg["data"]["phase_pick_dir"], set(pairs.keys()))

    rows: List[Dict[str, Any]] = []
    for fp in files:
        event_id, station = _extract_event_station(fp)
        event_row = event_catalog.get(event_id)
        phase_row = phase_map.get((event_id, station))
        if event_row is None or phase_row is None:
            continue
        sample = _sample_record_from_file(
            file_path=fp,
            event_row=event_row,
            phase_row=phase_row,
            velocity_layers=velocity_layers,
        )
        if sample is None:
            continue
        sample["station_idx"] = int(station_to_idx.get(station, 0))
        rows.append(sample)

    save_jsonl(manifest_path, rows)
    save_json(
        meta_path,
        {
            "waveform_dir": str(cfg["data"]["waveform_dir"]),
            "max_manifest_files": cfg_max_manifest,
            "num_rows": len(rows),
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
) -> Dict[str, Any]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio must be 1.0")

    events: Dict[str, datetime] = {}
    for row in manifest_rows:
        evt = str(row["event_id"])
        if evt in events:
            continue
        origin = _event_origin_dt(row)
        if origin is not None:
            events[evt] = origin

    sorted_events = sorted(events.items(), key=lambda kv: kv[1])
    event_ids_sorted = [eid for eid, _ in sorted_events]
    n_total = len(event_ids_sorted)
    n_ood = max(1, int(round(n_total * float(ood_event_ratio))))
    ood_events = set(event_ids_sorted[-n_ood:])
    in_domain_events = event_ids_sorted[:-n_ood]

    rng = np.random.default_rng(int(seed))
    shuffled = list(in_domain_events)
    rng.shuffle(shuffled)

    n_in = len(shuffled)
    n_train = int(n_in * train_ratio)
    n_val = int(n_in * val_ratio)

    train_events = set(shuffled[:n_train])
    val_events = set(shuffled[n_train : n_train + n_val])
    test_events = set(shuffled[n_train + n_val :])

    split = {
        "train": {"indices": [], "events": sorted(train_events)},
        "val": {"indices": [], "events": sorted(val_events)},
        "test": {"indices": [], "events": sorted(test_events)},
        "ood": {"indices": [], "events": sorted(ood_events)},
    }

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

    for key in split:
        split[key]["num_samples"] = len(split[key]["indices"])
        split[key]["num_events"] = len(split[key]["events"])
    return split


def build_or_load_frozen_splits(
    cfg: Mapping[str, Any],
    manifest_rows: Sequence[Mapping[str, Any]],
    force_rebuild: bool = False,
) -> Dict[str, Any]:
    split_file = Path(cfg["artifacts"]["frozen_split_file"])
    if split_file.exists() and not force_rebuild:
        payload = load_json(split_file)
        if int(payload.get("manifest_num_rows", -1)) == len(manifest_rows):
            return payload

    split = _split_events(
        manifest_rows=manifest_rows,
        seed=int(cfg["experiment"]["seed"]),
        train_ratio=float(cfg["split"]["train_ratio"]),
        val_ratio=float(cfg["split"]["val_ratio"]),
        test_ratio=float(cfg["split"]["test_ratio"]),
        ood_event_ratio=float(cfg["split"]["ood_event_ratio"]),
    )
    split["manifest_num_rows"] = len(manifest_rows)
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
    force_rebuild: bool = False,
) -> Dict[str, Any]:
    stats_file = Path(cfg["artifacts"]["normalization_stats_file"])
    if stats_file.exists() and not force_rebuild:
        payload = load_json(stats_file)
        if int(payload.get("manifest_num_rows", -1)) == len(manifest_rows):
            return payload

    train_indices = list(split_payload["train"]["indices"])
    feat_order = list(cfg["conditions"]["numeric_feature_order"])
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
        "manifest_num_rows": len(manifest_rows),
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
    split = build_or_load_frozen_splits(cfg, manifest, force_rebuild=force_rebuild_split)
    stats = build_or_load_normalization_stats(cfg, manifest, split, force_rebuild=force_rebuild_stats)
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
