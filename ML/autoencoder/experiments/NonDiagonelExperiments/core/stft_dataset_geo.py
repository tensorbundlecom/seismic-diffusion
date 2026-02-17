import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import obspy
import pandas as pd
import torch
from obspy import read
from scipy import signal
from torch.utils.data import Dataset

from ML.autoencoder.experiments.NonDiagonel.core.condition_utils import (
    condition_dict_to_vector,
    build_geometry_condition,
    fit_normalization_stats,
    load_json,
    normalize_condition_vector,
    save_json,
)


class SeismicSTFTDatasetGeoCondition(Dataset):
    """
    STFT dataset with geometry-aware conditioning:
      [magnitude, log1p(repi_km), depth_km, sin(azimuth), cos(azimuth)] + station embedding
    """

    def __init__(
        self,
        data_dir: str = "data/filtered_waveforms",
        event_file: str = "data/events/20140101_20251101_0.0_9.0_9_339.txt",
        station_coords_file: str = "",
        channels: List[str] = None,
        nperseg: int = 256,
        noverlap: int = 192,
        nfft: int = 256,
        normalize: bool = True,
        log_scale: bool = True,
        magnitude_col: str = "xM",
        station_list: Optional[List[str]] = None,
        condition_stats_file: Optional[str] = None,
    ):
        if channels is None:
            channels = ["HH", "HN", "EH", "BH"]

        self.data_dir = Path(data_dir)
        self.event_file = Path(event_file)
        self.channels = channels
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nfft
        self.normalize = normalize
        self.log_scale = log_scale
        self.magnitude_col = magnitude_col
        self.station_list_fixed = station_list

        print(f"Loading event catalog from {event_file}...")
        self.event_catalog = self._load_event_catalog()
        self.event_lookup = {
            str(row["event_id"]): row
            for _, row in self.event_catalog.iterrows()
        }
        print(f"Loaded {len(self.event_catalog)} events")

        self.file_paths = []
        for channel in channels:
            channel_dir = self.data_dir / channel
            if channel_dir.exists():
                self.file_paths.extend(sorted(channel_dir.glob("*.mseed")))

        if len(self.file_paths) == 0:
            raise ValueError(f"No mseed files found in {self.data_dir} for channels {channels}")

        print(f"Found {len(self.file_paths)} mseed files")

        self.station_names, self.station_to_idx = self._build_station_mapping()
        print(f"Found {len(self.station_names)} unique stations")

        if not station_coords_file:
            raise ValueError("station_coords_file is required for geometry conditions")
        self.station_coords = load_json(station_coords_file)
        self._check_station_coords_completeness()

        self.condition_stats = None
        if condition_stats_file and os.path.exists(condition_stats_file):
            self.condition_stats = load_json(condition_stats_file)
            print(f"Loaded condition stats from {condition_stats_file}")

        self._warned_no_stats = False

    def _load_event_catalog(self) -> pd.DataFrame:
        for encoding in ["latin1", "windows-1254", "iso-8859-9", "utf-8"]:
            try:
                df = pd.read_csv(self.event_file, sep="\t", encoding=encoding, skipinitialspace=True)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError(f"Could not decode {self.event_file}")

        df.columns = df.columns.str.strip()
        df["event_id"] = df["Deprem Kodu"].astype(str).str.strip()

        mag_columns = ["xM", "MD", "ML", "Mw", "Ms", "Mb"]
        for col in mag_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col] == 0.0, col] = np.nan

        df["latitude"] = pd.to_numeric(df["Enlem"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["Boylam"], errors="coerce")
        df["depth_km"] = pd.to_numeric(df["Der(km)"], errors="coerce")
        return df

    def _build_station_mapping(self) -> Tuple[List[str], Dict[str, int]]:
        if self.station_list_fixed is not None:
            print(f"Using fixed station list with {len(self.station_list_fixed)} stations.")
            station_names = sorted(self.station_list_fixed)
            return station_names, {s: i for i, s in enumerate(station_names)}

        station_names = set()
        for file_path in self.file_paths:
            station_names.add(self._extract_station_from_filename(file_path.name))
        station_names = sorted(list(station_names))
        return station_names, {s: i for i, s in enumerate(station_names)}

    def _check_station_coords_completeness(self) -> None:
        missing = []
        for s in self.station_names:
            c = self.station_coords.get(s)
            if c is None:
                missing.append(s)
                continue
            if any(k not in c for k in ["latitude", "longitude", "elevation_m"]):
                missing.append(s)
        if missing:
            raise ValueError(f"Missing station coordinates for {len(missing)} stations: {missing[:10]}")

    @staticmethod
    def _extract_station_from_filename(filename: str) -> str:
        # Expected patterns:
        # - YYYYMMDDHHMMSS_STATION_HH.mseed
        # - OOD_POST_01_STATION_HH.mseed
        stem = Path(filename).stem
        parts = stem.split("_")
        if len(parts) < 3:
            return "UNKNOWN"
        return parts[-2]

    @staticmethod
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

    def _get_event_info(self, event_id: str) -> Optional[Dict]:
        event = self.event_lookup.get(event_id)
        if event is None:
            return None
        magnitude = event[self.magnitude_col]
        if pd.isna(magnitude):
            for mag_col in ["Mw", "ML", "Ms", "Mb", "MD", "xM"]:
                if not pd.isna(event[mag_col]):
                    magnitude = event[mag_col]
                    break
        if pd.isna(magnitude):
            magnitude = 0.0

        return {
            "magnitude": float(magnitude),
            "latitude": float(event["latitude"]),
            "longitude": float(event["longitude"]),
            "depth_km": float(event["depth_km"]),
            "location_name": event.get("Yer", "UNKNOWN"),
        }

    def _build_condition_vector(self, event_info: Dict, station_name: str) -> Tuple[np.ndarray, Dict]:
        c = self.station_coords[station_name]
        cond = build_geometry_condition(
            magnitude=event_info["magnitude"],
            event_lat=event_info["latitude"],
            event_lon=event_info["longitude"],
            event_depth_km=event_info["depth_km"],
            station_lat=float(c["latitude"]),
            station_lon=float(c["longitude"]),
        )
        raw_vec = condition_dict_to_vector(cond)
        norm_vec = normalize_condition_vector(raw_vec, self.condition_stats)
        return norm_vec, cond

    def fit_condition_stats(self, indices: Optional[Sequence[int]] = None) -> Dict:
        if indices is None:
            indices = range(len(self.file_paths))

        rows = []
        for idx in indices:
            fp = self.file_paths[idx]
            event_id = self._extract_event_id_from_filename(fp.name)
            station_name = self._extract_station_from_filename(fp.name)
            if station_name not in self.station_coords:
                continue

            info = self._get_event_info(event_id)
            if info is None:
                continue

            cond = build_geometry_condition(
                magnitude=info["magnitude"],
                event_lat=info["latitude"],
                event_lon=info["longitude"],
                event_depth_km=info["depth_km"],
                station_lat=float(self.station_coords[station_name]["latitude"]),
                station_lon=float(self.station_coords[station_name]["longitude"]),
            )
            rows.append([cond["magnitude"], cond["log_repi_km"], cond["depth_km"]])

        if not rows:
            raise RuntimeError("No rows available to fit condition stats")

        self.condition_stats = fit_normalization_stats(rows)
        return self.condition_stats

    def save_condition_stats(self, path: str) -> None:
        if self.condition_stats is None:
            raise RuntimeError("condition_stats is not fitted")
        save_json(path, self.condition_stats)

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        file_path = self.file_paths[idx]
        try:
            stream = read(str(file_path))
            stream.merge(fill_value=0)

            component_groups = {"E": [], "N": [], "Z": []}
            for tr in stream:
                comp = tr.stats.channel[-1]
                if comp in component_groups:
                    component_groups[comp].append(tr)

            final_traces = []
            pref = {"HH": 0, "HN": 1, "BH": 2, "EH": 3}
            for comp in ["E", "N", "Z"]:
                comp_traces = component_groups[comp]
                if not comp_traces:
                    continue
                comp_traces.sort(key=lambda x: pref.get(x.stats.channel[:2], 99))
                final_traces.append(comp_traces[0])

            stream = obspy.Stream(traces=final_traces)
            if len(stream) != 3:
                raise ValueError(f"Expected E/N/Z components, found {[tr.stats.channel for tr in stream]}")
            stream.sort(keys=["channel"])

            event_id = self._extract_event_id_from_filename(file_path.name)
            station_name = stream[0].stats.station
            event_info = self._get_event_info(event_id)
            if event_info is None:
                raise ValueError(f"Event {event_id} not found in catalog")

            if station_name not in self.station_to_idx:
                raise ValueError(f"Station {station_name} not found in station mapping")
            if station_name not in self.station_coords:
                raise ValueError(f"Station {station_name} not found in station coordinate cache")

            if self.condition_stats is None and not self._warned_no_stats:
                print("[WARN] condition_stats is None; using raw numeric condition values.")
                self._warned_no_stats = True

            condition_vec, cond_meta = self._build_condition_vector(event_info, station_name)
            station_idx = self.station_to_idx[station_name]

            stft_channels = []
            sample_mag_min = np.inf
            sample_mag_max = -np.inf

            for trace in stream:
                data = trace.data.astype(np.float32)
                _, _, zxx = signal.stft(
                    data,
                    fs=trace.stats.sampling_rate,
                    nperseg=self.nperseg,
                    noverlap=self.noverlap,
                    nfft=self.nfft,
                    return_onesided=True,
                    boundary="zeros",
                    padded=True,
                )

                mag = np.abs(zxx)
                if self.log_scale:
                    mag = np.log1p(mag)

                if self.normalize:
                    cmin = mag.min()
                    cmax = mag.max()
                    sample_mag_min = min(sample_mag_min, cmin)
                    sample_mag_max = max(sample_mag_max, cmax)
                    if cmax > cmin:
                        mag = (mag - cmin) / (cmax - cmin)
                    else:
                        mag = np.zeros_like(mag)

                stft_channels.append(mag)

            min_time = min(c.shape[1] for c in stft_channels)
            stft_channels = [c[:, :min_time] for c in stft_channels]
            spectrogram = np.stack(stft_channels, axis=0)

            spec_tensor = torch.from_numpy(spectrogram).float()
            cond_tensor = torch.from_numpy(condition_vec).float()
            station_tensor = torch.tensor(station_idx, dtype=torch.long)

            metadata = {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "event_id": event_id,
                "station_name": station_name,
                "station_idx": station_idx,
                "magnitude": float(event_info["magnitude"]),
                "latitude": float(event_info["latitude"]),
                "longitude": float(event_info["longitude"]),
                "depth_km": float(event_info["depth_km"]),
                "log_repi_km": float(cond_meta["log_repi_km"]),
                "repi_km": float(cond_meta["repi_km"]),
                "azimuth_deg": float(cond_meta["azimuth_deg"]),
                "mag_min": float(sample_mag_min if np.isfinite(sample_mag_min) else 0.0),
                "mag_max": float(sample_mag_max if np.isfinite(sample_mag_max) else 1.0),
            }
            return spec_tensor, cond_tensor, station_tensor, metadata

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            dummy = torch.zeros((3, self.nfft // 2 + 1, 1), dtype=torch.float32)
            return dummy, torch.zeros(5, dtype=torch.float32), torch.tensor(0), {"error": str(e), "file_path": str(file_path)}


def collate_fn_geo(batch):
    specs = []
    conds = []
    stations = []
    meta = []

    valid = [b for b in batch if "error" not in b[3]]
    if not valid:
        return None, None, None, None

    max_time = max(item[0].shape[2] for item in valid)
    for spec, cond, sta, md in valid:
        if spec.shape[2] < max_time:
            pad = max_time - spec.shape[2]
            spec = torch.nn.functional.pad(spec, (0, pad), mode="constant", value=0)
        specs.append(spec)
        conds.append(cond)
        stations.append(sta)
        meta.append(md)

    return torch.stack(specs, 0), torch.stack(conds, 0), torch.stack(stations, 0), meta
