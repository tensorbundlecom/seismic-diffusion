import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import obspy
import pandas as pd
import torch
from obspy import read
from scipy import signal
from torch.utils.data import Dataset


class SeismicSTFTDatasetWithMetadata(Dataset):
    """
    Local copy of metadata-aware STFT dataset for isolated experiment usage.
    """

    def __init__(
        self,
        data_dir: str,
        event_file: str,
        channels: List[str],
        nperseg: int = 256,
        noverlap: int = 192,
        nfft: int = 256,
        normalize: bool = True,
        log_scale: bool = True,
        return_magnitude: bool = True,
        magnitude_col: str = "xM",
        station_list: Optional[List[str]] = None,
    ):
        self.data_dir = Path(data_dir)
        self.event_file = Path(event_file)
        self.channels = channels
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nfft
        self.normalize = normalize
        self.log_scale = log_scale
        self.return_magnitude = return_magnitude
        self.magnitude_col = magnitude_col
        self.station_list_fixed = station_list

        print(f"Loading event catalog from {self.event_file}...")
        self.event_catalog = self._load_event_catalog()
        print(f"Loaded {len(self.event_catalog)} events")

        self._calculate_location_bounds()

        self.file_paths: List[Path] = []
        for ch in channels:
            ch_dir = self.data_dir / ch
            if ch_dir.exists():
                self.file_paths.extend(sorted(ch_dir.glob("*.mseed")))

        if not self.file_paths:
            raise ValueError(f"No mseed files found in {self.data_dir} for channels {channels}")
        print(f"Found {len(self.file_paths)} mseed files")

        self.station_names, self.station_to_idx = self._build_station_mapping()
        print(f"Found {len(self.station_names)} unique stations")

    def _load_event_catalog(self) -> pd.DataFrame:
        df = None
        for enc in ("latin1", "windows-1254", "iso-8859-9", "utf-8"):
            try:
                df = pd.read_csv(self.event_file, sep="\t", encoding=enc, skipinitialspace=True)
                break
            except UnicodeDecodeError:
                continue
        if df is None:
            raise ValueError(f"Could not decode catalog: {self.event_file}")

        df.columns = df.columns.str.strip()
        df["event_id"] = df["Deprem Kodu"].astype(str).str.strip()

        for col in ["xM", "MD", "ML", "Mw", "Ms", "Mb"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col] == 0.0, col] = np.nan

        df["latitude"] = pd.to_numeric(df["Enlem"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["Boylam"], errors="coerce")
        df["depth"] = pd.to_numeric(df["Der(km)"], errors="coerce")
        return df

    def _calculate_location_bounds(self) -> None:
        self.lat_min = self.event_catalog["latitude"].min()
        self.lat_max = self.event_catalog["latitude"].max()
        self.lon_min = self.event_catalog["longitude"].min()
        self.lon_max = self.event_catalog["longitude"].max()
        self.depth_min = self.event_catalog["depth"].min()
        self.depth_max = self.event_catalog["depth"].max()
        print("Location bounds:")
        print(f"  Latitude: [{self.lat_min:.4f}, {self.lat_max:.4f}]")
        print(f"  Longitude: [{self.lon_min:.4f}, {self.lon_max:.4f}]")
        print(f"  Depth: [{self.depth_min:.4f}, {self.depth_max:.4f}] km")

    def _build_station_mapping(self) -> Tuple[List[str], Dict[str, int]]:
        if self.station_list_fixed is not None:
            print(f"Using fixed station list with {len(self.station_list_fixed)} stations.")
            names = sorted(self.station_list_fixed)
            return names, {s: i for i, s in enumerate(names)}

        station_names = set()
        for fp in self.file_paths:
            parts = fp.stem.split("_")
            if len(parts) >= 2:
                station_names.add(parts[1])
        names = sorted(station_names)
        return names, {s: i for i, s in enumerate(names)}

    @staticmethod
    def _extract_station_from_filename(filename: str) -> str:
        parts = Path(filename).stem.split("_")
        if len(parts) >= 2:
            return parts[1]
        return "UNKNOWN"

    @staticmethod
    def _extract_event_id_from_filename(filename: str) -> str:
        parts = Path(filename).stem.split("_")
        if len(parts) >= 3 and parts[0] == "OOD" and parts[1] in ("K", "POST"):
            return f"{parts[0]}_{parts[1]}_{parts[2]}"
        if len(parts) >= 2 and parts[0] == "OOD":
            try:
                return f"OOD_{int(parts[1]):02d}"
            except ValueError:
                return f"{parts[0]}_{parts[1]}"
        return parts[0]

    def _get_event_info(self, event_id: str) -> Optional[Dict]:
        rows = self.event_catalog[self.event_catalog["event_id"] == event_id]
        if len(rows) == 0:
            return None
        ev = rows.iloc[0]

        mag = ev[self.magnitude_col]
        if pd.isna(mag):
            for c in ["Mw", "ML", "Ms", "Mb", "MD", "xM"]:
                if not pd.isna(ev[c]):
                    mag = ev[c]
                    break
        if pd.isna(mag):
            mag = 0.0

        lat_norm = (ev["latitude"] - self.lat_min) / (self.lat_max - self.lat_min + 1e-8)
        lon_norm = (ev["longitude"] - self.lon_min) / (self.lon_max - self.lon_min + 1e-8)
        dep_norm = (ev["depth"] - self.depth_min) / (self.depth_max - self.depth_min + 1e-8)

        return {
            "magnitude": float(mag),
            "latitude": float(ev["latitude"]),
            "longitude": float(ev["longitude"]),
            "depth": float(ev["depth"]),
            "latitude_norm": float(lat_norm),
            "longitude_norm": float(lon_norm),
            "depth_norm": float(dep_norm),
            "location_name": str(ev["Yer"]),
        }

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        file_path = self.file_paths[idx]
        try:
            stream = read(str(file_path))
            stream.merge(fill_value=0)

            comp_groups = {"E": [], "N": [], "Z": []}
            for tr in stream:
                comp = tr.stats.channel[-1]
                if comp in comp_groups:
                    comp_groups[comp].append(tr)

            final_traces = []
            pref = {"HH": 0, "HN": 1, "BH": 2, "EH": 3}
            for comp in ("E", "N", "Z"):
                tr_list = comp_groups[comp]
                if not tr_list:
                    continue
                tr_list.sort(key=lambda x: pref.get(x.stats.channel[:2], 99))
                final_traces.append(tr_list[0])
            stream = obspy.Stream(traces=final_traces)

            if len(stream) != 3:
                raise ValueError(
                    f"Expected 3 components (E,N,Z), got {[tr.stats.channel for tr in stream]} for {file_path.name}"
                )
            stream.sort(keys=["channel"])

            event_id = self._extract_event_id_from_filename(file_path.name)
            station_name = stream[0].stats.station or self._extract_station_from_filename(file_path.name)
            event_info = self._get_event_info(event_id)
            if event_info is None:
                event_info = {
                    "magnitude": 0.0,
                    "latitude": 0.0,
                    "longitude": 0.0,
                    "depth": 0.0,
                    "latitude_norm": 0.5,
                    "longitude_norm": 0.5,
                    "depth_norm": 0.5,
                    "location_name": "UNKNOWN",
                }

            station_idx = self.station_to_idx.get(station_name, 0)

            stft_channels = []
            local_mag_min = 0.0
            local_mag_max = 1.0

            for tr in stream:
                data = tr.data.astype(np.float32)
                _, _, zxx = signal.stft(
                    data,
                    fs=tr.stats.sampling_rate,
                    nperseg=self.nperseg,
                    noverlap=self.noverlap,
                    nfft=self.nfft,
                    return_onesided=True,
                    boundary="zeros",
                    padded=True,
                )

                if self.return_magnitude:
                    spec = np.abs(zxx)
                    if self.log_scale:
                        spec = np.log1p(spec)
                    if self.normalize:
                        local_mag_min = float(spec.min())
                        local_mag_max = float(spec.max())
                        if local_mag_max > local_mag_min:
                            spec = (spec - local_mag_min) / (local_mag_max - local_mag_min)
                        else:
                            spec = np.zeros_like(spec)
                    stft_channels.append(spec)
                else:
                    stft_channels.append(zxx)

            min_t = min(ch.shape[1] for ch in stft_channels)
            stft_channels = [ch[:, :min_t] for ch in stft_channels]

            if self.return_magnitude:
                spectrogram = np.stack(stft_channels, axis=0)
            else:
                real_parts = [np.real(ch) for ch in stft_channels]
                imag_parts = [np.imag(ch) for ch in stft_channels]
                spectrogram = np.stack(real_parts + imag_parts, axis=0)

            spec_tensor = torch.from_numpy(spectrogram).float()
            mag_tensor = torch.tensor(event_info["magnitude"], dtype=torch.float32)
            loc_tensor = torch.tensor(
                [event_info["latitude_norm"], event_info["longitude_norm"], event_info["depth_norm"]],
                dtype=torch.float32,
            )
            sta_tensor = torch.tensor(station_idx, dtype=torch.long)

            metadata = {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "event_id": event_id,
                "station_name": station_name,
                "channel_type": file_path.parent.name,
                "sampling_rate": float(stream[0].stats.sampling_rate),
                "n_samples": int(len(stream[0].data)),
                "shape": tuple(spec_tensor.shape),
                "magnitude": event_info["magnitude"],
                "latitude": event_info["latitude"],
                "longitude": event_info["longitude"],
                "depth": event_info["depth"],
                "location_name": event_info["location_name"],
                "station_idx": int(station_idx),
                "mag_min": local_mag_min,
                "mag_max": local_mag_max,
            }
            return spec_tensor, mag_tensor, loc_tensor, sta_tensor, metadata

        except Exception as exc:
            print(f"Error loading {file_path}: {exc}")
            dummy_shape = (3 if self.return_magnitude else 6, self.nfft // 2 + 1, 1)
            return (
                torch.zeros(dummy_shape),
                torch.tensor(0.0, dtype=torch.float32),
                torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32),
                torch.tensor(0, dtype=torch.long),
                {"error": str(exc), "file_path": str(file_path)},
            )


def collate_fn_with_metadata(batch):
    valid = [item for item in batch if "error" not in item[4]]
    if not valid:
        return None, None, None, None, None

    max_t = max(spec.shape[2] for spec, _, _, _, _ in valid)
    specs = []
    mags = []
    locs = []
    stations = []
    metas = []
    for spec, mag, loc, sta, meta in valid:
        if spec.shape[2] < max_t:
            spec = torch.nn.functional.pad(spec, (0, max_t - spec.shape[2]), mode="constant", value=0)
        specs.append(spec)
        mags.append(mag)
        locs.append(loc)
        stations.append(sta)
        metas.append(meta)

    return (
        torch.stack(specs, dim=0),
        torch.stack(mags, dim=0),
        torch.stack(locs, dim=0),
        torch.stack(stations, dim=0),
        metas,
    )

