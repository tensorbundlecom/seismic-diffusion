from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import obspy
import torch
from obspy import read
from torch.utils.data import Dataset


def _read_json(path: str | Path) -> Dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_event_id_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    parts = stem.split("_")
    # OOD_POST_01_STATION_HH.mseed
    if len(parts) >= 4 and parts[0] == "OOD" and parts[1] == "POST":
        return f"{parts[0]}_{parts[1]}_{parts[2]}"
    # default: 20140101010203_STATION_HH.mseed
    return parts[0]


def _extract_station_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) < 3:
        return "UNKNOWN"
    return parts[-2]


def _read_lines(path: str | Path) -> List[str]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


class WaveformDataset(Dataset):
    """
    Unconditional waveform dataset (3C).

    Output:
      waveform: torch.FloatTensor [3, T]
      metadata: dict
    """

    def __init__(
        self,
        file_paths: Sequence[str | Path],
        segment_length: int = 7001,
        sample_rate_hz: float = 100.0,
        preprocess_demean: bool = True,
        preprocess_detrend: bool = True,
        bandpass_enabled: bool = True,
        bandpass_freqmin: float = 0.5,
        bandpass_freqmax: float = 20.0,
        bandpass_corners: int = 4,
        bandpass_zerophase: bool = True,
        normalization_stats_file: Optional[str] = None,
        allow_missing_stats: bool = False,
    ):
        self.file_paths = [Path(p) for p in file_paths]
        self.segment_length = int(segment_length)
        self.sample_rate_hz = float(sample_rate_hz)
        self.preprocess_demean = bool(preprocess_demean)
        self.preprocess_detrend = bool(preprocess_detrend)
        self.bandpass_enabled = bool(bandpass_enabled)
        self.bandpass_freqmin = float(bandpass_freqmin)
        self.bandpass_freqmax = float(bandpass_freqmax)
        self.bandpass_corners = int(bandpass_corners)
        self.bandpass_zerophase = bool(bandpass_zerophase)

        self.stats = None
        if normalization_stats_file:
            self.stats = _read_json(normalization_stats_file)
            # compact schema fallback
            if "mean" not in self.stats and "channels" in self.stats:
                ch = self.stats["channels"]
                self.stats = {
                    "mean": np.asarray(ch["mean"], dtype=np.float64).tolist(),
                    "std": np.asarray(ch["std"], dtype=np.float64).tolist(),
                }
        if (self.stats is None) and (not allow_missing_stats):
            raise ValueError("normalization_stats_file is required unless allow_missing_stats=True")

    def __len__(self) -> int:
        return len(self.file_paths)

    def _pick_enz_traces(self, stream: obspy.Stream) -> obspy.Stream:
        groups: Dict[str, List[obspy.Trace]] = {"E": [], "N": [], "Z": []}
        pref = {"HH": 0, "HN": 1, "BH": 2, "EH": 3}
        for tr in stream:
            comp = tr.stats.channel[-1].upper()
            if comp in groups:
                groups[comp].append(tr)
        out = []
        for comp in ["E", "N", "Z"]:
            traces = groups[comp]
            if not traces:
                continue
            traces.sort(key=lambda t: pref.get(t.stats.channel[:2].upper(), 99))
            out.append(traces[0])
        return obspy.Stream(traces=out)

    def _apply_preprocess(self, tr: obspy.Trace) -> obspy.Trace:
        tr = tr.copy()
        if self.preprocess_detrend:
            tr.detrend("linear")
        if self.preprocess_demean:
            tr.detrend("demean")
        if self.bandpass_enabled:
            tr.filter(
                "bandpass",
                freqmin=self.bandpass_freqmin,
                freqmax=self.bandpass_freqmax,
                corners=self.bandpass_corners,
                zerophase=self.bandpass_zerophase,
            )
        return tr

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        if self.stats is None:
            return x
        mean = np.asarray(self.stats["mean"], dtype=np.float32).reshape(3, 1)
        std = np.asarray(self.stats["std"], dtype=np.float32).reshape(3, 1)
        std = np.clip(std, 1e-8, None)
        return (x - mean) / std

    def __getitem__(self, idx: int):
        fp = self.file_paths[idx]
        try:
            stream = read(str(fp))
            stream.merge(fill_value=0)
            stream = self._pick_enz_traces(stream)
            if len(stream) != 3:
                raise ValueError(f"Expected E/N/Z components, found {[tr.stats.channel for tr in stream]}")

            processed = []
            for tr in stream:
                tr = self._apply_preprocess(tr)
                if float(tr.stats.sampling_rate) != self.sample_rate_hz:
                    tr.resample(self.sample_rate_hz)
                arr = tr.data.astype(np.float32)
                if arr.size < self.segment_length:
                    pad = self.segment_length - arr.size
                    arr = np.pad(arr, (0, pad), mode="constant", constant_values=0.0)
                elif arr.size > self.segment_length:
                    arr = arr[: self.segment_length]
                processed.append(arr)

            x = np.stack(processed, axis=0)  # [3, T]
            x = self._normalize(x)

            meta = {
                "file_path": str(fp),
                "file_name": fp.name,
                "event_id": _extract_event_id_from_filename(fp.name),
                "station_name": _extract_station_from_filename(fp.name),
                "sample_rate_hz": self.sample_rate_hz,
                "segment_length": self.segment_length,
            }
            return torch.from_numpy(x).float(), meta
        except Exception as e:
            return torch.zeros((3, self.segment_length), dtype=torch.float32), {
                "error": str(e),
                "file_path": str(fp),
                "file_name": fp.name,
            }


def collate_fn_waveform(batch):
    valid = [b for b in batch if "error" not in b[1]]
    if not valid:
        return None, None
    x = torch.stack([b[0] for b in valid], dim=0)
    meta = [b[1] for b in valid]
    return x, meta


def build_file_list_from_manifest(manifest_file: str | Path, split: str) -> List[str]:
    manifest = _read_json(manifest_file)
    key = f"{split}_files"
    if key not in manifest["splits"]:
        raise ValueError(f"Split '{split}' not found in manifest")
    return _read_lines(manifest["splits"][key]["file"])

