from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from torch.utils.data import Dataset

from .representation import PaperLogSpectrogram, PaperLogSpectrogramConfig
from setup.windowing import load_origin_window


def load_jsonl_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


class Stage1LogSpectrogramDataset(Dataset):
    def __init__(
        self,
        *,
        manifest_path: str | Path,
        splits: Iterable[str],
        num_samples: int,
        sample_rate_hz: float,
        components: Iterable[str],
        pre_origin_sec: float,
        padding_value: float,
        representation: PaperLogSpectrogram,
    ) -> None:
        all_rows = load_jsonl_rows(manifest_path)
        split_set = set(splits)
        self.rows = [row for row in all_rows if row.get("split") in split_set]
        self.num_samples = int(num_samples)
        self.sample_rate_hz = float(sample_rate_hz)
        self.components = tuple(components)
        self.pre_origin_sec = float(pre_origin_sec)
        self.padding_value = float(padding_value)
        self.representation = representation

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        waveform, window_info = load_origin_window(
            row["file_path"],
            origin_time_iso=row["origin_time"],
            num_samples=self.num_samples,
            sample_rate_hz=self.sample_rate_hz,
            components=self.components,
            pre_origin_sec=self.pre_origin_sec,
            padding_value=self.padding_value,
        )
        representation = self.representation.transform(waveform)
        return {
            "representation": representation.astype(np.float32, copy=False),
            "event_id": row["event_id"],
            "station_code": row["station_code"],
            "split": row["split"],
            "file_path": row["file_path"],
            "origin_time": row["origin_time"],
            "magnitude": float(row["magnitude"]),
            "hypocentral_distance_km": float(row["hypocentral_distance_km"]),
            "requires_left_pad": bool(window_info.requires_left_pad),
            "requires_right_pad": bool(window_info.requires_right_pad),
        }


def build_stage1_dataset_from_config(cfg: dict, *, splits: Iterable[str]) -> Stage1LogSpectrogramDataset:
    experiment_root = Path(__file__).resolve().parents[1]
    manifest_path = experiment_root / cfg["operations"]["artifacts"]["sample_manifest_jsonl"]
    representation = PaperLogSpectrogram(
        PaperLogSpectrogramConfig(
            n_fft=int(cfg["representation"]["n_fft"]),
            hop_length=int(cfg["representation"]["hop_length"]),
            clip_min=float(cfg["representation"]["normalization"]["clip_min"]),
            log_max=float(cfg["representation"]["normalization"]["log_max"]),
            drop_nyquist=bool(cfg["representation"]["drop_nyquist"]),
        )
    )
    return Stage1LogSpectrogramDataset(
        manifest_path=manifest_path,
        splits=splits,
        num_samples=int(cfg["data"]["window"]["num_samples"]),
        sample_rate_hz=float(cfg["data"]["sample_rate_hz"]),
        components=cfg["data"]["components"],
        pre_origin_sec=float(cfg["data"]["window"]["pre_origin_sec"]),
        padding_value=float(cfg["data"]["window"]["padding_value"]),
        representation=representation,
    )
