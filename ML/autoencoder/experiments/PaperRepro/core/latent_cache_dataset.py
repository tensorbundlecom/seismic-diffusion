from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class Stage2LatentCacheDataset(Dataset):
    def __init__(self, cache_path: str | Path, *, station_mapping: dict[str, int] | None = None) -> None:
        payload = torch.load(Path(cache_path), map_location="cpu")
        self.latents = payload["latent"].float()
        self.latent_mean = payload["latent_mean"].float()
        self.latent_log_std = payload["latent_log_std"].float()
        self.condition_raw = payload["condition_raw"].float()
        self.condition_normalized = payload["condition_normalized"].float()
        self.meta = payload["meta"]
        self.station_mapping = station_mapping or {}
        self.station_index = self._build_station_index_tensor()

    def _build_station_index_tensor(self) -> torch.Tensor:
        station_codes = list(self.meta.get("station_code", []))
        if not station_codes:
            return torch.full((len(self),), -1, dtype=torch.long)
        if not self.station_mapping:
            unique_codes = sorted(set(station_codes))
            self.station_mapping = {code: idx for idx, code in enumerate(unique_codes)}
        return torch.tensor([int(self.station_mapping[code]) for code in station_codes], dtype=torch.long)

    def __len__(self) -> int:
        return int(self.latents.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "latent": self.latents[index],
            "latent_mean": self.latent_mean[index],
            "latent_log_std": self.latent_log_std[index],
            "condition_raw": self.condition_raw[index],
            "condition_normalized": self.condition_normalized[index],
            "station_index": self.station_index[index],
        }


def load_cache_manifest(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_station_mapping(cache_root: str | Path) -> dict[str, int]:
    cache_root = Path(cache_root)
    station_codes: set[str] = set()
    for split in ("train", "validation", "test", "ood"):
        cache_path = cache_root / f"{split}_latent_cache.pt"
        if not cache_path.exists():
            continue
        payload = torch.load(cache_path, map_location="cpu")
        station_codes.update(payload.get("meta", {}).get("station_code", []))
    return {code: idx for idx, code in enumerate(sorted(station_codes))}
