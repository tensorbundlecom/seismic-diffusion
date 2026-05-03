from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .latent_cache_dataset import Stage2LatentCacheDataset, build_station_mapping


def build_stage2_dataloaders(
    cfg: dict,
    *,
    cache_root: str | Path,
    batch_size_override: int | None = None,
    num_workers_override: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    train_cfg = cfg["stage2"]["training"]
    batch_size = int(batch_size_override if batch_size_override is not None else train_cfg["batch_size"])
    num_workers = int(num_workers_override if num_workers_override is not None else train_cfg["num_workers"])

    cache_root = Path(cache_root)
    station_mapping = build_station_mapping(cache_root)
    train_dataset = Stage2LatentCacheDataset(cache_root / "train_latent_cache.pt", station_mapping=station_mapping)
    val_dataset = Stage2LatentCacheDataset(cache_root / "validation_latent_cache.pt", station_mapping=station_mapping)

    common_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, **common_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **common_kwargs)
    return train_loader, val_loader
