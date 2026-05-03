from __future__ import annotations

import math
from collections import Counter
from typing import Iterable

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .datasets import build_stage1_dataset_from_config


def _magnitude_bin_label(magnitude: float, edges: list[float]) -> str:
    for low, high in zip(edges[:-1], edges[1:]):
        if low <= magnitude < high:
            return f"[{low},{high})"
    return f"[{edges[-2]},{edges[-1]}]"


def build_soft_magnitude_sampler(rows: list[dict], *, edges: list[float], exponent: float) -> WeightedRandomSampler:
    bin_labels = [_magnitude_bin_label(float(row["magnitude"]), edges) for row in rows]
    counts = Counter(bin_labels)
    weights = [1.0 / math.pow(counts[label], exponent) for label in bin_labels]
    weight_tensor = torch.tensor(weights, dtype=torch.double)
    return WeightedRandomSampler(weight_tensor, num_samples=len(rows), replacement=True)


def build_stage1_dataloaders(cfg: dict, *, num_workers_override: int | None = None, batch_size_override: int | None = None) -> tuple[DataLoader, DataLoader]:
    train_dataset = build_stage1_dataset_from_config(cfg, splits=["train"])
    val_dataset = build_stage1_dataset_from_config(cfg, splits=["validation"])
    train_cfg = cfg["stage1"]["training"]
    batch_size = int(batch_size_override if batch_size_override is not None else train_cfg["batch_size"])
    num_workers = int(num_workers_override if num_workers_override is not None else train_cfg["num_workers"])

    sampler_cfg = train_cfg["sampler"]
    train_sampler = build_soft_magnitude_sampler(
        train_dataset.rows,
        edges=[float(x) for x in sampler_cfg["magnitude_bin_edges"]],
        exponent=float(sampler_cfg["exponent"]),
    )

    common_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        drop_last=False,
        **common_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        **common_kwargs,
    )
    return train_loader, val_loader
