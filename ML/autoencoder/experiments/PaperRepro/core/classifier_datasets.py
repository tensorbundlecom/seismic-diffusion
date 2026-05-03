from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from .binning import joint_class_index, magnitude_bin_edges_from_config, distance_bin_edges_from_config
from .datasets import build_stage1_dataset_from_config


class PaperReproClassifierDataset(Dataset):
    def __init__(self, cfg: dict, *, splits: Iterable[str]) -> None:
        self.base_dataset = build_stage1_dataset_from_config(cfg, splits=splits)
        self.magnitude_edges = magnitude_bin_edges_from_config(cfg)
        self.distance_edges = distance_bin_edges_from_config(cfg)
        self.labels = []
        for row in self.base_dataset.rows:
            label = joint_class_index(
                float(row["magnitude"]),
                float(row["hypocentral_distance_km"]),
                self.magnitude_edges,
                self.distance_edges,
            )
            if label < 0:
                raise ValueError(
                    f"sample falls outside classifier bins: magnitude={row['magnitude']} distance={row['hypocentral_distance_km']}"
                )
            self.labels.append(label)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.base_dataset[index]
        return {
            "signal": torch.from_numpy(item["representation"]).float(),
            "label": torch.tensor(self.labels[index], dtype=torch.long),
            "magnitude": float(item["magnitude"]),
            "hypocentral_distance_km": float(item["hypocentral_distance_km"]),
            "split": item["split"],
        }

    def class_counts(self) -> np.ndarray:
        counts = np.bincount(np.asarray(self.labels, dtype=np.int64))
        expected = (len(self.magnitude_edges) - 1) * (len(self.distance_edges) - 1)
        if counts.shape[0] < expected:
            counts = np.pad(counts, (0, expected - counts.shape[0]))
        return counts

    def class_weights(self) -> torch.Tensor:
        counts = self.class_counts().astype(np.float64)
        if np.any(counts <= 0):
            raise ValueError(f"classifier bins contain empty classes: {counts.tolist()}")
        weights = 1.0 / counts
        weights = weights / weights.mean()
        return torch.tensor(weights, dtype=torch.float32)


def save_classifier_bins(path: str | Path, *, magnitude_edges: list[float], distance_edges: list[float]) -> None:
    import json

    payload = {
        "magnitude_bin_edges": [float(v) for v in magnitude_edges],
        "distance_bin_edges": [float(v) for v in distance_edges],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
