from __future__ import annotations

from typing import Iterable

import numpy as np


def _as_float_edges(values: Iterable[float]) -> list[float]:
    edges = [float(v) for v in values]
    if len(edges) < 2:
        raise ValueError("at least two bin edges are required")
    return edges


def magnitude_bin_edges_from_config(cfg: dict) -> list[float]:
    return _as_float_edges(cfg["evaluation"]["reporting"]["magnitude_bin_edges"])


def distance_bin_edges_from_config(cfg: dict) -> list[float]:
    return _as_float_edges(cfg["evaluation"]["reporting"]["distance_bin_edges"])


def bin_label(edges: list[float], index: int) -> str:
    return f"[{edges[index]:g}, {edges[index + 1]:g})"


def bin_index(value: float, edges: list[float]) -> int:
    idx = int(np.digitize([float(value)], edges, right=False)[0] - 1)
    if idx < 0 or idx >= len(edges) - 1:
        return -1
    return idx


def joint_class_index(magnitude: float, distance_km: float, magnitude_edges: list[float], distance_edges: list[float]) -> int:
    mag_idx = bin_index(magnitude, magnitude_edges)
    dist_idx = bin_index(distance_km, distance_edges)
    if mag_idx < 0 or dist_idx < 0:
        return -1
    return dist_idx * (len(magnitude_edges) - 1) + mag_idx


def num_joint_classes(magnitude_edges: list[float], distance_edges: list[float]) -> int:
    return (len(magnitude_edges) - 1) * (len(distance_edges) - 1)
