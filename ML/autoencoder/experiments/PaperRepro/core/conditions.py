from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_condition_norm_stats(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_condition_vector(row: dict, norm_stats: dict) -> tuple[np.ndarray, np.ndarray]:
    zscore_features = list(norm_stats["zscore_features"])
    passthrough_features = list(norm_stats["passthrough_features"])

    raw_values = []
    normalized_values = []

    for feature in zscore_features:
        value = float(row[feature])
        stats = norm_stats["stats"][feature]
        raw_values.append(value)
        normalized_values.append((value - float(stats["mean"])) / float(stats["std"]))

    for feature in passthrough_features:
        value = float(row[feature])
        raw_values.append(value)
        normalized_values.append(value)

    return (
        np.asarray(raw_values, dtype=np.float32),
        np.asarray(normalized_values, dtype=np.float32),
    )
