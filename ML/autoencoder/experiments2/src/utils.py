"""Shared helper functions for experiments2."""

from __future__ import annotations

import json
import logging
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional

import numpy as np
import torch


def _deep_merge_dicts(base: MutableMapping[str, Any], override: Mapping[str, Any]) -> MutableMapping[str, Any]:
    """Merge ``override`` into ``base`` recursively."""
    for key, value in override.items():
        if (
            key in base
            and isinstance(base[key], MutableMapping)
            and isinstance(value, Mapping)
        ):
            _deep_merge_dicts(base[key], value)
        else:
            base[key] = value
    return base


def load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Load config with optional ``extends`` support.

    If ``extends`` exists, child keys override base keys recursively.
    """
    config_path = Path(config_path)
    cfg = load_json(config_path)
    extends = cfg.pop("extends", None)
    if extends is None:
        return cfg

    base_cfg = load_json(Path(extends))
    merged = _deep_merge_dicts(base_cfg, cfg)
    return dict(merged)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def now_run_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M")


def create_run_tree(run_root: str | Path, run_tag: str) -> Dict[str, Path]:
    """
    Create run directory tree:

    ``runs/exp001/run_YYYYMMDD_HHMM_<tag>/{checkpoints,metrics,plots,tmp}``
    """
    stamp = now_run_stamp()
    safe_tag = run_tag.strip().replace(" ", "_")
    run_dir = ensure_dir(Path(run_root) / f"run_{stamp}_{safe_tag}")
    checkpoints = ensure_dir(run_dir / "checkpoints")
    metrics = ensure_dir(run_dir / "metrics")
    plots = ensure_dir(run_dir / "plots")
    tmp = ensure_dir(run_dir / "tmp")
    return {
        "run_dir": run_dir,
        "checkpoints": checkpoints,
        "metrics": metrics,
        "plots": plots,
        "tmp": tmp,
    }


def save_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def save_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True))
            f.write("\n")


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def save_yaml_compatible(path: str | Path, payload: Mapping[str, Any]) -> None:
    """
    Save JSON text into ``.yaml`` path.

    JSON is valid YAML 1.2, so this avoids a hard dependency on PyYAML.
    """
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
        f.write("\n")


def configure_logger(log_path: str | Path) -> logging.Logger:
    logger = logging.getLogger(f"experiments2_{Path(log_path).name}")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class RunningMean:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += int(n)

    @property
    def avg(self) -> float:
        if self.count <= 0:
            return 0.0
        return self.total / self.count


@dataclass
class EarlyStopping:
    min_epochs: int
    patience: int
    min_delta: float
    best_value: float = math.inf
    bad_epochs: int = 0

    def step(self, epoch_idx_1based: int, value: float) -> bool:
        """
        Returns True when training should stop.
        """
        improved = value < (self.best_value - self.min_delta)
        if improved:
            self.best_value = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1

        if epoch_idx_1based < self.min_epochs:
            return False
        return self.bad_epochs >= self.patience


def beta_linear_warmup(epoch_idx_1based: int, beta_max: float, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return float(beta_max)
    ratio = min(1.0, float(epoch_idx_1based) / float(warmup_epochs))
    return float(beta_max) * ratio


def build_seed_bank(base_seed: int, k: int) -> List[int]:
    rng = random.Random(int(base_seed))
    return [rng.randrange(1, 2**31 - 1) for _ in range(int(k))]


def robust_median_mad(values: List[float]) -> tuple[float, float]:
    """
    Returns (median, MAD_scaled).
    """
    arr = np.asarray(values, dtype=np.float64)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    scale = 1.4826 * mad + 1e-8
    return med, scale


def zscore_robust(value: float, median: float, scale: float) -> float:
    z = (float(value) - float(median)) / float(scale)
    return float(np.clip(z, -4.0, 4.0))
