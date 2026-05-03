from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def experiment_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_config_path() -> Path:
    return experiment_root() / "configs" / "frozen_paper_repro_v1.yaml"


def load_frozen_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path is not None else default_config_path()
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
