"""Run-scoped output paths for collision-free evaluator artifacts."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_OUTPUT_ROOT = PROJECT_ROOT / "eval"
EVAL_OUTPUT_ROOT_ENV = "SEISMIC_EVAL_OUTPUT_ROOT"


def evaluation_output_root() -> Path:
    """Return the active suite root, preserving standalone evaluator defaults."""
    configured = os.environ.get(EVAL_OUTPUT_ROOT_ENV)
    return Path(configured).expanduser().resolve() if configured else DEFAULT_EVAL_OUTPUT_ROOT


def evaluation_output_dir(name: str, standalone_name: str | None = None) -> Path:
    """Return one evaluator directory, optionally preserving a legacy default."""
    for candidate in (name, standalone_name):
        if candidate is not None and (
            Path(candidate).name != candidate or candidate in {"", ".", ".."}
        ):
            raise ValueError(
                f"Evaluator output name must be one safe path component: {candidate!r}"
            )
    configured = os.environ.get(EVAL_OUTPUT_ROOT_ENV)
    directory_name = name if configured else (standalone_name or name)
    return evaluation_output_root() / directory_name
