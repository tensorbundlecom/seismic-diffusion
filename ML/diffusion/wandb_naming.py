"""Deterministic display-name helpers for Weights & Biases training runs."""

from __future__ import annotations

from argparse import Namespace
from typing import Any


def _format_value(value: Any) -> str:
    """Render CLI values consistently for a human-readable W&B display name."""
    if value is None:
        return "none"
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def parse_name_params(raw_params: str | None, args: Namespace) -> list[str]:
    """Validate and normalize a comma-separated list of parsed argument names."""
    if raw_params is None:
        return []

    names = [name.strip() for name in raw_params.split(",")]
    if not names or any(not name for name in names):
        raise ValueError(
            "--wandb_name_params must be a comma-separated list of non-empty "
            "parsed CLI parameter names."
        )

    available = vars(args)
    unknown = [name for name in names if name not in available]
    if unknown:
        available_names = ", ".join(sorted(available))
        raise ValueError(
            "Unknown --wandb_name_params value(s): "
            f"{', '.join(unknown)}. Available parsed CLI parameters: {available_names}."
        )

    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(
            "--wandb_name_params must not contain duplicate parameter names: "
            f"{', '.join(duplicates)}."
        )
    return names


def build_wandb_run_name(
    base_name: str | None,
    raw_params: str | None,
    args: Namespace,
    default_name: str,
) -> str:
    """Build a W&B display name without coupling it to filesystem run paths.

    ``base_name`` overrides the caller-provided default. Each requested parameter
    is appended as ``-name=value`` in the caller's specified order.
    """
    if base_name is None:
        name = default_name
    else:
        name = base_name.strip()
        if not name:
            raise ValueError("--wandb_run_name must not be empty.")

    names = parse_name_params(raw_params, args)
    if not names:
        return name
    suffixes = [f"{param}={_format_value(getattr(args, param))}" for param in names]
    return "-".join([name, *suffixes])
