"""Resolve artifacts belonging to one selected diffusion embedding export."""

from __future__ import annotations

from pathlib import Path
import hashlib
import math
from typing import Sequence, TypeVar


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EMBEDDINGS_DIR = ROOT / "ML" / "diffusion" / "embeddings"
T = TypeVar("T")


def resolve_embeddings_dir(path: str | Path | None) -> Path:
    """Return the selected export directory, preserving the legacy default."""
    return Path(path or DEFAULT_EMBEDDINGS_DIR).expanduser().resolve()


def artifact_path(embeddings_dir: str | Path | None, filename: str) -> Path:
    """Locate an artifact strictly inside the selected embedding export."""
    directory = resolve_embeddings_dir(embeddings_dir)
    return directory / filename


def deterministic_fraction_subset(
    values: Sequence[T], fraction: float = 1.0, limit: int = 0
) -> list[T]:
    """Return an evenly spaced, deterministic subset without random sampling."""
    if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction!r}")
    count = len(values)
    if count == 0:
        return []
    target = max(1, int(math.ceil(count * fraction)))
    if limit > 0:
        target = min(target, int(limit))
    if target >= count:
        return list(values)
    if target == 1:
        return [values[count // 2]]
    positions = [round(i * (count - 1) / (target - 1)) for i in range(target)]
    return [values[position] for position in positions]


def deterministic_selection_tag(values: Sequence[object]) -> str:
    """Return a compact stable identity for an ordered record selection."""
    payload = "\n".join(str(value) for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:10]
