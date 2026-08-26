"""Path helpers for AE-specific diffusion embedding exports."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable


_SAFE_AE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def project_root() -> Path:
    """Return the repository root, independent of the current directory."""
    return Path(__file__).resolve().parents[2]


def project_relative_path(path: str | Path, *, root: str | Path | None = None) -> str:
    """Serialize a project-local path without embedding a machine-specific root.

    Paths outside the repository deliberately remain absolute: there is no
    portable interpretation for an externally supplied checkpoint.
    """
    resolved = Path(path).expanduser().resolve()
    base = Path(root).expanduser().resolve() if root is not None else project_root()
    try:
        return resolved.relative_to(base).as_posix()
    except ValueError:
        return str(resolved)


def _legacy_project_candidates(recorded: Path, root: Path) -> Iterable[Path]:
    """Yield project-root candidates for an old absolute provenance path."""
    parts = recorded.parts
    # Historical exports used paths such as
    # /mnt/data/seismic-diffusion/ML/autoencoder/checkpoints/<name>/best_model.pt.
    # Retaining the repository-relative suffix makes those exports portable.
    for marker in ("ML",):
        if marker in parts:
            yield root.joinpath(*parts[parts.index(marker):])


def resolve_project_path(
    recorded_path: str | Path,
    *,
    root: str | Path | None = None,
    ae_name: str | None = None,
    require_exists: bool = False,
) -> Path:
    """Resolve a path stored in provenance against this checkout.

    New exports store project-relative paths.  For legacy absolute paths, the
    original path is preferred when it still exists; otherwise the ``ML/...``
    suffix is relocated into this checkout.  If a legacy source also records
    ``ae_name``, the canonical AE checkpoint location is an additional safe
    fallback. ``require_exists`` raises an actionable error after all safe
    candidates have been tried.
    """
    raw = Path(recorded_path).expanduser()
    base = Path(root).expanduser().resolve() if root is not None else project_root()
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
        candidates.extend(_legacy_project_candidates(raw, base))
    else:
        # Relative provenance is always rooted at the repository, never at the
        # process CWD or the embedding directory.
        candidates.append(base / raw)

    if ae_name and _SAFE_AE_NAME.fullmatch(ae_name):
        candidates.append(
            base / "ML" / "autoencoder" / "checkpoints" / ae_name / raw.name
        )

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    if not require_exists:
        return candidates[0].resolve()

    tried = "\n  - ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"Could not resolve AE checkpoint recorded as {str(recorded_path)!r}. "
        f"Tried:\n  - {tried}\n"
        "Create embeddings again in this checkout so source.json records a "
        "project-relative checkpoint path, or place the AE checkpoint at the "
        "matching ML/autoencoder/checkpoints/... location."
    )


def named_embedding_dir(embeddings_root: str | Path, ae_checkpoint: str | Path) -> Path:
    """Return ``embeddings_root/<AE run name>`` for a checkpoint.

    AE checkpoints are stored as ``checkpoints/<run name>/best_model.pt``.  The
    parent directory is therefore the stable run name used to isolate exports.
    """
    checkpoint = Path(ae_checkpoint).expanduser()
    ae_name = checkpoint.parent.name
    if not _SAFE_AE_NAME.fullmatch(ae_name):
        raise ValueError(
            "Cannot derive a safe AE name from checkpoint parent directory "
            f"{checkpoint.parent!s}."
        )
    return Path(embeddings_root).expanduser() / ae_name


def embedding_artifact_paths(embeddings_dir: str | Path) -> dict[str, Path]:
    """Return the core artifact paths for one embedding export directory."""
    directory = Path(embeddings_dir).expanduser()
    return {
        "embeddings": directory / "embeddings.pt",
        "metadata": directory / "metadata.json",
        "source": directory / "source.json",
    }
