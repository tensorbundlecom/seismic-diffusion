"""Small, dependency-free helpers for best-validation checkpoint selection."""

import math
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Optional
from uuid import uuid4


def is_strictly_better_validation_loss(
    candidate_loss: float, best_loss: Optional[float]
) -> bool:
    """Return whether a finite candidate strictly improves the recorded best loss.

    Equal losses intentionally do not replace the checkpoint, so the earliest
    epoch attaining a value remains the selected model.
    """
    if not math.isfinite(candidate_loss):
        return False
    return best_loss is None or not math.isfinite(best_loss) or candidate_loss < best_loss


def best_validation_checkpoint_path(
    checkpoint_root: Path, wandb_run_id: Optional[str] = None
) -> Path:
    """Return a stable best-checkpoint path, isolated for concurrent W&B runs."""
    return run_checkpoint_path(checkpoint_root, "best_val", wandb_run_id)


def run_checkpoint_path(
    checkpoint_root: Path, checkpoint_name: str, wandb_run_id: Optional[str] = None
) -> Path:
    """Return a checkpoint path, scoped to a W&B run when applicable."""
    if wandb_run_id:
        return checkpoint_root / "wandb_runs" / wandb_run_id / checkpoint_name
    return checkpoint_root / checkpoint_name


def replace_checkpoint_directory(
    staged_directory: Path,
    destination: Path,
    *,
    move: Optional[Callable[[Path, Path], None]] = None,
) -> None:
    """Replace a checkpoint directory while retaining the old one on save failure.

    ``staged_directory`` must be a completed sibling of ``destination``. The
    old destination is moved aside only after staging succeeds, and is restored
    if installing the staged directory fails.
    """
    if not staged_directory.is_dir():
        raise ValueError(f"Staged checkpoint directory does not exist: {staged_directory}")
    if staged_directory.parent != destination.parent:
        raise ValueError("Staged checkpoint directory must be a sibling of its destination.")

    def move_path(source: Path, target: Path) -> None:
        if move is not None:
            move(source, target)
        else:
            source.replace(target)

    if not destination.exists():
        move_path(staged_directory, destination)
        return

    backup = destination.parent / f".{destination.name}.backup-{uuid4().hex}"
    move_path(destination, backup)
    try:
        move_path(staged_directory, destination)
    except BaseException:
        if backup.exists():
            move_path(backup, destination)
        raise
    shutil.rmtree(backup)


def select_evaluation_checkpoint(
    best_checkpoint: Path, best_loss: Optional[float], final_checkpoint: Path
) -> Path:
    """Use only a finite best loss selected in this process for final evaluation."""
    if best_loss is not None and math.isfinite(best_loss) and best_checkpoint.is_dir():
        return best_checkpoint
    return final_checkpoint
