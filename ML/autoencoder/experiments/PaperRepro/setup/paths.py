from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PaperReproPaths:
    repo_root: Path
    experiment_root: Path
    source_repo_root: Path
    data_root: Path
    filtered_waveform_root: Path
    event_catalog: Path
    phase_pick_root: Path
    results_root: Path
    logs_root: Path


def default_paths() -> PaperReproPaths:
    repo_root = Path(__file__).resolve().parents[5]
    experiment_root = Path(__file__).resolve().parents[1]
    source_repo_root = Path("/home/gms/kalem_seismic")
    data_root = source_repo_root / "data" / "external_dataset" / "extracted" / "data"
    return PaperReproPaths(
        repo_root=repo_root,
        experiment_root=experiment_root,
        source_repo_root=source_repo_root,
        data_root=data_root,
        filtered_waveform_root=data_root / "filtered_waveforms",
        event_catalog=data_root / "events" / "20140101_20251101_0.0_9.0_9_339.txt",
        phase_pick_root=data_root / "phase_picks",
        results_root=experiment_root / "results",
        logs_root=experiment_root / "logs",
    )
