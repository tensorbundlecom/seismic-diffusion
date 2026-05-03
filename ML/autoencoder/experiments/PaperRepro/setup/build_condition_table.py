from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
import sys

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from core.frozen_config import load_frozen_config
from setup.metadata import (
    EARTH_KM_PER_DEG,
    compute_travel_times,
    dump_json,
    dump_jsonl,
    extract_event_station,
    inspect_waveform_head,
    parse_event_catalog,
    parse_phase_picks_targeted,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Build canonical PaperRepro condition manifest from external_dataset HH waveforms.")
    parser.add_argument("--config", type=str, default=None, help="Frozen config path.")
    parser.add_argument("--limit", type=int, default=0, help="Optional waveform file limit for smoke runs.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    return parser.parse_args()


def _resolve_artifact_paths(cfg: dict) -> dict[str, Path]:
    experiment_root = Path(__file__).resolve().parents[1]
    artifact_cfg = cfg["operations"]["artifacts"]
    return {key: experiment_root / rel_path for key, rel_path in artifact_cfg.items()}


def _iter_waveform_files(waveform_root: Path, waveform_family: str, limit: int) -> list[Path]:
    files = sorted((waveform_root / waveform_family).glob("*.mseed"))
    return files if limit <= 0 else files[:limit]


def build_manifest(cfg: dict, *, limit: int) -> tuple[list[dict], dict]:
    waveform_family = cfg["data"]["waveform_family"]
    waveform_files = _iter_waveform_files(
        waveform_root=Path(cfg["paths"]["filtered_waveform_root"]),
        waveform_family=waveform_family,
        limit=limit,
    )
    event_catalog = parse_event_catalog(cfg["paths"]["event_catalog"])

    needed_pairs = set()
    needed_events = set()
    for file_path in waveform_files:
        try:
            event_id, station_code = extract_event_station(file_path)
        except ValueError:
            continue
        needed_pairs.add((event_id, station_code))
        needed_events.add(event_id)

    phase_rows, event_gap_rows = parse_phase_picks_targeted(
        cfg["paths"]["phase_pick_root"],
        needed_pairs=needed_pairs,
        needed_events=needed_events,
    )

    rows: list[dict] = []
    issue_counts = Counter()
    full_coverage_count = 0
    padding_needed_count = 0
    left_padding_count = 0
    right_padding_count = 0
    missing_catalog_count = 0
    missing_phase_count = 0
    missing_gap_count = 0
    usable_count = 0

    components = tuple(cfg["data"]["components"])
    sample_rate_hz = float(cfg["data"]["sample_rate_hz"])
    pre_origin_sec = float(cfg["data"]["window"]["pre_origin_sec"])
    target_duration_s = float(cfg["data"]["window"]["num_samples"]) / sample_rate_hz
    expected_window_end_s = target_duration_s - (1.0 / sample_rate_hz)
    strict_full_coverage_required = bool(cfg["data"]["window"].get("strict_full_coverage_required", False))
    allow_padding = bool(cfg["data"]["window"].get("allow_padding", False))
    velocity_model_cfg = cfg["conditions"]["velocity_model_1d"]

    for file_path in waveform_files:
        event_id, station_code = extract_event_station(file_path)
        event_row = event_catalog.get(event_id)
        if event_row is None:
            missing_catalog_count += 1
            issue_counts["missing_catalog"] += 1
            rows.append(
                {
                    "file_path": str(file_path),
                    "event_id": event_id,
                    "station_code": station_code,
                    "sample_usable": False,
                    "failure_reason": "missing_catalog",
                }
            )
            continue

        head = inspect_waveform_head(
            file_path,
            origin_time_iso=event_row["origin_time"],
            expected_components=components,
            expected_sample_rate_hz=sample_rate_hz,
        )
        phase_row = phase_rows.get((event_id, station_code))
        if phase_row is None:
            missing_phase_count += 1
            issue_counts["missing_phase"] += 1
            rows.append(
                {
                    "file_path": str(file_path),
                    "event_id": event_id,
                    "station_code": station_code,
                    "origin_time": event_row["origin_time"],
                    "waveform_contract_ok": head.waveform_contract_ok,
                    "waveform_contract_issues": list(head.waveform_contract_issues),
                    "sample_usable": False,
                    "failure_reason": "missing_phase",
                }
            )
            continue

        azimuthal_gap_deg = event_gap_rows.get(event_id)
        if azimuthal_gap_deg is None:
            missing_gap_count += 1
            issue_counts["missing_gap"] += 1

        repi_km = float(phase_row["dist_deg"]) * EARTH_KM_PER_DEG
        travel = compute_travel_times(
            repi_km=repi_km,
            depth_km=float(event_row["depth_km"]),
            pre_origin_sec=pre_origin_sec,
            velocity_model_cfg=velocity_model_cfg,
        )

        missing_left_sec = max(0.0, head.start_offset_s + pre_origin_sec)
        missing_right_sec = max(0.0, expected_window_end_s - head.end_offset_s)
        requires_left_pad = missing_left_sec > 1e-6
        requires_right_pad = missing_right_sec > 1e-6
        window_has_full_coverage = not requires_left_pad and not requires_right_pad
        if window_has_full_coverage:
            full_coverage_count += 1
        else:
            padding_needed_count += 1
            if requires_left_pad:
                left_padding_count += 1
            if requires_right_pad:
                right_padding_count += 1

        sample_usable = head.waveform_contract_ok and (window_has_full_coverage or allow_padding)
        if strict_full_coverage_required:
            sample_usable = sample_usable and window_has_full_coverage

        if sample_usable:
            usable_count += 1

        row = {
            "file_path": str(file_path),
            "event_id": event_id,
            "station_code": station_code,
            "waveform_family": waveform_family,
            "origin_time": event_row["origin_time"],
            "origin_date": event_row["origin_date"],
            "origin_clock": event_row["origin_clock"],
            "latitude": float(event_row["latitude"]),
            "longitude": float(event_row["longitude"]),
            "magnitude": float(event_row["magnitude"]),
            "depth_km": float(event_row["depth_km"]),
            "repi_km": repi_km,
            "hypocentral_distance_km": float(travel["hypocentral_distance_km"]),
            "azimuth_deg": float(phase_row["azimuth_deg"]),
            "azimuth_sin": math.sin(math.radians(float(phase_row["azimuth_deg"]))),
            "azimuth_cos": math.cos(math.radians(float(phase_row["azimuth_deg"]))),
            "azimuthal_gap_deg": float(azimuthal_gap_deg) if azimuthal_gap_deg is not None else None,
            "tP_ref_s": float(travel["tP_ref_s"]),
            "tS_ref_s": float(travel["tS_ref_s"]),
            "dtPS_ref_s": float(travel["dtPS_ref_s"]),
            "phase_has_p": bool(phase_row["has_p"]),
            "phase_has_s": bool(phase_row["has_s"]),
            "waveform_start_time": head.start_time,
            "waveform_end_time": head.end_time,
            "waveform_start_offset_s": head.start_offset_s,
            "waveform_end_offset_s": head.end_offset_s,
            "window_has_full_coverage": window_has_full_coverage,
            "requires_left_pad": requires_left_pad,
            "requires_right_pad": requires_right_pad,
            "missing_left_sec": missing_left_sec,
            "missing_right_sec": missing_right_sec,
            "waveform_contract_ok": head.waveform_contract_ok,
            "waveform_contract_issues": list(head.waveform_contract_issues),
            "trace_count": head.trace_count,
            "sample_rate_hz": head.sample_rate_hz,
            "num_samples_raw": head.num_samples,
            "component_signature": head.component_signature,
            "sample_usable": sample_usable,
            "failure_reason": None if sample_usable else ("waveform_contract_failed" if not head.waveform_contract_ok else "window_coverage_failed"),
        }
        if not sample_usable and row["failure_reason"] is not None:
            issue_counts[str(row["failure_reason"])] += 1
        rows.append(row)

    meta = {
        "version": cfg["version"],
        "limit": limit,
        "waveform_family": waveform_family,
        "total_waveform_files_considered": len(waveform_files),
        "usable_sample_count": usable_count,
        "missing_catalog_count": missing_catalog_count,
        "missing_phase_count": missing_phase_count,
        "missing_gap_count": missing_gap_count,
        "full_coverage_count": full_coverage_count,
        "padding_needed_count": padding_needed_count,
        "left_padding_count": left_padding_count,
        "right_padding_count": right_padding_count,
        "issue_counts": dict(issue_counts),
        "strict_full_coverage_required": strict_full_coverage_required,
        "allow_padding": allow_padding,
        "target_duration_s": target_duration_s,
        "sample_rate_hz": sample_rate_hz,
    }
    return rows, meta


def write_outputs(cfg: dict, *, rows: list[dict], meta: dict) -> None:
    artifact_paths = _resolve_artifact_paths(cfg)
    dump_jsonl(artifact_paths["condition_manifest_jsonl"], rows)
    dump_json(artifact_paths["condition_manifest_meta_json"], meta)

    lines = [
        "# Condition Manifest",
        "",
        f"- `version`: `{meta['version']}`",
        f"- `total_waveform_files_considered`: `{meta['total_waveform_files_considered']}`",
        f"- `usable_sample_count`: `{meta['usable_sample_count']}`",
        f"- `full_coverage_count`: `{meta['full_coverage_count']}`",
        f"- `padding_needed_count`: `{meta['padding_needed_count']}`",
        f"- `left_padding_count`: `{meta['left_padding_count']}`",
        f"- `right_padding_count`: `{meta['right_padding_count']}`",
        f"- `missing_catalog_count`: `{meta['missing_catalog_count']}`",
        f"- `missing_phase_count`: `{meta['missing_phase_count']}`",
        f"- `missing_gap_count`: `{meta['missing_gap_count']}`",
        f"- `issue_counts`: `{meta['issue_counts']}`",
        "",
        "## Notes",
        "",
        "- `sample_usable` currently follows the frozen fixed-length policy with zero-padding allowed.",
        "- `window_has_full_coverage` stays available for later strict-coverage ablations.",
    ]
    artifact_paths["condition_manifest_md"].write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = load_frozen_config(args.config)
    artifact_paths = _resolve_artifact_paths(cfg)
    if not args.overwrite and artifact_paths["condition_manifest_jsonl"].exists():
        raise SystemExit(
            f"Refusing to overwrite existing manifest without --overwrite: {artifact_paths['condition_manifest_jsonl']}"
        )

    rows, meta = build_manifest(cfg, limit=args.limit)
    write_outputs(cfg, rows=rows, meta=meta)


if __name__ == "__main__":
    main()
