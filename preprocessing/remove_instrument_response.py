#!/usr/bin/env python3
"""Create a response-corrected, physical-unit waveform dataset.

This is intentionally a *new* preprocessing stage: its output must never be
the raw ``data/waveforms`` directory or an existing count-domain filtered
dataset.  Each accepted three-component MiniSEED file is processed as:

    demean -> linear detrend -> cosine taper -> remove response -> 2--15 Hz
    bandpass

The response is selected from StationXML using the trace's full SEED id and
its start time.  A file is rejected as a whole when a component is absent,
has no response valid at that time, or produces invalid samples.  This keeps
the output dataset internally consistent: every waveform is in the requested
physical unit (``ACC`` / m/s^2 by default).

Some legacy KO MiniSEED files have an empty network header.  By default those
traces are interpreted as network ``KO`` without modifying the raw archive;
the assumption and per-file inferred-trace count are recorded in provenance.

Run from the project root, for example:

    python preprocessing/remove_instrument_response.py \
      --inventory eval/station_responses.xml \
      --output-dir data/physical_waveforms_snr2_2-15hz

No network calls are made.  Generate/complete the StationXML inventory first.
The script is resumable: existing output files are skipped only when the
output directory contains a matching provenance manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from obspy import read, read_inventory
from obspy.core.inventory import Inventory
from obspy.core.util import AttribDict
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_NAME = "response_removal_manifest.json"
RESULTS_NAME = "response_removal_results.csv"
SCHEMA_VERSION = 1
UNIT_LABELS = {"ACC": "m/s^2", "VEL": "m/s", "DISP": "m"}

_WORKER_INVENTORY: Inventory | None = None
_WORKER_CONFIG: dict[str, Any] | None = None


def project_path(value: str | Path) -> Path:
    """Resolve command-line paths relative to the repository root."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_pre_filt(value: str) -> tuple[float, float, float, float]:
    try:
        values = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--pre-filt must contain four comma-separated numbers"
        ) from exc
    if len(values) != 4 or not all(math.isfinite(item) and item > 0 for item in values):
        raise argparse.ArgumentTypeError(
            "--pre-filt must contain four finite, positive frequencies"
        )
    if not values[0] < values[1] < values[2] < values[3]:
        raise argparse.ArgumentTypeError("--pre-filt frequencies must increase strictly")
    return values


def parse_optional_water_level(value: str) -> float | None:
    """Parse a non-negative dB value, or ``none`` to rely on ``pre_filt``."""
    if value.strip().lower() in {"none", "off", "null"}:
        return None
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--water-level must be non-negative or 'none'") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("--water-level must be non-negative or 'none'")
    return parsed


def parse_optional_network(value: str) -> str | None:
    """Parse an explicit SEED network code, or ``none`` to forbid inference."""
    value = value.strip()
    if value.lower() in {"none", "off", "null"}:
        return None
    if not value or "." in value or any(character.isspace() for character in value):
        raise argparse.ArgumentTypeError("--default-network must be a SEED network code or 'none'")
    return value


def resolve_summary_waveform(path_value: str, input_dir: Path) -> Path:
    """Resolve legacy summary paths while preserving the source directory."""
    listed = Path(path_value)
    candidates = [listed if listed.is_absolute() else ROOT / listed]
    # Existing summaries use ./data/waveforms/CHANNEL/file.mseed.  The final
    # two components make the selection portable to a different --input-dir.
    if len(listed.parts) >= 2:
        candidates.append(input_dir / listed.parts[-2] / listed.parts[-1])
    candidates.append(input_dir / listed.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not resolve summary waveform {path_value!r} under {input_dir}"
    )


def selected_waveforms(summary_path: Path, input_dir: Path, snr: float, require_gap_free: bool) -> list[Path]:
    summary = pd.read_csv(summary_path)
    required = {"waveform_file", "snr1", "snr2", "snr3", "gap1", "gap2", "gap3"}
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"Summary is missing required columns: {', '.join(missing)}")

    snr_max = summary[["snr1", "snr2", "snr3"]].max(axis=1)
    gap_max = summary[["gap1", "gap2", "gap3"]].max(axis=1)
    mask = snr_max > snr
    if require_gap_free:
        # Match preprocessing/preprocess_waveforms.py exactly so this physical
        # dataset has the same 108,421-file membership as the current AE set.
        mask &= gap_max == 0

    selected: list[Path] = []
    errors: list[str] = []
    for path_value in summary.loc[mask, "waveform_file"]:
        try:
            selected.append(resolve_summary_waveform(str(path_value), input_dir))
        except FileNotFoundError as exc:
            errors.append(str(exc))
    if errors:
        preview = "\n  ".join(errors[:5])
        suffix = "" if len(errors) <= 5 else f"\n  ... {len(errors) - 5} more"
        raise FileNotFoundError(
            f"{len(errors)} selected waveform(s) could not be found:\n  {preview}{suffix}"
        )

    # A duplicate summary row must not create concurrent writes to the same
    # output.  Sorting also gives deterministic manifests and progress order.
    return sorted(set(selected))


def output_path_for(raw_path: Path, input_dir: Path, output_dir: Path) -> Path:
    try:
        relative = raw_path.relative_to(input_dir)
    except ValueError as exc:
        raise ValueError(f"Input file {raw_path} is outside --input-dir {input_dir}") from exc
    return output_dir / relative


def validate_stream_components(stream: Any) -> None:
    """Require exactly one Z/N/E-like component for a complete waveform."""
    if len(stream) != 3:
        raise ValueError(f"expected exactly 3 traces, found {len(stream)}")
    components = [str(trace.stats.channel)[-1:] for trace in stream]
    if len(set(components)) != 3 or any(not item for item in components):
        raise ValueError(f"expected 3 distinct channel components, found {components}")
    station_ids = {(trace.stats.network, trace.stats.station, trace.stats.location) for trace in stream}
    if len(station_ids) != 1:
        raise ValueError("traces do not belong to a single network/station/location")
    if any(trace.stats.npts <= 1 or trace.stats.sampling_rate <= 0 for trace in stream):
        raise ValueError("trace has insufficient samples or invalid sampling rate")


def exact_inventory_for_trace(inventory: Inventory, trace: Any) -> Inventory:
    """Return only a full-ID response epoch covering the complete trace."""
    matching = inventory.select(
        network=trace.stats.network,
        station=trace.stats.station,
        location=trace.stats.location,
        channel=trace.stats.channel,
        time=trace.stats.starttime,
    )
    fully_covered = False
    for network in matching:
        for station in network:
            for channel in station:
                starts_before = channel.start_date is None or channel.start_date <= trace.stats.starttime
                ends_after = channel.end_date is None or channel.end_date >= trace.stats.endtime
                has_sensitivity = (
                    channel.response is not None
                    and channel.response.instrument_sensitivity is not None
                )
                if starts_before and ends_after and has_sensitivity:
                    fully_covered = True
                    break
    if not fully_covered:
        raise ValueError(
            f"no response epoch covers the full interval for {trace.id}: "
            f"{trace.stats.starttime} to {trace.stats.endtime}"
        )
    # get_response performs the same full-id/time lookup used by remove_response
    # and provides an explicit, useful failure for incomplete StationXML.
    response = matching.get_response(trace.id, trace.stats.starttime)
    if response.instrument_sensitivity is None:
        raise ValueError(f"response has no instrument sensitivity for {trace.id}")
    return matching


def process_stream(stream: Any, inventory: Inventory, config: dict[str, Any]) -> Any:
    processed = stream.copy()
    for trace in processed:
        if not str(trace.stats.network or ""):
            default_network = config.get("default_network")
            if not default_network:
                raise ValueError(
                    f"blank network code for {trace.id}; set --default-network explicitly"
                )
            trace.stats.network = default_network
    validate_stream_components(processed)
    for trace in processed:
        nyquist = float(trace.stats.sampling_rate) / 2.0
        if config["freqmax"] >= nyquist:
            raise ValueError(
                f"bandpass freqmax={config['freqmax']} Hz is at/above Nyquist {nyquist:g} Hz "
                f"for {trace.id}"
            )
        if config["pre_filt"][3] >= nyquist:
            raise ValueError(
                f"pre-filt upper corner {config['pre_filt'][3]} Hz is at/above Nyquist {nyquist:g} Hz "
                f"for {trace.id}"
            )
        exact_inventory = exact_inventory_for_trace(inventory, trace)
        trace.detrend("demean")
        trace.detrend("linear")
        trace.taper(max_percentage=config["taper_percentage"], type="cosine")
        trace.remove_response(
            inventory=exact_inventory,
            output=config["output_unit"],
            pre_filt=config["pre_filt"],
            water_level=config["water_level"],
            zero_mean=False,
            taper=False,
        )
        if not np.all(np.isfinite(trace.data)):
            raise ValueError(f"response removal produced non-finite samples for {trace.id}")
        trace.filter(
            "bandpass",
            freqmin=config["freqmin"],
            freqmax=config["freqmax"],
            corners=config["filter_corners"],
            zerophase=config["zero_phase"],
        )
        if not np.all(np.isfinite(trace.data)):
            raise ValueError(f"bandpass produced non-finite samples for {trace.id}")
    return processed


def atomic_write_stream(stream: Any, output_path: Path) -> None:
    """Atomically write physical data with a non-integer MiniSEED encoding.

    Raw downloads commonly retain ``stats.mseed.encoding=STEIM2``.  ObsPy
    otherwise warns and silently changes that incompatible integer encoding
    while writing the float response-corrected data.  Replace the entire
    MiniSEED writer metadata block so no count-domain encoding hint survives;
    standard SEED trace headers (network/station/location/channel/start time)
    remain on ``trace.stats`` untouched.
    """
    writable = stream.copy()
    for trace in writable:
        trace.data = np.ascontiguousarray(trace.data, dtype=np.float64)
        trace.stats.mseed = AttribDict({"encoding": "FLOAT64"})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}.", suffix=".mseed.tmp", dir=output_path.parent
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        writable.write(str(tmp_path), format="MSEED")
        os.replace(tmp_path, output_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _worker_init(inventory_path: str, config: dict[str, Any]) -> None:
    global _WORKER_INVENTORY, _WORKER_CONFIG
    _WORKER_INVENTORY = read_inventory(inventory_path)
    _WORKER_CONFIG = config


def process_one(job: tuple[str, str, bool]) -> dict[str, str]:
    raw_name, output_name, overwrite = job
    raw_path, output_path = Path(raw_name), Path(output_name)
    if output_path.exists() and not overwrite:
        try:
            header_stream = read(str(raw_path), headonly=True)
            inferred_network_traces = sum(
                not str(trace.stats.network or "") for trace in header_stream
            )
        except Exception:
            inferred_network_traces = 0
        return {
            "raw_path": str(raw_path),
            "output_path": str(output_path),
            "status": "skipped_existing",
            "reason": "",
            "inferred_network_traces": str(inferred_network_traces),
        }
    try:
        if _WORKER_INVENTORY is None or _WORKER_CONFIG is None:
            raise RuntimeError("worker was not initialized with StationXML/configuration")
        stream = read(str(raw_path))
        inferred_network_traces = sum(not str(trace.stats.network or "") for trace in stream)
        processed = process_stream(stream, _WORKER_INVENTORY, _WORKER_CONFIG)
        atomic_write_stream(processed, output_path)
        return {
            "raw_path": str(raw_path),
            "output_path": str(output_path),
            "status": "processed",
            "reason": "",
            "inferred_network_traces": str(inferred_network_traces),
        }
    except Exception as exc:
        return {
            "raw_path": str(raw_path),
            "output_path": str(output_path),
            "status": "failed",
            "reason": f"{type(exc).__name__}: {exc}",
            "inferred_network_traces": str(locals().get("inferred_network_traces", 0)),
        }


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def configuration_payload(args: argparse.Namespace, input_dir: Path, output_dir: Path, summary: Path, inventory: Path) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "selection": {
            "waveform_summary": str(summary),
            "waveform_summary_sha256": sha256_file(summary),
            "snr_threshold_strictly_greater_than": args.snr,
            "require_gap_free": args.require_gap_free,
        },
        "response_removal": {
            "inventory": str(inventory),
            "inventory_sha256": sha256_file(inventory),
            "output_unit": args.output_unit,
            "physical_unit": UNIT_LABELS[args.output_unit],
            "pre_filt_hz": list(args.pre_filt),
            "water_level_db": args.water_level,
            "default_network_for_blank_headers": args.default_network,
        },
        "processing": {
            "detrend": ["demean", "linear"],
            "taper": {"type": "cosine", "max_percentage": args.taper_percentage},
            "bandpass_hz": [args.freqmin, args.freqmax],
            "bandpass_corners": args.filter_corners,
            "bandpass_zero_phase": args.zero_phase,
        },
    }


def configuration_hash(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def check_or_create_manifest(
    manifest_path: Path, payload: dict[str, Any], selected_count: int, output_dir: Path
) -> dict[str, Any]:
    expected_hash = configuration_hash(payload)
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("configuration_sha256") != expected_hash:
            raise RuntimeError(
                f"{manifest_path} describes a different preprocessing configuration. "
                "Use a new --output-dir; mixing differently corrected waveforms is unsafe."
            )
        if existing.get("selected_files") != selected_count:
            raise RuntimeError(
                f"{manifest_path} was created for {existing.get('selected_files')} selected files, "
                f"but the current summary selects {selected_count}. Use a new --output-dir so the "
                "physical-unit dataset has an unambiguous membership."
            )
        return existing
    try:
        first_waveform = next(output_dir.rglob("*.mseed"))
    except StopIteration:
        first_waveform = None
    if first_waveform is not None:
        raise RuntimeError(
            f"Output directory already contains waveform {first_waveform}, but has no provenance manifest. "
            "Refusing to mix corrected and untracked files; choose a new --output-dir."
        )
    manifest = {
        **payload,
        "configuration_sha256": expected_hash,
        "created_at": utc_now(),
        "selected_files": selected_count,
        "results_file": RESULTS_NAME,
        "run_history": [],
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def write_results(path: Path, outcomes: Iterable[dict[str, str]]) -> None:
    rows = list(outcomes)
    fieldnames = ["raw_path", "output_path", "status", "reason", "inferred_network_traces"]
    # csv.writer into a StringIO would be equivalent, but this avoids a
    # non-atomic partially written results CSV when the process is interrupted.
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="data/waveforms", help="Raw count-domain MiniSEED root")
    parser.add_argument(
        "--output-dir",
        default="data/physical_waveforms_snr2_2-15hz",
        help="New physical-unit MiniSEED root (must differ from --input-dir)",
    )
    parser.add_argument("--waveform-summary", default="data/waveform_summary.csv")
    parser.add_argument("--inventory", default="eval/station_responses.xml", help="Combined StationXML inventory")
    parser.add_argument(
        "--default-network",
        type=parse_optional_network,
        default="KO",
        help=(
            "Interpret blank MiniSEED network headers as this code (default: KO); "
            "use 'none' to reject all blank-network traces."
        ),
    )
    parser.add_argument("--snr", type=float, default=2.0, help="Select summary rows with max component SNR > this value")
    parser.add_argument(
        "--require-gap-free",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require max component gap ratio == 0 (default: true)",
    )
    parser.add_argument("--output-unit", choices=sorted(UNIT_LABELS), default="ACC")
    parser.add_argument("--pre-filt", type=parse_pre_filt, default=(0.5, 1.0, 20.0, 22.5))
    parser.add_argument(
        "--water-level",
        type=parse_optional_water_level,
        default=None,
        help=(
            "Response-removal water level in dB, or 'none'. Default: none; the explicit "
            "pre-filter stabilizes mixed native sensor quantities without water-level suppression."
        ),
    )
    parser.add_argument("--taper-percentage", type=float, default=0.05)
    parser.add_argument("--freqmin", type=float, default=2.0)
    parser.add_argument("--freqmax", type=float, default=15.0)
    parser.add_argument("--filter-corners", type=int, default=4)
    parser.add_argument(
        "--zero-phase",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply the final bandpass forwards/backwards to avoid phase shift (default: true).",
    )
    parser.add_argument("--workers", type=int, default=1, help="Process count; each worker loads StationXML once")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N selected files (for a smoke test)")
    parser.add_argument("--overwrite", action="store_true", help="Reprocess existing outputs with the same manifest configuration")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.snr < 0 or not math.isfinite(args.snr):
        raise ValueError("--snr must be finite and non-negative")
    if args.water_level is not None and (args.water_level < 0 or not math.isfinite(args.water_level)):
        raise ValueError("--water-level must be finite and non-negative")
    if not 0 < args.taper_percentage <= 0.5:
        raise ValueError("--taper-percentage must be in (0, 0.5]")
    if not 0 < args.freqmin < args.freqmax:
        raise ValueError("require 0 < --freqmin < --freqmax")
    if args.filter_corners < 1:
        raise ValueError("--filter-corners must be positive")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")


def main() -> int:
    args = parse_args()
    validate_args(args)
    input_dir = project_path(args.input_dir).resolve()
    output_dir = project_path(args.output_dir).resolve()
    summary_path = project_path(args.waveform_summary).resolve()
    inventory_path = project_path(args.inventory).resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not summary_path.is_file():
        raise FileNotFoundError(f"Waveform summary does not exist: {summary_path}")
    if not inventory_path.is_file():
        raise FileNotFoundError(f"StationXML inventory does not exist: {inventory_path}")
    if input_dir == output_dir:
        raise ValueError("--output-dir must differ from --input-dir; in-place response removal is forbidden")

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = configuration_payload(args, input_dir, output_dir, summary_path, inventory_path)
    all_selected = selected_waveforms(summary_path, input_dir, args.snr, args.require_gap_free)
    manifest_path = output_dir / MANIFEST_NAME
    manifest = check_or_create_manifest(manifest_path, payload, len(all_selected), output_dir)
    selected = all_selected if args.limit is None else all_selected[: args.limit]
    config = {
        "output_unit": args.output_unit,
        "pre_filt": args.pre_filt,
        "water_level": args.water_level,
        "taper_percentage": args.taper_percentage,
        "freqmin": args.freqmin,
        "freqmax": args.freqmax,
        "filter_corners": args.filter_corners,
        "zero_phase": args.zero_phase,
        "default_network": args.default_network,
    }
    jobs = [(str(path), str(output_path_for(path, input_dir, output_dir)), args.overwrite) for path in selected]
    print(
        f"[response-removal] selected {len(all_selected):,} files "
        f"(running {len(jobs):,}); unit={args.output_unit} ({UNIT_LABELS[args.output_unit]}), "
        f"bandpass={args.freqmin:g}-{args.freqmax:g} Hz"
    )

    outcomes: list[dict[str, str]] = []
    if args.workers == 1:
        _worker_init(str(inventory_path), config)
        iterator = map(process_one, jobs)
        for outcome in tqdm(iterator, total=len(jobs), unit="file"):
            outcomes.append(outcome)
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_worker_init,
            initargs=(str(inventory_path), config),
        ) as executor:
            iterator = executor.map(process_one, jobs)
            for outcome in tqdm(iterator, total=len(jobs), unit="file"):
                outcomes.append(outcome)

    write_results(output_dir / RESULTS_NAME, outcomes)
    counts = {status: sum(row["status"] == status for row in outcomes) for status in ("processed", "skipped_existing", "failed")}
    manifest.setdefault("run_history", []).append(
        {
            "timestamp": utc_now(),
            "requested_files": len(jobs),
            "limit": args.limit,
            "overwrite": args.overwrite,
            **counts,
        }
    )
    atomic_write_json(manifest_path, manifest)
    print(
        "[response-removal] "
        + ", ".join(f"{key}={value:,}" for key, value in counts.items())
        + f"; results: {output_dir / RESULTS_NAME}"
    )
    if counts["failed"]:
        print("[response-removal] inspect failed rows in the results CSV before AE training.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
