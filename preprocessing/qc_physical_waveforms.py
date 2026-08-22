#!/usr/bin/env python3
"""Quality-control response-corrected waveform archives.

This script is deliberately downstream of response removal.  It does not
alter any waveform; it verifies the output archive and writes a reproducible
record of what was checked.  When the response-removal job emits its manifest,
pass it with ``--manifest``.  The manifest makes the raw-counts -> physical
unit provenance explicit and lets the review plots compare the two products.

Example
-------
    python preprocessing/qc_physical_waveforms.py \
        --corrected-dir data/physical_waveforms_snr2_2-15hz \
        --raw-dir data/waveforms \
        --manifest data/physical_waveforms_snr2_2-15hz/response_removal_manifest.json \
        --output-dir data/physical_waveforms_snr2_2-15hz/qc

The summary is suitable both for a quick review and as an input gate before
training a physical-unit autoencoder.  Thresholds are intentionally command
line options: their defaults flag clearly pathological deconvolutions, not
ordinary strong-motion records.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from obspy import Stream, read
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
PATH_FIELDS = ("output_path", "corrected_path", "physical_path", "path", "waveform_file")
RAW_PATH_FIELDS = ("raw_path", "input_path", "source_path", "raw_waveform_file")
# A manifest often includes both the ObsPy spelling (``ACC``) and the actual
# SI declaration.  The latter is what matters for global normalization.
UNIT_FIELDS = ("physical_unit", "unit", "units", "output_unit")


@dataclass
class InputRecord:
    corrected_path: Path
    raw_path: Path | None
    provenance: dict[str, Any]
    manifest_status: str | None = None


@dataclass
class QCRecord:
    corrected_path: str
    raw_path: str | None
    qc_status: str
    issues: str
    unit: str | None
    n_traces: int | None
    n_samples: int | None
    sampling_rates_hz: str | None
    duration_seconds: float | None
    max_abs: float | None
    rms: float | None
    peak_to_rms: float | None
    p99_abs: float | None
    clipped_samples: int | None
    clipped_fraction: float | None
    gap_count: int | None
    gap_seconds: float | None
    source_manifest_status: str | None


def _first(record: dict[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        value = record.get(name)
        if value not in (None, ""):
            return value
    return None


def _resolve_path(value: str | Path, *, base: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    # The response-removal manifest records project-relative paths (for
    # portability), whereas a hand-written manifest commonly uses paths
    # relative to the manifest. Support both without silently choosing a path
    # that does not exist.
    manifest_relative = base / path
    project_relative = ROOT / path
    if manifest_relative.exists() or not project_relative.exists():
        return manifest_relative
    return project_relative


def _load_manifest(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="") as handle:
            return list(csv.DictReader(handle)), {}
    with path.open() as handle:
        document = json.load(handle)
    if isinstance(document, list):
        return document, {}
    if isinstance(document, dict):
        for key in ("records", "files", "waveforms", "results"):
            if isinstance(document.get(key), list):
                return document[key], document
        # The preprocessing job deliberately separates immutable run-level
        # provenance (this JSON) from one row per output (the adjacent CSV).
        # This is the canonical format for remove_instrument_response.py.
        results_csv = path.parent / "response_removal_results.csv"
        if results_csv.exists():
            with results_csv.open(newline="") as handle:
                return list(csv.DictReader(handle)), document
    raise ValueError(
        f"{path} must contain records/files/waveforms/results, or have an adjacent "
        "response_removal_results.csv"
    )


def build_inputs(args: argparse.Namespace) -> list[InputRecord]:
    if args.manifest is None:
        paths = sorted(args.corrected_dir.rglob("*.mseed"))
        return [
            InputRecord(
                corrected_path=path,
                raw_path=(args.raw_dir / path.relative_to(args.corrected_dir)) if args.raw_dir else None,
                provenance={"unit": args.unit, "provenance_source": "command_line"},
            )
            for path in paths
        ]

    source = args.manifest
    inputs: list[InputRecord] = []
    rows, run_provenance = _load_manifest(source)
    response_removal = run_provenance.get("response_removal", {})
    processing = run_provenance.get("processing", {})
    for item in rows:
        if not isinstance(item, dict):
            continue
        output = _first(item, PATH_FIELDS)
        if output is None:
            # A manifest can include failures before an output was written; retain
            # them in the report using a stable descriptive pseudo-path.
            output = item.get("waveform_id") or item.get("id") or "<missing-output-path>"
        corrected = _resolve_path(str(output), base=source.parent)
        raw = _first(item, RAW_PATH_FIELDS)
        raw_path = _resolve_path(str(raw), base=source.parent) if raw else None
        if raw_path is None and args.raw_dir and corrected.is_relative_to(args.corrected_dir):
            raw_path = args.raw_dir / corrected.relative_to(args.corrected_dir)
        # Flatten the run-level declarations onto each row. This keeps both the
        # CSV and JSON summaries self-contained and makes a missing physical
        # unit/provenance a visible QC failure instead of an inference.
        provenance = dict(run_provenance)
        provenance.update(item)
        provenance.update({key: value for key, value in response_removal.items() if key not in provenance})
        provenance["processing"] = processing
        if "response_inventory" not in provenance and response_removal.get("inventory"):
            provenance["response_inventory"] = response_removal["inventory"]
        inputs.append(InputRecord(corrected, raw_path, provenance, str(item.get("status")) if item.get("status") else None))
    return inputs


def _unit(record: InputRecord) -> str | None:
    value = _first(record.provenance, UNIT_FIELDS)
    return str(value).strip() if value not in (None, "") else None


def _safe_gaps(stream: Stream) -> tuple[int, float]:
    # get_gaps compares only compatible traces, and is robust to a 3-component
    # stream.  Treat an overlap as a QC issue too because it changes sample count.
    gaps = stream.get_gaps()
    seconds = sum(abs(float(item[6])) for item in gaps)
    return len(gaps), seconds


def _longest_extremum_plateau(values: np.ndarray, peak: float) -> int:
    """Length of the longest contiguous run at an exact absolute maximum.

    Counting every recurrence of a sinusoid's peak creates false clipping
    alarms. Actual digitizer clipping instead produces a contiguous plateau.
    """
    if peak == 0:
        return int(values.size)
    mask = np.abs(values) == peak
    longest = current = 0
    for is_peak in mask:
        current = current + 1 if is_peak else 0
        longest = max(longest, current)
    return longest


def qc_one(record: InputRecord, args: argparse.Namespace) -> QCRecord:
    issues: list[str] = []
    unit = _unit(record)
    if not record.corrected_path.exists():
        return QCRecord(str(record.corrected_path), str(record.raw_path) if record.raw_path else None,
                        "failed", "missing_output", unit, None, None, None, None, None, None,
                        None, None, None, None, None, record.manifest_status)
    if unit is None:
        issues.append("missing_unit_provenance")
    elif unit != args.unit:
        issues.append(f"unexpected_unit:{unit}")
    if not any(key in record.provenance for key in ("response_inventory", "response_inventory_path", "response_removed", "processing", "provenance")):
        issues.append("missing_response_provenance")
    if record.manifest_status and record.manifest_status not in {"processed", "skipped_existing"}:
        issues.append(f"preprocessing_status:{record.manifest_status}")
    try:
        stream = read(str(record.corrected_path))
    except Exception as exc:  # Obspy wraps a broad set of parser failures.
        return QCRecord(str(record.corrected_path), str(record.raw_path) if record.raw_path else None,
                        "failed", f"unreadable:{type(exc).__name__}", unit, None, None, None, None,
                        None, None, None, None, None, None, None, record.manifest_status)

    if len(stream) != args.expected_traces:
        issues.append(f"wrong_trace_count:{len(stream)}")
    all_values = []
    sample_rates = []
    durations = []
    clipped_samples = 0
    for trace in stream:
        values = np.asarray(trace.data, dtype=np.float64)
        sample_rates.append(float(trace.stats.sampling_rate))
        durations.append((trace.stats.npts - 1) / trace.stats.sampling_rate if trace.stats.sampling_rate else math.nan)
        if values.size == 0:
            issues.append(f"empty_trace:{trace.id}")
            continue
        if not np.isfinite(values).all():
            issues.append(f"nonfinite:{trace.id}")
            continue
        peak = float(np.max(np.abs(values)))
        # A contiguous exact-extremum plateau is a useful clipping signal for
        # integer raw data and output accidentally cast back to an integer
        # format. Recurring sinusoid peaks are not treated as clipping.
        if peak > 0:
            clipped_samples += _longest_extremum_plateau(values, peak)
        all_values.append(values)
    gaps, gap_seconds = _safe_gaps(stream)
    if gaps and gap_seconds > args.max_gap_seconds:
        issues.append(f"gaps:{gaps}")
    if not all_values:
        issues.append("no_finite_samples")
        return QCRecord(str(record.corrected_path), str(record.raw_path) if record.raw_path else None,
                        "failed", ";".join(sorted(set(issues))), unit, len(stream), 0,
                        ";".join(map(str, sorted(set(sample_rates)))) or None, None, None, None,
                        None, None, clipped_samples, None, gaps, gap_seconds, record.manifest_status)

    data = np.concatenate(all_values)
    max_abs = float(np.max(np.abs(data)))
    rms = float(np.sqrt(np.mean(np.square(data))))
    peak_to_rms = max_abs / max(rms, np.finfo(float).tiny)
    p99_abs = float(np.percentile(np.abs(data), 99.0))
    clipped_fraction = clipped_samples / data.size
    if max_abs > args.max_abs:
        issues.append("extreme_amplitude")
    if peak_to_rms > args.max_peak_to_rms:
        issues.append("deconvolution_spike")
    if clipped_samples >= args.clip_min_samples and clipped_fraction >= args.clip_fraction:
        issues.append("possible_clipping")
    status = "ok" if not issues else "failed"
    return QCRecord(
        corrected_path=str(record.corrected_path), raw_path=str(record.raw_path) if record.raw_path else None,
        qc_status=status, issues=";".join(sorted(set(issues))), unit=unit, n_traces=len(stream),
        n_samples=int(data.size), sampling_rates_hz=";".join(map(str, sorted(set(sample_rates)))),
        duration_seconds=float(max(durations)) if durations else None, max_abs=max_abs, rms=rms,
        peak_to_rms=peak_to_rms, p99_abs=p99_abs, clipped_samples=clipped_samples,
        clipped_fraction=clipped_fraction, gap_count=gaps, gap_seconds=gap_seconds,
        source_manifest_status=record.manifest_status,
    )


def _matching_trace(stream: Stream, target) -> Any | None:
    matches = stream.select(id=target.id)
    if matches:
        return matches[0]
    channel_matches = stream.select(channel=target.stats.channel)
    return channel_matches[0] if channel_matches else None


def _spectrum(values: np.ndarray, sampling_rate: float) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2 or not np.isfinite(values).all():
        return np.array([]), np.array([])
    values = values - values.mean()
    taper = np.hanning(values.size)
    amplitude = np.abs(np.fft.rfft(values * taper)) / max(values.size, 1)
    frequencies = np.fft.rfftfreq(values.size, d=1.0 / sampling_rate)
    use = frequencies > 0
    return frequencies[use], amplitude[use]


def _bandpass_hz(record: InputRecord) -> tuple[float, float] | None:
    """Read the intended final passband from preprocessing provenance."""
    processing = record.provenance.get("processing")
    values = processing.get("bandpass_hz") if isinstance(processing, dict) else None
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        return None
    try:
        low, high = (float(values[0]), float(values[1]))
    except (TypeError, ValueError):
        return None
    return (low, high) if 0 < low < high else None


def _shade_passband(axis, bandpass: tuple[float, float] | None) -> None:
    if bandpass is None:
        return
    axis.axvspan(*bandpass, color="tab:green", alpha=0.13, label=f"{bandpass[0]:g}–{bandpass[1]:g} Hz passband")


def write_review_plot(record: InputRecord, png_path: Path) -> None:
    corrected = read(str(record.corrected_path))
    raw = read(str(record.raw_path)) if record.raw_path and record.raw_path.exists() else Stream()
    nrows = max(1, len(corrected))
    figure, axes = plt.subplots(nrows, 4, figsize=(22, 3.2 * nrows), squeeze=False)
    unit = _unit(record) or "physical units"
    bandpass = _bandpass_hz(record)
    for row, trace in enumerate(corrected):
        ax_raw_time, ax_acc_time, ax_raw_spectrum, ax_acc_spectrum = axes[row]
        raw_trace = _matching_trace(raw, trace)
        if raw_trace is not None:
            t_raw = np.arange(raw_trace.stats.npts) / raw_trace.stats.sampling_rate
            ax_raw_time.plot(t_raw, raw_trace.data, color="0.25", linewidth=0.65)
            ax_raw_time.set_title(f"{raw_trace.id} — raw")
            ax_raw_time.set_xlabel("seconds from trace start")
            ax_raw_time.set_ylabel("counts")
            ax_raw_time.grid(alpha=0.25)
            f_raw, a_raw = _spectrum(raw_trace.data, raw_trace.stats.sampling_rate)
            if f_raw.size:
                ax_raw_spectrum.loglog(f_raw, a_raw, color="0.25", linewidth=0.75)
            ax_raw_spectrum.set_title("raw-count spectrum")
            ax_raw_spectrum.set_xlabel("frequency (Hz)")
            ax_raw_spectrum.set_ylabel("Fourier amplitude (counts)")
            _shade_passband(ax_raw_spectrum, bandpass)
            ax_raw_spectrum.grid(alpha=0.25, which="both")
        else:
            for axis, title in ((ax_raw_time, "raw counts unavailable"), (ax_raw_spectrum, "raw-count spectrum unavailable")):
                axis.set_title(title)
                axis.text(0.5, 0.5, "No raw waveform supplied", ha="center", va="center", transform=axis.transAxes)
                axis.set_axis_off()
        t = np.arange(trace.stats.npts) / trace.stats.sampling_rate
        ax_acc_time.plot(t, trace.data, color="tab:blue", linewidth=0.7)
        ax_acc_time.set_title(f"{trace.id} — response corrected")
        ax_acc_time.set_xlabel("seconds from trace start")
        ax_acc_time.set_ylabel(unit)
        ax_acc_time.grid(alpha=0.25)
        f_acc, a_acc = _spectrum(trace.data, trace.stats.sampling_rate)
        if f_acc.size:
            ax_acc_spectrum.loglog(f_acc, a_acc, color="tab:blue", linewidth=0.75)
        ax_acc_spectrum.set_title("response-corrected spectrum")
        ax_acc_spectrum.set_xlabel("frequency (Hz)")
        ax_acc_spectrum.set_ylabel(f"Fourier amplitude ({unit})")
        _shade_passband(ax_acc_spectrum, bandpass)
        ax_acc_spectrum.grid(alpha=0.25, which="both")
        if bandpass is not None:
            # A single legend per spectral panel explains the shaded region;
            # raw and corrected amplitudes are intentionally never overlaid.
            ax_raw_spectrum.legend(loc="upper right", fontsize=7)
            ax_acc_spectrum.legend(loc="upper right", fontsize=7)
    passband_title = f"  |  final passband: {bandpass[0]:g}–{bandpass[1]:g} Hz" if bandpass else ""
    figure.suptitle(f"{record.corrected_path.name}{passband_title}", fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _record_key(record: InputRecord) -> str:
    return str(record.corrected_path.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corrected-dir", type=Path, required=True, help="Response-corrected MiniSEED root.")
    parser.add_argument("--raw-dir", type=Path, help="Raw counts root, preserving the same relative layout.")
    parser.add_argument("--manifest", type=Path, help="JSON/CSV response-removal manifest. Strongly recommended.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for QC summary and review PNGs.")
    parser.add_argument("--unit", default="m/s^2", help="Required physical output unit (default: m/s^2).")
    parser.add_argument("--expected-traces", type=int, default=3)
    parser.add_argument("--max-abs", type=float, default=100.0, help="Flag amplitudes above this value in physical units.")
    parser.add_argument("--max-peak-to-rms", type=float, default=100.0, help="Flag narrow deconvolution spikes above this ratio.")
    parser.add_argument("--max-gap-seconds", type=float, default=0.0)
    parser.add_argument("--clip-min-samples", type=int, default=5)
    parser.add_argument("--clip-fraction", type=float, default=1e-3)
    parser.add_argument("--sample-count", type=int, default=6, help="Number of deterministic successful records to plot.")
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--limit", type=int, help="Only inspect this many sorted records (for a smoke test).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.corrected_dir = args.corrected_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.raw_dir:
        args.raw_dir = args.raw_dir.resolve()
    if args.manifest:
        args.manifest = args.manifest.resolve()
    if not args.corrected_dir.is_dir():
        raise SystemExit(f"Corrected directory does not exist: {args.corrected_dir}")
    inputs = build_inputs(args)
    inputs.sort(key=_record_key)
    if args.limit is not None:
        inputs = inputs[:args.limit]
    if not inputs:
        raise SystemExit("No records found. Check --corrected-dir or --manifest.")

    records = [qc_one(item, args) for item in tqdm(inputs, desc="QC physical waveforms")]
    by_path = {_record_key(item): item for item in inputs}
    ok_inputs = [by_path[_record_key(item)] for item, result in zip(inputs, records) if result.qc_status == "ok"]
    selected = random.Random(args.seed).sample(ok_inputs, k=min(args.sample_count, len(ok_inputs)))
    review_dir = args.output_dir / "review_png"
    review_pngs = []
    for item in selected:
        stem = hashlib.sha1(_record_key(item).encode()).hexdigest()[:10]
        png = review_dir / f"{item.corrected_path.stem}_{stem}.png"
        try:
            write_review_plot(item, png)
            review_pngs.append(str(png))
        except Exception as exc:
            print(f"[qc] could not plot {item.corrected_path}: {exc}")

    csv_path = args.output_dir / "qc_summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
    counts = Counter(record.qc_status for record in records)
    issue_counts = Counter(issue for record in records for issue in record.issues.split(";") if issue)
    json_path = args.output_dir / "qc_summary.json"
    with json_path.open("w") as handle:
        json.dump({
            "schema_version": 1,
            "corrected_dir": str(args.corrected_dir),
            "raw_dir": str(args.raw_dir) if args.raw_dir else None,
            "manifest": str(args.manifest) if args.manifest else None,
            "required_unit": args.unit,
            "thresholds": {"max_abs": args.max_abs, "max_peak_to_rms": args.max_peak_to_rms,
                           "max_gap_seconds": args.max_gap_seconds, "clip_min_samples": args.clip_min_samples,
                           "clip_fraction": args.clip_fraction},
            "seed": args.seed,
            "summary": dict(counts),
            "issue_counts": dict(issue_counts),
            "review_pngs": review_pngs,
            "records": [asdict(record) for record in records],
        }, handle, indent=2)
    print(f"[qc] {len(records)} checked: {counts.get('ok', 0)} ok, {counts.get('failed', 0)} failed")
    print(f"[qc] CSV: {csv_path}")
    print(f"[qc] JSON: {json_path}")
    print(f"[qc] review PNGs: {len(review_pngs)} in {review_dir}")


if __name__ == "__main__":
    main()
