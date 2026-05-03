from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

from obspy import read

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from core.frozen_config import load_frozen_config
from setup.paths import default_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Audit HH external_dataset waveform contract for PaperRepro.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Frozen PaperRepro config path. Default: configs/frozen_paper_repro_v1.yaml",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Maximum number of HH waveform files to inspect. Use 0 for all files.",
    )
    return parser.parse_args()


def scan_waveforms(waveform_dir: Path, limit: int) -> dict:
    all_files = sorted(waveform_dir.glob("*.mseed"))
    files = all_files if limit == 0 else all_files[:limit]

    sample_rates = Counter()
    sample_lengths = Counter()
    trace_counts = Counter()
    component_signatures = Counter()
    issues: list[dict[str, str]] = []

    for path in files:
        try:
            stream = read(str(path))
        except Exception as exc:  # pragma: no cover - audit failure path
            issues.append({"file": str(path), "error": f"read_failed: {exc}"})
            continue

        trace_counts[len(stream)] += 1
        sample_rates.update({float(stream[0].stats.sampling_rate): 1})
        sample_lengths.update({int(stream[0].stats.npts): 1})
        signature = tuple(sorted(trace.stats.channel[-1] for trace in stream))
        component_signatures.update({"/".join(signature): 1})

    return {
        "waveform_dir": str(waveform_dir),
        "total_files_available": len(all_files),
        "total_files_scanned": len(files),
        "trace_counts": dict(trace_counts),
        "sample_rates_hz": dict(sample_rates),
        "sample_lengths": dict(sample_lengths),
        "component_signatures": dict(component_signatures),
        "issues": issues,
    }


def write_outputs(summary: dict, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "audit_external_dataset_hh.json"
    md_path = output_root / "audit_external_dataset_hh.md"

    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# HH Audit",
        "",
        f"- `waveform_dir`: `{summary['waveform_dir']}`",
        f"- `total_files_available`: `{summary['total_files_available']}`",
        f"- `total_files_scanned`: `{summary['total_files_scanned']}`",
        f"- `trace_counts`: `{summary['trace_counts']}`",
        f"- `sample_rates_hz`: `{summary['sample_rates_hz']}`",
        f"- `sample_lengths`: `{summary['sample_lengths']}`",
        f"- `component_signatures`: `{summary['component_signatures']}`",
        f"- `issue_count`: `{len(summary['issues'])}`",
    ]
    if summary["issues"]:
        lines.extend(["", "## Issues"])
        for issue in summary["issues"][:20]:
            lines.append(f"- `{issue['file']}`: `{issue['error']}`")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = load_frozen_config(args.config)
    paths = default_paths()

    waveform_family = cfg["data"]["waveform_family"]
    waveform_dir = Path(cfg["paths"]["filtered_waveform_root"]) / waveform_family
    summary = scan_waveforms(waveform_dir=waveform_dir, limit=args.limit)
    summary["expected"] = {
        "waveform_family": waveform_family,
        "components": cfg["data"]["components"],
        "sample_rate_hz": cfg["data"]["sample_rate_hz"],
    }

    write_outputs(summary, paths.results_root / "setup")


if __name__ == "__main__":
    main()
