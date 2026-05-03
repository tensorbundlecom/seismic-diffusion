from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from core.frozen_config import load_frozen_config
from core.representation import PaperLogSpectrogram, PaperLogSpectrogramConfig
from setup.windowing import load_origin_window


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Smoke-test origin-time window extraction and log-spectrogram generation.")
    parser.add_argument("--config", type=str, default=None, help="Frozen config path.")
    parser.add_argument("--count", type=int, default=8, help="How many samples to include in smoke artifacts.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing smoke artifacts.")
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    args = parse_args()
    cfg = load_frozen_config(args.config)
    experiment_root = Path(__file__).resolve().parents[1]
    artifact_cfg = cfg["operations"]["artifacts"]
    sample_manifest_path = experiment_root / artifact_cfg["sample_manifest_jsonl"]
    output_json = experiment_root / "results" / "setup" / "representation_smoke_paper_repro_v1.json"
    output_md = experiment_root / "results" / "setup" / "representation_smoke_paper_repro_v1.md"
    output_npz = experiment_root / "results" / "setup" / "representation_smoke_examples_paper_repro_v1.npz"

    if not args.overwrite and any(path.exists() for path in [output_json, output_md, output_npz]):
        raise SystemExit("Refusing to overwrite existing smoke artifacts without --overwrite.")

    sample_rows = _load_jsonl(sample_manifest_path)
    selected_rows = sample_rows[: args.count]
    if not selected_rows:
        raise SystemExit("No sample rows found for smoke test.")

    rep = PaperLogSpectrogram(
        PaperLogSpectrogramConfig(
            n_fft=int(cfg["representation"]["n_fft"]),
            hop_length=int(cfg["representation"]["hop_length"]),
            clip_min=float(cfg["representation"]["normalization"]["clip_min"]),
            log_max=float(cfg["representation"]["normalization"]["log_max"]),
            drop_nyquist=bool(cfg["representation"]["drop_nyquist"]),
        )
    )

    waveforms = []
    reps = []
    summaries = []
    for row in selected_rows:
        waveform, info = load_origin_window(
            row["file_path"],
            origin_time_iso=row["origin_time"],
            num_samples=int(cfg["data"]["window"]["num_samples"]),
            sample_rate_hz=float(cfg["data"]["sample_rate_hz"]),
            components=cfg["data"]["components"],
            pre_origin_sec=float(cfg["data"]["window"]["pre_origin_sec"]),
            padding_value=float(cfg["data"]["window"]["padding_value"]),
        )
        representation = rep.transform(waveform)
        waveforms.append(waveform)
        reps.append(representation)
        summaries.append(
            {
                "event_id": row["event_id"],
                "station_code": row["station_code"],
                "split": row["split"],
                "waveform_shape": list(waveform.shape),
                "representation_shape": list(representation.shape),
                "requires_left_pad": info.requires_left_pad,
                "requires_right_pad": info.requires_right_pad,
                "representation_min": float(representation.min()),
                "representation_max": float(representation.max()),
            }
        )

    waveform_stack = np.stack(waveforms, axis=0)
    representation_stack = np.stack(reps, axis=0)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        waveforms=waveform_stack,
        representations=representation_stack,
    )

    summary = {
        "count": len(selected_rows),
        "waveform_batch_shape": list(waveform_stack.shape),
        "representation_batch_shape": list(representation_stack.shape),
        "representation_global_min": float(representation_stack.min()),
        "representation_global_max": float(representation_stack.max()),
        "samples": summaries,
    }
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# Representation Smoke",
        "",
        f"- `count`: `{summary['count']}`",
        f"- `waveform_batch_shape`: `{summary['waveform_batch_shape']}`",
        f"- `representation_batch_shape`: `{summary['representation_batch_shape']}`",
        f"- `representation_global_min`: `{summary['representation_global_min']}`",
        f"- `representation_global_max`: `{summary['representation_global_max']}`",
        "",
        "## Samples",
        "",
    ]
    for sample in summaries:
        lines.append(
            f"- `{sample['event_id']}_{sample['station_code']}` split=`{sample['split']}` "
            f"waveform=`{sample['waveform_shape']}` rep=`{sample['representation_shape']}` "
            f"left_pad=`{sample['requires_left_pad']}` right_pad=`{sample['requires_right_pad']}` "
            f"rep_range=`[{sample['representation_min']:.4f}, {sample['representation_max']:.4f}]`"
        )
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
