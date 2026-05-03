from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from core.frozen_config import default_config_path, load_frozen_config
from core.representation import PaperLogSpectrogram, PaperLogSpectrogramConfig
from setup.windowing import load_origin_window


@dataclass
class BenchmarkPair:
    event_id: str
    station_code: str
    split: str
    magnitude: float
    hypocentral_distance_km: float
    azimuthal_gap_deg: float
    file_path: str
    origin_time: str
    note: str


BENCHMARK_SELECTIONS = [
    ("20200528182844", "GAZK", "M1.0 local"),
    ("20160108040543", "KCTX", "M2.5 local"),
    ("20160607080214", "MDNY", "M3.5 regional"),
    ("20151028162002", "RKY", "M4.5 regional"),
    ("20220721154423", "CRLT", "M4.7 distant"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Build the fixed PaperRepro benchmark reference set.")
    parser.add_argument("--config", type=str, default=None, help="Frozen config path.")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional output dir.")
    return parser.parse_args()


def default_output_dir() -> Path:
    return EXPERIMENT_ROOT / "results" / "benchmark_reference_v1"


def load_manifest_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_representation(cfg: dict) -> PaperLogSpectrogram:
    rep_cfg = cfg["representation"]
    return PaperLogSpectrogram(
        PaperLogSpectrogramConfig(
            n_fft=int(rep_cfg["n_fft"]),
            hop_length=int(rep_cfg["hop_length"]),
            clip_min=float(rep_cfg["normalization"]["clip_min"]),
            log_max=float(rep_cfg["normalization"]["log_max"]),
            drop_nyquist=bool(rep_cfg["drop_nyquist"]),
        )
    )


COMPONENT_INDEX = {"E": 0, "N": 1, "Z": 2}


def plot_waveform_grid(samples: list[dict], path: Path, component: str) -> None:
    cols = len(samples)
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 3.8), squeeze=False)
    component_idx = COMPONENT_INDEX[component]
    for col, sample in enumerate(samples):
        ax = axes[0][col]
        waveform = sample["waveform"]
        ax.plot(waveform[component_idx], color="tab:blue", linewidth=0.8, alpha=0.9, label=component)
        meta = sample["meta"]
        ax.set_title(
            f"{meta.event_id}\\n{meta.station_code} {meta.split}\\n{component}  M{meta.magnitude:.1f} D{meta.hypocentral_distance_km:.1f}km",
            fontsize=9,
        )
        ax.set_xlabel("sample")
        ax.set_ylabel("counts")
        if col == 0:
            ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_spectrogram_grid(samples: list[dict], path: Path, component: str) -> None:
    cols = len(samples)
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 3.8), squeeze=False)
    component_idx = COMPONENT_INDEX[component]
    for col, sample in enumerate(samples):
        ax = axes[0][col]
        single = sample["representation"][component_idx]
        im = ax.imshow(single, origin="lower", aspect="auto", cmap="magma")
        meta = sample["meta"]
        ax.set_title(
            f"{meta.event_id}\\n{meta.station_code} {component} {meta.note}",
            fontsize=9,
        )
        ax.set_xlabel("frame")
        ax.set_ylabel("freq")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = load_frozen_config(args.config or default_config_path())
    output_dir = Path(args.output_dir).resolve() if args.output_dir else default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = EXPERIMENT_ROOT / cfg["operations"]["artifacts"]["sample_manifest_jsonl"]
    manifest_rows = load_manifest_rows(manifest_path)
    row_map = {(row["event_id"], row["station_code"]): row for row in manifest_rows}

    representation = build_representation(cfg)
    benchmark_meta: list[BenchmarkPair] = []
    benchmark_samples = []

    for event_id, station_code, note in BENCHMARK_SELECTIONS:
        key = (event_id, station_code)
        if key not in row_map:
            raise KeyError(f"benchmark pair not found in manifest: {key}")
        row = row_map[key]
        meta = BenchmarkPair(
            event_id=event_id,
            station_code=station_code,
            split=str(row["split"]),
            magnitude=float(row["magnitude"]),
            hypocentral_distance_km=float(row["hypocentral_distance_km"]),
            azimuthal_gap_deg=float(row["azimuthal_gap_deg"]),
            file_path=str(row["file_path"]),
            origin_time=str(row["origin_time"]),
            note=note,
        )
        waveform, _ = load_origin_window(
            row["file_path"],
            origin_time_iso=row["origin_time"],
            num_samples=int(cfg["data"]["window"]["num_samples"]),
            sample_rate_hz=float(cfg["data"]["sample_rate_hz"]),
            components=cfg["data"]["components"],
            pre_origin_sec=float(cfg["data"]["window"]["pre_origin_sec"]),
            padding_value=float(cfg["data"]["window"]["padding_value"]),
        )
        spec = representation.transform(waveform).astype(np.float32, copy=False)
        benchmark_meta.append(meta)
        benchmark_samples.append(
            {
                "meta": meta,
                "waveform": waveform.astype(np.float32, copy=False),
                "representation": spec,
            }
        )

    payload = {
        "description": "Fixed benchmark event-station pairs for PaperRepro ablation comparisons.",
        "note": "No M5+ test/ood sample exists in the current frozen split; the highest selected benchmark magnitude is 4.7.",
        "pairs": [asdict(item) for item in benchmark_meta],
    }
    (output_dir / "benchmark_pairs.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    markdown_lines = [
        "# PaperRepro Benchmark Reference Set",
        "",
        "- Amaç: sonraki tüm denemelerde aynı event-station çiftlerini görselleştirip doğrudan karşılaştırmak.",
        "- Not: mevcut frozen `test+ood` havuzunda `M5+` event yok; bu yüzden en büyük seçilen benchmark `M4.7`.",
        "",
        "| event_id | station | split | magnitude | distance_km | azimuthal_gap_deg | note |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for item in benchmark_meta:
        markdown_lines.append(
            f"| {item.event_id} | {item.station_code} | {item.split} | {item.magnitude:.1f} | {item.hypocentral_distance_km:.1f} | {item.azimuthal_gap_deg:.1f} | {item.note} |"
        )
    (output_dir / "benchmark_pairs.md").write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")

    np.savez_compressed(
        output_dir / "benchmark_pairs_arrays.npz",
        waveforms=np.stack([sample["waveform"] for sample in benchmark_samples], axis=0),
        representations=np.stack([sample["representation"] for sample in benchmark_samples], axis=0),
    )
    for component in ("E", "N", "Z"):
        component_dir = output_dir / component
        component_dir.mkdir(parents=True, exist_ok=True)
        plot_waveform_grid(benchmark_samples, component_dir / "benchmark_waveform_grid.png", component)
        plot_spectrogram_grid(benchmark_samples, component_dir / "benchmark_spectrogram_grid.png", component)


if __name__ == "__main__":
    main()
