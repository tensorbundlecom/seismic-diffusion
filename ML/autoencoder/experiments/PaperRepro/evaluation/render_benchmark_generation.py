from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from core.conditions import build_condition_vector, load_condition_norm_stats
from core.latent_cache_dataset import build_station_mapping
from core.frozen_config import default_config_path, load_frozen_config
from core.representation import PaperLogSpectrogram, PaperLogSpectrogramConfig
from core.stage1_autoencoder import build_stage1_autoencoder_from_config
from core.stage2_unet import build_stage2_unet_from_config
from core.edm import Stage2EDM, heun_sampler
from evaluation.evaluate_stage1_autoencoder import griffin_lim_torch, normalized_to_magnitude
from setup.windowing import load_origin_window


COMPONENTS = ("E", "N", "Z")
COMPONENT_INDEX = {"E": 0, "N": 1, "Z": 2}


@dataclass
class BenchmarkSample:
    event_id: str
    station_code: str
    split: str
    magnitude: float
    hypocentral_distance_km: float
    azimuthal_gap_deg: float
    file_path: str
    origin_time: str
    note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Render the fixed benchmark set for a given Stage-2 checkpoint.")
    parser.add_argument("--config", type=str, default=None, help="Frozen config path.")
    parser.add_argument("--stage1-checkpoint", type=str, required=True, help="Stage-1 checkpoint path.")
    parser.add_argument("--stage2-checkpoint", type=str, required=True, help="Stage-2 checkpoint path.")
    parser.add_argument("--benchmark-json", type=str, default=None, help="Benchmark pair json path.")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda.")
    parser.add_argument("--cache-root", type=str, default=None, help="Optional Stage-2 cache root for station mapping.")
    parser.add_argument("--num-steps", type=int, default=None, help="Override Heun steps.")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional output dir.")
    return parser.parse_args()


def detect_device(override: str | None) -> torch.device:
    if override:
        return torch.device(override)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def benchmark_json_default() -> Path:
    return EXPERIMENT_ROOT / "results" / "benchmark_reference_v1" / "benchmark_pairs.json"


def default_output_dir(stage2_checkpoint: Path) -> Path:
    run_name = stage2_checkpoint.parents[1].name
    return EXPERIMENT_ROOT / "results" / "benchmark_generation" / run_name


def load_stage1(cfg: dict, checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_stage1_autoencoder_from_config(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def load_stage2(cfg: dict, checkpoint_path: Path, device: torch.device, *, num_stations: int | None = None):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    denoiser = build_stage2_unet_from_config(cfg, num_stations=num_stations)
    edm_cfg = cfg["stage2"]["edm"]
    model = Stage2EDM(
        denoiser,
        sigma_min=float(edm_cfg["sigma_min"]),
        sigma_max=float(edm_cfg["sigma_max"]),
        sigma_data=float(edm_cfg["sigma_data"]),
        p_mean=float(edm_cfg["P_mean"]),
        p_std=float(edm_cfg["P_std"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


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


def load_benchmark_samples(path: Path) -> list[BenchmarkSample]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [BenchmarkSample(**row) for row in payload["pairs"]]


def load_manifest_condition_rows(cfg: dict) -> dict[tuple[str, str], dict]:
    manifest_path = EXPERIMENT_ROOT / cfg["operations"]["artifacts"]["sample_manifest_jsonl"]
    rows = {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[(row["event_id"], row["station_code"])] = row
    return rows


def generate_latent(stage2_model: Stage2EDM, cond_norm: torch.Tensor, latent_shape: tuple[int, ...], device: torch.device, cfg: dict, num_steps: int, station_index: torch.Tensor | None = None) -> torch.Tensor:
    edm_cfg = cfg["stage2"]["edm"]
    return heun_sampler(
        stage2_model,
        cond=cond_norm,
        shape=(1, *latent_shape),
        station_index=station_index,
        sigma_min=float(edm_cfg["sigma_min"]),
        sigma_max=float(edm_cfg["sigma_max"]),
        rho=float(edm_cfg["rho"]),
        num_steps=num_steps,
        device=device,
    )


def plot_waveform_component(samples: list[dict], component: str, path: Path) -> None:
    cols = len(samples)
    fig, axes = plt.subplots(2, cols, figsize=(4 * cols, 5), squeeze=False)
    idx = COMPONENT_INDEX[component]
    for col, sample in enumerate(samples):
        for row, waveform, title in (
            (0, sample["target_waveform"], "target"),
            (1, sample["generated_waveform"], "generated"),
        ):
            ax = axes[row][col]
            ax.plot(waveform[idx], color="tab:blue", linewidth=0.8)
            meta = sample["meta"]
            ax.set_title(
                f"{title}\\n{meta.event_id} {meta.station_code}\\n{component} M{meta.magnitude:.1f} D{meta.hypocentral_distance_km:.1f}",
                fontsize=9,
            )
            ax.set_xlabel("sample")
            ax.set_ylabel("counts")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_spectrogram_component(samples: list[dict], component: str, path: Path) -> None:
    cols = len(samples)
    fig, axes = plt.subplots(2, cols, figsize=(4 * cols, 5), squeeze=False)
    idx = COMPONENT_INDEX[component]
    for col, sample in enumerate(samples):
        for row, spec, title in (
            (0, sample["target_repr"], "target"),
            (1, sample["generated_repr"], "generated"),
        ):
            ax = axes[row][col]
            im = ax.imshow(spec[idx], origin="lower", aspect="auto", cmap="magma")
            meta = sample["meta"]
            ax.set_title(
                f"{title}\\n{meta.event_id} {meta.station_code}\\n{component} {meta.note}",
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
    stage1_checkpoint = Path(args.stage1_checkpoint).resolve()
    stage2_checkpoint = Path(args.stage2_checkpoint).resolve()
    benchmark_path = Path(args.benchmark_json).resolve() if args.benchmark_json else benchmark_json_default()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else default_output_dir(stage2_checkpoint)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config or default_config_path(), output_dir / "config_snapshot.yaml")

    device = detect_device(args.device)
    cache_root = Path(args.cache_root).resolve() if args.cache_root else None
    station_mapping = build_station_mapping(cache_root) if cache_root else {}
    stage1_model = load_stage1(cfg, stage1_checkpoint, device)
    stage2_model = load_stage2(cfg, stage2_checkpoint, device, num_stations=(len(station_mapping) if station_mapping else None))
    benchmark_samples = load_benchmark_samples(benchmark_path)
    manifest_rows = load_manifest_condition_rows(cfg)
    representation = build_representation(cfg)
    norm_stats = load_condition_norm_stats(EXPERIMENT_ROOT / cfg["operations"]["artifacts"]["condition_norm_stats_json"])
    latent_shape = tuple(int(v) for v in cfg["stage2"]["latent_input_shape"])
    num_steps = int(args.num_steps or cfg["stage2"]["sampling"]["num_steps"])
    clip_min = float(cfg["representation"]["normalization"]["clip_min"])
    log_max = float(cfg["representation"]["normalization"]["log_max"])

    rendered = []
    for sample in benchmark_samples:
        row = manifest_rows[(sample.event_id, sample.station_code)]
        waveform, _ = load_origin_window(
            row["file_path"],
            origin_time_iso=row["origin_time"],
            num_samples=int(cfg["data"]["window"]["num_samples"]),
            sample_rate_hz=float(cfg["data"]["sample_rate_hz"]),
            components=cfg["data"]["components"],
            pre_origin_sec=float(cfg["data"]["window"]["pre_origin_sec"]),
            padding_value=float(cfg["data"]["window"]["padding_value"]),
        )
        target_repr = representation.transform(waveform).astype(np.float32, copy=False)
        cond_raw = {}
        for feature in list(norm_stats["zscore_features"]) + list(norm_stats["passthrough_features"]):
            cond_raw[feature] = float(row[feature])
        _, cond_norm = build_condition_vector(cond_raw, norm_stats)
        cond_tensor = torch.from_numpy(cond_norm).to(device=device, dtype=torch.float32).unsqueeze(0)
        station_index_tensor = None
        if station_mapping:
            station_index_tensor = torch.tensor([station_mapping[sample.station_code]], device=device, dtype=torch.long)

        with torch.no_grad():
            latent = generate_latent(stage2_model, cond_tensor, latent_shape, device, cfg, num_steps, station_index=station_index_tensor)
            generated_repr = stage1_model.decode(latent).detach().cpu()[0]
        generated_mag = normalized_to_magnitude(generated_repr, clip_min=clip_min, log_max=log_max)
        generated_waveform = griffin_lim_torch(
            generated_mag,
            n_fft=int(cfg["representation"]["n_fft"]),
            hop_length=int(cfg["representation"]["hop_length"]),
            num_iters=int(cfg["representation"]["inverse"]["n_iter"]),
            length=int(cfg["data"]["window"]["num_samples"]),
        ).detach().cpu().numpy()

        rendered.append(
            {
                "meta": sample,
                "target_waveform": waveform.astype(np.float32, copy=False),
                "generated_waveform": generated_waveform.astype(np.float32, copy=False),
                "target_repr": target_repr,
                "generated_repr": generated_repr.numpy().astype(np.float32, copy=False),
                "condition_raw": cond_raw,
            }
        )

    payload = {
        "stage1_checkpoint": str(stage1_checkpoint),
        "stage2_checkpoint": str(stage2_checkpoint),
        "num_steps": num_steps,
        "pairs": [
            {
                "event_id": item["meta"].event_id,
                "station_code": item["meta"].station_code,
                "split": item["meta"].split,
                "magnitude": item["meta"].magnitude,
                "hypocentral_distance_km": item["meta"].hypocentral_distance_km,
                "note": item["meta"].note,
                "condition_raw": item["condition_raw"],
            }
            for item in rendered
        ],
    }
    (output_dir / "benchmark_generation_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    np.savez_compressed(
        output_dir / "benchmark_generation_arrays.npz",
        target_waveforms=np.stack([item["target_waveform"] for item in rendered], axis=0),
        generated_waveforms=np.stack([item["generated_waveform"] for item in rendered], axis=0),
        target_representations=np.stack([item["target_repr"] for item in rendered], axis=0),
        generated_representations=np.stack([item["generated_repr"] for item in rendered], axis=0),
    )

    for component in COMPONENTS:
        component_dir = output_dir / component
        component_dir.mkdir(parents=True, exist_ok=True)
        plot_waveform_component(rendered, component, component_dir / "benchmark_generation_waveform_grid.png")
        plot_spectrogram_component(rendered, component, component_dir / "benchmark_generation_spectrogram_grid.png")


if __name__ == "__main__":
    main()
