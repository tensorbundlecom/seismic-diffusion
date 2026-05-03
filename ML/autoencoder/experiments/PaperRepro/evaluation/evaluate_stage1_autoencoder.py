from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from core.datasets import build_stage1_dataset_from_config
from core.frozen_config import default_config_path, load_frozen_config
from core.stage1_autoencoder import build_stage1_autoencoder_from_config
from setup.windowing import load_origin_window


@dataclass
class SplitMetrics:
    split: str
    num_samples_primary: int
    num_samples_waveform: int
    reconstruction_loss: float
    spec_corr: float
    lsd: float
    mr_lsd: float
    waveform_mse: float
    waveform_asd: float
    oracle_inversion_ceiling_waveform_mse: float
    oracle_inversion_ceiling_waveform_asd: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate PaperRepro Stage-1 autoencoder.")
    parser.add_argument("--config", type=str, default=None, help="Frozen config path.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to Stage-1 checkpoint.")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda")
    parser.add_argument("--batch-size", type=int, default=32, help="Evaluation batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--max-samples-per-split", type=int, default=None, help="Optional primary metric cap per split.")
    parser.add_argument("--waveform-eval-samples", type=int, default=32, help="Waveform metric subset size per split.")
    parser.add_argument("--visual-samples-per-split", type=int, default=6, help="Samples per split for PNG grids.")
    parser.add_argument("--seed", type=int, default=42, help="Selection seed.")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional output directory.")
    return parser.parse_args()


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def detect_device(override: str | None) -> torch.device:
    if override:
        return torch.device(override)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def checkpoint_run_name(checkpoint_path: Path) -> str:
    return checkpoint_path.parents[1].name


def default_output_dir(checkpoint_path: Path) -> Path:
    run_name = checkpoint_run_name(checkpoint_path)
    return EXPERIMENT_ROOT / "results" / "stage1_eval" / run_name


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def output_markdown(summary: dict, path: Path) -> None:
    lines = [
        "# Stage-1 Evaluation",
        "",
        f"- run: `{summary['run_name']}`",
        f"- checkpoint: `{summary['checkpoint_path']}`",
        f"- device: `{summary['device']}`",
        f"- waveform subset per split: `{summary['waveform_eval_samples']}`",
        f"- visual samples per split: `{summary['visual_samples_per_split']}`",
        "",
        "## Metrics",
        "",
        "| split | n_primary | n_waveform | recon_loss | spec_corr | lsd | mr_lsd | waveform_mse | waveform_asd | oracle_waveform_mse | oracle_waveform_asd |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split_metrics in summary["splits"]:
        lines.append(
            "| {split} | {num_samples_primary} | {num_samples_waveform} | {reconstruction_loss:.6f} | {spec_corr:.6f} | {lsd:.6f} | {mr_lsd:.6f} | {waveform_mse:.6f} | {waveform_asd:.6f} | {oracle_inversion_ceiling_waveform_mse:.6f} | {oracle_inversion_ceiling_waveform_asd:.6f} |".format(
                **split_metrics
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Primary metrics are computed on the full selected split or the configured primary cap.",
            "- Waveform metrics use deterministic Griffin-Lim reconstruction on a capped subset because the inverse pass is expensive.",
            "- Oracle inversion ceiling is reported as an approximate Griffin-Lim reference, not a strict mathematical upper bound.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_model(cfg: dict, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_stage1_autoencoder_from_config(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def build_split_dataset(cfg: dict, split: str, max_samples: int | None) -> Subset | torch.utils.data.Dataset:
    dataset = build_stage1_dataset_from_config(cfg, splits=[split])
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    indices = list(range(max_samples))
    return Subset(dataset, indices)


def unwrap_dataset(dataset) -> tuple[object, list[int]]:
    if isinstance(dataset, Subset):
        return dataset.dataset, list(dataset.indices)
    return dataset, list(range(len(dataset)))


def denormalize_log_spectrogram(x: torch.Tensor, *, clip_min: float, log_max: float) -> torch.Tensor:
    log_clip = float(np.log(clip_min))
    return ((x + 1.0) * 0.5) * (log_max - log_clip) + log_clip


def normalized_to_magnitude(x: torch.Tensor, *, clip_min: float, log_max: float) -> torch.Tensor:
    return torch.exp(denormalize_log_spectrogram(x, clip_min=clip_min, log_max=log_max))


def pearson_corr_per_sample(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_flat = x.flatten(start_dim=1)
    y_flat = y.flatten(start_dim=1)
    x_center = x_flat - x_flat.mean(dim=1, keepdim=True)
    y_center = y_flat - y_flat.mean(dim=1, keepdim=True)
    numerator = torch.sum(x_center * y_center, dim=1)
    denominator = torch.sqrt(torch.sum(x_center.square(), dim=1) * torch.sum(y_center.square(), dim=1) + 1.0e-12)
    return numerator / denominator


def lsd_per_sample(x_log: torch.Tensor, y_log: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((x_log - y_log).square(), dim=(1, 2, 3)))


def avgpool_by_scale(x: torch.Tensor, scale: int) -> torch.Tensor:
    if scale == 1:
        return x
    return F.avg_pool2d(x, kernel_size=scale, stride=scale, ceil_mode=False)


def mr_lsd_per_sample(x_log: torch.Tensor, y_log: torch.Tensor, scales: Iterable[int] = (1, 2, 4)) -> torch.Tensor:
    values = []
    for scale in scales:
        x_scaled = avgpool_by_scale(x_log, scale)
        y_scaled = avgpool_by_scale(y_log, scale)
        values.append(lsd_per_sample(x_scaled, y_scaled))
    return torch.stack(values, dim=0).mean(dim=0)


def waveform_asd(target_waveform: np.ndarray, pred_waveform: np.ndarray) -> float:
    target_fft = np.fft.rfft(target_waveform, axis=-1)
    pred_fft = np.fft.rfft(pred_waveform, axis=-1)
    target_log = np.log(np.clip(np.abs(target_fft), 1.0e-8, None))
    pred_log = np.log(np.clip(np.abs(pred_fft), 1.0e-8, None))
    return float(np.sqrt(np.mean((target_log - pred_log) ** 2)))


def griffin_lim_torch(
    magnitude: torch.Tensor,
    *,
    n_fft: int,
    hop_length: int,
    num_iters: int,
    length: int,
) -> torch.Tensor:
    if magnitude.ndim != 3:
        raise ValueError(f"Expected [channels, freq, frames], got {magnitude.shape}")
    device = magnitude.device
    if magnitude.shape[1] == n_fft // 2:
        pad = torch.zeros((magnitude.shape[0], 1, magnitude.shape[2]), device=device, dtype=magnitude.dtype)
        magnitude = torch.cat([magnitude, pad], dim=1)

    window = torch.hann_window(n_fft, periodic=True, device=device, dtype=magnitude.dtype)
    phase = torch.exp(2j * math.pi * torch.rand(magnitude.shape, device=device))
    complex_spec = magnitude.to(torch.complex64) * phase

    for _ in range(num_iters):
        waveform = torch.istft(
            complex_spec,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            center=True,
            length=length,
        )
        rebuilt = torch.stft(
            waveform,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            center=True,
            pad_mode="constant",
            return_complex=True,
        )
        phase = rebuilt / torch.clamp(rebuilt.abs(), min=1.0e-8)
        complex_spec = magnitude.to(torch.complex64) * phase

    return torch.istft(
        complex_spec,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=True,
        length=length,
    )


def magnitude_bin_label(magnitude: float) -> str:
    if magnitude < 2.0:
        return "<2"
    if magnitude < 3.0:
        return "2-3"
    if magnitude < 4.0:
        return "3-4"
    return "4+"


def distance_bin_label(distance_km: float) -> str:
    if distance_km < 20.0:
        return "<20"
    if distance_km < 40.0:
        return "20-40"
    if distance_km < 60.0:
        return "40-60"
    return "60+"


def select_diverse_indices(rows: list[dict], count: int, seed: int) -> list[int]:
    if len(rows) <= count:
        return list(range(len(rows)))
    rng = random.Random(seed)
    by_key: dict[tuple[str, str], list[int]] = {}
    for index, row in enumerate(rows):
        key = (magnitude_bin_label(float(row["magnitude"])), distance_bin_label(float(row["hypocentral_distance_km"])))
        by_key.setdefault(key, []).append(index)
    chosen: list[int] = []
    for key in sorted(by_key):
        rng.shuffle(by_key[key])
        chosen.append(by_key[key][0])
        if len(chosen) == count:
            return chosen
    remaining = [idx for idx in range(len(rows)) if idx not in set(chosen)]
    rng.shuffle(remaining)
    chosen.extend(remaining[: max(0, count - len(chosen))])
    return chosen[:count]


def concat_component_spectrogram(x: np.ndarray) -> np.ndarray:
    return np.concatenate([x[i] for i in range(x.shape[0])], axis=0)


def plot_spectrogram_grid(samples: list[dict], path: Path, title: str) -> None:
    cols = len(samples)
    fig, axes = plt.subplots(2, cols, figsize=(4 * cols, 6), squeeze=False)
    for col, sample in enumerate(samples):
        target = concat_component_spectrogram(sample["target_repr"])
        recon = concat_component_spectrogram(sample["recon_repr"])
        for row, image, row_title in (
            (0, target, "target"),
            (1, recon, "oracle"),
        ):
            ax = axes[row][col]
            im = ax.imshow(image, origin="lower", aspect="auto", cmap="magma")
            ax.set_title(f"{row_title}\nM{sample['magnitude']:.1f} D{sample['distance_km']:.1f}km")
            ax.set_xlabel("frame")
            ax.set_ylabel("stacked freq")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_waveform_grid(samples: list[dict], path: Path, title: str) -> None:
    cols = len(samples)
    fig, axes = plt.subplots(2, cols, figsize=(4 * cols, 5), squeeze=False, sharex=False, sharey=False)
    component_colors = ["tab:blue", "tab:orange", "tab:green"]
    component_labels = ["E", "N", "Z"]
    for col, sample in enumerate(samples):
        for row, waveform, row_title in (
            (0, sample["target_waveform"], "target"),
            (1, sample["recon_waveform"], "oracle"),
        ):
            ax = axes[row][col]
            for channel in range(waveform.shape[0]):
                ax.plot(waveform[channel], color=component_colors[channel], alpha=0.85, linewidth=0.8, label=component_labels[channel])
            ax.set_title(f"{row_title}\nM{sample['magnitude']:.1f} D{sample['distance_km']:.1f}km")
            ax.set_xlabel("sample")
            ax.set_ylabel("counts")
            if row == 0 and col == 0:
                ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def evaluate_primary_metrics(
    model: torch.nn.Module,
    dataset,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    clip_min: float,
    log_max: float,
) -> dict[str, float]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    total = 0
    recon_sum = 0.0
    spec_corr_sum = 0.0
    lsd_sum = 0.0
    mr_lsd_sum = 0.0
    with torch.no_grad():
        for batch in loader:
            x = batch["representation"].to(device)
            mean, _ = model.encode_stats(x)
            reconstruction = model.decode(mean)
            recon_loss = torch.mean((x - reconstruction).square(), dim=(1, 2, 3))
            x_log = denormalize_log_spectrogram(x, clip_min=clip_min, log_max=log_max)
            recon_log = denormalize_log_spectrogram(reconstruction, clip_min=clip_min, log_max=log_max)
            spec_corr = pearson_corr_per_sample(x, reconstruction)
            lsd = lsd_per_sample(x_log, recon_log)
            mr_lsd = mr_lsd_per_sample(x_log, recon_log)
            batch_size_actual = x.shape[0]
            total += batch_size_actual
            recon_sum += float(recon_loss.sum().item())
            spec_corr_sum += float(spec_corr.sum().item())
            lsd_sum += float(lsd.sum().item())
            mr_lsd_sum += float(mr_lsd.sum().item())
    return {
        "num_samples_primary": total,
        "reconstruction_loss": recon_sum / max(total, 1),
        "spec_corr": spec_corr_sum / max(total, 1),
        "lsd": lsd_sum / max(total, 1),
        "mr_lsd": mr_lsd_sum / max(total, 1),
    }


def evaluate_waveform_subset(
    model: torch.nn.Module,
    base_dataset,
    indices: list[int],
    *,
    device: torch.device,
    cfg: dict,
    split: str,
) -> tuple[dict[str, float], list[dict]]:
    rep_cfg = cfg["representation"]
    n_fft = int(rep_cfg["n_fft"])
    hop_length = int(rep_cfg["hop_length"])
    clip_min = float(rep_cfg["normalization"]["clip_min"])
    log_max = float(rep_cfg["normalization"]["log_max"])
    gl_iters = int(rep_cfg["inverse"]["n_iter"])
    num_samples = int(cfg["data"]["window"]["num_samples"])
    sample_rate_hz = float(cfg["data"]["sample_rate_hz"])
    components = cfg["data"]["components"]
    pre_origin_sec = float(cfg["data"]["window"]["pre_origin_sec"])
    padding_value = float(cfg["data"]["window"]["padding_value"])

    waveform_mse_values: list[float] = []
    waveform_asd_values: list[float] = []
    oracle_waveform_mse_values: list[float] = []
    oracle_waveform_asd_values: list[float] = []
    visual_samples: list[dict] = []

    with torch.no_grad():
        for index in indices:
            row = base_dataset.rows[index]
            waveform_np, _ = load_origin_window(
                row["file_path"],
                origin_time_iso=row["origin_time"],
                num_samples=num_samples,
                sample_rate_hz=sample_rate_hz,
                components=components,
                pre_origin_sec=pre_origin_sec,
                padding_value=padding_value,
            )
            target_repr_np = base_dataset.representation.transform(waveform_np)
            target_repr = torch.from_numpy(target_repr_np).unsqueeze(0).to(device)
            mean, _ = model.encode_stats(target_repr)
            recon_repr = model.decode(mean).squeeze(0)
            target_mag = normalized_to_magnitude(target_repr.squeeze(0), clip_min=clip_min, log_max=log_max)
            recon_mag = normalized_to_magnitude(recon_repr, clip_min=clip_min, log_max=log_max)

            oracle_waveform = griffin_lim_torch(
                recon_mag,
                n_fft=n_fft,
                hop_length=hop_length,
                num_iters=gl_iters,
                length=num_samples,
            ).detach().cpu().numpy()
            ceiling_waveform = griffin_lim_torch(
                target_mag,
                n_fft=n_fft,
                hop_length=hop_length,
                num_iters=gl_iters,
                length=num_samples,
            ).detach().cpu().numpy()

            waveform_mse_values.append(float(np.mean((waveform_np - oracle_waveform) ** 2)))
            waveform_asd_values.append(waveform_asd(waveform_np, oracle_waveform))
            oracle_waveform_mse_values.append(float(np.mean((waveform_np - ceiling_waveform) ** 2)))
            oracle_waveform_asd_values.append(waveform_asd(waveform_np, ceiling_waveform))
            visual_samples.append(
                {
                    "split": split,
                    "event_id": row["event_id"],
                    "station_code": row["station_code"],
                    "magnitude": float(row["magnitude"]),
                    "distance_km": float(row["hypocentral_distance_km"]),
                    "target_repr": target_repr.squeeze(0).detach().cpu().numpy(),
                    "recon_repr": recon_repr.detach().cpu().numpy(),
                    "target_waveform": waveform_np,
                    "recon_waveform": oracle_waveform,
                    "ceiling_waveform": ceiling_waveform,
                }
            )

    return (
        {
            "num_samples_waveform": len(indices),
            "waveform_mse": float(np.mean(waveform_mse_values)) if waveform_mse_values else float("nan"),
            "waveform_asd": float(np.mean(waveform_asd_values)) if waveform_asd_values else float("nan"),
            "oracle_inversion_ceiling_waveform_mse": float(np.mean(oracle_waveform_mse_values)) if oracle_waveform_mse_values else float("nan"),
            "oracle_inversion_ceiling_waveform_asd": float(np.mean(oracle_waveform_asd_values)) if oracle_waveform_asd_values else float("nan"),
        },
        visual_samples,
    )


def main() -> None:
    args = parse_args()
    cfg = load_frozen_config(args.config)
    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else default_output_dir(checkpoint_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    config_snapshot_path = output_dir / "config_snapshot.yaml"
    if not config_snapshot_path.exists():
        shutil.copyfile(args.config or default_config_path(), config_snapshot_path)

    device = detect_device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    log(f"Loading Stage-1 checkpoint from {checkpoint_path}")
    model = load_model(cfg, checkpoint_path, device)

    summary = {
        "run_name": checkpoint_run_name(checkpoint_path),
        "checkpoint_path": str(checkpoint_path),
        "device": str(device),
        "waveform_eval_samples": int(args.waveform_eval_samples),
        "visual_samples_per_split": int(args.visual_samples_per_split),
        "splits": [],
    }

    for split in ("test", "ood"):
        log(f"Evaluating split={split}")
        dataset = build_split_dataset(cfg, split, args.max_samples_per_split)
        base_dataset, indices = unwrap_dataset(dataset)
        clip_min = float(cfg["representation"]["normalization"]["clip_min"])
        log_max = float(cfg["representation"]["normalization"]["log_max"])

        primary = evaluate_primary_metrics(
            model,
            dataset,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            device=device,
            clip_min=clip_min,
            log_max=log_max,
        )

        candidate_rows = [base_dataset.rows[idx] for idx in indices]
        waveform_local_indices = select_diverse_indices(candidate_rows, min(int(args.waveform_eval_samples), len(candidate_rows)), args.seed + len(split))
        waveform_global_indices = [indices[idx] for idx in waveform_local_indices]
        waveform_metrics, visual_samples = evaluate_waveform_subset(
            model,
            base_dataset,
            waveform_global_indices,
            device=device,
            cfg=cfg,
            split=split,
        )

        visuals = visual_samples[: min(len(visual_samples), int(args.visual_samples_per_split))]
        plot_spectrogram_grid(visuals, figures_dir / f"{split}_spectrogram_grid.png", title=f"Stage-1 {split} target vs oracle spectrogram")
        plot_waveform_grid(visuals, figures_dir / f"{split}_waveform_grid.png", title=f"Stage-1 {split} target vs oracle waveform")

        split_metrics = SplitMetrics(
            split=split,
            num_samples_primary=int(primary["num_samples_primary"]),
            num_samples_waveform=int(waveform_metrics["num_samples_waveform"]),
            reconstruction_loss=float(primary["reconstruction_loss"]),
            spec_corr=float(primary["spec_corr"]),
            lsd=float(primary["lsd"]),
            mr_lsd=float(primary["mr_lsd"]),
            waveform_mse=float(waveform_metrics["waveform_mse"]),
            waveform_asd=float(waveform_metrics["waveform_asd"]),
            oracle_inversion_ceiling_waveform_mse=float(waveform_metrics["oracle_inversion_ceiling_waveform_mse"]),
            oracle_inversion_ceiling_waveform_asd=float(waveform_metrics["oracle_inversion_ceiling_waveform_asd"]),
        )
        payload = asdict(split_metrics)
        summary["splits"].append(payload)
        save_json(output_dir / f"metrics_{split}.json", payload)
        log(
            f"split={split} recon={split_metrics.reconstruction_loss:.6f} "
            f"spec_corr={split_metrics.spec_corr:.6f} "
            f"lsd={split_metrics.lsd:.6f} "
            f"waveform_mse={split_metrics.waveform_mse:.6f}"
        )

    save_json(output_dir / "summary.json", summary)
    output_markdown(summary, output_dir / "summary.md")
    log(f"Stage-1 evaluation finished. Artifacts written under {output_dir}")


if __name__ == "__main__":
    main()
