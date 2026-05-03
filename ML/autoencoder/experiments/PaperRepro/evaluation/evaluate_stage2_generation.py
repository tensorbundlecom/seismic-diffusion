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

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from core.binning import (
    bin_index,
    bin_label,
    distance_bin_edges_from_config,
    joint_class_index,
    magnitude_bin_edges_from_config,
    num_joint_classes,
)
from core.edm import Stage2EDM, heun_sampler
from core.frozen_config import default_config_path, load_frozen_config
from core.latent_cache_dataset import Stage2LatentCacheDataset, build_station_mapping
from core.paper_metrics_classifier import build_classifier_from_config
from core.stage1_autoencoder import build_stage1_autoencoder_from_config
from core.stage2_unet import build_stage2_unet_from_config
from evaluation.evaluate_stage1_autoencoder import (
    denormalize_log_spectrogram,
    griffin_lim_torch,
    lsd_per_sample,
    mr_lsd_per_sample,
    normalized_to_magnitude,
    pearson_corr_per_sample,
    waveform_asd,
)
from setup.windowing import load_origin_window


@dataclass
class SplitMetrics:
    split: str
    num_samples_primary: int
    num_samples_waveform: int
    spec_corr: float
    lsd: float
    mr_lsd: float
    envelope_similarity: float
    fourier_amplitude_fd: float
    waveform_mse: float
    waveform_asd: float
    oracle_inversion_ceiling_waveform_mse: float
    oracle_inversion_ceiling_waveform_asd: float
    classifier_accuracy: float
    classifier_embedding_fd: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate PaperRepro Stage-2 generation.")
    parser.add_argument("--config", type=str, default=None, help="Frozen config path.")
    parser.add_argument("--stage1-checkpoint", type=str, required=True, help="Stage-1 checkpoint path.")
    parser.add_argument("--stage2-checkpoint", type=str, required=True, help="Stage-2 checkpoint path.")
    parser.add_argument("--cache-root", type=str, required=True, help="Stage-2 cache root.")
    parser.add_argument("--classifier-checkpoint", type=str, required=True, help="Classifier checkpoint path.")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda.")
    parser.add_argument("--batch-size", type=int, default=32, help="Generation eval batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="Workers for cache loading.")
    parser.add_argument("--waveform-eval-samples", type=int, default=32, help="Waveform subset size per split.")
    parser.add_argument("--visual-samples-per-split", type=int, default=6, help="Visual samples per split.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional output dir.")
    parser.add_argument("--num-steps", type=int, default=None, help="Override Heun steps.")
    return parser.parse_args()


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def detect_device(override: str | None) -> torch.device:
    if override:
        return torch.device(override)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def default_output_dir(stage2_checkpoint: Path) -> Path:
    run_name = stage2_checkpoint.parents[1].name
    return EXPERIMENT_ROOT / "results" / "stage2_eval" / run_name


def checkpoint_run_name(checkpoint_path: Path) -> str:
    return checkpoint_path.parents[1].name


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


def load_classifier(cfg: dict, checkpoint_path: Path, device: torch.device):
    num_classes = num_joint_classes(
        magnitude_bin_edges_from_config(cfg),
        distance_bin_edges_from_config(cfg),
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_classifier_from_config(cfg, num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def output_markdown(summary: dict, path: Path) -> None:
    lines = [
        "# Stage-2 Evaluation",
        "",
        f"- run: `{summary['run_name']}`",
        f"- stage1 checkpoint: `{summary['stage1_checkpoint']}`",
        f"- stage2 checkpoint: `{summary['stage2_checkpoint']}`",
        f"- classifier checkpoint: `{summary['classifier_checkpoint']}`",
        f"- cache root: `{summary['cache_root']}`",
        f"- device: `{summary['device']}`",
        f"- waveform subset per split: `{summary['waveform_eval_samples']}`",
        f"- visual samples per split: `{summary['visual_samples_per_split']}`",
        f"- sampling steps: `{summary['num_steps']}`",
        "",
        "## Metrics",
        "",
        "| split | n_primary | n_waveform | spec_corr | lsd | mr_lsd | envelope_similarity | fourier_amplitude_fd | waveform_mse | waveform_asd | oracle_waveform_mse | oracle_waveform_asd | classifier_accuracy | classifier_embedding_fd |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split_metrics in summary["splits"]:
        lines.append(
            "| {split} | {num_samples_primary} | {num_samples_waveform} | {spec_corr:.6f} | {lsd:.6f} | {mr_lsd:.6f} | {envelope_similarity:.6f} | {fourier_amplitude_fd:.6f} | {waveform_mse:.6f} | {waveform_asd:.6f} | {oracle_inversion_ceiling_waveform_mse:.6f} | {oracle_inversion_ceiling_waveform_asd:.6f} | {classifier_accuracy:.6f} | {classifier_embedding_fd:.6f} |".format(
                **split_metrics
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Primary spectrogram metrics and classifier metrics are computed over the full split.",
            "- Waveform metrics, Fourier amplitude FD, envelope similarity, and visualizations use a capped subset with Griffin-Lim inversion.",
            "- Bin-wise tables are written separately as `bin_metrics_<split>.json` and `bin_metrics_<split>.md`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def output_bin_markdown(path: Path, payload: dict, split: str) -> None:
    lines = [
        f"# Stage-2 {split} Bin Metrics",
        "",
        "## Magnitude Bins",
        "",
        "| bin | count_primary | count_waveform | spec_corr | lsd | mr_lsd | envelope_similarity | fourier_amplitude_fd | classifier_accuracy | classifier_embedding_fd |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["magnitude_bins"]:
        lines.append(
            "| {label} | {count_primary} | {count_waveform} | {spec_corr:.6f} | {lsd:.6f} | {mr_lsd:.6f} | {envelope_similarity:.6f} | {fourier_amplitude_fd:.6f} | {classifier_accuracy:.6f} | {classifier_embedding_fd:.6f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Distance Bins",
            "",
            "| bin | count_primary | count_waveform | spec_corr | lsd | mr_lsd | envelope_similarity | fourier_amplitude_fd | classifier_accuracy | classifier_embedding_fd |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["distance_bins"]:
        lines.append(
            "| {label} | {count_primary} | {count_waveform} | {spec_corr:.6f} | {lsd:.6f} | {mr_lsd:.6f} | {envelope_similarity:.6f} | {fourier_amplitude_fd:.6f} | {classifier_accuracy:.6f} | {classifier_embedding_fd:.6f} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def select_diverse_indices(meta: dict, count: int, seed: int, *, magnitude_edges: list[float], distance_edges: list[float]) -> list[int]:
    total = len(meta["event_id"])
    if total <= count:
        return list(range(total))
    rng = random.Random(seed)
    by_key: dict[tuple[str, str], list[int]] = {}
    for index in range(total):
        mag_idx = bin_index(float(meta["magnitude"][index]), magnitude_edges)
        dist_idx = bin_index(float(meta["hypocentral_distance_km"][index]), distance_edges)
        key = (
            bin_label(magnitude_edges, mag_idx) if mag_idx >= 0 else "out",
            bin_label(distance_edges, dist_idx) if dist_idx >= 0 else "out",
        )
        by_key.setdefault(key, []).append(index)
    chosen: list[int] = []
    chosen_set: set[int] = set()
    for key in sorted(by_key):
        rng.shuffle(by_key[key])
        candidate = by_key[key][0]
        chosen.append(candidate)
        chosen_set.add(candidate)
        if len(chosen) == count:
            return chosen
    remaining = [idx for idx in range(total) if idx not in chosen_set]
    rng.shuffle(remaining)
    chosen.extend(remaining[: max(0, count - len(chosen))])
    return chosen[:count]


def select_bin_representatives(meta: dict, edges: list[float], field: str, *, seed: int) -> list[int]:
    rng = random.Random(seed)
    by_bin: dict[int, list[int]] = {}
    for index in range(len(meta["event_id"])):
        bin_idx = bin_index(float(meta[field][index]), edges)
        if bin_idx < 0:
            continue
        by_bin.setdefault(bin_idx, []).append(index)
    chosen: list[int] = []
    for bin_idx in range(len(edges) - 1):
        candidates = by_bin.get(bin_idx, [])
        if not candidates:
            continue
        rng.shuffle(candidates)
        chosen.append(candidates[0])
    return chosen


def concat_component_spectrogram(x: np.ndarray) -> np.ndarray:
    return np.concatenate([x[i] for i in range(x.shape[0])], axis=0)


def plot_spectrogram_grid(samples: list[dict], path: Path, title: str, key: str | None = None) -> None:
    cols = len(samples)
    fig, axes = plt.subplots(2, cols, figsize=(4 * cols, 6), squeeze=False)
    for col, sample in enumerate(samples):
        target = concat_component_spectrogram(sample["target_repr"])
        generated = concat_component_spectrogram(sample["generated_repr"])
        for row, image, row_title in (
            (0, target, "target"),
            (1, generated, "generated"),
        ):
            ax = axes[row][col]
            im = ax.imshow(image, origin="lower", aspect="auto", cmap="magma")
            suffix = sample[key] if key else f"M{sample['magnitude']:.1f} D{sample['distance_km']:.1f}km"
            ax.set_title(f"{row_title}\n{suffix}")
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
    fig, axes = plt.subplots(2, cols, figsize=(4 * cols, 5), squeeze=False)
    component_colors = ["tab:blue", "tab:orange", "tab:green"]
    component_labels = ["E", "N", "Z"]
    for col, sample in enumerate(samples):
        for row, waveform, row_title in (
            (0, sample["target_waveform"], "target"),
            (1, sample["generated_waveform"], "generated"),
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


def plot_envelope_distributions(target_env: np.ndarray, generated_env: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    max_value = max(float(target_env.max()), float(generated_env.max()), 1.0e-8)
    bins = np.linspace(0.0, max_value, 80)
    ax.hist(target_env, bins=bins, alpha=0.5, density=True, label="target")
    ax.hist(generated_env, bins=bins, alpha=0.5, density=True, label="generated")
    ax.set_title(title)
    ax.set_xlabel("mean envelope amplitude")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_fourier_distributions(target_log_fft: np.ndarray, generated_log_fft: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(target_log_fft.mean(axis=0), label="target")
    ax.plot(generated_log_fft.mean(axis=0), label="generated")
    ax.set_title(title)
    ax.set_xlabel("frequency bin")
    ax.set_ylabel("mean log amplitude")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def compute_envelope_similarity(target_waveform: np.ndarray, generated_waveform: np.ndarray) -> float:
    target_env = np.mean(np.abs(target_waveform), axis=0)
    generated_env = np.mean(np.abs(generated_waveform), axis=0)
    t = target_env - target_env.mean()
    g = generated_env - generated_env.mean()
    denom = math.sqrt(float(np.sum(t ** 2) * np.sum(g ** 2)) + 1.0e-12)
    return float(np.sum(t * g) / denom)


def frechet_distance(x: np.ndarray, y: np.ndarray, eps: float = 1.0e-6) -> float:
    from scipy import linalg

    mu_x = x.mean(axis=0)
    mu_y = y.mean(axis=0)
    cov_x = np.cov(x, rowvar=False)
    cov_y = np.cov(y, rowvar=False)
    covmean, _ = linalg.sqrtm(cov_x @ cov_y, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(cov_x.shape[0]) * eps
        covmean = linalg.sqrtm((cov_x + offset) @ (cov_y + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(np.sum((mu_x - mu_y) ** 2) + np.trace(cov_x) + np.trace(cov_y) - 2 * np.trace(covmean))


def compute_fourier_amplitude_fd(target_waveform: np.ndarray, generated_waveform: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    target_fft = np.abs(np.fft.rfft(target_waveform, axis=-1))
    generated_fft = np.abs(np.fft.rfft(generated_waveform, axis=-1))
    target_log = np.log(np.clip(target_fft, 1.0e-8, None)).reshape(target_fft.shape[0], -1)
    generated_log = np.log(np.clip(generated_fft, 1.0e-8, None)).reshape(generated_fft.shape[0], -1)
    fd = frechet_distance(generated_log, target_log)
    return fd, target_log, generated_log


def generate_latents(model: Stage2EDM, cond: torch.Tensor, latent_shape: tuple[int, ...], device: torch.device, cfg: dict, num_steps: int, station_index: torch.Tensor | None = None) -> torch.Tensor:
    edm_cfg = cfg["stage2"]["edm"]
    return heun_sampler(
        model,
        cond=cond,
        shape=(cond.shape[0], *latent_shape),
        station_index=station_index,
        sigma_min=float(edm_cfg["sigma_min"]),
        sigma_max=float(edm_cfg["sigma_max"]),
        rho=float(edm_cfg["rho"]),
        num_steps=num_steps,
        device=device,
    )


def classifier_forward_batches(classifier_model, representations: np.ndarray, device: torch.device, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    embeddings = []
    logits = []
    with torch.no_grad():
        for start in range(0, len(representations), batch_size):
            batch = torch.from_numpy(representations[start : start + batch_size]).to(device)
            emb = classifier_model.embed(batch)
            out = classifier_model.output_layer(emb)
            embeddings.append(emb.detach().cpu().numpy())
            logits.append(out.detach().cpu().numpy())
    return np.concatenate(embeddings, axis=0), np.concatenate(logits, axis=0)


def aggregate_primary_metrics(indices: list[int], *, spec_corr: np.ndarray, lsd: np.ndarray, mr_lsd: np.ndarray, classifier_accuracy: np.ndarray, target_emb: np.ndarray, generated_emb: np.ndarray) -> dict[str, float]:
    if not indices:
        return {
            "count_primary": 0,
            "spec_corr": float("nan"),
            "lsd": float("nan"),
            "mr_lsd": float("nan"),
            "classifier_accuracy": float("nan"),
            "classifier_embedding_fd": float("nan"),
        }
    return {
        "count_primary": len(indices),
        "spec_corr": float(np.mean(spec_corr[indices])),
        "lsd": float(np.mean(lsd[indices])),
        "mr_lsd": float(np.mean(mr_lsd[indices])),
        "classifier_accuracy": float(np.mean(classifier_accuracy[indices])),
        "classifier_embedding_fd": float(frechet_distance(generated_emb[indices], target_emb[indices])) if len(indices) >= 2 else float("nan"),
    }


def aggregate_waveform_metrics(indices: list[int], *, envelope_values: list[float], target_waveforms: list[np.ndarray], generated_waveforms: list[np.ndarray]) -> dict[str, float]:
    if not indices:
        return {
            "count_waveform": 0,
            "envelope_similarity": float("nan"),
            "fourier_amplitude_fd": float("nan"),
        }
    target = np.stack([target_waveforms[i] for i in indices], axis=0)
    generated = np.stack([generated_waveforms[i] for i in indices], axis=0)
    fourier_fd, _, _ = compute_fourier_amplitude_fd(target, generated)
    return {
        "count_waveform": len(indices),
        "envelope_similarity": float(np.mean([envelope_values[i] for i in indices])),
        "fourier_amplitude_fd": float(fourier_fd),
    }


def build_bin_tables(
    *,
    meta: dict,
    waveform_meta: list[dict],
    spec_corr_values: np.ndarray,
    lsd_values: np.ndarray,
    mr_lsd_values: np.ndarray,
    classifier_accuracy_values: np.ndarray,
    target_embeddings: np.ndarray,
    generated_embeddings: np.ndarray,
    waveform_envelope_values: list[float],
    target_waveforms: list[np.ndarray],
    generated_waveforms: list[np.ndarray],
    magnitude_edges: list[float],
    distance_edges: list[float],
) -> dict:
    magnitude_rows = []
    for idx in range(len(magnitude_edges) - 1):
        primary_indices = [
            i for i in range(len(spec_corr_values))
            if bin_index(float(meta["magnitude"][i]), magnitude_edges) == idx
        ]
        waveform_indices = [
            i for i, row in enumerate(waveform_meta)
            if bin_index(float(row["magnitude"]), magnitude_edges) == idx
        ]
        row = {"label": bin_label(magnitude_edges, idx)}
        row.update(
            aggregate_primary_metrics(
                primary_indices,
                spec_corr=spec_corr_values,
                lsd=lsd_values,
                mr_lsd=mr_lsd_values,
                classifier_accuracy=classifier_accuracy_values,
                target_emb=target_embeddings,
                generated_emb=generated_embeddings,
            )
        )
        row.update(
            aggregate_waveform_metrics(
                waveform_indices,
                envelope_values=waveform_envelope_values,
                target_waveforms=target_waveforms,
                generated_waveforms=generated_waveforms,
            )
        )
        magnitude_rows.append(row)

    distance_rows = []
    for idx in range(len(distance_edges) - 1):
        primary_indices = [
            i for i in range(len(spec_corr_values))
            if bin_index(float(meta["hypocentral_distance_km"][i]), distance_edges) == idx
        ]
        waveform_indices = [
            i for i, row in enumerate(waveform_meta)
            if bin_index(float(row["distance_km"]), distance_edges) == idx
        ]
        row = {"label": bin_label(distance_edges, idx)}
        row.update(
            aggregate_primary_metrics(
                primary_indices,
                spec_corr=spec_corr_values,
                lsd=lsd_values,
                mr_lsd=mr_lsd_values,
                classifier_accuracy=classifier_accuracy_values,
                target_emb=target_embeddings,
                generated_emb=generated_embeddings,
            )
        )
        row.update(
            aggregate_waveform_metrics(
                waveform_indices,
                envelope_values=waveform_envelope_values,
                target_waveforms=target_waveforms,
                generated_waveforms=generated_waveforms,
            )
        )
        distance_rows.append(row)
    return {
        "magnitude_bins": magnitude_rows,
        "distance_bins": distance_rows,
    }


def evaluate_split(
    *,
    split: str,
    cache_path: Path,
    stage1_model,
    stage2_model: Stage2EDM,
    classifier_model,
    cfg: dict,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    num_steps: int,
    waveform_eval_samples: int,
    visual_samples_per_split: int,
    output_dir: Path,
    seed: int,
) -> dict:
    station_mapping = build_station_mapping(cache_path.parent)
    dataset = Stage2LatentCacheDataset(cache_path, station_mapping=station_mapping)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    latent_shape = tuple(int(v) for v in cfg["stage2"]["latent_input_shape"])
    clip_min = float(cfg["representation"]["normalization"]["clip_min"])
    log_max = float(cfg["representation"]["normalization"]["log_max"])
    rep_cfg = cfg["representation"]
    num_samples = int(cfg["data"]["window"]["num_samples"])
    sample_rate_hz = float(cfg["data"]["sample_rate_hz"])
    components = cfg["data"]["components"]
    pre_origin_sec = float(cfg["data"]["window"]["pre_origin_sec"])
    padding_value = float(cfg["data"]["window"]["padding_value"])
    magnitude_edges = magnitude_bin_edges_from_config(cfg)
    distance_edges = distance_bin_edges_from_config(cfg)

    spec_corr_sum = 0.0
    lsd_sum = 0.0
    mr_lsd_sum = 0.0
    total = 0
    generated_repr_all = []
    target_repr_all = []
    spec_corr_values: list[float] = []
    lsd_values: list[float] = []
    mr_lsd_values: list[float] = []

    log(f"Evaluating split={split}")
    with torch.no_grad():
        for batch in loader:
            cond = batch["condition_normalized"].to(device)
            station_index = batch["station_index"].to(device)
            generated_latent = generate_latents(stage2_model, cond, latent_shape, device, cfg, num_steps, station_index=station_index)
            generated_repr = stage1_model.decode(generated_latent)
            target_repr = stage1_model.decode(batch["latent"].to(device))
            generated_repr_all.append(generated_repr.detach().cpu())
            target_repr_all.append(target_repr.detach().cpu())

            g_log = denormalize_log_spectrogram(generated_repr, clip_min=clip_min, log_max=log_max)
            t_log = denormalize_log_spectrogram(target_repr, clip_min=clip_min, log_max=log_max)
            spec_corr = pearson_corr_per_sample(generated_repr, target_repr)
            lsd = lsd_per_sample(g_log, t_log)
            mr_lsd = mr_lsd_per_sample(g_log, t_log)

            n = generated_repr.shape[0]
            total += n
            spec_corr_sum += float(spec_corr.sum().item())
            lsd_sum += float(lsd.sum().item())
            mr_lsd_sum += float(mr_lsd.sum().item())
            spec_corr_values.extend(spec_corr.detach().cpu().numpy().tolist())
            lsd_values.extend(lsd.detach().cpu().numpy().tolist())
            mr_lsd_values.extend(mr_lsd.detach().cpu().numpy().tolist())

    generated_repr_cat = torch.cat(generated_repr_all, dim=0).numpy().astype(np.float32, copy=False)
    target_repr_cat = torch.cat(target_repr_all, dim=0).numpy().astype(np.float32, copy=False)
    classifier_embedding_generated, classifier_logits_generated = classifier_forward_batches(
        classifier_model, generated_repr_cat, device, batch_size
    )
    classifier_embedding_target, classifier_logits_target = classifier_forward_batches(
        classifier_model, target_repr_cat, device, batch_size
    )
    classifier_labels = np.asarray(
        [
            joint_class_index(
                float(dataset.meta["magnitude"][i]),
                float(dataset.meta["hypocentral_distance_km"][i]),
                magnitude_edges,
                distance_edges,
            )
            for i in range(len(dataset))
        ],
        dtype=np.int64,
    )
    classifier_pred_generated = np.argmax(classifier_logits_generated, axis=1)
    classifier_accuracy_values = (classifier_pred_generated == classifier_labels).astype(np.float64)
    classifier_accuracy = float(np.mean(classifier_accuracy_values))
    classifier_embedding_fd = float(
        frechet_distance(classifier_embedding_generated, classifier_embedding_target)
    )

    chosen = select_diverse_indices(
        dataset.meta,
        min(waveform_eval_samples, len(dataset)),
        seed + len(split),
        magnitude_edges=magnitude_edges,
        distance_edges=distance_edges,
    )
    metric_indices = set(chosen)
    overview_visuals = chosen[: min(visual_samples_per_split, len(chosen))]
    magnitude_visuals = select_bin_representatives(
        dataset.meta,
        magnitude_edges,
        "magnitude",
        seed=seed + len(split) + 100,
    )
    distance_visuals = select_bin_representatives(
        dataset.meta,
        distance_edges,
        "hypocentral_distance_km",
        seed=seed + len(split) + 200,
    )
    visual_indices = []
    visual_seen: set[int] = set()
    for idx in overview_visuals + magnitude_visuals + distance_visuals + chosen:
        if idx in visual_seen:
            continue
        visual_indices.append(idx)
        visual_seen.add(idx)
    waveform_mse_values = []
    waveform_asd_values = []
    oracle_waveform_mse_values = []
    oracle_waveform_asd_values = []
    envelope_values = []
    target_waveforms = []
    generated_waveforms = []
    waveform_meta = []
    spectrogram_visual_samples = []
    magnitude_visual_samples = []
    distance_visual_samples = []

    with torch.no_grad():
        for idx in visual_indices:
            meta = {k: dataset.meta[k][idx] for k in dataset.meta}
            waveform_np, _ = load_origin_window(
                meta["file_path"],
                origin_time_iso=meta["origin_time"],
                num_samples=num_samples,
                sample_rate_hz=sample_rate_hz,
                components=components,
                pre_origin_sec=pre_origin_sec,
                padding_value=padding_value,
            )

            target_repr = torch.from_numpy(target_repr_cat[idx]).to(device)
            generated_repr = torch.from_numpy(generated_repr_cat[idx]).to(device)
            target_mag = normalized_to_magnitude(target_repr, clip_min=clip_min, log_max=log_max)
            generated_mag = normalized_to_magnitude(generated_repr, clip_min=clip_min, log_max=log_max)

            target_waveform = griffin_lim_torch(
                target_mag,
                n_fft=int(rep_cfg["n_fft"]),
                hop_length=int(rep_cfg["hop_length"]),
                num_iters=int(rep_cfg["inverse"]["n_iter"]),
                length=num_samples,
            ).detach().cpu().numpy()
            generated_waveform = griffin_lim_torch(
                generated_mag,
                n_fft=int(rep_cfg["n_fft"]),
                hop_length=int(rep_cfg["hop_length"]),
                num_iters=int(rep_cfg["inverse"]["n_iter"]),
                length=num_samples,
            ).detach().cpu().numpy()

            mag_idx = bin_index(float(meta["magnitude"]), magnitude_edges)
            dist_idx = bin_index(float(meta["hypocentral_distance_km"]), distance_edges)
            mag_label = bin_label(magnitude_edges, mag_idx) if mag_idx >= 0 else "out"
            dist_label = bin_label(distance_edges, dist_idx) if dist_idx >= 0 else "out"

            sample_payload = {
                "magnitude": float(meta["magnitude"]),
                "distance_km": float(meta["hypocentral_distance_km"]),
                "magnitude_bin": mag_label,
                "distance_bin": dist_label,
                "target_repr": target_repr.cpu().numpy(),
                "generated_repr": generated_repr.cpu().numpy(),
                "target_waveform": waveform_np,
                "generated_waveform": generated_waveform,
            }

            if idx in metric_indices:
                waveform_mse_values.append(float(np.mean((waveform_np - generated_waveform) ** 2)))
                waveform_asd_values.append(waveform_asd(waveform_np, generated_waveform))
                oracle_waveform_mse_values.append(float(np.mean((waveform_np - target_waveform) ** 2)))
                oracle_waveform_asd_values.append(waveform_asd(waveform_np, target_waveform))
                envelope_values.append(compute_envelope_similarity(waveform_np, generated_waveform))
                target_waveforms.append(waveform_np)
                generated_waveforms.append(generated_waveform)
                waveform_meta.append(
                    {
                        "magnitude": float(meta["magnitude"]),
                        "distance_km": float(meta["hypocentral_distance_km"]),
                        "magnitude_bin": mag_label,
                        "distance_bin": dist_label,
                    }
                )

            if idx in overview_visuals:
                spectrogram_visual_samples.append(sample_payload)
            if idx in magnitude_visuals:
                magnitude_visual_samples.append(sample_payload)
            if idx in distance_visuals:
                distance_visual_samples.append(sample_payload)

    target_waveforms_arr = np.stack(target_waveforms, axis=0)
    generated_waveforms_arr = np.stack(generated_waveforms, axis=0)
    fourier_fd, target_log_fft, generated_log_fft = compute_fourier_amplitude_fd(target_waveforms_arr, generated_waveforms_arr)

    plot_spectrogram_grid(
        spectrogram_visual_samples,
        output_dir / "figures" / f"{split}_spectrogram_grid.png",
        f"Stage-2 {split} target vs generated spectrogram",
    )
    plot_waveform_grid(
        spectrogram_visual_samples,
        output_dir / "figures" / f"{split}_waveform_grid.png",
        f"Stage-2 {split} target vs generated waveform",
    )
    plot_spectrogram_grid(
        magnitude_visual_samples,
        output_dir / "figures" / f"{split}_magnitude_bin_grid.png",
        f"Stage-2 {split} representative magnitude-bin examples",
        key="magnitude_bin",
    )
    plot_spectrogram_grid(
        distance_visual_samples,
        output_dir / "figures" / f"{split}_distance_bin_grid.png",
        f"Stage-2 {split} representative distance-bin examples",
        key="distance_bin",
    )
    plot_envelope_distributions(
        np.mean(np.abs(target_waveforms_arr), axis=1).reshape(-1),
        np.mean(np.abs(generated_waveforms_arr), axis=1).reshape(-1),
        output_dir / "figures" / f"{split}_envelope_distribution.png",
        f"Stage-2 {split} envelope distribution",
    )
    plot_fourier_distributions(
        target_log_fft,
        generated_log_fft,
        output_dir / "figures" / f"{split}_fourier_distribution.png",
        f"Stage-2 {split} Fourier amplitude distribution",
    )

    bin_payload = build_bin_tables(
        meta=dataset.meta,
        waveform_meta=waveform_meta,
        spec_corr_values=np.asarray(spec_corr_values, dtype=np.float64),
        lsd_values=np.asarray(lsd_values, dtype=np.float64),
        mr_lsd_values=np.asarray(mr_lsd_values, dtype=np.float64),
        classifier_accuracy_values=classifier_accuracy_values,
        target_embeddings=classifier_embedding_target,
        generated_embeddings=classifier_embedding_generated,
        waveform_envelope_values=envelope_values,
        target_waveforms=target_waveforms,
        generated_waveforms=generated_waveforms,
        magnitude_edges=magnitude_edges,
        distance_edges=distance_edges,
    )
    save_json(output_dir / f"bin_metrics_{split}.json", bin_payload)
    output_bin_markdown(output_dir / f"bin_metrics_{split}.md", bin_payload, split)

    split_metrics = SplitMetrics(
        split=split,
        num_samples_primary=total,
        num_samples_waveform=len(chosen),
        spec_corr=spec_corr_sum / max(total, 1),
        lsd=lsd_sum / max(total, 1),
        mr_lsd=mr_lsd_sum / max(total, 1),
        envelope_similarity=float(np.mean(envelope_values)) if envelope_values else float("nan"),
        fourier_amplitude_fd=float(fourier_fd),
        waveform_mse=float(np.mean(waveform_mse_values)) if waveform_mse_values else float("nan"),
        waveform_asd=float(np.mean(waveform_asd_values)) if waveform_asd_values else float("nan"),
        oracle_inversion_ceiling_waveform_mse=float(np.mean(oracle_waveform_mse_values)) if oracle_waveform_mse_values else float("nan"),
        oracle_inversion_ceiling_waveform_asd=float(np.mean(oracle_waveform_asd_values)) if oracle_waveform_asd_values else float("nan"),
        classifier_accuracy=classifier_accuracy,
        classifier_embedding_fd=classifier_embedding_fd,
    )
    return asdict(split_metrics)


def main() -> None:
    args = parse_args()
    cfg = load_frozen_config(args.config)
    stage1_checkpoint = Path(args.stage1_checkpoint).resolve()
    stage2_checkpoint = Path(args.stage2_checkpoint).resolve()
    classifier_checkpoint = Path(args.classifier_checkpoint).resolve()
    cache_root = Path(args.cache_root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else default_output_dir(stage2_checkpoint)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config or default_config_path(), output_dir / "config_snapshot.yaml")

    device = detect_device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    num_steps = int(args.num_steps or cfg["stage2"]["sampling"]["num_steps"])
    stage1_model = load_stage1(cfg, stage1_checkpoint, device)
    station_mapping = build_station_mapping(cache_root)
    stage2_model = load_stage2(cfg, stage2_checkpoint, device, num_stations=len(station_mapping))
    classifier_model = load_classifier(cfg, classifier_checkpoint, device)

    summary = {
        "run_name": checkpoint_run_name(stage2_checkpoint),
        "stage1_checkpoint": str(stage1_checkpoint),
        "stage2_checkpoint": str(stage2_checkpoint),
        "classifier_checkpoint": str(classifier_checkpoint),
        "cache_root": str(cache_root),
        "device": str(device),
        "waveform_eval_samples": int(args.waveform_eval_samples),
        "visual_samples_per_split": int(args.visual_samples_per_split),
        "num_steps": num_steps,
        "splits": [],
    }

    for split_name, cache_name in (("test", "test_latent_cache.pt"), ("ood", "ood_latent_cache.pt")):
        metrics = evaluate_split(
            split=split_name,
            cache_path=cache_root / cache_name,
            stage1_model=stage1_model,
            stage2_model=stage2_model,
            classifier_model=classifier_model,
            cfg=cfg,
            device=device,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            num_steps=num_steps,
            waveform_eval_samples=int(args.waveform_eval_samples),
            visual_samples_per_split=int(args.visual_samples_per_split),
            output_dir=output_dir,
            seed=int(args.seed),
        )
        summary["splits"].append(metrics)
        save_json(output_dir / f"metrics_{split_name}.json", metrics)
        log(
            f"split={split_name} spec_corr={metrics['spec_corr']:.6f} "
            f"lsd={metrics['lsd']:.6f} "
            f"envelope={metrics['envelope_similarity']:.6f} "
            f"fourier_fd={metrics['fourier_amplitude_fd']:.6f} "
            f"classifier_acc={metrics['classifier_accuracy']:.6f}"
        )

    save_json(output_dir / "summary.json", summary)
    output_markdown(summary, output_dir / "summary.md")
    log(f"Stage-2 evaluation finished. Artifacts written under {output_dir}")


if __name__ == "__main__":
    main()
