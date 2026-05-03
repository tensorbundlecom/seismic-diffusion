from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from core.conditions import build_condition_vector, load_condition_norm_stats
from core.datasets import build_stage1_dataset_from_config
from core.frozen_config import default_config_path, load_frozen_config
from core.stage1_autoencoder import build_stage1_autoencoder_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Build Stage-2 latent cache from trained Stage-1 autoencoder.")
    parser.add_argument("--config", type=str, default=None, help="Frozen config path.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Stage-1 checkpoint path.")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda.")
    parser.add_argument("--batch-size", type=int, default=128, help="Cache build batch size.")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Posterior sample seed.")
    parser.add_argument("--max-samples-per-split", type=int, default=None, help="Optional debug cap per split.")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional cache output directory.")
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


def build_split_dataset(cfg: dict, split: str, max_samples: int | None) -> Subset | torch.utils.data.Dataset:
    dataset = build_stage1_dataset_from_config(cfg, splits=[split])
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    return Subset(dataset, list(range(max_samples)))


def unwrap_dataset(dataset) -> tuple[object, list[int]]:
    if isinstance(dataset, Subset):
        return dataset.dataset, list(dataset.indices)
    return dataset, list(range(len(dataset)))


def checkpoint_run_name(checkpoint_path: Path) -> str:
    return checkpoint_path.parents[1].name


def default_output_dir(checkpoint_path: Path) -> Path:
    return EXPERIMENT_ROOT / "results" / "stage2_cache" / checkpoint_run_name(checkpoint_path)


def load_model(cfg: dict, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_stage1_autoencoder_from_config(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    cfg = load_frozen_config(args.config)
    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else default_output_dir(checkpoint_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    config_snapshot_path = output_dir / "cache_build_config.yaml"
    if not config_snapshot_path.exists():
        shutil.copyfile(args.config or default_config_path(), config_snapshot_path)

    device = detect_device(args.device)
    model = load_model(cfg, checkpoint_path, device)
    norm_stats_path = EXPERIMENT_ROOT / cfg["operations"]["artifacts"]["condition_norm_stats_json"]
    norm_stats = load_condition_norm_stats(norm_stats_path)

    station_codes = sorted({row["station_code"] for row in build_stage1_dataset_from_config(cfg, splits=["train", "validation", "test", "ood"]).rows})
    station_mapping = {code: idx for idx, code in enumerate(station_codes)}
    cache_manifest = {
        "run_name": checkpoint_run_name(checkpoint_path),
        "checkpoint_path": str(checkpoint_path),
        "cache_seed": seed,
        "device": str(device),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "num_stations": len(station_mapping),
        "splits": {},
    }
    save_json(output_dir / "cache_manifest.json", cache_manifest)

    start_time = time.time()
    latent_mean_stats = None
    latent_std_stats = None

    for split in ("train", "validation", "test", "ood"):
        log(f"Building latent cache for split={split}")
        dataset = build_split_dataset(cfg, split, args.max_samples_per_split)
        base_dataset, indices = unwrap_dataset(dataset)
        loader = DataLoader(
            dataset,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            pin_memory=device.type == "cuda",
            persistent_workers=int(args.num_workers) > 0,
        )

        latent_all = []
        latent_mean_all = []
        latent_log_std_all = []
        cond_raw_all = []
        cond_norm_all = []
        meta = {
            "event_id": [],
            "station_code": [],
            "split": [],
            "dataset_index": [],
            "magnitude": [],
            "hypocentral_distance_km": [],
            "file_path": [],
            "origin_time": [],
            "requires_left_pad": [],
            "requires_right_pad": [],
            "station_index": [],
        }

        row_offset = 0
        with torch.no_grad():
            for batch in loader:
                x = batch["representation"].to(device)
                mean, log_std = model.encode_stats(x)
                latent = model.reparameterize(mean, log_std)

                batch_size_actual = x.shape[0]
                batch_indices = indices[row_offset : row_offset + batch_size_actual]
                row_offset += batch_size_actual

                latent_all.append(latent.detach().cpu())
                latent_mean_all.append(mean.detach().cpu())
                latent_log_std_all.append(log_std.detach().cpu())

                raw_vectors = []
                norm_vectors = []
                for row_index in batch_indices:
                    row = base_dataset.rows[row_index]
                    raw_vec, norm_vec = build_condition_vector(row, norm_stats)
                    raw_vectors.append(raw_vec)
                    norm_vectors.append(norm_vec)
                    meta["event_id"].append(row["event_id"])
                    meta["station_code"].append(row["station_code"])
                    meta["station_index"].append(int(station_mapping[row["station_code"]]))
                    meta["split"].append(row["split"])
                    meta["dataset_index"].append(int(row_index))
                    meta["magnitude"].append(float(row["magnitude"]))
                    meta["hypocentral_distance_km"].append(float(row["hypocentral_distance_km"]))
                    meta["file_path"].append(row["file_path"])
                    meta["origin_time"].append(row["origin_time"])
                    meta["requires_left_pad"].append(bool(row["requires_left_pad"]))
                    meta["requires_right_pad"].append(bool(row["requires_right_pad"]))

                cond_raw_all.append(torch.from_numpy(np.stack(raw_vectors, axis=0)))
                cond_norm_all.append(torch.from_numpy(np.stack(norm_vectors, axis=0)))

        payload = {
            "latent": torch.cat(latent_all, dim=0),
            "latent_mean": torch.cat(latent_mean_all, dim=0),
            "latent_log_std": torch.cat(latent_log_std_all, dim=0),
            "condition_raw": torch.cat(cond_raw_all, dim=0).float(),
            "condition_normalized": torch.cat(cond_norm_all, dim=0).float(),
            "meta": meta,
        }

        out_path = output_dir / f"{split}_latent_cache.pt"
        torch.save(payload, out_path)
        cache_manifest["splits"][split] = {
            "num_samples": int(payload["latent"].shape[0]),
            "cache_path": str(out_path),
        }
        save_json(output_dir / "cache_manifest.json", cache_manifest)
        log(f"Saved {split} cache to {out_path}")

        if split == "train":
            latent_mean_stats = payload["latent"].mean(dim=0)
            latent_std_stats = payload["latent"].std(dim=0, unbiased=False).clamp(min=1.0e-6)

    latent_stats = {
        "latent_mean": latent_mean_stats,
        "latent_std": latent_std_stats,
    }
    torch.save(latent_stats, output_dir / "latent_stats.pt")
    save_json(
        output_dir / "latent_stats_summary.json",
        {
            "latent_shape": list(latent_mean_stats.shape),
            "latent_mean_abs_mean": float(latent_mean_stats.abs().mean().item()),
            "latent_std_mean": float(latent_std_stats.mean().item()),
            "latent_std_min": float(latent_std_stats.min().item()),
            "latent_std_max": float(latent_std_stats.max().item()),
        },
    )

    save_json(
        output_dir / "resource_summary.json",
        {
            "device": str(device),
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "cache_seed": seed,
            "wall_clock_sec": time.time() - start_time,
            "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "gpu_peak_memory_mb": (
                float(torch.cuda.max_memory_allocated(device) / (1024 * 1024)) if device.type == "cuda" else None
            ),
        },
    )
    log(f"Stage-2 latent cache build finished. Artifacts written under {output_dir}")


if __name__ == "__main__":
    main()
