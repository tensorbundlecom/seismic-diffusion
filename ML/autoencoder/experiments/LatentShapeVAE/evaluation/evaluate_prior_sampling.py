#!/usr/bin/env python3
"""
Evaluate prior-sampling realism for LatentShapeVAE checkpoints.

Realism feature set (per trace, averaged over channels):
  - Band energy ratios: [0.5-2], [2-8], [8-20] Hz
  - Envelope peak, kurtosis, duration_above_10pct_peak_sec
  - PSD slope (log power vs log freq), spectral centroid
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

# evaluation -> LatentShapeVAE -> experiments -> autoencoder -> ML -> repo
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))

from ML.autoencoder.experiments.LatentShapeVAE.core.model_vae_waveform import WaveformAE, WaveformDiagVAE
from ML.autoencoder.experiments.LatentShapeVAE.core.waveform_dataset import (
    WaveformDataset,
    collate_fn_waveform,
)


def _read_yaml(path: Path) -> Dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from exc
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _build_dataset(files: List[str], data_cfg: Dict, normalization_stats_file: str) -> WaveformDataset:
    preprocess = data_cfg.get("preprocess", {})
    band = preprocess.get("bandpass", {})
    return WaveformDataset(
        file_paths=files,
        segment_length=int(data_cfg.get("segment_length", 7001)),
        sample_rate_hz=float(data_cfg.get("sample_rate_hz", 100.0)),
        preprocess_demean=bool(preprocess.get("demean", True)),
        preprocess_detrend=bool(preprocess.get("detrend", True)),
        bandpass_enabled=bool(band.get("enabled", True)),
        bandpass_freqmin=float(band.get("freqmin", 0.5)),
        bandpass_freqmax=float(band.get("freqmax", 20.0)),
        bandpass_corners=int(band.get("corners", 4)),
        bandpass_zerophase=bool(band.get("zerophase", True)),
        normalization_stats_file=normalization_stats_file,
    )


def _build_model(run_cfg: Dict, device: torch.device):
    mode = str(run_cfg["ablation_mode"])
    latent_dim = int(run_cfg["latent_dim"])
    backbone = str(run_cfg["backbone"])
    input_length = int(run_cfg.get("segment_length", 7001))
    logvar_mode = str(run_cfg.get("logvar_mode", "legacy"))
    logvar_min = float(run_cfg.get("logvar_min", -30.0))
    logvar_max = float(run_cfg.get("logvar_max", 20.0))
    if mode == "ae":
        model = WaveformAE(in_channels=3, input_length=input_length, latent_dim=latent_dim, backbone=backbone)
    else:
        model = WaveformDiagVAE(
            in_channels=3,
            input_length=input_length,
            latent_dim=latent_dim,
            backbone=backbone,
            logvar_mode=logvar_mode,
            logvar_min=logvar_min,
            logvar_max=logvar_max,
        )
    return model.to(device)


def _denormalize(x: np.ndarray, stats: Dict) -> np.ndarray:
    # x: [B, C, T]
    mean = np.asarray(stats["mean"], dtype=np.float64).reshape(1, 3, 1)
    std = np.asarray(stats["std"], dtype=np.float64).reshape(1, 3, 1)
    std = np.clip(std, 1e-8, None)
    return x * std + mean


def _band_energy_ratio(psd: np.ndarray, freqs: np.ndarray, f0: float, f1: float) -> float:
    m = (freqs >= f0) & (freqs < f1)
    if not np.any(m):
        return 0.0
    return float(np.sum(psd[m]))


def _trace_features(x_ct: np.ndarray, fs: float) -> Dict[str, float]:
    # x_ct: [C, T]
    eps = 1e-12
    c, t = x_ct.shape
    freqs = np.fft.rfftfreq(t, d=1.0 / fs)

    rows = []
    for ch in range(c):
        y = x_ct[ch].astype(np.float64)
        y = y - y.mean()
        spec = np.fft.rfft(y)
        psd = np.abs(spec) ** 2
        e1 = _band_energy_ratio(psd, freqs, 0.5, 2.0)
        e2 = _band_energy_ratio(psd, freqs, 2.0, 8.0)
        e3 = _band_energy_ratio(psd, freqs, 8.0, 20.0)
        et = max(e1 + e2 + e3, eps)
        b1 = e1 / et
        b2 = e2 / et
        b3 = e3 / et

        env = np.abs(y)
        peak = float(np.max(env))
        m2 = float(np.mean((env - env.mean()) ** 2))
        m4 = float(np.mean((env - env.mean()) ** 4))
        kurt = float(m4 / (m2**2 + eps))
        thr = 0.1 * peak
        dur = float(np.sum(env >= thr) / fs)

        band_mask = (freqs >= 0.5) & (freqs <= 20.0)
        f_band = freqs[band_mask]
        p_band = psd[band_mask]
        p_sum = float(np.sum(p_band)) + eps
        centroid = float(np.sum(f_band * p_band) / p_sum)

        valid = (f_band > 0) & (p_band > 0)
        if np.sum(valid) >= 2:
            xlog = np.log(f_band[valid])
            ylog = np.log(p_band[valid] + eps)
            slope = float(np.polyfit(xlog, ylog, deg=1)[0])
        else:
            slope = 0.0

        rows.append([b1, b2, b3, peak, kurt, dur, slope, centroid])

    arr = np.asarray(rows, dtype=np.float64)
    mean_vec = arr.mean(axis=0)
    return {
        "band_ratio_0p5_2": float(mean_vec[0]),
        "band_ratio_2_8": float(mean_vec[1]),
        "band_ratio_8_20": float(mean_vec[2]),
        "env_peak": float(mean_vec[3]),
        "env_kurtosis": float(mean_vec[4]),
        "env_duration_10pct_sec": float(mean_vec[5]),
        "psd_slope_loglog": float(mean_vec[6]),
        "spectral_centroid_hz": float(mean_vec[7]),
    }


def _aggregate_features(x_bct: np.ndarray, fs: float) -> Dict[str, np.ndarray]:
    feats: Dict[str, List[float]] = {}
    for i in range(x_bct.shape[0]):
        row = _trace_features(x_bct[i], fs=fs)
        for k, v in row.items():
            feats.setdefault(k, []).append(float(v))
    return {k: np.asarray(v, dtype=np.float64) for k, v in feats.items()}


def _realism_score(real: Dict[str, np.ndarray], gen: Dict[str, np.ndarray]) -> Dict[str, float]:
    eps = 1e-8
    keys = sorted(real.keys())
    mean_z = []
    std_log = []
    per_feature = {}
    for k in keys:
        r = real[k]
        g = gen[k]
        r_mu = float(np.mean(r))
        r_std = float(np.std(r)) + eps
        g_mu = float(np.mean(g))
        g_std = float(np.std(g)) + eps
        mean_dev = abs(g_mu - r_mu) / r_std
        std_dev = abs(np.log(g_std / r_std))
        mean_z.append(mean_dev)
        std_log.append(std_dev)
        per_feature[k] = {
            "real_mean": r_mu,
            "real_std": r_std,
            "gen_mean": g_mu,
            "gen_std": g_std,
            "mean_dev_z": mean_dev,
            "std_dev_log": std_dev,
        }
    return {
        "feature_details": per_feature,
        "realism_mean_dev_z_avg": float(np.mean(mean_z)),
        "realism_std_dev_log_avg": float(np.mean(std_log)),
        "realism_composite": float(np.mean(mean_z) + np.mean(std_log)),
    }


def _collect_real_samples(loader: DataLoader, max_samples: int, stats: Dict, fs: float) -> Dict[str, np.ndarray]:
    chunks = []
    n = 0
    for batch in loader:
        x, _meta = batch
        if x is None:
            continue
        x_np = x.detach().cpu().numpy().astype(np.float64)
        x_np = _denormalize(x_np, stats)
        chunks.append(x_np)
        n += x_np.shape[0]
        if n >= max_samples:
            break
    if not chunks:
        raise RuntimeError("No real samples collected from dataset.")
    real = np.concatenate(chunks, axis=0)[:max_samples]
    return _aggregate_features(real, fs=fs)


def _collect_generated_samples(
    model: torch.nn.Module,
    latent_dim: int,
    batch_size: int,
    max_samples: int,
    segment_length: int,
    stats: Dict,
    device: torch.device,
    fs: float,
) -> Dict[str, np.ndarray]:
    chunks = []
    n = 0
    model.eval()
    with torch.no_grad():
        while n < max_samples:
            bs = min(batch_size, max_samples - n)
            z = torch.randn(bs, latent_dim, device=device)
            x_hat = model.decode(z, target_length=segment_length)
            x_np = x_hat.detach().cpu().numpy().astype(np.float64)
            x_np = _denormalize(x_np, stats)
            chunks.append(x_np)
            n += bs
    gen = np.concatenate(chunks, axis=0)[:max_samples]
    return _aggregate_features(gen, fs=fs)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate prior-sampling realism for LatentShapeVAE checkpoints.")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument(
        "--data_config",
        default="ML/autoencoder/experiments/LatentShapeVAE/configs/data_protocol_v1.yaml",
    )
    parser.add_argument(
        "--frozen_split_manifest",
        default="ML/autoencoder/experiments/LatentShapeVAE/protocol/frozen_event_splits_v1.json",
    )
    parser.add_argument("--normalization_stats_file", default="")
    parser.add_argument("--split", choices=["test", "ood_event"], default="ood_event")
    parser.add_argument("--num_real_samples", type=int, default=2000)
    parser.add_argument("--num_generated_samples", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data_cfg = _read_yaml(Path(args.data_config))
    manifest = _read_json(Path(args.frozen_split_manifest))
    norm_file = args.normalization_stats_file or data_cfg.get("normalization", {}).get("stats_file", "")
    if not norm_file:
        raise ValueError("normalization_stats_file not set in args or data config.")
    norm_stats = _read_json(Path(norm_file))

    split_key = f"{args.split}_files"
    if split_key not in manifest["splits"]:
        raise ValueError(f"Split key not found in manifest: {split_key}")
    files = _read_lines(Path(manifest["splits"][split_key]["file"]))
    if not files:
        raise RuntimeError(f"Split {args.split} is empty.")

    fs = float(data_cfg.get("sample_rate_hz", 100.0))
    segment_length = int(data_cfg.get("segment_length", 7001))
    ds = _build_dataset(files, data_cfg=data_cfg, normalization_stats_file=norm_file)

    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_fn_waveform,
    )
    real_features = _collect_real_samples(
        loader=loader,
        max_samples=args.num_real_samples,
        stats=norm_stats,
        fs=fs,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else Path("ML/autoencoder/experiments/LatentShapeVAE/results") / f"prior_sampling_{args.split}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for ckpt_arg in args.checkpoints:
        ckpt_path = Path(ckpt_arg)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        run_cfg = dict(ckpt.get("config", {}))
        run_cfg["segment_length"] = segment_length
        run_name = str(run_cfg.get("run_name", ckpt_path.stem.replace("_best", "")))
        latent_dim = int(run_cfg["latent_dim"])

        model = _build_model(run_cfg, device=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        gen_features = _collect_generated_samples(
            model=model,
            latent_dim=latent_dim,
            batch_size=args.batch_size,
            max_samples=args.num_generated_samples,
            segment_length=segment_length,
            stats=norm_stats,
            device=device,
            fs=fs,
        )
        sc = _realism_score(real=real_features, gen=gen_features)

        run_out = out_dir / run_name
        run_out.mkdir(parents=True, exist_ok=True)
        with (run_out / "real_features.json").open("w", encoding="utf-8") as f:
            json.dump({k: v.tolist() for k, v in real_features.items()}, f)
        with (run_out / "generated_features.json").open("w", encoding="utf-8") as f:
            json.dump({k: v.tolist() for k, v in gen_features.items()}, f)
        with (run_out / "realism_summary.json").open("w", encoding="utf-8") as f:
            json.dump(sc, f, indent=2)

        row = {
            "run_name": run_name,
            "checkpoint": ckpt_path.as_posix(),
            "ablation_mode": str(run_cfg.get("ablation_mode", "")),
            "backbone": str(run_cfg.get("backbone", "")),
            "latent_dim": latent_dim,
            "split": args.split,
            "num_real_samples": int(args.num_real_samples),
            "num_generated_samples": int(args.num_generated_samples),
            "realism_mean_dev_z_avg": float(sc["realism_mean_dev_z_avg"]),
            "realism_std_dev_log_avg": float(sc["realism_std_dev_log_avg"]),
            "realism_composite": float(sc["realism_composite"]),
        }
        rows.append(row)

    csv_path = out_dir / "prior_sampling_realism_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path = out_dir / "prior_sampling_realism_summary.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# Prior Sampling Realism Summary ({args.split})\n\n")
        f.write("| Run | Mode | Backbone | Latent | MeanDevZ | StdDevLog | Composite |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['run_name']} | {r['ablation_mode']} | {r['backbone']} | {r['latent_dim']} | "
                f"{r['realism_mean_dev_z_avg']:.4f} | {r['realism_std_dev_log_avg']:.4f} | {r['realism_composite']:.4f} |\n"
            )

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "output_dir": out_dir.as_posix(),
        "summary_csv": csv_path.as_posix(),
        "summary_md": md_path.as_posix(),
    }
    with (out_dir / "evaluation_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("[INFO] output_dir:", out_dir)
    print("[INFO] summary_csv:", csv_path)
    print("[INFO] summary_md :", md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
