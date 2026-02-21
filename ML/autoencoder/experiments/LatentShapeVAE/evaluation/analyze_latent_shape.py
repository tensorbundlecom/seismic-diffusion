#!/usr/bin/env python3
"""
Analyze latent-shape metrics for LatentShapeVAE checkpoints.

Metrics:
  Cov_mu, Mean_Sigma, Cov_agg
  ||mean||, diag_mae, offdiag_mean_abs(corr), eig_ratio, KL_moment, W2_moment
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def _safe_spd(cov: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    cov = (cov + cov.T) * 0.5
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, eps, None)
    cov_spd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    cov_spd = (cov_spd + cov_spd.T) * 0.5
    return cov_spd, eigvals


def _cov_to_corr(cov: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    d = np.clip(np.diag(cov), eps, None)
    s = np.sqrt(d)
    corr = cov / (s[:, None] * s[None, :] + eps)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def _offdiag_mean_abs(corr: np.ndarray) -> float:
    d = corr.shape[0]
    if d <= 1:
        return 0.0
    vals = np.abs(corr[np.triu_indices(d, k=1)])
    return float(vals.mean()) if vals.size > 0 else 0.0


def _kl_moment_gaussian_to_std_normal(mean: np.ndarray, cov_spd: np.ndarray, eigvals: np.ndarray) -> float:
    d = mean.shape[0]
    tr = float(np.trace(cov_spd))
    m2 = float(np.dot(mean, mean))
    logdet = float(np.sum(np.log(eigvals)))
    return 0.5 * (tr + m2 - d - logdet)


def _w2_moment_gaussian_to_std_normal(mean: np.ndarray, eigvals: np.ndarray) -> float:
    m2 = float(np.dot(mean, mean))
    c_term = float(np.sum((np.sqrt(eigvals) - 1.0) ** 2))
    return m2 + c_term


def _plot_heatmap(mat: np.ndarray, title: str, out_path: Path) -> None:
    plt.figure(figsize=(6, 5))
    plt.imshow(mat, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_eigvals(eigvals: np.ndarray, title: str, out_path: Path) -> None:
    x = np.arange(1, eigvals.size + 1)
    y = np.sort(np.clip(eigvals, 1e-12, None))[::-1]
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, marker="o", linewidth=1.0, markersize=2.5)
    plt.yscale("log")
    plt.xlabel("Rank")
    plt.ylabel("Eigenvalue (log)")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_pca_scatter(mu_all: np.ndarray, title: str, out_path: Path, max_points: int = 4000) -> None:
    if mu_all.shape[0] == 0 or mu_all.shape[1] < 2:
        return
    x = mu_all - mu_all.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    proj = u[:, :2] * s[:2]
    if proj.shape[0] > max_points:
        idx = np.linspace(0, proj.shape[0] - 1, max_points, dtype=int)
        proj = proj[idx]
    plt.figure(figsize=(6, 5))
    plt.scatter(proj[:, 0], proj[:, 1], s=4, alpha=0.25)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


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


def _iter_checkpoint_paths(arg_list: List[str]) -> List[Path]:
    out: List[Path] = []
    for a in arg_list:
        p = Path(a)
        if p.exists():
            out.append(p)
            continue
        # simple wildcard fallback
        for q in sorted(Path(".").glob(a)):
            if q.is_file():
                out.append(q)
    dedup = []
    seen = set()
    for p in out:
        k = p.resolve().as_posix()
        if k not in seen:
            dedup.append(p)
            seen.add(k)
    return dedup


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze latent-shape metrics for LatentShapeVAE checkpoints.")
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
    parser.add_argument("--split", choices=["train", "val", "test", "ood_event"], default="test")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=0, help="0 means all samples in split.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="")
    args = parser.parse_args()

    ckpt_paths = _iter_checkpoint_paths(args.checkpoints)
    if not ckpt_paths:
        raise RuntimeError("No checkpoint found from --checkpoints")

    data_cfg = _read_yaml(Path(args.data_config))
    manifest = _read_json(Path(args.frozen_split_manifest))
    split_key = f"{args.split}_files"
    if split_key not in manifest["splits"]:
        raise ValueError(f"Split key not found in manifest: {split_key}")
    split_files = _read_lines(Path(manifest["splits"][split_key]["file"]))
    if args.max_samples > 0:
        split_files = split_files[: args.max_samples]
    if not split_files:
        raise RuntimeError(f"Split {args.split} is empty.")

    norm_file = args.normalization_stats_file or data_cfg.get("normalization", {}).get("stats_file", "")
    if not norm_file:
        raise ValueError("normalization_stats_file not set in args or data config.")

    ds = _build_dataset(split_files, data_cfg=data_cfg, normalization_stats_file=norm_file)
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

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else Path("ML/autoencoder/experiments/LatentShapeVAE/results") / f"latent_shape_{args.split}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for ckpt_path in ckpt_paths:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        run_cfg = ckpt.get("config", {})
        run_name = str(run_cfg.get("run_name", ckpt_path.stem.replace("_best", "")))
        latent_dim = int(run_cfg["latent_dim"])
        mode = str(run_cfg["ablation_mode"])
        run_cfg = dict(run_cfg)
        run_cfg["segment_length"] = int(data_cfg.get("segment_length", 7001))

        model = _build_model(run_cfg=run_cfg, device=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        run_out = out_dir / run_name
        run_out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()

        mu_rows: List[np.ndarray] = []
        sum_sigma = np.zeros((latent_dim,), dtype=np.float64)
        n = 0

        with torch.no_grad():
            for batch in loader:
                x, _meta = batch
                if x is None:
                    continue
                x = x.to(device, non_blocking=use_cuda)
                if mode == "ae":
                    z = model.encode(x)
                    mu_np = z.detach().cpu().numpy().astype(np.float64)
                    mu_rows.append(mu_np)
                    n += mu_np.shape[0]
                else:
                    mu, logvar = model.encode(x)
                    mu_np = mu.detach().cpu().numpy().astype(np.float64)
                    lv_np = np.clip(logvar.detach().cpu().numpy().astype(np.float64), -30.0, 20.0)
                    var_np = np.exp(lv_np)
                    mu_rows.append(mu_np)
                    sum_sigma += var_np.sum(axis=0)
                    n += mu_np.shape[0]

        if n == 0:
            raise RuntimeError(f"No valid batch found while analyzing: {run_name}")

        mu_all = np.concatenate(mu_rows, axis=0)
        mean_mu = mu_all.mean(axis=0)
        cov_mu = np.cov(mu_all, rowvar=False) if mu_all.shape[0] > 1 else np.zeros((latent_dim, latent_dim), dtype=np.float64)
        cov_mu = (cov_mu + cov_mu.T) * 0.5
        mean_sigma = np.diag(sum_sigma / float(n)) if mode != "ae" else np.zeros((latent_dim, latent_dim), dtype=np.float64)
        cov_agg = cov_mu + mean_sigma
        cov_agg_spd, eigvals = _safe_spd(cov_agg, eps=1e-8)
        corr_agg = _cov_to_corr(cov_agg_spd)

        diag_vals = np.diag(cov_agg_spd)
        mean_norm = float(np.linalg.norm(mean_mu))
        diag_mae = float(np.mean(np.abs(diag_vals - 1.0)))
        offdiag_abs = _offdiag_mean_abs(corr_agg)
        eig_ratio = float(np.max(eigvals) / np.clip(np.min(eigvals), 1e-12, None))
        kl_moment = float(_kl_moment_gaussian_to_std_normal(mean_mu, cov_agg_spd, eigvals))
        w2_moment = float(_w2_moment_gaussian_to_std_normal(mean_mu, eigvals))
        runtime_sec = time.time() - t0

        np.save(run_out / "mean_mu.npy", mean_mu)
        np.save(run_out / "cov_mu.npy", cov_mu)
        np.save(run_out / "mean_sigma.npy", mean_sigma)
        np.save(run_out / "cov_agg.npy", cov_agg_spd)
        np.save(run_out / "corr_agg.npy", corr_agg)
        np.save(run_out / "eigvals.npy", eigvals)

        _plot_heatmap(corr_agg, f"{run_name} corr(Cov_agg)", run_out / "corr_agg_heatmap.png")
        _plot_eigvals(eigvals, f"{run_name} eigenvalues(Cov_agg)", run_out / "eigvals_cov_agg.png")
        _plot_pca_scatter(mu_all, f"{run_name} PCA(mu)", run_out / "pca_mu.png")

        row = {
            "run_name": run_name,
            "checkpoint": ckpt_path.as_posix(),
            "ablation_mode": mode,
            "backbone": str(run_cfg.get("backbone", "")),
            "latent_dim": latent_dim,
            "split": args.split,
            "n_samples": int(n),
            "mean_norm_l2": mean_norm,
            "diag_mae": diag_mae,
            "offdiag_mean_abs_corr": offdiag_abs,
            "eig_ratio": eig_ratio,
            "kl_moment_to_std_normal": kl_moment,
            "w2_moment_to_std_normal": w2_moment,
            "runtime_sec": runtime_sec,
        }
        summary_rows.append(row)

        with (run_out / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(row, f, indent=2)

    csv_path = out_dir / "latent_shape_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    md_path = out_dir / "latent_shape_summary.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# Latent Shape Summary ({args.split})\n\n")
        f.write("| Run | Mode | Backbone | Latent | N | MeanNorm | DiagMAE | OffDiagAbs | EigRatio | KL_moment | W2_moment |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in summary_rows:
            f.write(
                f"| {r['run_name']} | {r['ablation_mode']} | {r['backbone']} | {r['latent_dim']} | "
                f"{r['n_samples']} | {r['mean_norm_l2']:.4f} | {r['diag_mae']:.4f} | "
                f"{r['offdiag_mean_abs_corr']:.4f} | {r['eig_ratio']:.3f} | "
                f"{r['kl_moment_to_std_normal']:.4f} | {r['w2_moment_to_std_normal']:.4f} |\n"
            )

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "n_runs": len(summary_rows),
        "output_dir": out_dir.as_posix(),
        "summary_csv": csv_path.as_posix(),
        "summary_md": md_path.as_posix(),
    }
    with (out_dir / "analysis_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("[INFO] analysis_dir:", out_dir)
    print("[INFO] summary_csv:", csv_path)
    print("[INFO] summary_md :", md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
