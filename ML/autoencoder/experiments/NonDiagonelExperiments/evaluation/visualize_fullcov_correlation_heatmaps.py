import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import obspy
import torch
from scipy import signal

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))

from ML.autoencoder.experiments.NonDiagonel.core.model_full_cov_geo import GeoFullCovCVAE
from ML.autoencoder.experiments.NonDiagonel.core.stft_dataset_geo import SeismicSTFTDatasetGeoCondition


def load_fullcov_model(device: torch.device, station_count: int, checkpoint_path: str):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("config", {})
    latent_dim = int(cfg.get("latent_dim", 128))
    condition_dim = int(cfg.get("condition_dim", 64))
    num_stations = int(cfg.get("num_stations", station_count))

    model = GeoFullCovCVAE(
        in_channels=3,
        latent_dim=latent_dim,
        num_stations=num_stations,
        condition_dim=condition_dim,
        numeric_condition_dim=5,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def to_corr_matrix(L: torch.Tensor) -> np.ndarray:
    # L: [D, D] lower-triangular cholesky
    Li = L.detach().cpu().float().numpy()
    cov = Li @ Li.T
    var = np.clip(np.diag(cov), 1e-12, None)
    std = np.sqrt(var)
    corr = cov / (std[:, None] * std[None, :] + 1e-12)
    np.fill_diagonal(corr, 1.0)
    return corr


def plot_heatmap(mat: np.ndarray, out_path: Path, title: str, cmap: str, vmin: float, vmax: float):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Latent Index")
    ax.set_ylabel("Latent Index")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Correlation", rotation=270, labelpad=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize FullCov posterior correlation heatmaps on custom OOD.")
    parser.add_argument("--ood_data_dir", default="data/ood_waveforms/post_training_custom/filtered")
    parser.add_argument("--ood_catalog", default="data/events/ood_catalog_post_training.txt")
    parser.add_argument("--station_list_file", default="data/station_list_external_full.json")
    parser.add_argument("--station_subset_file", default="data/station_list_post_custom.json")
    parser.add_argument("--station_coords_file", default="ML/autoencoder/experiments/NonDiagonel/results/station_coords_external.json")
    parser.add_argument("--condition_stats_file", default="ML/autoencoder/experiments/NonDiagonel/results/condition_stats_external_seed42.json")
    parser.add_argument(
        "--fullcov_checkpoint",
        default="ML/autoencoder/experiments/NonDiagonel/checkpoints/fullcov_geo_repi_external_s42_20260215_best.pt",
    )
    parser.add_argument("--magnitude_col", default="ML")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output_dir",
        default="ML/autoencoder/experiments/NonDiagonel/results/evaluations/post_training_custom_geo_repi_s42_20260216_1942/heatmaps",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device if args.device in ["cpu", "cuda"] else "cpu")

    with open(args.station_list_file, "r") as f:
        station_list = json.load(f)
    station_subset = None
    if args.station_subset_file:
        with open(args.station_subset_file, "r") as f:
            station_subset = set(json.load(f))

    dataset = SeismicSTFTDatasetGeoCondition(
        data_dir=args.ood_data_dir,
        event_file=args.ood_catalog,
        station_coords_file=args.station_coords_file,
        channels=["HH"],
        magnitude_col=args.magnitude_col,
        station_list=station_list,
        condition_stats_file=args.condition_stats_file,
    )
    if args.max_samples > 0:
        dataset.file_paths = dataset.file_paths[: args.max_samples]

    model, ckpt = load_fullcov_model(
        device=device,
        station_count=len(station_list),
        checkpoint_path=args.fullcov_checkpoint,
    )

    corr_mats = []
    sample_info = []

    with torch.no_grad():
        for i in range(len(dataset)):
            spec, cond_numeric, station_idx, meta = dataset[i]
            if "error" in meta:
                continue

            station_name = meta.get("station_name", "")
            if station_subset is not None and station_name not in station_subset:
                continue

            file_path = meta["file_path"]
            st_gt = obspy.read(file_path)
            st_gt.resample(100.0)
            tr_gt = st_gt.select(component="Z")[0] if st_gt.select(component="Z") else st_gt[0]
            gt_wav = tr_gt.data.astype(np.float32)
            target_len = 7300
            if len(gt_wav) > target_len:
                gt_wav = gt_wav[:target_len]
            elif len(gt_wav) < target_len:
                gt_wav = np.pad(gt_wav, (0, target_len - len(gt_wav)))

            _, _, zxx = signal.stft(gt_wav, fs=100.0, nperseg=256, noverlap=192, nfft=256, boundary="zeros")
            mag_spec = np.log1p(np.abs(zxx))
            mag_min, mag_max = float(mag_spec.min()), float(mag_spec.max())
            if mag_max > mag_min:
                mag_spec = (mag_spec - mag_min) / (mag_max - mag_min + 1e-8)

            spec_in = torch.from_numpy(mag_spec).unsqueeze(0).unsqueeze(0).float().to(device).repeat(1, 3, 1, 1)
            if spec_in.shape[2:] != (129, 111):
                spec_in = torch.nn.functional.interpolate(spec_in, size=(129, 111), mode="bilinear", align_corners=False)

            cond_in = cond_numeric.unsqueeze(0).to(device)
            sta_in = station_idx.unsqueeze(0).to(device)
            _, _, L = model(spec_in, cond_in, sta_in)
            corr = to_corr_matrix(L[0])
            corr_mats.append(corr)

            off = np.abs(corr - np.eye(corr.shape[0]))
            max_abs_offdiag = float(np.max(off[np.triu_indices(corr.shape[0], 1)]))
            sample_info.append(
                {
                    "dataset_index": int(i),
                    "file_name": meta.get("file_name"),
                    "event_id": meta.get("event_id"),
                    "station_name": meta.get("station_name"),
                    "max_abs_offdiag": max_abs_offdiag,
                }
            )

    if not corr_mats:
        raise RuntimeError("No valid samples found for heatmap visualization.")

    corr_stack = np.stack(corr_mats, axis=0)
    mean_corr = np.mean(corr_stack, axis=0)
    mean_abs_corr = np.mean(np.abs(corr_stack), axis=0)

    # strongest sample by max off-diagonal absolute correlation
    strongest_idx = int(np.argmax([s["max_abs_offdiag"] for s in sample_info]))
    strongest_corr = corr_stack[strongest_idx]
    strongest_info = sample_info[strongest_idx]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "corr_stack.npy", corr_stack)
    np.save(out_dir / "corr_mean_signed.npy", mean_corr)
    np.save(out_dir / "corr_mean_abs.npy", mean_abs_corr)
    np.save(out_dir / "corr_strongest_sample.npy", strongest_corr)

    d = mean_corr.shape[0]
    n = corr_stack.shape[0]
    plot_heatmap(
        mean_corr,
        out_dir / "heatmap_mean_signed_corr.png",
        title=f"FullCov Mean Signed Correlation (N={n}, D={d})",
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
    )
    plot_heatmap(
        mean_abs_corr,
        out_dir / "heatmap_mean_abs_corr.png",
        title=f"FullCov Mean |Correlation| (N={n}, D={d})",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    plot_heatmap(
        strongest_corr,
        out_dir / "heatmap_strongest_offdiag_sample.png",
        title=f"Strongest Offdiag Sample: {strongest_info['file_name']}",
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
    )

    # summary JSON
    off_mean_abs = mean_abs_corr - np.eye(d)
    off_vals = np.abs(off_mean_abs[np.triu_indices(d, 1)])
    summary = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "checkpoint": args.fullcov_checkpoint,
        "checkpoint_epoch": ckpt.get("epoch"),
        "num_samples": n,
        "latent_dim": d,
        "mean_abs_corr_offdiag_mean_matrix": float(np.mean(off_vals)),
        "p95_abs_corr_offdiag_mean_matrix": float(np.quantile(off_vals, 0.95)),
        "max_abs_corr_offdiag_mean_matrix": float(np.max(off_vals)),
        "strongest_sample": strongest_info,
    }
    with open(out_dir / "heatmap_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "sample_info.jsonl", "w") as f:
        for row in sample_info:
            f.write(json.dumps(row) + "\n")

    print(f"[DONE] Heatmaps saved to: {out_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
