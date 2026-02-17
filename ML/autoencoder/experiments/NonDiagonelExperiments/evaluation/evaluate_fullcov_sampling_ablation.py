import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import obspy
import torch
from fastdtw import fastdtw
from scipy import signal
from scipy.ndimage import uniform_filter
from scipy.signal import hilbert
from scipy.spatial.distance import euclidean

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))

from ML.autoencoder.experiments.NonDiagonel.core.model_full_cov_geo import GeoFullCovCVAE
from ML.autoencoder.experiments.NonDiagonel.core.stft_dataset_geo import SeismicSTFTDatasetGeoCondition

try:
    from skimage.metrics import structural_similarity as ssim
except Exception:

    def ssim(img1, img2, data_range=1.0):
        k1 = 0.01
        k2 = 0.03
        c1 = (k1 * data_range) ** 2
        c2 = (k2 * data_range) ** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)

        mu1 = uniform_filter(img1, size=7)
        mu2 = uniform_filter(img2, size=7)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu12 = mu1 * mu2

        sigma1_sq = uniform_filter(img1 * img1, size=7) - mu1_sq
        sigma2_sq = uniform_filter(img2 * img2, size=7) - mu2_sq
        sigma12 = uniform_filter(img1 * img2, size=7) - mu12

        num = (2.0 * mu12 + c1) * (2.0 * sigma12 + c2)
        den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-12
        return float(np.mean(num / den))


METRICS = [
    "ssim",
    "lsd",
    "sc",
    "s_corr",
    "sta_lta_err",
    "mr_lsd",
    "arias_err",
    "env_corr",
    "dtw",
    "xcorr",
]

HIGHER_BETTER = {"ssim", "s_corr", "env_corr", "xcorr"}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path: Path, obj: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def reconstruct_signal(
    magnitude_spec: np.ndarray,
    mag_min: float,
    mag_max: float,
    fs: float,
    n_iter: int,
    init_phase: np.ndarray,
) -> np.ndarray:
    spec = magnitude_spec.copy()
    if mag_max > mag_min:
        spec = spec * (mag_max - mag_min) + mag_min
    spec = np.expm1(spec)

    phase = init_phase.copy()
    for _ in range(n_iter):
        stft_complex = spec * phase
        _, waveform = signal.istft(
            stft_complex,
            fs=fs,
            nperseg=256,
            noverlap=192,
            nfft=256,
            boundary="zeros",
        )
        _, _, new_zxx = signal.stft(
            waveform,
            fs=fs,
            nperseg=256,
            noverlap=192,
            nfft=256,
            boundary="zeros",
        )
        if new_zxx.shape != spec.shape:
            min_f = min(new_zxx.shape[0], spec.shape[0])
            min_t = min(new_zxx.shape[1], spec.shape[1])
            next_phase = np.zeros_like(spec, dtype=complex)
            next_phase[:min_f, :min_t] = np.exp(1j * np.angle(new_zxx[:min_f, :min_t]))
            phase = next_phase
        else:
            phase = np.exp(1j * np.angle(new_zxx))

    stft_complex = spec * phase
    _, waveform = signal.istft(
        stft_complex,
        fs=fs,
        nperseg=256,
        noverlap=192,
        nfft=256,
        boundary="zeros",
    )
    return waveform


def calculate_seismic_metrics(target_wav, pred_wav, target_spec, pred_spec, fs=100.0):
    metrics = {}

    s1 = (target_spec - np.min(target_spec)) / (np.max(target_spec) - np.min(target_spec) + 1e-8)
    s2 = (pred_spec - np.min(pred_spec)) / (np.max(pred_spec) - np.min(pred_spec) + 1e-8)
    metrics["ssim"] = float(ssim(s1, s2, data_range=1.0))
    metrics["lsd"] = float(np.sqrt(np.mean((np.log(target_spec + 1e-8) - np.log(pred_spec + 1e-8)) ** 2)))
    metrics["sc"] = float(np.linalg.norm(target_spec - pred_spec) / (np.linalg.norm(target_spec) + 1e-8))
    metrics["s_corr"] = float(np.corrcoef(target_spec.flatten(), pred_spec.flatten())[0, 1])

    spec_power_target = np.sum(target_spec, axis=0)
    spec_power_pred = np.sum(pred_spec, axis=0)

    def get_spectral_sta_lta(power, sta_len=5, lta_len=40):
        sta = np.convolve(power, np.ones(sta_len) / sta_len, mode="same")
        lta = np.convolve(power, np.ones(lta_len) / lta_len, mode="same")
        return sta / (lta + 1e-8)

    sl_target = get_spectral_sta_lta(spec_power_target)
    sl_pred = get_spectral_sta_lta(spec_power_pred)
    metrics["sta_lta_err"] = float(np.abs(np.max(sl_target) - np.max(sl_pred)) / (np.max(sl_target) + 1e-8))

    mr_lsd = []
    for n_fft in [64, 128, 512]:
        hop = n_fft // 4
        _, _, z1 = signal.stft(target_wav, fs=fs, nperseg=n_fft, noverlap=n_fft - hop)
        _, _, z2 = signal.stft(pred_wav, fs=fs, nperseg=n_fft, noverlap=n_fft - hop)
        t_spec = np.abs(z1)
        p_spec = np.abs(z2)
        min_f = min(t_spec.shape[0], p_spec.shape[0])
        min_t = min(t_spec.shape[1], p_spec.shape[1])
        mr_lsd.append(
            np.sqrt(
                np.mean(
                    (np.log(t_spec[:min_f, :min_t] + 1e-8) - np.log(p_spec[:min_f, :min_t] + 1e-8)) ** 2
                )
            )
        )
    metrics["mr_lsd"] = float(np.mean(mr_lsd))

    try:
        from scipy.integrate import trapezoid

        a_target = (np.pi / (2 * 9.81)) * trapezoid(target_wav**2, dx=1 / fs)
        a_pred = (np.pi / (2 * 9.81)) * trapezoid(pred_wav**2, dx=1 / fs)
    except Exception:
        a_target = (np.pi / (2 * 9.81)) * np.trapz(target_wav**2, dx=1 / fs)
        a_pred = (np.pi / (2 * 9.81)) * np.trapz(pred_wav**2, dx=1 / fs)
    metrics["arias_err"] = float(np.abs(a_target - a_pred) / (np.abs(a_target) + 1e-8))

    env1 = np.abs(hilbert(target_wav))
    env2 = np.abs(hilbert(pred_wav))
    min_len = min(len(env1), len(env2))
    metrics["env_corr"] = float(np.corrcoef(env1[:min_len], env2[:min_len])[0, 1])

    factor = max(1, len(target_wav) // 500)
    s_target = target_wav[::factor].reshape(-1, 1)
    s_pred = pred_wav[::factor].reshape(-1, 1)
    dtw_dist, _ = fastdtw(s_target, s_pred, dist=euclidean)
    metrics["dtw"] = float(dtw_dist / len(s_target))

    x1 = (target_wav - np.mean(target_wav)) / (np.std(target_wav) + 1e-8)
    x2 = (pred_wav - np.mean(pred_wav)) / (np.std(pred_wav) + 1e-8)
    min_len = min(len(x1), len(x2))
    xcorr = np.correlate(x1[:min_len], x2[:min_len], mode="full")
    metrics["xcorr"] = float(np.max(np.abs(xcorr)) / len(x1[:min_len]))
    return metrics


def signed_improvement(a: float, b: float, metric: str) -> float:
    # positive means A is better than B
    if metric in HIGHER_BETTER:
        return float(a - b)
    return float(b - a)


def aggregate_mode(mode_rows: List[Dict]) -> Dict[str, float]:
    out = {}
    for m in METRICS:
        vals = [r[m] for r in mode_rows]
        out[m] = float(np.mean(vals)) if vals else None
    return out


def bootstrap_mean_ci(values: np.ndarray, n_boot: int, seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return {"mean": None, "ci95_low": None, "ci95_high": None, "p_gt0": None}
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = values[idx].mean(axis=1)
    return {
        "mean": float(np.mean(values)),
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
        "p_gt0": float(np.mean(boot > 0.0)),
    }


def main():
    parser = argparse.ArgumentParser(description="FullCov sampling ablation: mu vs diag-sampled vs full-sampled.")
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
    parser.add_argument("--num_samples_per_event", type=int, default=8, help="Number of eps samples for diag/full modes.")
    parser.add_argument("--griffin_lim_iters", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_items", type=int, default=0)
    parser.add_argument("--bootstrap_iters", type=int, default=5000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output_dir",
        default="ML/autoencoder/experiments/NonDiagonel/results/evaluations/post_training_custom_geo_repi_s42_20260216_1942/sampling_ablation",
    )
    args = parser.parse_args()

    set_seed(args.seed)
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
    if args.max_items > 0:
        dataset.file_paths = dataset.file_paths[: args.max_items]

    ckpt = torch.load(args.fullcov_checkpoint, map_location=device)
    cfg = ckpt.get("config", {})
    model = GeoFullCovCVAE(
        in_channels=3,
        latent_dim=int(cfg.get("latent_dim", 128)),
        num_stations=int(cfg.get("num_stations", len(station_list))),
        condition_dim=int(cfg.get("condition_dim", 64)),
        numeric_condition_dim=5,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    fs = 100.0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_sample = []
    mode_metrics = {"mu": [], "diag_sampled": [], "full_sampled": []}
    processed = 0
    skipped = 0

    with torch.no_grad():
        for i in range(len(dataset)):
            spec, cond_numeric, station_idx, meta = dataset[i]
            if "error" in meta:
                skipped += 1
                continue
            if station_subset is not None and meta.get("station_name") not in station_subset:
                skipped += 1
                continue

            file_path = meta["file_path"]
            st_gt = obspy.read(file_path)
            st_gt.resample(fs)
            tr_gt = st_gt.select(component="Z")[0] if st_gt.select(component="Z") else st_gt[0]
            gt_wav = tr_gt.data.astype(np.float32)
            target_len = 7300
            if len(gt_wav) > target_len:
                gt_wav = gt_wav[:target_len]
            elif len(gt_wav) < target_len:
                gt_wav = np.pad(gt_wav, (0, target_len - len(gt_wav)))

            _, _, zxx = signal.stft(gt_wav, fs=fs, nperseg=256, noverlap=192, nfft=256, boundary="zeros")
            mag_spec = np.log1p(np.abs(zxx))
            mag_min, mag_max = float(mag_spec.min()), float(mag_spec.max())
            if mag_max > mag_min:
                mag_spec = (mag_spec - mag_min) / (mag_max - mag_min + 1e-8)

            spec_in = torch.from_numpy(mag_spec).unsqueeze(0).unsqueeze(0).to(device).float().repeat(1, 3, 1, 1)
            if spec_in.shape[2:] != (129, 111):
                spec_in = torch.nn.functional.interpolate(spec_in, size=(129, 111), mode="bilinear", align_corners=False)
            cond_in = cond_numeric.unsqueeze(0).to(device)
            sta_in = station_idx.unsqueeze(0).to(device)

            mu, L = model.encoder(spec_in, cond_in, sta_in)
            diag = torch.diag_embed(torch.diagonal(L, dim1=-2, dim2=-1))

            # deterministic phase seed per sample
            phase_rng = np.random.default_rng(args.seed + (i + 1) * 9176)
            init_phase = np.exp(2j * np.pi * phase_rng.random((129, 111)))

            def decode_and_score(z_tensor):
                recon = model.decoder(z_tensor, cond_in, sta_in)
                if recon.shape[2:] != (129, 111):
                    recon = torch.nn.functional.interpolate(recon, size=(129, 111), mode="bilinear", align_corners=False)
                pred_spec = recon[0, 2].cpu().numpy()
                pred_wav = reconstruct_signal(
                    pred_spec,
                    mag_min=mag_min,
                    mag_max=mag_max,
                    fs=fs,
                    n_iter=args.griffin_lim_iters,
                    init_phase=init_phase,
                )
                min_len = min(len(gt_wav), len(pred_wav))
                return calculate_seismic_metrics(
                    gt_wav[:min_len],
                    pred_wav[:min_len],
                    spec_in[0, 2].cpu().numpy(),
                    pred_spec,
                    fs=fs,
                )

            mu_metrics = decode_and_score(mu)
            mode_metrics["mu"].append(mu_metrics)

            diag_metrics_list = []
            full_metrics_list = []
            for k in range(args.num_samples_per_event):
                eps_rng = np.random.default_rng(args.seed + (i + 1) * 10000 + k * 31)
                eps_np = eps_rng.standard_normal(size=(1, mu.shape[1], 1)).astype(np.float32)
                eps = torch.from_numpy(eps_np).to(device)

                z_diag = (mu.unsqueeze(2) + torch.bmm(diag, eps)).squeeze(2)
                z_full = (mu.unsqueeze(2) + torch.bmm(L, eps)).squeeze(2)

                diag_metrics = decode_and_score(z_diag)
                full_metrics = decode_and_score(z_full)
                diag_metrics_list.append(diag_metrics)
                full_metrics_list.append(full_metrics)
                mode_metrics["diag_sampled"].append(diag_metrics)
                mode_metrics["full_sampled"].append(full_metrics)

            # Store sample-level averages for paired analysis.
            sample_avg_diag = {m: float(np.mean([d[m] for d in diag_metrics_list])) for m in METRICS}
            sample_avg_full = {m: float(np.mean([d[m] for d in full_metrics_list])) for m in METRICS}
            per_sample.append(
                {
                    "index": i,
                    "file_name": meta.get("file_name"),
                    "event_id": meta.get("event_id"),
                    "station_name": meta.get("station_name"),
                    "mu": mu_metrics,
                    "diag_sampled_avg": sample_avg_diag,
                    "full_sampled_avg": sample_avg_full,
                }
            )
            processed += 1

    aggregate = {mode: {m: float(np.mean([r[m] for r in rows])) for m in METRICS} for mode, rows in mode_metrics.items()}

    # Paired bootstrap on per-sample averages: full vs diag, diag vs mu, full vs mu.
    paired = {"full_vs_diag": {}, "diag_vs_mu": {}, "full_vs_mu": {}}
    for m in METRICS:
        f = np.array([r["full_sampled_avg"][m] for r in per_sample], dtype=np.float64)
        d = np.array([r["diag_sampled_avg"][m] for r in per_sample], dtype=np.float64)
        u = np.array([r["mu"][m] for r in per_sample], dtype=np.float64)

        full_vs_diag = np.array([signed_improvement(fv, dv, m) for fv, dv in zip(f, d)], dtype=np.float64)
        diag_vs_mu = np.array([signed_improvement(dv, uv, m) for dv, uv in zip(d, u)], dtype=np.float64)
        full_vs_mu = np.array([signed_improvement(fv, uv, m) for fv, uv in zip(f, u)], dtype=np.float64)

        paired["full_vs_diag"][m] = bootstrap_mean_ci(full_vs_diag, args.bootstrap_iters, args.seed + 100 + len(m))
        paired["diag_vs_mu"][m] = bootstrap_mean_ci(diag_vs_mu, args.bootstrap_iters, args.seed + 200 + len(m))
        paired["full_vs_mu"][m] = bootstrap_mean_ci(full_vs_mu, args.bootstrap_iters, args.seed + 300 + len(m))

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "fullcov_checkpoint": args.fullcov_checkpoint,
        "checkpoint_epoch": ckpt.get("epoch"),
        "num_ood_candidates": len(dataset),
        "processed_samples": processed,
        "skipped_samples": skipped,
        "num_samples_per_event": args.num_samples_per_event,
        "griffin_lim_iters": args.griffin_lim_iters,
        "seed": args.seed,
        "device": str(device),
    }

    save_json(out_dir / "manifest.json", manifest)
    save_json(out_dir / "metrics_aggregate_by_mode.json", aggregate)
    save_json(out_dir / "paired_bootstrap.json", paired)

    with open(out_dir / "per_sample_sampling_metrics.jsonl", "w") as f:
        for row in per_sample:
            f.write(json.dumps(row) + "\n")

    # Markdown summary
    lines = []
    lines.append("# FullCov Sampling Ablation Report")
    lines.append("")
    lines.append("Modes:")
    lines.append("- `mu`: deterministic decode (`z=mu`)")
    lines.append("- `diag_sampled`: stochastic decode with diagonal-only covariance")
    lines.append("- `full_sampled`: stochastic decode with full covariance")
    lines.append("")
    lines.append(f"Samples: `{processed}` | Replicates per sample: `{args.num_samples_per_event}`")
    lines.append("")
    lines.append("## Aggregate Metrics")
    lines.append("")
    lines.append("| Mode | SSIM ↑ | S-Corr ↑ | SC ↓ | STA/LTA Err ↓ | LSD ↓ | MR-LSD ↓ | Arias Err ↓ | Env Corr ↑ | DTW ↓ | XCorr ↑ |")
    lines.append("|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for mode in ["mu", "diag_sampled", "full_sampled"]:
        m = aggregate[mode]
        lines.append(
            f"| {mode} | {m['ssim']:.4f} | {m['s_corr']:.4f} | {m['sc']:.4f} | {m['sta_lta_err']:.4f} | "
            f"{m['lsd']:.4f} | {m['mr_lsd']:.4f} | {m['arias_err']:.4f} | {m['env_corr']:.4f} | {m['dtw']:.2f} | {m['xcorr']:.4f} |"
        )

    lines.append("")
    lines.append("## Paired Bootstrap (Signed Improvement)")
    lines.append("")
    lines.append("Positive signed diff means first mode is better.")
    lines.append("")
    for compare in ["full_vs_diag", "diag_vs_mu", "full_vs_mu"]:
        lines.append(f"### {compare}")
        lines.append("")
        lines.append("| Metric | Mean Signed Diff | 95% CI | p(diff>0) |")
        lines.append("|:---|---:|:---:|---:|")
        for metric in METRICS:
            s = paired[compare][metric]
            lines.append(
                f"| {metric} | {s['mean']:+.4f} | [{s['ci95_low']:.4f}, {s['ci95_high']:.4f}] | {s['p_gt0']:.3f} |"
            )
        lines.append("")

    with open(out_dir / "sampling_ablation_report.md", "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[DONE] Sampling ablation outputs: {out_dir}")


if __name__ == "__main__":
    main()
