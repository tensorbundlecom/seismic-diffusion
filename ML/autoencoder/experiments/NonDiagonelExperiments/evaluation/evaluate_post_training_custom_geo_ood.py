import argparse
import json
import logging
import os
import random
import sys
import time
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
try:
    from skimage.metrics import structural_similarity as ssim
except Exception:
    def ssim(img1, img2, data_range=1.0):
        # Fallback SSIM when scikit-image is unavailable.
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

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))

from ML.autoencoder.experiments.NonDiagonel.core.model_baseline_geo import GeoConditionalVariationalAutoencoder
from ML.autoencoder.experiments.NonDiagonel.core.model_full_cov_geo import GeoFullCovCVAE
from ML.autoencoder.experiments.NonDiagonel.core.stft_dataset_geo import SeismicSTFTDatasetGeoCondition


METRIC_KEYS = [
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


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(str(log_path))
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fmt = logging.Formatter("%(asctime)s | %(message)s")

    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path: Path, payload: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def reconstruct_signal(
    magnitude_spec: np.ndarray,
    mag_min: float = 0.0,
    mag_max: float = 1.0,
    fs: float = 100.0,
    n_iter: int = 64,
    init_phase: np.ndarray = None,
):
    spec = magnitude_spec.copy()
    if mag_max > mag_min:
        spec = spec * (mag_max - mag_min) + mag_min
    spec = np.expm1(spec)

    if init_phase is None:
        phase = np.exp(2j * np.pi * np.random.rand(*spec.shape))
    else:
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


def mean_metrics(metrics_dict: Dict[str, List[float]]) -> Dict[str, float]:
    out = {}
    for k, vals in metrics_dict.items():
        out[k] = float(np.nanmean(vals)) if vals else None
    return out


def offdiag_corr_stats(L_tensor: torch.Tensor) -> Dict[str, float]:
    # L: [B, D, D] lower-cholesky
    L = L_tensor.detach().cpu().float().numpy()
    sample_stats = []
    for i in range(L.shape[0]):
        Li = L[i]
        cov = Li @ Li.T
        var = np.clip(np.diag(cov), 1e-12, None)
        std = np.sqrt(var)
        corr = cov / (std[:, None] * std[None, :] + 1e-12)
        np.fill_diagonal(corr, 1.0)

        off = corr - np.eye(corr.shape[0], dtype=corr.dtype)
        off_vals = np.abs(off[np.triu_indices(corr.shape[0], 1)])
        ratio = float(np.linalg.norm(off, ord="fro") / (np.linalg.norm(corr, ord="fro") + 1e-12))

        sample_stats.append(
            {
                "mean_abs_corr_offdiag": float(np.mean(off_vals)),
                "max_abs_corr_offdiag": float(np.max(off_vals)),
                "p95_abs_corr_offdiag": float(np.quantile(off_vals, 0.95)),
                "offdiag_energy_ratio": ratio,
            }
        )

    keys = list(sample_stats[0].keys()) if sample_stats else []
    summary = {}
    for k in keys:
        vals = [s[k] for s in sample_stats]
        summary[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }
    return summary


def load_model_from_checkpoint(device: torch.device, station_count: int, checkpoint_path: str, model_type: str):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("config", {})
    latent_dim = int(cfg.get("latent_dim", 128))
    condition_dim = int(cfg.get("condition_dim", 64))
    num_stations = int(cfg.get("num_stations", station_count))

    if model_type == "baseline":
        model = GeoConditionalVariationalAutoencoder(
            in_channels=3,
            latent_dim=latent_dim,
            num_stations=num_stations,
            condition_dim=condition_dim,
            numeric_condition_dim=5,
        ).to(device)
    elif model_type == "fullcov":
        model = GeoFullCovCVAE(
            in_channels=3,
            latent_dim=latent_dim,
            num_stations=num_stations,
            condition_dim=condition_dim,
            numeric_condition_dim=5,
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def format_markdown_table(aggregated: Dict[str, Dict[str, float]]) -> str:
    headers = [
        "Model",
        "SSIM ↑",
        "S-Corr ↑",
        "SC ↓",
        "STA/LTA Err ↓",
        "LSD ↓",
        "MR-LSD ↓",
        "Arias Err ↓",
        "Env Corr ↑",
        "DTW ↓",
        "XCorr ↑",
    ]

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + " | ".join([":---"] + [":---:" for _ in headers[1:]]) + " |")

    def fmt(metric: str, v):
        if v is None:
            return "--"
        if metric == "dtw":
            return f"{v:.2f}"
        return f"{v:.4f}"

    for model_name, m in aggregated.items():
        row = [
            model_name,
            fmt("ssim", m.get("ssim")),
            fmt("s_corr", m.get("s_corr")),
            fmt("sc", m.get("sc")),
            fmt("sta_lta_err", m.get("sta_lta_err")),
            fmt("lsd", m.get("lsd")),
            fmt("mr_lsd", m.get("mr_lsd")),
            fmt("arias_err", m.get("arias_err")),
            fmt("env_corr", m.get("env_corr")),
            fmt("dtw", m.get("dtw")),
            fmt("xcorr", m.get("xcorr")),
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_summary_md(
    path: Path,
    eval_name: str,
    processed: int,
    skipped: int,
    aggregated: Dict[str, Dict[str, float]],
    fullcov_corr_summary: Dict,
    manifest: Dict,
):
    lines = []
    lines.append(f"# NonDiagonel Evaluation Summary: `{eval_name}`")
    lines.append("")
    lines.append("## Run Metadata")
    lines.append("")
    lines.append(f"- Timestamp (UTC): `{manifest['timestamp_utc']}`")
    lines.append(f"- Processed samples: `{processed}`")
    lines.append(f"- Skipped samples: `{skipped}`")
    lines.append(f"- OOD data dir: `{manifest['ood_data_dir']}`")
    lines.append(f"- OOD catalog: `{manifest['ood_catalog']}`")
    lines.append(f"- Station subset file: `{manifest['station_subset_file']}`")
    lines.append(f"- Baseline checkpoint: `{manifest['baseline_checkpoint']}`")
    lines.append(f"- FullCov checkpoint: `{manifest['fullcov_checkpoint']}`")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append(format_markdown_table(aggregated))
    lines.append("")
    lines.append("## FullCov Posterior Correlation (Off-Diagonal)")
    lines.append("")
    if not fullcov_corr_summary:
        lines.append("- No FullCov posterior stats collected.")
    else:
        for key, stats in fullcov_corr_summary.items():
            lines.append(
                f"- `{key}`: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
                f"min={stats['min']:.4f}, max={stats['max']:.4f}"
            )

    lines.append("")
    lines.append("## Metric Definitions")
    lines.append("")
    lines.append("- `SSIM`: Spectrogram structural similarity (higher is better).")
    lines.append("- `S-Corr`: Spectral correlation (higher is better).")
    lines.append("- `SC`: Spectral convergence (lower is better).")
    lines.append("- `STA/LTA Err`: Onset energy ratio error (lower is better).")
    lines.append("- `LSD`: Log-spectral distance (lower is better).")
    lines.append("- `MR-LSD`: Multi-resolution log-spectral distance (lower is better).")
    lines.append("- `Arias Err`: Arias intensity error (lower is better).")
    lines.append("- `Env Corr`: Envelope correlation (higher is better).")
    lines.append("- `DTW`: Dynamic time warping distance (lower is better).")
    lines.append("- `XCorr`: Max cross-correlation (higher is better).")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate NonDiagonel GeoBaseline vs GeoFullCov on custom post-training OOD.")
    parser.add_argument("--ood_data_dir", default="data/ood_waveforms/post_training_custom/filtered")
    parser.add_argument("--ood_catalog", default="data/events/ood_catalog_post_training.txt")
    parser.add_argument("--station_list_file", default="data/station_list_external_full.json")
    parser.add_argument("--station_subset_file", default="data/station_list_post_custom.json")
    parser.add_argument("--station_coords_file", default="ML/autoencoder/experiments/NonDiagonel/results/station_coords_external.json")
    parser.add_argument("--condition_stats_file", default="ML/autoencoder/experiments/NonDiagonel/results/condition_stats_external_seed42.json")
    parser.add_argument(
        "--baseline_checkpoint",
        default="ML/autoencoder/experiments/NonDiagonel/checkpoints/baseline_geo_repi_external_s42_20260215_best.pt",
    )
    parser.add_argument(
        "--fullcov_checkpoint",
        default="ML/autoencoder/experiments/NonDiagonel/checkpoints/fullcov_geo_repi_external_s42_20260215_best.pt",
    )
    parser.add_argument("--magnitude_col", default="ML")
    parser.add_argument("--griffin_lim_iters", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=0, help="Optional cap for quick debug.")
    parser.add_argument("--output_root", default="ML/autoencoder/experiments/NonDiagonel/results/evaluations")
    parser.add_argument("--eval_name", default="")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    set_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device if args.device in ["cpu", "cuda"] else "cpu")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_name = args.eval_name or f"post_training_custom_geo_{ts}"
    eval_dir = Path(args.output_root) / eval_name
    eval_dir.mkdir(parents=True, exist_ok=True)

    log_path = Path("ML/autoencoder/experiments/NonDiagonel/logs") / f"{eval_name}.log"
    logger = setup_logger(log_path)

    with open(args.station_list_file, "r") as f:
        station_list = json.load(f)
    station_subset = None
    if args.station_subset_file:
        with open(args.station_subset_file, "r") as f:
            station_subset = set(json.load(f))

    logger.info("Loading dataset...")
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
        logger.info(f"[INFO] max_samples applied: {len(dataset.file_paths)}")

    logger.info("Loading checkpoints...")
    baseline, baseline_ckpt = load_model_from_checkpoint(
        device=device,
        station_count=len(station_list),
        checkpoint_path=args.baseline_checkpoint,
        model_type="baseline",
    )
    fullcov, fullcov_ckpt = load_model_from_checkpoint(
        device=device,
        station_count=len(station_list),
        checkpoint_path=args.fullcov_checkpoint,
        model_type="fullcov",
    )

    results = {
        "BaselineGeo": {k: [] for k in METRIC_KEYS},
        "FullCovGeo": {k: [] for k in METRIC_KEYS},
    }
    fullcov_corr_sample_stats = []
    per_sample = []
    processed = 0
    skipped = 0

    start = time.time()
    with torch.no_grad():
        for i in range(len(dataset)):
            spec, cond_numeric, station_idx, meta = dataset[i]
            if "error" in meta:
                skipped += 1
                continue

            station_name = meta.get("station_name", "")
            if station_subset is not None and station_name not in station_subset:
                skipped += 1
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

            spec_in = torch.from_numpy(mag_spec).unsqueeze(0).unsqueeze(0).to(device).float().repeat(1, 3, 1, 1)
            if spec_in.shape[2:] != (129, 111):
                spec_in = torch.nn.functional.interpolate(spec_in, size=(129, 111), mode="bilinear", align_corners=False)

            cond_in = cond_numeric.unsqueeze(0).to(device)
            sta_in = station_idx.unsqueeze(0).to(device)

            r_base, _, _ = baseline(spec_in, cond_in, sta_in)
            r_full, _, L = fullcov(spec_in, cond_in, sta_in)

            orig_spec = spec_in[0, 2].cpu().numpy()
            base_spec = r_base[0, 2].cpu().numpy()
            full_spec = r_full[0, 2].cpu().numpy()

            phase_rng = np.random.default_rng(args.seed + (i + 1) * 1009)
            init_phase = np.exp(2j * np.pi * phase_rng.random(orig_spec.shape))

            wav_base = reconstruct_signal(
                base_spec,
                mag_min=mag_min,
                mag_max=mag_max,
                fs=100.0,
                n_iter=args.griffin_lim_iters,
                init_phase=init_phase,
            )
            wav_full = reconstruct_signal(
                full_spec,
                mag_min=mag_min,
                mag_max=mag_max,
                fs=100.0,
                n_iter=args.griffin_lim_iters,
                init_phase=init_phase,
            )

            min_len_base = min(len(gt_wav), len(wav_base))
            min_len_full = min(len(gt_wav), len(wav_full))
            m_base = calculate_seismic_metrics(
                gt_wav[:min_len_base],
                wav_base[:min_len_base],
                orig_spec,
                base_spec,
                fs=100.0,
            )
            m_full = calculate_seismic_metrics(
                gt_wav[:min_len_full],
                wav_full[:min_len_full],
                orig_spec,
                full_spec,
                fs=100.0,
            )

            for k in METRIC_KEYS:
                results["BaselineGeo"][k].append(m_base[k])
                results["FullCovGeo"][k].append(m_full[k])

            corr_stats = offdiag_corr_stats(L)
            corr_row = {k: v["mean"] for k, v in corr_stats.items()} if corr_stats else {}
            fullcov_corr_sample_stats.append(corr_row)

            per_sample.append(
                {
                    "index": i,
                    "file_name": meta.get("file_name"),
                    "event_id": meta.get("event_id"),
                    "station_name": station_name,
                    "metrics": {
                        "BaselineGeo": m_base,
                        "FullCovGeo": m_full,
                    },
                    "fullcov_posterior": corr_row,
                }
            )
            processed += 1

            if processed % 5 == 0:
                elapsed = time.time() - start
                logger.info(f"[PROGRESS] processed={processed} skipped={skipped} elapsed={elapsed:.1f}s")

    aggregated = {model_name: mean_metrics(model_metrics) for model_name, model_metrics in results.items()}

    # Aggregate fullcov off-diagonal stats across processed samples
    fullcov_corr_summary = {}
    if fullcov_corr_sample_stats:
        keys = fullcov_corr_sample_stats[0].keys()
        for k in keys:
            vals = np.array([row[k] for row in fullcov_corr_sample_stats], dtype=np.float64)
            fullcov_corr_summary[k] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }

    manifest = {
        "eval_name": eval_name,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "device": str(device),
        "seed": args.seed,
        "griffin_lim_iters": args.griffin_lim_iters,
        "magnitude_col": args.magnitude_col,
        "ood_data_dir": args.ood_data_dir,
        "ood_catalog": args.ood_catalog,
        "station_list_file": args.station_list_file,
        "station_subset_file": args.station_subset_file,
        "station_coords_file": args.station_coords_file,
        "condition_stats_file": args.condition_stats_file,
        "baseline_checkpoint": args.baseline_checkpoint,
        "fullcov_checkpoint": args.fullcov_checkpoint,
        "baseline_checkpoint_epoch": baseline_ckpt.get("epoch"),
        "fullcov_checkpoint_epoch": fullcov_ckpt.get("epoch"),
        "processed_samples": processed,
        "skipped_samples": skipped,
        "total_candidates": len(dataset),
    }

    save_json(eval_dir / "manifest.json", manifest)
    save_json(eval_dir / "metrics_aggregate.json", aggregated)
    save_json(eval_dir / "fullcov_posterior_offdiag_summary.json", fullcov_corr_summary)

    with open(eval_dir / "metrics_per_sample.jsonl", "w") as f:
        for row in per_sample:
            f.write(json.dumps(row) + "\n")

    write_summary_md(
        path=eval_dir / "summary.md",
        eval_name=eval_name,
        processed=processed,
        skipped=skipped,
        aggregated=aggregated,
        fullcov_corr_summary=fullcov_corr_summary,
        manifest=manifest,
    )

    elapsed = time.time() - start
    logger.info(f"[DONE] eval={eval_name} processed={processed} skipped={skipped} elapsed={elapsed:.1f}s")
    logger.info(f"[DONE] Outputs: {eval_dir}")


if __name__ == "__main__":
    main()
