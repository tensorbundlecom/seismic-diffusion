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

from ML.autoencoder.experiments.NonDiagonel.core.model_baseline_geo import GeoConditionalVariationalAutoencoder
from ML.autoencoder.experiments.NonDiagonel.core.model_baseline_geo_small import SmallGeoConditionalVariationalAutoencoder
from ML.autoencoder.experiments.NonDiagonel.core.model_full_cov_geo import GeoFullCovCVAE
from ML.autoencoder.experiments.NonDiagonel.core.model_full_cov_geo_small import SmallGeoFullCovCVAE
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

BASELINE_TYPES = {"baseline_geo", "baseline_geo_small"}
FULLCOV_TYPES = {"fullcov_geo", "fullcov_geo_small"}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reconstruct_signal(magnitude_spec, mag_min, mag_max, fs=100.0, n_iter=64, init_phase=None):
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
        _, waveform = signal.istft(stft_complex, fs=fs, nperseg=256, noverlap=192, nfft=256, boundary="zeros")
        _, _, new_zxx = signal.stft(waveform, fs=fs, nperseg=256, noverlap=192, nfft=256, boundary="zeros")
        if new_zxx.shape != spec.shape:
            min_f = min(new_zxx.shape[0], spec.shape[0])
            min_t = min(new_zxx.shape[1], spec.shape[1])
            next_phase = np.zeros_like(spec, dtype=complex)
            next_phase[:min_f, :min_t] = np.exp(1j * np.angle(new_zxx[:min_f, :min_t]))
            phase = next_phase
        else:
            phase = np.exp(1j * np.angle(new_zxx))
    stft_complex = spec * phase
    _, waveform = signal.istft(stft_complex, fs=fs, nperseg=256, noverlap=192, nfft=256, boundary="zeros")
    return waveform


def calculate_metrics(target_wav, pred_wav, target_spec, pred_spec, fs=100.0):
    out = {}
    s1 = (target_spec - np.min(target_spec)) / (np.max(target_spec) - np.min(target_spec) + 1e-8)
    s2 = (pred_spec - np.min(pred_spec)) / (np.max(pred_spec) - np.min(pred_spec) + 1e-8)
    out["ssim"] = float(ssim(s1, s2, data_range=1.0))
    out["lsd"] = float(np.sqrt(np.mean((np.log(target_spec + 1e-8) - np.log(pred_spec + 1e-8)) ** 2)))
    out["sc"] = float(np.linalg.norm(target_spec - pred_spec) / (np.linalg.norm(target_spec) + 1e-8))
    out["s_corr"] = float(np.corrcoef(target_spec.flatten(), pred_spec.flatten())[0, 1])

    spec_power_target = np.sum(target_spec, axis=0)
    spec_power_pred = np.sum(pred_spec, axis=0)

    def get_sta_lta(power, sta_len=5, lta_len=40):
        sta = np.convolve(power, np.ones(sta_len) / sta_len, mode="same")
        lta = np.convolve(power, np.ones(lta_len) / lta_len, mode="same")
        return sta / (lta + 1e-8)

    sl_target = get_sta_lta(spec_power_target)
    sl_pred = get_sta_lta(spec_power_pred)
    out["sta_lta_err"] = float(np.abs(np.max(sl_target) - np.max(sl_pred)) / (np.max(sl_target) + 1e-8))

    mr_lsd = []
    for n_fft in [64, 128, 512]:
        hop = n_fft // 4
        _, _, z1 = signal.stft(target_wav, fs=fs, nperseg=n_fft, noverlap=n_fft - hop)
        _, _, z2 = signal.stft(pred_wav, fs=fs, nperseg=n_fft, noverlap=n_fft - hop)
        t_spec = np.abs(z1)
        p_spec = np.abs(z2)
        min_f = min(t_spec.shape[0], p_spec.shape[0])
        min_t = min(t_spec.shape[1], p_spec.shape[1])
        mr_lsd.append(np.sqrt(np.mean((np.log(t_spec[:min_f, :min_t] + 1e-8) - np.log(p_spec[:min_f, :min_t] + 1e-8)) ** 2)))
    out["mr_lsd"] = float(np.mean(mr_lsd))

    try:
        from scipy.integrate import trapezoid

        a_target = (np.pi / (2 * 9.81)) * trapezoid(target_wav**2, dx=1 / fs)
        a_pred = (np.pi / (2 * 9.81)) * trapezoid(pred_wav**2, dx=1 / fs)
    except Exception:
        a_target = (np.pi / (2 * 9.81)) * np.trapz(target_wav**2, dx=1 / fs)
        a_pred = (np.pi / (2 * 9.81)) * np.trapz(pred_wav**2, dx=1 / fs)
    out["arias_err"] = float(np.abs(a_target - a_pred) / (np.abs(a_target) + 1e-8))

    env1 = np.abs(hilbert(target_wav))
    env2 = np.abs(hilbert(pred_wav))
    min_len = min(len(env1), len(env2))
    out["env_corr"] = float(np.corrcoef(env1[:min_len], env2[:min_len])[0, 1])

    factor = max(1, len(target_wav) // 500)
    s_target = target_wav[::factor].reshape(-1, 1)
    s_pred = pred_wav[::factor].reshape(-1, 1)
    dtw_dist, _ = fastdtw(s_target, s_pred, dist=euclidean)
    out["dtw"] = float(dtw_dist / len(s_target))

    x1 = (target_wav - np.mean(target_wav)) / (np.std(target_wav) + 1e-8)
    x2 = (pred_wav - np.mean(pred_wav)) / (np.std(pred_wav) + 1e-8)
    min_len = min(len(x1), len(x2))
    xcorr = np.correlate(x1[:min_len], x2[:min_len], mode="full")
    out["xcorr"] = float(np.max(np.abs(xcorr)) / len(x1[:min_len]))
    return out


def to_offdiag_summary(L_tensor):
    L = L_tensor.detach().cpu().float().numpy()[0]
    cov = L @ L.T
    var = np.clip(np.diag(cov), 1e-12, None)
    std = np.sqrt(var)
    corr = cov / (std[:, None] * std[None, :] + 1e-12)
    np.fill_diagonal(corr, 1.0)
    off = corr - np.eye(corr.shape[0], dtype=corr.dtype)
    off_vals = np.abs(off[np.triu_indices(corr.shape[0], 1)])
    return {
        "mean_abs_corr_offdiag": float(np.mean(off_vals)),
        "p95_abs_corr_offdiag": float(np.quantile(off_vals, 0.95)),
        "max_abs_corr_offdiag": float(np.max(off_vals)),
        "offdiag_energy_ratio": float(np.linalg.norm(off, ord="fro") / (np.linalg.norm(corr, ord="fro") + 1e-12)),
    }


def load_model(device, station_count, spec):
    ckpt = torch.load(spec["checkpoint"], map_location=device)
    cfg = ckpt.get("config", {})
    latent_dim = int(cfg.get("latent_dim", 128))
    condition_dim = int(cfg.get("condition_dim", 64))
    num_stations = int(cfg.get("num_stations", station_count))
    mtype = spec["type"]
    if mtype == "baseline_geo":
        model = GeoConditionalVariationalAutoencoder(
            in_channels=3,
            latent_dim=latent_dim,
            num_stations=num_stations,
            condition_dim=condition_dim,
            numeric_condition_dim=5,
        ).to(device)
    elif mtype == "baseline_geo_small":
        model = SmallGeoConditionalVariationalAutoencoder(
            in_channels=3,
            latent_dim=latent_dim,
            num_stations=num_stations,
            condition_dim=condition_dim,
            numeric_condition_dim=5,
        ).to(device)
    elif mtype == "fullcov_geo":
        model = GeoFullCovCVAE(
            in_channels=3,
            latent_dim=latent_dim,
            num_stations=num_stations,
            condition_dim=condition_dim,
            numeric_condition_dim=5,
        ).to(device)
    elif mtype == "fullcov_geo_small":
        model = SmallGeoFullCovCVAE(
            in_channels=3,
            latent_dim=latent_dim,
            num_stations=num_stations,
            condition_dim=condition_dim,
            numeric_condition_dim=5,
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {mtype}")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def write_markdown(path: Path, aggregated: Dict[str, Dict[str, float]], offdiag_summary: Dict[str, Dict[str, float]], manifest: Dict):
    lines = []
    lines.append("# NonDiagonel Model Family Evaluation")
    lines.append("")
    lines.append(f"- Timestamp (UTC): `{manifest['timestamp_utc']}`")
    lines.append(f"- Processed samples: `{manifest['processed_samples']}`")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append("| Model | SSIM ↑ | S-Corr ↑ | SC ↓ | STA/LTA Err ↓ | LSD ↓ | MR-LSD ↓ | Arias Err ↓ | Env Corr ↑ | DTW ↓ | XCorr ↑ |")
    lines.append("|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, m in aggregated.items():
        lines.append(
            f"| {name} | {m['ssim']:.4f} | {m['s_corr']:.4f} | {m['sc']:.4f} | {m['sta_lta_err']:.4f} | "
            f"{m['lsd']:.4f} | {m['mr_lsd']:.4f} | {m['arias_err']:.4f} | {m['env_corr']:.4f} | {m['dtw']:.2f} | {m['xcorr']:.4f} |"
        )

    if offdiag_summary:
        lines.append("")
        lines.append("## FullCov Off-Diagonal Summary")
        lines.append("")
        lines.append("| Model | mean | p95 | max | energy_ratio |")
        lines.append("|:---|---:|---:|---:|---:|")
        for name, s in offdiag_summary.items():
            lines.append(
                f"| {name} | {s['mean_abs_corr_offdiag']:.4f} | {s['p95_abs_corr_offdiag']:.4f} | "
                f"{s['max_abs_corr_offdiag']:.4f} | {s['offdiag_energy_ratio']:.4f} |"
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a model family for NonDiagonel experiment.")
    parser.add_argument("--ood_data_dir", default="data/ood_waveforms/post_training_custom/filtered")
    parser.add_argument("--ood_catalog", default="data/events/ood_catalog_post_training.txt")
    parser.add_argument("--station_list_file", default="data/station_list_external_full.json")
    parser.add_argument("--station_subset_file", default="data/station_list_post_custom.json")
    parser.add_argument("--station_coords_file", default="ML/autoencoder/experiments/NonDiagonel/results/station_coords_external.json")
    parser.add_argument("--condition_stats_file", default="ML/autoencoder/experiments/NonDiagonel/results/condition_stats_external_seed42.json")
    parser.add_argument("--magnitude_col", default="ML")
    parser.add_argument("--griffin_lim_iters", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_items", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--model_specs_json",
        default="ML/autoencoder/experiments/NonDiagonel/results/model_family_specs.json",
        help="JSON list with entries: {name,type,checkpoint}",
    )
    parser.add_argument(
        "--output_dir",
        default="ML/autoencoder/experiments/NonDiagonel/results/evaluations/post_training_custom_geo_repi_s42_20260216_1942/model_family_eval",
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
    with open(args.model_specs_json, "r") as f:
        model_specs = json.load(f)

    models = {}
    for spec in model_specs:
        model, ckpt = load_model(device, len(station_list), spec)
        models[spec["name"]] = {"model": model, "type": spec["type"], "checkpoint_epoch": ckpt.get("epoch")}

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

    per_model_metrics = {name: {m: [] for m in METRICS} for name in models}
    per_model_offdiag = {name: [] for name, d in models.items() if d["type"] in FULLCOV_TYPES}
    per_sample = []
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

            phase_rng = np.random.default_rng(args.seed + (i + 1) * 1007)
            init_phase = np.exp(2j * np.pi * phase_rng.random((129, 111)))

            row = {
                "index": i,
                "file_name": meta.get("file_name"),
                "event_id": meta.get("event_id"),
                "station_name": meta.get("station_name"),
                "metrics": {},
                "offdiag": {},
            }
            for name, entry in models.items():
                model = entry["model"]
                mtype = entry["type"]
                if mtype in BASELINE_TYPES:
                    recon, _, _ = model(spec_in, cond_in, sta_in)
                elif mtype in FULLCOV_TYPES:
                    recon, _, L = model(spec_in, cond_in, sta_in)
                    per_model_offdiag[name].append(to_offdiag_summary(L))
                    row["offdiag"][name] = to_offdiag_summary(L)
                else:
                    raise ValueError(f"Unsupported model type in runtime loop: {mtype}")

                pred_spec = recon[0, 2].cpu().numpy()
                pred_wav = reconstruct_signal(
                    pred_spec,
                    mag_min=mag_min,
                    mag_max=mag_max,
                    fs=100.0,
                    n_iter=args.griffin_lim_iters,
                    init_phase=init_phase,
                )
                min_len = min(len(gt_wav), len(pred_wav))
                metrics = calculate_metrics(gt_wav[:min_len], pred_wav[:min_len], spec_in[0, 2].cpu().numpy(), pred_spec, fs=100.0)
                row["metrics"][name] = metrics
                for m in METRICS:
                    per_model_metrics[name][m].append(metrics[m])

            per_sample.append(row)
            processed += 1
            if processed % 5 == 0:
                print(f"[PROGRESS] processed={processed} skipped={skipped} / total={len(dataset)}", flush=True)

    aggregated = {
        name: {m: float(np.mean(vals)) if vals else None for m, vals in metric_dict.items()}
        for name, metric_dict in per_model_metrics.items()
    }
    offdiag_summary = {}
    for name, rows in per_model_offdiag.items():
        if not rows:
            continue
        offdiag_summary[name] = {
            k: float(np.mean([r[k] for r in rows])) for k in rows[0].keys()
        }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "device": str(device),
        "seed": args.seed,
        "griffin_lim_iters": args.griffin_lim_iters,
        "processed_samples": processed,
        "skipped_samples": skipped,
        "model_specs_json": args.model_specs_json,
        "model_specs": model_specs,
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    with open(out_dir / "metrics_aggregate.json", "w") as f:
        json.dump(aggregated, f, indent=2)
    with open(out_dir / "offdiag_summary.json", "w") as f:
        json.dump(offdiag_summary, f, indent=2)
    with open(out_dir / "per_sample_metrics.jsonl", "w") as f:
        for row in per_sample:
            f.write(json.dumps(row) + "\n")

    write_markdown(out_dir / "summary.md", aggregated, offdiag_summary, manifest)
    print(f"[DONE] Model family evaluation outputs: {out_dir}")


if __name__ == "__main__":
    main()
