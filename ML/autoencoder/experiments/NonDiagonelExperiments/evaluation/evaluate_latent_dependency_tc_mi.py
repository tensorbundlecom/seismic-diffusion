import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))

from ML.autoencoder.experiments.NonDiagonel.core.model_baseline_geo import GeoConditionalVariationalAutoencoder
from ML.autoencoder.experiments.NonDiagonel.core.model_baseline_geo_small import SmallGeoConditionalVariationalAutoencoder
from ML.autoencoder.experiments.NonDiagonel.core.model_full_cov_geo import GeoFullCovCVAE
from ML.autoencoder.experiments.NonDiagonel.core.model_full_cov_geo_small import SmallGeoFullCovCVAE
from ML.autoencoder.experiments.NonDiagonel.core.stft_dataset_geo import SeismicSTFTDatasetGeoCondition


BASELINE_TYPES = {"baseline_geo", "baseline_geo_small"}
FULLCOV_TYPES = {"fullcov_geo", "fullcov_geo_small"}
HI_METRICS = {"ssim", "s_corr", "env_corr", "xcorr"}
LO_METRICS = {"sc", "sta_lta_err", "lsd", "mr_lsd", "arias_err", "dtw"}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def stable_logdet_spd(cov: np.ndarray, eps: float = 1e-8) -> float:
    eigvals = np.linalg.eigvalsh((cov + cov.T) * 0.5)
    eigvals = np.clip(eigvals, eps, None)
    return float(np.sum(np.log(eigvals)))


def gaussian_tc_from_cov(cov: np.ndarray, eps: float = 1e-8) -> float:
    diag = np.clip(np.diag(cov), eps, None)
    return float(0.5 * (np.sum(np.log(diag)) - stable_logdet_spd(cov, eps=eps)))


def cov_to_corr(cov: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.clip(np.diag(cov), eps, None)
    s = np.sqrt(v)
    corr = cov / (s[:, None] * s[None, :] + eps)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def offdiag_summary(corr: np.ndarray) -> Dict[str, float]:
    off = corr - np.eye(corr.shape[0], dtype=corr.dtype)
    vals = np.abs(off[np.triu_indices(corr.shape[0], 1)])
    return {
        "mean_abs_corr_offdiag": float(np.mean(vals)),
        "p95_abs_corr_offdiag": float(np.quantile(vals, 0.95)),
        "max_abs_corr_offdiag": float(np.max(vals)),
        "offdiag_energy_ratio": float(np.linalg.norm(off, ord="fro") / (np.linalg.norm(corr, ord="fro") + 1e-12)),
    }


def pairwise_mi_summary_from_corr(corr: np.ndarray) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    upper = np.triu_indices(corr.shape[0], 1)
    rho = np.clip(np.abs(corr[upper]), 0.0, 0.999999)
    mi = -0.5 * np.log(1.0 - rho * rho)
    order = np.argsort(mi)[::-1]
    top = []
    for idx in order[:10]:
        i = int(upper[0][idx])
        j = int(upper[1][idx])
        top.append({"i": i, "j": j, "abs_corr": float(rho[idx]), "gauss_mi": float(mi[idx])})
    return {
        "pairwise_mi_mean": float(np.mean(mi)),
        "pairwise_mi_p95": float(np.quantile(mi, 0.95)),
        "pairwise_mi_max": float(np.max(mi)),
    }, top


def random_orthogonal(dim: int, rng: np.random.Generator) -> np.ndarray:
    a = rng.standard_normal((dim, dim))
    q, r = np.linalg.qr(a)
    s = np.sign(np.diag(r))
    s[s == 0] = 1.0
    q = q * s
    return q


def summarize_rotation(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def normalize_score(value: float, min_v: float, max_v: float, higher_is_better: bool) -> float:
    if max_v <= min_v:
        return 0.5
    x = (value - min_v) / (max_v - min_v)
    return float(x if higher_is_better else (1.0 - x))


def plot_bar(values: Dict[str, float], title: str, ylabel: str, out_path: Path):
    names = list(values.keys())
    y = [values[n] for n in names]
    plt.figure(figsize=(12, 4))
    plt.bar(np.arange(len(names)), y)
    plt.xticks(np.arange(len(names)), names, rotation=20, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_box(values: Dict[str, List[float]], title: str, ylabel: str, out_path: Path):
    names = list(values.keys())
    data = [values[n] for n in names]
    plt.figure(figsize=(12, 4))
    plt.boxplot(data, tick_labels=names, showfliers=False)
    plt.xticks(rotation=20, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_scatter(scores: Dict[str, Dict[str, float]], out_path: Path):
    plt.figure(figsize=(6, 5))
    for name, row in scores.items():
        x = row["quality_score"]
        y = row["independence_score"]
        plt.scatter([x], [y], s=60)
        plt.text(x + 0.005, y + 0.005, name, fontsize=8)
    plt.xlabel("Quality Score (higher better)")
    plt.ylabel("Independence Score (higher better)")
    plt.title("Pareto View: Quality vs Independence")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_corr_heatmap(corr: np.ndarray, title: str, out_path: Path):
    plt.figure(figsize=(6, 5))
    plt.imshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Latent dependency analysis with TC/MI and basis-rotation controls.")
    parser.add_argument("--ood_data_dir", default="data/ood_waveforms/post_training_custom/filtered")
    parser.add_argument("--ood_catalog", default="data/events/ood_catalog_post_training.txt")
    parser.add_argument("--station_list_file", default="data/station_list_external_full.json")
    parser.add_argument("--station_subset_file", default="data/station_list_post_custom.json")
    parser.add_argument("--station_coords_file", default="ML/autoencoder/experiments/NonDiagonel/results/station_coords_external.json")
    parser.add_argument("--condition_stats_file", default="ML/autoencoder/experiments/NonDiagonel/results/condition_stats_external_seed42.json")
    parser.add_argument("--magnitude_col", default="ML")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_items", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_rotations", type=int, default=24)
    parser.add_argument(
        "--model_specs_json",
        default="ML/autoencoder/experiments/NonDiagonel/results/model_family_specs_small_phase_20260217.json",
    )
    parser.add_argument(
        "--ood_metrics_json",
        default="ML/autoencoder/experiments/NonDiagonel/results/evaluations/post_training_custom_geo_repi_s42_20260216_1942/model_family_eval_small_phase_20260217_1216/metrics_aggregate.json",
    )
    parser.add_argument(
        "--output_dir",
        default="ML/autoencoder/experiments/NonDiagonel/results/evaluations/post_training_custom_geo_repi_s42_20260216_1942/latent_dependency_tc_mi_20260217",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device if args.device in ["cpu", "cuda"] else "cpu")

    with open(args.station_list_file, "r") as f:
        station_list = json.load(f)
    station_subset = None
    if args.station_subset_file and os.path.exists(args.station_subset_file):
        with open(args.station_subset_file, "r") as f:
            station_subset = set(json.load(f))

    with open(args.model_specs_json, "r") as f:
        model_specs = json.load(f)

    models = {}
    for spec in model_specs:
        model, ckpt = load_model(device, len(station_list), spec)
        models[spec["name"]] = {
            "model": model,
            "type": spec["type"],
            "checkpoint": spec["checkpoint"],
            "checkpoint_epoch": int(ckpt.get("epoch", -1)),
            "latent_dim": int(ckpt.get("config", {}).get("latent_dim", 128)),
        }

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

    accum = {}
    for name, entry in models.items():
        d = entry["latent_dim"]
        accum[name] = {
            "mu_rows": [],
            "sum_sigma": np.zeros((d, d), dtype=np.float64),
            "posterior_tc": [],
        }

    per_sample_rows = []
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

            spec_in = spec.unsqueeze(0).to(device)
            if spec_in.shape[2:] != (129, 111):
                spec_in = torch.nn.functional.interpolate(spec_in, size=(129, 111), mode="bilinear", align_corners=False)
            cond_in = cond_numeric.unsqueeze(0).to(device)
            sta_in = station_idx.unsqueeze(0).to(device)

            row = {
                "index": i,
                "file_name": meta.get("file_name"),
                "event_id": meta.get("event_id"),
                "station_name": meta.get("station_name"),
                "posterior_tc": {},
            }

            for name, entry in models.items():
                model = entry["model"]
                mtype = entry["type"]
                if mtype in BASELINE_TYPES:
                    _, mu, logvar = model(spec_in, cond_in, sta_in)
                    mu_np = mu[0].detach().cpu().numpy().astype(np.float64)
                    var_np = np.exp(logvar[0].detach().cpu().numpy().astype(np.float64))
                    sigma = np.diag(var_np)
                elif mtype in FULLCOV_TYPES:
                    _, mu, L = model(spec_in, cond_in, sta_in)
                    mu_np = mu[0].detach().cpu().numpy().astype(np.float64)
                    L_np = L[0].detach().cpu().numpy().astype(np.float64)
                    sigma = L_np @ L_np.T
                else:
                    raise ValueError(f"Unsupported model type: {mtype}")

                tc_post = gaussian_tc_from_cov(sigma)
                accum[name]["mu_rows"].append(mu_np)
                accum[name]["sum_sigma"] += sigma
                accum[name]["posterior_tc"].append(tc_post)
                row["posterior_tc"][name] = float(tc_post)

            per_sample_rows.append(row)
            processed += 1
            if processed % 5 == 0:
                print(f"[PROGRESS] processed={processed} skipped={skipped} / total={len(dataset)}", flush=True)

    out_dir = Path(args.output_dir)
    plots_dir = out_dir / "plots"
    heatmap_dir = plots_dir / "corr_heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    rotation_raw = {}
    for name, dct in accum.items():
        z = np.stack(dct["mu_rows"], axis=0)  # [N, D]
        n = z.shape[0]
        cov_mu = np.cov(z, rowvar=False) if n > 1 else np.zeros_like(dct["sum_sigma"])
        mean_sigma = dct["sum_sigma"] / max(n, 1)
        sigma_agg = (cov_mu + mean_sigma + (cov_mu + mean_sigma).T) * 0.5

        corr_agg = cov_to_corr(sigma_agg)
        off = offdiag_summary(corr_agg)
        tc_agg = gaussian_tc_from_cov(sigma_agg)
        mi_summary, mi_top_pairs = pairwise_mi_summary_from_corr(corr_agg)
        eigvals = np.linalg.eigvalsh(sigma_agg)
        eigvals = np.clip(eigvals, 1e-12, None)
        p = eigvals / np.sum(eigvals)
        eff_rank = float(np.exp(-(p * np.log(p + 1e-12)).sum()))
        cond_num = float(np.max(eigvals) / np.min(eigvals))

        # Basis-rotation stress: same covariance under random orthogonal basis.
        rot_vals = {
            "tc_agg": [],
            "mean_abs_corr_offdiag": [],
            "p95_abs_corr_offdiag": [],
            "max_abs_corr_offdiag": [],
            "offdiag_energy_ratio": [],
        }
        for _ in range(args.num_rotations):
            q = random_orthogonal(sigma_agg.shape[0], rng)
            s_rot = q @ sigma_agg @ q.T
            c_rot = cov_to_corr(s_rot)
            o_rot = offdiag_summary(c_rot)
            rot_vals["tc_agg"].append(gaussian_tc_from_cov(s_rot))
            for k in ["mean_abs_corr_offdiag", "p95_abs_corr_offdiag", "max_abs_corr_offdiag", "offdiag_energy_ratio"]:
                rot_vals[k].append(o_rot[k])
        rotation_raw[name] = rot_vals

        summary[name] = {
            "samples": int(n),
            "latent_dim": int(z.shape[1]),
            "type": models[name]["type"],
            "checkpoint": models[name]["checkpoint"],
            "checkpoint_epoch": models[name]["checkpoint_epoch"],
            "posterior_tc_mean": float(np.mean(dct["posterior_tc"])),
            "posterior_tc_p95": float(np.quantile(dct["posterior_tc"], 0.95)),
            "posterior_tc_max": float(np.max(dct["posterior_tc"])),
            "tc_agg": float(tc_agg),
            "offdiag": off,
            "pairwise_mi": mi_summary,
            "top_mi_pairs": mi_top_pairs,
            "invariants": {
                "trace": float(np.sum(eigvals)),
                "logdet": float(np.sum(np.log(eigvals))),
                "effective_rank": eff_rank,
                "condition_number": cond_num,
            },
            "rotation_stress": {k: summarize_rotation(v) for k, v in rot_vals.items()},
        }

        plot_corr_heatmap(corr_agg, f"{name} | aggregated corr", heatmap_dir / f"{name}_corr_heatmap.png")

    with open(out_dir / "latent_dependency_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "rotation_stress_raw.json", "w") as f:
        json.dump(rotation_raw, f, indent=2)
    with open(out_dir / "per_sample_posterior_tc.jsonl", "w") as f:
        for row in per_sample_rows:
            f.write(json.dumps(row) + "\n")

    # Optional OOD quality integration (Pareto view).
    ood = None
    if args.ood_metrics_json and os.path.exists(args.ood_metrics_json):
        with open(args.ood_metrics_json, "r") as f:
            ood = json.load(f)

    scores = {}
    if ood is not None:
        model_names = [n for n in summary.keys() if n in ood]
        ood_metric_values = {}
        for met in sorted(HI_METRICS | LO_METRICS):
            vals = [ood[n][met] for n in model_names]
            ood_metric_values[met] = (min(vals), max(vals))

        dep_fields = ["tc_agg", "offdiag_mean", "offdiag_p95", "pairwise_mi_mean"]
        dep_values = {
            "tc_agg": [summary[n]["tc_agg"] for n in model_names],
            "offdiag_mean": [summary[n]["offdiag"]["mean_abs_corr_offdiag"] for n in model_names],
            "offdiag_p95": [summary[n]["offdiag"]["p95_abs_corr_offdiag"] for n in model_names],
            "pairwise_mi_mean": [summary[n]["pairwise_mi"]["pairwise_mi_mean"] for n in model_names],
        }
        dep_ranges = {k: (min(v), max(v)) for k, v in dep_values.items()}

        for name in model_names:
            q_parts = []
            for met in sorted(HI_METRICS | LO_METRICS):
                min_v, max_v = ood_metric_values[met]
                q_parts.append(normalize_score(ood[name][met], min_v, max_v, higher_is_better=(met in HI_METRICS)))
            quality_score = float(np.mean(q_parts))

            i_parts = [
                normalize_score(summary[name]["tc_agg"], *dep_ranges["tc_agg"], higher_is_better=False),
                normalize_score(summary[name]["offdiag"]["mean_abs_corr_offdiag"], *dep_ranges["offdiag_mean"], higher_is_better=False),
                normalize_score(summary[name]["offdiag"]["p95_abs_corr_offdiag"], *dep_ranges["offdiag_p95"], higher_is_better=False),
                normalize_score(summary[name]["pairwise_mi"]["pairwise_mi_mean"], *dep_ranges["pairwise_mi_mean"], higher_is_better=False),
            ]
            independence_score = float(np.mean(i_parts))
            scores[name] = {"quality_score": quality_score, "independence_score": independence_score}

        # Pareto non-dominated set in (quality max, independence max)
        names = list(scores.keys())
        pareto = []
        for i, a in enumerate(names):
            dominated = False
            for j, b in enumerate(names):
                if i == j:
                    continue
                qa, ia = scores[a]["quality_score"], scores[a]["independence_score"]
                qb, ib = scores[b]["quality_score"], scores[b]["independence_score"]
                if (qb >= qa and ib >= ia) and (qb > qa or ib > ia):
                    dominated = True
                    break
            if not dominated:
                pareto.append(a)
        for n in scores:
            scores[n]["pareto_nondominated"] = bool(n in pareto)

    # Plots
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_bar({n: summary[n]["tc_agg"] for n in summary}, "Aggregated Gaussian TC", "TC (lower better)", plots_dir / "tc_agg_bar.png")
    plot_bar(
        {n: summary[n]["offdiag"]["mean_abs_corr_offdiag"] for n in summary},
        "Aggregated Off-Diagonal Mean |Corr|",
        "mean |corr| offdiag (lower better)",
        plots_dir / "offdiag_mean_bar.png",
    )
    plot_bar(
        {n: summary[n]["pairwise_mi"]["pairwise_mi_mean"] for n in summary},
        "Aggregated Pairwise Gaussian MI (Mean)",
        "mean pairwise MI (lower better)",
        plots_dir / "pairwise_mi_mean_bar.png",
    )
    plot_box(
        {n: rotation_raw[n]["mean_abs_corr_offdiag"] for n in summary},
        "Rotation Stress: Offdiag Mean |Corr|",
        "mean |corr| offdiag",
        plots_dir / "rotation_stress_offdiag_mean_box.png",
    )
    plot_box(
        {n: rotation_raw[n]["tc_agg"] for n in summary},
        "Rotation Stress: TC",
        "TC",
        plots_dir / "rotation_stress_tc_box.png",
    )
    if scores:
        plot_scatter(scores, plots_dir / "pareto_quality_vs_independence.png")
        with open(out_dir / "pareto_scores.json", "w") as f:
            json.dump(scores, f, indent=2)

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "device": str(device),
        "seed": args.seed,
        "processed_samples": processed,
        "skipped_samples": skipped,
        "num_rotations": args.num_rotations,
        "model_specs_json": args.model_specs_json,
        "ood_metrics_json": args.ood_metrics_json if ood is not None else None,
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Markdown report
    lines = []
    lines.append("# Latent Dependency Report (TC/MI + Basis Control)")
    lines.append("")
    lines.append(f"- Timestamp (UTC): `{manifest['timestamp_utc']}`")
    lines.append(f"- Processed samples: `{processed}`")
    lines.append(f"- Skipped samples: `{skipped}`")
    lines.append(f"- Rotations per model: `{args.num_rotations}`")
    lines.append("")
    lines.append("## Aggregated Dependency Summary")
    lines.append("")
    lines.append("| Model | TC_agg ↓ | Offdiag Mean ↓ | Offdiag p95 ↓ | Pairwise MI Mean ↓ | Posterior TC Mean ↓ |")
    lines.append("|:---|---:|---:|---:|---:|---:|")
    for name, s in summary.items():
        lines.append(
            f"| {name} | {s['tc_agg']:.4f} | {s['offdiag']['mean_abs_corr_offdiag']:.4f} | "
            f"{s['offdiag']['p95_abs_corr_offdiag']:.4f} | {s['pairwise_mi']['pairwise_mi_mean']:.4f} | {s['posterior_tc_mean']:.4f} |"
        )

    lines.append("")
    lines.append("## Basis-Rotation Stress (Same Covariance, Rotated Coordinates)")
    lines.append("")
    lines.append("| Model | Offdiag Mean Min | Offdiag Mean Max | TC Min | TC Max |")
    lines.append("|:---|---:|---:|---:|---:|")
    for name, s in summary.items():
        rs = s["rotation_stress"]
        lines.append(
            f"| {name} | {rs['mean_abs_corr_offdiag']['min']:.4f} | {rs['mean_abs_corr_offdiag']['max']:.4f} | "
            f"{rs['tc_agg']['min']:.4f} | {rs['tc_agg']['max']:.4f} |"
        )

    if ood is not None:
        lines.append("")
        lines.append("## OOD + Dependency Combined View")
        lines.append("")
        lines.append("| Model | Quality Score ↑ | Independence Score ↑ | Pareto Non-Dominated |")
        lines.append("|:---|---:|---:|:---:|")
        for name, row in scores.items():
            lines.append(
                f"| {name} | {row['quality_score']:.4f} | {row['independence_score']:.4f} | "
                f"{'yes' if row['pareto_nondominated'] else 'no'} |"
            )

    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append("- `latent_dependency_summary.json`")
    lines.append("- `rotation_stress_raw.json`")
    lines.append("- `per_sample_posterior_tc.jsonl`")
    if ood is not None:
        lines.append("- `pareto_scores.json`")
    lines.append("- `plots/tc_agg_bar.png`")
    lines.append("- `plots/offdiag_mean_bar.png`")
    lines.append("- `plots/pairwise_mi_mean_bar.png`")
    lines.append("- `plots/rotation_stress_offdiag_mean_box.png`")
    lines.append("- `plots/rotation_stress_tc_box.png`")
    if ood is not None:
        lines.append("- `plots/pareto_quality_vs_independence.png`")
    lines.append("- `plots/corr_heatmaps/*.png`")
    with open(out_dir / "report.md", "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[DONE] latent dependency report: {out_dir}")


if __name__ == "__main__":
    main()
