#!/usr/bin/env python3
"""
Create compact visual summaries for stage2 beta=0.1 logvar-fix comparison.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_map(rows: List[Dict[str, str]], key: str = "run_name") -> Dict[str, Dict[str, str]]:
    return {r[key]: r for r in rows}


def _plot_corr_grid(
    test_dir: Path,
    ood_dir: Path,
    runs: List[str],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(
        nrows=len(runs),
        ncols=2,
        figsize=(8, 2.6 * len(runs)),
        constrained_layout=True,
    )
    if len(runs) == 1:
        axes = np.asarray([axes])

    for i, run in enumerate(runs):
        for j, (split, root) in enumerate([("test", test_dir), ("ood_event", ood_dir)]):
            ax = axes[i, j]
            p = root / run / "corr_agg.npy"
            if not p.exists():
                ax.axis("off")
                ax.set_title(f"{run}\n{split}\nmissing")
                continue
            corr = np.load(p)
            im = ax.imshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
            ax.set_title(f"{run}\n{split}")
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 1:
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                cbar.ax.tick_params(labelsize=7)

    fig.suptitle("corr(Cov_agg) Heatmap Grid", fontsize=12)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_metric_bars(
    test_summary: Dict[str, Dict[str, str]],
    prior_test: Dict[str, Dict[str, str]],
    prior_ood: Dict[str, Dict[str, str]],
    runs: List[str],
    out_path: Path,
) -> None:
    x = np.arange(len(runs))
    width = 0.24

    diag = np.array([float(test_summary[r]["diag_mae"]) for r in runs], dtype=np.float64)
    offdiag = np.array([float(test_summary[r]["offdiag_mean_abs_corr"]) for r in runs], dtype=np.float64)
    klm = np.array([float(test_summary[r]["kl_moment_to_std_normal"]) for r in runs], dtype=np.float64)
    prior_t = np.array([float(prior_test[r]["realism_composite"]) for r in runs], dtype=np.float64)
    prior_o = np.array([float(prior_ood[r]["realism_composite"]) for r in runs], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)

    ax = axes[0]
    ax.bar(x - width, diag, width=width, label="diag_mae (test)")
    ax.bar(x, offdiag, width=width, label="offdiag_abs_corr (test)")
    ax.bar(x + width, klm, width=width, label="KL_moment (test)")
    ax.set_yscale("log")
    ax.set_xticks(x, runs, rotation=25, ha="right")
    ax.set_title("Latent-shape Metrics (log scale, lower better)")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.bar(x - width / 2, prior_t, width=width, label="prior realism (test)")
    ax.bar(x + width / 2, prior_o, width=width, label="prior realism (ood)")
    ax.set_xticks(x, runs, rotation=25, ha="right")
    ax.set_title("Prior Sampling Composite (lower better)")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=8)

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize stage2 logvar-fix comparison artifacts.")
    parser.add_argument(
        "--latent_test_dir",
        default="ML/autoencoder/experiments/LatentShapeVAE/results/latent_shape_test_stage2_beta0p1_logvarfix_compare_v1",
    )
    parser.add_argument(
        "--latent_ood_dir",
        default="ML/autoencoder/experiments/LatentShapeVAE/results/latent_shape_ood_event_stage2_beta0p1_logvarfix_compare_v1",
    )
    parser.add_argument(
        "--prior_test_csv",
        default="ML/autoencoder/experiments/LatentShapeVAE/results/prior_sampling_test_stage2_beta0p1_logvarfix_compare_v1/prior_sampling_realism_summary.csv",
    )
    parser.add_argument(
        "--prior_ood_csv",
        default="ML/autoencoder/experiments/LatentShapeVAE/results/prior_sampling_ood_event_stage2_beta0p1_logvarfix_compare_v1/prior_sampling_realism_summary.csv",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        default=[
            "lsv_stage2_vae_base_ld64_b0p1_s42",
            "lsv_stage2_vae_base_ld64_b0p1_s43",
            "lsv_stage2_vae_base_ld64_b0p1_s43_logvfixv1",
            "lsv_stage2_vae_base_ld64_b0p1_s44",
            "lsv_stage2_vae_base_ld64_b0p1_s44_logvfixv1",
        ],
    )
    parser.add_argument(
        "--output_dir",
        default="ML/autoencoder/experiments/LatentShapeVAE/results/stage2_beta0p1_logvarfix_compare_v1/visuals",
    )
    args = parser.parse_args()

    latent_test_dir = Path(args.latent_test_dir)
    latent_ood_dir = Path(args.latent_ood_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_summary_rows = _read_csv(latent_test_dir / "latent_shape_summary.csv")
    test_summary = _to_map(test_summary_rows)
    prior_test = _to_map(_read_csv(Path(args.prior_test_csv)))
    prior_ood = _to_map(_read_csv(Path(args.prior_ood_csv)))

    runs = [r for r in args.runs if r in test_summary and r in prior_test and r in prior_ood]
    if not runs:
        raise RuntimeError("No overlapping runs found across summary files.")

    heatmap_path = out_dir / "corr_agg_grid_test_ood.png"
    metric_path = out_dir / "metric_bars.png"

    _plot_corr_grid(latent_test_dir, latent_ood_dir, runs, heatmap_path)
    _plot_metric_bars(test_summary, prior_test, prior_ood, runs, metric_path)

    print("[INFO] corr_grid:", heatmap_path.as_posix())
    print("[INFO] metric_bars:", metric_path.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
