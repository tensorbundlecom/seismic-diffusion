import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run the final instrumented DDPM vs DDIM sampler suite for the v3 checkpoint.")
    parser.add_argument(
        "--diffusion-checkpoint",
        default="ML/autoencoder/experiments/DDPMvsDDIM/runs/diffusion/diffusion_eventwise_v3_adaln_vpred/checkpoints/best.pt",
    )
    parser.add_argument(
        "--stage1-checkpoint",
        default="ML/autoencoder/experiments/DDPMvsDDIM/checkpoints/stage1_eventwise_v1_best.pt",
    )
    parser.add_argument(
        "--test-cache",
        default="ML/autoencoder/experiments/DDPMvsDDIM/data_cache/latent_cache_eventwise_v1/test_latent_cache.pt",
    )
    parser.add_argument(
        "--stats-file",
        default="ML/autoencoder/experiments/DDPMvsDDIM/data_cache/latent_cache_eventwise_v1/latent_stats.pt",
    )
    parser.add_argument("--max-samples", type=int, default=8838)
    parser.add_argument("--selection-mode", choices=["first", "evenly_spaced"], default="first")
    parser.add_argument("--ddim-steps-list", type=int, nargs="+", default=[25, 50, 100])
    parser.add_argument("--ddim-eta", type=float, default=0.0)
    parser.add_argument("--sampler-base-seed", type=int, default=1234)
    parser.add_argument("--resource-poll-interval-sec", type=float, default=0.5)
    parser.add_argument("--save-plots", choices=["none", "subset", "all"], default="none")
    parser.add_argument("--plot-count", type=int, default=25)
    parser.add_argument(
        "--output-root",
        default="ML/autoencoder/experiments/DDPMvsDDIM/results/final_sampler_suite_v3",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    suite_rows = []
    for ddim_steps in args.ddim_steps_list:
        run_dir = output_root / f"ddim_{ddim_steps:03d}"
        cmd = [
            sys.executable,
            "-m",
            "ML.autoencoder.experiments.DDPMvsDDIM.evaluation.run_sampler_comparison",
            "--diffusion-checkpoint",
            args.diffusion_checkpoint,
            "--stage1-checkpoint",
            args.stage1_checkpoint,
            "--test-cache",
            args.test_cache,
            "--stats-file",
            args.stats_file,
            "--max-samples",
            str(args.max_samples),
            "--selection-mode",
            args.selection_mode,
            "--ddim-steps",
            str(ddim_steps),
            "--ddim-eta",
            str(args.ddim_eta),
            "--sampler-base-seed",
            str(args.sampler_base_seed),
            "--resource-poll-interval-sec",
            str(args.resource_poll_interval_sec),
            "--save-plots",
            args.save_plots,
            "--plot-count",
            str(args.plot_count),
            "--output-dir",
            str(run_dir),
        ]
        print(f"[INFO] Launching sampler comparison for DDIM-{ddim_steps}: {' '.join(cmd)}", flush=True)
        start = time.perf_counter()
        subprocess.run(cmd, check=True)
        elapsed_sec = time.perf_counter() - start

        with open(run_dir / "summary.json", "r") as handle:
            summary = json.load(handle)
        suite_rows.append(
            {
                "ddim_steps": int(ddim_steps),
                "num_samples": int(summary["num_samples"]),
                "avg_ddpm_spec_corr": float(summary["avg_ddpm_spec_corr"]),
                "avg_ddim_spec_corr": float(summary["avg_ddim_spec_corr"]),
                "avg_ddpm_lsd": float(summary["avg_ddpm_lsd"]),
                "avg_ddim_lsd": float(summary["avg_ddim_lsd"]),
                "avg_ddpm_mr_lsd": float(summary["avg_ddpm_mr_lsd"]),
                "avg_ddim_mr_lsd": float(summary["avg_ddim_mr_lsd"]),
                "wall_time_min": float(summary["runtime"]["evaluation_wall_time_min"]),
                "samples_per_sec": float(summary["runtime"]["samples_per_sec"]),
                "avg_ddpm_total_time_ms": float(summary["runtime"]["avg_ddpm_total_time_ms"]),
                "avg_ddim_total_time_ms": float(summary["runtime"]["avg_ddim_total_time_ms"]),
                "peak_rss_mb": float(summary["resources"]["rss_mb_peak_process"]),
                "peak_gpu_alloc_mb": float(summary["resources"]["torch_peak_allocated_mb"]) if summary["resources"]["torch_peak_allocated_mb"] is not None else None,
                "gpu_util_avg": float(summary["resources"]["gpu_util_percent_avg"]) if summary["resources"]["gpu_util_percent_avg"] is not None else None,
                "outer_wall_time_min": float(elapsed_sec / 60.0),
                "summary_path": str(run_dir / "summary.json"),
            }
        )

    suite_summary = {
        "diffusion_checkpoint": args.diffusion_checkpoint,
        "stage1_checkpoint": args.stage1_checkpoint,
        "ddim_steps_list": [int(x) for x in args.ddim_steps_list],
        "rows": suite_rows,
    }
    with open(output_root / "suite_summary.json", "w") as handle:
        json.dump(suite_summary, handle, indent=2)

    md_lines = [
        "# Final Sampler Suite v3",
        "",
        f"- diffusion checkpoint: `{args.diffusion_checkpoint}`",
        f"- stage1 checkpoint: `{args.stage1_checkpoint}`",
        f"- samples: `{args.max_samples}`",
        "",
        "| DDIM steps | DDPM spec_corr | DDIM spec_corr | DDPM LSD | DDIM LSD | DDPM MR-LSD | DDIM MR-LSD | Wall time (min) | Samples/s | DDPM total ms | DDIM total ms | Peak RSS MB | Peak GPU alloc MB | GPU util avg |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in suite_rows:
        md_lines.append(
            "| "
            f"{row['ddim_steps']} | "
            f"{row['avg_ddpm_spec_corr']:.4f} | "
            f"{row['avg_ddim_spec_corr']:.4f} | "
            f"{row['avg_ddpm_lsd']:.4f} | "
            f"{row['avg_ddim_lsd']:.4f} | "
            f"{row['avg_ddpm_mr_lsd']:.4f} | "
            f"{row['avg_ddim_mr_lsd']:.4f} | "
            f"{row['wall_time_min']:.3f} | "
            f"{row['samples_per_sec']:.3f} | "
            f"{row['avg_ddpm_total_time_ms']:.3f} | "
            f"{row['avg_ddim_total_time_ms']:.3f} | "
            f"{row['peak_rss_mb']:.1f} | "
            f"{row['peak_gpu_alloc_mb']:.1f} | "
            f"{row['gpu_util_avg']:.1f} |"
        )
    with open(output_root / "suite_summary.md", "w") as handle:
        handle.write("\n".join(md_lines) + "\n")

    print(f"[INFO] Final sampler suite summary saved to: {output_root / 'suite_summary.json'}", flush=True)
    print(f"[INFO] Final sampler suite markdown saved to: {output_root / 'suite_summary.md'}", flush=True)


if __name__ == "__main__":
    main()
