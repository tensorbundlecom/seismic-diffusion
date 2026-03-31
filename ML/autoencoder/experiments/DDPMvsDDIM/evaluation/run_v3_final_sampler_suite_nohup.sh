#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="ML/autoencoder/experiments/DDPMvsDDIM"
LOG_DIR="$ROOT_DIR/logs/evaluation"
OUT_ROOT="$ROOT_DIR/results/final_sampler_suite_v3"
PY="/home/gms/miniconda3/bin/python3.12"
MODULE="ML.autoencoder.experiments.DDPMvsDDIM.evaluation.run_sampler_comparison"
DIFFUSION_CKPT="$ROOT_DIR/runs/diffusion/diffusion_eventwise_v3_adaln_vpred/checkpoints/best.pt"
STAGE1_CKPT="$ROOT_DIR/checkpoints/stage1_eventwise_v1_best.pt"
TEST_CACHE="$ROOT_DIR/data_cache/latent_cache_eventwise_v1/test_latent_cache.pt"
STATS_FILE="$ROOT_DIR/data_cache/latent_cache_eventwise_v1/latent_stats.pt"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

mkdir -p "$LOG_DIR" "$OUT_ROOT" /tmp/mpl_ddpmvsddim_final
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/mpl_ddpmvsddim_final

run_one() {
  local steps="$1"
  local outdir="$OUT_ROOT/ddim_$(printf '%03d' "$steps")"
  local step_log="$LOG_DIR/final_sampler_suite_v3_ddim_$(printf '%03d' "$steps")_${TIMESTAMP}.log"

  echo "[QUEUE] START DDIM-${steps} $(date --iso-8601=seconds)"
  echo "[QUEUE] STEP_LOG ${step_log}"
  rm -rf "$outdir"

  "$PY" -m "$MODULE" \
    --diffusion-checkpoint "$DIFFUSION_CKPT" \
    --stage1-checkpoint "$STAGE1_CKPT" \
    --test-cache "$TEST_CACHE" \
    --stats-file "$STATS_FILE" \
    --max-samples 8838 \
    --selection-mode first \
    --ddim-steps "$steps" \
    --ddim-eta 0.0 \
    --sampler-base-seed 1234 \
    --resource-poll-interval-sec 0.5 \
    --save-plots none \
    --plot-count 25 \
    --output-dir "$outdir" \
    > "$step_log" 2>&1

  echo "[QUEUE] END DDIM-${steps} $(date --iso-8601=seconds)"
}

run_one 25
run_one 50
run_one 100

"$PY" - <<'PY'
import json
from pathlib import Path

root = Path("ML/autoencoder/experiments/DDPMvsDDIM/results/final_sampler_suite_v3")
rows = []
for steps in [25, 50, 100]:
    summary_path = root / f"ddim_{steps:03d}" / "summary.json"
    with open(summary_path, "r") as handle:
        summary = json.load(handle)
    rows.append(
        {
            "ddim_steps": steps,
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
            "summary_path": str(summary_path),
        }
    )

suite_summary = {
    "diffusion_checkpoint": "ML/autoencoder/experiments/DDPMvsDDIM/runs/diffusion/diffusion_eventwise_v3_adaln_vpred/checkpoints/best.pt",
    "stage1_checkpoint": "ML/autoencoder/experiments/DDPMvsDDIM/checkpoints/stage1_eventwise_v1_best.pt",
    "ddim_steps_list": [25, 50, 100],
    "rows": rows,
}
with open(root / "suite_summary.json", "w") as handle:
    json.dump(suite_summary, handle, indent=2)

lines = [
    "# Final Sampler Suite v3",
    "",
    "| DDIM steps | DDPM spec_corr | DDIM spec_corr | DDPM LSD | DDIM LSD | DDPM MR-LSD | DDIM MR-LSD | Wall time (min) | Samples/s | DDPM total ms | DDIM total ms | Peak RSS MB | Peak GPU alloc MB | GPU util avg |",
    "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for row in rows:
    lines.append(
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
with open(root / "suite_summary.md", "w") as handle:
    handle.write("\n".join(lines) + "\n")
PY

echo "[QUEUE] SUITE_SUMMARY $OUT_ROOT/suite_summary.json"
echo "[QUEUE] SUITE_MARKDOWN $OUT_ROOT/suite_summary.md"
