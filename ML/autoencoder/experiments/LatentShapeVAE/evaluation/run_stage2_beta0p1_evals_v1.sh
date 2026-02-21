#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$ROOT_DIR"

EXP_DIR="ML/autoencoder/experiments/LatentShapeVAE"
EVAL_DIR="$EXP_DIR/evaluation"
LOG_DIR="$EXP_DIR/logs"
RES_DIR="$EXP_DIR/results"
PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/bin/python}"

mkdir -p "$LOG_DIR"

CKPT_S42="$EXP_DIR/checkpoints/lsv_stage2_vae_base_ld64_b0p1_s42_best.pt"
CKPT_S43="$EXP_DIR/checkpoints/lsv_stage2_vae_base_ld64_b0p1_s43_best.pt"
CKPT_S44="$EXP_DIR/checkpoints/lsv_stage2_vae_base_ld64_b0p1_s44_best.pt"

for ck in "$CKPT_S42" "$CKPT_S43" "$CKPT_S44"; do
  if [[ ! -f "$ck" ]]; then
    echo "[ERROR] Missing checkpoint: $ck"
    exit 1
  fi
done

echo "[$(date '+%F %T')] START stage2 beta=0.1 evaluation batch"
echo "[$(date '+%F %T')] PYTHON_BIN=$PYTHON_BIN"

"$PYTHON_BIN" "$EVAL_DIR/analyze_latent_shape.py" \
  --checkpoints "$CKPT_S42" "$CKPT_S43" "$CKPT_S44" \
  --split test \
  --output_dir "$RES_DIR/latent_shape_test_stage2_beta0p1_seeds_v1"

"$PYTHON_BIN" "$EVAL_DIR/analyze_latent_shape.py" \
  --checkpoints "$CKPT_S42" "$CKPT_S43" "$CKPT_S44" \
  --split ood_event \
  --output_dir "$RES_DIR/latent_shape_ood_event_stage2_beta0p1_seeds_v1"

"$PYTHON_BIN" "$EVAL_DIR/evaluate_prior_sampling.py" \
  --checkpoints "$CKPT_S42" "$CKPT_S43" "$CKPT_S44" \
  --split test \
  --num_real_samples 2000 \
  --num_generated_samples 2000 \
  --output_dir "$RES_DIR/prior_sampling_test_stage2_beta0p1_seeds_v1"

"$PYTHON_BIN" "$EVAL_DIR/evaluate_prior_sampling.py" \
  --checkpoints "$CKPT_S42" "$CKPT_S43" "$CKPT_S44" \
  --split ood_event \
  --num_real_samples 2000 \
  --num_generated_samples 2000 \
  --output_dir "$RES_DIR/prior_sampling_ood_event_stage2_beta0p1_seeds_v1"

python3 - <<'PY'
import csv
from pathlib import Path

base = Path("ML/autoencoder/experiments/LatentShapeVAE/results")
latent_test = base / "latent_shape_test_stage2_beta0p1_seeds_v1" / "latent_shape_summary.csv"
latent_ood = base / "latent_shape_ood_event_stage2_beta0p1_seeds_v1" / "latent_shape_summary.csv"
prior_test = base / "prior_sampling_test_stage2_beta0p1_seeds_v1" / "prior_sampling_realism_summary.csv"
prior_ood = base / "prior_sampling_ood_event_stage2_beta0p1_seeds_v1" / "prior_sampling_realism_summary.csv"

out_dir = base / "stage2_beta0p1_seed_eval_summary_v1"
out_dir.mkdir(parents=True, exist_ok=True)
out_csv = out_dir / "stage2_beta0p1_seed_eval_summary.csv"
out_md = out_dir / "stage2_beta0p1_seed_eval_summary.md"

def read_by_run(path):
    rows = list(csv.DictReader(path.open()))
    return {r["run_name"]: r for r in rows}

lt = read_by_run(latent_test)
lo = read_by_run(latent_ood)
pt = read_by_run(prior_test)
po = read_by_run(prior_ood)
runs = sorted(set(lt) & set(lo) & set(pt) & set(po))

fields = [
    "run_name",
    "test_diag_mae",
    "test_offdiag_mean_abs_corr",
    "test_kl_moment_to_std_normal",
    "test_w2_moment_to_std_normal",
    "ood_diag_mae",
    "ood_offdiag_mean_abs_corr",
    "ood_kl_moment_to_std_normal",
    "ood_w2_moment_to_std_normal",
    "prior_test_realism_composite",
    "prior_ood_realism_composite",
]

rows = []
for rn in runs:
    rows.append(
        {
            "run_name": rn,
            "test_diag_mae": float(lt[rn]["diag_mae"]),
            "test_offdiag_mean_abs_corr": float(lt[rn]["offdiag_mean_abs_corr"]),
            "test_kl_moment_to_std_normal": float(lt[rn]["kl_moment_to_std_normal"]),
            "test_w2_moment_to_std_normal": float(lt[rn]["w2_moment_to_std_normal"]),
            "ood_diag_mae": float(lo[rn]["diag_mae"]),
            "ood_offdiag_mean_abs_corr": float(lo[rn]["offdiag_mean_abs_corr"]),
            "ood_kl_moment_to_std_normal": float(lo[rn]["kl_moment_to_std_normal"]),
            "ood_w2_moment_to_std_normal": float(lo[rn]["w2_moment_to_std_normal"]),
            "prior_test_realism_composite": float(pt[rn]["realism_composite"]),
            "prior_ood_realism_composite": float(po[rn]["realism_composite"]),
        }
    )

with out_csv.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

with out_md.open("w", encoding="utf-8") as f:
    f.write("# Stage2 beta=0.1 Seed Evaluation Summary (v1)\n\n")
    f.write("| Run | test diag_mae | test offdiag | test KL_moment | test W2 | ood diag_mae | ood offdiag | ood KL_moment | ood W2 | prior test comp | prior ood comp |\n")
    f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in rows:
        f.write(
            f"| {r['run_name']} | {r['test_diag_mae']:.4f} | {r['test_offdiag_mean_abs_corr']:.4f} | "
            f"{r['test_kl_moment_to_std_normal']:.4f} | {r['test_w2_moment_to_std_normal']:.4f} | "
            f"{r['ood_diag_mae']:.4f} | {r['ood_offdiag_mean_abs_corr']:.4f} | "
            f"{r['ood_kl_moment_to_std_normal']:.4f} | {r['ood_w2_moment_to_std_normal']:.4f} | "
            f"{r['prior_test_realism_composite']:.4f} | {r['prior_ood_realism_composite']:.4f} |\n"
        )

print("[INFO] summary_csv:", out_csv.as_posix())
print("[INFO] summary_md :", out_md.as_posix())
PY

echo "[$(date '+%F %T')] DONE stage2 beta=0.1 evaluation batch"
