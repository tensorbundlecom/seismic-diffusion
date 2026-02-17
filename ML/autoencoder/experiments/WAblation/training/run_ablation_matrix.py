import argparse
import os
import subprocess
import sys
from pathlib import Path


def build_variants():
    return [
        ("A0_phys_raw_no_station", 0, 0),
        ("A1_phys_raw_station", 0, 1),
        ("A2_phys_w_no_station", 1, 0),
        ("A3_phys_w_station", 1, 1),
    ]


def parse_args():
    parser = argparse.ArgumentParser(description="Run WAblation variant matrix.")
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--max_items", type=int, default=0)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_val_batches", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--start_detached", action="store_true", help="Start each run with nohup in background")
    return parser.parse_args()


def main():
    args = parse_args()
    variants = build_variants()

    train_script = "ML/autoencoder/experiments/WAblation/training/train_w_ablation_external.py"
    logs_dir = Path("ML/autoencoder/experiments/WAblation/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    for variant_name, use_w, use_station in variants:
        cmd = [
            args.python_bin,
            train_script,
            "--variant_name",
            variant_name,
            "--use_mapping_network",
            str(use_w),
            "--use_station_embedding",
            str(use_station),
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
            "--beta",
            str(args.beta),
            "--lr",
            str(args.lr),
            "--seed",
            str(args.seed),
            "--num_workers",
            str(args.num_workers),
            "--max_items",
            str(args.max_items),
            "--max_train_batches",
            str(args.max_train_batches),
            "--max_val_batches",
            str(args.max_val_batches),
        ]

        if args.dry_run:
            print("[DRY-RUN]", " ".join(cmd))
            continue

        if args.start_detached:
            out = logs_dir / f"{variant_name}.out"
            pid = logs_dir / f"{variant_name}.pid"
            shell_cmd = f"nohup {' '.join(cmd)} > {out} 2>&1 & echo $! > {pid}"
            print("[START]", variant_name)
            subprocess.run(shell_cmd, shell=True, check=True)
            continue

        print("[RUN]", variant_name)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

