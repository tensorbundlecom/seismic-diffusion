import argparse
import subprocess
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))



def parse_args():
    p = argparse.ArgumentParser(description="Evaluate model on diverse OOD set using shared evaluator.")
    p.add_argument(
        "--config",
        default="ML/autoencoder/experiments/LegacyCondDiffusion/configs/diffusion_resmlp_default.json",
    )
    p.add_argument(
        "--ood_data_dir",
        default="data/ood_waveforms/post_training_custom/filtered",
    )
    p.add_argument(
        "--ood_catalog",
        default="data/events/ood_catalog_post_training.txt",
    )
    p.add_argument(
        "--output_metrics_json",
        default="ML/autoencoder/experiments/LegacyCondDiffusion/results/diverse_ood_metrics.json",
    )
    p.add_argument("--max_samples", type=int, default=-1)
    return p.parse_args()


def main():
    args = parse_args()
    cmd = [
        "/home/gms/.pyenv/shims/python",
        "ML/autoencoder/experiments/LegacyCondDiffusion/evaluation/evaluate_post_training_custom_ood.py",
        "--config",
        args.config,
        "--ood_data_dir",
        args.ood_data_dir,
        "--ood_catalog",
        args.ood_catalog,
        "--output_metrics_json",
        args.output_metrics_json,
        "--max_samples",
        str(args.max_samples),
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
