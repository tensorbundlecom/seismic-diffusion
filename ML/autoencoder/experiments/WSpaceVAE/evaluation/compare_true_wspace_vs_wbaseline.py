import argparse
import json
import os


def parse_args():
    p = argparse.ArgumentParser(description="Compare TrueWSpaceVAE metrics against LegacyCondBaseline metrics.")
    p.add_argument(
        "--true_wspace_json",
        default="ML/autoencoder/experiments/WSpaceVAE/results/post_training_custom_ood_metrics_true_wspace_vae.json",
    )
    p.add_argument(
        "--legacy_wbaseline_json",
        default="ML/autoencoder/experiments/LegacyCondBaseline/results/post_training_custom_ood_metrics.json",
    )
    p.add_argument(
        "--output_md",
        default="ML/autoencoder/experiments/WSpaceVAE/results/compare_true_wspace_vs_wbaseline.md",
    )
    return p.parse_args()


def fmt(v):
    return "NA" if v is None else f"{v:.4f}"


def main():
    args = parse_args()
    if not os.path.exists(args.true_wspace_json):
        raise FileNotFoundError(args.true_wspace_json)
    if not os.path.exists(args.legacy_wbaseline_json):
        raise FileNotFoundError(args.legacy_wbaseline_json)

    tw = json.load(open(args.true_wspace_json))
    wb = json.load(open(args.legacy_wbaseline_json))
    if "WBaseline" in wb and isinstance(wb["WBaseline"], dict):
        wb = wb["WBaseline"]

    metrics = [
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

    lines = []
    lines.append("# True W-Space VAE vs LegacyCondBaseline")
    lines.append("")
    lines.append("| Metric | TrueWSpaceVAE | LegacyCondBaseline |")
    lines.append("| :--- | ---: | ---: |")
    for m in metrics:
        lines.append(f"| {m} | {fmt(tw.get(m))} | {fmt(wb.get(m))} |")
    lines.append("")
    lines.append("Notes:")
    lines.append("- LegacyCondBaseline uses deterministic condition embedding (legacy `w_cond`).")
    lines.append("- TrueWSpaceVAE uses stochastic latent `u` mapped to `w = M(u)`.")

    os.makedirs(os.path.dirname(args.output_md), exist_ok=True)
    with open(args.output_md, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[INFO] Saved comparison: {args.output_md}")


if __name__ == "__main__":
    main()
