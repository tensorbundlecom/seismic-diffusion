import argparse
from pathlib import Path
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))


import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Visualize generated waveform/spec pairs from npy outputs.")
    p.add_argument(
        "--input_dir",
        default="ML/autoencoder/experiments/LegacyCondDiffusion/results/generated_samples",
    )
    p.add_argument(
        "--output_dir",
        default="ML/autoencoder/experiments/LegacyCondDiffusion/visualizations/generated_samples",
    )
    p.add_argument("--max_items", type=int, default=20)
    return p.parse_args()


def main():
    args = parse_args()
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spec_files = sorted(in_dir.glob("*_spec.npy"))
    count = 0
    for sf in spec_files:
        if count >= args.max_items:
            break
        stem = sf.stem.replace("_spec", "")
        wf = in_dir / f"{stem}_wav.npy"
        if not wf.exists():
            continue

        spec = np.load(sf)
        wav = np.load(wf)

        fig, axes = plt.subplots(2, 1, figsize=(10, 6))
        axes[0].plot(wav, linewidth=0.8)
        axes[0].set_title(f"{stem} - waveform")
        im = axes[1].imshow(spec, aspect="auto", origin="lower")
        axes[1].set_title(f"{stem} - spectrogram (normalized)")
        fig.colorbar(im, ax=axes[1], fraction=0.02, pad=0.02)
        fig.tight_layout()
        fig.savefig(out_dir / f"{stem}.png", dpi=150)
        plt.close(fig)
        count += 1

    print(f"[INFO] Saved {count} visualization files to {out_dir}")


if __name__ == "__main__":
    main()

