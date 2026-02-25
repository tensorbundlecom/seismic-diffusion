"""Visualization utilities for experiments2/exp001."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Mapping

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import signal
from torch.utils.data import DataLoader

from .dataset import ExternalHHComplexSTFTDataset, collate_exp001, prepare_exp001_artifacts
from .model import CVAEComplexSTFT
from .utils import build_seed_bank, load_config, load_json, resolve_device


def _complex_to_mag(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.square(x[0]) + np.square(x[1]) + 1e-8)


def _istft_from_complex(
    x: np.ndarray,
    cfg: Mapping[str, Any],
    stft_scale: float,
) -> np.ndarray:
    z = (x[0] + 1j * x[1]) * float(stft_scale)
    if bool(cfg["stft"]["drop_nyquist"]):
        # Reconstruct dropped Nyquist row as zeros.
        z = np.concatenate([z, np.zeros((1, z.shape[1]), dtype=z.dtype)], axis=0)

    fs = float(cfg["data"]["sampling_rate_hz"])
    n_fft = int(cfg["stft"]["n_fft"])
    win = int(cfg["stft"]["win_length"])
    hop = int(cfg["stft"]["hop_length"])
    noverlap = win - hop
    _, wav = signal.istft(
        z,
        fs=fs,
        nperseg=win,
        noverlap=noverlap,
        nfft=n_fft,
        input_onesided=True,
        boundary=True,
    )
    return wav.astype(np.float32, copy=False)


def _plot_triplet(
    out_path: Path,
    target: np.ndarray,
    recon: np.ndarray,
    cond: np.ndarray,
    cfg: Mapping[str, Any],
    stft_scale: float,
    title_prefix: str,
) -> None:
    t_mag = _complex_to_mag(target)
    r_mag = _complex_to_mag(recon)
    c_mag = _complex_to_mag(cond)

    t_wav = _istft_from_complex(target, cfg, stft_scale)
    r_wav = _istft_from_complex(recon, cfg, stft_scale)
    c_wav = _istft_from_complex(cond, cfg, stft_scale)

    fig, axs = plt.subplots(3, 2, figsize=(14, 9))
    ims = [
        axs[0, 0].imshow(np.log1p(t_mag), origin="lower", aspect="auto"),
        axs[1, 0].imshow(np.log1p(r_mag), origin="lower", aspect="auto"),
        axs[2, 0].imshow(np.log1p(c_mag), origin="lower", aspect="auto"),
    ]
    axs[0, 0].set_title("Target STFT log|X|")
    axs[1, 0].set_title("Reconstruction STFT log|X|")
    axs[2, 0].set_title("Condition-only STFT log|X|")
    for i in range(3):
        fig.colorbar(ims[i], ax=axs[i, 0], fraction=0.046, pad=0.04)

    axs[0, 1].plot(t_wav, lw=0.8)
    axs[0, 1].set_title("Target waveform")
    axs[1, 1].plot(r_wav, lw=0.8)
    axs[1, 1].set_title("Reconstruction waveform")
    axs[2, 1].plot(c_wav, lw=0.8)
    axs[2, 1].set_title("Condition-only waveform")

    for ax_row in axs:
        for ax in ax_row:
            ax.grid(alpha=0.2)
    fig.suptitle(title_prefix)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize exp001 model outputs.")
    p.add_argument(
        "--config",
        default="ML/autoencoder/experiments2/configs/exp001_base.json",
    )
    p.add_argument(
        "--checkpoint",
        required=True,
        help="Path to checkpoint (.pt)",
    )
    p.add_argument("--split", default="test", choices=["train", "val", "test", "ood"])
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: checkpoint parent / ../plots/visualize_<split>).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = resolve_device()

    manifest, split, norm_stats = prepare_exp001_artifacts(cfg)
    ds = ExternalHHComplexSTFTDataset(cfg, manifest, split[args.split]["indices"], norm_stats)
    loader = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_exp001,
    )

    num_stations = len(load_json(cfg["data"]["station_list_file"]))
    model = CVAEComplexSTFT(
        numeric_cond_dim=len(cfg["conditions"]["numeric_feature_order"]),
        num_stations=num_stations,
        latent_dim=int(cfg["model"]["latent_dim"]),
        station_embedding_dim=int(cfg["model"]["station_embedding_dim"]),
        condition_hidden_dim=int(cfg["model"]["condition_hidden_dim"]),
        encoder_channels=tuple(cfg["model"]["encoder_channels"]),
        decoder_channels=tuple(cfg["model"]["decoder_channels"]),
        input_shape=(2, int(cfg["stft"]["target_freq_bins"]), int(cfg["stft"]["target_time_frames"])),
    ).to(device)
    payload = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    ckpt_path = Path(args.checkpoint)
    if args.out_dir is None:
        out_dir = ckpt_path.parent.parent / "plots" / f"visualize_{args.split}"
    else:
        out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_bank = build_seed_bank(int(cfg["evaluation"]["seed_bank_base"]), 1)
    gen = torch.Generator(device=device.type)
    gen.manual_seed(seed_bank[0])

    produced = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            cond = batch["cond"].to(device)
            station = batch["station_idx"].to(device)
            recon, _, _ = model(x, cond, station)
            cond_pred = model.sample_condition_only(cond, station, generator=gen)

            x_np = x.cpu().numpy()
            r_np = recon.cpu().numpy()
            c_np = cond_pred.cpu().numpy()

            for i in range(x_np.shape[0]):
                meta = batch["meta"][i]
                title = f"{args.split} | event={meta['event_id']} station={meta['station_code']} M={meta['magnitude']:.2f}"
                _plot_triplet(
                    out_path=out_dir / f"{produced:04d}_{meta['event_id']}_{meta['station_code']}.png",
                    target=x_np[i],
                    recon=r_np[i],
                    cond=c_np[i],
                    cfg=cfg,
                    stft_scale=float(norm_stats["stft_global_rms"]),
                    title_prefix=title,
                )
                produced += 1
                if produced >= int(args.num_samples):
                    return


if __name__ == "__main__":
    main()
