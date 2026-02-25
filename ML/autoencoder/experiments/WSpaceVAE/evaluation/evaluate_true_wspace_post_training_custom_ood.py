import argparse
import json
import os
import sys

import numpy as np
import obspy
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))

from ML.autoencoder.experiments.General.core.stft_dataset import SeismicSTFTDatasetWithMetadata  # noqa: E402
from ML.autoencoder.experiments.WSpaceVAE.core.config_utils import load_config, save_json  # noqa: E402
from ML.autoencoder.experiments.WSpaceVAE.core.model_true_wspace_vae import TrueWSpaceCVAE  # noqa: E402
from ML.autoencoder.experiments.WSpaceVAE.evaluation.metrics import (  # noqa: E402
    calculate_all_metrics,
    reconstruct_signal_griffin_lim,
)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate true W-space VAE on post-training custom OOD.")
    p.add_argument(
        "--config",
        default="ML/autoencoder/experiments/WSpaceVAE/configs/train_true_wspace_vae_external.json",
    )
    p.add_argument(
        "--checkpoint",
        default="ML/autoencoder/experiments/WSpaceVAE/checkpoints/true_wspace_vae_best.pt",
    )
    p.add_argument("--max_samples", type=int, default=-1)
    p.add_argument(
        "--mode",
        choices=["reconstruct", "sample"],
        default="reconstruct",
        help="reconstruct: model(x,c) path, sample: decoder generation from u~N(0,I)",
    )
    p.add_argument(
        "--output_json",
        default="ML/autoencoder/experiments/WSpaceVAE/results/post_training_custom_ood_metrics_true_wspace_vae.json",
    )
    return p.parse_args()


def load_gt_waveform(path, fs=100.0, target_len=7300):
    st = obspy.read(path)
    st.resample(fs)
    tr = st.select(component="Z")[0] if st.select(component="Z") else st[0]
    wav = tr.data.astype(np.float32)
    if len(wav) > target_len:
        wav = wav[:target_len]
    elif len(wav) < target_len:
        wav = np.pad(wav, (0, target_len - len(wav)))
    return wav


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(cfg["data"]["station_list_file"], "r") as f:
        station_list = json.load(f)

    dataset = SeismicSTFTDatasetWithMetadata(
        data_dir=cfg["eval"]["ood_data_dir"],
        event_file=cfg["eval"]["ood_catalog"],
        channels=["HH"],
        magnitude_col="xM",
        station_list=station_list,
    )

    model = TrueWSpaceCVAE(
        in_channels=cfg["model"]["in_channels"],
        u_dim=cfg["model"]["u_dim"],
        w_dim=cfg["model"]["w_dim"],
        cond_dim=cfg["model"]["cond_dim"],
        num_stations=len(station_list),
        station_emb_dim=cfg["model"]["station_emb_dim"],
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    metric_keys = [
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
    agg = {k: [] for k in metric_keys}
    used = 0

    with torch.no_grad():
        for i in range(len(dataset)):
            if args.max_samples > 0 and used >= args.max_samples:
                break
            spec, mag, loc, sta, meta = dataset[i]
            if "error" in meta:
                continue

            mag_b = mag.unsqueeze(0).to(device)
            loc_b = loc.unsqueeze(0).to(device)
            sta_b = sta.unsqueeze(0).to(device)
            if args.mode == "reconstruct":
                pred, _, _, _, _ = model(spec.unsqueeze(0).to(device), mag_b, loc_b, sta_b)
                pred = pred[0, 2].cpu().numpy()
            else:
                pred, _, _ = model.sample(1, mag_b, loc_b, sta_b, device=device)
                pred = pred[0, 2].cpu().numpy()
            target = spec[2].cpu().numpy()

            gt_wav = load_gt_waveform(meta["file_path"])
            pred_wav = reconstruct_signal_griffin_lim(
                pred, mag_min=meta.get("mag_min", 0.0), mag_max=meta.get("mag_max", 1.0), fs=100.0, n_iter=64
            )
            m = min(len(gt_wav), len(pred_wav))
            mtr = calculate_all_metrics(gt_wav[:m], pred_wav[:m], target, pred)
            for k in metric_keys:
                agg[k].append(mtr[k])
            used += 1
            if used % 10 == 0:
                print(f"[INFO] Processed {used} samples")

    summary = {k: float(np.nanmean(v)) if v else None for k, v in agg.items()}
    summary["num_samples"] = used
    summary["model"] = "TrueWSpaceVAE"
    summary["mode"] = args.mode
    save_json(summary, args.output_json)
    print(json.dumps(summary, indent=2))
    print(f"[INFO] Saved: {args.output_json}")


if __name__ == "__main__":
    main()
