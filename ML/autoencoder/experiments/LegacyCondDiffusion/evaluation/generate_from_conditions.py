import argparse
from pathlib import Path
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))


import numpy as np
import torch

from ML.autoencoder.experiments.LegacyCondDiffusion.core.config_utils import load_config
from ML.autoencoder.experiments.LegacyCondDiffusion.core.dataset_stft import SeismicSTFTDatasetWithMetadata
from ML.autoencoder.experiments.LegacyCondDiffusion.evaluation.inference_utils import (
    load_diffusion_model,
    load_stage1_model,
    sample_latent_from_condition,
)
from ML.autoencoder.experiments.LegacyCondDiffusion.evaluation.metrics import reconstruct_signal_griffin_lim


def parse_args():
    p = argparse.ArgumentParser(description="Generate samples from conditions using trained diffusion model.")
    p.add_argument(
        "--config",
        default="ML/autoencoder/experiments/LegacyCondDiffusion/configs/diffusion_resmlp_default.json",
    )
    p.add_argument("--num_samples", type=int, default=10)
    p.add_argument(
        "--output_dir",
        default="ML/autoencoder/experiments/LegacyCondDiffusion/results/generated_samples",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    eval_cfg = cfg["eval"]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage1, station_list = load_stage1_model(
        stage1_ckpt=eval_cfg["stage1_checkpoint"],
        station_list_file=eval_cfg["station_list_file"],
        device=device,
    )
    diffusion_model, diff_cfg = load_diffusion_model(
        diff_ckpt=eval_cfg["diffusion_checkpoint"],
        device=device,
    )

    dataset = SeismicSTFTDatasetWithMetadata(
        data_dir=eval_cfg["ood_data_dir"],
        event_file=eval_cfg["ood_catalog"],
        channels=["HH"],
        magnitude_col="xM",
        station_list=station_list,
    )

    used = 0
    with torch.no_grad():
        for i in range(len(dataset)):
            if used >= args.num_samples:
                break
            spec, mag, loc, sta, meta = dataset[i]
            if "error" in meta:
                continue

            z, w = sample_latent_from_condition(
                stage1_model=stage1,
                diffusion_model=diffusion_model,
                diffusion_cfg=diff_cfg,
                stats_file=cfg["data"]["stats_file"],
                magnitude=mag.unsqueeze(0).to(device),
                location=loc.unsqueeze(0).to(device),
                station_idx=sta.unsqueeze(0).to(device),
                device=device,
            )
            pred = stage1.decoder(z, w)
            if pred.shape[2:] != spec.shape[1:]:
                pred = torch.nn.functional.interpolate(pred, size=spec.shape[1:], mode="bilinear", align_corners=False)

            pred_spec = pred[0, 2].cpu().numpy()
            pred_wav = reconstruct_signal_griffin_lim(
                pred_spec,
                mag_min=meta.get("mag_min", 0.0),
                mag_max=meta.get("mag_max", 1.0),
                fs=100.0,
                n_iter=64,
            )

            stem = f"{used:03d}_{meta.get('event_id','UNK')}_{meta.get('station_name','UNK')}"
            np.save(out_dir / f"{stem}_spec.npy", pred_spec)
            np.save(out_dir / f"{stem}_wav.npy", pred_wav)
            used += 1

    print(f"[INFO] Saved {used} generated samples to {out_dir}")


if __name__ == "__main__":
    main()

