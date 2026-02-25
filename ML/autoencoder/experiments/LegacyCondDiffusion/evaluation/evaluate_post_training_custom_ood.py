import argparse
import json
import os
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))


import numpy as np
import obspy
import torch

from ML.autoencoder.experiments.LegacyCondDiffusion.core.config_utils import load_config, save_json
from ML.autoencoder.experiments.LegacyCondDiffusion.core.dataset_stft import SeismicSTFTDatasetWithMetadata
from ML.autoencoder.experiments.LegacyCondDiffusion.evaluation.inference_utils import (
    load_diffusion_model,
    load_stage1_model,
    sample_latent_from_condition,
)
from ML.autoencoder.experiments.LegacyCondDiffusion.evaluation.metrics import (
    calculate_all_metrics,
    reconstruct_signal_griffin_lim,
)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate LegacyCondDiffusion on post-training custom OOD.")
    p.add_argument(
        "--config",
        default="ML/autoencoder/experiments/LegacyCondDiffusion/configs/diffusion_resmlp_default.json",
    )
    p.add_argument("--max_samples", type=int, default=-1, help="Optional cap for quick runs.")
    p.add_argument("--ood_data_dir", default=None)
    p.add_argument("--ood_catalog", default=None)
    p.add_argument("--station_subset_file", default=None)
    p.add_argument("--output_metrics_json", default=None)
    return p.parse_args()


def load_station_subset(path):
    if path is None:
        return None
    with open(path, "r") as f:
        return set(json.load(f))


def load_gt_waveform(file_path, fs=100.0, target_len=7300):
    st = obspy.read(file_path)
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
    eval_cfg = cfg["eval"]
    if args.ood_data_dir is not None:
        eval_cfg["ood_data_dir"] = args.ood_data_dir
    if args.ood_catalog is not None:
        eval_cfg["ood_catalog"] = args.ood_catalog
    if args.station_subset_file is not None:
        eval_cfg["station_subset_file"] = args.station_subset_file
    if args.output_metrics_json is not None:
        eval_cfg["output_metrics_json"] = args.output_metrics_json

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    stage1, station_list = load_stage1_model(
        stage1_ckpt=eval_cfg["stage1_checkpoint"],
        station_list_file=eval_cfg["station_list_file"],
        device=device,
    )
    diffusion_model, diff_cfg = load_diffusion_model(
        diff_ckpt=eval_cfg["diffusion_checkpoint"],
        device=device,
    )
    subset = load_station_subset(eval_cfg.get("station_subset_file"))

    dataset = SeismicSTFTDatasetWithMetadata(
        data_dir=eval_cfg["ood_data_dir"],
        event_file=eval_cfg["ood_catalog"],
        channels=["HH"],
        magnitude_col="xM",
        station_list=station_list,
    )

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
    n_used = 0

    with torch.no_grad():
        for i in range(len(dataset)):
            if args.max_samples > 0 and n_used >= args.max_samples:
                break

            spec, mag, loc, sta, meta = dataset[i]
            if "error" in meta:
                continue

            station_name = meta.get("station_name", "UNKNOWN")
            if subset is not None and station_name not in subset:
                continue

            mag_b = mag.unsqueeze(0).to(device)
            loc_b = loc.unsqueeze(0).to(device)
            sta_b = sta.unsqueeze(0).to(device)

            z, w = sample_latent_from_condition(
                stage1_model=stage1,
                diffusion_model=diffusion_model,
                diffusion_cfg=diff_cfg,
                stats_file=cfg["data"]["stats_file"],
                magnitude=mag_b,
                location=loc_b,
                station_idx=sta_b,
                device=device,
            )
            pred = stage1.decoder(z, w)
            if pred.shape[2:] != spec.shape[1:]:
                pred = torch.nn.functional.interpolate(
                    pred, size=spec.shape[1:], mode="bilinear", align_corners=False
                )

            target_spec = spec[2].cpu().numpy()
            pred_spec = pred[0, 2].cpu().numpy()

            gt_wav = load_gt_waveform(meta["file_path"], fs=100.0, target_len=7300)
            pred_wav = reconstruct_signal_griffin_lim(
                pred_spec,
                mag_min=meta.get("mag_min", 0.0),
                mag_max=meta.get("mag_max", 1.0),
                fs=100.0,
                n_iter=64,
            )
            m = min(len(gt_wav), len(pred_wav))
            metrics = calculate_all_metrics(
                target_wav=gt_wav[:m],
                pred_wav=pred_wav[:m],
                target_spec=target_spec,
                pred_spec=pred_spec,
                fs=100.0,
            )
            for k in metric_keys:
                agg[k].append(metrics[k])
            n_used += 1

            if (n_used % 10) == 0:
                print(f"[INFO] Processed {n_used} samples")

    summary = {k: (float(np.nanmean(v)) if v else None) for k, v in agg.items()}
    summary["num_samples"] = n_used
    summary["run_name"] = cfg["run"]["name"]
    summary["denoiser"] = cfg["model"]["denoiser"]
    summary["cond_mode"] = cfg["model"]["cond_mode"]

    output_path = eval_cfg["output_metrics_json"]
    save_json(summary, output_path)
    print(f"[INFO] Metrics saved: {output_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
