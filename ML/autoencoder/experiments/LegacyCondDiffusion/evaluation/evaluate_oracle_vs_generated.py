import argparse
import json
import os
import sys

import numpy as np
import obspy
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))

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
    p = argparse.ArgumentParser(description="Compare oracle reconstruction vs diffusion generation.")
    p.add_argument(
        "--config",
        default="ML/autoencoder/experiments/LegacyCondDiffusion/configs/diffusion_resmlp_default.json",
    )
    p.add_argument("--max_samples", type=int, default=20)
    p.add_argument(
        "--output_json",
        default="ML/autoencoder/experiments/LegacyCondDiffusion/results/oracle_vs_generated_metrics.json",
    )
    return p.parse_args()


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


def mean_dict(metric_list):
    keys = metric_list[0].keys()
    return {k: float(np.nanmean([m[k] for m in metric_list])) for k in keys}


def main():
    args = parse_args()
    cfg = load_config(args.config)
    eval_cfg = cfg["eval"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage1, station_list = load_stage1_model(
        stage1_ckpt=eval_cfg["stage1_checkpoint"],
        station_list_file=eval_cfg["station_list_file"],
        device=device,
    )
    diffusion_model, diffusion_cfg = load_diffusion_model(
        diff_ckpt=eval_cfg["diffusion_checkpoint"], device=device
    )

    dataset = SeismicSTFTDatasetWithMetadata(
        data_dir=eval_cfg["ood_data_dir"],
        event_file=eval_cfg["ood_catalog"],
        channels=["HH"],
        magnitude_col="xM",
        station_list=station_list,
    )

    oracle_metrics = []
    generated_metrics = []
    per_sample = []

    with torch.no_grad():
        used = 0
        for i in range(len(dataset)):
            if used >= args.max_samples:
                break
            spec, mag, loc, sta, meta = dataset[i]
            if "error" in meta:
                continue

            x = spec.unsqueeze(0).to(device)
            mag_b = mag.unsqueeze(0).to(device)
            loc_b = loc.unsqueeze(0).to(device)
            sta_b = sta.unsqueeze(0).to(device)

            # Oracle reconstruction: z = mu(x,c)
            mu, _, w_oracle = stage1.encode_distribution(x, mag_b, loc_b, sta_b)
            pred_oracle = stage1.decoder(mu, w_oracle)
            if pred_oracle.shape[2:] != spec.shape[1:]:
                pred_oracle = torch.nn.functional.interpolate(
                    pred_oracle, size=spec.shape[1:], mode="bilinear", align_corners=False
                )

            # Diffusion generation: z from noise conditioned on c
            z_gen, w_gen = sample_latent_from_condition(
                stage1_model=stage1,
                diffusion_model=diffusion_model,
                diffusion_cfg=diffusion_cfg,
                stats_file=cfg["data"]["stats_file"],
                magnitude=mag_b,
                location=loc_b,
                station_idx=sta_b,
                device=device,
            )
            pred_gen = stage1.decoder(z_gen, w_gen)
            if pred_gen.shape[2:] != spec.shape[1:]:
                pred_gen = torch.nn.functional.interpolate(
                    pred_gen, size=spec.shape[1:], mode="bilinear", align_corners=False
                )

            target_spec = spec[2].cpu().numpy()
            oracle_spec = pred_oracle[0, 2].cpu().numpy()
            gen_spec = pred_gen[0, 2].cpu().numpy()

            gt_wav = load_gt_waveform(meta["file_path"], fs=100.0, target_len=7300)
            oracle_wav = reconstruct_signal_griffin_lim(
                oracle_spec, mag_min=meta.get("mag_min", 0.0), mag_max=meta.get("mag_max", 1.0), fs=100.0, n_iter=64
            )
            gen_wav = reconstruct_signal_griffin_lim(
                gen_spec, mag_min=meta.get("mag_min", 0.0), mag_max=meta.get("mag_max", 1.0), fs=100.0, n_iter=64
            )

            mo = min(len(gt_wav), len(oracle_wav))
            mg = min(len(gt_wav), len(gen_wav))
            m_oracle = calculate_all_metrics(
                target_wav=gt_wav[:mo],
                pred_wav=oracle_wav[:mo],
                target_spec=target_spec,
                pred_spec=oracle_spec,
                fs=100.0,
            )
            m_gen = calculate_all_metrics(
                target_wav=gt_wav[:mg],
                pred_wav=gen_wav[:mg],
                target_spec=target_spec,
                pred_spec=gen_spec,
                fs=100.0,
            )
            oracle_metrics.append(m_oracle)
            generated_metrics.append(m_gen)
            per_sample.append(
                {
                    "file_name": meta.get("file_name", "unknown"),
                    "oracle": m_oracle,
                    "generated": m_gen,
                }
            )
            used += 1

    result = {
        "num_samples": used,
        "oracle_mean": mean_dict(oracle_metrics) if oracle_metrics else {},
        "generated_mean": mean_dict(generated_metrics) if generated_metrics else {},
        "per_sample": per_sample,
    }
    save_json(result, args.output_json)
    print(json.dumps(result["oracle_mean"], indent=2))
    print(json.dumps(result["generated_mean"], indent=2))
    print(f"[INFO] Saved: {args.output_json}")


if __name__ == "__main__":
    main()
