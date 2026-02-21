#!/usr/bin/env python3
"""
Compute ELBO-like terms (reconstruction proxy + KL) on a frozen split.

For AE runs, KL is reported as 0.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

# evaluation -> LatentShapeVAE -> experiments -> autoencoder -> ML -> repo
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))

from ML.autoencoder.experiments.LatentShapeVAE.core.loss_utils import ae_composite_loss, vae_composite_loss
from ML.autoencoder.experiments.LatentShapeVAE.core.model_vae_waveform import WaveformAE, WaveformDiagVAE
from ML.autoencoder.experiments.LatentShapeVAE.core.waveform_dataset import WaveformDataset, collate_fn_waveform


def _read_yaml(path: Path) -> Dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from exc
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _build_dataset(files: List[str], data_cfg: Dict, normalization_stats_file: str) -> WaveformDataset:
    preprocess = data_cfg.get("preprocess", {})
    band = preprocess.get("bandpass", {})
    return WaveformDataset(
        file_paths=files,
        segment_length=int(data_cfg.get("segment_length", 7001)),
        sample_rate_hz=float(data_cfg.get("sample_rate_hz", 100.0)),
        preprocess_demean=bool(preprocess.get("demean", True)),
        preprocess_detrend=bool(preprocess.get("detrend", True)),
        bandpass_enabled=bool(band.get("enabled", True)),
        bandpass_freqmin=float(band.get("freqmin", 0.5)),
        bandpass_freqmax=float(band.get("freqmax", 20.0)),
        bandpass_corners=int(band.get("corners", 4)),
        bandpass_zerophase=bool(band.get("zerophase", True)),
        normalization_stats_file=normalization_stats_file,
    )


def _build_model(run_cfg: Dict, device: torch.device):
    mode = str(run_cfg["ablation_mode"])
    latent_dim = int(run_cfg["latent_dim"])
    backbone = str(run_cfg["backbone"])
    input_length = int(run_cfg.get("segment_length", 7001))
    logvar_mode = str(run_cfg.get("logvar_mode", "legacy"))
    logvar_min = float(run_cfg.get("logvar_min", -30.0))
    logvar_max = float(run_cfg.get("logvar_max", 20.0))
    if mode == "ae":
        return WaveformAE(3, input_length, latent_dim, backbone).to(device)
    return WaveformDiagVAE(
        3,
        input_length,
        latent_dim,
        backbone,
        logvar_mode=logvar_mode,
        logvar_min=logvar_min,
        logvar_max=logvar_max,
    ).to(device)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze ELBO terms for LatentShapeVAE checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--data_config",
        default="ML/autoencoder/experiments/LatentShapeVAE/configs/data_protocol_v1.yaml",
    )
    parser.add_argument(
        "--loss_config",
        default="ML/autoencoder/experiments/LatentShapeVAE/configs/loss_config_v1.yaml",
    )
    parser.add_argument(
        "--frozen_split_manifest",
        default="ML/autoencoder/experiments/LatentShapeVAE/protocol/frozen_event_splits_v1.json",
    )
    parser.add_argument("--normalization_stats_file", default="")
    parser.add_argument("--split", choices=["train", "val", "test", "ood_event"], default="test")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    data_cfg = _read_yaml(Path(args.data_config))
    loss_cfg = _read_yaml(Path(args.loss_config))
    manifest = _read_json(Path(args.frozen_split_manifest))
    split_key = f"{args.split}_files"
    files = _read_lines(Path(manifest["splits"][split_key]["file"]))
    norm_file = args.normalization_stats_file or data_cfg.get("normalization", {}).get("stats_file", "")
    if not norm_file:
        raise ValueError("normalization_stats_file not set in args or data config.")

    ds = _build_dataset(files, data_cfg=data_cfg, normalization_stats_file=norm_file)
    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_fn_waveform,
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    run_cfg = dict(ckpt.get("config", {}))
    run_cfg["segment_length"] = int(data_cfg.get("segment_length", 7001))
    mode = str(run_cfg.get("ablation_mode", "vae"))
    run_name = str(run_cfg.get("run_name", ckpt_path.stem.replace("_best", "")))
    beta = float(run_cfg.get("beta_target", 0.1))
    lambda_mr = float(run_cfg.get("lambda_mr", loss_cfg.get("mrstft", {}).get("weight", 0.5)))
    mr_cfg = loss_cfg.get("mrstft", {})
    mr_n_ffts = [int(x) for x in run_cfg.get("mr_n_ffts", mr_cfg.get("n_ffts", [64, 256, 1024]))]
    mr_hops = [int(x) for x in run_cfg.get("mr_hop_lengths", mr_cfg.get("hop_lengths", [16, 64, 256]))]
    mr_wins = [int(x) for x in run_cfg.get("mr_win_lengths", mr_cfg.get("win_lengths", [64, 256, 1024]))]
    mr_eps = float(run_cfg.get("mr_eps", 1e-7))
    free_bits = float(run_cfg.get("free_bits", 0.0))

    model = _build_model(run_cfg, device=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    sums = {
        "loss_total": 0.0,
        "recon_total": 0.0,
        "time_mse": 0.0,
        "mrstft": 0.0,
        "kl_raw": 0.0,
        "beta_kl": 0.0,
    }
    n_steps = 0
    n_skipped_nonfinite = 0
    with torch.no_grad():
        for batch in loader:
            x, _meta = batch
            if x is None:
                continue
            x = x.to(device, non_blocking=use_cuda)
            if mode == "ae":
                x_hat, _z = model(x)
                _loss, terms = ae_composite_loss(
                    x_hat=x_hat,
                    x=x,
                    lambda_mr=lambda_mr,
                    mr_n_ffts=mr_n_ffts,
                    mr_hop_lengths=mr_hops,
                    mr_win_lengths=mr_wins,
                    mr_eps=mr_eps,
                )
            else:
                x_hat, mu, logvar = model(x)
                _loss, terms, _ = vae_composite_loss(
                    x_hat=x_hat,
                    x=x,
                    mu=mu,
                    logvar=logvar,
                    beta=beta,
                    lambda_mr=lambda_mr,
                    mr_n_ffts=mr_n_ffts,
                    mr_hop_lengths=mr_hops,
                    mr_win_lengths=mr_wins,
                    mr_eps=mr_eps,
                    free_bits=free_bits,
                )
            if not all(torch.isfinite(torch.tensor(float(terms[k]))).item() for k in sums.keys()):
                n_skipped_nonfinite += 1
                continue
            for k in sums:
                sums[k] += float(terms[k])
            n_steps += 1

    n_steps = max(n_steps, 1)
    means = {k: v / n_steps for k, v in sums.items()}
    out = {
        "run_name": run_name,
        "checkpoint": ckpt_path.as_posix(),
        "ablation_mode": mode,
        "split": args.split,
        "n_eval_steps": n_steps,
        "n_skipped_nonfinite": int(n_skipped_nonfinite),
        "E_log_p_x_given_z_proxy": -float(means["recon_total"]),
        "E_KL_q_to_p": float(means["kl_raw"]),
        "loss_total": float(means["loss_total"]),
        "recon_total": float(means["recon_total"]),
        "time_mse": float(means["time_mse"]),
        "mrstft": float(means["mrstft"]),
        "kl_raw": float(means["kl_raw"]),
        "beta_kl": float(means["beta_kl"]),
    }

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else Path("ML/autoencoder/experiments/LatentShapeVAE/results") / f"elbo_terms_{args.split}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{run_name}_elbo_terms.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    csv_path = out_dir / "elbo_terms_summary.csv"
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(out)

    print("[INFO] elbo_terms_json:", json_path)
    print("[INFO] elbo_summary_csv:", csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
