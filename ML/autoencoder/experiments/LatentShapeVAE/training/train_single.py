#!/usr/bin/env python3
"""
Train one LatentShapeVAE run (AE / beta0 / VAE).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# training -> LatentShapeVAE -> experiments -> autoencoder -> ML -> repo
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))

from ML.autoencoder.experiments.LatentShapeVAE.core.loss_utils import (
    active_units_from_mu,
    ae_composite_loss,
    vae_composite_loss,
)
from ML.autoencoder.experiments.LatentShapeVAE.core.model_vae_waveform import WaveformAE, WaveformDiagVAE
from ML.autoencoder.experiments.LatentShapeVAE.core.waveform_dataset import (
    WaveformDataset,
    collate_fn_waveform,
)


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


def _setup_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger(str(path))
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fmt = logging.Formatter("%(asctime)s | %(message)s")

    fh = logging.FileHandler(path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_csv_row(path: Path, row: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


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


def _build_model(
    ablation_mode: str,
    backbone: str,
    latent_dim: int,
    input_length: int,
    device: torch.device,
    logvar_mode: str = "legacy",
    logvar_min: float = -30.0,
    logvar_max: float = 20.0,
):
    if ablation_mode == "ae":
        model = WaveformAE(
            in_channels=3,
            input_length=input_length,
            latent_dim=latent_dim,
            backbone=backbone,
        )
    else:
        model = WaveformDiagVAE(
            in_channels=3,
            input_length=input_length,
            latent_dim=latent_dim,
            backbone=backbone,
            logvar_mode=logvar_mode,
            logvar_min=logvar_min,
            logvar_max=logvar_max,
        )
    return model.to(device)


def _effective_beta(
    step: int,
    ablation_mode: str,
    beta_target: float,
    anneal_enabled: bool,
    anneal_start: float,
    anneal_end: float,
    anneal_warmup_steps: int,
) -> float:
    if ablation_mode == "ae":
        return 0.0
    if ablation_mode == "beta0":
        return 0.0
    if not anneal_enabled or anneal_warmup_steps <= 0:
        return float(beta_target)
    t = min(1.0, max(0.0, step / float(anneal_warmup_steps)))
    return float(anneal_start + t * (anneal_end - anneal_start))


def _evaluate(
    model: torch.nn.Module,
    ablation_mode: str,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    beta_eff: float,
    lambda_mr: float,
    mr_n_ffts: List[int],
    mr_hop_lengths: List[int],
    mr_win_lengths: List[int],
    mr_eps: float,
    free_bits: float,
    latent_dim: int,
) -> Dict[str, float]:
    model.eval()
    totals = {
        "loss_total": 0.0,
        "recon_total": 0.0,
        "time_mse": 0.0,
        "mrstft": 0.0,
        "kl_raw": 0.0,
        "beta_kl": 0.0,
    }
    n_steps = 0
    n_nonfinite = 0
    n_samples = 0
    kl_dim_sum: Optional[torch.Tensor] = None
    mu_rows: List[torch.Tensor] = []

    def amp_ctx():
        if device.type == "cuda":
            return torch.autocast(device_type="cuda", enabled=amp)
        return nullcontext()

    with torch.no_grad():
        for batch in loader:
            x, _meta = batch
            if x is None:
                continue
            x = x.to(device, non_blocking=(device.type == "cuda"))
            bs = x.size(0)

            with amp_ctx():
                if ablation_mode == "ae":
                    x_hat, z = model(x)
                    _loss, terms = ae_composite_loss(
                        x_hat=x_hat,
                        x=x,
                        lambda_mr=lambda_mr,
                        mr_n_ffts=mr_n_ffts,
                        mr_hop_lengths=mr_hop_lengths,
                        mr_win_lengths=mr_win_lengths,
                        mr_eps=mr_eps,
                    )
                    mu_rows.append(z.detach().cpu())
                elif ablation_mode == "beta0":
                    # beta=0 path: deterministic mean-path to prevent variance drift blow-up.
                    mu, logvar = model.encode(x)
                    x_hat = model.decode(mu, target_length=int(x.shape[-1]))
                    _loss, terms, _ = vae_composite_loss(
                        x_hat=x_hat,
                        x=x,
                        mu=mu,
                        logvar=logvar,
                        beta=0.0,
                        lambda_mr=lambda_mr,
                        mr_n_ffts=mr_n_ffts,
                        mr_hop_lengths=mr_hop_lengths,
                        mr_win_lengths=mr_win_lengths,
                        mr_eps=mr_eps,
                        free_bits=free_bits,
                    )
                    mu_rows.append(mu.detach().cpu())
                    lv = torch.clamp(logvar.detach(), min=-30.0, max=20.0)
                    k = 0.5 * (mu.detach().pow(2) + lv.exp() - 1.0 - lv)  # [B, D]
                    k_sum = k.sum(dim=0).detach().cpu()
                    kl_dim_sum = k_sum if kl_dim_sum is None else kl_dim_sum + k_sum
                    n_samples += bs
                else:
                    x_hat, mu, logvar = model(x)
                    _loss, terms, _ = vae_composite_loss(
                        x_hat=x_hat,
                        x=x,
                        mu=mu,
                        logvar=logvar,
                        beta=beta_eff,
                        lambda_mr=lambda_mr,
                        mr_n_ffts=mr_n_ffts,
                        mr_hop_lengths=mr_hop_lengths,
                        mr_win_lengths=mr_win_lengths,
                        mr_eps=mr_eps,
                        free_bits=free_bits,
                    )
                    mu_rows.append(mu.detach().cpu())

                    # Raw per-dim KL for collapse diagnostics (without free-bits clamp).
                    lv = torch.clamp(logvar.detach(), min=-30.0, max=20.0)
                    k = 0.5 * (mu.detach().pow(2) + lv.exp() - 1.0 - lv)  # [B, D]
                    k_sum = k.sum(dim=0).detach().cpu()
                    kl_dim_sum = k_sum if kl_dim_sum is None else kl_dim_sum + k_sum
                    n_samples += bs

            if not all(math.isfinite(float(v)) for v in terms.values()):
                n_nonfinite += 1
                continue

            for k, v in terms.items():
                if k in totals:
                    totals[k] += float(v)
            n_steps += 1

    n_steps = max(n_steps, 1)
    out = {k: v / n_steps for k, v in totals.items()}

    mu_all = torch.cat(mu_rows, dim=0) if mu_rows else torch.zeros((0, latent_dim))
    if mu_all.numel() > 0:
        au, _ = active_units_from_mu(mu_all, threshold=1e-2)
        out["active_units"] = float(au)
    else:
        out["active_units"] = 0.0

    if ablation_mode == "ae":
        out["kl_per_dim_median"] = 0.0
    elif kl_dim_sum is not None and n_samples > 0:
        kl_per_dim = kl_dim_sum / float(n_samples)
        out["kl_per_dim_median"] = float(torch.median(kl_per_dim).item())
    else:
        out["kl_per_dim_median"] = 0.0

    out["nonfinite_eval_batches"] = float(n_nonfinite)
    return out


def _save_checkpoint(path: Path, payload: Dict, logger: logging.Logger) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    except OSError as exc:
        logger.error(f"[CKPT-ERROR] failed to write checkpoint to {path}: {exc}")
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Train one LatentShapeVAE run.")
    parser.add_argument("--run_name", default="")
    parser.add_argument("--ablation_mode", choices=["vae", "beta0", "ae"], default="vae")
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--backbone", choices=["small", "base", "large"], default="base")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--amp", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)

    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--free_bits", type=float, default=0.0)
    parser.add_argument("--logvar_mode", choices=["legacy", "bounded_sigmoid"], default="legacy")
    parser.add_argument("--logvar_min", type=float, default=-30.0)
    parser.add_argument("--logvar_max", type=float, default=20.0)
    parser.add_argument("--lambda_mr", type=float, default=-1.0, help="If <0, read from loss config.")
    parser.add_argument("--mr_n_ffts", default="", help="Comma list override, e.g. 64,256,1024")
    parser.add_argument("--mr_hop_lengths", default="")
    parser.add_argument("--mr_win_lengths", default="")
    parser.add_argument("--mr_eps", type=float, default=1e-7)

    parser.add_argument("--anneal_enabled", type=int, default=-1)
    parser.add_argument("--anneal_beta_start", type=float, default=0.0)
    parser.add_argument("--anneal_beta_end", type=float, default=-1.0)
    parser.add_argument("--anneal_warmup_steps", type=int, default=-1)

    parser.add_argument("--max_steps", type=int, default=12000)
    parser.add_argument("--val_check_every_steps", type=int, default=1000)
    parser.add_argument("--train_log_every_steps", type=int, default=100)
    parser.add_argument("--patience_evals", type=int, default=6)
    parser.add_argument("--min_steps_before_early_stop", type=int, default=4000)
    parser.add_argument("--min_delta", type=float, default=0.0)

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
    parser.add_argument("--root_dir", default="ML/autoencoder/experiments/LatentShapeVAE")
    args = parser.parse_args()

    _set_seed(args.seed)

    data_cfg = _read_yaml(Path(args.data_config))
    loss_cfg = _read_yaml(Path(args.loss_config))
    split_manifest = _read_json(Path(args.frozen_split_manifest))

    norm_file = args.normalization_stats_file or data_cfg.get("normalization", {}).get("stats_file", "")
    if not norm_file:
        raise ValueError("normalization_stats_file not set in args or data config.")

    train_files = _read_lines(Path(split_manifest["splits"]["train_files"]["file"]))
    val_files = _read_lines(Path(split_manifest["splits"]["val_files"]["file"]))
    if not train_files or not val_files:
        raise RuntimeError("Train/val split is empty. Freeze splits first.")

    mr_cfg = loss_cfg.get("mrstft", {})
    kl_cfg = loss_cfg.get("kl", {})
    lambda_mr = float(args.lambda_mr if args.lambda_mr >= 0 else mr_cfg.get("weight", 0.5))
    mr_n_ffts = [int(x) for x in args.mr_n_ffts.split(",") if x.strip()] if args.mr_n_ffts else [int(x) for x in mr_cfg.get("n_ffts", [64, 256, 1024])]
    mr_hops = [int(x) for x in args.mr_hop_lengths.split(",") if x.strip()] if args.mr_hop_lengths else [int(x) for x in mr_cfg.get("hop_lengths", [16, 64, 256])]
    mr_wins = [int(x) for x in args.mr_win_lengths.split(",") if x.strip()] if args.mr_win_lengths else [int(x) for x in mr_cfg.get("win_lengths", [64, 256, 1024])]
    if not (len(mr_n_ffts) == len(mr_hops) == len(mr_wins)):
        raise ValueError("MR-STFT lists must have same length.")

    beta_target = 0.0 if args.ablation_mode == "beta0" else float(args.beta)
    if args.ablation_mode == "vae" and args.beta == 0.1 and "beta_default" in kl_cfg:
        beta_target = float(kl_cfg["beta_default"])
    free_bits = float(args.free_bits if args.free_bits > 0 else kl_cfg.get("free_bits", 0.0))

    anneal_cfg = kl_cfg.get("anneal", {})
    anneal_enabled = bool(args.anneal_enabled) if args.anneal_enabled >= 0 else bool(anneal_cfg.get("enabled", False))
    anneal_warmup_steps = int(args.anneal_warmup_steps if args.anneal_warmup_steps >= 0 else anneal_cfg.get("warmup_steps", 0))
    anneal_beta_start = float(args.anneal_beta_start if args.anneal_beta_start >= 0 else anneal_cfg.get("beta_start", 0.0))
    anneal_beta_end = float(args.anneal_beta_end if args.anneal_beta_end >= 0 else anneal_cfg.get("beta_end", beta_target))
    if args.ablation_mode in ["ae", "beta0"]:
        anneal_enabled = False
        anneal_warmup_steps = 0
        anneal_beta_start = 0.0
        anneal_beta_end = 0.0

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"lsv_{args.ablation_mode}_{args.backbone}_ld{args.latent_dim}_b{str(beta_target).replace('.', 'p')}_s{args.seed}_{ts}"

    root = Path(args.root_dir)
    ckpt_dir = root / "checkpoints"
    logs_dir = root / "logs"
    run_dir = root / "results" / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = _setup_logger(logs_dir / f"{run_name}.log")
    history_csv = run_dir / "history.csv"
    run_cfg_json = run_dir / "run_config.json"
    summary_json = run_dir / "summary.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    persistent_workers = args.num_workers > 0

    train_ds = _build_dataset(train_files, data_cfg=data_cfg, normalization_stats_file=norm_file)
    val_ds = _build_dataset(val_files, data_cfg=data_cfg, normalization_stats_file=norm_file)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn_waveform,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn_waveform,
        drop_last=False,
    )

    input_length = int(data_cfg.get("segment_length", 7001))
    model = _build_model(
        ablation_mode=args.ablation_mode,
        backbone=args.backbone,
        latent_dim=args.latent_dim,
        input_length=input_length,
        device=device,
        logvar_mode=args.logvar_mode,
        logvar_min=args.logvar_min,
        logvar_max=args.logvar_max,
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(device=device.type, enabled=(bool(args.amp) and device.type == "cuda"))

    run_cfg = {
        "run_name": run_name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "ablation_mode": args.ablation_mode,
        "latent_dim": int(args.latent_dim),
        "backbone": args.backbone,
        "seed": int(args.seed),
        "device": str(device),
        "lr": float(args.lr),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "amp": int(args.amp),
        "weight_decay": float(args.weight_decay),
        "grad_clip_norm": float(args.grad_clip_norm),
        "beta_target": float(beta_target),
        "free_bits": float(free_bits),
        "logvar_mode": str(args.logvar_mode),
        "logvar_min": float(args.logvar_min),
        "logvar_max": float(args.logvar_max),
        "lambda_mr": float(lambda_mr),
        "mr_n_ffts": mr_n_ffts,
        "mr_hop_lengths": mr_hops,
        "mr_win_lengths": mr_wins,
        "mr_eps": float(args.mr_eps),
        "anneal_enabled": bool(anneal_enabled),
        "anneal_beta_start": float(anneal_beta_start),
        "anneal_beta_end": float(anneal_beta_end),
        "anneal_warmup_steps": int(anneal_warmup_steps),
        "max_steps": int(args.max_steps),
        "val_check_every_steps": int(args.val_check_every_steps),
        "train_log_every_steps": int(args.train_log_every_steps),
        "patience_evals": int(args.patience_evals),
        "min_steps_before_early_stop": int(args.min_steps_before_early_stop),
        "min_delta": float(args.min_delta),
        "frozen_split_manifest": Path(args.frozen_split_manifest).as_posix(),
        "normalization_stats_file": Path(norm_file).as_posix(),
        "data_config": Path(args.data_config).as_posix(),
        "loss_config": Path(args.loss_config).as_posix(),
        "split_counts": {"train_files": len(train_files), "val_files": len(val_files)},
    }
    with run_cfg_json.open("w", encoding="utf-8") as f:
        json.dump(run_cfg, f, indent=2)

    logger.info(
        f"[START] run={run_name} mode={args.ablation_mode} backbone={args.backbone} "
        f"latent_dim={args.latent_dim} beta_target={beta_target} device={device}"
    )
    logger.info(
        f"[DATA] train_files={len(train_files)} val_files={len(val_files)} "
        f"batch_size={args.batch_size} workers={args.num_workers}"
    )

    step = 0
    nonfinite_train_steps = 0
    best_val = math.inf
    best_step = 0
    no_improve_evals = 0
    stop_reason = "max_steps_reached"
    train_meter = {k: 0.0 for k in ["loss_total", "recon_total", "time_mse", "mrstft", "kl_raw", "beta_kl"]}
    train_meter_count = 0
    t_start = time.time()

    train_iter = iter(train_loader)
    while step < args.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        x, _meta = batch
        if x is None:
            continue
        step += 1
        model.train()
        x = x.to(device, non_blocking=pin_memory)
        beta_eff = _effective_beta(
            step=step,
            ablation_mode=args.ablation_mode,
            beta_target=beta_target,
            anneal_enabled=anneal_enabled,
            anneal_start=anneal_beta_start,
            anneal_end=anneal_beta_end,
            anneal_warmup_steps=anneal_warmup_steps,
        )

        optimizer.zero_grad(set_to_none=True)

        autocast_ctx = torch.autocast(device_type="cuda", enabled=(bool(args.amp) and device.type == "cuda")) if device.type == "cuda" else nullcontext()
        with autocast_ctx:
            if args.ablation_mode == "ae":
                x_hat, _z = model(x)
                loss, terms = ae_composite_loss(
                    x_hat=x_hat,
                    x=x,
                    lambda_mr=lambda_mr,
                    mr_n_ffts=mr_n_ffts,
                    mr_hop_lengths=mr_hops,
                    mr_win_lengths=mr_wins,
                    mr_eps=args.mr_eps,
                )
            elif args.ablation_mode == "beta0":
                # beta=0 path: deterministic mean-path to keep training stable.
                mu, logvar = model.encode(x)
                x_hat = model.decode(mu, target_length=int(x.shape[-1]))
                loss, terms, _ = vae_composite_loss(
                    x_hat=x_hat,
                    x=x,
                    mu=mu,
                    logvar=logvar,
                    beta=0.0,
                    lambda_mr=lambda_mr,
                    mr_n_ffts=mr_n_ffts,
                    mr_hop_lengths=mr_hops,
                    mr_win_lengths=mr_wins,
                    mr_eps=args.mr_eps,
                    free_bits=free_bits,
                )
            else:
                x_hat, mu, logvar = model(x)
                loss, terms, _ = vae_composite_loss(
                    x_hat=x_hat,
                    x=x,
                    mu=mu,
                    logvar=logvar,
                    beta=beta_eff,
                    lambda_mr=lambda_mr,
                    mr_n_ffts=mr_n_ffts,
                    mr_hop_lengths=mr_hops,
                    mr_win_lengths=mr_wins,
                    mr_eps=args.mr_eps,
                    free_bits=free_bits,
                )

        if not torch.isfinite(loss):
            nonfinite_train_steps += 1
            logger.warning(
                f"[NONFINITE-TRAIN] step={step} mode={args.ablation_mode} "
                f"loss={float(loss.detach().cpu().item())} skipped_updates={nonfinite_train_steps}"
            )
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()
        if args.grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        for k in train_meter:
            train_meter[k] += float(terms.get(k, 0.0))
        train_meter_count += 1

        if step % args.train_log_every_steps == 0:
            denom = max(1, train_meter_count)
            logger.info(
                f"[TRAIN] step={step}/{args.max_steps} "
                f"loss={train_meter['loss_total']/denom:.6f} "
                f"recon={train_meter['recon_total']/denom:.6f} "
                f"time={train_meter['time_mse']/denom:.6f} "
                f"mr={train_meter['mrstft']/denom:.6f} "
                f"kl={train_meter['kl_raw']/denom:.6f} "
                f"beta={beta_eff:.6f}"
            )
            train_meter = {k: 0.0 for k in train_meter}
            train_meter_count = 0

        if (step % args.val_check_every_steps == 0) or (step == args.max_steps):
            val_metrics = _evaluate(
                model=model,
                ablation_mode=args.ablation_mode,
                loader=val_loader,
                device=device,
                amp=bool(args.amp),
                beta_eff=beta_eff,
                lambda_mr=lambda_mr,
                mr_n_ffts=mr_n_ffts,
                mr_hop_lengths=mr_hops,
                mr_win_lengths=mr_wins,
                mr_eps=args.mr_eps,
                free_bits=free_bits,
                latent_dim=args.latent_dim,
            )
            current_val = float(val_metrics["loss_total"])
            improved = current_val < (best_val - args.min_delta)
            if improved:
                best_val = current_val
                best_step = step
                no_improve_evals = 0
            else:
                no_improve_evals += 1

            row = {
                "step": step,
                "beta_eff": beta_eff,
                "val_loss_total": val_metrics["loss_total"],
                "val_recon_total": val_metrics["recon_total"],
                "val_time_mse": val_metrics["time_mse"],
                "val_mrstft": val_metrics["mrstft"],
                "val_kl_raw": val_metrics["kl_raw"],
                "val_beta_kl": val_metrics["beta_kl"],
                "val_active_units": val_metrics["active_units"],
                "val_kl_per_dim_median": val_metrics["kl_per_dim_median"],
                "val_nonfinite_eval_batches": val_metrics["nonfinite_eval_batches"],
                "best_val_so_far": best_val,
                "is_best": int(improved),
                "elapsed_sec": time.time() - t_start,
            }
            _write_csv_row(history_csv, row)

            logger.info(
                f"[VAL] step={step} loss={val_metrics['loss_total']:.6f} "
                f"recon={val_metrics['recon_total']:.6f} time={val_metrics['time_mse']:.6f} "
                f"mr={val_metrics['mrstft']:.6f} kl={val_metrics['kl_raw']:.6f} "
                f"au={val_metrics['active_units']:.1f}/{args.latent_dim} "
                f"kl_med={val_metrics['kl_per_dim_median']:.6f} "
                f"val_nonfinite={int(val_metrics['nonfinite_eval_batches'])} "
                f"best={best_val:.6f} no_improve={no_improve_evals}"
            )

            ckpt_payload = {
                "step": int(step),
                "best_step": int(best_step),
                "best_val_loss": float(best_val),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "config": run_cfg,
                "val_metrics": val_metrics,
            }
            _save_checkpoint(ckpt_dir / f"{run_name}_last.pt", ckpt_payload, logger)
            if improved:
                _save_checkpoint(ckpt_dir / f"{run_name}_best.pt", ckpt_payload, logger)

            if (
                step >= args.min_steps_before_early_stop
                and no_improve_evals >= args.patience_evals
            ):
                stop_reason = "early_stop_patience"
                logger.info(
                    f"[EARLY-STOP] step={step} best_step={best_step} "
                    f"best_val={best_val:.6f} patience={args.patience_evals}"
                )
                break

    elapsed = time.time() - t_start
    summary = {
        "run_name": run_name,
        "stop_reason": stop_reason,
        "final_step": int(step),
        "best_step": int(best_step),
        "best_val_loss": float(best_val),
        "nonfinite_train_steps": int(nonfinite_train_steps),
        "elapsed_sec": float(elapsed),
        "elapsed_hr": float(elapsed / 3600.0),
        "checkpoint_best": (ckpt_dir / f"{run_name}_best.pt").as_posix(),
        "checkpoint_last": (ckpt_dir / f"{run_name}_last.pt").as_posix(),
        "history_csv": history_csv.as_posix(),
    }
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        f"[DONE] run={run_name} stop_reason={stop_reason} "
        f"best_step={best_step} best_val={best_val:.6f} elapsed_min={elapsed/60.0:.1f}"
    )
    logger.info(f"[ARTIFACT] history={history_csv}")
    logger.info(f"[ARTIFACT] summary={summary_json}")
    logger.info(f"[ARTIFACT] best_ckpt={ckpt_dir / (run_name + '_best.pt')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
