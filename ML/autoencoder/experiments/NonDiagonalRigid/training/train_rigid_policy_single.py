#!/usr/bin/env python3
"""
Train one NonDiagonalRigid Phase-1 policy run with frozen splits/stats.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

# project root: training -> NonDiagonalRigid -> experiments -> autoencoder -> ML -> root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))

from ML.autoencoder.experiments.NonDiagonel.core.loss_utils import beta_cvae_loss, full_cov_loss
from ML.autoencoder.experiments.NonDiagonel.core.stft_dataset_geo import SeismicSTFTDatasetGeoCondition, collate_fn_geo
from ML.autoencoder.experiments.NonDiagonalRigid.core.model_policy_geo import (
    PolicyGeoConditionalVariationalAutoencoder,
    PolicyGeoFullCovCVAE,
    SUPPORTED_POLICIES,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(str(log_path))
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fmt = logging.Formatter("%(asctime)s | %(message)s")

    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def parse_channels(arg: str) -> List[int]:
    out = []
    for x in arg.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(int(x))
    if len(out) != 4:
        raise ValueError(f"base_channels must have 4 integers, got: {arg}")
    return out


def map_files_to_indices(dataset_files: List[Path], wanted_files: Iterable[str]) -> List[int]:
    idx_by_file = {Path(p).as_posix(): i for i, p in enumerate(dataset_files)}
    out = []
    missing = []
    for fp in wanted_files:
        key = Path(fp).as_posix()
        if key in idx_by_file:
            out.append(idx_by_file[key])
        else:
            missing.append(key)
    if missing:
        raise RuntimeError(f"{len(missing)} files from frozen split not found in dataset. Example: {missing[:3]}")
    return out


def write_csv_row(path: Path, row: Dict) -> None:
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def build_model(
    model_family: str,
    policy: str,
    scale: float,
    latent_dim: int,
    condition_dim: int,
    num_stations: int,
    base_channels: Sequence[int],
    device: torch.device,
):
    if model_family == "baseline_diag":
        model = PolicyGeoConditionalVariationalAutoencoder(
            in_channels=3,
            latent_dim=latent_dim,
            num_stations=num_stations,
            condition_dim=condition_dim,
            numeric_condition_dim=5,
            policy=policy,
            scale=scale,
            base_channels=base_channels,
        )
    elif model_family == "fullcov":
        model = PolicyGeoFullCovCVAE(
            in_channels=3,
            latent_dim=latent_dim,
            num_stations=num_stations,
            condition_dim=condition_dim,
            numeric_condition_dim=5,
            policy=policy,
            scale=scale,
            base_channels=base_channels,
        )
    else:
        raise ValueError(f"Unsupported model_family: {model_family}")
    return model.to(device)


def evaluate_val(
    model: torch.nn.Module,
    model_family: str,
    val_loader: DataLoader,
    beta: float,
    device: torch.device,
    autocast_ctx,
) -> Tuple[float, float, float, int]:
    model.eval()
    va_loss = va_rec = va_kl = 0.0
    va_steps = 0
    with torch.no_grad():
        for specs, conds, stations, _ in val_loader:
            if specs is None:
                continue
            specs = specs.to(device, non_blocking=True)
            conds = conds.to(device, non_blocking=True)
            stations = stations.to(device, non_blocking=True)
            with autocast_ctx():
                if model_family == "baseline_diag":
                    recon, mu, logvar = model(specs, conds, stations)
                    loss, rec, kl = beta_cvae_loss(recon, specs, mu, logvar, beta=beta)
                else:
                    recon, mu, L = model(specs, conds, stations)
                    loss, rec, kl = full_cov_loss(recon, specs, mu, L, beta=beta)
            va_loss += float(loss.item())
            va_rec += float(rec.item())
            va_kl += float(kl.item())
            va_steps += 1
    va_steps = max(va_steps, 1)
    return va_loss / va_steps, va_rec / va_steps, va_kl / va_steps, va_steps


def save_ckpt(path: Path, payload: Dict, logger: logging.Logger) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    except OSError as e:
        logger.error(f"[CKPT-ERROR] path={path} error={e}")
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Train one NonDiagonalRigid policy-aware run with frozen protocol.")
    parser.add_argument("--model_family", choices=["baseline_diag", "fullcov"], required=True)
    parser.add_argument("--policy", choices=sorted(list(SUPPORTED_POLICIES)), required=True)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--base_channels", default="32,64,128,256")

    parser.add_argument("--condition_dim", type=int, default=64)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--amp", type=int, default=1)

    parser.add_argument("--max_steps", type=int, default=12000)
    parser.add_argument("--val_check_every_steps", type=int, default=2000)
    parser.add_argument("--patience_evals", type=int, default=9999)
    parser.add_argument("--min_delta", type=float, default=0.0)
    parser.add_argument("--min_steps_before_early_stop", type=int, default=0)
    parser.add_argument("--train_log_every_steps", type=int, default=100)

    parser.add_argument(
        "--frozen_split_manifest",
        default="ML/autoencoder/experiments/NonDiagonalRigid/protocol/frozen_splits_v1.json",
    )
    parser.add_argument(
        "--normalization_stats_file",
        default="ML/autoencoder/experiments/NonDiagonalRigid/protocol/normalization_stats_v1.json",
    )
    parser.add_argument(
        "--station_coords_file",
        default="ML/autoencoder/experiments/NonDiagonel/results/station_coords_external.json",
    )
    parser.add_argument(
        "--root_dir",
        default="ML/autoencoder/experiments/NonDiagonalRigid",
    )
    parser.add_argument("--run_name", default="")
    args = parser.parse_args()

    set_seed(args.seed)
    if args.min_steps_before_early_stop <= 0:
        args.min_steps_before_early_stop = max(
            5 * args.val_check_every_steps,
            int(0.20 * args.max_steps),
        )

    base_channels = parse_channels(args.base_channels)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    scale_tag = str(args.scale).replace(".", "p")
    run_name = args.run_name or (
        f"rigid_policy_{args.model_family}_{args.policy}_sc{scale_tag}_ld{args.latent_dim}_s{args.seed}_{ts}"
    )

    root = Path(args.root_dir)
    ckpt_dir = root / "checkpoints"
    log_dir = root / "logs"
    res_dir = root / "results" / run_name
    for d in [ckpt_dir, log_dir, res_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(log_dir / f"{run_name}.log")
    history_csv = res_dir / "history.csv"
    run_cfg_json = res_dir / "run_config.json"

    split_manifest = read_json(Path(args.frozen_split_manifest))
    norm_stats = read_json(Path(args.normalization_stats_file))
    train_files = read_lines(Path(split_manifest["splits"]["train_files"]["file"]))
    val_files = read_lines(Path(split_manifest["splits"]["val_files"]["file"]))
    station_list = split_manifest["stations"]

    id_root = split_manifest["policy"]["id_channel_dir"]
    event_file = split_manifest["policy"]["event_file"]

    dataset = SeismicSTFTDatasetGeoCondition(
        data_dir=str(Path(id_root).parent),
        event_file=event_file,
        station_coords_file=args.station_coords_file,
        channels=["HH"],
        magnitude_col="ML",
        station_list=station_list,
        condition_stats_file=str(Path(args.normalization_stats_file)),
    )

    if "features" in norm_stats and "mean" in norm_stats["features"] and "mean" not in norm_stats:
        dataset.condition_stats = norm_stats["features"]

    train_idx = map_files_to_indices(dataset.file_paths, train_files)
    val_idx = map_files_to_indices(dataset.file_paths, val_files)
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = torch.cuda.is_available()
    persistent_workers = args.num_workers > 0
    train_gen = torch.Generator().manual_seed(args.seed)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        generator=train_gen,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn_geo,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn_geo,
    )

    model = build_model(
        model_family=args.model_family,
        policy=args.policy,
        scale=args.scale,
        latent_dim=args.latent_dim,
        condition_dim=args.condition_dim,
        num_stations=len(station_list),
        base_channels=base_channels,
        device=device,
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    use_amp = (args.amp == 1 and device.type == "cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        autocast_ctx = lambda: torch.amp.autocast("cuda", enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        autocast_ctx = lambda: torch.cuda.amp.autocast(enabled=use_amp)

    run_cfg = {
        "version": "NonDiagonalRigid-v1-policy",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": run_name,
        "model_family": args.model_family,
        "policy": args.policy,
        "scale": args.scale,
        "base_channels": base_channels,
        "latent_dim": args.latent_dim,
        "seed": args.seed,
        "condition_dim": args.condition_dim,
        "beta": args.beta,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "amp": args.amp,
        "max_steps": args.max_steps,
        "val_check_every_steps": args.val_check_every_steps,
        "patience_evals": args.patience_evals,
        "min_delta": args.min_delta,
        "min_steps_before_early_stop": args.min_steps_before_early_stop,
        "train_log_every_steps": args.train_log_every_steps,
        "frozen_split_manifest": args.frozen_split_manifest,
        "normalization_stats_file": args.normalization_stats_file,
        "station_coords_file": args.station_coords_file,
        "train_count": len(train_idx),
        "val_count": len(val_idx),
        "num_stations": len(station_list),
        "device": str(device),
        "backbone_plan": getattr(model, "backbone_plan", None),
    }
    with run_cfg_json.open("w", encoding="utf-8") as f:
        json.dump(run_cfg, f, indent=2)

    logger.info(
        f"RUN={run_name} model={args.model_family} policy={args.policy} scale={args.scale} ld={args.latent_dim} "
        f"seed={args.seed} device={device} train/val={len(train_idx)}/{len(val_idx)}"
    )
    logger.info(
        f"SCHED max_steps={args.max_steps} val_every={args.val_check_every_steps} "
        f"patience_evals={args.patience_evals} min_steps_before_early_stop={args.min_steps_before_early_stop}"
    )

    global_step = 0
    epoch = 0
    best_val = float("inf")
    best_step = -1
    bad_evals = 0
    stop_reason = "max_steps"
    t_start = time.time()

    while global_step < args.max_steps:
        epoch += 1
        model.train()
        for specs, conds, stations, _ in train_loader:
            if specs is None:
                continue

            specs = specs.to(device, non_blocking=True)
            conds = conds.to(device, non_blocking=True)
            stations = stations.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx():
                if args.model_family == "baseline_diag":
                    recon, mu, logvar = model(specs, conds, stations)
                    loss, rec, kl = beta_cvae_loss(recon, specs, mu, logvar, beta=args.beta)
                else:
                    recon, mu, L = model(specs, conds, stations)
                    loss, rec, kl = full_cov_loss(recon, specs, mu, L, beta=args.beta)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            global_step += 1

            if args.train_log_every_steps > 0 and (global_step % args.train_log_every_steps == 0):
                logger.info(
                    f"[step {global_step:>7}] train_loss={float(loss.item()):.4f} "
                    f"train_rec={float(rec.item()):.4f} train_kl={float(kl.item()):.4f} "
                    f"elapsed={time.time() - t_start:.1f}s"
                )

            if global_step % args.val_check_every_steps == 0 or global_step >= args.max_steps:
                t_eval = time.time()
                val_loss, val_rec, val_kl, val_steps = evaluate_val(
                    model=model,
                    model_family=args.model_family,
                    val_loader=val_loader,
                    beta=args.beta,
                    device=device,
                    autocast_ctx=autocast_ctx,
                )
                eval_sec = time.time() - t_eval

                improved = val_loss < (best_val - args.min_delta)
                if improved:
                    best_val = val_loss
                    best_step = global_step
                    bad_evals = 0
                else:
                    bad_evals += 1

                row = {
                    "step": global_step,
                    "epoch": epoch,
                    "train_loss_step": float(loss.item()),
                    "train_recon_step": float(rec.item()),
                    "train_kl_step": float(kl.item()),
                    "val_loss": val_loss,
                    "val_recon": val_rec,
                    "val_kl": val_kl,
                    "val_steps": val_steps,
                    "improved": int(improved),
                    "best_val_so_far": best_val,
                    "best_step_so_far": best_step,
                    "bad_evals": bad_evals,
                    "eval_seconds": eval_sec,
                    "elapsed_seconds": time.time() - t_start,
                }
                write_csv_row(history_csv, row)

                logger.info(
                    f"[step {global_step:>7}] val_loss={val_loss:.4f} val_rec={val_rec:.4f} val_kl={val_kl:.4f} "
                    f"best={best_val:.4f}@{best_step} bad_evals={bad_evals} improved={improved}"
                )

                payload = {
                    "step": global_step,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val": best_val,
                    "best_step": best_step,
                    "config": run_cfg,
                }
                save_ckpt(ckpt_dir / f"{run_name}_latest.pt", payload, logger)
                if improved:
                    save_ckpt(ckpt_dir / f"{run_name}_best.pt", payload, logger)

                can_early_stop = global_step >= args.min_steps_before_early_stop
                if can_early_stop and bad_evals >= args.patience_evals:
                    stop_reason = "early_stop"
                    break

            if global_step >= args.max_steps:
                stop_reason = "max_steps"
                break
        if stop_reason == "early_stop":
            break

    final_summary = {
        "run_name": run_name,
        "stop_reason": stop_reason,
        "global_step": global_step,
        "epochs_seen": epoch,
        "best_val": best_val,
        "best_step": best_step,
        "elapsed_seconds": time.time() - t_start,
    }
    with (res_dir / "final_summary.json").open("w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2)

    logger.info(
        f"FINISHED run={run_name} stop_reason={stop_reason} "
        f"best_val={best_val:.4f} best_step={best_step} elapsed={final_summary['elapsed_seconds']:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
