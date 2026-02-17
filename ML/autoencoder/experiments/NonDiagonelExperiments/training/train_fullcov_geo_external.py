import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# project root: training -> NonDiagonel -> experiments -> autoencoder -> ML -> root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))

from ML.autoencoder.experiments.NonDiagonel.core.loss_utils import full_cov_loss
from ML.autoencoder.experiments.NonDiagonel.core.model_full_cov_geo import GeoFullCovCVAE
from ML.autoencoder.experiments.NonDiagonel.core.stft_dataset_geo import SeismicSTFTDatasetGeoCondition, collate_fn_geo


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


def save_history_row(csv_path: Path, row: dict):
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Train geometry-conditioned full-cov CVAE on external HH dataset.")
    parser.add_argument("--data_dir", default="data/external_dataset/extracted/data/filtered_waveforms")
    parser.add_argument("--event_file", default="data/external_dataset/extracted/data/events/20140101_20251101_0.0_9.0_9_339.txt")
    parser.add_argument("--station_list_file", default="data/station_list_external_full.json")
    parser.add_argument("--station_coords_file", default="ML/autoencoder/experiments/NonDiagonel/results/station_coords_external.json")
    parser.add_argument("--condition_stats_file", default="")
    parser.add_argument("--run_name", default="")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--condition_dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_items", type=int, default=0, help="Optional cap for fast pilot runs.")
    parser.add_argument("--amp", type=int, default=1, help="Use mixed precision on CUDA (1/0).")
    args = parser.parse_args()

    if not os.path.exists(args.station_coords_file):
        raise FileNotFoundError(
            f"Station coordinate cache not found: {args.station_coords_file}. "
            f"Run setup/build_station_geometry_cache.py first."
        )

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"fullcov_geo_external_{ts}"

    root = Path("ML/autoencoder/experiments/NonDiagonel")
    ckpt_dir = root / "checkpoints"
    log_dir = root / "logs"
    res_dir = root / "results"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(log_dir / f"{run_name}.log")
    hist_path = res_dir / f"{run_name}_history.csv"
    cfg_path = res_dir / f"{run_name}_config.json"

    with open(args.station_list_file, "r") as f:
        station_list = json.load(f)

    dataset = SeismicSTFTDatasetGeoCondition(
        data_dir=args.data_dir,
        event_file=args.event_file,
        station_coords_file=args.station_coords_file,
        channels=["HH"],
        magnitude_col="ML",
        station_list=station_list,
        condition_stats_file=(args.condition_stats_file if args.condition_stats_file else None),
    )

    if args.max_items and args.max_items > 0:
        dataset.file_paths = dataset.file_paths[: args.max_items]
        logger.info(f"[INFO] max_items applied: {len(dataset.file_paths)}")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    gen = torch.Generator().manual_seed(args.seed)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=gen)

    # Prefer reusing baseline stats if provided; otherwise fit from this train split.
    if not args.condition_stats_file:
        stats_path = res_dir / f"{run_name}_condition_stats.json"
        dataset.fit_condition_stats(train_set.indices)
        dataset.save_condition_stats(str(stats_path))
    else:
        stats_path = Path(args.condition_stats_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = torch.cuda.is_available()
    persistent_workers = args.num_workers > 0

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
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

    model = GeoFullCovCVAE(
        in_channels=3,
        latent_dim=args.latent_dim,
        num_stations=len(dataset.station_names),
        condition_dim=args.condition_dim,
        numeric_condition_dim=5,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    use_amp = (args.amp == 1 and device.type == "cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        autocast_ctx = lambda: torch.amp.autocast("cuda", enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        autocast_ctx = lambda: torch.cuda.amp.autocast(enabled=use_amp)

    config = {
        "run_name": run_name,
        "model": "GeoFullCovCVAE",
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "lr": args.lr,
        "beta": args.beta,
        "latent_dim": args.latent_dim,
        "condition_dim": args.condition_dim,
        "seed": args.seed,
        "max_items": args.max_items,
        "station_coords_file": args.station_coords_file,
        "condition_stats_file": str(stats_path),
        "dataset_size": len(dataset),
        "train_size": train_size,
        "val_size": val_size,
        "num_stations": len(dataset.station_names),
    }
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2)

    logger.info(
        f"RUN={run_name} device={device} data={len(dataset)} train/val={train_size}/{val_size} "
        f"batch={args.batch_size} epochs={args.epochs} lr={args.lr} beta={args.beta}"
    )

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        tr_loss = tr_rec = tr_kl = 0.0
        tr_steps = 0

        for batch in train_loader:
            specs, conds, stations, _ = batch
            if specs is None:
                continue
            specs = specs.to(device, non_blocking=True)
            conds = conds.to(device, non_blocking=True)
            stations = stations.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx():
                recon, mu, L = model(specs, conds, stations)
                loss, rec, kl = full_cov_loss(recon, specs, mu, L, beta=args.beta)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            tr_loss += float(loss.item())
            tr_rec += float(rec.item())
            tr_kl += float(kl.item())
            tr_steps += 1

        model.eval()
        va_loss = va_rec = va_kl = 0.0
        va_steps = 0
        with torch.no_grad():
            for batch in val_loader:
                specs, conds, stations, _ = batch
                if specs is None:
                    continue
                specs = specs.to(device, non_blocking=True)
                conds = conds.to(device, non_blocking=True)
                stations = stations.to(device, non_blocking=True)
                with autocast_ctx():
                    recon, mu, L = model(specs, conds, stations)
                    loss, rec, kl = full_cov_loss(recon, specs, mu, L, beta=args.beta)
                va_loss += float(loss.item())
                va_rec += float(rec.item())
                va_kl += float(kl.item())
                va_steps += 1

        tr_steps = max(tr_steps, 1)
        va_steps = max(va_steps, 1)
        row = {
            "epoch": epoch,
            "train_loss": tr_loss / tr_steps,
            "train_recon": tr_rec / tr_steps,
            "train_kl": tr_kl / tr_steps,
            "val_loss": va_loss / va_steps,
            "val_recon": va_rec / va_steps,
            "val_kl": va_kl / va_steps,
            "epoch_seconds": time.time() - t0,
        }
        save_history_row(hist_path, row)
        logger.info(
            f"[E{epoch:03d}/{args.epochs:03d}] "
            f"train_loss={row['train_loss']:.2f} train_recon={row['train_recon']:.2f} train_kl={row['train_kl']:.2f} | "
            f"val_loss={row['val_loss']:.2f} val_recon={row['val_recon']:.2f} val_kl={row['val_kl']:.2f} | "
            f"time={row['epoch_seconds']:.1f}s"
        )

        latest_path = ckpt_dir / f"{run_name}_latest.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": row["val_loss"],
                "config": config,
                "condition_stats_file": str(stats_path),
            },
            latest_path,
        )

        if row["val_loss"] < best_val:
            best_val = row["val_loss"]
            best_path = ckpt_dir / f"{run_name}_best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": row["val_loss"],
                    "config": config,
                    "condition_stats_file": str(stats_path),
                },
                best_path,
            )
            logger.info(f"[BEST] epoch={epoch} val_loss={best_val:.2f} checkpoint={best_path.name}")

        if epoch % 10 == 0:
            ep_path = ckpt_dir / f"{run_name}_epoch_{epoch:03d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": row["val_loss"],
                    "config": config,
                    "condition_stats_file": str(stats_path),
                },
                ep_path,
            )

    logger.info(f"FINISHED run={run_name} best_val={best_val:.2f}")


if __name__ == "__main__":
    main()
