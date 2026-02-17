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

# project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))

from ML.autoencoder.experiments.NonDiagonel.core.loss_utils import full_cov_loss, offdiag_l2_penalty
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


def sample_z(mu: torch.Tensor, L: torch.Tensor, training: bool) -> torch.Tensor:
    if not training:
        return mu
    b, d = mu.shape
    eps = torch.randn(b, d, 1, device=mu.device, dtype=mu.dtype)
    z = mu.unsqueeze(2) + torch.bmm(L, eps)
    return z.squeeze(2)


def forward_with_mode(model: GeoFullCovCVAE, specs, conds, stations, mode: str, training: bool):
    mu, L_raw = model.encoder(specs, conds, stations)

    if mode == "diag_only":
        L_used = torch.diag_embed(torch.diagonal(L_raw, dim1=-2, dim2=-1))
    else:
        L_used = L_raw

    z = sample_z(mu, L_used, training=training)
    recon = model.decoder(z, conds, stations)
    if recon.shape[2:] != specs.shape[2:]:
        recon = torch.nn.functional.interpolate(
            recon,
            size=specs.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
    return recon, mu, L_raw, L_used


def main():
    parser = argparse.ArgumentParser(description="Train FullCov controls: diag-only and offdiag-penalized variants.")
    parser.add_argument("--data_dir", default="data/external_dataset/extracted/data/filtered_waveforms")
    parser.add_argument("--event_file", default="data/external_dataset/extracted/data/events/20140101_20251101_0.0_9.0_9_339.txt")
    parser.add_argument("--station_list_file", default="data/station_list_external_full.json")
    parser.add_argument("--station_coords_file", default="ML/autoencoder/experiments/NonDiagonel/results/station_coords_external.json")
    parser.add_argument("--condition_stats_file", default="ML/autoencoder/experiments/NonDiagonel/results/condition_stats_external_seed42.json")
    parser.add_argument("--init_checkpoint", default="", help="Optional checkpoint to initialize model weights.")
    parser.add_argument("--run_name", default="")
    parser.add_argument("--mode", choices=["fullcov", "diag_only", "offdiag_penalty"], required=True)
    parser.add_argument("--offdiag_lambda", type=float, default=0.0, help="Used only when mode=offdiag_penalty.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--condition_dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_items", type=int, default=0)
    parser.add_argument("--amp", type=int, default=1)
    args = parser.parse_args()

    if args.mode != "offdiag_penalty" and args.offdiag_lambda != 0.0:
        raise ValueError("--offdiag_lambda should be 0 unless mode=offdiag_penalty")
    if args.mode == "offdiag_penalty" and args.offdiag_lambda <= 0.0:
        raise ValueError("mode=offdiag_penalty requires --offdiag_lambda > 0")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.run_name:
        run_name = args.run_name
    else:
        suffix = args.mode if args.mode != "offdiag_penalty" else f"{args.mode}_l{args.offdiag_lambda:g}"
        run_name = f"fullcov_geo_control_{suffix}_{ts}"

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
        condition_stats_file=args.condition_stats_file,
    )

    if args.max_items and args.max_items > 0:
        dataset.file_paths = dataset.file_paths[: args.max_items]
        logger.info(f"[INFO] max_items applied: {len(dataset.file_paths)}")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    gen = torch.Generator().manual_seed(args.seed)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=gen)

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

    if args.init_checkpoint:
        init_state = torch.load(args.init_checkpoint, map_location=device)
        model.load_state_dict(init_state["model_state_dict"])
        logger.info(f"[INIT] Loaded weights from {args.init_checkpoint}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    use_amp = args.amp == 1 and device.type == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        autocast_ctx = lambda: torch.amp.autocast("cuda", enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        autocast_ctx = lambda: torch.cuda.amp.autocast(enabled=use_amp)

    config = {
        "run_name": run_name,
        "model": "GeoFullCovCVAE",
        "mode": args.mode,
        "offdiag_lambda": args.offdiag_lambda,
        "init_checkpoint": args.init_checkpoint,
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
        "condition_stats_file": args.condition_stats_file,
        "dataset_size": len(dataset),
        "train_size": train_size,
        "val_size": val_size,
        "num_stations": len(dataset.station_names),
    }
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2)

    logger.info(
        f"RUN={run_name} mode={args.mode} device={device} data={len(dataset)} train/val={train_size}/{val_size} "
        f"batch={args.batch_size} epochs={args.epochs} lr={args.lr} beta={args.beta} offdiag_lambda={args.offdiag_lambda}"
    )

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        tr_loss = tr_rec = tr_kl = tr_pen = tr_pen_scaled = 0.0
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
                recon, mu, L_raw, L_used = forward_with_mode(model, specs, conds, stations, mode=args.mode, training=True)
                loss, rec, kl = full_cov_loss(recon, specs, mu, L_used, beta=args.beta)
                pen = torch.tensor(0.0, device=device)
                pen_scaled = torch.tensor(0.0, device=device)
                if args.mode == "offdiag_penalty":
                    pen = offdiag_l2_penalty(L_raw)
                    pen_scaled = args.offdiag_lambda * pen
                    loss = loss + pen_scaled

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            tr_loss += float(loss.item())
            tr_rec += float(rec.item())
            tr_kl += float(kl.item())
            tr_pen += float(pen.item())
            tr_pen_scaled += float(pen_scaled.item())
            tr_steps += 1

        model.eval()
        va_loss = va_rec = va_kl = va_pen = va_pen_scaled = 0.0
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
                    recon, mu, L_raw, L_used = forward_with_mode(model, specs, conds, stations, mode=args.mode, training=False)
                    loss, rec, kl = full_cov_loss(recon, specs, mu, L_used, beta=args.beta)
                    pen = torch.tensor(0.0, device=device)
                    pen_scaled = torch.tensor(0.0, device=device)
                    if args.mode == "offdiag_penalty":
                        pen = offdiag_l2_penalty(L_raw)
                        pen_scaled = args.offdiag_lambda * pen
                        loss = loss + pen_scaled

                va_loss += float(loss.item())
                va_rec += float(rec.item())
                va_kl += float(kl.item())
                va_pen += float(pen.item())
                va_pen_scaled += float(pen_scaled.item())
                va_steps += 1

        tr_steps = max(tr_steps, 1)
        va_steps = max(va_steps, 1)
        row = {
            "epoch": epoch,
            "train_loss": tr_loss / tr_steps,
            "train_recon": tr_rec / tr_steps,
            "train_kl": tr_kl / tr_steps,
            "train_offdiag_pen": tr_pen / tr_steps,
            "train_offdiag_pen_scaled": tr_pen_scaled / tr_steps,
            "val_loss": va_loss / va_steps,
            "val_recon": va_rec / va_steps,
            "val_kl": va_kl / va_steps,
            "val_offdiag_pen": va_pen / va_steps,
            "val_offdiag_pen_scaled": va_pen_scaled / va_steps,
            "epoch_seconds": time.time() - t0,
        }
        save_history_row(hist_path, row)

        logger.info(
            f"[E{epoch:03d}/{args.epochs:03d}] "
            f"train_loss={row['train_loss']:.2f} rec={row['train_recon']:.2f} kl={row['train_kl']:.2f} "
            f"pen={row['train_offdiag_pen']:.6f} pen_scaled={row['train_offdiag_pen_scaled']:.6f} | "
            f"val_loss={row['val_loss']:.2f} rec={row['val_recon']:.2f} kl={row['val_kl']:.2f} "
            f"pen={row['val_offdiag_pen']:.6f} pen_scaled={row['val_offdiag_pen_scaled']:.6f} | "
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
                },
                ep_path,
            )

    logger.info(f"FINISHED run={run_name} best_val={best_val:.2f}")


if __name__ == "__main__":
    main()
