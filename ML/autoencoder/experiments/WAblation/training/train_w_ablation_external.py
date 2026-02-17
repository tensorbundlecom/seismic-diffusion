import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))

from ML.autoencoder.experiments.General.core.stft_dataset import (  # noqa: E402
    SeismicSTFTDatasetWithMetadata,
    collate_fn_with_metadata,
)
from ML.autoencoder.experiments.WAblation.core.loss_utils import ablation_cvae_loss  # noqa: E402
from ML.autoencoder.experiments.WAblation.core.model_w_ablation import WAblationCVAE  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Train W ablation CVAE variant on external HH data.")
    parser.add_argument("--variant_name", required=True)
    parser.add_argument("--use_mapping_network", type=int, default=1, choices=[0, 1])
    parser.add_argument("--use_station_embedding", type=int, default=1, choices=[0, 1])
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--w_dim", type=int, default=64)
    parser.add_argument("--station_emb_dim", type=int, default=16)
    parser.add_argument("--map_hidden_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--max_items", type=int, default=0, help="0 means full dataset")
    parser.add_argument("--max_train_batches", type=int, default=0, help="0 means full epoch")
    parser.add_argument("--max_val_batches", type=int, default=0, help="0 means full validation")
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--channels", type=str, default="HH")
    parser.add_argument("--data_dir", type=str, default="data/external_dataset/extracted/data/filtered_waveforms")
    parser.add_argument(
        "--event_file",
        type=str,
        default="data/external_dataset/extracted/data/events/20140101_20251101_0.0_9.0_9_339.txt",
    )
    parser.add_argument("--station_list_file", type=str, default="data/station_list_external_full.json")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def maybe_limit_subset(dataset, max_items, seed):
    if max_items <= 0 or max_items >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    subset, _ = random_split(dataset, [max_items, len(dataset) - max_items], generator=generator)
    return subset


def train():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = torch.cuda.is_available()

    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
    if not os.path.exists(args.station_list_file):
        raise FileNotFoundError(f"Station list not found: {args.station_list_file}")

    with open(args.station_list_file, "r") as f:
        station_list = json.load(f)

    dataset = SeismicSTFTDatasetWithMetadata(
        data_dir=args.data_dir,
        event_file=args.event_file,
        channels=[args.channels],
        magnitude_col="ML",
        station_list=station_list,
    )
    dataset = maybe_limit_subset(dataset, args.max_items, args.seed)
    num_stations = len(dataset.dataset.station_names) if hasattr(dataset, "dataset") else len(dataset.station_names)

    train_size = int(args.train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn_with_metadata,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=bool(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn_with_metadata,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=bool(args.num_workers > 0),
    )

    model = WAblationCVAE(
        in_channels=3,
        latent_dim=args.latent_dim,
        num_stations=num_stations,
        station_emb_dim=args.station_emb_dim,
        use_station_embedding=bool(args.use_station_embedding),
        use_mapping_network=bool(args.use_mapping_network),
        w_dim=args.w_dim,
        map_hidden_dim=args.map_hidden_dim,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    exp_dir = Path("ML/autoencoder/experiments/WAblation")
    ckpt_dir = exp_dir / "checkpoints"
    log_dir = exp_dir / "logs"
    res_dir = exp_dir / "results"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    run_id = f"{args.variant_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    log_path = log_dir / f"{run_id}.log"
    hist_path = res_dir / f"{run_id}_history.csv"
    cfg_path = res_dir / f"{run_id}_config.json"

    config_payload = vars(args).copy()
    config_payload["device"] = str(device)
    config_payload["dataset_size"] = len(dataset)
    config_payload["train_size"] = train_size
    config_payload["val_size"] = val_size
    config_payload["num_stations"] = num_stations
    with open(cfg_path, "w") as f:
        json.dump(config_payload, f, indent=2)

    with open(hist_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_recon",
                "train_kl",
                "val_loss",
                "val_recon",
                "val_kl",
                "epoch_seconds",
            ]
        )

    def log(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    log(
        f"RUN={run_id} device={device} data={len(dataset)} train/val={train_size}/{val_size} "
        f"mapping={bool(args.use_mapping_network)} station={bool(args.use_station_embedding)}"
    )
    log(
        f"hyperparams batch={args.batch_size} epochs={args.epochs} lr={args.lr} beta={args.beta} "
        f"latent={args.latent_dim} w_dim={args.w_dim}"
    )

    best_val = float("inf")
    ckpt_best = ckpt_dir / f"{run_id}_best.pt"
    ckpt_latest = ckpt_dir / f"{run_id}_latest.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = train_recon = train_kl = 0.0
        train_steps = 0

        for batch_idx, batch in enumerate(train_loader, start=1):
            specs, mags, locs, stations, _ = batch
            if specs is None:
                continue

            specs = specs.to(device)
            mags = mags.to(device)
            locs = locs.to(device)
            stations = stations.to(device)

            optimizer.zero_grad(set_to_none=True)
            recon, mu, logvar = model(specs, mags, locs, stations)
            loss, recon_l, kl_l = ablation_cvae_loss(recon, specs, mu, logvar, beta=args.beta)
            loss.backward()
            optimizer.step()

            train_loss += float(loss.item())
            train_recon += float(recon_l.item())
            train_kl += float(kl_l.item())
            train_steps += 1

            if args.max_train_batches > 0 and train_steps >= args.max_train_batches:
                break

        if train_steps == 0:
            log(f"[E{epoch:03d}] skipped (no train batches)")
            continue

        model.eval()
        val_loss = val_recon = val_kl = 0.0
        val_steps = 0
        with torch.no_grad():
            for batch in val_loader:
                specs, mags, locs, stations, _ = batch
                if specs is None:
                    continue

                specs = specs.to(device)
                mags = mags.to(device)
                locs = locs.to(device)
                stations = stations.to(device)

                recon, mu, logvar = model(specs, mags, locs, stations)
                loss, recon_l, kl_l = ablation_cvae_loss(recon, specs, mu, logvar, beta=args.beta)

                val_loss += float(loss.item())
                val_recon += float(recon_l.item())
                val_kl += float(kl_l.item())
                val_steps += 1

                if args.max_val_batches > 0 and val_steps >= args.max_val_batches:
                    break

        if val_steps == 0:
            log(f"[E{epoch:03d}] skipped (no val batches)")
            continue

        avg_train_loss = train_loss / train_steps
        avg_train_recon = train_recon / train_steps
        avg_train_kl = train_kl / train_steps
        avg_val_loss = val_loss / val_steps
        avg_val_recon = val_recon / val_steps
        avg_val_kl = val_kl / val_steps
        elapsed = time.time() - t0

        log(
            f"[E{epoch:03d}/{args.epochs:03d}] "
            f"train_loss={avg_train_loss:.2f} train_recon={avg_train_recon:.2f} train_kl={avg_train_kl:.2f} | "
            f"val_loss={avg_val_loss:.2f} val_recon={avg_val_recon:.2f} val_kl={avg_val_kl:.2f} | "
            f"time={elapsed:.1f}s"
        )

        payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_val_loss,
            "config": config_payload,
            "run_id": run_id,
            "variant_name": args.variant_name,
        }
        torch.save(payload, ckpt_latest)
        if args.save_every > 0 and epoch % args.save_every == 0:
            torch.save(payload, ckpt_dir / f"{run_id}_epoch_{epoch:03d}.pt")

        if avg_val_loss < best_val:
            best_val = avg_val_loss
            torch.save(payload, ckpt_best)
            log(f"[BEST] epoch={epoch} val_loss={best_val:.2f} checkpoint={ckpt_best.name}")

        with open(hist_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    avg_train_loss,
                    avg_train_recon,
                    avg_train_kl,
                    avg_val_loss,
                    avg_val_recon,
                    avg_val_kl,
                    elapsed,
                ]
            )

    log(f"FINISHED run={run_id} best_val={best_val:.2f}")


if __name__ == "__main__":
    train()

