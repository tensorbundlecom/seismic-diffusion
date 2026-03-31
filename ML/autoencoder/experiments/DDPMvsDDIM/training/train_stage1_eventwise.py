import argparse
import csv
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from ML.autoencoder.experiments.DDPMvsDDIM.core.loss_utils import legacy_cond_baseline_cvae_loss
from ML.autoencoder.experiments.DDPMvsDDIM.core.model_legacy_cond_baseline import LegacyCondBaselineCVAE
from ML.autoencoder.experiments.DDPMvsDDIM.core.split_utils import (
    build_hybrid_eventwise_split,
    load_split_indices,
    save_split_artifacts,
)
from ML.autoencoder.experiments.DDPMvsDDIM.core.stft_dataset import (
    SeismicSTFTDatasetWithMetadata,
    collate_fn_with_metadata,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Stage-1 localized baseline VAE with frozen event-wise split.")
    parser.add_argument("--data-dir", default="data/external_dataset/extracted/data/filtered_waveforms")
    parser.add_argument(
        "--event-file",
        default="data/external_dataset/extracted/data/events/20140101_20251101_0.0_9.0_9_339.txt",
    )
    parser.add_argument("--station-list-file", default="data/station_list_external_full.json")
    parser.add_argument("--channels", nargs="+", default=["HH"])
    parser.add_argument("--magnitude-col", default="ML")
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--cond-embedding-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--limit-train-batches", type=int, default=0)
    parser.add_argument("--limit-val-batches", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument(
        "--split-file",
        default="ML/autoencoder/experiments/DDPMvsDDIM/protocol/eventwise_split_v1.json",
    )
    parser.add_argument(
        "--split-summary-file",
        default="ML/autoencoder/experiments/DDPMvsDDIM/protocol/eventwise_split_summary_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        default="ML/autoencoder/experiments/DDPMvsDDIM/runs/stage1",
    )
    parser.add_argument("--run-name", default="stage1_eventwise_v1")
    parser.add_argument(
        "--stable-checkpoint-dir",
        default="ML/autoencoder/experiments/DDPMvsDDIM/checkpoints",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def limit_reached(step_idx, limit):
    return limit > 0 and step_idx > limit


def make_loader(dataset, batch_size, shuffle, num_workers, pin_memory):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn_with_metadata,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def ensure_split(dataset, args):
    event_ids = [dataset._extract_event_id_from_filename(path.name) for path in dataset.file_paths]
    split_file = Path(args.split_file)
    summary_file = Path(args.split_summary_file)

    if split_file.exists():
        print(f"[INFO] Loaded frozen event-wise split: {split_file}")
        return load_split_indices(str(split_file))

    split_indices, event_info = build_hybrid_eventwise_split(
        event_ids=event_ids,
        event_catalog=dataset.event_catalog,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        primary_mag_col="ML",
        fallback_mag_col="xM",
    )
    save_split_artifacts(split_indices, event_ids, event_info, str(split_file), str(summary_file))
    print(f"[INFO] Saved frozen event-wise split: {split_file}")
    print(f"[INFO] Saved split summary: {summary_file}")
    return split_indices


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.output_dir) / args.run_name
    checkpoint_dir = run_dir / "checkpoints"
    metrics_dir = run_dir / "metrics"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    stable_ckpt_dir = Path(args.stable_checkpoint_dir)
    stable_ckpt_dir.mkdir(parents=True, exist_ok=True)

    with open(args.station_list_file, "r") as handle:
        station_list = json.load(handle)
    print(f"[INFO] Loaded fixed station list: {len(station_list)} stations")

    dataset = SeismicSTFTDatasetWithMetadata(
        data_dir=args.data_dir,
        event_file=args.event_file,
        channels=args.channels,
        magnitude_col=args.magnitude_col,
        station_list=station_list,
    )
    num_stations = len(dataset.station_names)
    print(f"[INFO] Dataset size={len(dataset)} | stations in mapping={num_stations}")

    split_indices = ensure_split(dataset, args)
    train_dataset = Subset(dataset, split_indices["train"])
    val_dataset = Subset(dataset, split_indices["val"])
    print(f"[INFO] Event-wise split sizes | train={len(train_dataset)} val={len(val_dataset)} test={len(split_indices['test'])}")

    pin_memory = torch.cuda.is_available()
    train_loader = make_loader(train_dataset, args.batch_size, True, args.num_workers, pin_memory)
    val_loader = make_loader(val_dataset, args.batch_size, False, args.num_workers, pin_memory)

    model = LegacyCondBaselineCVAE(
        in_channels=3,
        latent_dim=args.latent_dim,
        num_stations=num_stations,
        cond_embedding_dim=args.cond_embedding_dim,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    history_path = metrics_dir / "history.csv"
    with open(history_path, "w", newline="") as history_file:
        writer = csv.writer(history_file)
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

        print("[INFO] Starting Stage-1 event-wise training")
        print(
            "[INFO] "
            f"device={device} latent_dim={args.latent_dim} cond_embedding_dim={args.cond_embedding_dim} "
            f"batch_size={args.batch_size} epochs={args.epochs} lr={args.lr} beta={args.beta}"
        )
        print(f"[INFO] Outputs will be written to: {run_dir}")

        best_val_loss = float("inf")
        for epoch in range(args.epochs):
            epoch_start = time.time()
            model.train()
            train_loss = 0.0
            train_recon = 0.0
            train_kl = 0.0
            train_steps = 0

            for batch_idx, batch in enumerate(train_loader, start=1):
                if limit_reached(batch_idx, args.limit_train_batches):
                    break
                specs, mags, locs, stations, _ = batch
                if specs is None:
                    continue

                specs = specs.to(device)
                mags = mags.to(device)
                locs = locs.to(device)
                stations = stations.to(device)

                optimizer.zero_grad(set_to_none=True)
                recon, mu, logvar = model(specs, mags, locs, stations)
                loss, recon_loss, kl_loss = legacy_cond_baseline_cvae_loss(recon, specs, mu, logvar, beta=args.beta)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_recon += recon_loss.item()
                train_kl += kl_loss.item()
                train_steps += 1

            if train_steps == 0:
                print(f"[WARN] Epoch {epoch + 1}: no valid training batches")
                continue

            avg_train_loss = train_loss / train_steps
            avg_train_recon = train_recon / train_steps
            avg_train_kl = train_kl / train_steps

            model.eval()
            val_loss = 0.0
            val_recon = 0.0
            val_kl = 0.0
            val_steps = 0
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader, start=1):
                    if limit_reached(batch_idx, args.limit_val_batches):
                        break
                    specs, mags, locs, stations, _ = batch
                    if specs is None:
                        continue

                    specs = specs.to(device)
                    mags = mags.to(device)
                    locs = locs.to(device)
                    stations = stations.to(device)

                    recon, mu, logvar = model(specs, mags, locs, stations)
                    loss, recon_loss, kl_loss = legacy_cond_baseline_cvae_loss(
                        recon, specs, mu, logvar, beta=args.beta
                    )
                    val_loss += loss.item()
                    val_recon += recon_loss.item()
                    val_kl += kl_loss.item()
                    val_steps += 1

            if val_steps == 0:
                print(f"[WARN] Epoch {epoch + 1}: no valid validation batches")
                continue

            avg_val_loss = val_loss / val_steps
            avg_val_recon = val_recon / val_steps
            avg_val_kl = val_kl / val_steps
            elapsed = time.time() - epoch_start
            print(
                f"[EPOCH {epoch + 1:03d}/{args.epochs}] "
                f"train_loss={avg_train_loss:.2f} train_recon={avg_train_recon:.2f} train_kl={avg_train_kl:.2f} | "
                f"val_loss={avg_val_loss:.2f} val_recon={avg_val_recon:.2f} val_kl={avg_val_kl:.2f} | "
                f"time={elapsed:.1f}s"
            )

            writer.writerow(
                [
                    epoch + 1,
                    avg_train_loss,
                    avg_train_recon,
                    avg_train_kl,
                    avg_val_loss,
                    avg_val_recon,
                    avg_val_kl,
                    elapsed,
                ]
            )
            history_file.flush()

            payload = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_val_loss,
                "config": {
                    "latent_dim": args.latent_dim,
                    "cond_embedding_dim": args.cond_embedding_dim,
                    "w_dim": args.cond_embedding_dim,
                    "num_stations": num_stations,
                    "in_channels": 3,
                    "beta": args.beta,
                    "lr": args.lr,
                    "seed": args.seed,
                    "split_file": args.split_file,
                    "run_name": args.run_name,
                },
            }

            latest_path = checkpoint_dir / "latest.pt"
            torch.save(payload, latest_path)
            shutil.copy2(latest_path, stable_ckpt_dir / f"{args.run_name}_latest.pt")

            if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
                torch.save(payload, checkpoint_dir / f"epoch_{epoch + 1:03d}.pt")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_path = checkpoint_dir / "best.pt"
                torch.save(payload, best_path)
                shutil.copy2(best_path, stable_ckpt_dir / f"{args.run_name}_best.pt")
                print(f"[INFO] New best model saved: val_loss={best_val_loss:.2f}")

    with open(metrics_dir / "run_config.json", "w") as handle:
        json.dump(vars(args), handle, indent=2)
    print("[INFO] Stage-1 event-wise training completed.")


if __name__ == "__main__":
    main()
