import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from ML.autoencoder.experiments.DDPMvsDDIM.core.diffusion_utils import (
    compute_min_snr_weights,
    DiffusionSchedule,
    build_condition_tensor,
    build_training_target,
    q_sample,
)
from ML.autoencoder.experiments.DDPMvsDDIM.core.latent_cache_dataset import LatentCacheDataset
from ML.autoencoder.experiments.DDPMvsDDIM.core.model_diffusion_resmlp import build_diffusion_denoiser


def parse_args():
    parser = argparse.ArgumentParser(description="Train latent DDPM-style denoiser for DDPM vs DDIM comparison.")
    parser.add_argument(
        "--train-cache",
        default="ML/autoencoder/experiments/DDPMvsDDIM/data_cache/latent_cache_eventwise_v1/train_latent_cache.pt",
    )
    parser.add_argument(
        "--val-cache",
        default="ML/autoencoder/experiments/DDPMvsDDIM/data_cache/latent_cache_eventwise_v1/val_latent_cache.pt",
    )
    parser.add_argument(
        "--stats-file",
        default="ML/autoencoder/experiments/DDPMvsDDIM/data_cache/latent_cache_eventwise_v1/latent_stats.pt",
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--beta-schedule", choices=["linear", "cosine"], default="cosine")
    parser.add_argument("--cond-mode", choices=["embedding_only", "raw_only", "embedding_plus_raw"], default="embedding_plus_raw")
    parser.add_argument("--model-type", choices=["resmlp", "adaln_resmlp"], default="resmlp")
    parser.add_argument("--prediction-target", choices=["epsilon", "v"], default="epsilon")
    parser.add_argument("--loss-weighting", choices=["none", "min_snr"], default="none")
    parser.add_argument("--min-snr-gamma", type=float, default=5.0)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--limit-train-batches", type=int, default=0)
    parser.add_argument("--limit-val-batches", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--output-dir", default="ML/autoencoder/experiments/DDPMvsDDIM/runs/diffusion")
    parser.add_argument("--run-name", default="diffusion_eventwise_v1")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def limit_reached(step_idx, limit):
    return limit > 0 and step_idx > limit


def diffusion_loss(prediction, target, schedule, timesteps, prediction_target, loss_weighting, min_snr_gamma):
    per_sample = torch.nn.functional.mse_loss(prediction, target, reduction="none").mean(dim=1)
    if loss_weighting == "none":
        return per_sample.mean()
    if loss_weighting == "min_snr":
        weights = compute_min_snr_weights(
            schedule=schedule,
            timesteps=timesteps,
            prediction_target=prediction_target,
            gamma=min_snr_gamma,
        )
        return (per_sample * weights).mean()
    raise ValueError(f"Unsupported loss_weighting: {loss_weighting}")


def main():
    args = parse_args()
    set_seed(args.seed)
    run_dir = Path(args.output_dir) / args.run_name
    checkpoint_dir = run_dir / "checkpoints"
    metrics_dir = run_dir / "metrics"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = LatentCacheDataset(args.train_cache, stats_pt_path=args.stats_file)
    val_ds = LatentCacheDataset(args.val_cache, stats_pt_path=args.stats_file)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        collate_fn=LatentCacheDataset.collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        collate_fn=LatentCacheDataset.collate_fn,
    )

    cond_embedding_dim = int(train_ds.cond_embedding.size(1))
    raw_condition_dim = int(train_ds.raw_condition.size(1))
    latent_dim = int(train_ds.z_mu.size(1))
    if args.cond_mode == "embedding_only":
        cond_dim = cond_embedding_dim
    elif args.cond_mode == "raw_only":
        cond_dim = raw_condition_dim
    else:
        cond_dim = cond_embedding_dim + raw_condition_dim

    model = build_diffusion_denoiser(
        model_type=args.model_type,
        latent_dim=latent_dim,
        cond_dim=cond_dim,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        dropout=args.dropout,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    schedule = DiffusionSchedule(args.timesteps, beta_schedule=args.beta_schedule).to(device)

    history = []
    best_val_loss = float("inf")

    print(
        f"[INFO] Starting diffusion training | device={device} latent_dim={latent_dim} "
        f"cond_mode={args.cond_mode} cond_dim={cond_dim} timesteps={args.timesteps}"
    )

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        train_steps = 0

        for batch_idx, batch in enumerate(train_loader, start=1):
            if limit_reached(batch_idx, args.limit_train_batches):
                break
            z0 = batch["z"].to(device)
            cond_embedding = batch["cond_embedding"].to(device)
            raw_condition = batch["raw_condition"].to(device)
            cond = build_condition_tensor(args.cond_mode, cond_embedding=cond_embedding, raw_condition=raw_condition)

            timesteps = torch.randint(0, args.timesteps, (z0.size(0),), device=device)
            noise = torch.randn_like(z0)
            z_t = q_sample(schedule, z0, timesteps, noise=noise)
            target = build_training_target(args.prediction_target, schedule, z0, z_t, timesteps, noise)
            prediction = model(z_t, timesteps, cond)

            loss = diffusion_loss(
                prediction=prediction,
                target=target,
                schedule=schedule,
                timesteps=timesteps,
                prediction_target=args.prediction_target,
                loss_weighting=args.loss_weighting,
                min_snr_gamma=args.min_snr_gamma,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()

            train_loss += float(loss.item())
            train_steps += 1

        model.eval()
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader, start=1):
                if limit_reached(batch_idx, args.limit_val_batches):
                    break
                z0 = batch["z"].to(device)
                cond_embedding = batch["cond_embedding"].to(device)
                raw_condition = batch["raw_condition"].to(device)
                cond = build_condition_tensor(args.cond_mode, cond_embedding=cond_embedding, raw_condition=raw_condition)

                timesteps = torch.randint(0, args.timesteps, (z0.size(0),), device=device)
                noise = torch.randn_like(z0)
                z_t = q_sample(schedule, z0, timesteps, noise=noise)
                target = build_training_target(args.prediction_target, schedule, z0, z_t, timesteps, noise)
                prediction = model(z_t, timesteps, cond)
                loss = diffusion_loss(
                    prediction=prediction,
                    target=target,
                    schedule=schedule,
                    timesteps=timesteps,
                    prediction_target=args.prediction_target,
                    loss_weighting=args.loss_weighting,
                    min_snr_gamma=args.min_snr_gamma,
                )
                val_loss += float(loss.item())
                val_steps += 1

        if train_steps == 0 or val_steps == 0:
            print(f"[WARN] Empty epoch {epoch + 1}")
            continue

        train_loss /= train_steps
        val_loss /= val_steps
        elapsed = time.time() - t0
        print(f"[EPOCH {epoch + 1:03d}/{args.epochs}] train_loss={train_loss:.6f} val_loss={val_loss:.6f} time={elapsed:.1f}s")

        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "epoch_seconds": elapsed})

        payload = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "config": vars(args),
            "latent_dim": latent_dim,
            "cond_embedding_dim": cond_embedding_dim,
            "raw_condition_dim": raw_condition_dim,
        }
        torch.save(payload, checkpoint_dir / "latest.pt")
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            torch.save(payload, checkpoint_dir / f"epoch_{epoch + 1:03d}.pt")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(payload, checkpoint_dir / "best.pt")
            print(f"[INFO] New best diffusion checkpoint saved: val_loss={best_val_loss:.6f}")

    with open(metrics_dir / "history.json", "w") as handle:
        json.dump({"history": history, "best_val_loss": best_val_loss, "config": vars(args)}, handle, indent=2)
    print("[INFO] Diffusion training completed.")


if __name__ == "__main__":
    main()
