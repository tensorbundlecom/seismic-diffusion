import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))

from ML.autoencoder.experiments.General.core.stft_dataset import (  # noqa: E402
    SeismicSTFTDatasetWithMetadata,
    collate_fn_with_metadata,
)
from ML.autoencoder.experiments.WSpaceVAE.core.config_utils import load_config, save_json  # noqa: E402
from ML.autoencoder.experiments.WSpaceVAE.core.loss_utils import vae_loss  # noqa: E402
from ML.autoencoder.experiments.WSpaceVAE.core.model_true_wspace_vae import TrueWSpaceCVAE  # noqa: E402
from ML.autoencoder.experiments.WSpaceVAE.core.split_utils import (  # noqa: E402
    build_eventwise_split_indices,
    save_split,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser(description="Train true W-space VAE (StyleGAN-like mapping u->w).")
    p.add_argument(
        "--config",
        default="ML/autoencoder/experiments/WSpaceVAE/configs/train_true_wspace_vae_external.json",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["train"]["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_root = Path("ML/autoencoder/experiments/WSpaceVAE")
    ckpt_dir = exp_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with open(cfg["data"]["station_list_file"], "r") as f:
        station_list = json.load(f)

    dataset = SeismicSTFTDatasetWithMetadata(
        data_dir=cfg["data"]["data_dir"],
        event_file=cfg["data"]["event_file"],
        channels=cfg["data"]["channels"],
        magnitude_col=cfg["data"]["magnitude_col"],
        station_list=station_list,
    )

    event_ids = [dataset._extract_event_id_from_filename(p.name) for p in dataset.file_paths]
    split_file = cfg["data"]["split_file"]
    if os.path.exists(split_file):
        payload = json.load(open(split_file))
        split = {k: payload[k]["indices"] for k in ("train", "val", "test")}
        print(f"[INFO] Loaded frozen split: {split_file}")
    else:
        split = build_eventwise_split_indices(
            event_ids,
            train_ratio=cfg["data"]["train_ratio"],
            val_ratio=cfg["data"]["val_ratio"],
            test_ratio=cfg["data"]["test_ratio"],
            seed=cfg["train"]["seed"],
        )
        save_split(split, event_ids, split_file)
        print(f"[INFO] Saved split: {split_file}")

    train_ds = Subset(dataset, split["train"])
    val_ds = Subset(dataset, split["val"])

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg["train"]["num_workers"] > 0,
        collate_fn=collate_fn_with_metadata,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg["train"]["num_workers"] > 0,
        collate_fn=collate_fn_with_metadata,
    )

    model = TrueWSpaceCVAE(
        in_channels=cfg["model"]["in_channels"],
        u_dim=cfg["model"]["u_dim"],
        w_dim=cfg["model"]["w_dim"],
        cond_dim=cfg["model"]["cond_dim"],
        num_stations=len(station_list),
        station_emb_dim=cfg["model"]["station_emb_dim"],
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
    beta = float(cfg["train"]["beta"])
    epochs = int(cfg["train"]["epochs"])
    best_val = float("inf")
    history = []

    print(
        f"[INFO] Training TrueWSpaceVAE | device={device} epochs={epochs} batch={cfg['train']['batch_size']} "
        f"lr={cfg['train']['lr']} beta={beta}"
    )

    for ep in range(epochs):
        t0 = time.time()
        model.train()
        tr_loss = tr_rec = tr_kl = 0.0
        tr_steps = 0
        for bi, batch in enumerate(train_loader):
            specs, mags, locs, stas, _ = batch
            if specs is None:
                continue
            specs = specs.to(device)
            mags = mags.to(device)
            locs = locs.to(device)
            stas = stas.to(device)

            optimizer.zero_grad(set_to_none=True)
            recon, mu, logvar, _, _ = model(specs, mags, locs, stas)
            loss, rec, kl = vae_loss(recon, specs, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()

            tr_loss += float(loss.item())
            tr_rec += float(rec.item())
            tr_kl += float(kl.item())
            tr_steps += 1
            if (bi + 1) % cfg["train"]["log_every_batches"] == 0:
                print(
                    f"[TRAIN][E{ep+1:03d}][B{bi+1:05d}] "
                    f"loss={loss.item():.6f} rec={rec.item():.6f} kl={kl.item():.6f}"
                )

        model.eval()
        va_loss = va_rec = va_kl = 0.0
        va_steps = 0
        with torch.no_grad():
            for batch in val_loader:
                specs, mags, locs, stas, _ = batch
                if specs is None:
                    continue
                specs = specs.to(device)
                mags = mags.to(device)
                locs = locs.to(device)
                stas = stas.to(device)

                recon, mu, logvar, _, _ = model(specs, mags, locs, stas)
                loss, rec, kl = vae_loss(recon, specs, mu, logvar, beta=beta)
                va_loss += float(loss.item())
                va_rec += float(rec.item())
                va_kl += float(kl.item())
                va_steps += 1

        tr_loss /= max(1, tr_steps)
        tr_rec /= max(1, tr_steps)
        tr_kl /= max(1, tr_steps)
        va_loss /= max(1, va_steps)
        va_rec /= max(1, va_steps)
        va_kl /= max(1, va_steps)
        dt = time.time() - t0
        print(
            f"[EPOCH {ep+1:03d}/{epochs}] train_loss={tr_loss:.6f} train_rec={tr_rec:.6f} train_kl={tr_kl:.6f} | "
            f"val_loss={va_loss:.6f} val_rec={va_rec:.6f} val_kl={va_kl:.6f} | time={dt:.1f}s"
        )
        history.append(
            {
                "epoch": ep + 1,
                "train_loss": tr_loss,
                "train_rec": tr_rec,
                "train_kl": tr_kl,
                "val_loss": va_loss,
                "val_rec": va_rec,
                "val_kl": va_kl,
            }
        )

        payload = {
            "epoch": ep + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": va_loss,
            "config": cfg,
        }
        torch.save(payload, ckpt_dir / "true_wspace_vae_latest.pt")
        if (ep + 1) % cfg["train"]["save_every_epochs"] == 0:
            torch.save(payload, ckpt_dir / f"true_wspace_vae_epoch_{ep+1:03d}.pt")
        if va_loss < best_val:
            best_val = va_loss
            torch.save(payload, ckpt_dir / "true_wspace_vae_best.pt")
            print(f"[INFO] New best checkpoint: val_loss={best_val:.6f}")

    save_json({"history": history, "best_val_loss": best_val}, str(exp_root / "results" / "train_history.json"))
    print("[INFO] TrueWSpaceVAE training completed.")


if __name__ == "__main__":
    main()

