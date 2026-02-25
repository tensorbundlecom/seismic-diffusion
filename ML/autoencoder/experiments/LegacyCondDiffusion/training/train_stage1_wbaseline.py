import argparse
import json
import os
import random
import time
from pathlib import Path
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))


import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from ML.autoencoder.experiments.LegacyCondDiffusion.core.config_utils import load_config, save_json
from ML.autoencoder.experiments.LegacyCondDiffusion.core.dataset_stft import (
    SeismicSTFTDatasetWithMetadata,
    collate_fn_with_metadata,
)
from ML.autoencoder.experiments.LegacyCondDiffusion.core.loss_utils import cvae_loss
from ML.autoencoder.experiments.LegacyCondDiffusion.core.model_stage1_wbaseline import WBaselineStage1
from ML.autoencoder.experiments.LegacyCondDiffusion.core.split_utils import (
    build_eventwise_split_indices,
    save_split_file,
    summarize_split,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser(description="Train Stage-1 legacy-condition backbone (isolated).")
    p.add_argument(
        "--config",
        default="ML/autoencoder/experiments/LegacyCondDiffusion/configs/stage1_default.json",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    seed = int(cfg["train"]["seed"])
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_root = Path("ML/autoencoder/experiments/LegacyCondDiffusion")
    ckpt_dir = exp_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    protocol_dir = exp_root / "protocol"
    protocol_dir.mkdir(parents=True, exist_ok=True)

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
        with open(split_file, "r") as f:
            payload = json.load(f)
        split_indices = {
            "train": payload["train"]["indices"],
            "val": payload["val"]["indices"],
            "test": payload["test"]["indices"],
        }
        print(f"[INFO] Loaded frozen split: {split_file}")
    else:
        split_indices = build_eventwise_split_indices(
            event_ids=event_ids,
            train_ratio=cfg["data"]["train_ratio"],
            val_ratio=cfg["data"]["val_ratio"],
            test_ratio=cfg["data"]["test_ratio"],
            seed=seed,
        )
        save_split_file(split_indices, event_ids=event_ids, out_path=split_file)
        print(f"[INFO] Saved new frozen split: {split_file}")

    split_summary = summarize_split(split_indices, event_ids=event_ids)
    save_json(split_summary, str(protocol_dir / "split_summary_stage1.json"))
    print(f"[INFO] Split summary: {split_summary}")

    train_ds = Subset(dataset, split_indices["train"])
    val_ds = Subset(dataset, split_indices["val"])

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

    model = WBaselineStage1(
        in_channels=cfg["model"]["in_channels"],
        latent_dim=cfg["model"]["latent_dim"],
        num_stations=len(station_list),
        w_dim=cfg["model"]["w_dim"],
        station_emb_dim=cfg["model"]["station_emb_dim"],
        map_hidden_dim=cfg["model"]["map_hidden_dim"],
        mag_min=cfg["data"]["mag_min"],
        mag_max=cfg["data"]["mag_max"],
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
    best_val = float("inf")

    epochs = int(cfg["train"]["epochs"])
    beta = float(cfg["train"]["beta"])
    print(
        f"[INFO] Stage-1 training starts | device={device} epochs={epochs} "
        f"batch={cfg['train']['batch_size']} lr={cfg['train']['lr']} beta={beta}"
    )

    for ep in range(epochs):
        t0 = time.time()
        model.train()
        tr_loss = tr_rec = tr_kl = 0.0
        tr_steps = 0

        for batch_idx, batch in enumerate(train_loader):
            specs, mags, locs, stas, _ = batch
            if specs is None:
                continue
            specs = specs.to(device)
            mags = mags.to(device)
            locs = locs.to(device)
            stas = stas.to(device)

            optimizer.zero_grad(set_to_none=True)
            recon, mu, logvar = model(specs, mags, locs, stas)
            loss, rec, kl = cvae_loss(recon, specs, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()

            tr_loss += float(loss.item())
            tr_rec += float(rec.item())
            tr_kl += float(kl.item())
            tr_steps += 1

            if (batch_idx + 1) % cfg["train"]["log_every_batches"] == 0:
                print(
                    f"[TRAIN][E{ep+1:03d}][B{batch_idx+1:05d}] "
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
                recon, mu, logvar = model(specs, mags, locs, stas)
                loss, rec, kl = cvae_loss(recon, specs, mu, logvar, beta=beta)
                va_loss += float(loss.item())
                va_rec += float(rec.item())
                va_kl += float(kl.item())
                va_steps += 1

        if tr_steps == 0 or va_steps == 0:
            print(f"[WARN] Empty epoch {ep+1}.")
            continue

        tr_loss /= tr_steps
        tr_rec /= tr_steps
        tr_kl /= tr_steps
        va_loss /= va_steps
        va_rec /= va_steps
        va_kl /= va_steps

        elapsed = time.time() - t0
        print(
            f"[EPOCH {ep+1:03d}/{epochs}] "
            f"train_loss={tr_loss:.6f} train_rec={tr_rec:.6f} train_kl={tr_kl:.6f} | "
            f"val_loss={va_loss:.6f} val_rec={va_rec:.6f} val_kl={va_kl:.6f} | "
            f"time={elapsed:.1f}s"
        )

        ckpt_payload = {
            "epoch": ep + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": va_loss,
            "config": cfg,
        }

        torch.save(ckpt_payload, ckpt_dir / "stage1_wbaseline_latest.pt")
        if (ep + 1) % cfg["train"]["save_every_epochs"] == 0:
            torch.save(ckpt_payload, ckpt_dir / f"stage1_wbaseline_epoch_{ep+1:03d}.pt")
        if va_loss < best_val:
            best_val = va_loss
            torch.save(ckpt_payload, ckpt_dir / "stage1_wbaseline_best.pt")
            print(f"[INFO] New best checkpoint saved: val_loss={best_val:.6f}")

    print("[INFO] Stage-1 training completed.")


if __name__ == "__main__":
    main()
