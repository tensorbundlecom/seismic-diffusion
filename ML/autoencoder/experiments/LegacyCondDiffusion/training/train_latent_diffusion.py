import argparse
import random
import time
from pathlib import Path
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))


import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from ML.autoencoder.experiments.LegacyCondDiffusion.core.config_utils import load_config, save_json
from ML.autoencoder.experiments.LegacyCondDiffusion.core.diffusion_utils import (
    build_condition_tensor,
    choose_denoiser,
    sample_ve_noisy_latent,
    weighted_denoise_loss,
)
from ML.autoencoder.experiments.LegacyCondDiffusion.core.latent_cache_dataset import LatentCacheDataset


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser(description="Train latent diffusion model.")
    p.add_argument(
        "--config",
        default="ML/autoencoder/experiments/LegacyCondDiffusion/configs/diffusion_resmlp_default.json",
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
    results_dir = exp_root / "results"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    train_ds = LatentCacheDataset(cfg["data"]["train_cache"], stats_pt_path=cfg["data"]["stats_file"])
    val_ds = LatentCacheDataset(cfg["data"]["val_cache"], stats_pt_path=cfg["data"]["stats_file"])

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg["train"]["num_workers"] > 0,
        collate_fn=LatentCacheDataset.collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg["train"]["num_workers"] > 0,
        collate_fn=LatentCacheDataset.collate_fn,
    )

    cond_mode = cfg["model"]["cond_mode"]
    latent_dim = int(cfg["model"]["latent_dim"])
    w_dim = int(cfg["model"]["w_dim"])
    c_dim = int(cfg["model"]["c_phys_dim"])
    cond_dim = w_dim if cond_mode == "w_only" else c_dim if cond_mode == "c_only" else (w_dim + c_dim)

    denoiser = choose_denoiser(
        denoiser_name=cfg["model"]["denoiser"],
        latent_dim=latent_dim,
        cond_dim=cond_dim,
        hidden_dim=cfg["model"].get("hidden_dim", 512),
        depth=cfg["model"].get("depth", 6),
        dropout=cfg["model"].get("dropout", 0.0),
        base_channels=cfg["model"].get("base_channels", 64),
    ).to(device)

    optimizer = optim.Adam(denoiser.parameters(), lr=cfg["train"]["lr"])
    best_val = float("inf")
    history = []

    t_min = float(cfg["diffusion"]["t_min"])
    t_max = float(cfg["diffusion"]["t_max"])
    loss_weight_mode = cfg["train"]["loss_weight_mode"]

    print(
        f"[INFO] Diffusion training starts | denoiser={cfg['model']['denoiser']} cond_mode={cond_mode} "
        f"latent_dim={latent_dim} cond_dim={cond_dim} device={device}"
    )

    for ep in range(int(cfg["train"]["epochs"])):
        t0 = time.time()
        denoiser.train()
        tr_loss = 0.0
        tr_steps = 0

        for batch_idx, batch in enumerate(train_loader):
            z0 = batch["z"].to(device)
            w = batch["w"].to(device)
            c_phys = batch["c_phys"].to(device)
            cond = build_condition_tensor(cond_mode, w=w, c_phys=c_phys)

            t = torch.rand(z0.size(0), device=device) * (t_max - t_min) + t_min
            z_t, _ = sample_ve_noisy_latent(z0, t)
            pred_z0 = denoiser(z_t, t, cond)

            loss = weighted_denoise_loss(pred_z0, z0, t, weight_mode=loss_weight_mode)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg["train"]["grad_clip_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(denoiser.parameters(), cfg["train"]["grad_clip_norm"])
            optimizer.step()

            tr_loss += float(loss.item())
            tr_steps += 1

            if (batch_idx + 1) % cfg["train"]["log_every_batches"] == 0:
                print(f"[TRAIN][E{ep+1:03d}][B{batch_idx+1:05d}] loss={loss.item():.6f}")

        denoiser.eval()
        va_loss = 0.0
        va_steps = 0
        with torch.no_grad():
            for batch in val_loader:
                z0 = batch["z"].to(device)
                w = batch["w"].to(device)
                c_phys = batch["c_phys"].to(device)
                cond = build_condition_tensor(cond_mode, w=w, c_phys=c_phys)
                t = torch.rand(z0.size(0), device=device) * (t_max - t_min) + t_min
                z_t, _ = sample_ve_noisy_latent(z0, t)
                pred_z0 = denoiser(z_t, t, cond)
                loss = weighted_denoise_loss(pred_z0, z0, t, weight_mode=loss_weight_mode)
                va_loss += float(loss.item())
                va_steps += 1

        if tr_steps == 0 or va_steps == 0:
            print(f"[WARN] Empty epoch {ep+1}.")
            continue

        tr_loss /= tr_steps
        va_loss /= va_steps
        elapsed = time.time() - t0
        print(f"[EPOCH {ep+1:03d}/{cfg['train']['epochs']}] train_loss={tr_loss:.6f} val_loss={va_loss:.6f} time={elapsed:.1f}s")

        hist_row = {"epoch": ep + 1, "train_loss": tr_loss, "val_loss": va_loss}
        history.append(hist_row)

        ckpt_payload = {
            "epoch": ep + 1,
            "model_state_dict": denoiser.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": va_loss,
            "config": cfg,
        }
        tag = cfg["run"]["name"]
        torch.save(ckpt_payload, ckpt_dir / f"{tag}_latest.pt")
        if (ep + 1) % cfg["train"]["save_every_epochs"] == 0:
            torch.save(ckpt_payload, ckpt_dir / f"{tag}_epoch_{ep+1:03d}.pt")
        if va_loss < best_val:
            best_val = va_loss
            torch.save(ckpt_payload, ckpt_dir / f"{tag}_best.pt")
            print(f"[INFO] New best checkpoint saved: val_loss={best_val:.6f}")

    save_json(
        {
            "run_name": cfg["run"]["name"],
            "denoiser": cfg["model"]["denoiser"],
            "cond_mode": cond_mode,
            "best_val_loss": best_val,
            "history": history,
        },
        str(results_dir / f"{cfg['run']['name']}_train_history.json"),
    )
    print("[INFO] Diffusion training completed.")


if __name__ == "__main__":
    main()

