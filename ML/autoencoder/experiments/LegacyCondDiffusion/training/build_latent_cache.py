import argparse
import json
from pathlib import Path
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))


import torch
from torch.utils.data import DataLoader, Subset

from ML.autoencoder.experiments.LegacyCondDiffusion.core.config_utils import load_config, save_json
from ML.autoencoder.experiments.LegacyCondDiffusion.core.dataset_stft import (
    SeismicSTFTDatasetWithMetadata,
    collate_fn_with_metadata,
)
from ML.autoencoder.experiments.LegacyCondDiffusion.core.model_stage1_wbaseline import WBaselineStage1


def parse_args():
    p = argparse.ArgumentParser(description="Build latent cache from trained Stage-1 model.")
    p.add_argument(
        "--config",
        default="ML/autoencoder/experiments/LegacyCondDiffusion/configs/stage1_default.json",
    )
    p.add_argument(
        "--checkpoint",
        default="ML/autoencoder/experiments/LegacyCondDiffusion/checkpoints/stage1_wbaseline_best.pt",
    )
    return p.parse_args()


@torch.no_grad()
def extract_split_cache(model, loader, device):
    z_mu_all = []
    w_all = []
    c_all = []
    sta_all = []
    mag_all = []
    loc_all = []
    meta = {"event_id": [], "file_name": [], "file_path": []}

    for batch in loader:
        specs, mags, locs, stas, metas = batch
        if specs is None:
            continue
        specs = specs.to(device)
        mags = mags.to(device)
        locs = locs.to(device)
        stas = stas.to(device)

        mu, _, w = model.encode_distribution(specs, mags, locs, stas)
        c_phys = model.build_raw_physical_condition(mags, locs)

        z_mu_all.append(mu.cpu())
        w_all.append(w.cpu())
        c_all.append(c_phys.cpu())
        sta_all.append(stas.cpu())
        mag_all.append(mags.cpu())
        loc_all.append(locs.cpu())

        for m in metas:
            meta["event_id"].append(m.get("event_id", "UNKNOWN"))
            meta["file_name"].append(m.get("file_name", "UNKNOWN"))
            meta["file_path"].append(m.get("file_path", ""))

    return {
        "z_mu": torch.cat(z_mu_all, dim=0),
        "w": torch.cat(w_all, dim=0),
        "c_phys": torch.cat(c_all, dim=0),
        "station_idx": torch.cat(sta_all, dim=0),
        "magnitude": torch.cat(mag_all, dim=0),
        "location": torch.cat(loc_all, dim=0),
        "meta": meta,
    }


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_root = Path("ML/autoencoder/experiments/LegacyCondDiffusion")
    cache_dir = exp_root / "data_cache"
    results_dir = exp_root / "results"
    cache_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(cfg["data"]["station_list_file"], "r") as f:
        station_list = json.load(f)

    dataset = SeismicSTFTDatasetWithMetadata(
        data_dir=cfg["data"]["data_dir"],
        event_file=cfg["data"]["event_file"],
        channels=cfg["data"]["channels"],
        magnitude_col=cfg["data"]["magnitude_col"],
        station_list=station_list,
    )

    with open(cfg["data"]["split_file"], "r") as f:
        split_payload = json.load(f)
    split_indices = {
        "train": split_payload["train"]["indices"],
        "val": split_payload["val"]["indices"],
        "test": split_payload["test"]["indices"],
    }

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

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    cache_payloads = {}
    for split_name in ("train", "val", "test"):
        subset = Subset(dataset, split_indices[split_name])
        loader = DataLoader(
            subset,
            batch_size=cfg["cache"]["batch_size"],
            shuffle=False,
            num_workers=cfg["cache"]["num_workers"],
            pin_memory=torch.cuda.is_available(),
            persistent_workers=cfg["cache"]["num_workers"] > 0,
            collate_fn=collate_fn_with_metadata,
        )
        print(f"[INFO] Extracting cache for split={split_name} n={len(subset)}")
        payload = extract_split_cache(model, loader, device)
        cache_payloads[split_name] = payload
        out_path = cache_dir / f"{split_name}_latent_cache.pt"
        torch.save(payload, out_path)
        print(f"[INFO] Saved cache: {out_path}")

    z_train = cache_payloads["train"]["z_mu"]
    z_mean = z_train.mean(dim=0)
    z_std = z_train.std(dim=0, unbiased=False).clamp(min=1e-6)

    stats_pt = cache_dir / "latent_stats.pt"
    torch.save({"z_mean": z_mean, "z_std": z_std}, stats_pt)
    stats_json = {
        "latent_dim": int(z_mean.numel()),
        "z_mean_abs_mean": float(z_mean.abs().mean().item()),
        "z_std_mean": float(z_std.mean().item()),
        "z_std_min": float(z_std.min().item()),
        "z_std_max": float(z_std.max().item()),
    }
    save_json(stats_json, str(results_dir / "latent_norm_stats.json"))
    print(f"[INFO] Saved latent stats: {stats_pt}")
    print(f"[INFO] Saved latent norm summary: {results_dir / 'latent_norm_stats.json'}")


if __name__ == "__main__":
    main()

