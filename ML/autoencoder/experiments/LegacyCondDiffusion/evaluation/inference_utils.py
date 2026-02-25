import json

import torch

from ML.autoencoder.experiments.LegacyCondDiffusion.core.config_utils import load_config
from ML.autoencoder.experiments.LegacyCondDiffusion.core.diffusion_utils import (
    build_condition_tensor,
    choose_denoiser,
    heun_sample_ve,
)
from ML.autoencoder.experiments.LegacyCondDiffusion.core.model_stage1_wbaseline import WBaselineStage1


def load_stage1_model(stage1_ckpt: str, station_list_file: str, device: torch.device):
    with open(station_list_file, "r") as f:
        station_list = json.load(f)

    ckpt = torch.load(stage1_ckpt, map_location=device)
    cfg = ckpt["config"]
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
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, station_list


def load_diffusion_model(diff_ckpt: str, device: torch.device):
    ckpt = torch.load(diff_ckpt, map_location=device)
    cfg = ckpt["config"]

    cond_mode = cfg["model"]["cond_mode"]
    w_dim = cfg["model"]["w_dim"]
    c_dim = cfg["model"]["c_phys_dim"]
    cond_dim = w_dim if cond_mode == "w_only" else c_dim if cond_mode == "c_only" else (w_dim + c_dim)

    denoiser = choose_denoiser(
        denoiser_name=cfg["model"]["denoiser"],
        latent_dim=cfg["model"]["latent_dim"],
        cond_dim=cond_dim,
        hidden_dim=cfg["model"].get("hidden_dim", 512),
        depth=cfg["model"].get("depth", 6),
        dropout=cfg["model"].get("dropout", 0.0),
        base_channels=cfg["model"].get("base_channels", 64),
    ).to(device)
    denoiser.load_state_dict(ckpt["model_state_dict"])
    denoiser.eval()
    return denoiser, cfg


@torch.no_grad()
def sample_latent_from_condition(
    stage1_model: WBaselineStage1,
    diffusion_model,
    diffusion_cfg,
    stats_file: str,
    magnitude: torch.Tensor,
    location: torch.Tensor,
    station_idx: torch.Tensor,
    device: torch.device,
):
    stats = torch.load(stats_file, map_location=device)
    z_mean = stats["z_mean"].to(device)
    z_std = stats["z_std"].to(device)

    w = stage1_model.build_w(magnitude, location, station_idx)
    c_phys = stage1_model.build_raw_physical_condition(magnitude, location)
    cond = build_condition_tensor(diffusion_cfg["model"]["cond_mode"], w=w, c_phys=c_phys)

    z_norm = heun_sample_ve(
        model=diffusion_model,
        cond=cond,
        latent_dim=diffusion_cfg["model"]["latent_dim"],
        num_steps=diffusion_cfg["diffusion"]["sampler_steps"],
        t_min=diffusion_cfg["diffusion"]["t_min"],
        t_max=diffusion_cfg["diffusion"]["t_max"],
        device=device,
    )
    z = z_norm * (z_std.unsqueeze(0) + 1e-8) + z_mean.unsqueeze(0)
    return z, w


def load_eval_config(config_path: str):
    return load_config(config_path)

