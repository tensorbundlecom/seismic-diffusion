import math
from typing import Callable

import torch
import torch.nn.functional as F


def sinusoidal_timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """
    t: [B] in continuous range.
    Returns [B, dim].
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, device=t.device, dtype=torch.float32) / max(half - 1, 1)
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb


def sample_ve_noisy_latent(z0: torch.Tensor, t: torch.Tensor):
    eps = torch.randn_like(z0)
    z_t = z0 + t.unsqueeze(1) * eps
    return z_t, eps


def weighted_denoise_loss(pred_z0: torch.Tensor, target_z0: torch.Tensor, t: torch.Tensor, weight_mode: str = "none"):
    if weight_mode == "inv_t2":
        w = 1.0 / (t.unsqueeze(1).pow(2) + 1e-6)
        return (w * (pred_z0 - target_z0).pow(2)).mean()
    if weight_mode == "inv_t":
        w = 1.0 / (t.unsqueeze(1) + 1e-6)
        return (w * (pred_z0 - target_z0).pow(2)).mean()
    return F.mse_loss(pred_z0, target_z0, reduction="mean")


def build_condition_tensor(cond_mode: str, w: torch.Tensor, c_phys: torch.Tensor) -> torch.Tensor:
    if cond_mode == "w_only":
        return w
    if cond_mode == "c_only":
        return c_phys
    if cond_mode == "w_plus_c":
        return torch.cat([w, c_phys], dim=1)
    raise ValueError(f"Unsupported cond_mode: {cond_mode}")


@torch.no_grad()
def heun_sample_ve(
    model,
    cond: torch.Tensor,
    latent_dim: int,
    num_steps: int,
    t_min: float,
    t_max: float,
    device: torch.device,
):
    """
    Deterministic reverse ODE sampler for VE setting.
    Model is trained to predict clean z0 from zt.
    ODE drift: dz/dt = (z - D(z,t,c)) / t
    """
    z = torch.randn(cond.size(0), latent_dim, device=device) * t_max
    ts = torch.linspace(t_max, t_min, steps=num_steps, device=device)

    def drift(z_in: torch.Tensor, t_scalar: torch.Tensor) -> torch.Tensor:
        t_batch = torch.full((z_in.size(0),), float(t_scalar.item()), device=device)
        d = model(z_in, t_batch, cond)
        return (z_in - d) / (t_batch.unsqueeze(1) + 1e-6)

    for i in range(num_steps - 1):
        t_cur = ts[i]
        t_next = ts[i + 1]
        dt = t_next - t_cur

        k1 = drift(z, t_cur)
        z_euler = z + dt * k1
        k2 = drift(z_euler, t_next)
        z = z + dt * 0.5 * (k1 + k2)

    return z


def choose_denoiser(
    denoiser_name: str,
    latent_dim: int,
    cond_dim: int,
    hidden_dim: int = 512,
    depth: int = 6,
    dropout: float = 0.0,
    base_channels: int = 64,
):
    if denoiser_name == "resmlp":
        from .model_diffusion_resmlp import ResMLPDenoiser

        return ResMLPDenoiser(
            latent_dim=latent_dim,
            cond_dim=cond_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
        )
    if denoiser_name == "unet1d":
        from .model_diffusion_unet1d import UNet1DDenoiser

        return UNet1DDenoiser(
            latent_dim=latent_dim,
            cond_dim=cond_dim,
            base_channels=base_channels,
        )
    raise ValueError(f"Unsupported denoiser_name: {denoiser_name}")

