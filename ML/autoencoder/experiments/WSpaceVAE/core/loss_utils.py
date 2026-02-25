import torch
import torch.nn.functional as F


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()


def vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
):
    rec = F.mse_loss(recon, target, reduction="mean")
    kl = kl_divergence(mu, logvar)
    return rec + beta * kl, rec, kl

