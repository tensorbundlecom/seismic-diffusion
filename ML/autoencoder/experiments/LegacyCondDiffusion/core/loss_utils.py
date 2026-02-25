import torch
import torch.nn.functional as F


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()


def cvae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 0.1,
):
    recon_loss = F.mse_loss(recon, target, reduction="mean")
    kl = kl_divergence(mu, logvar)
    total = recon_loss + beta * kl
    return total, recon_loss, kl

