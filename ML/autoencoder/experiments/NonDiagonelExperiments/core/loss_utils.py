import torch
import torch.nn as nn


def beta_cvae_loss(reconstructed: torch.Tensor, original: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float = 0.1):
    recon_loss = nn.functional.mse_loss(reconstructed, original, reduction="sum") / original.size(0)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / original.size(0)
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def multivariate_kl_divergence(mu: torch.Tensor, L: torch.Tensor, beta: float = 0.1):
    batch_size, latent_dim = mu.shape
    mu_sq = torch.sum(mu.pow(2))
    trace_sigma = torch.sum(L.pow(2))
    diag_L = torch.diagonal(L, dim1=-2, dim2=-1)
    log_det_sigma = 2.0 * torch.sum(torch.log(diag_L + 1e-8))
    kl = 0.5 * (trace_sigma + mu_sq - (batch_size * latent_dim) - log_det_sigma)
    return beta * (kl / batch_size)


def full_cov_loss(reconstructed: torch.Tensor, original: torch.Tensor, mu: torch.Tensor, L: torch.Tensor, beta: float = 0.1):
    recon_loss = nn.functional.mse_loss(reconstructed, original, reduction="sum") / original.size(0)
    kl_loss = multivariate_kl_divergence(mu, L, beta=beta)
    return recon_loss + kl_loss, recon_loss, kl_loss


def offdiag_l2_penalty(L: torch.Tensor) -> torch.Tensor:
    """
    Mean squared magnitude of off-diagonal Cholesky entries.
    Returns a scalar tensor suitable for weighting with lambda.
    """
    b, d, _ = L.shape
    mask = 1.0 - torch.eye(d, device=L.device, dtype=L.dtype).unsqueeze(0)
    off = L * mask
    denom = max(d * (d - 1), 1)
    return torch.sum(off.pow(2), dim=(1, 2)).mean() / denom
