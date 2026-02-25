import torch
import torch.nn as nn


def multivariate_kl_divergence(mu, L, beta=1.0):
    """KL[ N(mu, LL^T) || N(0, I) ] with Cholesky factor L."""
    batch_size, latent_dim = mu.shape
    mu_sq = torch.sum(mu.pow(2))
    trace_sigma = torch.sum(L.pow(2))
    diag_l = torch.diagonal(L, dim1=-2, dim2=-1)
    log_det_sigma = 2.0 * torch.sum(torch.log(diag_l + 1e-8))
    kl = 0.5 * (trace_sigma + mu_sq - (batch_size * latent_dim) - log_det_sigma)
    return beta * kl


def wfullcov_loss_function(reconstructed, original, mu, L, beta=1.0):
    recon_loss = nn.functional.mse_loss(reconstructed, original, reduction='sum')
    kl_loss = multivariate_kl_divergence(mu, L, beta=beta)
    total_loss = recon_loss + kl_loss
    return total_loss, recon_loss, kl_loss
