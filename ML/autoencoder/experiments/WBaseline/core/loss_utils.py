import torch


def wbaseline_cvae_loss_function(reconstructed, original, mu, logvar, beta=1.0):
    """Base beta-CVAE loss for W-conditioned baseline model."""
    recon_loss = torch.nn.functional.mse_loss(reconstructed, original, reduction='sum')
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + beta * kl
    return total_loss, recon_loss, kl
