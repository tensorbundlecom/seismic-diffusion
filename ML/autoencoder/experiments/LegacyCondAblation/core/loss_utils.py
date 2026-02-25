import torch


def ablation_cvae_loss(reconstructed, original, mu, logvar, beta=1.0):
    """Standard beta-CVAE loss used across all ablation variants."""
    recon_loss = torch.nn.functional.mse_loss(reconstructed, original, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon_loss + beta * kl_loss
    return total, recon_loss, kl_loss

