import torch


def legacy_cond_baseline_cvae_loss(reconstructed, original, mu, logvar, beta=1.0):
    """Beta-CVAE objective used by the localized LegacyCondBaseline copy."""
    recon_loss = torch.nn.functional.mse_loss(reconstructed, original, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss


# Checkpoint/runtime compatibility with older naming.
wbaseline_cvae_loss_function = legacy_cond_baseline_cvae_loss
