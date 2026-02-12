import torch


def wflow_cvae_loss_function(reconstructed, original, mu, logvar, zk, log_det, beta=1.0):
    """Flow-CVAE loss used by W-space conditioned Normalizing Flow."""
    recon_loss = torch.nn.functional.mse_loss(reconstructed, original, reduction='sum')

    kl_gaussian = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Clamp only for numeric stability in long trainings.
    safe_log_det = torch.clamp(log_det, min=-100.0, max=100.0)
    kl_flow = kl_gaussian - safe_log_det.sum()

    total_loss = recon_loss + beta * kl_flow
    return total_loss, recon_loss, kl_flow
