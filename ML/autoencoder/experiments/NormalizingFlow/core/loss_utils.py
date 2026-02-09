import torch

def flow_cvae_loss_function(reconstructed, original, mu, logvar, zk, log_det, beta=1.0):
    """
    Compute the Flow-CVAE loss.
    
    The KL divergence with Normalizing Flows is:
    KL = E_q0 [ log q0(z0|x) - log |det J| - log p(zK) ]
    
    where:
    - log q0(z0|x) is the log-density of the initial Gaussian posterior.
    - log |det J| is the log-determinant of the Jacobian of the flow.
    - log p(zK) is the log-density of the prior (Standard Normal).
    """
    # 1. Reconstruction loss (MSE)
    recon_loss = torch.nn.functional.mse_loss(reconstructed, original, reduction='sum')
    
    # 2. KL Divergence for Flow
    # log q0(z0|x) = -0.5 * (log(2*pi) + logvar + (z0-mu)^2 / exp(logvar))
    # We can simplify this if we use the expectation form.
    # Standard VAE KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    
    # For Flow-VAE, we use zk:
    # log p(zK) = -0.5 * (latent_dim * log(2*pi) + sum(zk^2))
    # log qK(zK|x) = log q0(z0|x) - log_det
    
    # A common way is to compute log qK and log p explicitly:
    batch_size = zk.size(0)
    latent_dim = zk.size(1)
    
    # Sample z0 used to get zk (we need it for log q0)
    # Since we don't have z0 here, we can use the identity:
    # KL = KL_gaussian - log_det
    
    # Standard Gaussian KL
    kl_gaussian = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Adjust for flow: higher log_det reduces the KL penalty (as it spreads the density)
    # Note: log_det is log|det(dzK/dz0)|. If the flow expands, log_det > 0.
    kl_flow = kl_gaussian - log_det.sum()
    
    # Total loss
    total_loss = recon_loss + beta * kl_flow
    
    return total_loss, recon_loss, kl_flow
