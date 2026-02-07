import torch
import torch.nn as nn
import numpy as np

def multivariate_kl_divergence(mu, L, beta=1.0):
    """
    Computes the KL divergence between a multivariate Gaussian Q(z|x) = N(mu, L L^T)
    and a standard normal prior P(z) = N(0, I).
    
    Formula: KL(Q||P) = 0.5 * (trace(sigma) + mu^T mu - k - log(det(sigma)))
    where sigma = L L^T and k is the latent dimension.
    
    Using Cholesky L:
    - trace(sigma) = sum of squares of elements of L
    - log(det(sigma)) = 2 * sum(log(diag(L)))
    
    Args:
        mu: Mean vector (batch_size, latent_dim)
        L: Lower triangular Cholesky factor (batch_size, latent_dim, latent_dim)
        beta: Weight for KL term
        
    Returns:
        KL divergence loss (scalar)
    """
    batch_size, latent_dim = mu.shape
    
    # 1. mu^T mu
    mu_sq = torch.sum(mu.pow(2))
    
    # 2. trace(sigma) = trace(L L^T) = sum(L_{ij}^2)
    trace_sigma = torch.sum(L.pow(2))
    
    # 3. log(det(sigma)) = 2 * sum(log(diag(L)))
    # We add a small epsilon for numerical stability
    diag_L = torch.diagonal(L, dim1=-2, dim2=-1)
    log_det_sigma = 2.0 * torch.sum(torch.log(diag_L + 1e-8))
    
    # KL Calculation
    kl = 0.5 * (trace_sigma + mu_sq - (batch_size * latent_dim) - log_det_sigma)
    
    return beta * kl

def full_cov_loss_function(reconstructed, original, mu, L, beta=1.0):
    """
    Combined loss for Full Covariance VAE.
    """
    recon_loss = nn.functional.mse_loss(reconstructed, original, reduction='sum')
    kl_loss = multivariate_kl_divergence(mu, L, beta=beta)
    
    total_loss = recon_loss + kl_loss
    
    return total_loss, recon_loss, kl_loss
