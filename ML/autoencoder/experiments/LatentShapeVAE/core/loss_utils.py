from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F


def time_mse_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # Compute in fp32 for numerical stability.
    return F.mse_loss(x_hat.float(), x.float(), reduction="mean")


def _stft_logmag(x: torch.Tensor, n_fft: int, hop_length: int, win_length: int, eps: float = 1e-7) -> torch.Tensor:
    # x: [B, C, T] -> flatten channels
    b, c, t = x.shape
    # Force fp32 STFT even under AMP; fp16 STFT can produce non-finite values.
    x_flat = x.reshape(b * c, t).float()
    window = torch.hann_window(win_length, device=x.device, dtype=torch.float32)
    stft = torch.stft(
        x_flat,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        return_complex=True,
    )
    mag = torch.abs(stft)
    mag = torch.nan_to_num(mag, nan=0.0, posinf=1e12, neginf=0.0)
    mag = torch.clamp(mag, min=eps, max=1e12)
    return torch.log(mag)


def mrstft_logmag_loss(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    n_ffts: List[int],
    hop_lengths: List[int],
    win_lengths: List[int],
    eps: float = 1e-7,
) -> torch.Tensor:
    if not (len(n_ffts) == len(hop_lengths) == len(win_lengths)):
        raise ValueError("n_ffts, hop_lengths, win_lengths must have same length")
    losses = []
    for n_fft, hop, win in zip(n_ffts, hop_lengths, win_lengths):
        y_hat = _stft_logmag(x_hat, n_fft=n_fft, hop_length=hop, win_length=win, eps=eps)
        y = _stft_logmag(x, n_fft=n_fft, hop_length=hop, win_length=win, eps=eps)
        losses.append(F.l1_loss(y_hat, y, reduction="mean"))
    return torch.stack(losses).mean()


def kl_raw_diag_gaussian(mu: torch.Tensor, logvar: torch.Tensor, free_bits: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      kl_raw: scalar KL (batch-avg)
      kl_per_dim: [D] batch-avg KL contribution per latent dim
    """
    # Clamp log-variance to keep KL numerically stable on outlier batches.
    logvar = torch.clamp(logvar, min=-30.0, max=20.0)
    k = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
    kl_per_dim = k.mean(dim=0)  # [D]
    if free_bits > 0:
        kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
    kl_raw = kl_per_dim.sum()
    return kl_raw, kl_per_dim


def vae_composite_loss(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
    lambda_mr: float,
    mr_n_ffts: List[int],
    mr_hop_lengths: List[int],
    mr_win_lengths: List[int],
    mr_eps: float = 1e-7,
    free_bits: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
    l_time = time_mse_loss(x_hat, x)
    l_mr = mrstft_logmag_loss(
        x_hat,
        x,
        n_ffts=mr_n_ffts,
        hop_lengths=mr_hop_lengths,
        win_lengths=mr_win_lengths,
        eps=mr_eps,
    )
    recon = l_time + (lambda_mr * l_mr)
    kl_raw, kl_per_dim = kl_raw_diag_gaussian(mu, logvar, free_bits=free_bits)
    beta_kl = beta * kl_raw
    total = recon + beta_kl
    terms = {
        "time_mse": float(l_time.item()),
        "mrstft": float(l_mr.item()),
        "recon_total": float(recon.item()),
        "kl_raw": float(kl_raw.item()),
        "beta_kl": float(beta_kl.item()),
        "loss_total": float(total.item()),
    }
    return total, terms, kl_per_dim


def ae_composite_loss(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    lambda_mr: float,
    mr_n_ffts: List[int],
    mr_hop_lengths: List[int],
    mr_win_lengths: List[int],
    mr_eps: float = 1e-7,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    l_time = time_mse_loss(x_hat, x)
    l_mr = mrstft_logmag_loss(
        x_hat,
        x,
        n_ffts=mr_n_ffts,
        hop_lengths=mr_hop_lengths,
        win_lengths=mr_win_lengths,
        eps=mr_eps,
    )
    recon = l_time + (lambda_mr * l_mr)
    terms = {
        "time_mse": float(l_time.item()),
        "mrstft": float(l_mr.item()),
        "recon_total": float(recon.item()),
        "kl_raw": 0.0,
        "beta_kl": 0.0,
        "loss_total": float(recon.item()),
    }
    return recon, terms


def active_units_from_mu(mu_all: torch.Tensor, threshold: float = 1e-2) -> Tuple[int, torch.Tensor]:
    """
    mu_all: [N, D]
    """
    var = mu_all.var(dim=0, unbiased=True)
    au = int((var > threshold).sum().item())
    return au, var
