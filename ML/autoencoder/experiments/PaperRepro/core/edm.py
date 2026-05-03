from __future__ import annotations

import math

import torch
from torch import nn


class Stage2EDM(nn.Module):
    def __init__(
        self,
        denoiser: nn.Module,
        *,
        sigma_min: float,
        sigma_max: float,
        sigma_data: float,
        p_mean: float,
        p_std: float,
    ) -> None:
        super().__init__()
        self.denoiser = denoiser
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.sigma_data = float(sigma_data)
        self.p_mean = float(p_mean)
        self.p_std = float(p_std)

    def sample_training_sigma(self, batch_size: int, device: torch.device) -> torch.Tensor:
        rnd = torch.randn(batch_size, device=device)
        return torch.exp(rnd * self.p_std + self.p_mean).clamp(min=self.sigma_min, max=self.sigma_max)

    def precondition(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        cond: torch.Tensor,
        station_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sigma = sigma.reshape(-1, 1, 1, 1)
        sigma_data_sq = self.sigma_data ** 2
        c_in = 1.0 / torch.sqrt(sigma.square() + sigma_data_sq)
        c_skip = sigma_data_sq / (sigma.square() + sigma_data_sq)
        c_out = sigma * self.sigma_data / torch.sqrt(sigma.square() + sigma_data_sq)
        model_out = self.denoiser(c_in * x, sigma.flatten(), cond, station_index=station_index)
        return c_skip * x + c_out * model_out

    def loss_weight(self, sigma: torch.Tensor) -> torch.Tensor:
        sigma_sq = sigma.square()
        sigma_data_sq = self.sigma_data ** 2
        return (sigma_sq + sigma_data_sq) / ((sigma * self.sigma_data) ** 2)

    def compute_loss(
        self,
        clean_latent: torch.Tensor,
        cond: torch.Tensor,
        station_index: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        sigma = self.sample_training_sigma(clean_latent.shape[0], clean_latent.device)
        noise = torch.randn_like(clean_latent) * sigma.view(-1, 1, 1, 1)
        noisy = clean_latent + noise
        prediction = self.precondition(noisy, sigma, cond, station_index=station_index)
        weight = self.loss_weight(sigma).view(-1, 1, 1, 1)
        loss = torch.mean(weight * (prediction - clean_latent) ** 2)
        return {
            "loss": loss,
            "prediction": prediction,
            "sigma": sigma,
        }


def heun_sampler(
    model: Stage2EDM,
    *,
    cond: torch.Tensor,
    shape: tuple[int, ...],
    station_index: torch.Tensor | None = None,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    num_steps: int,
    device: torch.device,
) -> torch.Tensor:
    step_indices = torch.arange(num_steps, device=device, dtype=torch.float32)
    sigmas = (sigma_max ** (1 / rho) + step_indices / max(num_steps - 1, 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    sigmas = torch.cat([sigmas, torch.zeros(1, device=device)])
    x = torch.randn(shape, device=device) * sigmas[0]
    for i in range(num_steps):
        sigma = sigmas[i].repeat(shape[0])
        sigma_next = sigmas[i + 1].repeat(shape[0])
        denoised = model.precondition(x, sigma, cond, station_index=station_index)
        d = (x - denoised) / sigma.view(-1, 1, 1, 1)
        x_next = x + (sigma_next - sigma).view(-1, 1, 1, 1) * d
        if i < num_steps - 1:
            denoised_next = model.precondition(x_next, sigma_next, cond, station_index=station_index)
            d_next = (x_next - denoised_next) / sigma_next.view(-1, 1, 1, 1).clamp(min=1.0e-8)
            x = x + (sigma_next - sigma).view(-1, 1, 1, 1) * 0.5 * (d + d_next)
        else:
            x = x_next
    return x
