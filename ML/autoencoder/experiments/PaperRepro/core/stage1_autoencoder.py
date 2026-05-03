from __future__ import annotations

import torch
from torch import nn

from .blocks import Decoder, Encoder


class Stage1Autoencoder(nn.Module):
    def __init__(
        self,
        *,
        encoder_config: dict,
        decoder_config: dict,
        kl_weight: float,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(**encoder_config)
        self.decoder = Decoder(**decoder_config)
        self.kl_weight = float(kl_weight)

    def encode_stats(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = torch.chunk(self.encoder(x), 2, dim=1)
        return mean, log_std

    def reparameterize(self, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
        return mean + torch.randn_like(mean) * torch.exp(log_std)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        mean, log_std = self.encode_stats(x)
        z = self.reparameterize(mean, log_std)
        reconstruction = self.decode(z)
        return {
            "reconstruction": reconstruction,
            "mean": mean,
            "log_std": log_std,
            "latent": z,
        }

    @staticmethod
    def kl_divergence(mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
        log_var = 2.0 * log_std
        return 0.5 * torch.sum(mean.square() + torch.exp(log_var) - log_var - 1.0, dim=1)

    def compute_losses(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.forward(x)
        reconstruction = outputs["reconstruction"]
        mean = outputs["mean"]
        log_std = outputs["log_std"]
        recon_loss = torch.mean((x - reconstruction) ** 2)
        kl_div = torch.mean(self.kl_divergence(mean, log_std))
        loss = recon_loss + self.kl_weight * kl_div
        return {
            "loss": loss,
            "reconstruction_loss": recon_loss,
            "kl_divergence": kl_div,
            **outputs,
        }


def build_stage1_autoencoder_from_config(cfg: dict) -> Stage1Autoencoder:
    stage1_cfg = cfg["stage1"]
    encdec = stage1_cfg["encoder_decoder"]
    latent_channels = int(stage1_cfg["latent_shape"][0])
    encoder_config = {
        "in_channels": int(cfg["representation"]["shape"][0]),
        "out_channels": latent_channels * 2,
        "model_channels": int(encdec["model_channels"]),
        "channel_mult": tuple(int(v) for v in encdec["channel_mult"]),
        "num_res_blocks": int(encdec["num_res_blocks"]),
        "dropout": float(encdec["dropout"]),
        "conv_kernel_size": int(encdec["conv_kernel_size"]),
        "dims": int(encdec["dims"]),
    }
    decoder_config = {
        "in_channels": latent_channels,
        "out_channels": int(cfg["representation"]["shape"][0]),
        "model_channels": int(encdec["model_channels"]),
        "channel_mult": tuple(int(v) for v in encdec["channel_mult"]),
        "num_res_blocks": int(encdec["num_res_blocks"]),
        "dropout": float(encdec["dropout"]),
        "conv_kernel_size": int(encdec["conv_kernel_size"]),
        "dims": int(encdec["dims"]),
    }
    return Stage1Autoencoder(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        kl_weight=float(stage1_cfg["objective"]["kl_weight"]),
    )
