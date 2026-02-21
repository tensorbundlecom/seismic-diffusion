from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


BACKBONE_CHANNELS: Dict[str, List[int]] = {
    "small": [16, 32, 64, 96, 128, 160],
    "base": [32, 64, 96, 128, 192, 256],
    "large": [32, 64, 128, 192, 256, 320],
}


def _make_encoder_1d(in_channels: int, channels: List[int]) -> nn.Sequential:
    layers: List[nn.Module] = []
    c_in = in_channels
    for c_out in channels:
        layers.extend(
            [
                nn.Conv1d(c_in, c_out, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(c_out),
                nn.SiLU(inplace=True),
            ]
        )
        c_in = c_out
    return nn.Sequential(*layers)


def _make_decoder_1d(out_channels: int, channels: List[int]) -> nn.Sequential:
    layers: List[nn.Module] = []
    # reverse from deepest channel to shallow
    rev = list(reversed(channels))
    c_in = rev[0]
    targets = rev[1:] + [out_channels]
    for i, c_out in enumerate(targets):
        is_last = i == len(targets) - 1
        layers.append(
            nn.ConvTranspose1d(
                c_in,
                c_out,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            )
        )
        if is_last:
            # waveform output (linear)
            pass
        else:
            layers.extend([nn.BatchNorm1d(c_out), nn.SiLU(inplace=True)])
        c_in = c_out
    return nn.Sequential(*layers)


class WaveformDiagVAE(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        input_length: int = 7001,
        latent_dim: int = 128,
        backbone: str = "base",
        logvar_mode: str = "legacy",
        logvar_min: float = -30.0,
        logvar_max: float = 20.0,
    ):
        super().__init__()
        if backbone not in BACKBONE_CHANNELS:
            raise ValueError(f"Unsupported backbone '{backbone}'. choices={list(BACKBONE_CHANNELS)}")
        if logvar_mode not in {"legacy", "bounded_sigmoid"}:
            raise ValueError("Unsupported logvar_mode. choices=['legacy', 'bounded_sigmoid']")
        if float(logvar_min) >= float(logvar_max):
            raise ValueError("logvar_min must be < logvar_max")
        self.in_channels = in_channels
        self.input_length = int(input_length)
        self.latent_dim = int(latent_dim)
        self.backbone = backbone
        self.channels = BACKBONE_CHANNELS[backbone]
        self.logvar_mode = str(logvar_mode)
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)

        self.encoder = _make_encoder_1d(in_channels, self.channels)
        self.decoder = _make_decoder_1d(in_channels, self.channels)

        # Infer flattened encoder feature shape with a dummy pass.
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, self.input_length)
            h = self.encoder(dummy)
            self.enc_c = int(h.shape[1])
            self.enc_l = int(h.shape[2])
            self.enc_flat = int(self.enc_c * self.enc_l)

        self.fc_mu = nn.Linear(self.enc_flat, self.latent_dim)
        self.fc_logvar = nn.Linear(self.enc_flat, self.latent_dim)
        self.fc_proj = nn.Linear(self.latent_dim, self.enc_flat)

    def _transform_logvar(self, raw_logvar: torch.Tensor) -> torch.Tensor:
        if self.logvar_mode == "legacy":
            return raw_logvar
        # Bounded parameterization to prevent rare, extreme posterior variance spikes.
        return self.logvar_min + (self.logvar_max - self.logvar_min) * torch.sigmoid(raw_logvar)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        h_flat = h.reshape(h.size(0), -1)
        mu = self.fc_mu(h_flat)
        logvar = self._transform_logvar(self.fc_logvar(h_flat))
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        logvar = torch.clamp(logvar, min=-30.0, max=20.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, target_length: int | None = None) -> torch.Tensor:
        h = self.fc_proj(z).reshape(z.size(0), self.enc_c, self.enc_l)
        x_hat = self.decoder(h)
        tlen = self.input_length if target_length is None else int(target_length)
        if x_hat.shape[-1] != tlen:
            x_hat = F.interpolate(x_hat, size=tlen, mode="linear", align_corners=False)
        return x_hat

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target_length = int(x.shape[-1])
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, target_length=target_length)
        return x_hat, mu, logvar


class WaveformAE(nn.Module):
    """
    Deterministic AE baseline with same backbone and bottleneck size.
    """

    def __init__(
        self,
        in_channels: int = 3,
        input_length: int = 7001,
        latent_dim: int = 128,
        backbone: str = "base",
    ):
        super().__init__()
        if backbone not in BACKBONE_CHANNELS:
            raise ValueError(f"Unsupported backbone '{backbone}'. choices={list(BACKBONE_CHANNELS)}")
        self.in_channels = in_channels
        self.input_length = int(input_length)
        self.latent_dim = int(latent_dim)
        self.backbone = backbone
        self.channels = BACKBONE_CHANNELS[backbone]

        self.encoder = _make_encoder_1d(in_channels, self.channels)
        self.decoder = _make_decoder_1d(in_channels, self.channels)

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, self.input_length)
            h = self.encoder(dummy)
            self.enc_c = int(h.shape[1])
            self.enc_l = int(h.shape[2])
            self.enc_flat = int(self.enc_c * self.enc_l)

        self.fc_latent = nn.Linear(self.enc_flat, self.latent_dim)
        self.fc_proj = nn.Linear(self.latent_dim, self.enc_flat)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h_flat = h.reshape(h.size(0), -1)
        z = self.fc_latent(h_flat)
        return z

    def decode(self, z: torch.Tensor, target_length: int | None = None) -> torch.Tensor:
        h = self.fc_proj(z).reshape(z.size(0), self.enc_c, self.enc_l)
        x_hat = self.decoder(h)
        tlen = self.input_length if target_length is None else int(target_length)
        if x_hat.shape[-1] != tlen:
            x_hat = F.interpolate(x_hat, size=tlen, mode="linear", align_corners=False)
        return x_hat

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        tlen = int(x.shape[-1])
        z = self.encode(x)
        x_hat = self.decode(z, target_length=tlen)
        return x_hat, z
