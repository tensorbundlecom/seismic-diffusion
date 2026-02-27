"""Model definitions for experiments2/exp001."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class ConditionEncoder(nn.Module):
    """Encodes numeric condition vector + station embedding into a compact condition state."""

    def __init__(
        self,
        numeric_dim: int,
        num_stations: int,
        station_embedding_dim: int,
        condition_hidden_dim: int,
    ) -> None:
        super().__init__()
        self.station_embedding = nn.Embedding(num_stations, station_embedding_dim)
        in_dim = numeric_dim + station_embedding_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, condition_hidden_dim),
            nn.LayerNorm(condition_hidden_dim),
            nn.SiLU(),
            nn.Linear(condition_hidden_dim, condition_hidden_dim),
            nn.SiLU(),
        )

    def forward(self, cond_numeric: torch.Tensor, station_idx: torch.Tensor) -> torch.Tensor:
        emb = self.station_embedding(station_idx)
        return self.net(torch.cat([cond_numeric, emb], dim=1))


class Encoder2D(nn.Module):
    """Convolutional encoder for 2x128x220 complex-STFT input."""

    def __init__(self, in_channels: int, channels: Sequence[int]) -> None:
        super().__init__()
        layers = []
        c_prev = in_channels
        for c_out in channels:
            layers.extend(
                [
                    nn.Conv2d(c_prev, c_out, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(c_out),
                    nn.SiLU(),
                ]
            )
            c_prev = c_out
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder2D(nn.Module):
    """Transposed-conv decoder from latent+condition state."""

    def __init__(self, out_channels: int, channels: Sequence[int]) -> None:
        super().__init__()
        # channels are expected like [256, 128, 64, 32]
        layers = []
        for i in range(len(channels) - 1):
            layers.extend(
                [
                    nn.ConvTranspose2d(
                        channels[i],
                        channels[i + 1],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(channels[i + 1]),
                    nn.SiLU(),
                ]
            )
        layers.append(
            nn.ConvTranspose2d(
                channels[-1],
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            )
        )
        self.net = nn.Sequential(*layers)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class CVAEComplexSTFT(nn.Module):
    """
    Conditional VAE for Z-only complex STFT.

    Input shape is fixed to ``[B, 2, 128, 220]``.
    """

    def __init__(
        self,
        numeric_cond_dim: int,
        num_stations: int,
        latent_dim: int = 128,
        station_embedding_dim: int = 16,
        condition_hidden_dim: int = 128,
        encoder_channels: Sequence[int] = (32, 64, 128, 256),
        decoder_channels: Sequence[int] = (256, 128, 64, 32),
        input_shape: tuple[int, int, int] = (2, 128, 220),
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.input_shape = input_shape

        self.condition_encoder = ConditionEncoder(
            numeric_dim=numeric_cond_dim,
            num_stations=num_stations,
            station_embedding_dim=station_embedding_dim,
            condition_hidden_dim=condition_hidden_dim,
        )
        self.encoder = Encoder2D(in_channels=input_shape[0], channels=encoder_channels)

        # Infer encoded spatial shape once.
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            enc = self.encoder(dummy)
            self.enc_shape = tuple(enc.shape[1:])
            enc_flat_dim = int(enc.numel())

        self.fc_mu = nn.Linear(enc_flat_dim + condition_hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(enc_flat_dim + condition_hidden_dim, latent_dim)

        dec_flat_dim = int(decoder_channels[0] * self.enc_shape[1] * self.enc_shape[2])
        self.fc_decode = nn.Linear(latent_dim + condition_hidden_dim, dec_flat_dim)
        self.decoder = Decoder2D(out_channels=input_shape[0], channels=decoder_channels)

    def encode_distribution(
        self,
        x: torch.Tensor,
        cond_numeric: torch.Tensor,
        station_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cond_state = self.condition_encoder(cond_numeric, station_idx)
        h = self.encoder(x).flatten(start_dim=1)
        h_joint = torch.cat([h, cond_state], dim=1)
        mu = self.fc_mu(h_joint)
        # Minimal numeric-stability guard for KL: keep posterior log-variance in a safe range.
        logvar = torch.clamp(self.fc_logvar(h_joint), min=-8.0, max=4.0)
        return mu, logvar, cond_state

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_from_latent(self, z: torch.Tensor, cond_state: torch.Tensor) -> torch.Tensor:
        h = self.fc_decode(torch.cat([z, cond_state], dim=1))
        h = h.view(z.size(0), -1, self.enc_shape[1], self.enc_shape[2])
        out = self.decoder(h)
        # Decoder produces width 224 with this architecture; crop to 220.
        out = out[:, :, : self.input_shape[1], : self.input_shape[2]]
        return out

    def forward(
        self,
        x: torch.Tensor,
        cond_numeric: torch.Tensor,
        station_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar, cond_state = self.encode_distribution(x, cond_numeric, station_idx)
        z = self.reparameterize(mu, logvar)
        recon = self.decode_from_latent(z, cond_state)
        return recon, mu, logvar

    def sample_condition_only(
        self,
        cond_numeric: torch.Tensor,
        station_idx: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        cond_state = self.condition_encoder(cond_numeric, station_idx)
        z = torch.randn(
            cond_numeric.size(0),
            self.latent_dim,
            device=cond_numeric.device,
            generator=generator,
        )
        return self.decode_from_latent(z, cond_state)
