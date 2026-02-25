import torch
import torch.nn as nn


class WMapping(nn.Module):
    """
    StyleGAN-like mapping: stochastic base latent u -> w.
    """

    def __init__(self, u_dim: int, w_dim: int, hidden_dim: int = 256, depth: int = 4):
        super().__init__()
        layers = []
        in_dim = u_dim
        for _ in range(depth - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, w_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.net(u)


class TrueWSpaceCVAE(nn.Module):
    """
    Conditional VAE where the stochastic latent is u and decoder consumes w = M(u).
    This is a true W-latent design (not deterministic condition embedding).
    """

    def __init__(
        self,
        in_channels: int = 3,
        u_dim: int = 128,
        w_dim: int = 128,
        cond_dim: int = 64,
        num_stations: int = 100,
        station_emb_dim: int = 16,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.u_dim = u_dim
        self.w_dim = w_dim
        self.cond_dim = cond_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8)),
        )

        self.station_embedding = nn.Embedding(num_stations, station_emb_dim)
        self.cond_net = nn.Sequential(
            nn.Linear(1 + 3 + station_emb_dim, cond_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cond_dim, cond_dim),
            nn.ReLU(inplace=True),
        )

        flat_dim = 256 * 8 * 8
        self.fc_mu = nn.Linear(flat_dim + cond_dim, u_dim)
        self.fc_logvar = nn.Linear(flat_dim + cond_dim, u_dim)

        self.mapping = WMapping(u_dim=u_dim, w_dim=w_dim, hidden_dim=256, depth=4)

        self.spatial_dim = 16
        self.proj_dim = 256 * self.spatial_dim * self.spatial_dim
        self.fc_decode = nn.Linear(w_dim + cond_dim, self.proj_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def build_condition(self, magnitude: torch.Tensor, location: torch.Tensor, station_idx: torch.Tensor):
        mag = magnitude.unsqueeze(1)
        loc = torch.clamp(location, 0.0, 1.0)
        sta = self.station_embedding(station_idx)
        cond_in = torch.cat([mag, loc, sta], dim=1)
        return self.cond_net(cond_in)

    def encode_distribution(self, x, magnitude, location, station_idx):
        cond = self.build_condition(magnitude, location, station_idx)
        h = self.encoder(x).view(x.size(0), -1)
        h = torch.cat([h, cond], dim=1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar, cond

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_from_u(self, u: torch.Tensor, cond: torch.Tensor):
        w = self.mapping(u)
        z = torch.cat([w, cond], dim=1)
        h = self.fc_decode(z).view(u.size(0), 256, self.spatial_dim, self.spatial_dim)
        return self.decoder(h), w

    def forward(self, x, magnitude, location, station_idx):
        original_size = x.shape[2:]
        mu, logvar, cond = self.encode_distribution(x, magnitude, location, station_idx)
        u = self.reparameterize(mu, logvar)
        recon, w = self.decode_from_u(u, cond)
        if recon.shape[2:] != original_size:
            recon = torch.nn.functional.interpolate(
                recon, size=original_size, mode="bilinear", align_corners=False
            )
        return recon, mu, logvar, u, w

    @torch.no_grad()
    def sample(self, num_samples, magnitude, location, station_idx, device):
        cond = self.build_condition(magnitude, location, station_idx)
        u = torch.randn(num_samples, self.u_dim, device=device)
        recon, w = self.decode_from_u(u, cond)
        return recon, u, w

