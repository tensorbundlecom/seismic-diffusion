import torch
import torch.nn as nn


class MappingNetwork(nn.Module):
    """Maps physical condition + station embedding to legacy condition embedding."""

    def __init__(self, cond_input_dim: int, w_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, w_dim),
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return self.net(cond)


class WConditionedDecoder(nn.Module):
    """Decoder conditioned on legacy `w_cond` vector only."""

    def __init__(self, out_channels: int = 3, latent_dim: int = 128, w_dim: int = 64):
        super().__init__()
        self.spatial_dim = 16
        self.projection_dim = 256 * self.spatial_dim * self.spatial_dim
        self.fc_projection = nn.Linear(latent_dim + w_dim, self.projection_dim)

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
            nn.ConvTranspose2d(32, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        batch_size = z.size(0)
        h = self.fc_projection(torch.cat([z, w], dim=1))
        h = h.view(batch_size, 256, self.spatial_dim, self.spatial_dim)
        return self.decoder(h)


class WBaselineStage1(nn.Module):
    """Legacy condition-embedding CVAE backbone used for Stage-1 training."""

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 128,
        num_stations: int = 100,
        w_dim: int = 64,
        station_emb_dim: int = 16,
        map_hidden_dim: int = 128,
        mag_min: float = 0.0,
        mag_max: float = 9.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.num_stations = num_stations
        self.w_dim = w_dim
        self.station_emb_dim = station_emb_dim
        self.mag_min = float(mag_min)
        self.mag_max = float(mag_max)

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
        )
        self.encoder_pool = nn.AdaptiveAvgPool2d((8, 8))

        self.station_embedding = nn.Embedding(num_stations, station_emb_dim)
        self.mapping_net = MappingNetwork(
            cond_input_dim=1 + 3 + station_emb_dim,
            w_dim=w_dim,
            hidden_dim=map_hidden_dim,
        )

        flattened_dim = 256 * 8 * 8
        self.fc_mu = nn.Linear(flattened_dim + self.w_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(flattened_dim + self.w_dim, self.latent_dim)

        self.decoder = WConditionedDecoder(out_channels=in_channels, latent_dim=latent_dim, w_dim=w_dim)
        self._dummy_init()

    def _dummy_init(self) -> None:
        with torch.no_grad():
            x = torch.zeros(1, self.in_channels, 129, 111)
            m = torch.zeros(1)
            l = torch.zeros(1, 3)
            s = torch.zeros(1, dtype=torch.long)
            self.forward(x, m, l, s)

    def _normalize_magnitude(self, magnitude: torch.Tensor) -> torch.Tensor:
        denom = max(self.mag_max - self.mag_min, 1e-6)
        mag_norm = (magnitude - self.mag_min) / denom
        return torch.clamp(mag_norm, 0.0, 1.0)

    def build_raw_physical_condition(self, magnitude: torch.Tensor, location: torch.Tensor) -> torch.Tensor:
        """
        Returns physical condition vector used in diffusion hybrid conditioning:
        [mag_norm, lat_norm, lon_norm, depth_norm].
        """
        mag_norm = self._normalize_magnitude(magnitude).unsqueeze(1)
        loc_norm = torch.clamp(location, 0.0, 1.0)
        return torch.cat([mag_norm, loc_norm], dim=1)

    def build_w(self, magnitude: torch.Tensor, location: torch.Tensor, station_idx: torch.Tensor) -> torch.Tensor:
        mag_norm = self._normalize_magnitude(magnitude).unsqueeze(1)
        loc_norm = torch.clamp(location, 0.0, 1.0)
        station_emb = self.station_embedding(station_idx)
        cond = torch.cat([mag_norm, loc_norm, station_emb], dim=1)
        return self.mapping_net(cond)

    def encode_distribution(
        self,
        x: torch.Tensor,
        magnitude: torch.Tensor,
        location: torch.Tensor,
        station_idx: torch.Tensor,
    ):
        batch_size = x.size(0)
        w = self.build_w(magnitude, location, station_idx)
        h = self.encoder_pool(self.encoder(x)).view(batch_size, -1)
        posterior_input = torch.cat([h, w], dim=1)
        mu = self.fc_mu(posterior_input)
        logvar = self.fc_logvar(posterior_input)
        return mu, logvar, w

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, magnitude: torch.Tensor, location: torch.Tensor, station_idx: torch.Tensor):
        w = self.build_w(magnitude, location, station_idx)
        return self.decoder(z, w)

    def forward(
        self,
        x: torch.Tensor,
        magnitude: torch.Tensor,
        location: torch.Tensor,
        station_idx: torch.Tensor,
    ):
        original_size = x.shape[2:]
        mu, logvar, w = self.encode_distribution(x, magnitude, location, station_idx)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z, w)
        if recon.shape[2:] != original_size:
            recon = torch.nn.functional.interpolate(
                recon, size=original_size, mode="bilinear", align_corners=False
            )
        return recon, mu, logvar
