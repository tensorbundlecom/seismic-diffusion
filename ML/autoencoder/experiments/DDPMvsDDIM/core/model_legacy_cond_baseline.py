import torch
import torch.nn as nn


class MappingNetwork(nn.Module):
    """Maps normalized physical parameters and station embedding into a condition embedding."""

    def __init__(self, cond_input_dim, embedding_dim=64, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, condition):
        return self.net(condition)


class ConditionedDecoder(nn.Module):
    """Decoder conditioned on latent z and deterministic condition embedding."""

    def __init__(self, out_channels=3, latent_dim=128, cond_embedding_dim=64):
        super().__init__()
        self.spatial_dim = 16
        self.projection_dim = 256 * self.spatial_dim * self.spatial_dim

        self.fc_projection = nn.Linear(latent_dim + cond_embedding_dim, self.projection_dim)
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

    def forward(self, latent, cond_embedding):
        batch_size = latent.size(0)
        hidden = self.fc_projection(torch.cat([latent, cond_embedding], dim=1))
        hidden = hidden.view(batch_size, 256, self.spatial_dim, self.spatial_dim)
        return self.decoder(hidden)


class LegacyCondBaselineCVAE(nn.Module):
    """
    Localized copy of the LegacyCondBaseline CVAE.

    This is the Stage-0 backbone for the DDPM vs DDIM box before any diffusion code is added.
    """

    def __init__(
        self,
        in_channels=3,
        latent_dim=128,
        num_stations=100,
        cond_embedding_dim=64,
        w_dim=None,
        station_emb_dim=16,
        map_hidden_dim=128,
        mag_min=0.0,
        mag_max=9.0,
    ):
        super().__init__()
        if w_dim is not None:
            cond_embedding_dim = w_dim
        self.latent_dim = latent_dim
        self.cond_embedding_dim = cond_embedding_dim
        self.w_dim = cond_embedding_dim
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

        self.station_embedding = nn.Embedding(num_stations, station_emb_dim)
        cond_input_dim = 1 + 3 + station_emb_dim
        self.mapping_net = MappingNetwork(
            cond_input_dim=cond_input_dim,
            embedding_dim=cond_embedding_dim,
            hidden_dim=map_hidden_dim,
        )

        self.fc_mu = None
        self.fc_logvar = None
        self.decoder = ConditionedDecoder(
            out_channels=in_channels,
            latent_dim=latent_dim,
            cond_embedding_dim=cond_embedding_dim,
        )
        self._dummy_init()

    def _normalize_magnitude(self, magnitude):
        denom = max(self.mag_max - self.mag_min, 1e-6)
        magnitude_norm = (magnitude - self.mag_min) / denom
        return torch.clamp(magnitude_norm, 0.0, 1.0)

    def build_condition_embedding(self, magnitude, location, station_idx):
        magnitude_norm = self._normalize_magnitude(magnitude).unsqueeze(1)
        location_norm = torch.clamp(location, 0.0, 1.0)
        station_emb = self.station_embedding(station_idx)
        condition = torch.cat([magnitude_norm, location_norm, station_emb], dim=1)
        return self.mapping_net(condition)

    def _initialize_fc_layers(self, flattened_dim, device):
        self.fc_mu = nn.Linear(flattened_dim + self.cond_embedding_dim, self.latent_dim).to(device)
        self.fc_logvar = nn.Linear(flattened_dim + self.cond_embedding_dim, self.latent_dim).to(device)

    def _dummy_init(self):
        with torch.no_grad():
            sample_x = torch.randn(1, 3, 129, 111)
            sample_m = torch.zeros(1)
            sample_l = torch.zeros(1, 3)
            sample_s = torch.zeros(1, dtype=torch.long)
            self.forward(sample_x, sample_m, sample_l, sample_s)

    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x, magnitude, location, station_idx):
        batch_size = x.size(0)
        cond_embedding = self.build_condition_embedding(magnitude, location, station_idx)
        hidden = self.encoder(x)
        hidden_flat = hidden.view(batch_size, -1)

        if self.fc_mu is None:
            self._initialize_fc_layers(hidden_flat.size(1), x.device)

        posterior_input = torch.cat([hidden_flat, cond_embedding], dim=1)
        mu = self.fc_mu(posterior_input)
        logvar = self.fc_logvar(posterior_input)
        return mu, logvar, cond_embedding

    def decode(self, latent, cond_embedding, original_size=None):
        reconstructed = self.decoder(latent, cond_embedding)
        if original_size is not None and reconstructed.shape[2:] != original_size:
            reconstructed = torch.nn.functional.interpolate(
                reconstructed,
                size=original_size,
                mode="bilinear",
                align_corners=False,
            )
        return reconstructed

    def forward(self, x, magnitude, location, station_idx):
        original_size = x.shape[2:]
        mu, logvar, cond_embedding = self.encode(x, magnitude, location, station_idx)
        latent = self.reparameterize(mu, logvar)
        reconstructed = self.decode(latent, cond_embedding, original_size=original_size)
        return reconstructed, mu, logvar

    def sample(self, num_samples, magnitude, location, station_idx, device=None, output_size=(129, 111)):
        if device is None:
            device = magnitude.device
        latent = torch.randn(num_samples, self.latent_dim, device=device)
        cond_embedding = self.build_condition_embedding(magnitude, location, station_idx)
        return self.decode(latent, cond_embedding, original_size=output_size)


# Checkpoint/runtime compatibility with older naming.
WBaselineCVAE = LegacyCondBaselineCVAE
