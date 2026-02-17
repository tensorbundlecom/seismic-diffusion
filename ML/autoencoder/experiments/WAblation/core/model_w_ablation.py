import torch
import torch.nn as nn


class MappingNetwork(nn.Module):
    """Maps raw condition vector into W-space."""

    def __init__(self, in_dim, w_dim=64, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, w_dim),
        )

    def forward(self, x):
        return self.net(x)


class ConditionedDecoder(nn.Module):
    """Decoder that receives latent z plus condition vector."""

    def __init__(self, out_channels=3, latent_dim=128, cond_dim=64):
        super().__init__()
        self.spatial_dim = 16
        self.proj_dim = 256 * self.spatial_dim * self.spatial_dim

        self.fc_projection = nn.Linear(latent_dim + cond_dim, self.proj_dim)
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

    def forward(self, z, cond):
        b = z.size(0)
        h = self.fc_projection(torch.cat([z, cond], dim=1))
        h = h.view(b, 256, self.spatial_dim, self.spatial_dim)
        return self.decoder(h)


class WAblationCVAE(nn.Module):
    """
    Configurable CVAE for W ablation.
    - use_mapping_network=False: raw condition used directly.
    - use_station_embedding=False: station index is ignored.
    """

    def __init__(
        self,
        in_channels=3,
        latent_dim=128,
        num_stations=100,
        station_emb_dim=16,
        use_station_embedding=True,
        use_mapping_network=True,
        w_dim=64,
        map_hidden_dim=128,
        mag_min=0.0,
        mag_max=9.0,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.use_station_embedding = bool(use_station_embedding)
        self.use_mapping_network = bool(use_mapping_network)
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

        # raw condition = magnitude(1) + location(3) + optional station_emb
        self.raw_cond_dim = 4 + (int(station_emb_dim) if self.use_station_embedding else 0)

        if self.use_station_embedding:
            self.station_embedding = nn.Embedding(num_stations, station_emb_dim)
        else:
            self.station_embedding = None

        if self.use_mapping_network:
            self.mapping_net = MappingNetwork(in_dim=self.raw_cond_dim, w_dim=w_dim, hidden_dim=map_hidden_dim)
            self.cond_dim = int(w_dim)
        else:
            self.mapping_net = None
            self.cond_dim = self.raw_cond_dim

        self.fc_mu = None
        self.fc_logvar = None
        self.decoder = ConditionedDecoder(out_channels=in_channels, latent_dim=self.latent_dim, cond_dim=self.cond_dim)

        self._dummy_init()

    def _normalize_magnitude(self, magnitude):
        denom = max(self.mag_max - self.mag_min, 1e-6)
        mag_norm = (magnitude - self.mag_min) / denom
        return torch.clamp(mag_norm, 0.0, 1.0)

    def _build_condition(self, magnitude, location, station_idx):
        mag_norm = self._normalize_magnitude(magnitude).unsqueeze(1)
        loc_norm = torch.clamp(location, 0.0, 1.0)

        parts = [mag_norm, loc_norm]
        if self.use_station_embedding:
            station_emb = self.station_embedding(station_idx)
            parts.append(station_emb)

        raw_cond = torch.cat(parts, dim=1)
        if self.mapping_net is None:
            return raw_cond
        return self.mapping_net(raw_cond)

    def _init_fc_layers(self, flat_dim, device):
        self.fc_mu = nn.Linear(flat_dim + self.cond_dim, self.latent_dim).to(device)
        self.fc_logvar = nn.Linear(flat_dim + self.cond_dim, self.latent_dim).to(device)

    def _dummy_init(self):
        with torch.no_grad():
            x = torch.randn(1, 3, 129, 111)
            m = torch.zeros(1)
            l = torch.zeros(1, 3)
            s = torch.zeros(1, dtype=torch.long)
            self.forward(x, m, l, s)

    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, magnitude, location, station_idx):
        b = x.size(0)
        original_size = x.shape[2:]
        cond = self._build_condition(magnitude, location, station_idx)

        h = self.encoder(x)
        h_flat = h.view(b, -1)

        if self.fc_mu is None:
            self._init_fc_layers(h_flat.size(1), x.device)

        posterior_in = torch.cat([h_flat, cond], dim=1)
        mu = self.fc_mu(posterior_in)
        logvar = self.fc_logvar(posterior_in)

        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z, cond)

        if recon.shape[2:] != original_size:
            recon = torch.nn.functional.interpolate(recon, size=original_size, mode="bilinear", align_corners=False)

        return recon, mu, logvar

