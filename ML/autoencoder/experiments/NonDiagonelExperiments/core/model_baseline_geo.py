import torch
import torch.nn as nn


class GeoCVAEEncoder(nn.Module):
    """
    Conditional encoder with numeric geometry features + station embedding.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 128,
        num_stations: int = 100,
        condition_dim: int = 64,
        numeric_condition_dim: int = 5,
    ):
        super().__init__()
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

        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.numeric_condition_dim = numeric_condition_dim

        self.station_embedding = nn.Embedding(num_stations, condition_dim // 4)
        condition_input_dim = numeric_condition_dim + (condition_dim // 4)
        self.condition_network = nn.Sequential(
            nn.Linear(condition_input_dim, condition_dim),
            nn.ReLU(inplace=True),
            nn.Linear(condition_dim, condition_dim),
            nn.ReLU(inplace=True),
        )

        self.fc_mu = None
        self.fc_logvar = None

    def _initialize_fc_layers(self, flattened_dim: int, device: torch.device) -> None:
        combined_dim = flattened_dim + self.condition_dim
        self.fc_mu = nn.Linear(combined_dim, self.latent_dim).to(device)
        self.fc_logvar = nn.Linear(combined_dim, self.latent_dim).to(device)

    def forward(self, x: torch.Tensor, cond_numeric: torch.Tensor, station_idx: torch.Tensor):
        batch_size = x.size(0)
        h = self.encoder(x)
        h_flat = h.view(batch_size, -1)

        station_emb = self.station_embedding(station_idx)
        cond_input = torch.cat([cond_numeric, station_emb], dim=1)
        cond_features = self.condition_network(cond_input)

        if self.fc_mu is None:
            self._initialize_fc_layers(h_flat.size(1), x.device)

        combined = torch.cat([h_flat, cond_features], dim=1)
        mu = self.fc_mu(combined)
        logvar = self.fc_logvar(combined)
        return mu, logvar


class GeoCVAEDecoder(nn.Module):
    def __init__(
        self,
        out_channels: int = 3,
        latent_dim: int = 128,
        num_stations: int = 100,
        condition_dim: int = 64,
        numeric_condition_dim: int = 5,
    ):
        super().__init__()
        self.condition_dim = condition_dim
        self.numeric_condition_dim = numeric_condition_dim

        self.station_embedding = nn.Embedding(num_stations, condition_dim // 4)
        condition_input_dim = numeric_condition_dim + (condition_dim // 4)
        self.condition_network = nn.Sequential(
            nn.Linear(condition_input_dim, condition_dim),
            nn.ReLU(inplace=True),
            nn.Linear(condition_dim, condition_dim),
            nn.ReLU(inplace=True),
        )

        self.spatial_dim = 16
        self.projection_dim = 256 * self.spatial_dim * self.spatial_dim
        self.fc_projection = nn.Linear(latent_dim + condition_dim, self.projection_dim)

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

    def forward(self, z: torch.Tensor, cond_numeric: torch.Tensor, station_idx: torch.Tensor):
        batch_size = z.size(0)
        station_emb = self.station_embedding(station_idx)
        cond_input = torch.cat([cond_numeric, station_emb], dim=1)
        cond_features = self.condition_network(cond_input)

        combined = torch.cat([z, cond_features], dim=1)
        h = self.fc_projection(combined)
        h = h.view(batch_size, 256, self.spatial_dim, self.spatial_dim)
        return self.decoder(h)


class GeoConditionalVariationalAutoencoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 128,
        num_stations: int = 100,
        condition_dim: int = 64,
        numeric_condition_dim: int = 5,
    ):
        super().__init__()
        self.encoder = GeoCVAEEncoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
            num_stations=num_stations,
            condition_dim=condition_dim,
            numeric_condition_dim=numeric_condition_dim,
        )
        self.decoder = GeoCVAEDecoder(
            out_channels=in_channels,
            latent_dim=latent_dim,
            num_stations=num_stations,
            condition_dim=condition_dim,
            numeric_condition_dim=numeric_condition_dim,
        )
        self.latent_dim = latent_dim
        self._dummy_init()

    def _dummy_init(self):
        with torch.no_grad():
            x = torch.randn(1, 3, 129, 111)
            c = torch.zeros(1, 5)
            s = torch.zeros(1, dtype=torch.long)
            self.forward(x, c, s)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor, cond_numeric: torch.Tensor, station_idx: torch.Tensor):
        original_size = x.shape[2:]
        mu, logvar = self.encoder(x, cond_numeric, station_idx)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z, cond_numeric, station_idx)
        if recon.shape[2:] != original_size:
            recon = torch.nn.functional.interpolate(
                recon,
                size=original_size,
                mode="bilinear",
                align_corners=False,
            )
        return recon, mu, logvar

    def decode(self, z: torch.Tensor, cond_numeric: torch.Tensor, station_idx: torch.Tensor):
        return self.decoder(z, cond_numeric, station_idx)

    def sample(self, num_samples: int, cond_numeric: torch.Tensor, station_idx: torch.Tensor, device: str = "cuda"):
        z = torch.randn(num_samples, self.latent_dim).to(device)
        return self.decode(z, cond_numeric, station_idx)

