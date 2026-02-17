import torch
import torch.nn as nn

from ML.autoencoder.experiments.NonDiagonel.core.model_baseline_geo import GeoCVAEDecoder, GeoCVAEEncoder


class GeoFullCovEncoder(nn.Module):
    """
    Full-covariance encoder producing mean and Cholesky factor L.
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
        self.base = GeoCVAEEncoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
            num_stations=num_stations,
            condition_dim=condition_dim,
            numeric_condition_dim=numeric_condition_dim,
        )
        self.latent_dim = latent_dim
        self.num_chol = (latent_dim * (latent_dim + 1)) // 2
        self.fc_mu = None
        self.fc_chol = None

    def forward(self, x: torch.Tensor, cond_numeric: torch.Tensor, station_idx: torch.Tensor):
        b = x.size(0)
        h = self.base.encoder(x)
        h_flat = h.view(b, -1)

        station_emb = self.base.station_embedding(station_idx)
        cond_input = torch.cat([cond_numeric, station_emb], dim=1)
        cond_features = self.base.condition_network(cond_input)
        combined = torch.cat([h_flat, cond_features], dim=1)

        if self.fc_mu is None:
            d = combined.size(1)
            self.fc_mu = nn.Linear(d, self.latent_dim).to(x.device)
            self.fc_chol = nn.Linear(d, self.num_chol).to(x.device)

        mu = self.fc_mu(combined)
        chol_flat = self.fc_chol(combined)

        L = torch.zeros(b, self.latent_dim, self.latent_dim, device=x.device, dtype=mu.dtype)
        tril_idx = torch.tril_indices(row=self.latent_dim, col=self.latent_dim, offset=0, device=x.device)
        L[:, tril_idx[0], tril_idx[1]] = chol_flat

        diag_idx = torch.arange(self.latent_dim, device=x.device)
        diag_vals = torch.exp(L[:, diag_idx, diag_idx].float()).to(L.dtype)
        L[:, diag_idx, diag_idx] = diag_vals
        return mu, L


class GeoFullCovCVAE(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 128,
        num_stations: int = 100,
        condition_dim: int = 64,
        numeric_condition_dim: int = 5,
    ):
        super().__init__()
        self.encoder = GeoFullCovEncoder(
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

    def reparameterize(self, mu: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        b = mu.size(0)
        eps = torch.randn(b, self.latent_dim, 1, device=mu.device, dtype=mu.dtype)
        z = mu.unsqueeze(2) + torch.bmm(L, eps)
        return z.squeeze(2)

    def forward(self, x: torch.Tensor, cond_numeric: torch.Tensor, station_idx: torch.Tensor):
        original_size = x.shape[2:]
        mu, L = self.encoder(x, cond_numeric, station_idx)
        z = self.reparameterize(mu, L)
        recon = self.decoder(z, cond_numeric, station_idx)
        if recon.shape[2:] != original_size:
            recon = torch.nn.functional.interpolate(
                recon,
                size=original_size,
                mode="bilinear",
                align_corners=False,
            )
        return recon, mu, L
