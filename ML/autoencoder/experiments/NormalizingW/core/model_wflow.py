import os
import sys
import torch
import torch.nn as nn

# Add path to project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../')))


class MappingNetwork(nn.Module):
    """Maps normalized physical parameters + station embedding into W space."""

    def __init__(self, cond_input_dim, w_dim=64, hidden_dim=128):
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

    def forward(self, c):
        return self.net(c)


class ConditionalAffineCouplingLayerW(nn.Module):
    """RealNVP affine coupling conditioned on W-space vectors."""

    def __init__(self, dim, w_dim=64, hidden_dim=256, mask_type='even'):
        super().__init__()
        self.dim = dim
        self.d = dim // 2
        self.mask_type = mask_type

        input_dim = self.d + w_dim
        out_dim = dim - self.d

        self.st_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, out_dim * 2),
        )

    def _split(self, x):
        if self.mask_type == 'even':
            x1, x2 = x[:, :self.d], x[:, self.d:]
        else:
            x2, x1 = x[:, :self.d], x[:, self.d:]
        return x1, x2

    def _merge(self, y1, y2):
        if self.mask_type == 'even':
            return torch.cat([y1, y2], dim=1)
        return torch.cat([y2, y1], dim=1)

    def forward(self, x, w):
        x1, x2 = self._split(x)
        st = self.st_net(torch.cat([x1, w], dim=1))
        s, t = st.chunk(2, dim=1)

        # Bound scale for stable log-det during long runs.
        s = torch.tanh(s)
        y2 = x2 * torch.exp(s) + t

        y = self._merge(x1, y2)
        log_det = s.sum(dim=1)
        return y, log_det

    def inverse(self, y, w):
        y1, y2 = self._split(y)
        st = self.st_net(torch.cat([y1, w], dim=1))
        s, t = st.chunk(2, dim=1)
        s = torch.tanh(s)

        x2 = (y2 - t) * torch.exp(-s)
        return self._merge(y1, x2)


class NormalizingFlowW(nn.Module):
    """Stack of W-conditioned affine coupling layers."""

    def __init__(self, dim, w_dim=64, num_layers=8, hidden_dim=256):
        super().__init__()
        self.layers = nn.ModuleList([
            ConditionalAffineCouplingLayerW(
                dim=dim,
                w_dim=w_dim,
                hidden_dim=hidden_dim,
                mask_type='even' if i % 2 == 0 else 'odd',
            )
            for i in range(num_layers)
        ])

    def forward(self, z, w):
        log_det_total = torch.zeros(z.size(0), device=z.device)
        zk = z
        for layer in self.layers:
            zk, log_det = layer(zk, w)
            log_det_total += log_det
        return zk, log_det_total

    def inverse(self, z, w):
        x = z
        for layer in reversed(self.layers):
            x = layer.inverse(x, w)
        return x


class WConditionedDecoder(nn.Module):
    """Decoder conditioned only on W vectors (no direct raw condition path)."""

    def __init__(self, out_channels=3, latent_dim=128, w_dim=64):
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

    def forward(self, z, w):
        batch_size = z.size(0)
        h = self.fc_projection(torch.cat([z, w], dim=1))
        h = h.view(batch_size, 256, self.spatial_dim, self.spatial_dim)
        return self.decoder(h)


class WFlowCVAE(nn.Module):
    """CVAE + RealNVP where conditioning is mapped into W and used everywhere."""

    def __init__(
        self,
        in_channels=3,
        latent_dim=128,
        num_stations=100,
        flow_layers=8,
        w_dim=64,
        station_emb_dim=16,
        map_hidden_dim=128,
        mag_min=0.0,
        mag_max=9.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.w_dim = w_dim
        self.mag_min = float(mag_min)
        self.mag_max = float(mag_max)

        # Image encoder branch
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
        self.mapping_net = MappingNetwork(cond_input_dim=cond_input_dim, w_dim=w_dim, hidden_dim=map_hidden_dim)

        self.fc_mu = None
        self.fc_logvar = None

        self.flow = NormalizingFlowW(dim=latent_dim, w_dim=w_dim, num_layers=flow_layers)
        self.decoder = WConditionedDecoder(out_channels=in_channels, latent_dim=latent_dim, w_dim=w_dim)

        self._dummy_init()

    def _normalize_magnitude(self, magnitude):
        denom = max(self.mag_max - self.mag_min, 1e-6)
        mag_norm = (magnitude - self.mag_min) / denom
        return torch.clamp(mag_norm, 0.0, 1.0)

    def _build_w(self, magnitude, location, station_idx):
        mag_norm = self._normalize_magnitude(magnitude).unsqueeze(1)
        loc_norm = torch.clamp(location, 0.0, 1.0)
        station_emb = self.station_embedding(station_idx)
        cond = torch.cat([mag_norm, loc_norm, station_emb], dim=1)
        return self.mapping_net(cond)

    def _initialize_fc_layers(self, flattened_dim, device):
        self.fc_mu = nn.Linear(flattened_dim + self.w_dim, self.latent_dim).to(device)
        self.fc_logvar = nn.Linear(flattened_dim + self.w_dim, self.latent_dim).to(device)

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
        batch_size = x.size(0)
        original_size = x.shape[2:]

        w = self._build_w(magnitude, location, station_idx)

        h = self.encoder(x)
        h_flat = h.view(batch_size, -1)

        if self.fc_mu is None:
            self._initialize_fc_layers(h_flat.size(1), x.device)

        posterior_input = torch.cat([h_flat, w], dim=1)
        mu = self.fc_mu(posterior_input)
        logvar = self.fc_logvar(posterior_input)

        z0 = self.reparameterize(mu, logvar)
        zk, log_det = self.flow(z0, w)

        reconstructed = self.decoder(zk, w)
        if reconstructed.shape[2:] != original_size:
            reconstructed = torch.nn.functional.interpolate(
                reconstructed, size=original_size, mode='bilinear', align_corners=False
            )

        return reconstructed, mu, logvar, zk, log_det

    def inverse(self, zk, magnitude, location, station_idx):
        w = self._build_w(magnitude, location, station_idx)
        return self.flow.inverse(zk, w)

    def sample(self, num_samples, magnitude, location, station_idx, device='cuda'):
        z = torch.randn(num_samples, self.latent_dim, device=device)
        w = self._build_w(magnitude, location, station_idx)
        return self.decoder(z, w)
