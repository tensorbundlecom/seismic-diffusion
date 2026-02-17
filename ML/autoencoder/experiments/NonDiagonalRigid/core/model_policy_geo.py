from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn


SUPPORTED_POLICIES = {"width_only", "depth_only", "hybrid"}


def _round_to_multiple(value: float, multiple: int = 8, min_value: int = 8) -> int:
    out = int(round(value / multiple) * multiple)
    return max(min_value, out)


def _scaled_channels(base_channels: Sequence[int], scale: float) -> List[int]:
    return [_round_to_multiple(float(c) * float(scale), multiple=8, min_value=8) for c in base_channels]


def _depth_profile(policy: str, scale: float) -> List[int]:
    # Keep downsample stages fixed; only internal per-stage depth changes.
    if policy == "width_only":
        return [1, 1, 1, 1]
    if policy == "depth_only":
        if scale >= 0.99:
            return [2, 2, 2, 2]
        if scale >= 0.74:
            return [2, 2, 1, 1]
        return [1, 1, 1, 1]
    if policy == "hybrid":
        if scale >= 0.99:
            return [2, 2, 2, 2]
        if scale >= 0.74:
            return [2, 1, 1, 1]
        return [1, 1, 1, 1]
    raise ValueError(f"Unknown policy: {policy}")


def resolve_backbone_plan(
    policy: str,
    scale: float,
    base_channels: Sequence[int] = (32, 64, 128, 256),
) -> Dict:
    if policy not in SUPPORTED_POLICIES:
        raise ValueError(f"Unsupported policy: {policy}")
    if scale <= 0:
        raise ValueError(f"scale must be > 0, got {scale}")

    if policy == "depth_only":
        channels = list(base_channels)
    else:
        channels = _scaled_channels(base_channels, scale)
    depth_repeats = _depth_profile(policy, scale)

    return {
        "policy": policy,
        "scale": float(scale),
        "base_channels": [int(c) for c in base_channels],
        "channels": [int(c) for c in channels],
        "depth_repeats": [int(r) for r in depth_repeats],
    }


def _make_encoder_stack(in_channels: int, channels: Sequence[int], depth_repeats: Sequence[int]) -> nn.Sequential:
    layers: List[nn.Module] = []
    c_in = in_channels
    for out_c, repeats in zip(channels, depth_repeats):
        # Stage downsample block (fixed across policies/scales).
        layers.extend(
            [
                nn.Conv2d(c_in, out_c, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            ]
        )
        # Optional internal depth blocks for depth/hybrid policies.
        for _ in range(max(0, int(repeats) - 1)):
            layers.extend(
                [
                    nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(inplace=True),
                ]
            )
        c_in = out_c
    return nn.Sequential(*layers)


def _make_decoder_stack(out_channels: int, channels: Sequence[int], depth_repeats: Sequence[int]) -> nn.Sequential:
    layers: List[nn.Module] = []
    decode_targets = list(reversed(channels[:-1])) + [out_channels]
    decode_repeats = list(reversed(depth_repeats))

    c_in = channels[-1]
    for i, c_out in enumerate(decode_targets):
        is_last = i == (len(decode_targets) - 1)
        layers.append(nn.ConvTranspose2d(c_in, c_out, kernel_size=3, stride=2, padding=1, output_padding=1))
        if is_last:
            layers.append(nn.Sigmoid())
        else:
            layers.extend([nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)])
            repeats = decode_repeats[i] if i < len(decode_repeats) else 1
            for _ in range(max(0, int(repeats) - 1)):
                layers.extend(
                    [
                        nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(c_out),
                        nn.ReLU(inplace=True),
                    ]
                )
        c_in = c_out
    return nn.Sequential(*layers)


class PolicyGeoConditionedFeatureExtractor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_stations: int,
        condition_dim: int,
        numeric_condition_dim: int,
        channels: Sequence[int],
        depth_repeats: Sequence[int],
    ):
        super().__init__()
        self.encoder = _make_encoder_stack(
            in_channels=in_channels,
            channels=channels,
            depth_repeats=depth_repeats,
        )
        self.condition_dim = condition_dim
        self.station_embedding = nn.Embedding(num_stations, condition_dim // 4)
        condition_input_dim = numeric_condition_dim + (condition_dim // 4)
        self.condition_network = nn.Sequential(
            nn.Linear(condition_input_dim, condition_dim),
            nn.ReLU(inplace=True),
            nn.Linear(condition_dim, condition_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, cond_numeric: torch.Tensor, station_idx: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h_flat = h.view(h.size(0), -1)
        station_emb = self.station_embedding(station_idx)
        cond_input = torch.cat([cond_numeric, station_emb], dim=1)
        cond_features = self.condition_network(cond_input)
        return torch.cat([h_flat, cond_features], dim=1)


class PolicyGeoCVAEEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 128,
        num_stations: int = 100,
        condition_dim: int = 64,
        numeric_condition_dim: int = 5,
        channels: Sequence[int] = (32, 64, 128, 256),
        depth_repeats: Sequence[int] = (1, 1, 1, 1),
    ):
        super().__init__()
        self.feature_extractor = PolicyGeoConditionedFeatureExtractor(
            in_channels=in_channels,
            num_stations=num_stations,
            condition_dim=condition_dim,
            numeric_condition_dim=numeric_condition_dim,
            channels=channels,
            depth_repeats=depth_repeats,
        )
        self.latent_dim = latent_dim
        self.fc_mu = None
        self.fc_logvar = None

    def _initialize_fc_layers(self, in_dim: int, device: torch.device) -> None:
        self.fc_mu = nn.Linear(in_dim, self.latent_dim).to(device)
        self.fc_logvar = nn.Linear(in_dim, self.latent_dim).to(device)

    def forward(self, x: torch.Tensor, cond_numeric: torch.Tensor, station_idx: torch.Tensor):
        combined = self.feature_extractor(x, cond_numeric, station_idx)
        if self.fc_mu is None:
            self._initialize_fc_layers(combined.size(1), x.device)
        mu = self.fc_mu(combined)
        logvar = self.fc_logvar(combined)
        return mu, logvar


class PolicyGeoCVAEDecoder(nn.Module):
    def __init__(
        self,
        out_channels: int = 3,
        latent_dim: int = 128,
        num_stations: int = 100,
        condition_dim: int = 64,
        numeric_condition_dim: int = 5,
        channels: Sequence[int] = (32, 64, 128, 256),
        depth_repeats: Sequence[int] = (1, 1, 1, 1),
    ):
        super().__init__()
        self.condition_dim = condition_dim
        self.station_embedding = nn.Embedding(num_stations, condition_dim // 4)
        condition_input_dim = numeric_condition_dim + (condition_dim // 4)
        self.condition_network = nn.Sequential(
            nn.Linear(condition_input_dim, condition_dim),
            nn.ReLU(inplace=True),
            nn.Linear(condition_dim, condition_dim),
            nn.ReLU(inplace=True),
        )

        self.spatial_dim = 16
        self.deep_channels = int(channels[-1])
        self.projection_dim = self.deep_channels * self.spatial_dim * self.spatial_dim
        self.fc_projection = nn.Linear(latent_dim + condition_dim, self.projection_dim)
        self.decoder = _make_decoder_stack(
            out_channels=out_channels,
            channels=channels,
            depth_repeats=depth_repeats,
        )

    def forward(self, z: torch.Tensor, cond_numeric: torch.Tensor, station_idx: torch.Tensor):
        batch_size = z.size(0)
        station_emb = self.station_embedding(station_idx)
        cond_input = torch.cat([cond_numeric, station_emb], dim=1)
        cond_features = self.condition_network(cond_input)
        combined = torch.cat([z, cond_features], dim=1)
        h = self.fc_projection(combined)
        h = h.view(batch_size, self.deep_channels, self.spatial_dim, self.spatial_dim)
        return self.decoder(h)


class PolicyGeoConditionalVariationalAutoencoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 128,
        num_stations: int = 100,
        condition_dim: int = 64,
        numeric_condition_dim: int = 5,
        policy: str = "width_only",
        scale: float = 1.0,
        base_channels: Sequence[int] = (32, 64, 128, 256),
    ):
        super().__init__()
        plan = resolve_backbone_plan(policy=policy, scale=scale, base_channels=base_channels)
        self.backbone_plan = plan
        channels = plan["channels"]
        depth_repeats = plan["depth_repeats"]

        self.encoder = PolicyGeoCVAEEncoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
            num_stations=num_stations,
            condition_dim=condition_dim,
            numeric_condition_dim=numeric_condition_dim,
            channels=channels,
            depth_repeats=depth_repeats,
        )
        self.decoder = PolicyGeoCVAEDecoder(
            out_channels=in_channels,
            latent_dim=latent_dim,
            num_stations=num_stations,
            condition_dim=condition_dim,
            numeric_condition_dim=numeric_condition_dim,
            channels=channels,
            depth_repeats=depth_repeats,
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


class PolicyGeoFullCovEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 128,
        num_stations: int = 100,
        condition_dim: int = 64,
        numeric_condition_dim: int = 5,
        channels: Sequence[int] = (32, 64, 128, 256),
        depth_repeats: Sequence[int] = (1, 1, 1, 1),
    ):
        super().__init__()
        self.feature_extractor = PolicyGeoConditionedFeatureExtractor(
            in_channels=in_channels,
            num_stations=num_stations,
            condition_dim=condition_dim,
            numeric_condition_dim=numeric_condition_dim,
            channels=channels,
            depth_repeats=depth_repeats,
        )
        self.latent_dim = latent_dim
        self.num_chol = (latent_dim * (latent_dim + 1)) // 2
        self.fc_mu = None
        self.fc_chol = None

    def _initialize_fc_layers(self, in_dim: int, device: torch.device) -> None:
        self.fc_mu = nn.Linear(in_dim, self.latent_dim).to(device)
        self.fc_chol = nn.Linear(in_dim, self.num_chol).to(device)

    def forward(self, x: torch.Tensor, cond_numeric: torch.Tensor, station_idx: torch.Tensor):
        combined = self.feature_extractor(x, cond_numeric, station_idx)
        if self.fc_mu is None:
            self._initialize_fc_layers(combined.size(1), x.device)

        mu = self.fc_mu(combined)
        chol_flat = self.fc_chol(combined)
        b = mu.size(0)

        L = torch.zeros(b, self.latent_dim, self.latent_dim, device=x.device, dtype=mu.dtype)
        tril_idx = torch.tril_indices(row=self.latent_dim, col=self.latent_dim, offset=0, device=x.device)
        L[:, tril_idx[0], tril_idx[1]] = chol_flat

        diag_idx = torch.arange(self.latent_dim, device=x.device)
        diag_vals = torch.exp(L[:, diag_idx, diag_idx].float()).to(L.dtype)
        L[:, diag_idx, diag_idx] = diag_vals
        return mu, L


class PolicyGeoFullCovCVAE(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 128,
        num_stations: int = 100,
        condition_dim: int = 64,
        numeric_condition_dim: int = 5,
        policy: str = "width_only",
        scale: float = 1.0,
        base_channels: Sequence[int] = (32, 64, 128, 256),
    ):
        super().__init__()
        plan = resolve_backbone_plan(policy=policy, scale=scale, base_channels=base_channels)
        self.backbone_plan = plan
        channels = plan["channels"]
        depth_repeats = plan["depth_repeats"]

        self.encoder = PolicyGeoFullCovEncoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
            num_stations=num_stations,
            condition_dim=condition_dim,
            numeric_condition_dim=numeric_condition_dim,
            channels=channels,
            depth_repeats=depth_repeats,
        )
        self.decoder = PolicyGeoCVAEDecoder(
            out_channels=in_channels,
            latent_dim=latent_dim,
            num_stations=num_stations,
            condition_dim=condition_dim,
            numeric_condition_dim=numeric_condition_dim,
            channels=channels,
            depth_repeats=depth_repeats,
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
