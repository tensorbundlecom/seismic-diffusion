from itertools import chain
import math
from pathlib import Path
from typing import Dict, Optional

import torch
from diffusers import UNet2DConditionModel

NUM_CONTINUOUS = 6  # magnitude, 2D distance (km), sin(azimuth), cos(azimuth), depth, snr


class DiffusionUNet2D:
    """UNet2DConditionModel wrapper with learned station-id embedding."""

    def __init__(
        self,
        in_channels,
        out_channels,
        num_stations,
        station_emb_dim=64,
        num_continuous=NUM_CONTINUOUS,
        base_channels=64,
        num_channels=0,
        channel_emb_dim=16,
    ):
        self.num_continuous = num_continuous
        self.num_stations = int(num_stations)
        self.station_emb_dim = int(station_emb_dim)
        # Optional learned embedding for the instrument/channel type (HH/HN/EH/BH).
        # num_channels=0 keeps the legacy station-only conditioning for old checkpoints.
        self.num_channels = int(num_channels)
        self.use_channel = self.num_channels > 0
        self.channel_emb_dim = int(channel_emb_dim) if self.use_channel else 0
        self.cond_dim = self.num_continuous + self.station_emb_dim + self.channel_emb_dim

        self.model = UNet2DConditionModel(
            sample_size=None,
            in_channels=in_channels,
            out_channels=out_channels,
            # 3 resolution levels: (64, 128, 256)
            # The first block preserves the in_channels count so there
            # is no information bottleneck at the very start.
            block_out_channels=(
                base_channels,
                base_channels * 2,
                base_channels * 4,
            ),
            layers_per_block=2,
            down_block_types=(
                "DownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
            ),
            up_block_types=(
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "UpBlock2D",
            ),
            cross_attention_dim=self.cond_dim,
        )
        self.station_embedding = torch.nn.Embedding(self.num_stations, self.station_emb_dim)
        if self.use_channel:
            self.channel_embedding = torch.nn.Embedding(self.num_channels, self.channel_emb_dim)

    @property
    def cond_input_width(self):
        """Raw conditioning-vector width this model expects (before embedding lookups)."""
        return self.num_continuous + 1 + (1 if self.use_channel else 0)

    def _embeddings(self):
        embs = [self.station_embedding]
        if self.use_channel:
            embs.append(self.channel_embedding)
        return embs

    def parameters(self):
        return chain(self.model.parameters(), *(e.parameters() for e in self._embeddings()))

    def to(self, device):
        self.model.to(device)
        for e in self._embeddings():
            e.to(device)
        return self

    def train(self):
        self.model.train()
        for e in self._embeddings():
            e.train()

    def eval(self):
        self.model.eval()
        for e in self._embeddings():
            e.eval()

    def _encode_conditioning(self, cond):
        """
        Input cond shape: (B, seq_len, cond_input_width).
        Layout: [num_continuous continuous dims, station_idx, (channel_idx if use_channel)].
        """
        if cond.shape[-1] != self.cond_input_width:
            raise ValueError(
                f"Expected conditioning width {self.cond_input_width}, got {cond.shape[-1]}"
            )
        continuous = cond[..., : self.num_continuous]
        station_idx = cond[..., self.num_continuous].round().long().clamp(0, self.num_stations - 1)
        parts = [continuous, self.station_embedding(station_idx)]
        if self.use_channel:
            channel_idx = (
                cond[..., self.num_continuous + 1].round().long().clamp(0, self.num_channels - 1)
            )
            parts.append(self.channel_embedding(channel_idx))
        return torch.cat(parts, dim=-1)

    def forward(self, x, timesteps, cond):
        cond_encoded = self._encode_conditioning(cond)
        return self.model(x, timestep=timesteps, encoder_hidden_states=cond_encoded)

    def save_pretrained(self, save_dir):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(save_dir))
        payload = {
            "state_dict": self.station_embedding.state_dict(),
            "num_stations": self.num_stations,
            "station_emb_dim": self.station_emb_dim,
            "num_continuous": self.num_continuous,
            "num_channels": self.num_channels,
            "channel_emb_dim": self.channel_emb_dim,
        }
        if self.use_channel:
            payload["channel_state_dict"] = self.channel_embedding.state_dict()
        torch.save(payload, save_dir / "station_embedding.pt")

    @classmethod
    def load_pretrained(cls, load_dir):
        load_dir = Path(load_dir)
        unet = UNet2DConditionModel.from_pretrained(str(load_dir))
        emb_payload = torch.load(load_dir / "station_embedding.pt", map_location="cpu")
        wrapper = cls(
            in_channels=unet.config.in_channels,
            out_channels=unet.config.out_channels,
            num_stations=int(emb_payload["num_stations"]),
            station_emb_dim=int(emb_payload["station_emb_dim"]),
            num_continuous=int(emb_payload.get("num_continuous", NUM_CONTINUOUS)),
            base_channels=int(unet.config.block_out_channels[0]),
            num_channels=int(emb_payload.get("num_channels", 0)),
            channel_emb_dim=int(emb_payload.get("channel_emb_dim", 16)),
        )
        wrapper.model = unet
        wrapper.station_embedding.load_state_dict(emb_payload["state_dict"])
        if wrapper.use_channel and "channel_state_dict" in emb_payload:
            wrapper.channel_embedding.load_state_dict(emb_payload["channel_state_dict"])
        return wrapper


def create_conditioning_vector(
    metadata: Dict,
    station_locations: Dict[str, Dict[str, float]],
    station_vs30: Optional[Dict[str, float]] = None,
):
    """
    Returns conditioning vector of shape (8,) as:
    [magnitude, 2d_distance_km, sin(azimuth), cos(azimuth), depth_km, snr,
     station_idx, channel_idx]

    When ``station_vs30`` is provided, the site Vs30 (m/s) is inserted as an extra
    continuous feature right after snr, giving shape (9,):
    [..., snr, vs30, station_idx, channel_idx]. It stays inside the leading
    continuous block so it is z-score normalized alongside the other scalars.

    Azimuth is encoded as sin/cos to preserve its circular topology — raw degrees
    would make 1° and 359° appear maximally different after z-score normalization.

    channel_idx (HH/HN/EH/BH) is appended last so consumers that only need the
    station tail (e.g. the amplitude MLP) keep working unchanged.
    """
    station_name = metadata["station_name"]
    if station_name not in station_locations:
        raise KeyError(
            f"Missing station coordinates for '{station_name}'. "
            "Generate station locations first (see fetch_station_locations.py)."
        )

    station_lat = float(station_locations[station_name]["latitude"])
    station_lon = float(station_locations[station_name]["longitude"])
    event_lat = float(metadata["latitude"])
    event_lon = float(metadata["longitude"])

    event_lat_rad = math.radians(event_lat)
    event_lon_rad = math.radians(event_lon)
    station_lat_rad = math.radians(station_lat)
    station_lon_rad = math.radians(station_lon)

    dlat = station_lat_rad - event_lat_rad
    dlon = station_lon_rad - event_lon_rad

    # Haversine distance on WGS84 sphere approximation.
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(event_lat_rad) * math.cos(station_lat_rad) * math.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(1.0 - a, 0.0)))
    distance_km = 6371.0 * c

    # Forward azimuth (event -> station), in [0, 360).
    y = math.sin(dlon) * math.cos(station_lat_rad)
    x = (
        math.cos(event_lat_rad) * math.sin(station_lat_rad)
        - math.sin(event_lat_rad) * math.cos(station_lat_rad) * math.cos(dlon)
    )
    azimuth_deg = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0

    continuous_values = [
        float(metadata["magnitude"]),
        distance_km,
        math.sin(math.radians(azimuth_deg)),
        math.cos(math.radians(azimuth_deg)),
        float(metadata["depth"]),
        float(metadata["snr"]),
    ]
    if station_vs30 is not None:
        if station_name not in station_vs30:
            raise KeyError(
                f"Missing Vs30 for station '{station_name}'. "
                "Generate it first (see compute_station_vs30.py)."
            )
        continuous_values.append(float(station_vs30[station_name]))

    continuous = torch.tensor(continuous_values, dtype=torch.float32)
    station_idx = torch.tensor([float(metadata["station_idx"])], dtype=torch.float32)
    channel_idx = torch.tensor([float(metadata.get("channel_idx", 0))], dtype=torch.float32)
    return torch.cat([continuous, station_idx, channel_idx])  # (8,) or (9,) with vs30
