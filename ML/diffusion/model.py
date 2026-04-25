from itertools import chain
import math
from pathlib import Path
from typing import Dict

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
    ):
        self.num_continuous = num_continuous
        self.num_stations = int(num_stations)
        self.station_emb_dim = int(station_emb_dim)
        self.cond_dim = self.num_continuous + self.station_emb_dim

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

    def parameters(self):
        return chain(self.model.parameters(), self.station_embedding.parameters())

    def to(self, device):
        self.model.to(device)
        self.station_embedding.to(device)
        return self

    def train(self):
        self.model.train()
        self.station_embedding.train()

    def eval(self):
        self.model.eval()
        self.station_embedding.eval()

    def _encode_conditioning(self, cond):
        """
        Input cond shape: (B, seq_len, num_continuous + 1)
        Last column is raw station_idx.
        """
        if cond.shape[-1] != self.num_continuous + 1:
            raise ValueError(
                f"Expected conditioning width {self.num_continuous + 1}, got {cond.shape[-1]}"
            )
        continuous = cond[..., : self.num_continuous]
        station_idx = cond[..., self.num_continuous].round().long().clamp(0, self.num_stations - 1)
        station_emb = self.station_embedding(station_idx)
        return torch.cat([continuous, station_emb], dim=-1)

    def forward(self, x, timesteps, cond):
        cond_encoded = self._encode_conditioning(cond)
        return self.model(x, timestep=timesteps, encoder_hidden_states=cond_encoded)

    def save_pretrained(self, save_dir):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(save_dir))
        torch.save(
            {
                "state_dict": self.station_embedding.state_dict(),
                "num_stations": self.num_stations,
                "station_emb_dim": self.station_emb_dim,
                "num_continuous": self.num_continuous,
            },
            save_dir / "station_embedding.pt",
        )

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
        )
        wrapper.model = unet
        wrapper.station_embedding.load_state_dict(emb_payload["state_dict"])
        return wrapper


def create_conditioning_vector(metadata: Dict, station_locations: Dict[str, Dict[str, float]]):
    """
    Returns conditioning vector of shape (7,) as:
    [magnitude, 2d_distance_km, sin(azimuth), cos(azimuth), depth_km, snr, station_idx]

    Azimuth is encoded as sin/cos to preserve its circular topology — raw degrees
    would make 1° and 359° appear maximally different after z-score normalization.
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

    continuous = torch.tensor(
        [
            float(metadata["magnitude"]),
            distance_km,
            math.sin(math.radians(azimuth_deg)),
            math.cos(math.radians(azimuth_deg)),
            float(metadata["depth"]),
            float(metadata["snr"]),
        ],
        dtype=torch.float32,
    )
    station_idx = torch.tensor([float(metadata["station_idx"])], dtype=torch.float32)
    return torch.cat([continuous, station_idx])  # (7,)
