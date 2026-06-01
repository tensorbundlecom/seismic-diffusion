from pathlib import Path

import torch
import torch.nn as nn

NUM_CONTINUOUS = 6  # must match diffusion model


class AmplitudeMLP(nn.Module):
    """
    Predicts per-channel waveform amplitude (std) from metadata conditioning.

    Input:  (B, num_continuous + 1) — same format as the diffusion model:
            first num_continuous dims are z-scored continuous features,
            last dim is raw station_idx (integer stored as float).
    Output: (B, 3) — predicted std for each channel (E, N, Z), always positive.

    The model outputs raw values; callers must apply the inverse of the
    log-std normalization used during training (see amp_stats.json).
    """

    def __init__(
        self,
        num_stations: int,
        num_continuous: int = NUM_CONTINUOUS,
        station_emb_dim: int = 32,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.num_continuous = num_continuous
        self.num_stations = num_stations
        self.station_emb_dim = station_emb_dim
        self.hidden_dim = hidden_dim

        self.station_embedding = nn.Embedding(num_stations, station_emb_dim)

        input_dim = num_continuous + station_emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        continuous = cond[:, : self.num_continuous]
        station_idx = (
            cond[:, self.num_continuous].round().long().clamp(0, self.num_stations - 1)
        )
        station_emb = self.station_embedding(station_idx)
        x = torch.cat([continuous, station_emb], dim=-1)
        return self.mlp(x)  # (B, 3) — raw log-std values (normalized)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "num_stations": self.num_stations,
                "num_continuous": self.num_continuous,
                "station_emb_dim": self.station_emb_dim,
                "hidden_dim": self.hidden_dim,
            },
            path,
        )

    @classmethod
    def load(cls, path, device="cpu"):
        payload = torch.load(path, map_location=device)
        model = cls(
            num_stations=payload["num_stations"],
            num_continuous=payload["num_continuous"],
            station_emb_dim=payload["station_emb_dim"],
            hidden_dim=payload["hidden_dim"],
        )
        model.load_state_dict(payload["state_dict"])
        return model.to(device)
