import torch
import torch.nn as nn

from .diffusion_utils import sinusoidal_timestep_embedding


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResMLPDenoiser(nn.Module):
    """
    Predicts clean latent z0 from noisy latent zt, timestep t, and condition vector.
    """

    def __init__(
        self,
        latent_dim: int,
        cond_dim: int,
        hidden_dim: int = 512,
        depth: int = 6,
        time_emb_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.time_emb_dim = time_emb_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.in_proj = nn.Linear(latent_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim, dropout=dropout) for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, latent_dim))

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if t.dim() == 2:
            t = t.squeeze(-1)
        t_emb = sinusoidal_timestep_embedding(t, self.time_emb_dim)
        h = self.in_proj(z_t) + self.time_mlp(t_emb) + self.cond_mlp(cond)
        for blk in self.blocks:
            h = blk(h)
        return self.out(h)

