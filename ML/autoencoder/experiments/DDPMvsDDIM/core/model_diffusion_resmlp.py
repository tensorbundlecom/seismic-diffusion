import torch
import torch.nn as nn

from ML.autoencoder.experiments.DDPMvsDDIM.core.diffusion_utils import sinusoidal_timestep_embedding


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

    def forward(self, x: torch.Tensor):
        return x + self.block(x)


class ResMLPDenoiser(nn.Module):
    """Predicts epsilon for latent DDPM/DDIM training."""

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

    def forward(self, noisy_latent: torch.Tensor, timesteps: torch.Tensor, cond: torch.Tensor):
        if timesteps.dim() == 2:
            timesteps = timesteps.squeeze(-1)
        time_emb = sinusoidal_timestep_embedding(timesteps, self.time_emb_dim)
        hidden = self.in_proj(noisy_latent) + self.time_mlp(time_emb) + self.cond_mlp(cond)
        for block in self.blocks:
            hidden = block(hidden)
        return self.out(hidden)


class AdaLNResidualBlock(nn.Module):
    def __init__(self, dim: int, cond_dim: int, dropout: float = 0.0, expansion: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.modulation = nn.Sequential(
            nn.SiLU(inplace=False),
            nn.Linear(cond_dim, dim * 3),
        )
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion, dim),
        )

    def forward(self, x: torch.Tensor, cond_context: torch.Tensor):
        shift, scale, gate = self.modulation(cond_context).chunk(3, dim=-1)
        hidden = self.norm(x) * (1.0 + scale) + shift
        hidden = self.mlp(hidden)
        return x + torch.sigmoid(gate) * hidden


class AdaLNResMLPDenoiser(nn.Module):
    """Condition-aware epsilon denoiser with per-block adaptive layer norm modulation."""

    def __init__(
        self,
        latent_dim: int,
        cond_dim: int,
        hidden_dim: int = 768,
        depth: int = 10,
        time_emb_dim: int = 128,
        dropout: float = 0.0,
        expansion: int = 4,
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
        self.blocks = nn.ModuleList(
            [AdaLNResidualBlock(hidden_dim, hidden_dim, dropout=dropout, expansion=expansion) for _ in range(depth)]
        )
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.final_modulation = nn.Sequential(
            nn.SiLU(inplace=False),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )
        self.out = nn.Linear(hidden_dim, latent_dim)

    def forward(self, noisy_latent: torch.Tensor, timesteps: torch.Tensor, cond: torch.Tensor):
        if timesteps.dim() == 2:
            timesteps = timesteps.squeeze(-1)
        time_emb = sinusoidal_timestep_embedding(timesteps, self.time_emb_dim)
        cond_context = self.time_mlp(time_emb) + self.cond_mlp(cond)
        hidden = self.in_proj(noisy_latent)
        for block in self.blocks:
            hidden = block(hidden, cond_context)
        out_shift, out_scale = self.final_modulation(cond_context).chunk(2, dim=-1)
        hidden = self.final_norm(hidden) * (1.0 + out_scale) + out_shift
        return self.out(hidden)


def build_diffusion_denoiser(
    model_type: str,
    latent_dim: int,
    cond_dim: int,
    hidden_dim: int,
    depth: int,
    dropout: float,
):
    if model_type == "resmlp":
        return ResMLPDenoiser(
            latent_dim=latent_dim,
            cond_dim=cond_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
        )
    if model_type == "adaln_resmlp":
        return AdaLNResMLPDenoiser(
            latent_dim=latent_dim,
            cond_dim=cond_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
        )
    raise ValueError(f"Unsupported diffusion model_type: {model_type}")
