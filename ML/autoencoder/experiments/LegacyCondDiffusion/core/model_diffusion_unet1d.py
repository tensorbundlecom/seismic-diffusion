import torch
import torch.nn as nn

from .diffusion_utils import sinusoidal_timestep_embedding


class FiLMResBlock1D(nn.Module):
    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.cond_to_scale_shift = nn.Linear(cond_dim, channels * 2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        scale, shift = self.cond_to_scale_shift(cond).chunk(2, dim=1)
        scale = scale.unsqueeze(-1)
        shift = shift.unsqueeze(-1)
        h = h * (1.0 + scale) + shift
        h = self.conv2(self.act(self.norm2(h)))
        return x + h


class UNet1DDenoiser(nn.Module):
    """
    1D U-Net denoiser for vector latent.
    Input/Output are [B, latent_dim], internally treated as [B, 1, latent_dim].
    """

    def __init__(
        self,
        latent_dim: int,
        cond_dim: int,
        base_channels: int = 64,
        time_emb_dim: int = 128,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.time_emb_dim = time_emb_dim

        cond_total = cond_dim * 2
        self.time_proj = nn.Sequential(
            nn.Linear(time_emb_dim, cond_dim),
            nn.SiLU(inplace=True),
            nn.Linear(cond_dim, cond_dim),
        )

        self.stem = nn.Conv1d(1, base_channels, kernel_size=3, padding=1)
        self.down1 = nn.Conv1d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.down2 = nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1)

        self.rb1 = FiLMResBlock1D(base_channels, cond_total)
        self.rb2 = FiLMResBlock1D(base_channels * 2, cond_total)
        self.mid = FiLMResBlock1D(base_channels * 4, cond_total)

        self.up1 = nn.ConvTranspose1d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose1d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)

        self.up_rb1 = FiLMResBlock1D(base_channels * 2, cond_total)
        self.up_rb2 = FiLMResBlock1D(base_channels, cond_total)

        self.out = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(inplace=True),
            nn.Conv1d(base_channels, 1, kernel_size=3, padding=1),
        )

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if t.dim() == 2:
            t = t.squeeze(-1)
        t_emb = sinusoidal_timestep_embedding(t, self.time_emb_dim)
        t_cond = self.time_proj(t_emb)
        full_cond = torch.cat([cond, t_cond], dim=1)

        x = z_t.unsqueeze(1)  # [B,1,D]
        x0 = self.rb1(self.stem(x), full_cond)
        x1 = self.rb2(self.down1(x0), full_cond)
        x2 = self.mid(self.down2(x1), full_cond)

        u1 = self.up1(x2)
        if u1.shape[-1] != x1.shape[-1]:
            u1 = torch.nn.functional.interpolate(u1, size=x1.shape[-1], mode="nearest")
        u1 = self.up_rb1(u1 + x1, full_cond)

        u2 = self.up2(u1)
        if u2.shape[-1] != x0.shape[-1]:
            u2 = torch.nn.functional.interpolate(u2, size=x0.shape[-1], mode="nearest")
        u2 = self.up_rb2(u2 + x0, full_cond)

        out = self.out(u2).squeeze(1)
        if out.shape[1] != self.latent_dim:
            out = torch.nn.functional.interpolate(
                out.unsqueeze(1), size=self.latent_dim, mode="linear", align_corners=False
            ).squeeze(1)
        return out
