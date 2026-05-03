from __future__ import annotations

import math

import torch
from torch import nn


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb


class Stage2ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, emb_channels: int, dropout: float) -> None:
        super().__init__()
        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, out_channels),
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type_as(h).unsqueeze(-1).unsqueeze(-1)
        h = h + emb_out
        h = self.out_layers(h)
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.proj_out = nn.Conv1d(channels, channels, kernel_size=1)
        self.num_heads = num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x).reshape(b, c, h * w)
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        head_dim = c // self.num_heads
        q = q.reshape(b, self.num_heads, head_dim, h * w)
        k = k.reshape(b, self.num_heads, head_dim, h * w)
        v = v.reshape(b, self.num_heads, head_dim, h * w)
        scale = head_dim ** -0.5
        attn = torch.softmax(torch.einsum("bhcn,bhcm->bhnm", q * scale, k * scale), dim=-1)
        out = torch.einsum("bhnm,bhcm->bhcn", attn, v).reshape(b, c, h * w)
        out = self.proj_out(out).reshape(b, c, h, w)
        return x_in + out


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.op(x)


class Stage2UNet(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        channel_mult: tuple[int, ...],
        num_res_blocks: int,
        attention_resolutions: tuple[int, ...],
        num_heads: int,
        dropout: float,
        cond_features: int,
        station_embedding_enabled: bool = False,
        num_stations: int | None = None,
        station_embedding_dim: int = 0,
    ) -> None:
        super().__init__()
        self.model_channels = model_channels
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, model_channels * 4),
            nn.SiLU(),
            nn.Linear(model_channels * 4, model_channels * 4),
        )
        self.cond_embed = nn.Sequential(
            nn.Linear(cond_features, model_channels * 4),
            nn.SiLU(),
            nn.Linear(model_channels * 4, model_channels * 4),
        )
        self.station_embedding_enabled = bool(station_embedding_enabled)
        if self.station_embedding_enabled:
            if num_stations is None or num_stations <= 0:
                raise ValueError("num_stations must be provided when station embedding is enabled")
            if station_embedding_dim <= 0:
                raise ValueError("station_embedding_dim must be positive when station embedding is enabled")
            self.station_embedding = nn.Embedding(int(num_stations), int(station_embedding_dim))
            self.station_embed = nn.Sequential(
                nn.Linear(int(station_embedding_dim), model_channels * 4),
                nn.SiLU(),
                nn.Linear(model_channels * 4, model_channels * 4),
            )
        else:
            self.station_embedding = None
            self.station_embed = None
        self.input_conv = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)

        self.input_blocks = nn.ModuleList()
        self.input_block_channels = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                out_ch = model_channels * mult
                layers = [Stage2ResBlock(ch, out_ch, model_channels * 4, dropout)]
                ch = out_ch
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads))
                self.input_blocks.append(nn.ModuleList(layers))
                self.input_block_channels.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(nn.ModuleList([Downsample(ch)]))
                self.input_block_channels.append(ch)
                ds *= 2

        self.middle_block = nn.ModuleList(
            [
                Stage2ResBlock(ch, ch, model_channels * 4, dropout),
                AttentionBlock(ch, num_heads),
                Stage2ResBlock(ch, ch, model_channels * 4, dropout),
            ]
        )

        self.output_blocks = nn.ModuleList()
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for block_index in range(num_res_blocks + 1):
                skip_ch = self.input_block_channels.pop()
                out_ch = model_channels * mult
                layers = [Stage2ResBlock(ch + skip_ch, out_ch, model_channels * 4, dropout)]
                ch = out_ch
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads))
                if level and block_index == num_res_blocks:
                    layers.append(Upsample(ch))
                    ds //= 2
                self.output_blocks.append(nn.ModuleList(layers))

        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, kernel_size=3, padding=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        cond: torch.Tensor,
        station_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sigma_embed = timestep_embedding(torch.log(sigma.clamp(min=1.0e-6)) / 4.0, self.model_channels)
        emb = self.time_embed(sigma_embed) + self.cond_embed(cond)
        if self.station_embedding_enabled:
            if station_index is None:
                raise ValueError("station_index is required when station embedding is enabled")
            emb = emb + self.station_embed(self.station_embedding(station_index.long()))

        hs = []
        h = self.input_conv(x)
        hs.append(h)
        for block in self.input_blocks:
            for layer in block:
                if isinstance(layer, Stage2ResBlock):
                    h = layer(h, emb)
                else:
                    h = layer(h)
            hs.append(h)

        for layer in self.middle_block:
            if isinstance(layer, Stage2ResBlock):
                h = layer(h, emb)
            else:
                h = layer(h)

        for block in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            for layer in block:
                if isinstance(layer, Stage2ResBlock):
                    h = layer(h, emb)
                else:
                    h = layer(h)

        return self.out(h)


def build_stage2_unet_from_config(cfg: dict, *, num_stations: int | None = None) -> Stage2UNet:
    stage2_cfg = cfg["stage2"]
    unet_cfg = stage2_cfg["unet"]
    conditioning_cfg = stage2_cfg["conditioning"]
    station_cfg = conditioning_cfg.get("station_embedding", {})
    cond_features = len(cfg["conditions"]["scalar_features"])
    return Stage2UNet(
        in_channels=int(unet_cfg["in_channels"]),
        model_channels=int(unet_cfg["model_channels"]),
        out_channels=int(unet_cfg["out_channels"]),
        channel_mult=tuple(int(v) for v in unet_cfg["channel_mult"]),
        num_res_blocks=int(unet_cfg["num_res_blocks"]),
        attention_resolutions=tuple(int(v) for v in unet_cfg["attention_resolutions"]),
        num_heads=int(unet_cfg["num_heads"]),
        dropout=float(unet_cfg["dropout"]),
        cond_features=cond_features,
        station_embedding_enabled=bool(station_cfg.get("enabled", False)),
        num_stations=num_stations,
        station_embedding_dim=int(station_cfg.get("embedding_dim", 0)),
    )
