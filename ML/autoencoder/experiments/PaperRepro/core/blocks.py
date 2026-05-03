from __future__ import annotations

import torch.nn.functional as F
from torch import nn

from .nn import avg_pool_nd, conv_nd, normalization, zero_module


class Upsample(nn.Module):
    def __init__(self, channels: int, use_conv: bool, dims: int = 2, out_channels: int | None = None, kernel_size: int = 3):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        self.conv = (
            conv_nd(dims, channels, self.out_channels, kernel_size, padding="same")
            if use_conv
            else None
        )

    def forward(self, x):
        if self.dims == 3:
            x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest")
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.conv is not None:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, channels: int, use_conv: bool, dims: int = 2, out_channels: int | None = None, kernel_size: int = 3):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims,
                channels,
                self.out_channels,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2,
            )
        else:
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        return self.op(x)


class ResBlock(nn.Module):
    def __init__(self, channels: int, dropout: float, out_channels: int | None = None, kernel_size: int = 3, dims: int = 2):
        super().__init__()
        out_channels = out_channels or channels
        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, out_channels, kernel_size, padding="same"),
        )
        self.out_layers = nn.Sequential(
            normalization(out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, out_channels, out_channels, kernel_size, padding="same")),
        )
        self.skip_connection = nn.Identity() if out_channels == channels else conv_nd(dims, channels, out_channels, 1)

    def forward(self, x):
        return self.skip_connection(x) + self.out_layers(self.in_layers(x))


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        num_res_blocks: int,
        dropout: float = 0.0,
        channel_mult: tuple[int, ...] = (1, 2, 4),
        conv_kernel_size: int = 3,
        conv_resample: bool = True,
        dims: int = 2,
    ) -> None:
        super().__init__()
        ch = int(channel_mult[0] * model_channels)
        self.input_layer = conv_nd(dims, in_channels, ch, conv_kernel_size, padding="same")
        blocks = []
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                blocks.append(
                    ResBlock(
                        ch,
                        dropout,
                        out_channels=int(mult * model_channels),
                        kernel_size=conv_kernel_size,
                        dims=dims,
                    )
                )
                ch = int(mult * model_channels)
            if level != len(channel_mult) - 1:
                blocks.append(Downsample(ch, conv_resample, dims=dims, out_channels=ch, kernel_size=conv_kernel_size))
        self.blocks = nn.Sequential(*blocks)
        self.output_layer = conv_nd(dims, ch, out_channels, conv_kernel_size, padding="same")

    def forward(self, x):
        x = self.input_layer(x)
        x = self.blocks(x)
        return self.output_layer(x)


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        num_res_blocks: int,
        dropout: float = 0.0,
        channel_mult: tuple[int, ...] = (1, 2, 4),
        conv_kernel_size: int = 3,
        conv_resample: bool = True,
        dims: int = 2,
    ) -> None:
        super().__init__()
        ch = int(channel_mult[-1] * model_channels)
        self.input_layer = conv_nd(dims, in_channels, ch, conv_kernel_size, padding="same")
        blocks = []
        for level, mult in reversed(list(enumerate(channel_mult))):
            if level != len(channel_mult) - 1:
                blocks.append(Upsample(ch, conv_resample, dims=dims, out_channels=ch, kernel_size=conv_kernel_size))
            for _ in range(num_res_blocks):
                blocks.append(
                    ResBlock(
                        ch,
                        dropout,
                        out_channels=int(mult * model_channels),
                        kernel_size=conv_kernel_size,
                        dims=dims,
                    )
                )
                ch = int(mult * model_channels)
        self.blocks = nn.Sequential(*blocks)
        self.output_layer = conv_nd(dims, ch, out_channels, conv_kernel_size, padding="same")

    def forward(self, x):
        x = self.input_layer(x)
        x = self.blocks(x)
        return self.output_layer(x)
