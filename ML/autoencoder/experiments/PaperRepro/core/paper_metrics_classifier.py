from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .blocks import Encoder


@dataclass
class ClassifierConfig:
    dims: int
    in_channels: int
    model_channels: int
    out_channels: int
    num_res_blocks: int
    dropout: float
    channel_mult: tuple[int, ...]
    conv_kernel_size: int
    embedding_dim: int
    num_classes: int


class PaperMetricsClassifier(nn.Module):
    def __init__(self, cfg: ClassifierConfig) -> None:
        super().__init__()
        self.encoder = Encoder(
            in_channels=cfg.in_channels,
            model_channels=cfg.model_channels,
            out_channels=cfg.out_channels,
            num_res_blocks=cfg.num_res_blocks,
            dropout=cfg.dropout,
            channel_mult=cfg.channel_mult,
            conv_kernel_size=cfg.conv_kernel_size,
            dims=cfg.dims,
        )
        self.embedding_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cfg.out_channels, cfg.embedding_dim),
            nn.SiLU(),
            nn.Linear(cfg.embedding_dim, cfg.embedding_dim),
        )
        self.output_layer = nn.Linear(cfg.embedding_dim, cfg.num_classes)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = torch.mean(h, dim=tuple(range(2, h.ndim)))
        h = self.embedding_mlp(h)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.embed(x))


def build_classifier_from_config(cfg: dict, *, num_classes: int) -> PaperMetricsClassifier:
    cls_cfg = cfg["evaluation"]["classifier"]
    return PaperMetricsClassifier(
        ClassifierConfig(
            dims=int(cls_cfg["dims"]),
            in_channels=int(cls_cfg["in_channels"]),
            model_channels=int(cls_cfg["model_channels"]),
            out_channels=int(cls_cfg["out_channels"]),
            num_res_blocks=int(cls_cfg["num_res_blocks"]),
            dropout=float(cls_cfg["dropout"]),
            channel_mult=tuple(int(v) for v in cls_cfg["channel_mult"]),
            conv_kernel_size=int(cls_cfg["conv_kernel_size"]),
            embedding_dim=int(cls_cfg["embedding_dim"]),
            num_classes=int(num_classes),
        )
    )
