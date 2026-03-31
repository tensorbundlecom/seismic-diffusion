from typing import Dict

import torch
from torch.utils.data import Dataset


class LatentCacheDataset(Dataset):
    def __init__(self, cache_pt_path: str, stats_pt_path: str = None):
        payload = torch.load(cache_pt_path, map_location="cpu")
        self.z_mu = payload["z_mu"].float()
        self.z_logvar = payload["z_logvar"].float()
        self.cond_embedding = payload["cond_embedding"].float()
        self.raw_condition = payload["raw_condition"].float()
        self.station_idx = payload["station_idx"].long()
        self.magnitude = payload["magnitude"].float()
        self.location = payload["location"].float()
        self.meta = payload.get("meta", {})

        self.apply_norm = False
        self.z_mean = None
        self.z_std = None

        if stats_pt_path is not None:
            stats = torch.load(stats_pt_path, map_location="cpu")
            self.z_mean = stats["z_mean"].float()
            self.z_std = stats["z_std"].float()
            self.apply_norm = True

    def __len__(self):
        return self.z_mu.size(0)

    def __getitem__(self, idx):
        z = self.z_mu[idx]
        if self.apply_norm:
            z = (z - self.z_mean) / (self.z_std + 1e-8)
        return {
            "z": z,
            "z_mu": self.z_mu[idx],
            "z_logvar": self.z_logvar[idx],
            "cond_embedding": self.cond_embedding[idx],
            "raw_condition": self.raw_condition[idx],
            "station_idx": self.station_idx[idx],
            "magnitude": self.magnitude[idx],
            "location": self.location[idx],
        }

    @staticmethod
    def collate_fn(batch):
        out: Dict[str, torch.Tensor] = {}
        for key in batch[0].keys():
            out[key] = torch.stack([item[key] for item in batch], dim=0)
        return out
