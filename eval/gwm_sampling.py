"""
Shared GWM sampling stack for scenario-sweep eval scripts.

Wraps the model loading, conditioning construction, and batch sampling
pattern of evaluate_first_order.py so sweep-style evaluations (attenuation,
scaling, ...) don't each re-copy it. Waveforms are returned in the counts
domain before response deconvolution. Legacy per-event AEs retain the
historical AmplitudeMLP rescale; global AEs bypass it.

Usage:
    sampler = GwmSampler(checkpoint=None, ae_checkpoint=None)
    waves = sampler.generate(metas, steps=1000, gl_iters=200,
                             seed=0, pool=process_pool)

Each meta dict needs the fields create_conditioning_vector expects
(magnitude, latitude, longitude, depth, snr, station_name, station_idx,
channel_idx, channel_type). An optional key ``vs30_override`` replaces the
station's Vs30 in the conditioning (both diffusion and amplitude model),
which is how Vs30 sweeps are driven.
"""

import json
from pathlib import Path

import numpy as np

from ML.diffusion.reconstruction import (
    checkpoint_stft_config,
    diffusion_cache_tag,
    resolve_reconstruction_spec,
)

ROOT = Path(__file__).resolve().parents[1]
DIFF_DIR = ROOT / "ML" / "diffusion"


class GwmSampler:
    def __init__(self, checkpoint=None, ae_checkpoint=None):
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import torch
        import evaluate_first_order as fo
        from ML.diffusion.model import create_conditioning_vector

        self.torch, self.fo = torch, fo
        self._create_cond = create_conditioning_vector

        ckpt_dir = Path(checkpoint) if checkpoint else fo.find_latest_checkpoint()
        self.unet, self.scheduler, cfg = fo.load_diffusion(ckpt_dir)
        self.training_type = cfg.get("training_type", "ddpm")
        self.data_shape = tuple(cfg["data_shape"])
        self.data_mode = cfg.get("data_mode", "latent")
        data_normalization = cfg.get("data_normalization", {})
        self.emb_std = float(data_normalization.get("std", cfg.get("emb_std", 1.0)))
        self.emb_mean = float(data_normalization.get("mean", cfg.get("emb_mean", 0.0)))
        source_path = ckpt_dir / "embedding_source.json"
        if not source_path.exists():
            source_path = DIFF_DIR / "embeddings" / "source.json"
        self.reconstruction = resolve_reconstruction_spec(
            cfg, embeddings_source_path=source_path,
            ae_checkpoint_override=ae_checkpoint,
        )
        fo.configure_stft(checkpoint_stft_config(cfg, embeddings_source_path=source_path))
        self.cache_tag = diffusion_cache_tag(ckpt_dir, cfg, self.reconstruction)
        self.ae = None
        if self.data_mode == "latent":
            ae_ckpt = self.reconstruction.ae_checkpoint
            if not ae_ckpt:
                raise ValueError("Latent diffusion sampling needs an AE checkpoint in checkpoint provenance "
                                 "or an explicit ae_checkpoint.")
            self.ae, _ = fo.load_model(ae_ckpt, device=fo.DEVICE)
            self.ae.eval()

        t = torch.tensor
        self.amp = None
        self.amp_metric = "max"
        if self.reconstruction.uses_amplitude_model:
            self.amp, amp_stats = fo.load_amplitude()
            self.amp_metric = amp_stats.get("metric", "std")
            self.log_std_mean = t(amp_stats["log_std_mean"], dtype=torch.float32)
            self.log_std_scale = t(amp_stats["log_std_scale"], dtype=torch.float32)
            self.gain_mean = t(amp_stats["gain_mean"], dtype=torch.float32)
            self.gain_scale = t(amp_stats["gain_scale"], dtype=torch.float32)

        condition_normalization = cfg.get("conditioning_normalization", {})
        self.cond_mean = t(condition_normalization.get("mean", fo._scale["cond_mean"]),
                           dtype=torch.float32)
        self.cond_std = t(condition_normalization.get("std", fo._scale["cond_std"]),
                          dtype=torch.float32).clamp(min=1e-8)
        self.diff_nc = int(self.unet.num_continuous)
        self.amp_nc = int(self.amp.num_continuous) if self.amp is not None else 0
        self.need_vs30 = max(self.diff_nc, self.amp_nc) >= 7

        self.station_locations = json.load(
            open(DIFF_DIR / "embeddings" / "station_locations.json"))
        vs30_path = DIFF_DIR / "embeddings" / "station_vs30.json"
        self.station_vs30 = json.load(open(vs30_path)) if vs30_path.exists() else None

    def conds(self, meta):
        """(diffusion cond, legacy amplitude cond) for one meta dict."""
        torch = self.torch
        vs30_map = self.station_vs30
        if self.need_vs30 and "vs30_override" in meta:
            vs30_map = {meta["station_name"]: float(meta["vs30_override"])}
        full = self._create_cond(meta, self.station_locations,
                                 vs30_map if self.need_vs30 else None)
        nc_full = full.shape[0] - 2

        def normed(nc):
            v = full.clone()
            v[:nc] = (v[:nc] - self.cond_mean[:nc]) / self.cond_std[:nc]
            return v

        d = normed(self.diff_nc)
        parts = [d[:self.diff_nc], d[nc_full:nc_full + 1]]
        if self.unet.use_channel:
            parts.append(d[nc_full + 1:nc_full + 2])
        if self.amp is None:
            return torch.cat(parts), None
        a = normed(self.amp_nc)
        return torch.cat(parts), torch.cat([a[:self.amp_nc], a[nc_full:nc_full + 1]])

    def generate(self, metas, steps, gl_iters, seed, pool, channel_idx=0):
        """One counts-domain waveform per meta (None on Griffin-Lim failure)."""
        torch, fo = self.torch, self.fo
        pairs = [self.conds(m) for m in metas]
        diff_cond = torch.stack([p[0] for p in pairs]).unsqueeze(1).to(fo.DEVICE)
        if self.amp is not None:
            amp_cond = torch.stack([p[1] for p in pairs]).to(fo.DEVICE)
            with torch.no_grad():
                raw = self.amp(amp_cond).cpu()
            amp_scales = torch.exp(raw[:, :3] * self.log_std_scale + self.log_std_mean)
            if raw.shape[1] >= 6:
                gains = (raw[:, 3:6] * self.gain_scale + self.gain_mean).clamp(0.1, 20.0)
            else:
                gains = torch.full_like(amp_scales, 1.0)
        else:
            amp_scales = gains = None

        x = fo.sample_batch(self.unet, self.scheduler, diff_cond, self.data_shape,
                            steps, self.training_type, seed=seed)
        x = x * self.emb_std + self.emb_mean
        if self.data_mode == "latent":
            with torch.no_grad():
                specs = self.ae.decode(x)
        else:
            specs = x
        specs = specs[:, :3, :fo.FREQ_BINS, :].cpu().numpy()

        jobs = [
            (specs[k, channel_idx], self.reconstruction,
             float(gains[k, channel_idx]) if gains is not None else 1.0,
             float(amp_scales[k, channel_idx]) if amp_scales is not None else None,
             self.amp_metric, gl_iters,
             m["station_name"],
             f"{m.get('channel_type', 'HH')}{['E', 'N', 'Z'][channel_idx]}",
             str(m.get("event_id", "")))
            for k, m in enumerate(metas)
        ]
        out = []
        for res in pool.map(fo._process_synth, jobs):
            out.append(None if res is None else res[0].astype(np.float64))
        return out
