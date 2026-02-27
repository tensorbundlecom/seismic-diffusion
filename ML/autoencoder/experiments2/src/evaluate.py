"""Evaluation and metric utilities for experiments2/exp001."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F


LOWER_BETTER_METRICS = {
    "complex_l1",
    "mr_lsd",
    "abs_xcorr_lag_s",
    "band_energy_ratio_error",
    "mid_band_error",
    "onset_mae_p_s",
    "onset_mae_s_s",
    "onset_mae_dtps_s",
}

HIGHER_BETTER_METRICS = {"xcorr_max", "envelope_corr"}


def complex_l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dr = pred[:, 0] - target[:, 0]
    di = pred[:, 1] - target[:, 1]
    return torch.mean(torch.sqrt(dr * dr + di * di + 1e-8))


def logmag_l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_mag = torch.sqrt(pred[:, 0] * pred[:, 0] + pred[:, 1] * pred[:, 1] + 1e-8)
    true_mag = torch.sqrt(target[:, 0] * target[:, 0] + target[:, 1] * target[:, 1] + 1e-8)
    return F.l1_loss(torch.log1p(pred_mag), torch.log1p(true_mag), reduction="mean")


def kl_terms(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    free_bits_per_dim: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    kl_dim = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
    kl_raw = kl_dim.mean()
    kl_fb = torch.maximum(kl_dim, torch.full_like(kl_dim, float(free_bits_per_dim))).mean()
    return kl_raw, kl_fb


def total_vae_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    lambda_complex: float,
    lambda_logmag: float,
    beta_t: float,
    free_bits_per_dim: float,
) -> Dict[str, torch.Tensor]:
    loss_complex = complex_l1_loss(pred, target)
    loss_logmag = logmag_l1_loss(pred, target)
    recon = float(lambda_complex) * loss_complex + float(lambda_logmag) * loss_logmag
    kl_raw, kl_fb = kl_terms(mu, logvar, free_bits_per_dim=float(free_bits_per_dim))
    total = recon + float(beta_t) * kl_fb
    return {
        "total": total,
        "recon_total": recon,
        "loss_complex": loss_complex,
        "loss_logmag": loss_logmag,
        "kl_raw": kl_raw,
        "kl_fb": kl_fb,
    }


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _magnitudes_from_complex(x: np.ndarray) -> np.ndarray:
    # x shape: [B, 2, F, T]
    return np.sqrt(np.square(x[:, 0]) + np.square(x[:, 1]) + 1e-8)


def _mr_lsd_batch(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Multi-resolution log-spectral distance with pooling scales.
    Returns per-sample value, shape [B].
    """
    pred_t = torch.from_numpy(pred)
    target_t = torch.from_numpy(target)
    pred_mag = torch.sqrt(pred_t[:, 0] ** 2 + pred_t[:, 1] ** 2 + 1e-8)
    true_mag = torch.sqrt(target_t[:, 0] ** 2 + target_t[:, 1] ** 2 + 1e-8)

    vals: List[torch.Tensor] = []
    for s in (1, 2, 4):
        if s == 1:
            p = pred_mag
            t = true_mag
        else:
            p = F.avg_pool2d(pred_mag.unsqueeze(1), kernel_size=(s, s), stride=(s, s)).squeeze(1)
            t = F.avg_pool2d(true_mag.unsqueeze(1), kernel_size=(s, s), stride=(s, s)).squeeze(1)
        d = torch.log1p(p) - torch.log1p(t)
        vals.append(torch.sqrt(torch.mean(d * d, dim=(1, 2)) + 1e-12))
    return torch.stack(vals, dim=0).mean(dim=0).cpu().numpy()


def _envelope_from_complex(x: np.ndarray) -> np.ndarray:
    # x: [B,2,F,T]
    mag = _magnitudes_from_complex(x)
    return np.sum(np.square(mag), axis=1)


def _xcorr_stats_batch(env_pred: np.ndarray, env_true: np.ndarray, frame_sec: float) -> tuple[np.ndarray, np.ndarray]:
    b = env_pred.shape[0]
    xcorr_max = np.zeros(b, dtype=np.float64)
    lag_s = np.zeros(b, dtype=np.float64)
    for i in range(b):
        a = env_true[i]
        c = env_pred[i]
        a = (a - np.mean(a)) / (np.std(a) + 1e-8)
        c = (c - np.mean(c)) / (np.std(c) + 1e-8)
        corr = np.correlate(a, c, mode="full") / max(1, len(a))
        arg = int(np.argmax(np.abs(corr)))
        lag_frames = arg - (len(a) - 1)
        xcorr_max[i] = float(np.abs(corr[arg]))
        lag_s[i] = float(lag_frames * frame_sec)
    return xcorr_max, lag_s


def _band_ratio_errors(
    pred: np.ndarray,
    target: np.ndarray,
    cfg: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    stft_cfg = cfg["stft"]
    eval_cfg = cfg["evaluation"]
    fs = float(cfg["data"]["sampling_rate_hz"])
    n_fft = int(stft_cfg["n_fft"])
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    if bool(stft_cfg["drop_nyquist"]):
        freqs = freqs[:-1]
    f_target = int(stft_cfg["target_freq_bins"])
    if len(freqs) > f_target:
        freqs = freqs[:f_target]
    elif len(freqs) < f_target:
        pad = np.full((f_target - len(freqs),), freqs[-1] if len(freqs) else 0.0)
        freqs = np.concatenate([freqs, pad], axis=0)

    low_lo, low_hi = eval_cfg["metric_bands_hz"]["low"]
    mid_lo, mid_hi = eval_cfg["metric_bands_hz"]["mid"]
    high_lo, high_hi = eval_cfg["metric_bands_hz"]["high"]
    low_mask = (freqs >= float(low_lo)) & (freqs < float(low_hi))
    mid_mask = (freqs >= float(mid_lo)) & (freqs < float(mid_hi))
    high_mask = (freqs >= float(high_lo)) & (freqs < float(high_hi))
    if not np.any(low_mask):
        raise ValueError("metric_bands_hz.low produced an empty frequency mask.")
    if not np.any(mid_mask):
        raise ValueError("metric_bands_hz.mid produced an empty frequency mask.")
    if not np.any(high_mask):
        raise ValueError("metric_bands_hz.high produced an empty frequency mask.")

    pred_mag2 = np.square(pred[:, 0]) + np.square(pred[:, 1])
    true_mag2 = np.square(target[:, 0]) + np.square(target[:, 1])

    def _band_share(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        total = np.sum(x, axis=(1, 2)) + 1e-8
        band = np.sum(x[:, mask, :], axis=(1, 2))
        return band / total

    pred_low = _band_share(pred_mag2, low_mask)
    true_low = _band_share(true_mag2, low_mask)
    pred_mid = _band_share(pred_mag2, mid_mask)
    true_mid = _band_share(true_mag2, mid_mask)
    pred_high = _band_share(pred_mag2, high_mask)
    true_high = _band_share(true_mag2, high_mask)

    low_err = np.abs(pred_low - true_low)
    mid_err = np.abs(pred_mid - true_mid)
    high_err = np.abs(pred_high - true_high)
    ratio_err = 0.5 * (low_err + high_err)
    return ratio_err, low_err, mid_err, high_err


@dataclass
class PickResult:
    p_pick_s: float
    s_pick_s: float
    failure_p: bool
    failure_s: bool


def _pick_onsets_from_energy(
    env: np.ndarray,
    tP_ref_s: float,
    tS_ref_s: float,
    frame_sec: float,
    smooth_frames: int,
    p_window_sec: float,
    s_window_sec: float,
    s_min_after_p_sec: float,
    confidence_threshold: float,
) -> PickResult:
    kernel = np.ones(max(1, int(smooth_frames)), dtype=np.float64) / max(1, int(smooth_frames))
    e_log = np.log1p(np.maximum(env, 0.0))
    e_s = np.convolve(e_log, kernel, mode="same")
    d = np.diff(e_s, prepend=e_s[0])
    n = len(d)

    def _pick_in_window(center_s: float, half_window_s: float, min_frame: int = 1) -> tuple[float, bool]:
        center_frame = int(round(center_s / frame_sec))
        half_frames = int(round(half_window_s / frame_sec))
        start = max(min_frame, center_frame - half_frames)
        end = min(n - 1, center_frame + half_frames)
        if end <= start:
            return float(center_frame * frame_sec), True
        seg = d[start : end + 1]
        local_idx = int(np.argmax(seg))
        peak_idx = start + local_idx
        peak_val = float(seg[local_idx])
        med = float(np.median(seg))
        mad = float(np.median(np.abs(seg - med)))
        conf = (peak_val - med) / (mad + 1e-8)
        fail = conf < float(confidence_threshold)
        return float(peak_idx * frame_sec), bool(fail)

    p_pick_s, fail_p = _pick_in_window(tP_ref_s, p_window_sec, min_frame=1)
    min_s_frame = max(1, int(round((p_pick_s + s_min_after_p_sec) / frame_sec)))
    s_pick_s, fail_s = _pick_in_window(tS_ref_s, s_window_sec, min_frame=min_s_frame)
    if s_pick_s <= p_pick_s:
        fail_s = True
    return PickResult(p_pick_s=p_pick_s, s_pick_s=s_pick_s, failure_p=fail_p, failure_s=fail_s)


class StreamingMetricAccumulator:
    def __init__(self) -> None:
        self.sum: Dict[str, float] = {}
        self.count: Dict[str, int] = {}

    def add(self, key: str, values: np.ndarray, mask: Optional[np.ndarray] = None) -> None:
        if mask is not None:
            values = values[mask]
        if values.size == 0:
            return
        self.sum[key] = self.sum.get(key, 0.0) + float(np.sum(values))
        self.count[key] = self.count.get(key, 0) + int(values.size)

    def add_scalar(self, key: str, value: float, n: int = 1) -> None:
        self.sum[key] = self.sum.get(key, 0.0) + float(value) * int(n)
        self.count[key] = self.count.get(key, 0) + int(n)

    def mean(self, key: str, default: float = 0.0) -> float:
        c = self.count.get(key, 0)
        if c <= 0:
            return float(default)
        return self.sum[key] / c

    def to_dict(self) -> Dict[str, float]:
        out = {}
        for key in sorted(self.sum.keys()):
            out[key] = self.mean(key)
        return out


def _batch_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    cond_raw: torch.Tensor,
    cfg: Mapping[str, Any],
) -> Dict[str, np.ndarray]:
    pred_np = pred.detach().cpu().numpy()
    true_np = target.detach().cpu().numpy()
    cond_raw_np = cond_raw.detach().cpu().numpy()

    feat_order = list(cfg["conditions"]["numeric_feature_order"])
    try:
        idx_mag = int(feat_order.index("magnitude"))
        idx_tp = int(feat_order.index("tP_ref_s"))
        idx_ts = int(feat_order.index("tS_ref_s"))
    except ValueError as exc:
        raise ValueError(
            "numeric_feature_order must include magnitude, tP_ref_s, and tS_ref_s for evaluation metrics."
        ) from exc
    max_idx = max(idx_mag, idx_tp, idx_ts)
    if cond_raw_np.shape[1] <= max_idx:
        raise ValueError(
            f"cond_raw feature width={cond_raw_np.shape[1]} is incompatible with numeric_feature_order "
            f"(requires index up to {max_idx})."
        )

    tP_ref = cond_raw_np[:, idx_tp]
    tS_ref = cond_raw_np[:, idx_ts]

    out: Dict[str, np.ndarray] = {}
    dr = pred_np[:, 0] - true_np[:, 0]
    di = pred_np[:, 1] - true_np[:, 1]
    out["complex_l1"] = np.mean(np.sqrt(dr * dr + di * di + 1e-8), axis=(1, 2))
    out["mr_lsd"] = _mr_lsd_batch(pred_np, true_np)

    env_pred = _envelope_from_complex(pred_np)
    env_true = _envelope_from_complex(true_np)

    hop = int(cfg["stft"]["hop_length"])
    fs = float(cfg["data"]["sampling_rate_hz"])
    frame_sec = hop / fs
    xcorr_max, lag_s = _xcorr_stats_batch(env_pred, env_true, frame_sec=frame_sec)
    out["xcorr_max"] = xcorr_max
    out["xcorr_lag_s"] = lag_s
    out["abs_xcorr_lag_s"] = np.abs(lag_s)

    env_corr = np.asarray([_safe_corr(env_true[i], env_pred[i]) for i in range(env_true.shape[0])], dtype=np.float64)
    out["envelope_corr"] = env_corr

    ratio_err, low_err, mid_err, high_err = _band_ratio_errors(pred_np, true_np, cfg=cfg)
    out["band_energy_ratio_error"] = ratio_err
    out["low_band_error"] = low_err
    out["mid_band_error"] = mid_err
    out["high_band_error"] = high_err

    pick_cfg = cfg["evaluation"]["onset_picker"]
    p_mae = np.full((pred_np.shape[0],), np.nan, dtype=np.float64)
    s_mae = np.full((pred_np.shape[0],), np.nan, dtype=np.float64)
    dtps_mae = np.full((pred_np.shape[0],), np.nan, dtype=np.float64)
    failure_p = np.zeros((pred_np.shape[0],), dtype=bool)
    failure_s = np.zeros((pred_np.shape[0],), dtype=bool)

    for i in range(pred_np.shape[0]):
        pred_pick = _pick_onsets_from_energy(
            env=env_pred[i],
            tP_ref_s=float(tP_ref[i]),
            tS_ref_s=float(tS_ref[i]),
            frame_sec=frame_sec,
            smooth_frames=int(pick_cfg["smooth_frames"]),
            p_window_sec=float(pick_cfg["p_window_sec"]),
            s_window_sec=float(pick_cfg["s_window_sec"]),
            s_min_after_p_sec=float(pick_cfg["s_min_after_p_sec"]),
            confidence_threshold=float(pick_cfg["confidence_threshold"]),
        )
        true_pick = _pick_onsets_from_energy(
            env=env_true[i],
            tP_ref_s=float(tP_ref[i]),
            tS_ref_s=float(tS_ref[i]),
            frame_sec=frame_sec,
            smooth_frames=int(pick_cfg["smooth_frames"]),
            p_window_sec=float(pick_cfg["p_window_sec"]),
            s_window_sec=float(pick_cfg["s_window_sec"]),
            s_min_after_p_sec=float(pick_cfg["s_min_after_p_sec"]),
            confidence_threshold=float(pick_cfg["confidence_threshold"]),
        )

        failure_p[i] = bool(pred_pick.failure_p)
        failure_s[i] = bool(pred_pick.failure_s)

        p_eval = (not pred_pick.failure_p) and (not true_pick.failure_p)
        s_eval = (not pred_pick.failure_s) and (not true_pick.failure_s)
        if p_eval:
            p_mae[i] = abs(pred_pick.p_pick_s - true_pick.p_pick_s)
        if s_eval:
            s_mae[i] = abs(pred_pick.s_pick_s - true_pick.s_pick_s)
        if p_eval and s_eval:
            pred_dt = pred_pick.s_pick_s - pred_pick.p_pick_s
            true_dt = true_pick.s_pick_s - true_pick.p_pick_s
            dtps_mae[i] = abs(pred_dt - true_dt)

    out["onset_mae_p_s"] = p_mae
    out["onset_mae_s_s"] = s_mae
    out["onset_mae_dtps_s"] = dtps_mae
    out["failure_p"] = failure_p.astype(np.float64)
    out["failure_s"] = failure_s.astype(np.float64)
    out["evaluable"] = ((~failure_p) & (~failure_s)).astype(np.float64)

    mags = cond_raw_np[:, idx_mag]
    out["mag"] = mags
    return out


def _add_metric_arrays_to_acc(
    acc: StreamingMetricAccumulator,
    arrays: Dict[str, np.ndarray],
    sample_mask: Optional[np.ndarray] = None,
) -> None:
    if sample_mask is None:
        sample_mask = np.ones_like(arrays["complex_l1"], dtype=bool)
    for key in (
        "complex_l1",
        "mr_lsd",
        "xcorr_max",
        "xcorr_lag_s",
        "abs_xcorr_lag_s",
        "envelope_corr",
        "band_energy_ratio_error",
        "low_band_error",
        "mid_band_error",
        "high_band_error",
    ):
        vals = np.asarray(arrays[key])
        acc.add(key, vals, mask=sample_mask)

    p_vals = np.asarray(arrays["onset_mae_p_s"])
    s_vals = np.asarray(arrays["onset_mae_s_s"])
    d_vals = np.asarray(arrays["onset_mae_dtps_s"])
    acc.add("onset_mae_p_s", p_vals, mask=sample_mask & np.isfinite(p_vals))
    acc.add("onset_mae_s_s", s_vals, mask=sample_mask & np.isfinite(s_vals))
    acc.add("onset_mae_dtps_s", d_vals, mask=sample_mask & np.isfinite(d_vals))

    failure_p = np.asarray(arrays["failure_p"])
    failure_s = np.asarray(arrays["failure_s"])
    evaluable = np.asarray(arrays["evaluable"])
    acc.add("onset_failure_rate_p", failure_p, mask=sample_mask)
    acc.add("onset_failure_rate_s", failure_s, mask=sample_mask)
    acc.add("onset_evaluable_rate", evaluable, mask=sample_mask)


def _bin_masks_from_magnitude(mag: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "all": np.ones_like(mag, dtype=bool),
        "lt3": mag < 3.0,
        "m3to5": (mag >= 3.0) & (mag < 5.0),
        "ge5": mag >= 5.0,
    }


def evaluate_reconstruction(
    model: torch.nn.Module,
    dataloader: Iterable[Dict[str, Any]],
    cfg: Mapping[str, Any],
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()
    acc = StreamingMetricAccumulator()
    bin_acc = {k: StreamingMetricAccumulator() for k in ("all", "lt3", "m3to5", "ge5")}
    with torch.no_grad():
        for b_idx, batch in enumerate(dataloader):
            if max_batches is not None and b_idx >= max_batches:
                break
            x = batch["x"].to(device)
            cond = batch["cond"].to(device)
            station = batch["station_idx"].to(device)
            cond_raw = batch["cond_raw"].to(device)
            pred, _, _ = model(x, cond, station)
            arrays = _batch_metrics(pred=pred, target=x, cond_raw=cond_raw, cfg=cfg)
            _add_metric_arrays_to_acc(acc, arrays)
            mag = np.asarray(arrays["mag"])
            for name, mask in _bin_masks_from_magnitude(mag).items():
                _add_metric_arrays_to_acc(bin_acc[name], arrays, sample_mask=mask)
    return {"global": acc.to_dict(), "by_bin": {k: v.to_dict() for k, v in bin_acc.items()}}


def _subset_batch(batch: Dict[str, Any], n_keep: int) -> Dict[str, Any]:
    if batch["x"].size(0) <= n_keep:
        return batch
    return {
        "x": batch["x"][:n_keep],
        "cond": batch["cond"][:n_keep],
        "cond_raw": batch["cond_raw"][:n_keep],
        "station_idx": batch["station_idx"][:n_keep],
        "meta": batch["meta"][:n_keep],
    }


def evaluate_condition_only(
    model: torch.nn.Module,
    dataloader: Iterable[Dict[str, Any]],
    cfg: Mapping[str, Any],
    device: torch.device,
    k_samples: int,
    seed_bank: List[int],
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    model.eval()
    per_k_results: List[Dict[str, float]] = []
    per_k_results_by_bin: List[Dict[str, Dict[str, float]]] = []
    total_seen = 0

    for k in range(int(k_samples)):
        acc = StreamingMetricAccumulator()
        bin_acc = {name: StreamingMetricAccumulator() for name in ("all", "lt3", "m3to5", "ge5")}
        seen_this_k = 0
        gen = torch.Generator(device=device.type)
        gen.manual_seed(int(seed_bank[k]))
        with torch.no_grad():
            for batch in dataloader:
                bsz = int(batch["x"].size(0))
                if max_samples is not None:
                    remaining = int(max_samples) - seen_this_k
                    if remaining <= 0:
                        break
                    if bsz > remaining:
                        batch = _subset_batch(batch, remaining)
                        bsz = remaining

                x = batch["x"].to(device)
                cond = batch["cond"].to(device)
                cond_raw = batch["cond_raw"].to(device)
                station = batch["station_idx"].to(device)

                pred = model.sample_condition_only(cond_numeric=cond, station_idx=station, generator=gen)
                arrays = _batch_metrics(pred=pred, target=x, cond_raw=cond_raw, cfg=cfg)
                _add_metric_arrays_to_acc(acc, arrays)
                mag = np.asarray(arrays["mag"])
                for name, mask in _bin_masks_from_magnitude(mag).items():
                    _add_metric_arrays_to_acc(bin_acc[name], arrays, sample_mask=mask)
                seen_this_k += bsz
        total_seen = max(total_seen, seen_this_k)
        per_k_results.append(acc.to_dict())
        per_k_results_by_bin.append({k: v.to_dict() for k, v in bin_acc.items()})

    metric_names = sorted({k for row in per_k_results for k in row.keys()})
    out_mean: Dict[str, float] = {}
    out_std: Dict[str, float] = {}
    for m in metric_names:
        vals = [float(row.get(m, float("nan"))) for row in per_k_results]
        vals = [v for v in vals if math.isfinite(v)]
        if not vals:
            continue
        out_mean[m] = float(np.mean(vals))
        out_std[m] = float(np.std(vals))

    # Bin-wise mean/std across K runs.
    bin_names = ("all", "lt3", "m3to5", "ge5")
    out_by_bin_mean: Dict[str, Dict[str, float]] = {}
    out_by_bin_std: Dict[str, Dict[str, float]] = {}
    for bname in bin_names:
        metric_names_b = sorted(
            {m for per_k in per_k_results_by_bin for m in per_k.get(bname, {}).keys()}
        )
        out_by_bin_mean[bname] = {}
        out_by_bin_std[bname] = {}
        for m in metric_names_b:
            vals = [
                float(per_k[bname].get(m, float("nan")))
                for per_k in per_k_results_by_bin
                if bname in per_k
            ]
            vals = [v for v in vals if math.isfinite(v)]
            if not vals:
                continue
            out_by_bin_mean[bname][m] = float(np.mean(vals))
            out_by_bin_std[bname][m] = float(np.std(vals))

    return {
        "k_samples": int(k_samples),
        "num_samples_evaluated": int(total_seen),
        "mean": out_mean,
        "std": out_std,
        "per_k": per_k_results,
        "by_bin_mean": out_by_bin_mean,
        "by_bin_std": out_by_bin_std,
        "per_k_by_bin": per_k_results_by_bin,
    }


def _load_model_for_eval(
    cfg: Mapping[str, Any],
    checkpoint_path: str,
    device: torch.device,
) -> torch.nn.Module:
    from .model import CVAEComplexSTFT
    from .utils import load_json

    num_stations = len(load_json(cfg["data"]["station_list_file"]))
    model = CVAEComplexSTFT(
        numeric_cond_dim=len(cfg["conditions"]["numeric_feature_order"]),
        num_stations=num_stations,
        latent_dim=int(cfg["model"]["latent_dim"]),
        station_embedding_dim=int(cfg["model"]["station_embedding_dim"]),
        condition_hidden_dim=int(cfg["model"]["condition_hidden_dim"]),
        encoder_channels=tuple(cfg["model"]["encoder_channels"]),
        decoder_channels=tuple(cfg["model"]["decoder_channels"]),
        input_shape=(2, int(cfg["stft"]["target_freq_bins"]), int(cfg["stft"]["target_time_frames"])),
    ).to(device)
    payload = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def main() -> None:
    import argparse
    from torch.utils.data import DataLoader

    from .dataset import ExternalHHComplexSTFTDataset, collate_exp001, prepare_exp001_artifacts
    from .utils import build_seed_bank, load_config, resolve_device, save_json

    p = argparse.ArgumentParser(description="Evaluate exp001 checkpoint.")
    p.add_argument(
        "--config",
        default="ML/autoencoder/experiments2/configs/exp001_base.json",
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test", "ood", "all"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--k-samples", type=int, default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--out-json", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    device = resolve_device()
    manifest, split, norm_stats = prepare_exp001_artifacts(cfg)
    model = _load_model_for_eval(cfg, args.checkpoint, device)

    if args.split == "all":
        split_names = ["val", "test", "ood"]
    else:
        split_names = [args.split]

    k = int(cfg["evaluation"]["k_samples_final"] if args.k_samples is None else args.k_samples)
    seed_bank = build_seed_bank(int(cfg["evaluation"]["seed_bank_base"]), k)

    out: Dict[str, Any] = {
        "checkpoint": args.checkpoint,
        "k_samples": k,
        "splits": {},
    }
    for split_name in split_names:
        ds = ExternalHHComplexSTFTDataset(cfg, manifest, split[split_name]["indices"], norm_stats)
        dl = DataLoader(
            ds,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            collate_fn=collate_exp001,
        )
        recon = evaluate_reconstruction(
            model=model,
            dataloader=dl,
            cfg=cfg,
            device=device,
            max_batches=None,
        )
        cond = evaluate_condition_only(
            model=model,
            dataloader=dl,
            cfg=cfg,
            device=device,
            k_samples=k,
            seed_bank=seed_bank,
            max_samples=args.max_samples,
        )
        out["splits"][split_name] = {"reconstruction": recon, "condition_only": cond}

    if args.out_json is None:
        out_json = Path(args.checkpoint).parent.parent / "metrics" / "evaluation_exp001.json"
    else:
        out_json = Path(args.out_json)
    save_json(out_json, out)
    print(f"[INFO] Evaluation saved to: {out_json}")


if __name__ == "__main__":
    main()
