from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class PaperLogSpectrogramConfig:
    n_fft: int = 256
    hop_length: int = 32
    clip_min: float = 1.0e-8
    log_max: float = 3.0
    drop_nyquist: bool = True
    center: bool = True
    pad_mode: str = "constant"


class PaperLogSpectrogram:
    """
    Paper-compatible magnitude-only log-spectrogram transform.

    Notes:
    - Released `tqdne` code uses `librosa.stft(..., n_fft=256, hop_length=32)` with
      center padding and then drops the Nyquist row.
    - `librosa` is unavailable in this environment, so we mirror the same tensor
      contract with `torch.stft(center=True, pad_mode="constant")`.
    """

    def __init__(self, config: PaperLogSpectrogramConfig | None = None) -> None:
        self.config = config or PaperLogSpectrogramConfig()
        self._window = torch.hann_window(self.config.n_fft, periodic=True)
        self._log_clip = float(np.log(self.config.clip_min))

    @property
    def shape(self) -> tuple[int, int]:
        freq_bins = self.config.n_fft // 2
        return (freq_bins, 128)

    def transform(self, waveform: np.ndarray) -> np.ndarray:
        """
        Convert waveform `[channels, time]` to normalized log-spectrogram
        `[channels, freq, frames]` in `[-1, 1]`.
        """
        if waveform.ndim != 2:
            raise ValueError(f"Expected waveform with shape [channels, time], got {waveform.shape}")

        tensor = torch.as_tensor(waveform, dtype=torch.float32)
        spec = torch.stft(
            tensor,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.n_fft,
            window=self._window.to(tensor.device),
            center=self.config.center,
            pad_mode=self.config.pad_mode,
            return_complex=True,
        )
        magnitude = spec.abs()
        if self.config.drop_nyquist:
            magnitude = magnitude[:, :-1, :]
        log_spec = torch.log(torch.clamp(magnitude, min=self.config.clip_min))
        norm = (log_spec - self._log_clip) / (self.config.log_max - self._log_clip)
        norm = norm * 2.0 - 1.0
        return norm.detach().cpu().numpy().astype(np.float32, copy=False)
