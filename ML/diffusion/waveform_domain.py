"""Waveform-domain provenance and physical ground-motion conversion helpers."""

from __future__ import annotations

from typing import Optional

import numpy as np


INSTRUMENT_COUNTS = "instrument_counts"
PHYSICAL_ACCELERATION = "physical_acceleration"
WAVEFORM_DOMAINS = (INSTRUMENT_COUNTS, PHYSICAL_ACCELERATION)


def normalize_waveform_domain(value: str, *, field: str = "waveform_domain") -> str:
    domain = str(value).strip().lower()
    if domain not in WAVEFORM_DOMAINS:
        raise ValueError(
            f"Unsupported {field}={value!r}; expected one of: {', '.join(WAVEFORM_DOMAINS)}."
        )
    return domain


def resolve_waveform_domain(
    recorded: Optional[str], override: Optional[str] = None
) -> str:
    """Resolve an optional override, retaining counts for legacy checkpoints."""
    if override is not None and str(override).strip().lower() != "auto":
        return normalize_waveform_domain(override, field="waveform_domain override")
    if recorded is None:
        return INSTRUMENT_COUNTS
    return normalize_waveform_domain(recorded)


def physical_acceleration_to_motion(
    acceleration: np.ndarray, output: str, sample_rate: float
) -> np.ndarray:
    """Return acceleration or velocity from an acceleration-domain waveform.

    Velocity is obtained by frequency-domain integration with the undefined DC
    component fixed to zero. The physical archive is already bandpass filtered,
    so this avoids cumulative-integration drift without adding a sensor response.
    """
    acc = np.asarray(acceleration, dtype=np.float64)
    kind = str(output).strip().upper()
    if kind == "ACC":
        return acc.copy()
    if kind != "VEL":
        raise ValueError(f"Physical acceleration supports ACC or VEL output, got {output!r}.")
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive and finite, got {sample_rate!r}.")
    if acc.size == 0:
        return acc.copy()
    spectrum = np.fft.rfft(acc - np.mean(acc))
    frequencies = np.fft.rfftfreq(acc.size, d=1.0 / float(sample_rate))
    velocity_spectrum = np.zeros_like(spectrum)
    nonzero = frequencies > 0.0
    velocity_spectrum[nonzero] = spectrum[nonzero] / (2j * np.pi * frequencies[nonzero])
    return np.fft.irfft(velocity_spectrum, n=acc.size)
