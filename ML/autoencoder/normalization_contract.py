"""Shared waveform-domain normalization defaults for autoencoder training."""

from __future__ import annotations

import math
from typing import Optional


WAVEFORM_DOMAINS = ("instrument_counts", "physical_acceleration")


def resolve_amplitude_epsilon(
    waveform_domain: str, amplitude_epsilon: Optional[float]
) -> float:
    """Resolve an explicit or domain-appropriate magnitude floor.

    Count-domain training historically used ``log1p(magnitude)``, whose exact
    equivalent is ``log(magnitude + 1.0)``. Physical acceleration needs a much
    smaller floor so that low-amplitude signals remain representable.
    """
    if waveform_domain not in WAVEFORM_DOMAINS:
        raise ValueError(
            f"Unsupported waveform domain {waveform_domain!r}. "
            f"Expected one of: {', '.join(WAVEFORM_DOMAINS)}."
        )

    if amplitude_epsilon is None:
        return 1.0 if waveform_domain == "instrument_counts" else 1e-12

    resolved = float(amplitude_epsilon)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError("amplitude_epsilon must be finite and greater than zero.")
    return resolved
