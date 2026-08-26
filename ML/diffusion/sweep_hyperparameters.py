"""Validation helpers for the continuously sweepable diffusion hyperparameters."""

import argparse
import math


def positive_finite_float(value: str) -> float:
    """Return a strictly positive finite float suitable for ``argparse.type``."""
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"must be a number; got {value!r}.") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive finite number; got {value!r}.")
    return parsed


def nonnegative_finite_float(value: str) -> float:
    """Return a finite float that may be zero, suitable for ``argparse.type``."""
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"must be a number; got {value!r}.") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError(
            f"must be a non-negative finite number; got {value!r}."
        )
    return parsed


def positive_int(value: str) -> int:
    """Return a strictly positive integer suitable for ``argparse.type``."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"must be an integer; got {value!r}.") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer; got {value!r}.")
    return parsed


def groupnorm_base_channels(value: str) -> int:
    """Validate a base U-Net width against Diffusers' 32-group GroupNorm."""
    parsed = positive_int(value)
    if parsed % 32 != 0:
        raise argparse.ArgumentTypeError(
            "must be a positive multiple of 32: Diffusers UNet uses GroupNorm with "
            f"32 groups; got {parsed}."
        )
    return parsed
