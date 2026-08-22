"""Checkpoint-bound spectrogram reconstruction helpers.

The diffusion model learns AE latents, not a particular representation of
amplitude.  The representation therefore has to travel with the diffusion
checkpoint.  In particular, a globally-normalised AE can recover physical
STFT magnitudes directly and must not use the legacy AmplitudeMLP correction.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read reconstruction metadata from {path}: {exc}") from exc


def _normalization_fields(payload: Mapping[str, Any]) -> dict:
    """Extract source.json-compatible normalization fields from a payload."""
    nested = payload.get("normalization")
    source = payload.get("source")
    source = source if isinstance(source, Mapping) else {}
    source_nested = source.get("normalization")
    source_nested = source_nested if isinstance(source_nested, Mapping) else {}

    def first(*keys: str):
        # Accept both the direct source.json representation and the aliases
        # copied into diffusion checkpoint provenance. This also tolerates a
        # complete source snapshot nested under ``source``.
        for values in (nested, payload, source_nested, source):
            if not isinstance(values, Mapping):
                continue
            for key in keys:
                if values.get(key) is not None:
                    return values[key]
        return None

    return {
        "mode": first("normalization_mode", "mode"),
        "global_min": first("global_min"),
        "global_max": first("global_max"),
        "amplitude_epsilon": first("amplitude_epsilon"),
    }


@dataclass(frozen=True)
class ReconstructionSpec:
    """The immutable AE amplitude contract for one diffusion checkpoint."""

    mode: str
    ae_checkpoint: Optional[str]
    global_min: Optional[float] = None
    global_max: Optional[float] = None
    source_identity: str = "legacy"
    source_origin: str = "legacy"
    amplitude_epsilon: float = 1.0

    @property
    def uses_global_normalization(self) -> bool:
        return self.mode == "global"

    @property
    def uses_amplitude_model(self) -> bool:
        return not self.uses_global_normalization

    @property
    def cache_tag(self) -> str:
        # Short but content-addressed: changing the AE source contract cannot
        # accidentally reuse waveform/evaluation artifacts.
        payload = {
            "mode": self.mode,
            "global_min": self.global_min,
            "global_max": self.global_max,
            "amplitude_epsilon": self.amplitude_epsilon,
            "ae_checkpoint": self.ae_checkpoint,
            "source_identity": self.source_identity,
        }
        return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()[:12]

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "normalization_mode": self.mode,
            "global_min": self.global_min,
            "global_max": self.global_max,
            "amplitude_epsilon": self.amplitude_epsilon,
            "ae_checkpoint": self.ae_checkpoint,
            "source_identity": self.source_identity,
            "source_origin": self.source_origin,
        }


def _validate_spec(fields: Mapping[str, Any], ae_checkpoint: Optional[str], *, origin: str,
                   identity_payload: Mapping[str, Any]) -> ReconstructionSpec:
    mode = fields.get("mode") or "per_event"
    if mode not in {"global", "per_event"}:
        raise ValueError(f"Unsupported AE normalization mode {mode!r} from {origin}.")
    lo, hi = fields.get("global_min"), fields.get("global_max")
    amplitude_epsilon = fields.get("amplitude_epsilon")
    if mode == "global":
        if lo is None or hi is None:
            raise ValueError(f"Global AE normalization from {origin} is missing global_min/global_max.")
        lo, hi = float(lo), float(hi)
        if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
            raise ValueError(
                f"Invalid global AE normalization range from {origin}: min={lo}, max={hi}."
            )
        # Checkpoints created before physical-unit normalization used log1p,
        # whose exact inverse is recovered by epsilon=1.
        amplitude_epsilon = 1.0 if amplitude_epsilon is None else float(amplitude_epsilon)
        if not math.isfinite(amplitude_epsilon) or amplitude_epsilon <= 0:
            raise ValueError(
                f"Invalid amplitude_epsilon from {origin}: {amplitude_epsilon}."
            )
    else:
        lo = hi = None
        amplitude_epsilon = 1.0
    identity = hashlib.sha256(_canonical_json(identity_payload).encode()).hexdigest()[:16]
    return ReconstructionSpec(
        mode=mode,
        ae_checkpoint=ae_checkpoint,
        global_min=lo,
        global_max=hi,
        amplitude_epsilon=amplitude_epsilon,
        source_identity=identity,
        source_origin=origin,
    )


def resolve_reconstruction_spec(
    diffusion_config: Mapping[str, Any],
    *,
    embeddings_source_path: str | Path | None = None,
    ae_checkpoint_override: str | Path | None = None,
) -> ReconstructionSpec:
    """Resolve AE normalization, preferring the checkpoint's provenance snapshot.

    New diffusion checkpoints save ``embedding_provenance``.  Older models
    predate that field, so ``embeddings/source.json`` remains a compatibility
    fallback.  The AE checkpoint itself is only inspected when neither source
    carries normalization fields, which lets manually-built old source files
    remain usable without making a mutable source file authoritative for new
    checkpoints.
    """
    provenance = diffusion_config.get("embedding_provenance")
    if isinstance(provenance, Mapping):
        fields = _normalization_fields(provenance)
        ae_ckpt = ae_checkpoint_override or provenance.get("ae_checkpoint")
        if fields["mode"] is not None:
            return _validate_spec(
                fields, str(ae_ckpt) if ae_ckpt else None,
                origin="diffusion checkpoint embedding_provenance",
                identity_payload=dict(provenance),
            )

    source = None
    if embeddings_source_path is not None:
        path = Path(embeddings_source_path)
        if path.exists():
            source = _read_json(path)
            fields = _normalization_fields(source)
            ae_ckpt = ae_checkpoint_override or source.get("ae_checkpoint")
            if fields["mode"] is not None:
                return _validate_spec(
                    fields, str(ae_ckpt) if ae_ckpt else None,
                    origin=f"compatibility source file {path}",
                    identity_payload=source,
                )

    ae_ckpt = ae_checkpoint_override
    if ae_ckpt is None and isinstance(provenance, Mapping):
        ae_ckpt = provenance.get("ae_checkpoint")
    if ae_ckpt is None and source is not None:
        ae_ckpt = source.get("ae_checkpoint")

    # Last-resort compatibility for old source.json produced before the
    # normalization fields were added.  Such AEs used per-event scaling.
    return _validate_spec(
        {"mode": "per_event"}, str(ae_ckpt) if ae_ckpt else None,
        origin="legacy per-event default",
        identity_payload={"ae_checkpoint": str(ae_ckpt) if ae_ckpt else None, "mode": "per_event"},
    )


def decoded_to_magnitude(decoded, spec: ReconstructionSpec, *, legacy_inv_log_gain=1.0):
    """Invert the AE output into linear STFT magnitude.

    For global AEs this deliberately does *not* clamp ``decoded`` to [0, 1].
    Samples outside that range carry extrapolated log-amplitude information.
    New physical-unit AEs use ``log(magnitude + amplitude_epsilon)``. Older
    global AEs omit epsilon and therefore default to 1, exactly preserving
    their historical ``expm1`` inverse.
    """
    if spec.uses_global_normalization:
        log_magnitude = decoded * (spec.global_max - spec.global_min) + spec.global_min
    else:
        # Historical per-event output needs the learned inverse log gain.
        log_magnitude = decoded * legacy_inv_log_gain

    epsilon = spec.amplitude_epsilon if spec.uses_global_normalization else 1.0

    if isinstance(log_magnitude, np.ndarray):
        if epsilon == 1.0:
            return np.maximum(np.expm1(log_magnitude), 0.0)
        return np.maximum(np.exp(log_magnitude) - epsilon, 0.0)

    # Keep torch an optional dependency for worker-only numpy operations.
    import torch
    if torch.is_tensor(log_magnitude):
        if epsilon == 1.0:
            return torch.clamp(torch.expm1(log_magnitude), min=0.0)
        return torch.clamp(torch.exp(log_magnitude) - epsilon, min=0.0)
    if epsilon == 1.0:
        return max(float(np.expm1(float(log_magnitude))), 0.0)
    return max(float(np.exp(float(log_magnitude))) - epsilon, 0.0)


def postprocess_griffinlim_waveform(wave: np.ndarray, spec: ReconstructionSpec,
                                    *, amp_scale: Optional[float] = None,
                                    metric: str = "max") -> np.ndarray:
    """Apply the legacy amplitude correction, or preserve global-AE amplitude."""
    wave = np.asarray(wave)
    if not spec.uses_amplitude_model:
        return wave
    if amp_scale is None:
        raise ValueError("Legacy per-event reconstruction requires an amplitude-model scale.")
    ref = float(np.max(np.abs(wave))) if metric == "max" else float(np.std(wave))
    if ref > 1e-10:
        wave = wave / ref * float(amp_scale)
    return wave


def diffusion_cache_tag(checkpoint_dir: str | Path, diffusion_config: Mapping[str, Any],
                        reconstruction: ReconstructionSpec) -> str:
    """Content-addressed cache identity for generated waveform artifacts."""
    checkpoint_dir = str(Path(checkpoint_dir).resolve())
    # The config is cheap to hash and changes when the training contract does;
    # the path distinguishes separately trained checkpoints with the same cfg.
    payload = {
        "diffusion_checkpoint": checkpoint_dir,
        "training_config": dict(diffusion_config),
        "reconstruction": reconstruction.as_dict(),
    }
    return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()[:12]


def checkpoint_stft_config(diffusion_config: Mapping[str, Any], *,
                           embeddings_source_path: str | Path | None = None) -> dict | None:
    """Return the checkpoint's STFT snapshot, with old-source fallback.

    New configs carry it both directly in ``embedding_provenance.stft`` and in
    the complete source snapshot. Reading a live source file is deliberately
    a last resort for historical checkpoints only.
    """
    provenance = diffusion_config.get("embedding_provenance")
    if isinstance(provenance, Mapping):
        stft = provenance.get("stft")
        if isinstance(stft, Mapping):
            return dict(stft)
        source = provenance.get("source")
        if isinstance(source, Mapping) and isinstance(source.get("stft"), Mapping):
            return dict(source["stft"])
    if embeddings_source_path is not None:
        path = Path(embeddings_source_path)
        if path.exists():
            source = _read_json(path)
            if isinstance(source.get("stft"), Mapping):
                return dict(source["stft"])
    return None
