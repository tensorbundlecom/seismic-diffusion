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

from ML.diffusion.waveform_domain import resolve_waveform_domain
from ML.diffusion.embedding_paths import resolve_project_path


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
    waveform_domain: str = "instrument_counts"

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
            "waveform_domain": self.waveform_domain,
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
            "waveform_domain": self.waveform_domain,
        }


def _validate_spec(fields: Mapping[str, Any], ae_checkpoint: Optional[str], *, origin: str,
                   identity_payload: Mapping[str, Any],
                   waveform_domain: Optional[str] = None,
                   ae_name: Optional[str] = None) -> ReconstructionSpec:
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
    resolved_checkpoint = (
        str(resolve_project_path(ae_checkpoint, ae_name=ae_name)) if ae_checkpoint else None
    )
    return ReconstructionSpec(
        mode=mode,
        ae_checkpoint=resolved_checkpoint,
        global_min=lo,
        global_max=hi,
        amplitude_epsilon=amplitude_epsilon,
        source_identity=identity,
        source_origin=origin,
        waveform_domain=resolve_waveform_domain(waveform_domain),
    )


def resolve_reconstruction_spec(
    diffusion_config: Mapping[str, Any],
    *,
    embeddings_source_path: str | Path | None = None,
    ae_checkpoint_override: str | Path | None = None,
    waveform_domain_override: str | None = None,
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
                waveform_domain=(waveform_domain_override or provenance.get(
                    "waveform_domain", diffusion_config.get("waveform_domain")
                )),
                ae_name=(provenance.get("ae_name") or (
                    provenance.get("source", {}).get("ae_name")
                    if isinstance(provenance.get("source"), Mapping) else None
                )),
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
                    waveform_domain=(waveform_domain_override or source.get(
                        "waveform_domain", diffusion_config.get("waveform_domain")
                    )),
                    ae_name=source.get("ae_name"),
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
        waveform_domain=(waveform_domain_override or diffusion_config.get("waveform_domain")),
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


_CHECKPOINT_MODEL_ARTIFACTS = (
    # Diffusers UNet architecture and weights.  Both safetensors and PyTorch
    # binary checkpoints are supported because save_pretrained can be
    # configured either way.
    "config.json",
    "diffusion_pytorch_model.safetensors",
    "diffusion_pytorch_model.bin",
    "pytorch_model.bin",
    # The wrapper's learned conditioning embedding and the inference schedule
    # also affect every generated waveform.
    "station_embedding.pt",
    "scheduler_config.json",
)


def checkpoint_model_identity(checkpoint_dir: str | Path) -> str:
    """SHA-256 identity of the checkpoint artifacts that affect sampling.

    A checkpoint directory is commonly reused at ``.../unet2d`` between
    training runs.  Path/config-only cache names would then silently reuse
    samples from the previous weights.  Hash only the model, wrapper, and
    scheduler artifacts rather than every file in the directory, so logs and
    provenance sidecars do not invalidate expensive waveform caches.
    """
    checkpoint_dir = Path(checkpoint_dir).resolve()
    artifacts = [checkpoint_dir / name for name in _CHECKPOINT_MODEL_ARTIFACTS]
    artifacts = [path for path in artifacts if path.is_file()]
    if not artifacts:
        expected = ", ".join(_CHECKPOINT_MODEL_ARTIFACTS)
        raise FileNotFoundError(
            f"No model artifacts found in checkpoint {checkpoint_dir}; expected one of: {expected}."
        )

    digest = hashlib.sha256()
    for path in artifacts:
        # Include the filename as well as bytes: moving a scheduler config into
        # a weight filename must not retain the same cache identity.
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        try:
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError as exc:
            raise ValueError(f"Could not hash checkpoint artifact {path}: {exc}") from exc
        digest.update(b"\0")
    return digest.hexdigest()


def diffusion_cache_tag(checkpoint_dir: str | Path, diffusion_config: Mapping[str, Any],
                        reconstruction: ReconstructionSpec) -> str:
    """Content-addressed cache identity for generated waveform artifacts."""
    checkpoint_dir = Path(checkpoint_dir).resolve()
    # The model-artifact identity is essential: the final checkpoint directory
    # is intentionally reused by training, so its path/config can stay stable
    # while its learned weights change.
    payload = {
        "diffusion_checkpoint": str(checkpoint_dir),
        "training_config": dict(diffusion_config),
        "reconstruction": reconstruction.as_dict(),
        "model_artifacts_sha256": checkpoint_model_identity(checkpoint_dir),
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
