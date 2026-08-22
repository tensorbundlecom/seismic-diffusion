"""Reusable, content-aware cache for global STFT normalization bounds."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Iterable, Tuple

from tqdm import tqdm


# Bump this whenever the preprocessing used by fit_global_normalization changes.
CACHE_FORMAT_VERSION = 1
PREPROCESSING_VERSION = 1


@dataclass(frozen=True)
class GlobalNormalizationCacheResult:
    """The fitted bounds and the cache entry that supplied them."""

    global_min: float
    global_max: float
    cache_path: Path
    cache_hit: bool


def _dataset_settings(dataset) -> dict:
    """Return every dataset setting that can affect fitted STFT values."""
    return {
        "dataset_type": f"{type(dataset).__module__}.{type(dataset).__qualname__}",
        "return_magnitude": bool(dataset.return_magnitude),
        "log_scale": bool(dataset.log_scale),
        "nperseg": int(dataset.nperseg),
        "noverlap": int(dataset.noverlap),
        "nfft": int(dataset.nfft),
        "target_freq_bins": dataset.target_freq_bins,
        "target_time_bins": dataset.target_time_bins,
        "resample_hz": dataset.resample_hz,
        "target_seconds": dataset.target_seconds,
        "target_samples": dataset.target_samples,
        # These are fixed in the dataset implementation, but are part of the
        # transform and therefore deliberately represented in the cache key.
        "stft_boundary": "zeros",
        "stft_padded": True,
        "stft_return_onesided": True,
        "preprocessing_version": PREPROCESSING_VERSION,
    }


def _cache_key(dataset, indices: Iterable[int]) -> Tuple[str, list[int]]:
    """Hash the ordered split, selected file identities, and STFT settings."""
    normalized_indices = [int(index) for index in indices]
    digest = hashlib.sha256()

    def update(value: object) -> None:
        digest.update(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")

    update({"cache_format_version": CACHE_FORMAT_VERSION, "settings": _dataset_settings(dataset)})
    # Index order is included intentionally: it captures the exact random_split
    # result, while the paths/stat fields invalidate the entry when train data changes.
    for index in tqdm(
        normalized_indices,
        desc="Fingerprinting normalization cache",
        unit="file",
        dynamic_ncols=True,
        mininterval=0.5,
    ):
        file_path = Path(dataset.file_paths[index])
        file_stat = file_path.stat()
        update(
            {
                "index": index,
                "path": str(file_path.resolve()),
                "size": file_stat.st_size,
                "mtime_ns": file_stat.st_mtime_ns,
            }
        )
    return digest.hexdigest(), normalized_indices


def _read_cache(cache_path: Path, cache_key: str) -> tuple[float, float] | None:
    try:
        with cache_path.open("r", encoding="utf-8") as cache_file:
            cached = json.load(cache_file)
        if (
            cached.get("cache_format_version") != CACHE_FORMAT_VERSION
            or cached.get("cache_key") != cache_key
        ):
            return None
        global_min = float(cached["global_min"])
        global_max = float(cached["global_max"])
        if not math.isfinite(global_min) or not math.isfinite(global_max):
            return None
        if global_min > global_max:
            return None
        return global_min, global_max
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def _write_cache_atomically(cache_path: Path, cache_key: str, global_min: float, global_max: float) -> None:
    """Publish a complete JSON cache file in one atomic rename."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "cache_key": cache_key,
        "global_min": global_min,
        "global_max": global_max,
    }
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=cache_path.parent,
            prefix=f".{cache_path.name}.", suffix=".tmp", delete=False,
        ) as temp_file:
            temp_name = temp_file.name
            json.dump(payload, temp_file, sort_keys=True)
            temp_file.write("\n")
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_name, cache_path)
        temp_name = None
    finally:
        if temp_name is not None:
            try:
                Path(temp_name).unlink()
            except FileNotFoundError:
                pass


def fit_or_load_global_normalization(dataset, indices: Iterable[int], cache_root: str | Path) -> GlobalNormalizationCacheResult:
    """Load matching global bounds or fit and cache them atomically.

    The cache entry is independent of a run name. Its key includes the exact
    training indices, every selected file's resolved path/size/mtime, and all
    preprocessing settings that influence post-log, post-resize STFT values.
    """
    print("Checking global normalization cache...")
    cache_key, normalized_indices = _cache_key(dataset, indices)
    cache_path = Path(cache_root) / f"global_normalization_{cache_key}.json"
    cached_bounds = _read_cache(cache_path, cache_key)
    if cached_bounds is not None:
        global_min, global_max = dataset.set_global_normalization_stats(*cached_bounds)
        print(f"Global normalization cache hit: {cache_path}")
        return GlobalNormalizationCacheResult(global_min, global_max, cache_path, True)

    print(f"Global normalization cache miss: {cache_path}")
    print("Fitting global normalization over training samples...")
    global_min, global_max = dataset.fit_global_normalization(normalized_indices)
    _write_cache_atomically(cache_path, cache_key, global_min, global_max)
    print(f"Saved global normalization cache: {cache_path}")
    return GlobalNormalizationCacheResult(global_min, global_max, cache_path, False)
